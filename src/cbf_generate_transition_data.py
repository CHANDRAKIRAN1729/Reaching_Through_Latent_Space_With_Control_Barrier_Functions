"""
CBF Transition Data Generation — Run Goal+Prior-only planner and record transitions.

Generates (z_k, z_{k+1}^nom, obs) transition pairs by running the nominal planner
(Goal + Prior losses only, NO collision loss) on random scenarios.

Critical: The transitions MUST come from the Goal+Prior-only planner because that is
what the CBF will see at inference time. Training on transitions from a different
planner would create a distribution mismatch that breaks the CBF guarantees.
"""

from __future__ import print_function
import argparse
import json
import logging
import numpy as np
import os
import time
import torch
import torch.optim as optim
import warnings

warnings.filterwarnings('ignore', category=FutureWarning, module='torch')

from vae import VAE
from robot_state_dataset import RobotStateDataset
from robot_obs_dataset import RobotObstacleDataset
from sim.panda import Panda
from sim.robot3d import Robo3D
from evaluate_planning import ObstacleScenarioGenerator
import cbf_config as cfg

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


def load_frozen_vae(device):
    """Load the frozen VAE model."""
    with open(cfg.VAE_CONFIG, 'r') as f:
        config = json.load(f)
        if 'parsed_args' in config:
            config = config['parsed_args']

    model = VAE(
        config['input_dim'], config['latent_dim'],
        config['units_per_layer'], config['num_hidden_layers']
    )
    checkpoint = torch.load(cfg.VAE_CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    logging.info(f"Frozen VAE loaded from {cfg.VAE_CHECKPOINT}")
    return model


def get_normalization_stats():
    """Get normalization statistics from existing datasets."""
    dataset = RobotStateDataset(
        cfg.DATA_PATH, train=0, train_data_name='free_space_100k_train.dat'
    )
    mean_train = dataset.get_mean_train()
    std_train = dataset.get_std_train()
    return mean_train, std_train


def run_nominal_planner_and_record_transitions(
        model, robot, robo3d,
        q_start, e_start, e_target, obstacles_raw,
        mean_train_t, std_train_t,
        device, planning_lr, lambda_prior, max_steps):
    """
    Run the Goal+Prior-only planner and record (z_k, z_{k+1}^nom, obs) at each step.

    This is the NOMINAL planner — no collision loss. The transitions capture
    how the planner would move through latent space if unconstrained by safety,
    which is exactly what the CBF will see at inference time.

    Returns:
        transitions: list of dicts with 'z_k', 'z_nom', 'obs', 'safe_k', 'safe_nom'
    """
    # Normalize start state
    x_start = torch.cat([q_start, e_start], dim=1)  # [1, 10]
    x_start_norm = (x_start - mean_train_t[:, :10]) / std_train_t[:, :10]

    # Encode to latent — use sampled z (index [0]) to match inference planner
    with torch.no_grad():
        z_init = model.encoder(x_start_norm)[0]

    z = z_init.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([z], lr=planning_lr)

    transitions = []
    obstacles_xyhr = [obs.tolist() for obs in obstacles_raw]

    for step in range(max_steps):
        # Save z_k BEFORE the optimizer step
        z_before = z.data.clone()

        optimizer.zero_grad()

        # Decode
        x_decoded_norm = model.decoder(z)
        x_decoded = x_decoded_norm * std_train_t[:, :10] + mean_train_t[:, :10]
        q_decoded = x_decoded[:, :7]
        e_decoded = x_decoded[:, 7:10]

        # === NOMINAL LOSS: Goal + Prior ONLY (no collision!) ===
        L_goal = torch.norm(e_decoded - e_target)
        L_prior = 0.5 * torch.sum(z ** 2)
        L_nominal = L_goal + lambda_prior * L_prior

        # Check if goal reached
        if L_goal.item() < cfg.SUCCESS_THRESHOLD:
            break

        L_nominal.backward()
        optimizer.step()

        # z now contains z_{k+1}^nom
        z_after = z.data.clone()

        # Check collision status for z_before and z_after using Robo3D
        with torch.no_grad():
            # Decode z_before
            x_before_norm = model.decoder(z_before)
            x_before = x_before_norm * std_train_t[:, :10] + mean_train_t[:, :10]
            q_before = x_before[:, :7].cpu().numpy()[0]
            q_before_deg = np.degrees(q_before).tolist()
            safe_k = 0.0 if robo3d.check_for_collision(q_before_deg, obstacles_xyhr) else 1.0

            # Decode z_after
            x_after_norm = model.decoder(z_after)
            x_after = x_after_norm * std_train_t[:, :10] + mean_train_t[:, :10]
            q_after = x_after[:, :7].cpu().numpy()[0]
            q_after_deg = np.degrees(q_after).tolist()
            safe_nom = 0.0 if robo3d.check_for_collision(q_after_deg, obstacles_xyhr) else 1.0

        transitions.append({
            'z_k': z_before.cpu().squeeze(0),
            'z_nom': z_after.cpu().squeeze(0),
            'obs': torch.tensor(obstacles_raw[0], dtype=torch.float32),  # first obstacle
            'safe_k': safe_k,
            'safe_nom': safe_nom,
        })

    return transitions


def main():
    parser = argparse.ArgumentParser(description='Generate CBF transition-pair dataset')
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=cfg.SEED)
    parser.add_argument('--num_scenarios', type=int, default=cfg.NUM_TRANSITION_SCENARIOS,
                        help='Number of planning scenarios to run')
    parser.add_argument('--max_steps', type=int, default=cfg.TRANSITION_MAX_STEPS)
    parser.add_argument('--planning_lr', type=float, default=cfg.TRANSITION_PLANNING_LR)
    parser.add_argument('--lambda_prior', type=float, default=cfg.TRANSITION_LAMBDA_PRIOR)
    parser.add_argument('--num_obstacles', type=int, default=1)
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Fraction of scenarios for validation')
    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logging.info(f"Using device: {device}")

    # Create output directory
    os.makedirs(cfg.CBF_DATA_DIR, exist_ok=True)

    # Load models
    model = load_frozen_vae(device)
    mean_train, std_train = get_normalization_stats()
    mean_train_t = torch.tensor(mean_train, dtype=torch.float32).to(device)
    std_train_t = torch.tensor(std_train, dtype=torch.float32).to(device)

    # Initialize robot
    robot = Panda()
    robot.to(device)
    robo3d = Robo3D(Panda())
    scenario_gen = ObstacleScenarioGenerator(robot)

    q_min_rad = robot.joint_min_limits_tensor * (torch.pi / 180.0)
    q_max_rad = robot.joint_max_limits_tensor * (torch.pi / 180.0)

    # =========================================================================
    # Generate transitions
    # =========================================================================
    logging.info(f"\n{'=' * 60}")
    logging.info(f"Generating transition pairs from {args.num_scenarios} scenarios")
    logging.info(f"Planner: Goal + Prior only (lr={args.planning_lr}, λ_prior={args.lambda_prior})")
    logging.info(f"{'=' * 60}\n")

    all_transitions = []
    scenario_boundaries = [0]  # track where each scenario's transitions start
    start_time = time.time()

    for scenario_id in range(args.num_scenarios):
        # Sample random start and target configurations
        q_start = torch.rand(1, 7, device=device) * (q_max_rad - q_min_rad) + q_min_rad
        e_start = robot.FK(q_start.clone(), device, rad=True)
        q_target = torch.rand(1, 7, device=device) * (q_max_rad - q_min_rad) + q_min_rad
        e_target = robot.FK(q_target.clone(), device, rad=True)

        # Generate obstacles
        obstacles_raw = scenario_gen.generate_scenario(
            q_start.cpu().numpy()[0],
            e_start.cpu().numpy()[0],
            e_target.cpu().numpy()[0],
            num_obstacles=args.num_obstacles
        )

        if len(obstacles_raw) == 0:
            continue

        # Run nominal planner and record transitions
        transitions = run_nominal_planner_and_record_transitions(
            model, robot, robo3d,
            q_start, e_start, e_target, obstacles_raw,
            mean_train_t, std_train_t,
            device, args.planning_lr, args.lambda_prior, args.max_steps
        )

        all_transitions.extend(transitions)
        scenario_boundaries.append(len(all_transitions))

        if (scenario_id + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (scenario_id + 1) / elapsed
            logging.info(f"Progress: {scenario_id + 1}/{args.num_scenarios} "
                         f"({rate:.1f} scenarios/s), "
                         f"total transitions: {len(all_transitions)}")

    elapsed = time.time() - start_time
    logging.info(f"\nGenerated {len(all_transitions)} transitions from "
                 f"{args.num_scenarios} scenarios in {elapsed:.1f}s")

    if len(all_transitions) == 0:
        logging.error("No transitions generated! Check scenario generation.")
        return

    # =========================================================================
    # Collate into tensors
    # =========================================================================
    z_k_all = torch.stack([t['z_k'] for t in all_transitions])
    z_nom_all = torch.stack([t['z_nom'] for t in all_transitions])
    obs_all = torch.stack([t['obs'] for t in all_transitions])
    safe_k_all = torch.tensor([t['safe_k'] for t in all_transitions], dtype=torch.float32)
    safe_nom_all = torch.tensor([t['safe_nom'] for t in all_transitions], dtype=torch.float32)

    # Statistics
    total = len(all_transitions)
    safe_k_count = (safe_k_all == 1).sum().item()
    unsafe_k_count = (safe_k_all == 0).sum().item()
    safe_nom_count = (safe_nom_all == 1).sum().item()
    unsafe_nom_count = (safe_nom_all == 0).sum().item()

    logging.info(f"\nTransition statistics:")
    logging.info(f"  z_k:   safe={safe_k_count} ({safe_k_count/total*100:.1f}%), "
                 f"unsafe={unsafe_k_count} ({unsafe_k_count/total*100:.1f}%)")
    logging.info(f"  z_nom: safe={safe_nom_count} ({safe_nom_count/total*100:.1f}%), "
                 f"unsafe={unsafe_nom_count} ({unsafe_nom_count/total*100:.1f}%)")

    # =========================================================================
    # Split by scenario (NOT by individual transition — avoid data leakage)
    # =========================================================================
    num_scenarios = len(scenario_boundaries) - 1
    n_val_scenarios = max(1, int(num_scenarios * args.val_split))
    n_train_scenarios = num_scenarios - n_val_scenarios

    # Shuffle scenario indices
    scenario_perm = torch.randperm(num_scenarios).tolist()
    train_scenarios = scenario_perm[:n_train_scenarios]
    val_scenarios = scenario_perm[n_train_scenarios:]

    def gather_scenario_indices(scenario_ids):
        indices = []
        for s in scenario_ids:
            start = scenario_boundaries[s]
            end = scenario_boundaries[s + 1]
            indices.extend(range(start, end))
        return indices

    train_idx = gather_scenario_indices(train_scenarios)
    val_idx = gather_scenario_indices(val_scenarios)

    # =========================================================================
    # Save transition data
    # =========================================================================
    train_data = {
        'z_k': z_k_all[train_idx],
        'z_nom': z_nom_all[train_idx],
        'obs': obs_all[train_idx],
        'safe_k': safe_k_all[train_idx],
        'safe_nom': safe_nom_all[train_idx],
    }
    val_data = {
        'z_k': z_k_all[val_idx],
        'z_nom': z_nom_all[val_idx],
        'obs': obs_all[val_idx],
        'safe_k': safe_k_all[val_idx],
        'safe_nom': safe_nom_all[val_idx],
    }

    torch.save(train_data, cfg.TRANSITIONS_TRAIN)
    logging.info(f"Saved training transitions: {len(train_idx)} → {cfg.TRANSITIONS_TRAIN}")

    torch.save(val_data, cfg.TRANSITIONS_VAL)
    logging.info(f"Saved validation transitions: {len(val_idx)} → {cfg.TRANSITIONS_VAL}")

    # =========================================================================
    # Extract and save state-label data from the SAME rollouts
    #
    # Each transition gives two state-label samples:
    #   (z_k, obs, safe_k)  and  (z_nom, obs, safe_nom)
    #
    # Label convention: safe_k=1 → label=0 (safe), safe_k=0 → label=1 (unsafe)
    # =========================================================================
    def extract_state_labels(indices):
        z_k_sub = z_k_all[indices]
        z_nom_sub = z_nom_all[indices]
        obs_sub = obs_all[indices]
        safe_k_sub = safe_k_all[indices]
        safe_nom_sub = safe_nom_all[indices]

        # Concatenate z_k and z_nom as separate state samples
        z_states = torch.cat([z_k_sub, z_nom_sub], dim=0)
        obs_states = torch.cat([obs_sub, obs_sub], dim=0)
        # Convert: safe=1 → label=0, unsafe=0 → label=1
        labels = torch.cat([1.0 - safe_k_sub, 1.0 - safe_nom_sub], dim=0)

        # Remove exact duplicates (z_nom at step k == z_k at step k+1 within
        # the same trajectory, so many will be duplicated). We deduplicate by
        # keeping unique rows. This is approximate but reduces redundancy.
        # For large datasets the overlap is acceptable so we skip dedup for speed.

        return {'z': z_states, 'obs': obs_states, 'label': labels}

    train_state_labels = extract_state_labels(train_idx)
    val_state_labels = extract_state_labels(val_idx)

    torch.save(train_state_labels, cfg.STATE_LABELS_TRAIN)
    n_safe_train = (train_state_labels['label'] == 0).sum().item()
    n_unsafe_train = (train_state_labels['label'] == 1).sum().item()
    logging.info(f"Saved training state-labels: {len(train_state_labels['z'])} "
                 f"(safe={n_safe_train}, unsafe={n_unsafe_train}) → {cfg.STATE_LABELS_TRAIN}")

    torch.save(val_state_labels, cfg.STATE_LABELS_VAL)
    n_safe_val = (val_state_labels['label'] == 0).sum().item()
    n_unsafe_val = (val_state_labels['label'] == 1).sum().item()
    logging.info(f"Saved validation state-labels: {len(val_state_labels['z'])} "
                 f"(safe={n_safe_val}, unsafe={n_unsafe_val}) → {cfg.STATE_LABELS_VAL}")

    logging.info(f"\n{'=' * 60}")
    logging.info("Data generation complete!")
    logging.info(f"  Transitions:  train={len(train_idx)}, val={len(val_idx)}")
    logging.info(f"  State-labels: train={len(train_state_labels['z'])}, "
                 f"val={len(val_state_labels['z'])}")
    logging.info(f"{'=' * 60}")


if __name__ == '__main__':
    main()

