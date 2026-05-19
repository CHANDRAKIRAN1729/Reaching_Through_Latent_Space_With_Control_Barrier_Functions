"""
Fixed-Obstacle CBF Data Generation.

Generates transition + state-label data using ONE fixed obstacle and ONE fixed goal.
Only the start configuration varies across scenarios.

Outputs (all into cbf_fixed_data/):
    - transitions_train.pt: {z_k, z_nom, safe_k, safe_nom}   (NO obs)
    - transitions_val.pt:   same format
    - state_labels_train.pt: {z, label}                       (NO obs)
    - state_labels_val.pt:   same format
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
from sim.panda import Panda
from sim.robot3d import Robo3D
import cbf_config as cfg
import cbf_fixed_config as fcfg

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
        q_start, e_start, e_target, obstacle_xyhr,
        mean_train_t, std_train_t,
        device, planning_lr, lambda_prior, max_steps):
    """
    Run the Goal+Prior-only planner and record transitions.
    Uses the FIXED obstacle for collision checking.

    Returns:
        transitions: list of dicts with 'z_k', 'z_nom', 'safe_k', 'safe_nom' (no obs)
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
        if L_goal.item() < fcfg.SUCCESS_THRESHOLD:
            break

        L_nominal.backward()
        optimizer.step()

        # z now contains z_{k+1}^nom
        z_after = z.data.clone()

        # Check collision status using Robo3D with the FIXED obstacle
        with torch.no_grad():
            # Decode z_before
            x_before_norm = model.decoder(z_before)
            x_before = x_before_norm * std_train_t[:, :10] + mean_train_t[:, :10]
            q_before = x_before[:, :7].cpu().numpy()[0]
            q_before_deg = np.degrees(q_before).tolist()
            safe_k = 0.0 if robo3d.check_for_collision(q_before_deg, [obstacle_xyhr]) else 1.0

            # Decode z_after
            x_after_norm = model.decoder(z_after)
            x_after = x_after_norm * std_train_t[:, :10] + mean_train_t[:, :10]
            q_after = x_after[:, :7].cpu().numpy()[0]
            q_after_deg = np.degrees(q_after).tolist()
            safe_nom = 0.0 if robo3d.check_for_collision(q_after_deg, [obstacle_xyhr]) else 1.0

        transitions.append({
            'z_k': z_before.cpu().squeeze(0),
            'z_nom': z_after.cpu().squeeze(0),
            'safe_k': safe_k,
            'safe_nom': safe_nom,
        })

    return transitions


def main():
    parser = argparse.ArgumentParser(description='Generate FIXED-obstacle CBF data')
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=fcfg.SEED)
    parser.add_argument('--num_scenarios', type=int, default=fcfg.NUM_SCENARIOS)
    parser.add_argument('--max_steps', type=int, default=fcfg.MAX_STEPS)
    parser.add_argument('--planning_lr', type=float, default=fcfg.PLANNING_LR)
    parser.add_argument('--lambda_prior', type=float, default=fcfg.LAMBDA_PRIOR)
    parser.add_argument('--val_split', type=float, default=0.1)
    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logging.info(f"Using device: {device}")

    # Create output directory
    os.makedirs(fcfg.FIXED_DATA_DIR, exist_ok=True)

    # Load models
    model = load_frozen_vae(device)
    mean_train, std_train = get_normalization_stats()
    mean_train_t = torch.tensor(mean_train, dtype=torch.float32).to(device)
    std_train_t = torch.tensor(std_train, dtype=torch.float32).to(device)

    # Initialize robot
    robot = Panda()
    robot.to(device)
    robo3d = Robo3D(Panda())

    q_min_rad = robot.joint_min_limits_tensor * (torch.pi / 180.0)
    q_max_rad = robot.joint_max_limits_tensor * (torch.pi / 180.0)

    # Fixed obstacle and goal
    obstacle_xyhr = fcfg.FIXED_OBSTACLE
    q_target = torch.tensor([fcfg.FIXED_GOAL_Q_RAD], dtype=torch.float32, device=device)
    e_target = robot.FK(q_target.clone(), device, rad=True)

    logging.info(f"\n{'=' * 60}")
    logging.info(f"FIXED-OBSTACLE DATA GENERATION")
    logging.info(f"  Obstacle: {obstacle_xyhr}")
    logging.info(f"  Goal q:   {fcfg.FIXED_GOAL_Q_RAD}")
    logging.info(f"  Goal ee:  {e_target.cpu().numpy()[0].tolist()}")
    logging.info(f"  Scenarios: {args.num_scenarios}")
    logging.info(f"{'=' * 60}\n")

    # =========================================================================
    # Generate transitions
    # =========================================================================
    all_transitions = []
    scenario_boundaries = [0]
    start_time = time.time()

    for scenario_id in range(args.num_scenarios):
        # Random start only — obstacle and goal are FIXED
        q_start = torch.rand(1, 7, device=device) * (q_max_rad - q_min_rad) + q_min_rad
        e_start = robot.FK(q_start.clone(), device, rad=True)

        # Run nominal planner and record transitions
        transitions = run_nominal_planner_and_record_transitions(
            model, robot, robo3d,
            q_start, e_start, e_target, obstacle_xyhr,
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
    # Collate into tensors (NO obs field)
    # =========================================================================
    z_k_all = torch.stack([t['z_k'] for t in all_transitions])
    z_nom_all = torch.stack([t['z_nom'] for t in all_transitions])
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
    # Split by scenario
    # =========================================================================
    num_scenarios = len(scenario_boundaries) - 1
    n_val_scenarios = max(1, int(num_scenarios * args.val_split))
    n_train_scenarios = num_scenarios - n_val_scenarios

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
    # Save transition data (NO obs)
    # =========================================================================
    train_data = {
        'z_k': z_k_all[train_idx],
        'z_nom': z_nom_all[train_idx],
        'safe_k': safe_k_all[train_idx],
        'safe_nom': safe_nom_all[train_idx],
    }
    val_data = {
        'z_k': z_k_all[val_idx],
        'z_nom': z_nom_all[val_idx],
        'safe_k': safe_k_all[val_idx],
        'safe_nom': safe_nom_all[val_idx],
    }

    torch.save(train_data, fcfg.FIXED_TRANSITIONS_TRAIN)
    logging.info(f"Saved training transitions: {len(train_idx)} → {fcfg.FIXED_TRANSITIONS_TRAIN}")

    torch.save(val_data, fcfg.FIXED_TRANSITIONS_VAL)
    logging.info(f"Saved validation transitions: {len(val_idx)} → {fcfg.FIXED_TRANSITIONS_VAL}")

    # =========================================================================
    # Extract and save state-label data (NO obs)
    # =========================================================================
    def extract_state_labels(indices):
        z_k_sub = z_k_all[indices]
        z_nom_sub = z_nom_all[indices]
        safe_k_sub = safe_k_all[indices]
        safe_nom_sub = safe_nom_all[indices]

        z_states = torch.cat([z_k_sub, z_nom_sub], dim=0)
        # Convert: safe=1 → label=0, unsafe=0 → label=1
        labels = torch.cat([1.0 - safe_k_sub, 1.0 - safe_nom_sub], dim=0)

        return {'z': z_states, 'label': labels}

    train_state_labels = extract_state_labels(train_idx)
    val_state_labels = extract_state_labels(val_idx)

    torch.save(train_state_labels, fcfg.FIXED_STATE_LABELS_TRAIN)
    n_safe_train = (train_state_labels['label'] == 0).sum().item()
    n_unsafe_train = (train_state_labels['label'] == 1).sum().item()
    logging.info(f"Saved training state-labels: {len(train_state_labels['z'])} "
                 f"(safe={n_safe_train}, unsafe={n_unsafe_train}) → {fcfg.FIXED_STATE_LABELS_TRAIN}")

    torch.save(val_state_labels, fcfg.FIXED_STATE_LABELS_VAL)
    n_safe_val = (val_state_labels['label'] == 0).sum().item()
    n_unsafe_val = (val_state_labels['label'] == 1).sum().item()
    logging.info(f"Saved validation state-labels: {len(val_state_labels['z'])} "
                 f"(safe={n_safe_val}, unsafe={n_unsafe_val}) → {fcfg.FIXED_STATE_LABELS_VAL}")

    logging.info(f"\n{'=' * 60}")
    logging.info("Data generation complete!")
    logging.info(f"  Transitions:  train={len(train_idx)}, val={len(val_idx)}")
    logging.info(f"  State-labels: train={len(train_state_labels['z'])}, "
                 f"val={len(val_state_labels['z'])}")
    logging.info(f"{'=' * 60}")


if __name__ == '__main__':
    main()
