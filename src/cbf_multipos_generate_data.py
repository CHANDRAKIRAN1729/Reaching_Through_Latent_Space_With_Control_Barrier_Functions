"""
Multi-Position CBF Data Generation.

Generates data across 3-4 obstacle positions (fixed h, r).
Each scenario randomly picks one position. The obs_xy = [x, y] is recorded.

Outputs:
    - transitions_train/val.pt: {z_k, z_nom, obs_xy, safe_k, safe_nom}
    - state_labels_train/val.pt: {z, obs_xy, label}
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
import cbf_multipos_config as mcfg

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
    return dataset.get_mean_train(), dataset.get_std_train()


def run_nominal_planner_and_record_transitions(
        model, robot, robo3d,
        q_start, e_start, e_target, obstacle_xyhr, obs_xy,
        mean_train_t, std_train_t,
        device, planning_lr, lambda_prior, max_steps):
    """
    Run Goal+Prior planner and record transitions with obs_xy.

    Returns:
        transitions: list of dicts with 'z_k', 'z_nom', 'obs_xy', 'safe_k', 'safe_nom'
    """
    x_start = torch.cat([q_start, e_start], dim=1)
    x_start_norm = (x_start - mean_train_t[:, :10]) / std_train_t[:, :10]

    with torch.no_grad():
        z_init = model.encoder(x_start_norm)[0]

    z = z_init.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([z], lr=planning_lr)

    transitions = []
    obs_xy_tensor = torch.tensor(obs_xy, dtype=torch.float32)

    for step in range(max_steps):
        z_before = z.data.clone()
        optimizer.zero_grad()

        x_decoded_norm = model.decoder(z)
        x_decoded = x_decoded_norm * std_train_t[:, :10] + mean_train_t[:, :10]
        e_decoded = x_decoded[:, 7:10]

        L_goal = torch.norm(e_decoded - e_target)
        L_prior = 0.5 * torch.sum(z ** 2)
        L_nominal = L_goal + lambda_prior * L_prior

        if L_goal.item() < mcfg.SUCCESS_THRESHOLD:
            break

        L_nominal.backward()
        optimizer.step()

        z_after = z.data.clone()

        # Collision check with full obstacle [x, y, h, r]
        with torch.no_grad():
            x_before_norm = model.decoder(z_before)
            x_before = x_before_norm * std_train_t[:, :10] + mean_train_t[:, :10]
            q_before = x_before[:, :7].cpu().numpy()[0]
            q_before_deg = np.degrees(q_before).tolist()
            safe_k = 0.0 if robo3d.check_for_collision(q_before_deg, [obstacle_xyhr]) else 1.0

            x_after_norm = model.decoder(z_after)
            x_after = x_after_norm * std_train_t[:, :10] + mean_train_t[:, :10]
            q_after = x_after[:, :7].cpu().numpy()[0]
            q_after_deg = np.degrees(q_after).tolist()
            safe_nom = 0.0 if robo3d.check_for_collision(q_after_deg, [obstacle_xyhr]) else 1.0

        transitions.append({
            'z_k': z_before.cpu().squeeze(0),
            'z_nom': z_after.cpu().squeeze(0),
            'obs_xy': obs_xy_tensor.clone(),
            'safe_k': safe_k,
            'safe_nom': safe_nom,
        })

    return transitions


def main():
    parser = argparse.ArgumentParser(description='Generate multi-position CBF data')
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=mcfg.SEED)
    parser.add_argument('--num_scenarios_per_pos', type=int, default=mcfg.NUM_SCENARIOS,
                        help='Number of scenarios PER obstacle position')
    parser.add_argument('--max_steps', type=int, default=mcfg.MAX_STEPS)
    parser.add_argument('--planning_lr', type=float, default=mcfg.PLANNING_LR)
    parser.add_argument('--lambda_prior', type=float, default=mcfg.LAMBDA_PRIOR)
    parser.add_argument('--val_split', type=float, default=0.1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logging.info(f"Using device: {device}")

    os.makedirs(mcfg.MULTIPOS_DATA_DIR, exist_ok=True)

    model = load_frozen_vae(device)
    mean_train, std_train = get_normalization_stats()
    mean_train_t = torch.tensor(mean_train, dtype=torch.float32).to(device)
    std_train_t = torch.tensor(std_train, dtype=torch.float32).to(device)

    robot = Panda()
    robot.to(device)
    robo3d = Robo3D(Panda())

    q_min_rad = robot.joint_min_limits_tensor * (torch.pi / 180.0)
    q_max_rad = robot.joint_max_limits_tensor * (torch.pi / 180.0)

    # Fixed goal
    q_target = torch.tensor([mcfg.FIXED_GOAL_Q_RAD], dtype=torch.float32, device=device)
    e_target = robot.FK(q_target.clone(), device, rad=True)

    num_positions = len(mcfg.OBSTACLE_POSITIONS)
    total_scenarios = args.num_scenarios_per_pos * num_positions

    logging.info(f"\n{'=' * 60}")
    logging.info(f"MULTI-POSITION DATA GENERATION")
    logging.info(f"  Obstacle positions: {mcfg.OBSTACLE_POSITIONS}")
    logging.info(f"  Fixed shape: h={mcfg.FIXED_H}, r={mcfg.FIXED_R}")
    logging.info(f"  Goal ee: {e_target.cpu().numpy()[0].tolist()}")
    logging.info(f"  Scenarios per position: {args.num_scenarios_per_pos}")
    logging.info(f"  Total scenarios: {total_scenarios}")
    logging.info(f"{'=' * 60}\n")

    all_transitions = []
    scenario_boundaries = [0]
    start_time = time.time()
    global_scenario = 0

    for pos_idx, (obs_x, obs_y) in enumerate(mcfg.OBSTACLE_POSITIONS):
        obstacle_xyhr = [obs_x, obs_y, mcfg.FIXED_H, mcfg.FIXED_R]
        obs_xy = [obs_x, obs_y]

        logging.info(f"Position {pos_idx+1}/{num_positions}: obstacle={obstacle_xyhr}")

        for scenario_id in range(args.num_scenarios_per_pos):
            q_start = torch.rand(1, 7, device=device) * (q_max_rad - q_min_rad) + q_min_rad
            e_start = robot.FK(q_start.clone(), device, rad=True)

            transitions = run_nominal_planner_and_record_transitions(
                model, robot, robo3d,
                q_start, e_start, e_target, obstacle_xyhr, obs_xy,
                mean_train_t, std_train_t,
                device, args.planning_lr, args.lambda_prior, args.max_steps
            )

            all_transitions.extend(transitions)
            scenario_boundaries.append(len(all_transitions))
            global_scenario += 1

            if global_scenario % 500 == 0:
                elapsed = time.time() - start_time
                rate = global_scenario / elapsed
                logging.info(f"  Progress: {global_scenario}/{total_scenarios} "
                             f"({rate:.1f} scenarios/s), "
                             f"transitions: {len(all_transitions)}")

    elapsed = time.time() - start_time
    logging.info(f"\nGenerated {len(all_transitions)} transitions from "
                 f"{total_scenarios} scenarios in {elapsed:.1f}s")

    if len(all_transitions) == 0:
        logging.error("No transitions generated!")
        return

    # =========================================================================
    # Collate (with obs_xy)
    # =========================================================================
    z_k_all = torch.stack([t['z_k'] for t in all_transitions])
    z_nom_all = torch.stack([t['z_nom'] for t in all_transitions])
    obs_xy_all = torch.stack([t['obs_xy'] for t in all_transitions])
    safe_k_all = torch.tensor([t['safe_k'] for t in all_transitions], dtype=torch.float32)
    safe_nom_all = torch.tensor([t['safe_nom'] for t in all_transitions], dtype=torch.float32)

    total = len(all_transitions)
    safe_k_count = (safe_k_all == 1).sum().item()
    unsafe_k_count = (safe_k_all == 0).sum().item()

    logging.info(f"\nTransition statistics:")
    logging.info(f"  z_k: safe={safe_k_count} ({safe_k_count/total*100:.1f}%), "
                 f"unsafe={unsafe_k_count} ({unsafe_k_count/total*100:.1f}%)")

    # =========================================================================
    # Split by scenario
    # =========================================================================
    num_scenarios = len(scenario_boundaries) - 1
    n_val = max(1, int(num_scenarios * args.val_split))
    n_train = num_scenarios - n_val

    perm = torch.randperm(num_scenarios).tolist()
    train_scenarios = perm[:n_train]
    val_scenarios = perm[n_train:]

    def gather(scenario_ids):
        indices = []
        for s in scenario_ids:
            indices.extend(range(scenario_boundaries[s], scenario_boundaries[s + 1]))
        return indices

    train_idx = gather(train_scenarios)
    val_idx = gather(val_scenarios)

    # =========================================================================
    # Save transitions (with obs_xy)
    # =========================================================================
    for name, idx, path in [('train', train_idx, mcfg.MULTIPOS_TRANSITIONS_TRAIN),
                            ('val', val_idx, mcfg.MULTIPOS_TRANSITIONS_VAL)]:
        data = {
            'z_k': z_k_all[idx],
            'z_nom': z_nom_all[idx],
            'obs_xy': obs_xy_all[idx],
            'safe_k': safe_k_all[idx],
            'safe_nom': safe_nom_all[idx],
        }
        torch.save(data, path)
        logging.info(f"Saved {name} transitions: {len(idx)} → {path}")

    # =========================================================================
    # Extract and save state-labels (with obs_xy)
    # =========================================================================
    def extract_state_labels(indices):
        z_k_sub = z_k_all[indices]
        z_nom_sub = z_nom_all[indices]
        obs_sub = obs_xy_all[indices]
        safe_k_sub = safe_k_all[indices]
        safe_nom_sub = safe_nom_all[indices]

        z_states = torch.cat([z_k_sub, z_nom_sub], dim=0)
        obs_states = torch.cat([obs_sub, obs_sub], dim=0)
        labels = torch.cat([1.0 - safe_k_sub, 1.0 - safe_nom_sub], dim=0)
        return {'z': z_states, 'obs_xy': obs_states, 'label': labels}

    for name, idx, path in [('train', train_idx, mcfg.MULTIPOS_STATE_LABELS_TRAIN),
                            ('val', val_idx, mcfg.MULTIPOS_STATE_LABELS_VAL)]:
        sl = extract_state_labels(idx)
        torch.save(sl, path)
        n_safe = (sl['label'] == 0).sum().item()
        n_unsafe = (sl['label'] == 1).sum().item()
        logging.info(f"Saved {name} state-labels: {len(sl['z'])} "
                     f"(safe={n_safe}, unsafe={n_unsafe}) → {path}")

    logging.info(f"\n{'=' * 60}")
    logging.info("Data generation complete!")
    logging.info(f"  Transitions: train={len(train_idx)}, val={len(val_idx)}")
    logging.info(f"{'=' * 60}")


if __name__ == '__main__':
    main()
