"""
Multi-Position CBF Evaluation — Planning with B(z, x, y) safety filter.

Tests on all obstacle positions together (random position per scenario).
"""

from __future__ import print_function
import argparse
import json
import logging
import numpy as np
import os
import time
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

warnings.filterwarnings('ignore', category=FutureWarning, module='torch')

from vae import VAE
from robot_state_dataset import RobotStateDataset
from sim.panda import Panda
from sim.robot3d import Robo3D
from evaluate_planning import validate_path_with_geometric_checker, compute_path_length
import cbf_config as cfg
import cbf_multipos_config as mcfg

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


# =============================================================================
# MultiPosBarrierNet — same as in cbf_multipos_train.py
# =============================================================================
class MultiPosBarrierNet(nn.Module):
    def __init__(self, latent_dim=7, obs_xy_dim=2, hidden_units=2048, num_hidden=4):
        super(MultiPosBarrierNet, self).__init__()
        self.latent_dim = latent_dim
        self.obs_xy_dim = obs_xy_dim
        self.fc_in = nn.Linear(latent_dim + obs_xy_dim, hidden_units)
        self.fc_hidden = nn.ModuleList(
            [nn.Linear(hidden_units, hidden_units) for _ in range(num_hidden - 1)]
        )
        self.fc_out = nn.Linear(hidden_units, 1)

    def forward(self, z, obs_xy):
        x = torch.cat([z.view(-1, self.latent_dim), obs_xy.view(-1, self.obs_xy_dim)], dim=-1)
        h = F.elu(self.fc_in(x))
        for fc in self.fc_hidden:
            h = F.elu(fc(h))
        return self.fc_out(h).view(-1)


# =============================================================================
# CBF Safety Correction — B(z, x, y) version
# =============================================================================
def cbf_multipos_safety_correction(cbf_net, z_current, z_nominal, obs_xy,
                                    alpha, delta_t, lambda_max=1.0):
    """CBF correction with obs_xy conditioning."""
    with torch.no_grad():
        B_current = cbf_net(z_current.detach(), obs_xy)

    B_target = (1.0 - alpha * delta_t) * B_current

    z_nom = z_nominal.detach().clone().requires_grad_(True)
    B_nom = cbf_net(z_nom, obs_xy)
    B_nom_val = B_nom.item()

    if B_nom_val >= B_target.item():
        return z_nominal.detach().clone(), {
            'lambda_val': 0.0, 'B_current': B_current.item(),
            'B_nominal': B_nom_val, 'correction_applied': False,
        }

    grad_B = torch.autograd.grad(
        B_nom, z_nom, grad_outputs=torch.ones_like(B_nom),
        create_graph=False, retain_graph=False
    )[0]

    d_norm_sq = torch.sum(grad_B ** 2) + 1e-8
    lambda_linear = ((B_target - B_nom) / d_norm_sq).item()
    lambda_val = min(max(0.0, lambda_linear), lambda_max)

    z_safe = z_nom.detach() + lambda_val * grad_B.detach()

    with torch.no_grad():
        B_safe_val = cbf_net(z_safe, obs_xy).item()

    return z_safe.detach(), {
        'lambda_val': lambda_val, 'B_current': B_current.item(),
        'B_nominal': B_nom_val, 'B_safe': B_safe_val,
        'correction_applied': lambda_val > 0.0,
    }


# =============================================================================
# Planning with B(z, x, y)
# =============================================================================
def plan_with_multipos_cbf(model, cbf_net,
                           q_start, e_start, e_target, obs_xy_tensor,
                           mean_train, std_train, device, args):
    mean_train_t = torch.tensor(mean_train, dtype=torch.float32).to(device)
    std_train_t = torch.tensor(std_train, dtype=torch.float32).to(device)

    x_start = torch.cat([q_start, e_start], dim=1)
    x_start_norm = (x_start - mean_train_t[:, :10]) / std_train_t[:, :10]
    with torch.no_grad():
        z_init = model.encoder(x_start_norm)[0]

    z = z_init.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([z], lr=args.planning_lr)

    decoded_path = []
    latent_path = []
    min_dist = float('inf')
    goal_reached = False
    start_time = time.time()

    lambda_values = []
    corrections_applied = []

    for step in range(args.max_steps):
        z_before = z.data.clone()
        optimizer.zero_grad()

        x_decoded_norm = model.decoder(z)
        x_decoded = x_decoded_norm * std_train_t[:, :10] + mean_train_t[:, :10]
        e_decoded = x_decoded[:, 7:10]

        L_goal = torch.norm(e_decoded - e_target)
        L_prior = 0.5 * torch.sum(z ** 2)
        L_nominal = L_goal + args.lambda_prior * L_prior

        L_nominal.backward()
        optimizer.step()

        # CBF correction with obs_xy
        z_safe, info = cbf_multipos_safety_correction(
            cbf_net, z_before, z.data, obs_xy_tensor,
            alpha=args.cbf_alpha, delta_t=args.cbf_delta_t,
            lambda_max=args.lambda_max
        )
        z.data = z_safe

        if info['correction_applied']:
            optimizer.state[z] = {}

        lambda_values.append(info['lambda_val'])
        corrections_applied.append(info['correction_applied'])

        with torch.no_grad():
            x_safe_norm = model.decoder(z)
            x_safe = x_safe_norm * std_train_t[:, :10] + mean_train_t[:, :10]
            q_safe = x_safe[:, :7]
            e_safe = x_safe[:, 7:10]

        decoded_path.append({
            'q': q_safe.detach().cpu().numpy()[0],
            'e': e_safe.detach().cpu().numpy()[0],
            'step': step
        })
        latent_path.append(z.detach().cpu().numpy().copy())

        dist_to_goal = torch.norm(e_safe - e_target).item()
        min_dist = min(min_dist, dist_to_goal)

        if dist_to_goal < args.success_threshold:
            goal_reached = True
            break

    cbf_metrics = {}
    if lambda_values:
        cbf_metrics = {
            'avg_lambda': float(np.mean(lambda_values)),
            'max_lambda': float(np.max(lambda_values)),
            'intervention_rate': float(np.mean(corrections_applied)),
        }

    return {
        'goal_reached': goal_reached,
        'min_distance': min_dist,
        'planning_time_ms': (time.time() - start_time) * 1000,
        'path_length': compute_path_length(latent_path),
        'num_steps': len(decoded_path),
        'decoded_path': decoded_path,
        'cbf_metrics': cbf_metrics,
    }


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Evaluate multi-position CBF')
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=mcfg.SEED)
    parser.add_argument('--num_problems', type=int, default=100,
                        help='Total problems (distributed across positions)')
    parser.add_argument('--max_steps', type=int, default=mcfg.MAX_STEPS)
    parser.add_argument('--planning_lr', type=float, default=mcfg.PLANNING_LR)
    parser.add_argument('--lambda_prior', type=float, default=mcfg.LAMBDA_PRIOR)
    parser.add_argument('--success_threshold', type=float, default=mcfg.SUCCESS_THRESHOLD)
    parser.add_argument('--cbf_alpha', type=float, default=mcfg.CBF_ALPHA)
    parser.add_argument('--cbf_delta_t', type=float, default=mcfg.CBF_DELTA_T)
    parser.add_argument('--lambda_max', type=float, default=1.0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logging.info(f"Using device: {device}")

    # Load VAE
    with open(cfg.VAE_CONFIG, 'r') as f:
        vae_config = json.load(f)
        if 'parsed_args' in vae_config:
            vae_config = vae_config['parsed_args']
    model = VAE(
        vae_config['input_dim'], vae_config['latent_dim'],
        vae_config['units_per_layer'], vae_config['num_hidden_layers']
    )
    checkpoint = torch.load(cfg.VAE_CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Load MultiPosBarrierNet
    cbf_net = MultiPosBarrierNet(
        latent_dim=mcfg.LATENT_DIM,
        obs_xy_dim=mcfg.OBS_XY_DIM,
        hidden_units=mcfg.CBF_HIDDEN_UNITS,
        num_hidden=mcfg.CBF_NUM_HIDDEN
    )
    cbf_ckpt = torch.load(mcfg.MULTIPOS_BEST_CHECKPOINT, map_location=device, weights_only=False)
    cbf_net.load_state_dict(cbf_ckpt['model_state_dict'])
    cbf_net.to(device)
    cbf_net.eval()
    logging.info(f"MultiPosBarrierNet loaded (epoch {cbf_ckpt['epoch']})")

    # Normalization
    dataset = RobotStateDataset(
        cfg.DATA_PATH, train=0, train_data_name='free_space_100k_train.dat'
    )
    mean_train = dataset.get_mean_train()
    std_train = dataset.get_std_train()

    # Robot
    robot = Panda()
    robot.to(device)
    robo3d = Robo3D(Panda())

    q_min_rad = robot.joint_min_limits_tensor * (torch.pi / 180.0)
    q_max_rad = robot.joint_max_limits_tensor * (torch.pi / 180.0)

    # Fixed goal
    q_target = torch.tensor([mcfg.FIXED_GOAL_Q_RAD], dtype=torch.float32, device=device)
    e_target = robot.FK(q_target.clone(), device, rad=True)

    logging.info(f"\n{'=' * 70}")
    logging.info(f"MULTI-POSITION CBF EVALUATION")
    logging.info(f"  Positions: {mcfg.OBSTACLE_POSITIONS}")
    logging.info(f"  Shape: h={mcfg.FIXED_H}, r={mcfg.FIXED_R}")
    logging.info(f"  Goal ee: {e_target.cpu().numpy()[0].tolist()}")
    logging.info(f"  Scenarios: {args.num_problems}")
    logging.info(f"  CBF: α={args.cbf_alpha}, Δt={args.cbf_delta_t}")
    logging.info(f"{'=' * 70}\n")

    # Per-position tracking
    pos_stats = {tuple(pos): {'success': 0, 'goal': 0, 'cf': 0, 'total': 0}
                 for pos in mcfg.OBSTACLE_POSITIONS}
    overall = {'success': 0, 'goal': 0, 'cf': 0}

    for i in range(args.num_problems):
        # Cycle through positions
        pos_idx = i % len(mcfg.OBSTACLE_POSITIONS)
        obs_xy = mcfg.OBSTACLE_POSITIONS[pos_idx]
        obstacle_xyhr = mcfg.OBSTACLES_FULL[pos_idx]

        obs_xy_tensor = torch.tensor(obs_xy, dtype=torch.float32).unsqueeze(0).to(device)
        obstacle_raw = [np.array(obstacle_xyhr, dtype=np.float32)]

        # Random start
        q_start = torch.rand(1, 7, device=device) * (q_max_rad - q_min_rad) + q_min_rad
        e_start = robot.FK(q_start.clone(), device, rad=True)

        # Plan with B(z, x, y)
        plan_result = plan_with_multipos_cbf(
            model, cbf_net,
            q_start, e_start, e_target, obs_xy_tensor,
            mean_train, std_train, device, args
        )

        # Validate
        is_cf, _, _ = validate_path_with_geometric_checker(
            plan_result['decoded_path'], obstacle_raw, robo3d
        )

        goal = plan_result['goal_reached']
        success = goal and is_cf

        # Track
        key = tuple(obs_xy)
        pos_stats[key]['total'] += 1
        if goal:
            pos_stats[key]['goal'] += 1
            overall['goal'] += 1
        if is_cf:
            pos_stats[key]['cf'] += 1
            overall['cf'] += 1
        if success:
            pos_stats[key]['success'] += 1
            overall['success'] += 1

        if (i + 1) % 50 == 0:
            n = i + 1
            logging.info(
                f"Progress: {n}/{args.num_problems} | "
                f"Success: {overall['success']/n*100:.1f}% | "
                f"Goal: {overall['goal']/n*100:.1f}% | "
                f"CF: {overall['cf']/n*100:.1f}%"
            )

    # =========================================================================
    # Summary
    # =========================================================================
    n = args.num_problems
    logging.info(f"\n{'=' * 70}")
    logging.info("MULTI-POSITION CBF EVALUATION RESULTS")
    logging.info(f"{'=' * 70}")
    logging.info(f"  Overall ({n} scenarios):")
    logging.info(f"    Success Rate:    {overall['success']/n*100:.1f}%")
    logging.info(f"    Goal Reached:    {overall['goal']/n*100:.1f}%")
    logging.info(f"    Collision-free:  {overall['cf']/n*100:.1f}%")

    logging.info(f"\n  Per-position breakdown:")
    for pos, stats in pos_stats.items():
        t = max(stats['total'], 1)
        logging.info(
            f"    Obs ({pos[0]:.1f}, {pos[1]:.1f}): "
            f"Success={stats['success']/t*100:.1f}% "
            f"Goal={stats['goal']/t*100:.1f}% "
            f"CF={stats['cf']/t*100:.1f}% "
            f"(n={stats['total']})"
        )
    logging.info(f"{'=' * 70}")


if __name__ == '__main__':
    main()
