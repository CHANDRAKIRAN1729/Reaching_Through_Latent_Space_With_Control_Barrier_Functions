"""
Fixed-Obstacle CBF Evaluation — Planning with B(z) safety filter.

Uses the SAME fixed obstacle and goal for all scenarios.
Only start configurations vary.
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
import cbf_fixed_config as fcfg

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


# =============================================================================
# FixedBarrierNet — same as in cbf_fixed_train.py
# =============================================================================
class FixedBarrierNet(nn.Module):
    """Neural Control Barrier Function B(z) for a FIXED obstacle."""

    def __init__(self, latent_dim=7, hidden_units=2048, num_hidden=4):
        super(FixedBarrierNet, self).__init__()
        self.latent_dim = latent_dim
        self.fc_in = nn.Linear(latent_dim, hidden_units)
        self.fc_hidden = nn.ModuleList(
            [nn.Linear(hidden_units, hidden_units) for _ in range(num_hidden - 1)]
        )
        self.fc_out = nn.Linear(hidden_units, 1)

    def forward(self, z):
        h = F.elu(self.fc_in(z.view(-1, self.latent_dim)))
        for fc in self.fc_hidden:
            h = F.elu(fc(h))
        return self.fc_out(h).view(-1)


# =============================================================================
# CBF Safety Correction — B(z) version (no obs)
# =============================================================================
def cbf_fixed_safety_correction(cbf_net, z_current, z_nominal, alpha, delta_t,
                                 lambda_max=1.0):
    """
    Single-step closed-form safe latent update for B(z).

    Same formula as cbf_model.py but without obs parameter:
        z_safe = z_nom + λ · ∇B(z_nom)
        λ = max(0, (B_target - B_nom) / ||∇B||²)
    """
    # Step 1: B(z_k)
    with torch.no_grad():
        B_current = cbf_net(z_current.detach())

    B_target = (1.0 - alpha * delta_t) * B_current

    # Step 2: B(z_nom) and ∇B(z_nom)
    z_nom = z_nominal.detach().clone().requires_grad_(True)
    B_nom = cbf_net(z_nom)
    B_nom_val = B_nom.item()

    # Check: is correction needed?
    if B_nom_val >= B_target.item():
        return z_nominal.detach().clone(), {
            'lambda_val': 0.0,
            'B_current': B_current.item(),
            'B_nominal': B_nom_val,
            'B_target': B_target.item(),
            'B_safe': B_nom_val,
            'correction_applied': False,
        }

    # Step 3: ∇B(z_nom)
    grad_B = torch.autograd.grad(
        B_nom, z_nom, grad_outputs=torch.ones_like(B_nom),
        create_graph=False, retain_graph=False
    )[0]

    d_norm_sq = torch.sum(grad_B ** 2) + 1e-8

    # Step 4: λ
    lambda_linear = ((B_target - B_nom) / d_norm_sq).item()
    lambda_val = min(max(0.0, lambda_linear), lambda_max)

    # Step 5: z_safe = z_nom + λ · ∇B
    z_safe = z_nom.detach() + lambda_val * grad_B.detach()

    with torch.no_grad():
        B_safe_val = cbf_net(z_safe).item()

    info = {
        'lambda_val': lambda_val,
        'B_current': B_current.item(),
        'B_nominal': B_nom_val,
        'B_target': B_target.item(),
        'B_safe': B_safe_val,
        'correction_applied': lambda_val > 0.0,
    }

    return z_safe.detach(), info


# =============================================================================
# Planning with B(z)
# =============================================================================
def plan_with_fixed_cbf(model, cbf_net,
                        q_start, e_start, e_target,
                        mean_train, std_train,
                        device, args):
    """Plan trajectory using Goal+Prior + fixed-obstacle CBF correction."""
    mean_train_t = torch.tensor(mean_train, dtype=torch.float32).to(device)
    std_train_t = torch.tensor(std_train, dtype=torch.float32).to(device)

    # Encode start
    x_start = torch.cat([q_start, e_start], dim=1)
    x_start_norm = (x_start - mean_train_t[:, :10]) / std_train_t[:, :10]
    with torch.no_grad():
        z_init = model.encoder(x_start_norm)[0]

    z = z_init.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([z], lr=args.planning_lr)

    # Tracking
    decoded_path = []
    latent_path = []
    min_dist = float('inf')
    goal_reached = False
    start_time = time.time()

    lambda_values = []
    B_current_values = []
    corrections_applied = []

    for step in range(args.max_steps):
        z_before = z.data.clone()

        optimizer.zero_grad()

        # Decode
        x_decoded_norm = model.decoder(z)
        x_decoded = x_decoded_norm * std_train_t[:, :10] + mean_train_t[:, :10]
        q_decoded = x_decoded[:, :7]
        e_decoded = x_decoded[:, 7:10]

        # Nominal loss
        L_goal = torch.norm(e_decoded - e_target)
        L_prior = 0.5 * torch.sum(z ** 2)
        L_nominal = L_goal + args.lambda_prior * L_prior

        L_nominal.backward()
        optimizer.step()

        # CBF safety filter — B(z) without obs
        z_safe, info = cbf_fixed_safety_correction(
            cbf_net, z_before, z.data,
            alpha=args.cbf_alpha, delta_t=args.cbf_delta_t,
            lambda_max=args.lambda_max
        )

        z.data = z_safe

        if info['correction_applied']:
            optimizer.state[z] = {}

        lambda_values.append(info['lambda_val'])
        B_current_values.append(info['B_current'])
        corrections_applied.append(info['correction_applied'])

        # Record waypoint
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

    planning_time = (time.time() - start_time) * 1000

    cbf_metrics = {}
    if lambda_values:
        cbf_metrics = {
            'avg_lambda': float(np.mean(lambda_values)),
            'max_lambda': float(np.max(lambda_values)),
            'intervention_rate': float(np.mean(corrections_applied)),
            'avg_B_current': float(np.mean(B_current_values)),
            'min_B_current': float(np.min(B_current_values)),
        }

    return {
        'goal_reached': goal_reached,
        'min_distance': min_dist,
        'planning_time_ms': planning_time,
        'path_length': compute_path_length(latent_path),
        'num_steps': len(decoded_path),
        'decoded_path': decoded_path,
        'cbf_metrics': cbf_metrics,
    }


# =============================================================================
# Main Evaluation
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Evaluate FIXED-obstacle CBF planning')
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=fcfg.SEED)
    parser.add_argument('--num_problems', type=int, default=100)
    parser.add_argument('--max_steps', type=int, default=fcfg.MAX_STEPS)
    parser.add_argument('--planning_lr', type=float, default=fcfg.PLANNING_LR)
    parser.add_argument('--lambda_prior', type=float, default=fcfg.LAMBDA_PRIOR)
    parser.add_argument('--success_threshold', type=float, default=fcfg.SUCCESS_THRESHOLD)
    parser.add_argument('--cbf_alpha', type=float, default=fcfg.CBF_ALPHA)
    parser.add_argument('--cbf_delta_t', type=float, default=fcfg.CBF_DELTA_T)
    parser.add_argument('--lambda_max', type=float, default=1.0)
    args = parser.parse_args()

    # Setup
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
    logging.info(f"VAE loaded from {cfg.VAE_CHECKPOINT}")

    # Load FixedBarrierNet
    cbf_net = FixedBarrierNet(
        latent_dim=fcfg.LATENT_DIM,
        hidden_units=fcfg.CBF_HIDDEN_UNITS,
        num_hidden=fcfg.CBF_NUM_HIDDEN
    )
    cbf_checkpoint = torch.load(fcfg.FIXED_BEST_CHECKPOINT, map_location=device, weights_only=False)
    cbf_net.load_state_dict(cbf_checkpoint['model_state_dict'])
    cbf_net.to(device)
    cbf_net.eval()
    logging.info(f"FixedBarrierNet loaded (epoch {cbf_checkpoint['epoch']})")

    # Normalization stats
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

    # Fixed obstacle and goal
    obstacle_raw = [np.array(fcfg.FIXED_OBSTACLE, dtype=np.float32)]
    q_target = torch.tensor([fcfg.FIXED_GOAL_Q_RAD], dtype=torch.float32, device=device)
    e_target = robot.FK(q_target.clone(), device, rad=True)

    logging.info(f"\n{'=' * 70}")
    logging.info(f"FIXED-OBSTACLE CBF EVALUATION")
    logging.info(f"  Obstacle: {fcfg.FIXED_OBSTACLE}")
    logging.info(f"  Goal ee:  {e_target.cpu().numpy()[0].tolist()}")
    logging.info(f"  Scenarios: {args.num_problems} (random start only)")
    logging.info(f"  CBF: α={args.cbf_alpha}, Δt={args.cbf_delta_t}")
    logging.info(f"{'=' * 70}\n")

    # =========================================================================
    # Evaluation loop
    # =========================================================================
    successes = 0
    goal_reached_count = 0
    collision_free_count = 0
    planning_times = []
    all_cbf_metrics = []

    for i in range(args.num_problems):
        # Random start only
        q_start = torch.rand(1, 7, device=device) * (q_max_rad - q_min_rad) + q_min_rad
        e_start = robot.FK(q_start.clone(), device, rad=True)

        # Plan with B(z)
        plan_result = plan_with_fixed_cbf(
            model, cbf_net,
            q_start, e_start, e_target,
            mean_train, std_train,
            device, args
        )

        # Validate with Robo3D
        is_collision_free, num_collisions, collision_waypoints = \
            validate_path_with_geometric_checker(
                plan_result['decoded_path'],
                obstacle_raw,
                robo3d
            )

        goal_reached = plan_result['goal_reached']
        success = goal_reached and is_collision_free

        if goal_reached:
            goal_reached_count += 1
        if is_collision_free:
            collision_free_count += 1
        if success:
            successes += 1
            planning_times.append(plan_result['planning_time_ms'])
        all_cbf_metrics.append(plan_result['cbf_metrics'])

        if (i + 1) % 10 == 0:
            sr = successes / (i + 1) * 100
            gr = goal_reached_count / (i + 1) * 100
            cf = collision_free_count / (i + 1) * 100
            logging.info(
                f"Progress: {i+1}/{args.num_problems} | "
                f"Success: {sr:.1f}% | Goal: {gr:.1f}% | "
                f"Collision-free: {cf:.1f}%"
            )

    # =========================================================================
    # Summary
    # =========================================================================
    n = args.num_problems
    success_rate = successes / n * 100
    goal_rate = goal_reached_count / n * 100
    cf_rate = collision_free_count / n * 100

    avg_lambda = np.mean([m.get('avg_lambda', 0) for m in all_cbf_metrics if m])
    avg_interv = np.mean([m.get('intervention_rate', 0) for m in all_cbf_metrics if m])

    logging.info(f"\n{'=' * 70}")
    logging.info("FIXED-OBSTACLE CBF EVALUATION RESULTS")
    logging.info(f"{'=' * 70}")
    logging.info(f"  Obstacle:         {fcfg.FIXED_OBSTACLE}")
    logging.info(f"  Test Scenarios:   {n}")
    logging.info(f"  Success Rate:     {success_rate:.1f}%")
    logging.info(f"  Goal Reached:     {goal_rate:.1f}%")
    logging.info(f"  Collision-free:   {cf_rate:.1f}%")
    if planning_times:
        logging.info(f"  Avg Planning Time: {np.mean(planning_times):.1f} ± {np.std(planning_times):.1f} ms")
    logging.info(f"  Avg λ:            {avg_lambda:.6f}")
    logging.info(f"  Intervention Rate: {avg_interv*100:.1f}%")
    logging.info(f"{'=' * 70}")


if __name__ == '__main__':
    main()
