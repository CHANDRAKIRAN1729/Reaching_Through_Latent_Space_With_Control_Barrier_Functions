"""
Fixed-Obstacle Classifier Evaluation — Planning with C(z).

Plans using Goal + Prior + Collision(C(z)) losses.
Fixed obstacle, fixed goal, random starts.
Validates with Robo3D ground-truth collision checker.

Analogous to cbf_fixed_evaluate.py but using classifier instead of CBF.
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
import classifier_fixed_config as clcfg

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


# =============================================================================
# FixedCollisionClassifier C(z) — same as in training script
# =============================================================================
class FixedCollisionClassifier(nn.Module):
    def __init__(self, latent_dim=7, hidden_units=2048, num_hidden=4):
        super(FixedCollisionClassifier, self).__init__()
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
# Planning with Goal + Prior + C(z) collision loss
# =============================================================================
def plan_with_fixed_classifier(model, classifier,
                                q_start, e_start, e_target,
                                mean_train, std_train,
                                device, args):
    """
    Path planning using fixed collision classifier C(z).

    Three loss terms:
        L_goal:      ||ee_decoded - ee_target||
        L_collision:  -log(1 - sigmoid(C(z)/T))
        L_prior:     0.5 * ||z||^2
    """
    mean_train_t = torch.tensor(mean_train, dtype=torch.float32).to(device)
    std_train_t = torch.tensor(std_train, dtype=torch.float32).to(device)

    # Encode start
    x_start = torch.cat([q_start, e_start], dim=1)
    x_start_norm = (x_start - mean_train_t[:, :10]) / std_train_t[:, :10]
    with torch.no_grad():
        z_init = model.encoder(x_start_norm)[0]

    z = z_init.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([z], lr=args.planning_lr)

    latent_path = [z.detach().cpu().numpy().copy()]
    decoded_path = []
    min_dist = 1e5
    goal_reached = False
    start_time = time.time()

    for step in range(args.max_steps):
        optimizer.zero_grad()

        # Decode
        x_decoded_norm = model.decoder(z)
        x_decoded = x_decoded_norm * std_train_t[:, :10] + mean_train_t[:, :10]
        q_decoded = x_decoded[:, :7]
        e_decoded = x_decoded[:, 7:10]

        # Loss 1: Goal
        L_goal = torch.norm(e_decoded - e_target)

        # Loss 2: Prior
        L_prior = 0.5 * torch.sum(z ** 2)

        # Loss 3: Collision from C(z)
        logit = classifier(z)
        p_collision = torch.sigmoid(logit / args.temperature)
        L_collision = -torch.log(1 - p_collision + 1e-8)

        # Combined loss
        L_total = L_goal + args.lambda_prior * L_prior + \
                  args.lambda_collision * L_collision

        dist_to_goal = L_goal.item()
        min_dist = min(min_dist, dist_to_goal)

        decoded_path.append({
            'q': q_decoded.detach().cpu().numpy()[0],
            'e': e_decoded.detach().cpu().numpy()[0],
            'step': step
        })
        latent_path.append(z.detach().cpu().numpy().copy())

        if dist_to_goal < args.success_threshold:
            goal_reached = True
            break

        L_total.backward()
        optimizer.step()

    planning_time = (time.time() - start_time) * 1000

    return {
        'goal_reached': goal_reached,
        'min_distance': min_dist,
        'planning_time_ms': planning_time,
        'path_length': compute_path_length(latent_path),
        'num_steps': len(decoded_path),
        'decoded_path': decoded_path,
    }


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Evaluate fixed-obstacle classifier C(z)')
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=clcfg.SEED)
    parser.add_argument('--num_problems', type=int, default=500)
    parser.add_argument('--max_steps', type=int, default=300)
    parser.add_argument('--planning_lr', type=float, default=0.03)
    parser.add_argument('--success_threshold', type=float, default=clcfg.SUCCESS_THRESHOLD)
    parser.add_argument('--lambda_prior', type=float, default=0.01)
    parser.add_argument('--lambda_collision', type=float, default=0.5)
    parser.add_argument('--temperature', type=float, default=1.0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logging.info(f"Using device: {device}")

    # =========================================================================
    # Load VAE
    # =========================================================================
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

    # =========================================================================
    # Load fixed classifier C(z)
    # =========================================================================
    classifier = FixedCollisionClassifier(
        latent_dim=clcfg.LATENT_DIM,
        hidden_units=clcfg.HIDDEN_UNITS,
        num_hidden=clcfg.NUM_HIDDEN
    )
    cl_ckpt = torch.load(clcfg.BEST_CHECKPOINT, map_location=device, weights_only=False)
    classifier.load_state_dict(cl_ckpt['model_state_dict'])
    classifier.to(device)
    classifier.eval()
    logging.info(f"FixedCollisionClassifier loaded (epoch {cl_ckpt['epoch']}, "
                 f"val_acc={cl_ckpt['val_accuracy']*100:.1f}%)")

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
    q_target = torch.tensor([clcfg.FIXED_GOAL_Q_RAD], dtype=torch.float32, device=device)
    e_target = robot.FK(q_target.clone(), device, rad=True)

    obstacle_raw = np.array(clcfg.FIXED_OBSTACLE, dtype=np.float32)

    logging.info(f"\n{'=' * 70}")
    logging.info(f"FIXED-OBSTACLE CLASSIFIER EVALUATION")
    logging.info(f"  Obstacle: {clcfg.FIXED_OBSTACLE}")
    logging.info(f"  Goal ee:  {e_target.cpu().numpy()[0].tolist()}")
    logging.info(f"  Scenarios: {args.num_problems} (random start only)")
    logging.info(f"  Planner: Goal + Prior + C(z) collision loss")
    logging.info(f"  λ_prior={args.lambda_prior}, λ_collision={args.lambda_collision}, "
                 f"T={args.temperature}")
    logging.info(f"{'=' * 70}\n")

    # =========================================================================
    # Evaluation loop
    # =========================================================================
    successes = 0
    goal_reached_count = 0
    collision_free_count = 0
    planning_times = []

    for i in range(args.num_problems):
        # Random start
        q_start = torch.rand(1, 7, device=device) * (q_max_rad - q_min_rad) + q_min_rad
        e_start = robot.FK(q_start.clone(), device, rad=True)

        # Plan with C(z)
        plan_result = plan_with_fixed_classifier(
            model, classifier,
            q_start, e_start, e_target,
            mean_train, std_train,
            device, args
        )

        # Validate with Robo3D
        is_cf, _, _ = validate_path_with_geometric_checker(
            plan_result['decoded_path'],
            [obstacle_raw],
            robo3d
        )

        goal = plan_result['goal_reached']
        success = goal and is_cf

        if goal:
            goal_reached_count += 1
        if is_cf:
            collision_free_count += 1
        if success:
            successes += 1
            planning_times.append(plan_result['planning_time_ms'])

        if (i + 1) % 50 == 0:
            n = i + 1
            logging.info(
                f"Progress: {n}/{args.num_problems} | "
                f"Success: {successes/n*100:.1f}% | "
                f"Goal: {goal_reached_count/n*100:.1f}% | "
                f"Collision-free: {collision_free_count/n*100:.1f}%"
            )

    # =========================================================================
    # Summary
    # =========================================================================
    n = args.num_problems
    logging.info(f"\n{'=' * 70}")
    logging.info("FIXED-OBSTACLE CLASSIFIER EVALUATION RESULTS")
    logging.info(f"{'=' * 70}")
    logging.info(f"  Method:           Collision Classifier C(z) (baseline, fixed)")
    logging.info(f"  Obstacle:         {clcfg.FIXED_OBSTACLE}")
    logging.info(f"  Test Scenarios:   {n}")
    logging.info(f"  Success Rate:     {successes/n*100:.1f}%")
    logging.info(f"  Goal Reached:     {goal_reached_count/n*100:.1f}%")
    logging.info(f"  Collision-free:   {collision_free_count/n*100:.1f}%")
    if planning_times:
        logging.info(f"  Avg Planning Time: {np.mean(planning_times):.1f} ± "
                     f"{np.std(planning_times):.1f} ms")
    logging.info(f"{'=' * 70}")


if __name__ == '__main__':
    main()
