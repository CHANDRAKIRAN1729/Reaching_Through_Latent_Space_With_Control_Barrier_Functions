"""
New Evaluation Script for "Reaching Through Latent Space"

Implements the CORRECT two-stage pipeline as described by the paper's author:

  Stage 1 - PLANNING (gradient-based optimization in latent space):
    Uses the LEARNED CLASSIFIER (model_obs) for collision avoidance.
    The classifier is differentiable, enabling backpropagation through
    L_collision to steer the latent trajectory away from obstacles.

  Stage 2 - EVALUATION (ground-truth collision validation):
    Uses Robo3D(Panda).check_for_collision() — the geometric capsule-based
    collision checker — to validate every waypoint in the planned trajectory.
    This is the author's recommended approach (email correspondence).

Author's instruction:
  "To check whether there is any collision with the obstacles in a path
   (a sequence of joint angles), one can instantiate a Robo3D with Panda
   as the definition and use the function check_for_collision."

Paper Reference: Section V - Experiments
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
import torch.optim as optim
from torch import nn
import torch.nn.functional as F

# Suppress FutureWarnings from torch.load
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')

from robot_state_dataset import RobotStateDataset
from robot_obs_dataset import RobotObstacleDataset
from vae import VAE
from vae_obs import VAEObstacleBCE
from sim.panda import Panda
from sim.robot3d import Robo3D

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

FMAX = 1e5


def convert_to_json_serializable(obj):
    """Recursively convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(item) for item in obj)
    else:
        return obj


class ObstacleScenarioGenerator:
    """
    Generates obstacle scenarios as described in paper Section IV-C:
    - First obstacle between start and goal
    - Subsequent obstacles: 50% between start-goal, 50% random
    """
    MIN_DIST_FROM_BASE = 0.3  # meters

    def __init__(self, robot, workspace_bounds=None):
        self.robot = robot
        if workspace_bounds is None:
            self.workspace_bounds = {
                'x': (0.4, 0.8),
                'y': (-0.5, 0.5),
                'z': (0.0, 1.0)
            }
        else:
            self.workspace_bounds = workspace_bounds

    def sample_obstacle_between_points(self, p1, p2):
        """Sample obstacle on line segment between p1 and p2."""
        max_attempts = 20
        for _ in range(max_attempts):
            t = np.random.uniform(0.3, 0.7)
            pos = p1 + t * (p2 - p1)
            x, y = pos[0], pos[1]
            dist_from_base = np.sqrt(x**2 + y**2)
            if dist_from_base >= self.MIN_DIST_FROM_BASE:
                h = np.random.uniform(0.5, 1.0)
                r = np.random.uniform(0.05, 0.15)
                return np.array([x, y, h, r], dtype=np.float32)
        return self.sample_random_obstacle()

    def sample_random_obstacle(self):
        """Sample random obstacle in avoidable region of workspace."""
        max_attempts = 50
        for _ in range(max_attempts):
            x = np.random.uniform(*self.workspace_bounds['x'])
            y = np.random.uniform(*self.workspace_bounds['y'])
            dist_from_base = np.sqrt(x**2 + y**2)
            if dist_from_base >= self.MIN_DIST_FROM_BASE:
                h = np.random.uniform(0.5, 1.0)
                r = np.random.uniform(0.05, 0.15)
                return np.array([x, y, h, r], dtype=np.float32)
        return np.array([0.6, 0.0, 0.7, 0.1], dtype=np.float32)

    def generate_scenario(self, q_start, e_start, e_target, num_obstacles=1):
        """Generate obstacle scenario ensuring obstacles are between start and goal."""
        obstacles = []
        if num_obstacles == 0:
            return obstacles
        # First obstacle always between start and goal
        obstacles.append(self.sample_obstacle_between_points(e_start, e_target))
        # Subsequent: 50% between, 50% random
        for i in range(1, num_obstacles):
            if np.random.random() < 0.5:
                obstacles.append(self.sample_obstacle_between_points(e_start, e_target))
            else:
                obstacles.append(self.sample_random_obstacle())
        return obstacles


def compute_path_length(latent_path):
    """Compute path length in latent space."""
    if len(latent_path) < 2:
        return 0.0
    total_length = 0.0
    for i in range(1, len(latent_path)):
        total_length += np.linalg.norm(latent_path[i] - latent_path[i-1])
    return total_length


# =============================================================================
# STAGE 1: PLANNING — Gradient-based optimization using learned classifier
# =============================================================================

def plan_with_latent_optimization(
    model, model_obs,
    q_start, e_start, e_target,
    obstacles_normalized,
    mean_train, std_train,
    device, args):
    """
    Path planning via latent space optimization (Algorithm 2 from paper).

    Uses three loss terms:
    - L_goal:      distance to target end-effector position
    - L_collision: collision penalty from LEARNED CLASSIFIER (differentiable)
    - L_prior:     keep latent near origin (regularization)

    Returns:
        dict with planning results including decoded_path (sequence of joint angles)
    """
    mean_train_t = torch.tensor(mean_train, dtype=torch.float32).to(device)
    std_train_t = torch.tensor(std_train, dtype=torch.float32).to(device)

    # Normalize start state
    x_start = torch.cat([q_start, e_start], dim=1)  # [1, 10]
    x_start_norm = (x_start - mean_train_t[:, :10]) / std_train_t[:, :10]

    # Get initial latent code
    with torch.no_grad():
        z_init = model.encoder(x_start_norm)[0]

    z = z_init.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([z], lr=args.planning_lr)

    # Track path
    latent_path = [z.detach().cpu().numpy().copy()]
    decoded_path = []  # list of {'q': array(7,), 'e': array(3,), 'step': int}

    # Normalize goal
    e_target_norm = (e_target - mean_train_t[:, 7:10]) / std_train_t[:, 7:10]

    # Convert normalized obstacles to tensors
    obs_tensors = [torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                   for obs in obstacles_normalized]

    # GECO state
    lambda_prior = args.lambda_prior
    lambda_collision = args.lambda_collision
    C_prior_ma = None
    C_collision_ma = None

    start_time = time.time()
    min_dist = FMAX
    goal_reached = False

    for step in range(args.max_steps):
        optimizer.zero_grad()

        # Decode latent to configuration
        x_decoded_norm = model.decoder(z)
        x_decoded = x_decoded_norm * std_train_t[:, :10] + mean_train_t[:, :10]
        q_decoded = x_decoded[:, :7]
        e_decoded = x_decoded[:, 7:10]

        # --- Loss 1: Goal reaching (Eq. 4) ---
        L_goal = torch.norm(e_decoded - e_target)

        # --- Loss 2: Prior (Eq. 5) ---
        L_prior = 0.5 * torch.sum(z ** 2)

        # --- Loss 3: Collision from LEARNED CLASSIFIER (Eq. 6) ---
        L_collision = torch.tensor(0.0, device=device)
        if model_obs is not None and len(obstacles_normalized) > 0:
            for obs_tensor in obs_tensors:
                logit = model_obs.obstacle_collision_classifier(z, obs_tensor)
                p_collision = torch.sigmoid(logit / args.temperature)
                L_collision = L_collision + (-torch.log(1 - p_collision + 1e-8))
            L_collision = L_collision / len(obstacles_normalized)

        # Log initial losses
        if step == 0:
            logging.info(f"Initial losses: L_goal={L_goal.item():.4f}, "
                        f"L_prior={L_prior.item():.4f}, L_collision={L_collision.item():.4f}")

        # --- GECO adaptive lambda (Algorithm 2) ---
        if args.use_geco:
            C_prior = L_prior.item() - args.tau_prior_goal
            C_collision = L_collision.item() - args.tau_obs_goal

            if step == 0:
                C_prior_ma = C_prior
                C_collision_ma = C_collision
            else:
                C_prior_ma = args.alpha_ma_prior * C_prior_ma + (1 - args.alpha_ma_prior) * C_prior
                C_collision_ma = args.alpha_ma_obs * C_collision_ma + (1 - args.alpha_ma_obs) * C_collision

            kappa_prior = np.exp(args.alpha_geco * C_prior_ma)
            kappa_collision = np.exp(args.alpha_geco * C_collision_ma)

            lambda_prior = np.clip(kappa_prior * lambda_prior, 1e-6, 1000.0)
            lambda_collision = np.clip(kappa_collision * lambda_collision, 1e-6, 1000.0)

            if step % 50 == 0:
                logging.info(f"Step {step}: lambda_prior={lambda_prior:.4f}, lambda_collision={lambda_collision:.4f}, "
                             f"L_prior={L_prior.item():.4f}, L_collision={L_collision.item():.4f}")

        # Combined loss
        L_total = L_goal + lambda_prior * L_prior + lambda_collision * L_collision

        # Check distance to goal
        dist_to_goal = L_goal.item()
        min_dist = min(min_dist, dist_to_goal)

        # Record waypoint
        decoded_path.append({
            'q': q_decoded.detach().cpu().numpy()[0],
            'e': e_decoded.detach().cpu().numpy()[0],
            'step': step
        })
        latent_path.append(z.detach().cpu().numpy().copy())

        if dist_to_goal < args.success_threshold:
            goal_reached = True
            break

        # Gradient descent
        L_total.backward()
        optimizer.step()

    end_time = time.time()
    planning_time = (end_time - start_time) * 1000
    path_length = compute_path_length(latent_path)

    return {
        'goal_reached': goal_reached,
        'planning_time_ms': planning_time,
        'path_length': path_length,
        'min_distance': min_dist,
        'num_steps': step + 1,
        'decoded_path': decoded_path,
        'latent_path': latent_path,
        'final_ee_pos': decoded_path[-1]['e'] if decoded_path else e_start.cpu().numpy()[0]
    }


# =============================================================================
# STAGE 2: EVALUATION — Ground-truth collision checking using Robo3D
# =============================================================================

def validate_path_with_geometric_checker(decoded_path, obstacles_raw, robo3d):
    """
    Validate every waypoint in the planned trajectory using the geometric
    capsule-based collision checker (Robo3D.check_for_collision).

    This is the author's recommended evaluation method.

    Args:
        decoded_path: list of dicts with 'q' (joint angles in RADIANS, shape (7,))
        obstacles_raw: list of obstacle parameters [x, y, h, r] (raw, NOT normalized)
        robo3d: Robo3D instance (instantiated with Panda definition)

    Returns:
        is_path_collision_free: bool — True if NO waypoint collides
        num_collisions: int — number of waypoints that collide
        collision_waypoints: list of step indices where collision occurred
    """
    if len(obstacles_raw) == 0:
        return True, 0, []

    # Robo3D.check_for_collision expects:
    #   jpos: list of joint angles in DEGREES
    #   obstacles_xyhr: list of [x, y, h, r]
    obstacles_xyhr = [obs.tolist() if isinstance(obs, np.ndarray) else obs
                      for obs in obstacles_raw]

    collision_waypoints = []
    for waypoint in decoded_path:
        q_rad = waypoint['q']  # shape (7,), in radians
        q_deg = np.degrees(q_rad).tolist()  # convert to degrees for Robo3D

        if robo3d.check_for_collision(q_deg, obstacles_xyhr):
            collision_waypoints.append(waypoint['step'])

    num_collisions = len(collision_waypoints)
    is_collision_free = (num_collisions == 0)
    return is_collision_free, num_collisions, collision_waypoints


# =============================================================================
# MAIN EVALUATION LOOP — Combines Stage 1 + Stage 2
# =============================================================================

def evaluate_path_planning(model, model_obs, robot, robo3d,
                           mean_train, std_train, mean_obs, std_obs,
                           device, args):
    """
    Main evaluation function.

    For each scenario:
      1. Plan a path using latent-space optimization (learned classifier for gradients)
      2. Validate the path using Robo3D.check_for_collision (geometric ground truth)
    """
    model.eval()
    if model_obs is not None:
        model_obs.eval()

    scenario_gen = ObstacleScenarioGenerator(robot)

    results_list = []
    successes = 0
    goal_reached_count = 0
    collision_free_count = 0
    planning_times = []
    path_lengths = []
    min_distances = []

    logging.info("=" * 70)
    logging.info(f"Evaluating {args.num_problems} planning scenarios")
    logging.info(f"Success threshold: {args.success_threshold}m")
    logging.info(f"Max planning steps: {args.max_steps}")
    logging.info(f"Number of obstacles: {args.num_obstacles}")
    logging.info(f"GECO adaptive weighting: {'ENABLED' if args.use_geco else 'DISABLED'}")
    logging.info(f"Lambda prior (initial): {args.lambda_prior}")
    logging.info(f"Lambda collision (initial): {args.lambda_collision}")
    if args.use_geco:
        logging.info(f"  GECO alpha: {args.alpha_geco}")
        logging.info(f"  tau_prior: {args.tau_prior_goal}, tau_obs: {args.tau_obs_goal}")
        logging.info(f"  alpha_ma_prior: {args.alpha_ma_prior}, alpha_ma_obs: {args.alpha_ma_obs}")
    logging.info(f"Collision evaluation: Robo3D geometric checker (author-recommended)")
    logging.info("=" * 70)

    # Load or generate scenarios
    loaded_scenarios = None
    if args.load_scenes:
        with open(args.load_scenes, 'r') as f:
            loaded_scenarios = json.load(f)
        logging.info(f"Loaded {len(loaded_scenarios['scenarios'])} scenarios from {args.load_scenes}")
        args.num_problems = len(loaded_scenarios['scenarios'])

    scenarios_to_save = []

    for i in range(args.num_problems):

        # --- Generate / load scenario ---
        if loaded_scenarios:
            scenario = loaded_scenarios['scenarios'][i]
            q_start = torch.tensor(scenario['q_start'], device=device, dtype=torch.float32).unsqueeze(0)
            e_start = torch.tensor(scenario['e_start'], device=device, dtype=torch.float32).unsqueeze(0)
            q_target = torch.tensor(scenario['q_target'], device=device, dtype=torch.float32).unsqueeze(0)
            e_target = torch.tensor(scenario['e_target'], device=device, dtype=torch.float32).unsqueeze(0)
            obstacles_raw = [np.array(obs) for obs in scenario['obstacles']]
        else:
            q_min_rad = robot.joint_min_limits_tensor * (torch.pi / 180.0)
            q_max_rad = robot.joint_max_limits_tensor * (torch.pi / 180.0)
            q_start = torch.rand(1, 7, device=device) * (q_max_rad - q_min_rad) + q_min_rad
            e_start = robot.FK(q_start.clone(), device, rad=True)
            q_target = torch.rand(1, 7, device=device) * (q_max_rad - q_min_rad) + q_min_rad
            e_target = robot.FK(q_target.clone(), device, rad=True)
            obstacles_raw = scenario_gen.generate_scenario(
                q_start.cpu().numpy()[0],
                e_start.cpu().numpy()[0],
                e_target.cpu().numpy()[0],
                num_obstacles=args.num_obstacles
            )

        # Save scenario for reproducibility
        if args.save_scenes:
            scenarios_to_save.append({
                'scenario_id': i,
                'q_start': q_start.cpu().numpy()[0].tolist(),
                'e_start': e_start.cpu().numpy()[0].tolist(),
                'q_target': q_target.cpu().numpy()[0].tolist(),
                'e_target': e_target.cpu().numpy()[0].tolist(),
                'obstacles': [obs.tolist() for obs in obstacles_raw]
            })

        # Normalize obstacles for the learned classifier (Stage 1)
        obstacles_normalized = [(obs - mean_obs) / std_obs for obs in obstacles_raw]

        # =============================================================
        # STAGE 1: Plan path (learned classifier provides gradients)
        # =============================================================
        plan_result = plan_with_latent_optimization(
            model, model_obs,
            q_start, e_start, e_target,
            obstacles_normalized,
            mean_train, std_train,
            device, args
        )

        # =============================================================
        # STAGE 2: Validate path with geometric ground truth
        #   Robo3D(Panda).check_for_collision — author's recommendation
        # =============================================================
        is_collision_free, num_collisions, collision_waypoints = \
            validate_path_with_geometric_checker(
                plan_result['decoded_path'],
                obstacles_raw,       # raw (un-normalized) obstacles
                robo3d
            )

        # --- Determine success ---
        goal_reached = plan_result['goal_reached']
        success = goal_reached and is_collision_free

        # --- Accumulate statistics ---
        if goal_reached:
            goal_reached_count += 1
        if is_collision_free:
            collision_free_count += 1
        if success:
            successes += 1
            planning_times.append(plan_result['planning_time_ms'])
            path_lengths.append(plan_result['path_length'])
        min_distances.append(plan_result['min_distance'])

        # Store per-scenario result
        results_list.append({
            'scenario_id': i,
            'num_obstacles': len(obstacles_raw),
            'start_ee': e_start.cpu().numpy()[0].tolist(),
            'target_ee': e_target.cpu().numpy()[0].tolist(),
            'obstacles': [obs.tolist() for obs in obstacles_raw],
            'goal_reached': goal_reached,
            'success': success,
            'is_collision_free': is_collision_free,
            'num_collision_waypoints': num_collisions,
            'collision_waypoint_steps': collision_waypoints,
            'planning_time_ms': plan_result['planning_time_ms'],
            'path_length': plan_result['path_length'],
            'min_distance': plan_result['min_distance'],
            'num_steps': plan_result['num_steps'],
            'final_ee_pos': plan_result['final_ee_pos'].tolist()
                           if isinstance(plan_result['final_ee_pos'], np.ndarray)
                           else plan_result['final_ee_pos'],
        })

        # Log progress
        if (i + 1) % 100 == 0:
            sr = successes / (i + 1) * 100
            gr = goal_reached_count / (i + 1) * 100
            cf = collision_free_count / (i + 1) * 100
            logging.info(
                f"Progress: {i+1}/{args.num_problems} | "
                f"Success: {sr:.1f}% | Goal Reached: {gr:.1f}% | "
                f"Collision-free (Robo3D): {cf:.1f}%"
            )
        elif (i + 1) % 10 == 0 and success:
            logging.info(
                f"Scenario {i+1}: SUCCESS in {plan_result['num_steps']} steps, "
                f"{plan_result['planning_time_ms']:.2f}ms"
            )

    # --- Summary statistics ---
    n = args.num_problems
    success_rate = successes / n * 100
    goal_reached_rate = goal_reached_count / n * 100
    collision_free_rate = collision_free_count / n * 100
    avg_time = np.mean(planning_times) if planning_times else 0
    std_time = np.std(planning_times) if planning_times else 0
    avg_path = np.mean(path_lengths) if path_lengths else 0
    std_path = np.std(path_lengths) if path_lengths else 0
    avg_min_dist = np.mean(min_distances)

    results_summary = {
        'num_problems': n,
        'num_obstacles': args.num_obstacles,
        'success_threshold_m': args.success_threshold,
        'max_steps': args.max_steps,
        'use_geco': args.use_geco,
        'lambda_prior': args.lambda_prior,
        'lambda_collision': args.lambda_collision,
        'collision_checker': 'Robo3D.check_for_collision (geometric ground truth)',
        'successes': successes,
        'success_rate_percent': success_rate,
        'goal_reached_count': goal_reached_count,
        'goal_reached_rate_percent': goal_reached_rate,
        'collision_free_count': collision_free_count,
        'collision_free_rate_percent': collision_free_rate,
        'avg_planning_time_ms': avg_time,
        'std_planning_time_ms': std_time,
        'avg_path_length': avg_path,
        'std_path_length': std_path,
        'avg_min_distance_m': avg_min_dist,
        'planning_times_ms': planning_times,
        'path_lengths': path_lengths,
        'min_distances': min_distances,
        'detailed_results': results_list
    }

    # Save scenarios for reproducibility
    if args.save_scenes and scenarios_to_save:
        scenes_data = {
            'num_scenarios': len(scenarios_to_save),
            'num_obstacles': args.num_obstacles,
            'seed': args.seed,
            'scenarios': scenarios_to_save
        }
        with open(args.save_scenes, 'w') as f:
            json.dump(scenes_data, f, indent=2)
        logging.info(f"Saved {len(scenarios_to_save)} scenarios to {args.save_scenes}")

    return results_summary


def print_results(results):
    """Print results in paper-ready format."""

    logging.info("\n" + "=" * 70)
    logging.info("PATH PLANNING EVALUATION RESULTS")
    logging.info("=" * 70)
    logging.info(f"Configuration:")
    logging.info(f"  Test Scenarios:        {results['num_problems']}")
    logging.info(f"  Obstacles per scene:   {results['num_obstacles']}")
    logging.info(f"  Success threshold:     {results['success_threshold_m']}m")
    logging.info(f"  Max planning steps:    {results['max_steps']}")
    logging.info(f"  GECO adaptive weights: {'ENABLED' if results['use_geco'] else 'DISABLED'}")
    logging.info(f"  Lambda prior (init):   {results['lambda_prior']}")
    logging.info(f"  Lambda collision (init): {results['lambda_collision']}")
    logging.info(f"  Collision checker:     {results['collision_checker']}")
    logging.info("-" * 70)
    logging.info(f"Results:")
    logging.info(f"  Successful plans (goal + collision-free): {results['successes']}")
    logging.info(f"  Success Rate:          {results['success_rate_percent']:.2f}%")
    logging.info(f"  Goal Reached:          {results['goal_reached_count']}")
    logging.info(f"  Goal Reached Rate:     {results['goal_reached_rate_percent']:.2f}%")
    logging.info(f"  Collision-free paths:  {results['collision_free_count']}")
    logging.info(f"  Collision-free Rate:   {results['collision_free_rate_percent']:.2f}%")
    logging.info(f"  Avg Planning Time:     {results['avg_planning_time_ms']:.2f} +/- {results['std_planning_time_ms']:.2f} ms")
    logging.info(f"  Avg Path Length:       {results['avg_path_length']:.4f} +/- {results['std_path_length']:.4f}")
    logging.info(f"  Avg Min Distance:      {results['avg_min_distance_m']:.4f}m")
    logging.info("=" * 70 + "\n")

    logging.info("Comparison with Paper Results:")
    logging.info("  Paper (1 obstacle, 5mm threshold):")
    logging.info("    Success Rate: ~90%")
    logging.info("    Planning Time: ~180ms")
    logging.info(f"  Your Results:")
    logging.info(f"    Success Rate: {results['success_rate_percent']:.2f}%")
    logging.info(f"    Goal Reached: {results['goal_reached_rate_percent']:.2f}%")
    logging.info(f"    Planning Time: {results['avg_planning_time_ms']:.2f}ms")
    logging.info("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='New evaluation for Reaching Through Latent Space (author-recommended pipeline)'
    )

    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to VAE checkpoint')
    parser.add_argument('--checkpoint_obs', type=str, required=True,
                       help='Path to collision classifier checkpoint')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to VAE training config JSON file')
    parser.add_argument('--config_obs', type=str, required=True,
                       help='Path to collision classifier training config JSON file')

    # Evaluation arguments
    parser.add_argument('--num_problems', type=int, default=1000,
                       help='Number of planning scenarios (paper uses 1000)')
    parser.add_argument('--num_obstacles', type=int, default=1,
                       help='Number of obstacles per scenario')
    parser.add_argument('--max_steps', type=int, default=300,
                       help='Maximum planning steps per problem')
    parser.add_argument('--planning_lr', type=float, default=0.03,
                       help='Learning rate for latent space optimization')
    parser.add_argument('--success_threshold', type=float, default=0.01,
                       help='Distance threshold for success (meters)')

    # Loss weights
    parser.add_argument('--lambda_prior', type=float, default=0.01,
                       help='Initial weight for prior loss')
    parser.add_argument('--lambda_collision', type=float, default=0.5,
                       help='Initial weight for collision loss')

    # GECO parameters
    parser.add_argument('--use_geco', action='store_true', default=False,
                       help='Enable GECO adaptive loss weighting')
    parser.add_argument('--alpha_geco', type=float, default=0.01,
                       help='GECO learning rate')
    parser.add_argument('--tau_prior_goal', type=float, default=5.0,
                       help='GECO prior loss target')
    parser.add_argument('--tau_obs_goal', type=float, default=18.0,
                       help='GECO obstacle loss target')
    parser.add_argument('--alpha_ma_prior', type=float, default=0.95,
                       help='Moving average factor for prior loss')
    parser.add_argument('--alpha_ma_obs', type=float, default=0.4,
                       help='Moving average factor for obstacle loss')

    # Temperature
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Temperature for collision classifier sigmoid')

    # Data
    parser.add_argument('--data_path', type=str, default='../data',
                       help='Path to dataset')

    # Output
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file for results')

    # Scene save/load
    parser.add_argument('--save_scenes', type=str, default=None,
                       help='Save generated scenarios to JSON file')
    parser.add_argument('--load_scenes', type=str, default=None,
                       help='Load pre-generated scenarios from JSON file')

    # Device
    parser.add_argument('--no_cuda', action='store_true', default=False,
                       help='Disable CUDA')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    logging.info(f"Using device: {device}")

    # Load configs
    with open(args.config, 'r') as f:
        config = json.load(f)
        if 'parsed_args' in config:
            config = config['parsed_args']

    with open(args.config_obs, 'r') as f:
        config_obs = json.load(f)
        if 'parsed_args' in config_obs:
            config_obs = config_obs['parsed_args']

    # Get normalization parameters
    dataset = RobotStateDataset(
        args.data_path, train=0,
        train_data_name='free_space_100k_train.dat'
    )
    mean_train = dataset.get_mean_train()
    std_train = dataset.get_std_train()

    obs_dataset = RobotObstacleDataset(
        args.data_path, train=0,
        train_data_name='collision_100k_train.dat',
        test_data_name='collision_10k_test.dat',
        free_space_train_name='free_space_100k_train.dat',
        free_space_test_name='free_space_10k_test.dat'
    )
    mean_obs = obs_dataset.get_mean_train()[0, 10:14]
    std_obs = obs_dataset.get_std_train()[0, 10:14]
    logging.info(f"Obstacle normalization: mean={mean_obs}, std={std_obs}")

    # Initialize Panda (for FK and joint limits — torch-based)
    robot = Panda()
    robot.to(device)

    # =========================================================================
    # KEY: Instantiate Robo3D with Panda definition (author's recommendation)
    # This is the geometric collision checker used in Stage 2 (evaluation).
    # =========================================================================
    robo3d = Robo3D(Panda())
    logging.info("Robo3D geometric collision checker instantiated with Panda definition")

    # Load VAE model
    logging.info(f"Loading VAE from {args.checkpoint}")
    model = VAE(
        config['input_dim'],
        config['latent_dim'],
        config['units_per_layer'],
        config['num_hidden_layers']
    )
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning)
        checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    logging.info(f"VAE loaded: {config['units_per_layer']} units, {config['latent_dim']}D latent")

    # Load collision classifier
    logging.info(f"Loading collision classifier from {args.checkpoint_obs}")
    model_obs = VAEObstacleBCE(
        config['input_dim'],
        config['latent_dim'],
        config['units_per_layer'],
        config['num_hidden_layers']
    )
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning)
        checkpoint_obs = torch.load(args.checkpoint_obs, map_location=device)
    model_obs.load_state_dict(checkpoint_obs['model_state_dict'])
    model_obs.to(device)
    model_obs.eval()
    logging.info("Collision classifier loaded")

    # Copy encoder weights from VAE to classifier (ensure consistency)
    model_obs.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # Run evaluation
    logging.info("\nStarting evaluation (two-stage pipeline)...")
    logging.info("  Stage 1: Planning with learned classifier (differentiable)")
    logging.info("  Stage 2: Validation with Robo3D.check_for_collision (geometric ground truth)")
    logging.info("")

    results = evaluate_path_planning(
        model, model_obs, robot, robo3d,
        mean_train, std_train, mean_obs, std_obs,
        device, args
    )

    # Print results
    print_results(results)

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            results_to_save = {k: v for k, v in results.items() if k != 'detailed_results'}
            results_to_save = convert_to_json_serializable(results_to_save)
            json.dump(results_to_save, f, indent=2)
        logging.info(f"Results saved to {args.output}")

        detailed_output = args.output.replace('.json', '_detailed.json')
        with open(detailed_output, 'w') as f:
            results_serializable = convert_to_json_serializable(results)
            json.dump(results_serializable, f, indent=2)
        logging.info(f"Detailed results saved to {detailed_output}")


if __name__ == "__main__":
    main()
