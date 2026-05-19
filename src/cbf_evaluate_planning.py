"""
CBF Planning & Evaluation — Latent-space planning with CBF safety filter.

Replaces the collision classifier with the CBF safety correction.
Same two-stage pipeline as evaluate_planning.py:
    Stage 1: Planning (Goal + Prior nominal → CBF safety correction)
    Stage 2: Validation (Robo3D geometric ground truth)

Key change from baseline:
    BASELINE:  L_total = L_goal + λ_prior·L_prior + λ_collision·L_collision
               optimizer.step()

    CBF:       L_nominal = L_goal + λ_prior·L_prior  (NO collision loss)
               optimizer.step()  →  z_nom
               z_safe = CBF_correction(z_nom)  →  z ← z_safe
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

warnings.filterwarnings('ignore', category=FutureWarning, module='torch')

from vae import VAE
from robot_state_dataset import RobotStateDataset
from robot_obs_dataset import RobotObstacleDataset
from sim.panda import Panda
from sim.robot3d import Robo3D
from evaluate_planning import (
    ObstacleScenarioGenerator,
    validate_path_with_geometric_checker,
    compute_path_length,
    convert_to_json_serializable,
)
from cbf_model import BarrierNet, cbf_safety_correction
import cbf_config as cfg

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


# =============================================================================
# STAGE 1: PLANNING — Goal+Prior nominal planner + CBF safety filter
# =============================================================================

def plan_with_cbf(model, cbf_net,
                  q_start, e_start, e_target,
                  obstacles_raw,
                  mean_train, std_train,
                  device, args):
    """
    Plan a trajectory using Goal+Prior nominal planner + CBF safety filter.

    At each optimization step:
        1. Compute L_goal + L_prior  (NO collision loss)
        2. optimizer.step()  →  z_{k+1}^{nom}
        3. CBF safety correction  →  z_{k+1}^{safe}
        4. z ← z_{k+1}^{safe}

    Args:
        model:       Frozen VAE (encoder + decoder)
        cbf_net:     Trained BarrierNet
        q_start:     (1, 7) start joint angles (radians)
        e_start:     (1, 3) start end-effector position
        e_target:    (1, 3) target end-effector position
        obstacles_raw: list of [x, y, h, r] obstacle parameters (raw, unnormalized)
        mean_train:  normalization mean
        std_train:   normalization std
        device:      torch device
        args:        planning parameters

    Returns:
        dict with planning results + CBF-specific metrics
    """
    mean_train_t = torch.tensor(mean_train, dtype=torch.float32).to(device)
    std_train_t = torch.tensor(std_train, dtype=torch.float32).to(device)

    # Encode start configuration
    x_start = torch.cat([q_start, e_start], dim=1)
    x_start_norm = (x_start - mean_train_t[:, :10]) / std_train_t[:, :10]

    with torch.no_grad():
        # Use sampled z (index [0]), NOT mu (index [1]).
        # The baseline uses encoder(x)[0] which includes reparameterization noise.
        # Using mu would place z in a different distribution than training.
        z_init = model.encoder(x_start_norm)[0]

    z = z_init.clone().detach().requires_grad_(True)
    # Use Adam (same as baseline) for fast convergence.
    # When CBF intervenes (λ > 0), we reset Adam's momentum state to avoid
    # state corruption from the external z.data overwrite.
    optimizer = optim.Adam([z], lr=args.planning_lr)

    # Prepare obstacle tensor for CBF
    obs_tensors = [
        torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        for obs in obstacles_raw
    ]

    # Tracking
    decoded_path = []
    latent_path = []
    min_dist = float('inf')
    goal_reached = False
    start_time = time.time()

    # CBF-specific metrics
    lambda_values = []
    B_current_values = []
    B_nominal_values = []
    corrections_applied = []
    rejections = []

    for step in range(args.max_steps):
        # Save z_k BEFORE the optimizer step
        z_before = z.data.clone()

        optimizer.zero_grad()

        # Decode
        x_decoded_norm = model.decoder(z)
        x_decoded = x_decoded_norm * std_train_t[:, :10] + mean_train_t[:, :10]
        q_decoded = x_decoded[:, :7]
        e_decoded = x_decoded[:, 7:10]

        # --- NOMINAL LOSS: Goal + Prior only (NO collision term!) ---
        L_goal = torch.norm(e_decoded - e_target)
        L_prior = 0.5 * torch.sum(z ** 2)
        L_nominal = L_goal + args.lambda_prior * L_prior

        # Log initial losses
        if step == 0:
            logging.debug(f"Initial losses: L_goal={L_goal.item():.4f}, "
                          f"L_prior={L_prior.item():.4f}")

        # Gradient descent
        L_nominal.backward()
        optimizer.step()

        # z now contains z_{k+1}^{nom}

        # --- CBF SAFETY FILTER ---
        if cbf_net is not None and len(obstacles_raw) > 0:
            # Apply CBF correction for each obstacle, take the most conservative
            max_lambda = 0.0
            best_z_safe = z.data.clone()
            agg_info = {'B_current': 0.0, 'B_nominal': 0.0, 'lambda_val': 0.0,
                        'correction_applied': False}

            for obs_tensor in obs_tensors:
                z_safe, info = cbf_safety_correction(
                    cbf_net, z_before, z.data, obs_tensor,
                    alpha=args.cbf_alpha, delta_t=args.cbf_delta_t,
                    lambda_max=args.lambda_max,
                    safe_threshold=args.cbf_safe_threshold,
                    max_iters=args.cbf_max_iters
                )

                if info['lambda_val'] > max_lambda:
                    max_lambda = info['lambda_val']
                    best_z_safe = z_safe
                    agg_info = info

            # Apply the most conservative correction
            z.data = best_z_safe

            # Reset Adam's internal state when CBF intervenes.
            # Adam's momentum (m) and variance (v) estimates become stale
            # after z.data is externally modified. Resetting prevents the
            # optimizer from fighting against the CBF correction.
            if agg_info['correction_applied']:
                optimizer.state[z] = {}

            # Record CBF metrics
            lambda_values.append(agg_info['lambda_val'])
            B_current_values.append(agg_info['B_current'])
            B_nominal_values.append(agg_info['B_nominal'])
            corrections_applied.append(agg_info['correction_applied'])
            rejections.append(agg_info.get('rejected', False))

        # Record waypoint (after CBF correction)
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

    planning_time = (time.time() - start_time) * 1000  # ms
    path_length = compute_path_length(latent_path)

    # CBF summary metrics
    cbf_metrics = {}
    if lambda_values:
        cbf_metrics = {
            'avg_lambda': float(np.mean(lambda_values)),
            'max_lambda': float(np.max(lambda_values)),
            'intervention_rate': float(np.mean(corrections_applied)),
            'num_interventions': int(sum(corrections_applied)),
            'rejection_rate': float(np.mean(rejections)) if rejections else 0.0,
            'num_rejections': int(sum(rejections)) if rejections else 0,
            'avg_B_current': float(np.mean(B_current_values)),
            'min_B_current': float(np.min(B_current_values)),
            'avg_B_nominal': float(np.mean(B_nominal_values)),
            'min_B_nominal': float(np.min(B_nominal_values)),
        }

    return {
        'goal_reached': goal_reached,
        'min_distance': min_dist,
        'planning_time_ms': planning_time,
        'path_length': path_length,
        'num_steps': len(decoded_path),
        'decoded_path': decoded_path,
        'latent_path': latent_path,
        'final_ee_pos': e_safe.detach().cpu().numpy()[0] if len(decoded_path) > 0 else None,
        'cbf_metrics': cbf_metrics,
    }


# =============================================================================
# MAIN EVALUATION LOOP — Stage 1 (CBF planning) + Stage 2 (Robo3D validation)
# =============================================================================

def evaluate_cbf_planning(model, cbf_net, robot, robo3d,
                          mean_train, std_train,
                          device, args):
    """
    Main evaluation function with CBF safety filter.

    For each scenario:
        1. Plan using Goal+Prior + CBF safety filter
        2. Validate with Robo3D.check_for_collision (geometric ground truth)
    """
    model.eval()
    cbf_net.eval()

    scenario_gen = ObstacleScenarioGenerator(robot)

    results_list = []
    successes = 0
    goal_reached_count = 0
    collision_free_count = 0
    planning_times = []
    path_lengths = []
    min_distances = []
    all_cbf_metrics = []

    logging.info("=" * 70)
    logging.info(f"CBF Evaluation: {args.num_problems} planning scenarios")
    logging.info(f"Planner: Goal + Prior only (lr={args.planning_lr}, λ_prior={args.lambda_prior})")
    logging.info(f"Safety: CBF filter (α={args.cbf_alpha}, Δt={args.cbf_delta_t})")
    logging.info(f"Max steps: {args.max_steps}, Success threshold: {args.success_threshold}m")
    logging.info("=" * 70)

    # Load or generate scenarios
    loaded_scenarios = None
    if args.load_scenes:
        with open(args.load_scenes, 'r') as f:
            loaded_scenarios = json.load(f)
        logging.info(f"Loaded {len(loaded_scenarios['scenarios'])} scenarios from {args.load_scenes}")
        args.num_problems = len(loaded_scenarios['scenarios'])

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

        # =============================================================
        # STAGE 1: Plan with CBF safety filter
        # =============================================================
        plan_result = plan_with_cbf(
            model, cbf_net,
            q_start, e_start, e_target,
            obstacles_raw,
            mean_train, std_train,
            device, args
        )

        # =============================================================
        # STAGE 2: Validate path with Robo3D geometric ground truth
        # =============================================================
        is_collision_free, num_collisions, collision_waypoints = \
            validate_path_with_geometric_checker(
                plan_result['decoded_path'],
                obstacles_raw,
                robo3d
            )

        # Determine success
        goal_reached = plan_result['goal_reached']
        success = goal_reached and is_collision_free

        # Accumulate statistics
        if goal_reached:
            goal_reached_count += 1
        if is_collision_free:
            collision_free_count += 1
        if success:
            successes += 1
            planning_times.append(plan_result['planning_time_ms'])
            path_lengths.append(plan_result['path_length'])
        min_distances.append(plan_result['min_distance'])
        all_cbf_metrics.append(plan_result['cbf_metrics'])

        # Store result
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
            'cbf_metrics': plan_result['cbf_metrics'],
        })

        # Log progress
        if (i + 1) % 100 == 0:
            sr = successes / (i + 1) * 100
            gr = goal_reached_count / (i + 1) * 100
            cf = collision_free_count / (i + 1) * 100
            avg_lambda = np.mean([m.get('avg_lambda', 0) for m in all_cbf_metrics])
            avg_interv = np.mean([m.get('intervention_rate', 0) for m in all_cbf_metrics])
            logging.info(
                f"Progress: {i+1}/{args.num_problems} | "
                f"Success: {sr:.1f}% | Goal: {gr:.1f}% | "
                f"Collision-free: {cf:.1f}% | "
                f"CBF: λ_avg={avg_lambda:.4f}, interv={avg_interv*100:.1f}%"
            )

    # --- Summary ---
    n = args.num_problems
    success_rate = successes / n * 100
    goal_reached_rate = goal_reached_count / n * 100
    collision_free_rate = collision_free_count / n * 100
    avg_time = np.mean(planning_times) if planning_times else 0
    std_time = np.std(planning_times) if planning_times else 0
    avg_path = np.mean(path_lengths) if path_lengths else 0
    std_path = np.std(path_lengths) if path_lengths else 0
    avg_min_dist = np.mean(min_distances)

    # CBF aggregate metrics
    cbf_agg = {}
    if all_cbf_metrics:
        cbf_agg = {
            'avg_lambda_overall': float(np.mean([m.get('avg_lambda', 0) for m in all_cbf_metrics])),
            'max_lambda_overall': float(np.max([m.get('max_lambda', 0) for m in all_cbf_metrics])),
            'avg_intervention_rate': float(np.mean([m.get('intervention_rate', 0) for m in all_cbf_metrics])),
            'avg_min_B_current': float(np.mean([m.get('min_B_current', 0) for m in all_cbf_metrics])),
        }

    results_summary = {
        'method': 'CBF_safety_filter',
        'num_problems': n,
        'num_obstacles': args.num_obstacles,
        'success_threshold_m': args.success_threshold,
        'max_steps': args.max_steps,
        'planning_lr': args.planning_lr,
        'lambda_prior': args.lambda_prior,
        'cbf_alpha': args.cbf_alpha,
        'cbf_delta_t': args.cbf_delta_t,
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
        'cbf_aggregate_metrics': cbf_agg,
        'planning_times_ms': planning_times,
        'path_lengths': path_lengths,
        'min_distances': min_distances,
        'detailed_results': results_list,
    }

    return results_summary


def print_results(results):
    """Print CBF evaluation results."""
    logging.info("\n" + "=" * 70)
    logging.info("CBF PLANNING EVALUATION RESULTS")
    logging.info("=" * 70)
    logging.info(f"Configuration:")
    logging.info(f"  Method:              CBF Safety Filter")
    logging.info(f"  Test Scenarios:      {results['num_problems']}")
    logging.info(f"  Obstacles per scene: {results['num_obstacles']}")
    logging.info(f"  Success threshold:   {results['success_threshold_m']}m")
    logging.info(f"  Max planning steps:  {results['max_steps']}")
    logging.info(f"  Planning LR:         {results['planning_lr']}")
    logging.info(f"  Lambda prior:        {results['lambda_prior']}")
    logging.info(f"  CBF alpha:           {results['cbf_alpha']}")
    logging.info(f"  CBF delta_t:         {results['cbf_delta_t']}")
    logging.info(f"  Collision checker:   {results['collision_checker']}")
    logging.info("-" * 70)
    logging.info(f"Task Performance:")
    logging.info(f"  Success Rate:        {results['success_rate_percent']:.2f}%")
    logging.info(f"  Goal Reached:        {results['goal_reached_rate_percent']:.2f}%")
    logging.info(f"  Collision-free:      {results['collision_free_rate_percent']:.2f}%")
    logging.info(f"  Avg Planning Time:   {results['avg_planning_time_ms']:.2f} ± {results['std_planning_time_ms']:.2f} ms")
    logging.info(f"  Avg Path Length:     {results['avg_path_length']:.4f} ± {results['std_path_length']:.4f}")

    cbf_agg = results.get('cbf_aggregate_metrics', {})
    if cbf_agg:
        logging.info("-" * 70)
        logging.info(f"CBF-Specific Metrics:")
        logging.info(f"  Avg λ correction:     {cbf_agg.get('avg_lambda_overall', 0):.6f}")
        logging.info(f"  Max λ correction:     {cbf_agg.get('max_lambda_overall', 0):.6f}")
        logging.info(f"  Avg intervention rate: {cbf_agg.get('avg_intervention_rate', 0)*100:.1f}%")
        logging.info(f"  Avg min B(z_k):       {cbf_agg.get('avg_min_B_current', 0):.4f}")
    logging.info("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='CBF Planning Evaluation for Reaching Through Latent Space'
    )

    # Model arguments
    parser.add_argument('--checkpoint', type=str, default=cfg.VAE_CHECKPOINT,
                        help='Path to frozen VAE checkpoint')
    parser.add_argument('--config', type=str, default=cfg.VAE_CONFIG,
                        help='Path to VAE config JSON')
    parser.add_argument('--cbf_checkpoint', type=str, default=cfg.CBF_BEST_CHECKPOINT,
                        help='Path to trained BarrierNet checkpoint')

    # Evaluation
    parser.add_argument('--num_problems', type=int, default=1000)
    parser.add_argument('--num_obstacles', type=int, default=1)
    parser.add_argument('--max_steps', type=int, default=cfg.MAX_STEPS)
    parser.add_argument('--planning_lr', type=float, default=cfg.PLANNING_LR)
    parser.add_argument('--success_threshold', type=float, default=cfg.SUCCESS_THRESHOLD)
    parser.add_argument('--lambda_prior', type=float, default=cfg.LAMBDA_PRIOR)

    # CBF parameters
    parser.add_argument('--cbf_alpha', type=float, default=cfg.CBF_ALPHA)
    parser.add_argument('--cbf_delta_t', type=float, default=cfg.CBF_DELTA_T)
    parser.add_argument('--lambda_max', type=float, default=1.0,
                        help='Maximum CBF correction magnitude per iteration')
    parser.add_argument('--cbf_safe_threshold', type=float, default=None,
                        help='Skip CBF correction when B(z_nom) > this value')
    parser.add_argument('--cbf_max_iters', type=int, default=5,
                        help='Max iterative correction steps (default 5)')

    # Data
    parser.add_argument('--data_path', type=str, default=cfg.DATA_PATH)

    # Scenes
    parser.add_argument('--load_scenes', type=str, default=None,
                        help='Load scenarios from JSON (use test_scenes.json for fair comparison)')
    parser.add_argument('--save_scenes', type=str, default=None)

    # Output
    parser.add_argument('--output', type=str, default=None)

    # Device
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=cfg.SEED)

    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logging.info(f"Using device: {device}")

    # Load VAE config
    with open(args.config, 'r') as f:
        config = json.load(f)
        if 'parsed_args' in config:
            config = config['parsed_args']

    # Get normalization stats
    dataset = RobotStateDataset(args.data_path, train=0, train_data_name='free_space_100k_train.dat')
    mean_train = dataset.get_mean_train()
    std_train = dataset.get_std_train()

    # Initialize robot
    robot = Panda()
    robot.to(device)
    robo3d = Robo3D(Panda())
    logging.info("Robo3D geometric collision checker instantiated")

    # Load frozen VAE
    logging.info(f"Loading VAE from {args.checkpoint}")
    model = VAE(
        config['input_dim'], config['latent_dim'],
        config['units_per_layer'], config['num_hidden_layers']
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    logging.info(f"VAE loaded: {config['latent_dim']}D latent")

    # Load trained CBF
    logging.info(f"Loading BarrierNet from {args.cbf_checkpoint}")
    cbf_ckpt = torch.load(args.cbf_checkpoint, map_location=device, weights_only=False)
    cbf_config = cbf_ckpt.get('config', {})
    cbf_net = BarrierNet(
        latent_dim=cbf_config.get('latent_dim', cfg.LATENT_DIM),
        obs_dim=cbf_config.get('obs_dim', cfg.OBS_DIM),
        hidden_units=cbf_config.get('hidden_units', cfg.CBF_HIDDEN_UNITS),
        num_hidden=cbf_config.get('num_hidden', cfg.CBF_NUM_HIDDEN)
    )
    cbf_net.load_state_dict(cbf_ckpt['model_state_dict'])
    cbf_net.to(device)
    cbf_net.eval()
    logging.info(f"BarrierNet loaded (epoch {cbf_ckpt.get('epoch', '?')})")

    # Run evaluation
    logging.info("\nStarting CBF evaluation (two-stage pipeline)...")
    logging.info("  Stage 1: Planning with Goal+Prior + CBF safety filter")
    logging.info("  Stage 2: Validation with Robo3D.check_for_collision")
    logging.info("")

    results = evaluate_cbf_planning(
        model, cbf_net, robot, robo3d,
        mean_train, std_train,
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


if __name__ == '__main__':
    main()
