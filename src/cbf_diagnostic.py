"""
CBF Diagnostic — Test barrier function accuracy against Robo3D ground truth.

Answers the key question: Does B(z,o) > 0 actually mean safe?
If accuracy < 70%, the barrier is fundamentally broken and ablation won't help.
If accuracy > 85%, the barrier works on static data and the issue is elsewhere.

Run: python cbf_diagnostic.py
"""

import json
import logging
import numpy as np
import os
import torch
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

from vae import VAE
from cbf_model import BarrierNet
from robot_state_dataset import RobotStateDataset
from sim.panda import Panda
from sim.robot3d import Robo3D
from evaluate_planning import ObstacleScenarioGenerator
import cbf_config as cfg

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    # Load VAE
    with open(cfg.VAE_CONFIG, 'r') as f:
        config = json.load(f)
        if 'parsed_args' in config:
            config = config['parsed_args']
    model = VAE(config['input_dim'], config['latent_dim'],
                config['units_per_layer'], config['num_hidden_layers'])
    ckpt = torch.load(cfg.VAE_CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()

    # Load CBF
    cbf_ckpt = torch.load(cfg.CBF_BEST_CHECKPOINT, map_location=device, weights_only=False)
    cbf_net = BarrierNet(latent_dim=cfg.LATENT_DIM, obs_dim=cfg.OBS_DIM,
                         hidden_units=cfg.CBF_HIDDEN_UNITS, num_hidden=cfg.CBF_NUM_HIDDEN)
    cbf_net.load_state_dict(cbf_ckpt['model_state_dict'])
    cbf_net.to(device).eval()

    logging.info(f"CBF checkpoint epoch: {cbf_ckpt.get('epoch', '?')}, "
                 f"score: {cbf_ckpt.get('score', '?')}")
    train_m = cbf_ckpt.get('train_metrics', {})
    val_m = cbf_ckpt.get('val_metrics', {})
    logging.info(f"Train safe_acc={train_m.get('safe_accuracy',0)*100:.1f}%, "
                 f"unsafe_acc={train_m.get('unsafe_accuracy',0)*100:.1f}%")
    logging.info(f"Val   safe_acc={val_m.get('safe_accuracy',0)*100:.1f}%, "
                 f"unsafe_acc={val_m.get('unsafe_accuracy',0)*100:.1f}%")

    # Load normalization stats
    dataset = RobotStateDataset(cfg.DATA_PATH, train=0,
                                train_data_name='free_space_100k_train.dat')
    mean_train = dataset.get_mean_train()
    std_train = dataset.get_std_train()
    mean_t = torch.tensor(mean_train[:, :10], dtype=torch.float32).to(device)
    std_t = torch.tensor(std_train[:, :10], dtype=torch.float32).to(device)

    # Initialize robot
    robot = Panda()
    robot.to(device)
    robo3d = Robo3D(Panda())
    scenario_gen = ObstacleScenarioGenerator(robot)

    q_min = robot.joint_min_limits_tensor * (torch.pi / 180.0)
    q_max = robot.joint_max_limits_tensor * (torch.pi / 180.0)

    # =========================================================================
    # TEST 1: Random static samples — B accuracy vs Robo3D ground truth
    # =========================================================================
    logging.info("\n" + "=" * 60)
    logging.info("TEST 1: Static accuracy (1000 random configs × random obstacles)")
    logging.info("=" * 60)

    N = 1000
    correct = 0
    B_vals_safe = []
    B_vals_unsafe = []

    for i in range(N):
        q = torch.rand(1, 7, device=device) * (q_max - q_min) + q_min
        ee = robot.FK(q.clone(), device, rad=True)

        # Random obstacle
        obs_raw = scenario_gen.generate_scenario(
            q.cpu().numpy()[0], ee.cpu().numpy()[0],
            ee.cpu().numpy()[0],  # target doesn't matter for obstacle gen
            num_obstacles=1
        )
        if len(obs_raw) == 0:
            continue

        obs_tensor = torch.tensor(obs_raw[0], dtype=torch.float32).unsqueeze(0).to(device)

        # Encode to latent
        x = torch.cat([q, ee], dim=1)
        x_norm = (x - mean_t) / std_t
        with torch.no_grad():
            z = model.encoder(x_norm)[0]
            B_val = cbf_net(z, obs_tensor).item()

        # Ground truth from Robo3D
        q_deg = np.degrees(q.cpu().numpy()[0]).tolist()
        obs_list = [obs_raw[0].tolist()]
        gt_collision = robo3d.check_for_collision(q_deg, obs_list)
        gt_safe = not gt_collision

        # Compare
        B_predicts_safe = B_val >= 0
        if B_predicts_safe == gt_safe:
            correct += 1

        if gt_safe:
            B_vals_safe.append(B_val)
        else:
            B_vals_unsafe.append(B_val)

    acc = correct / N * 100
    logging.info(f"Static accuracy: {acc:.1f}% ({correct}/{N})")
    logging.info(f"  Safe samples:   {len(B_vals_safe)}, avg B = {np.mean(B_vals_safe):.4f}")
    logging.info(f"  Unsafe samples: {len(B_vals_unsafe)}, avg B = {np.mean(B_vals_unsafe):.4f}")

    if B_vals_safe:
        safe_correct = sum(1 for b in B_vals_safe if b >= 0)
        logging.info(f"  Safe accuracy:   {safe_correct/len(B_vals_safe)*100:.1f}%")
    if B_vals_unsafe:
        unsafe_correct = sum(1 for b in B_vals_unsafe if b < 0)
        logging.info(f"  Unsafe accuracy: {unsafe_correct/len(B_vals_unsafe)*100:.1f}%")

    # =========================================================================
    # TEST 2: Along planning trajectories — does B track safety?
    # =========================================================================
    logging.info("\n" + "=" * 60)
    logging.info("TEST 2: Trajectory accuracy (50 planning runs)")
    logging.info("=" * 60)

    import torch.optim as optim

    traj_correct = 0
    traj_total = 0
    traj_B_safe = []
    traj_B_unsafe = []

    for scenario_id in range(50):
        q_start = torch.rand(1, 7, device=device) * (q_max - q_min) + q_min
        e_start = robot.FK(q_start.clone(), device, rad=True)
        q_target = torch.rand(1, 7, device=device) * (q_max - q_min) + q_min
        e_target = robot.FK(q_target.clone(), device, rad=True)

        obs_raw = scenario_gen.generate_scenario(
            q_start.cpu().numpy()[0], e_start.cpu().numpy()[0],
            e_target.cpu().numpy()[0], num_obstacles=1
        )
        if len(obs_raw) == 0:
            continue

        obs_tensor = torch.tensor(obs_raw[0], dtype=torch.float32).unsqueeze(0).to(device)
        obs_list = [obs_raw[0].tolist()]

        # Run nominal planner (Goal+Prior, no CBF)
        x_start = torch.cat([q_start, e_start], dim=1)
        x_start_norm = (x_start - mean_t) / std_t
        with torch.no_grad():
            z_init = model.encoder(x_start_norm)[0]

        z = z_init.clone().detach().requires_grad_(True)
        optimizer = optim.Adam([z], lr=0.03)

        for step in range(100):  # 100 steps for speed
            optimizer.zero_grad()
            x_dec_norm = model.decoder(z)
            x_dec = x_dec_norm * std_t + mean_t
            L = torch.norm(x_dec[:, 7:10] - e_target) + 0.01 * 0.5 * torch.sum(z ** 2)
            L.backward()
            optimizer.step()

            # Check B and ground truth at each waypoint
            with torch.no_grad():
                B_val = cbf_net(z, obs_tensor).item()
                x_cur = model.decoder(z) * std_t + mean_t
                q_cur = x_cur[:, :7].cpu().numpy()[0]
                q_cur_deg = np.degrees(q_cur).tolist()
                gt_collision = robo3d.check_for_collision(q_cur_deg, obs_list)
                gt_safe = not gt_collision

            B_pred_safe = B_val >= 0
            if B_pred_safe == gt_safe:
                traj_correct += 1
            traj_total += 1

            if gt_safe:
                traj_B_safe.append(B_val)
            else:
                traj_B_unsafe.append(B_val)

    traj_acc = traj_correct / max(traj_total, 1) * 100
    logging.info(f"Trajectory accuracy: {traj_acc:.1f}% ({traj_correct}/{traj_total})")
    logging.info(f"  Safe waypoints:   {len(traj_B_safe)}, avg B = {np.mean(traj_B_safe):.4f}" if traj_B_safe else "  No safe waypoints")
    logging.info(f"  Unsafe waypoints: {len(traj_B_unsafe)}, avg B = {np.mean(traj_B_unsafe):.4f}" if traj_B_unsafe else "  No unsafe waypoints")

    if traj_B_safe:
        ts_correct = sum(1 for b in traj_B_safe if b >= 0)
        logging.info(f"  Traj safe accuracy:   {ts_correct/len(traj_B_safe)*100:.1f}%")
    if traj_B_unsafe:
        tu_correct = sum(1 for b in traj_B_unsafe if b < 0)
        logging.info(f"  Traj unsafe accuracy: {tu_correct/len(traj_B_unsafe)*100:.1f}%")

    # =========================================================================
    # DIAGNOSIS
    # =========================================================================
    logging.info("\n" + "=" * 60)
    logging.info("DIAGNOSIS")
    logging.info("=" * 60)

    if acc < 60:
        logging.info("❌ CRITICAL: Static accuracy < 60% — barrier is fundamentally broken")
        logging.info("   → Fix training data or training procedure BEFORE ablation")
    elif acc < 75:
        logging.info("⚠️  WARNING: Static accuracy 60-75% — barrier is weak")
        logging.info("   → Ablation on training params (margin, LR, epochs) may help")
    else:
        logging.info("✅ Static accuracy > 75% — barrier classifies well on static data")

    if traj_total > 0 and traj_acc < acc - 10:
        logging.info("⚠️  Distribution mismatch: trajectory accuracy much lower than static")
        logging.info("   → Need DAgger-style data collection or trajectory-aware training")
    elif traj_total > 0:
        logging.info("✅ Trajectory accuracy similar to static — no distribution mismatch")


if __name__ == '__main__':
    main()
