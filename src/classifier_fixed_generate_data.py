"""
Fixed-Obstacle Classifier Data Generation.

Generates (z, label) pairs for training C(z):
    1. Sample random joint configurations
    2. Check collision with the FIXED obstacle using Robo3D
    3. Encode to z using frozen VAE
    4. Store (z, label) — no obs in the data

Labels: 0 = safe (no collision), 1 = unsafe (collision)
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

warnings.filterwarnings('ignore', category=FutureWarning, module='torch')

from vae import VAE
from robot_state_dataset import RobotStateDataset
from sim.panda import Panda
from sim.robot3d import Robo3D
import cbf_config as cfg
import classifier_fixed_config as clcfg

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


def main():
    parser = argparse.ArgumentParser(
        description='Generate data for fixed-obstacle classifier C(z)')
    parser.add_argument('--num_samples', type=int, default=clcfg.NUM_SAMPLES)
    parser.add_argument('--val_split', type=float, default=clcfg.VAL_SPLIT)
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=clcfg.SEED)
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size for VAE encoding')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logging.info(f"Using device: {device}")

    os.makedirs(clcfg.DATA_DIR, exist_ok=True)

    # =========================================================================
    # Load frozen VAE
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
    logging.info(f"Frozen VAE loaded from {cfg.VAE_CHECKPOINT}")

    # Normalization stats
    dataset = RobotStateDataset(
        cfg.DATA_PATH, train=0, train_data_name='free_space_100k_train.dat'
    )
    mean_train = dataset.get_mean_train()
    std_train = dataset.get_std_train()
    mean_train_t = torch.tensor(mean_train[:, :10], dtype=torch.float32).to(device)
    std_train_t = torch.tensor(std_train[:, :10], dtype=torch.float32).to(device)

    # Robot
    robot = Panda()
    robot.to(device)
    robo3d = Robo3D(Panda())

    q_min_rad = robot.joint_min_limits_tensor * (torch.pi / 180.0)
    q_max_rad = robot.joint_max_limits_tensor * (torch.pi / 180.0)

    # Fixed obstacle
    obstacle_xyhr = clcfg.FIXED_OBSTACLE

    logging.info(f"\n{'=' * 60}")
    logging.info(f"FIXED-OBSTACLE CLASSIFIER DATA GENERATION")
    logging.info(f"  Obstacle: {obstacle_xyhr}")
    logging.info(f"  Samples:  {args.num_samples}")
    logging.info(f"  Val split: {args.val_split}")
    logging.info(f"{'=' * 60}\n")

    # =========================================================================
    # Sample random configs and check collisions
    # =========================================================================
    all_z = []
    all_labels = []
    safe_count = 0
    unsafe_count = 0

    start_time = time.time()
    batch_size = args.batch_size
    num_batches = (args.num_samples + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        current_batch_size = min(batch_size, args.num_samples - batch_idx * batch_size)
        if current_batch_size <= 0:
            break

        # Random joint configurations
        q_batch = torch.rand(current_batch_size, 7, device=device) * \
                  (q_max_rad - q_min_rad) + q_min_rad

        # Forward kinematics
        ee_batch = robot.FK(q_batch.clone(), device, rad=True)

        # Encode to latent space
        x_batch = torch.cat([q_batch, ee_batch], dim=1)
        x_norm = (x_batch - mean_train_t) / std_train_t

        with torch.no_grad():
            z_batch = model.encoder(x_norm)[0]

        # Check collision with fixed obstacle using Robo3D
        q_np = q_batch.cpu().numpy()
        labels = []
        for i in range(current_batch_size):
            q_deg = np.degrees(q_np[i]).tolist()
            is_collision = robo3d.check_for_collision(q_deg, [obstacle_xyhr])
            labels.append(1.0 if is_collision else 0.0)

        labels_tensor = torch.tensor(labels, dtype=torch.float32)

        all_z.append(z_batch.cpu())
        all_labels.append(labels_tensor)

        batch_safe = (labels_tensor == 0).sum().item()
        batch_unsafe = (labels_tensor == 1).sum().item()
        safe_count += batch_safe
        unsafe_count += batch_unsafe

        if (batch_idx + 1) % 50 == 0 or batch_idx == num_batches - 1:
            total = safe_count + unsafe_count
            logging.info(
                f"Batch {batch_idx+1}/{num_batches} | "
                f"Total: {total} | "
                f"Safe: {safe_count} ({safe_count/total*100:.1f}%) | "
                f"Unsafe: {unsafe_count} ({unsafe_count/total*100:.1f}%)"
            )

    elapsed = time.time() - start_time
    logging.info(f"\nGenerated {safe_count + unsafe_count} samples in {elapsed:.1f}s")
    logging.info(f"  Safe: {safe_count} ({safe_count/(safe_count+unsafe_count)*100:.1f}%)")
    logging.info(f"  Unsafe: {unsafe_count} ({unsafe_count/(safe_count+unsafe_count)*100:.1f}%)")

    # =========================================================================
    # Concatenate and split
    # =========================================================================
    z_all = torch.cat(all_z, dim=0)
    labels_all = torch.cat(all_labels, dim=0)

    n = len(z_all)
    n_val = int(n * args.val_split)
    n_train = n - n_val

    # Shuffle
    perm = torch.randperm(n)
    z_all = z_all[perm]
    labels_all = labels_all[perm]

    z_train, z_val = z_all[:n_train], z_all[n_train:]
    labels_train, labels_val = labels_all[:n_train], labels_all[n_train:]

    # Save
    torch.save({'z': z_train, 'label': labels_train}, clcfg.TRAIN_DATA)
    torch.save({'z': z_val, 'label': labels_val}, clcfg.VAL_DATA)

    logging.info(f"\nSaved:")
    logging.info(f"  Train: {clcfg.TRAIN_DATA} ({n_train} samples)")
    logging.info(f"  Val:   {clcfg.VAL_DATA} ({n_val} samples)")

    train_safe = (labels_train == 0).sum().item()
    train_unsafe = (labels_train == 1).sum().item()
    val_safe = (labels_val == 0).sum().item()
    val_unsafe = (labels_val == 1).sum().item()

    logging.info(f"\n  Train — safe: {train_safe}, unsafe: {train_unsafe}")
    logging.info(f"  Val   — safe: {val_safe}, unsafe: {val_unsafe}")


if __name__ == '__main__':
    main()
