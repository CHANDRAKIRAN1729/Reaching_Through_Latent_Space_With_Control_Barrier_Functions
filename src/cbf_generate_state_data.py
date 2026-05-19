"""
CBF State-Label Data Generation — Encode existing collision data into latent space.

Converts the existing collision_*.dat and free_space_*.dat files into
latent-space samples (z, obs, label) for CBF training.

Processing:
    1. Load collision data → normalize [q, ee] → encode through frozen VAE → z
    2. Extract obstacle [x, y, h, r] and collision label
    3. Split into train/val/test with scenario-level separation
    4. Save as .pt tensors
"""

from __future__ import print_function
import argparse
import json
import logging
import numpy as np
import os
import torch
import warnings

warnings.filterwarnings('ignore', category=FutureWarning, module='torch')

from vae import VAE
from robot_state_dataset import RobotStateDataset
from robot_obs_dataset import RobotObstacleDataset
from sim.panda import Panda
from sim.robot3d import Robo3D
import cbf_config as cfg

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


def load_frozen_vae(device):
    """Load the frozen VAE model (encoder only needed)."""
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

    obs_dataset = RobotObstacleDataset(
        cfg.DATA_PATH, train=0,
        train_data_name='collision_100k_train.dat',
        test_data_name='collision_10k_test.dat',
        free_space_train_name='free_space_100k_train.dat',
        free_space_test_name='free_space_10k_test.dat'
    )
    mean_obs = obs_dataset.get_mean_train()[0, 10:14]
    std_obs = obs_dataset.get_std_train()[0, 10:14]

    return mean_train, std_train, mean_obs, std_obs


def encode_collision_data(model, data_path, data_name, mean_train, std_train,
                          device, batch_size=4096):
    """
    Encode collision dataset samples into latent space.

    Returns:
        z_all:     (N, latent_dim) latent codes (using mu, deterministic)
        obs_all:   (N, 4) raw obstacle parameters
        label_all: (N,) collision labels (0=safe, 1=unsafe)
    """
    # Load raw data (numpy .npy format, loaded via np.load)
    with open(os.path.join(data_path, data_name), 'rb') as f:
        raw_data = np.load(f)

    logging.info(f"Loaded {len(raw_data)} samples from {data_name}")

    # Extract components
    jpos_ee = raw_data[:, :10]     # [q(7), ee(3)]
    obs_raw = raw_data[:, 10:14]   # [x, y, h, r]
    labels = raw_data[:, 14]       # collision label

    # Normalize robot state for VAE encoder
    mean_t = torch.tensor(mean_train[:, :10], dtype=torch.float32).to(device)
    std_t = torch.tensor(std_train[:, :10], dtype=torch.float32).to(device)

    z_list = []
    with torch.no_grad():
        for i in range(0, len(jpos_ee), batch_size):
            batch = torch.tensor(jpos_ee[i:i+batch_size], dtype=torch.float32).to(device)
            batch_norm = (batch - mean_t) / std_t
            # Use mu (deterministic encoding) not sampled z
            _, mu, _ = model.encoder(batch_norm)
            z_list.append(mu.cpu())

    z_all = torch.cat(z_list, dim=0)                                    # (N, 7)
    obs_all = torch.tensor(obs_raw, dtype=torch.float32)                # (N, 4)
    label_all = torch.tensor(labels, dtype=torch.float32)               # (N,)

    logging.info(f"Encoded: {z_all.shape[0]} samples → z shape {z_all.shape}")
    logging.info(f"  Safe: {(label_all == 0).sum().item()}, Unsafe: {(label_all == 1).sum().item()}")

    return z_all, obs_all, label_all


def encode_free_space_data(model, data_path, data_name, mean_train, std_train,
                           device, batch_size=4096):
    """
    Encode free-space dataset samples into latent space.
    All samples are safe (label=0). No obstacle info — returns dummy obstacles.

    Returns:
        z_all:     (N, latent_dim) latent codes
        label_all: (N,) all zeros (safe)
    """
    with open(os.path.join(data_path, data_name), 'rb') as f:
        raw_data = np.load(f)

    logging.info(f"Loaded {len(raw_data)} free-space samples from {data_name}")

    jpos_ee = raw_data[:, :10]

    mean_t = torch.tensor(mean_train[:, :10], dtype=torch.float32).to(device)
    std_t = torch.tensor(std_train[:, :10], dtype=torch.float32).to(device)

    z_list = []
    with torch.no_grad():
        for i in range(0, len(jpos_ee), batch_size):
            batch = torch.tensor(jpos_ee[i:i+batch_size], dtype=torch.float32).to(device)
            batch_norm = (batch - mean_t) / std_t
            _, mu, _ = model.encoder(batch_norm)
            z_list.append(mu.cpu())

    z_all = torch.cat(z_list, dim=0)
    label_all = torch.zeros(len(z_all), dtype=torch.float32)

    logging.info(f"Encoded: {z_all.shape[0]} free-space samples (all safe)")

    return z_all, label_all


def generate_boundary_samples(model, robot, robo3d, mean_train, std_train,
                              device, num_samples=10000, distance_threshold=0.05):
    """
    Generate hard samples near obstacle boundaries using Robo3D distance checking.

    Samples random configurations and obstacles, keeps those with small
    positive distance (near-miss) for enriched boundary coverage.
    """
    logging.info(f"Generating boundary samples (target: {num_samples}, threshold: {distance_threshold}m)...")

    q_min_rad = robot.joint_min_limits_tensor * (torch.pi / 180.0)
    q_max_rad = robot.joint_max_limits_tensor * (torch.pi / 180.0)

    mean_t = torch.tensor(mean_train[:, :10], dtype=torch.float32).to(device)
    std_t = torch.tensor(std_train[:, :10], dtype=torch.float32).to(device)

    z_boundary = []
    obs_boundary = []
    label_boundary = []

    attempts = 0
    max_attempts = num_samples * 20

    while len(z_boundary) < num_samples and attempts < max_attempts:
        attempts += 1

        # Sample random configuration
        q = torch.rand(1, 7, device=device) * (q_max_rad - q_min_rad) + q_min_rad
        e = robot.FK(q.clone(), device, rad=True)

        # Sample random obstacle near the end-effector
        e_np = e.cpu().numpy()[0]
        offset = np.random.uniform(-0.2, 0.2, size=2)
        obs_x = e_np[0] + offset[0]
        obs_y = e_np[1] + offset[1]
        obs_h = np.random.uniform(0.3, 1.2)
        obs_r = np.random.uniform(0.03, 0.2)

        obs_raw = np.array([obs_x, obs_y, obs_h, obs_r], dtype=np.float32)

        # Check distance using Robo3D
        q_deg = np.degrees(q.cpu().numpy()[0]).tolist()
        dist = robo3d.dist_jpos_to_obstacles(q_deg, [obs_raw.tolist()])

        # Keep if near boundary (small distance)
        if dist <= distance_threshold:
            # Encode to latent
            x_state = torch.cat([q, e], dim=1)
            x_norm = (x_state - mean_t) / std_t
            with torch.no_grad():
                _, mu, _ = model.encoder(x_norm)

            z_boundary.append(mu.cpu().squeeze(0))
            obs_boundary.append(torch.tensor(obs_raw))
            label_boundary.append(0.0 if dist > 0 else 1.0)

    if len(z_boundary) > 0:
        z_boundary = torch.stack(z_boundary)
        obs_boundary = torch.stack(obs_boundary)
        label_boundary = torch.tensor(label_boundary, dtype=torch.float32)
        logging.info(f"Generated {len(z_boundary)} boundary samples in {attempts} attempts")
        logging.info(f"  Safe (near-miss): {(label_boundary == 0).sum().item()}, "
                     f"Unsafe (collision): {(label_boundary == 1).sum().item()}")
    else:
        logging.warning("No boundary samples generated!")
        z_boundary = torch.zeros(0, cfg.LATENT_DIM)
        obs_boundary = torch.zeros(0, cfg.OBS_DIM)
        label_boundary = torch.zeros(0)

    return z_boundary, obs_boundary, label_boundary


def main():
    parser = argparse.ArgumentParser(description='Generate CBF state-label dataset')
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=cfg.SEED)
    parser.add_argument('--boundary_samples', type=int, default=50000,
                        help='Number of boundary samples to generate (more = better CBF accuracy)')
    parser.add_argument('--boundary_threshold', type=float, default=0.05,
                        help='Distance threshold for boundary samples (meters)')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Fraction of data for validation')
    parser.add_argument('--skip_boundary', action='store_true', default=False,
                        help='Skip boundary sample generation (faster)')
    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logging.info(f"Using device: {device}")

    # Create output directory
    os.makedirs(cfg.CBF_DATA_DIR, exist_ok=True)

    # Load frozen VAE
    model = load_frozen_vae(device)

    # Get normalization stats
    mean_train, std_train, mean_obs, std_obs = get_normalization_stats()
    logging.info(f"Normalization stats loaded")

    # Initialize robot for boundary samples
    robot = Panda()
    robot.to(device)
    robo3d = Robo3D(Panda())

    # =========================================================================
    # Encode collision training data
    # =========================================================================
    z_coll_train, obs_coll_train, label_coll_train = encode_collision_data(
        model, cfg.DATA_PATH, 'collision_100k_train.dat',
        mean_train, std_train, device
    )

    # =========================================================================
    # Encode collision test data
    # =========================================================================
    z_coll_test, obs_coll_test, label_coll_test = encode_collision_data(
        model, cfg.DATA_PATH, 'collision_10k_test.dat',
        mean_train, std_train, device
    )

    # =========================================================================
    # Generate boundary samples (enriched near-collision data)
    # =========================================================================
    if not args.skip_boundary:
        z_bnd, obs_bnd, label_bnd = generate_boundary_samples(
            model, robot, robo3d, mean_train, std_train, device,
            num_samples=args.boundary_samples,
            distance_threshold=args.boundary_threshold
        )
    else:
        z_bnd = torch.zeros(0, cfg.LATENT_DIM)
        obs_bnd = torch.zeros(0, cfg.OBS_DIM)
        label_bnd = torch.zeros(0)

    # =========================================================================
    # Combine and split training data
    # =========================================================================
    z_all_train = torch.cat([z_coll_train, z_bnd], dim=0)
    obs_all_train = torch.cat([obs_coll_train, obs_bnd], dim=0)
    label_all_train = torch.cat([label_coll_train, label_bnd], dim=0)

    # Shuffle
    perm = torch.randperm(len(z_all_train))
    z_all_train = z_all_train[perm]
    obs_all_train = obs_all_train[perm]
    label_all_train = label_all_train[perm]

    # Split into train/val
    n_val = int(len(z_all_train) * args.val_split)
    n_train = len(z_all_train) - n_val

    train_data = {
        'z': z_all_train[:n_train],
        'obs': obs_all_train[:n_train],
        'label': label_all_train[:n_train],
    }
    val_data = {
        'z': z_all_train[n_train:],
        'obs': obs_all_train[n_train:],
        'label': label_all_train[n_train:],
    }
    test_data = {
        'z': z_coll_test,
        'obs': obs_coll_test,
        'label': label_coll_test,
    }

    # =========================================================================
    # Save datasets
    # =========================================================================
    torch.save(train_data, cfg.STATE_LABELS_TRAIN)
    logging.info(f"Saved training data: {n_train} samples → {cfg.STATE_LABELS_TRAIN}")
    logging.info(f"  Safe: {(train_data['label'] == 0).sum().item()}, "
                 f"Unsafe: {(train_data['label'] == 1).sum().item()}")

    torch.save(val_data, cfg.STATE_LABELS_VAL)
    logging.info(f"Saved validation data: {n_val} samples → {cfg.STATE_LABELS_VAL}")

    torch.save(test_data, cfg.STATE_LABELS_TEST)
    logging.info(f"Saved test data: {len(z_coll_test)} samples → {cfg.STATE_LABELS_TEST}")

    # Save normalization stats for reference
    norm_stats = {
        'mean_train': mean_train,
        'std_train': std_train,
        'mean_obs': mean_obs,
        'std_obs': std_obs,
    }
    torch.save(norm_stats, os.path.join(cfg.CBF_DATA_DIR, 'normalization_stats.pt'))
    logging.info(f"Saved normalization stats")

    logging.info("\n" + "=" * 60)
    logging.info("State-label dataset generation complete!")
    logging.info("=" * 60)


if __name__ == '__main__':
    main()
