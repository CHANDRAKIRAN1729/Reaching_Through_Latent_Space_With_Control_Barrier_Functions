"""
Multi-Position CBF Training — B(z, x, y) conditioned on obstacle position.

BarrierNet input: z(7) + obs_xy(2) = 9D.
Fixed obstacle shape (h, r), only position (x, y) varies.

Same three-term loss as fixed experiment:
    1. Safe sign:    relu(-B + γ)
    2. Unsafe sign:  relu(B + γ)     (symmetric)
    3. Decrease:     relu(target - B_nom)
"""

from __future__ import print_function
import argparse
import json
import logging
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings('ignore', category=FutureWarning, module='torch')

import cbf_multipos_config as mcfg

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


# =============================================================================
# BarrierNet B(z, x, y) — conditioned on obstacle position
# =============================================================================
class MultiPosBarrierNet(nn.Module):
    """
    Neural CBF B(z, x, y) conditioned on obstacle (x, y) position.
    Fixed shape (h, r) not included in input.
    """

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
# Dataset classes (with obs_xy)
# =============================================================================
class MultiPosStateLabelDataset(Dataset):
    def __init__(self, data_path):
        data = torch.load(data_path, weights_only=False)
        self.z = data['z'].float()
        self.obs_xy = data['obs_xy'].float()
        self.label = data['label'].float()
        self.num_safe = (self.label == 0).sum().item()
        self.num_unsafe = (self.label == 1).sum().item()

    def __len__(self):
        return len(self.z)

    def __getitem__(self, idx):
        return self.z[idx], self.obs_xy[idx], self.label[idx]

    def get_stats(self):
        return {'total': len(self.z), 'safe': self.num_safe, 'unsafe': self.num_unsafe}


class MultiPosTransitionDataset(Dataset):
    def __init__(self, data_path):
        data = torch.load(data_path, weights_only=False)
        self.z_k = data['z_k'].float()
        self.z_nom = data['z_nom'].float()
        self.obs_xy = data['obs_xy'].float()

    def __len__(self):
        return len(self.z_k)

    def __getitem__(self, idx):
        return self.z_k[idx], self.z_nom[idx], self.obs_xy[idx]

    def get_stats(self):
        return {'total': len(self.z_k)}


# =============================================================================
# Loss function — B(z, x, y)
# =============================================================================
def compute_cbf_loss(cbf_net, z, obs_xy, label, z_k, z_nom, obs_xy_trans,
                     lambda_s, lambda_u, lambda_d, alpha, delta_t,
                     safety_margin=1.0):
    safe_mask = (label == 0)
    unsafe_mask = (label == 1)

    # Term 1: Safe sign loss
    if safe_mask.sum() > 0:
        B_safe = cbf_net(z[safe_mask], obs_xy[safe_mask])
        L_safe = torch.mean(F.relu(-B_safe + safety_margin))
        safe_accuracy = (B_safe >= 0).float().mean().item()
        mean_B_safe = B_safe.mean().item()
    else:
        L_safe = torch.tensor(0.0, device=z.device)
        safe_accuracy, mean_B_safe = 0.0, 0.0

    # Term 2: Unsafe sign loss (symmetric)
    if unsafe_mask.sum() > 0:
        B_unsafe = cbf_net(z[unsafe_mask], obs_xy[unsafe_mask])
        L_unsafe = torch.mean(F.relu(B_unsafe + safety_margin))
        unsafe_accuracy = (B_unsafe < 0).float().mean().item()
        mean_B_unsafe = B_unsafe.mean().item()
    else:
        L_unsafe = torch.tensor(0.0, device=z.device)
        unsafe_accuracy, mean_B_unsafe = 0.0, 0.0

    # Term 3: Decrease condition
    B_k = cbf_net(z_k, obs_xy_trans)
    B_nom = cbf_net(z_nom, obs_xy_trans)
    target = (1.0 - alpha * delta_t) * B_k
    L_decrease = torch.mean(F.relu(target - B_nom))
    violation_rate = (B_nom < target).float().mean().item()

    loss = lambda_s * L_safe + lambda_u * L_unsafe + lambda_d * L_decrease

    metrics = {
        'L_safe': L_safe.item(), 'L_unsafe': L_unsafe.item(),
        'L_decrease': L_decrease.item(), 'L_total': loss.item(),
        'safe_accuracy': safe_accuracy, 'unsafe_accuracy': unsafe_accuracy,
        'violation_rate': violation_rate,
        'mean_B_safe': mean_B_safe, 'mean_B_unsafe': mean_B_unsafe,
    }
    return loss, metrics


# =============================================================================
# Training & Validation
# =============================================================================
def train_epoch(cbf_net, optimizer, label_loader, trans_loader, device,
                lambda_s, lambda_u, lambda_d, alpha, delta_t, safety_margin):
    cbf_net.train()
    epoch_metrics = {k: 0.0 for k in [
        'L_safe', 'L_unsafe', 'L_decrease', 'L_total',
        'safe_accuracy', 'unsafe_accuracy', 'violation_rate',
        'mean_B_safe', 'mean_B_unsafe'
    ]}
    num_batches = 0

    trans_iter = iter(trans_loader)
    for z, obs_xy, label in label_loader:
        try:
            z_k, z_nom, obs_xy_trans = next(trans_iter)
        except StopIteration:
            trans_iter = iter(trans_loader)
            z_k, z_nom, obs_xy_trans = next(trans_iter)

        z, obs_xy, label = z.to(device), obs_xy.to(device), label.to(device)
        z_k, z_nom, obs_xy_trans = z_k.to(device), z_nom.to(device), obs_xy_trans.to(device)

        optimizer.zero_grad()
        loss, metrics = compute_cbf_loss(
            cbf_net, z, obs_xy, label, z_k, z_nom, obs_xy_trans,
            lambda_s, lambda_u, lambda_d, alpha, delta_t, safety_margin
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(cbf_net.parameters(), 1.0)
        optimizer.step()

        for k, v in metrics.items():
            epoch_metrics[k] += v
        num_batches += 1

    for k in epoch_metrics:
        epoch_metrics[k] /= max(num_batches, 1)
    return epoch_metrics


def validate(cbf_net, label_loader, trans_loader, device,
             lambda_s, lambda_u, lambda_d, alpha, delta_t, safety_margin):
    cbf_net.eval()
    epoch_metrics = {k: 0.0 for k in [
        'L_safe', 'L_unsafe', 'L_decrease', 'L_total',
        'safe_accuracy', 'unsafe_accuracy', 'violation_rate',
        'mean_B_safe', 'mean_B_unsafe'
    ]}
    num_batches = 0

    trans_iter = iter(trans_loader)
    with torch.no_grad():
        for z, obs_xy, label in label_loader:
            try:
                z_k, z_nom, obs_xy_trans = next(trans_iter)
            except StopIteration:
                trans_iter = iter(trans_loader)
                z_k, z_nom, obs_xy_trans = next(trans_iter)

            z, obs_xy, label = z.to(device), obs_xy.to(device), label.to(device)
            z_k, z_nom, obs_xy_trans = z_k.to(device), z_nom.to(device), obs_xy_trans.to(device)

            _, metrics = compute_cbf_loss(
                cbf_net, z, obs_xy, label, z_k, z_nom, obs_xy_trans,
                lambda_s, lambda_u, lambda_d, alpha, delta_t, safety_margin
            )
            for k, v in metrics.items():
                epoch_metrics[k] += v
            num_batches += 1

    for k in epoch_metrics:
        epoch_metrics[k] /= max(num_batches, 1)
    return epoch_metrics


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Train multi-position CBF B(z, x, y)')
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=mcfg.SEED)
    parser.add_argument('--epochs', type=int, default=mcfg.CBF_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=mcfg.CBF_BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=mcfg.CBF_LR)
    parser.add_argument('--lambda_safe', type=float, default=mcfg.LAMBDA_SAFE)
    parser.add_argument('--lambda_unsafe', type=float, default=mcfg.LAMBDA_UNSAFE)
    parser.add_argument('--lambda_decrease', type=float, default=mcfg.LAMBDA_DECREASE)
    parser.add_argument('--alpha', type=float, default=mcfg.CBF_ALPHA)
    parser.add_argument('--delta_t', type=float, default=mcfg.CBF_DELTA_T)
    parser.add_argument('--safety_margin', type=float, default=mcfg.SAFETY_MARGIN)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--log_interval', type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logging.info(f"Using device: {device}")

    os.makedirs(mcfg.MULTIPOS_SNAPSHOT_DIR, exist_ok=True)

    # Load datasets
    logging.info("Loading datasets...")
    train_label = MultiPosStateLabelDataset(mcfg.MULTIPOS_STATE_LABELS_TRAIN)
    val_label = MultiPosStateLabelDataset(mcfg.MULTIPOS_STATE_LABELS_VAL)
    train_trans = MultiPosTransitionDataset(mcfg.MULTIPOS_TRANSITIONS_TRAIN)
    val_trans = MultiPosTransitionDataset(mcfg.MULTIPOS_TRANSITIONS_VAL)

    logging.info(f"State-label train: {train_label.get_stats()}")
    logging.info(f"State-label val:   {val_label.get_stats()}")
    logging.info(f"Transition train:  {train_trans.get_stats()}")
    logging.info(f"Transition val:    {val_trans.get_stats()}")

    train_label_loader = DataLoader(train_label, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_label_loader = DataLoader(val_label, batch_size=args.batch_size, shuffle=False)
    train_trans_loader = DataLoader(train_trans, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_trans_loader = DataLoader(val_trans, batch_size=args.batch_size, shuffle=False)

    # Model
    cbf_net = MultiPosBarrierNet(
        latent_dim=mcfg.LATENT_DIM,
        obs_xy_dim=mcfg.OBS_XY_DIM,
        hidden_units=mcfg.CBF_HIDDEN_UNITS,
        num_hidden=mcfg.CBF_NUM_HIDDEN
    )
    cbf_net.to(device)
    optimizer = optim.Adam(cbf_net.parameters(), lr=args.lr)

    param_count = sum(p.numel() for p in cbf_net.parameters())
    logging.info(f"MultiPosBarrierNet: {param_count:,} parameters")
    logging.info(f"  Input: z({mcfg.LATENT_DIM}) + obs_xy({mcfg.OBS_XY_DIM}) = {mcfg.LATENT_DIM + mcfg.OBS_XY_DIM}D")
    logging.info(f"  Positions: {mcfg.OBSTACLE_POSITIONS}")
    logging.info(f"  γ={args.safety_margin}, α={args.alpha}, Δt={args.delta_t}")

    best_score = -float('inf')
    best_epoch = 0

    logging.info(f"\n{'=' * 60}")
    logging.info(f"Starting B(z, x, y) training for {args.epochs} epochs")
    logging.info(f"{'=' * 60}\n")

    for epoch in range(1, args.epochs + 1):
        train_m = train_epoch(
            cbf_net, optimizer,
            train_label_loader, train_trans_loader, device,
            args.lambda_safe, args.lambda_unsafe, args.lambda_decrease,
            args.alpha, args.delta_t, args.safety_margin
        )
        val_m = validate(
            cbf_net,
            val_label_loader, val_trans_loader, device,
            args.lambda_safe, args.lambda_unsafe, args.lambda_decrease,
            args.alpha, args.delta_t, args.safety_margin
        )

        score = val_m['safe_accuracy'] + val_m['unsafe_accuracy']
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': cbf_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'score': score,
            }, mcfg.MULTIPOS_BEST_CHECKPOINT)

        if epoch % args.log_interval == 0:
            logging.info(
                f"Epoch {epoch}/{args.epochs} | "
                f"Train L={train_m['L_total']:.4f} "
                f"(S={train_m['L_safe']:.4f} U={train_m['L_unsafe']:.4f} "
                f"D={train_m['L_decrease']:.4f}) | "
                f"SafeAcc={train_m['safe_accuracy']*100:.1f}% "
                f"UnsafeAcc={train_m['unsafe_accuracy']*100:.1f}% | "
                f"Val SafeAcc={val_m['safe_accuracy']*100:.1f}% "
                f"UnsafeAcc={val_m['unsafe_accuracy']*100:.1f}%"
            )

        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': cbf_net.state_dict(),
                'score': score,
            }, os.path.join(mcfg.MULTIPOS_SNAPSHOT_DIR, f'barrier_net_epoch{epoch}.pt'))

    logging.info(f"\n{'=' * 60}")
    logging.info(f"Training complete! Best: epoch {best_epoch}, score={best_score:.4f}")
    logging.info(f"Saved to: {mcfg.MULTIPOS_BEST_CHECKPOINT}")
    logging.info(f"{'=' * 60}")


if __name__ == '__main__':
    main()
