"""
Fixed-Obstacle CBF Training — B(z) without obstacle input.

BarrierNet takes only latent code z as input (no obs concatenation).
The obstacle is baked into the data — all samples share the same fixed obstacle.

Three-term loss:
    1. Safe sign loss:    relu(-B(z) + γ)     — push B ≥ γ for safe states
    2. Unsafe sign loss:  relu(B(z) + γ)      — push B ≤ -γ for unsafe states (symmetric)
    3. Decrease condition: relu(target - B_nom) — enforce forward invariance
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

import cbf_fixed_config as fcfg

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


# =============================================================================
# BarrierNet B(z) — NO obstacle input
# =============================================================================
class FixedBarrierNet(nn.Module):
    """
    Neural Control Barrier Function B(z) for a FIXED obstacle.

    Input:  z (latent code only, no obstacle descriptor)
    Output: Scalar barrier value B (signed, unbounded)
            B(z) ≥ 0  →  Safe
            B(z) < 0  →  Unsafe
    """

    def __init__(self, latent_dim=7, hidden_units=2048, num_hidden=4):
        super(FixedBarrierNet, self).__init__()

        self.latent_dim = latent_dim

        # Input layer: z only (no obs concatenation)
        self.fc_in = nn.Linear(latent_dim, hidden_units)

        # Hidden layers
        self.fc_hidden = nn.ModuleList(
            [nn.Linear(hidden_units, hidden_units) for _ in range(num_hidden - 1)]
        )

        # Output layer: scalar barrier value
        self.fc_out = nn.Linear(hidden_units, 1)

    def forward(self, z):
        """
        Compute barrier value B(z).

        Args:
            z: (batch, latent_dim) latent codes

        Returns:
            B: (batch,) scalar barrier values
        """
        h = F.elu(self.fc_in(z.view(-1, self.latent_dim)))
        for fc in self.fc_hidden:
            h = F.elu(fc(h))
        return self.fc_out(h).view(-1)


# =============================================================================
# Dataset classes (no obs field)
# =============================================================================
class FixedStateLabelDataset(Dataset):
    """State-label dataset: (z, label) — no obs."""

    def __init__(self, data_path):
        data = torch.load(data_path, weights_only=False)
        self.z = data['z'].float()
        self.label = data['label'].float()
        self.safe_mask = (self.label == 0)
        self.unsafe_mask = (self.label == 1)
        self.num_safe = self.safe_mask.sum().item()
        self.num_unsafe = self.unsafe_mask.sum().item()

    def __len__(self):
        return len(self.z)

    def __getitem__(self, idx):
        return self.z[idx], self.label[idx]

    def get_stats(self):
        return {
            'total': len(self.z),
            'num_safe': self.num_safe,
            'num_unsafe': self.num_unsafe,
            'safe_ratio': self.num_safe / len(self.z) if len(self.z) > 0 else 0,
        }


class FixedTransitionDataset(Dataset):
    """Transition dataset: (z_k, z_nom) — no obs."""

    def __init__(self, data_path):
        data = torch.load(data_path, weights_only=False)
        self.z_k = data['z_k'].float()
        self.z_nom = data['z_nom'].float()
        self.safe_k = data.get('safe_k', None)
        self.safe_nom = data.get('safe_nom', None)

    def __len__(self):
        return len(self.z_k)

    def __getitem__(self, idx):
        return self.z_k[idx], self.z_nom[idx]

    def get_stats(self):
        stats = {'total_transitions': len(self.z_k)}
        if self.safe_k is not None:
            stats['safe_k_count'] = (self.safe_k == 1).sum().item()
            stats['unsafe_k_count'] = (self.safe_k == 0).sum().item()
        return stats


# =============================================================================
# Loss function — B(z) with no obs
# =============================================================================
def compute_cbf_loss(cbf_net, z, label, z_k, z_nom,
                     lambda_s, lambda_u, lambda_d, alpha, delta_t,
                     safety_margin=1.0):
    """
    Three-term CBF loss for fixed-obstacle B(z).

    Same structure as main cbf_train.py but without obs arguments.
    Uses symmetric margin: safe → B ≥ γ, unsafe → B ≤ -γ.
    """
    safe_mask = (label == 0)
    unsafe_mask = (label == 1)

    # Term 1: Safe sign loss — B(z) ≥ safety_margin
    if safe_mask.sum() > 0:
        B_safe = cbf_net(z[safe_mask])
        L_safe = torch.mean(F.relu(-B_safe + safety_margin))
        safe_accuracy = (B_safe >= 0).float().mean().item()
        mean_B_safe = B_safe.mean().item()
    else:
        L_safe = torch.tensor(0.0, device=z.device)
        safe_accuracy = 0.0
        mean_B_safe = 0.0

    # Term 2: Unsafe sign loss — B(z) ≤ -safety_margin (symmetric)
    if unsafe_mask.sum() > 0:
        B_unsafe = cbf_net(z[unsafe_mask])
        L_unsafe = torch.mean(F.relu(B_unsafe + safety_margin))
        unsafe_accuracy = (B_unsafe < 0).float().mean().item()
        mean_B_unsafe = B_unsafe.mean().item()
    else:
        L_unsafe = torch.tensor(0.0, device=z.device)
        unsafe_accuracy = 0.0
        mean_B_unsafe = 0.0

    # Term 3: Decrease condition — B(z_nom) ≥ (1 - α·Δ) · B(z_k)
    B_k = cbf_net(z_k)
    B_nom = cbf_net(z_nom)
    target = (1.0 - alpha * delta_t) * B_k
    L_decrease = torch.mean(F.relu(target - B_nom))
    violation_rate = (B_nom < target).float().mean().item()

    # Combined loss
    loss = lambda_s * L_safe + lambda_u * L_unsafe + lambda_d * L_decrease

    metrics = {
        'L_safe': L_safe.item(),
        'L_unsafe': L_unsafe.item(),
        'L_decrease': L_decrease.item(),
        'L_total': loss.item(),
        'safe_accuracy': safe_accuracy,
        'unsafe_accuracy': unsafe_accuracy,
        'violation_rate': violation_rate,
        'mean_B_safe': mean_B_safe,
        'mean_B_unsafe': mean_B_unsafe,
    }

    return loss, metrics


# =============================================================================
# Training & Validation
# =============================================================================
def train_epoch(cbf_net, optimizer, label_loader, trans_loader, device,
                lambda_s, lambda_u, lambda_d, alpha, delta_t, safety_margin,
                max_grad_norm=1.0):
    """One training epoch with dual-batch loading."""
    cbf_net.train()
    epoch_metrics = {k: 0.0 for k in [
        'L_safe', 'L_unsafe', 'L_decrease', 'L_total',
        'safe_accuracy', 'unsafe_accuracy', 'violation_rate',
        'mean_B_safe', 'mean_B_unsafe'
    ]}
    num_batches = 0

    trans_iter = iter(trans_loader)
    for z, label in label_loader:
        try:
            trans_batch = next(trans_iter)
        except StopIteration:
            trans_iter = iter(trans_loader)
            trans_batch = next(trans_iter)

        z_k, z_nom = trans_batch[0], trans_batch[1]

        z, label = z.to(device), label.to(device)
        z_k, z_nom = z_k.to(device), z_nom.to(device)

        optimizer.zero_grad()
        loss, metrics = compute_cbf_loss(
            cbf_net, z, label, z_k, z_nom,
            lambda_s, lambda_u, lambda_d, alpha, delta_t, safety_margin
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(cbf_net.parameters(), max_grad_norm)
        optimizer.step()

        for k, v in metrics.items():
            epoch_metrics[k] += v
        num_batches += 1

    for k in epoch_metrics:
        epoch_metrics[k] /= max(num_batches, 1)

    return epoch_metrics


def validate(cbf_net, label_loader, trans_loader, device,
             lambda_s, lambda_u, lambda_d, alpha, delta_t, safety_margin):
    """Validation (no gradient)."""
    cbf_net.eval()
    epoch_metrics = {k: 0.0 for k in [
        'L_safe', 'L_unsafe', 'L_decrease', 'L_total',
        'safe_accuracy', 'unsafe_accuracy', 'violation_rate',
        'mean_B_safe', 'mean_B_unsafe'
    ]}
    num_batches = 0

    trans_iter = iter(trans_loader)
    with torch.no_grad():
        for z, label in label_loader:
            try:
                trans_batch = next(trans_iter)
            except StopIteration:
                trans_iter = iter(trans_loader)
                trans_batch = next(trans_iter)

            z_k, z_nom = trans_batch[0], trans_batch[1]

            z, label = z.to(device), label.to(device)
            z_k, z_nom = z_k.to(device), z_nom.to(device)

            _, metrics = compute_cbf_loss(
                cbf_net, z, label, z_k, z_nom,
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
    parser = argparse.ArgumentParser(description='Train FIXED-obstacle CBF B(z)')
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=fcfg.SEED)
    parser.add_argument('--epochs', type=int, default=fcfg.CBF_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=fcfg.CBF_BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=fcfg.CBF_LR)
    parser.add_argument('--lambda_safe', type=float, default=fcfg.LAMBDA_SAFE)
    parser.add_argument('--lambda_unsafe', type=float, default=fcfg.LAMBDA_UNSAFE)
    parser.add_argument('--lambda_decrease', type=float, default=fcfg.LAMBDA_DECREASE)
    parser.add_argument('--alpha', type=float, default=fcfg.CBF_ALPHA)
    parser.add_argument('--delta_t', type=float, default=fcfg.CBF_DELTA_T)
    parser.add_argument('--safety_margin', type=float, default=fcfg.SAFETY_MARGIN)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--log_interval', type=int, default=1)
    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logging.info(f"Using device: {device}")

    os.makedirs(fcfg.FIXED_SNAPSHOT_DIR, exist_ok=True)

    # =========================================================================
    # Load datasets
    # =========================================================================
    logging.info("Loading datasets...")

    train_label_dataset = FixedStateLabelDataset(fcfg.FIXED_STATE_LABELS_TRAIN)
    val_label_dataset = FixedStateLabelDataset(fcfg.FIXED_STATE_LABELS_VAL)
    train_trans_dataset = FixedTransitionDataset(fcfg.FIXED_TRANSITIONS_TRAIN)
    val_trans_dataset = FixedTransitionDataset(fcfg.FIXED_TRANSITIONS_VAL)

    logging.info(f"State-label train: {train_label_dataset.get_stats()}")
    logging.info(f"State-label val:   {val_label_dataset.get_stats()}")
    logging.info(f"Transition train:  {train_trans_dataset.get_stats()}")
    logging.info(f"Transition val:    {val_trans_dataset.get_stats()}")

    train_label_loader = DataLoader(
        train_label_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    val_label_loader = DataLoader(
        val_label_dataset, batch_size=args.batch_size, shuffle=False
    )
    train_trans_loader = DataLoader(
        train_trans_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    val_trans_loader = DataLoader(
        val_trans_dataset, batch_size=args.batch_size, shuffle=False
    )

    # =========================================================================
    # Initialize model — B(z) with NO obs input
    # =========================================================================
    cbf_net = FixedBarrierNet(
        latent_dim=fcfg.LATENT_DIM,
        hidden_units=fcfg.CBF_HIDDEN_UNITS,
        num_hidden=fcfg.CBF_NUM_HIDDEN
    )
    cbf_net.to(device)
    optimizer = optim.Adam(cbf_net.parameters(), lr=args.lr)

    param_count = sum(p.numel() for p in cbf_net.parameters())
    logging.info(f"FixedBarrierNet: {param_count:,} parameters")
    logging.info(f"  Input: z ({fcfg.LATENT_DIM}D) — NO obs input")
    logging.info(f"  Hidden: {fcfg.CBF_HIDDEN_UNITS} × {fcfg.CBF_NUM_HIDDEN}")
    logging.info(f"  Safety margin: γ = {args.safety_margin}")
    logging.info(f"  α = {args.alpha}, Δt = {args.delta_t}")

    # =========================================================================
    # Training loop
    # =========================================================================
    best_score = -float('inf')
    best_epoch = 0
    best_checkpoint_path = os.path.join(fcfg.FIXED_SNAPSHOT_DIR, 'barrier_net_best.pt')

    logging.info(f"\n{'=' * 60}")
    logging.info(f"Starting FIXED-obstacle B(z) training for {args.epochs} epochs")
    logging.info(f"{'=' * 60}\n")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            cbf_net, optimizer,
            train_label_loader, train_trans_loader, device,
            args.lambda_safe, args.lambda_unsafe, args.lambda_decrease,
            args.alpha, args.delta_t, args.safety_margin
        )

        val_metrics = validate(
            cbf_net,
            val_label_loader, val_trans_loader, device,
            args.lambda_safe, args.lambda_unsafe, args.lambda_decrease,
            args.alpha, args.delta_t, args.safety_margin
        )

        # Score: safe_acc + unsafe_acc (higher = better)
        score = val_metrics['safe_accuracy'] + val_metrics['unsafe_accuracy']

        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': cbf_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'score': score,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
            }, best_checkpoint_path)

        if epoch % args.log_interval == 0:
            logging.info(
                f"Epoch {epoch}/{args.epochs} | "
                f"Train L={train_metrics['L_total']:.4f} "
                f"(S={train_metrics['L_safe']:.4f} U={train_metrics['L_unsafe']:.4f} "
                f"D={train_metrics['L_decrease']:.4f}) | "
                f"SafeAcc={train_metrics['safe_accuracy']*100:.1f}% "
                f"UnsafeAcc={train_metrics['unsafe_accuracy']*100:.1f}% "
                f"ViolRate={train_metrics['violation_rate']*100:.1f}% | "
                f"Val L={val_metrics['L_total']:.4f} "
                f"SafeAcc={val_metrics['safe_accuracy']*100:.1f}% "
                f"UnsafeAcc={val_metrics['unsafe_accuracy']*100:.1f}%"
            )

        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(fcfg.FIXED_SNAPSHOT_DIR, f'barrier_net_epoch{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': cbf_net.state_dict(),
                'score': score,
            }, ckpt_path)

    logging.info(f"\n{'=' * 60}")
    logging.info(f"Training complete!")
    logging.info(f"Best model: epoch {best_epoch}, score={best_score:.4f}")
    logging.info(f"Saved to: {best_checkpoint_path}")
    logging.info(f"{'=' * 60}")


if __name__ == '__main__':
    main()
