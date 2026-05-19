"""
CBF Training — Dual-batch training of the neural barrier function B_θ(z, o).

Training uses three loss terms:
    1. Safe sign loss:    λ_s · E[max(-B(z,o), 0)]     — push B ≥ 0 for safe states
    2. Unsafe sign loss:  λ_u · E[max(B(z,o), 0)]      — push B < 0 for unsafe states
    3. Decrease condition: λ_d · E[max(target - B_nom, 0)] — enforce forward invariance

Dual-batch strategy (Implementation Plan Section 5.3):
    Each iteration draws from two DataLoaders:
    - Batch 1: state-label dataset  (for Terms 1 & 2)
    - Batch 2: transition dataset   (for Term 3)
"""

from __future__ import print_function
import argparse
import json
import logging
import numpy as np
import os
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import warnings

warnings.filterwarnings('ignore', category=FutureWarning, module='torch')

from cbf_model import BarrierNet
from cbf_dataset import CBFStateLabelDataset, CBFTransitionDataset
import cbf_config as cfg

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


def compute_cbf_loss(cbf_net, z, obs, label, z_k, z_nom, obs_trans,
                     lambda_s, lambda_u, lambda_d, alpha, delta_t,
                     safety_margin=0.0):
    """
    Compute the three-term CBF training loss.

    Args:
        cbf_net:    BarrierNet model
        z, obs, label: from state-label batch (label: 0=safe, 1=unsafe)
        z_k, z_nom, obs_trans: from transition batch
        lambda_s, lambda_u, lambda_d: loss weights
        alpha, delta_t: CBF parameters
        safety_margin: margin γ — boundary at B = γ

    Returns:
        loss: total weighted loss
        metrics: dict with individual loss components and accuracies
    """
    # Split state-label batch into safe and unsafe
    safe_mask = (label == 0)
    unsafe_mask = (label == 1)

    # =========================================================================
    # Term 1: Safe sign loss
    # B(z, o) ≥ safety_margin for safe states
    # =========================================================================
    if safe_mask.sum() > 0:
        B_safe = cbf_net(z[safe_mask], obs[safe_mask])
        L_safe = torch.mean(F.relu(-B_safe + safety_margin))
        safe_accuracy = (B_safe >= 0).float().mean().item()
        mean_B_safe = B_safe.mean().item()
    else:
        L_safe = torch.tensor(0.0, device=z.device)
        safe_accuracy = 0.0
        mean_B_safe = 0.0

    # =========================================================================
    # Term 2: Unsafe sign loss
    # B(z, o) ≤ -safety_margin for unsafe states (symmetric margin)
    # =========================================================================
    if unsafe_mask.sum() > 0:
        B_unsafe = cbf_net(z[unsafe_mask], obs[unsafe_mask])
        L_unsafe = torch.mean(F.relu(B_unsafe + safety_margin))
        unsafe_accuracy = (B_unsafe < 0).float().mean().item()
        mean_B_unsafe = B_unsafe.mean().item()
    else:
        L_unsafe = torch.tensor(0.0, device=z.device)
        unsafe_accuracy = 0.0
        mean_B_unsafe = 0.0

    # =========================================================================
    # Term 3: CBF decrease condition
    # B(z_{k+1}^nom, o) ≥ (1 - α·Δ) · B(z_k, o)
    # =========================================================================
    B_k = cbf_net(z_k, obs_trans)
    B_nom = cbf_net(z_nom, obs_trans)
    target = (1.0 - alpha * delta_t) * B_k
    L_decrease = torch.mean(F.relu(target - B_nom))
    violation_rate = (B_nom < target).float().mean().item()

    # =========================================================================
    # Combined loss
    # =========================================================================
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


def train_epoch(cbf_net, optimizer, label_loader, trans_loader, device,
                lambda_s, lambda_u, lambda_d, alpha, delta_t, safety_margin,
                max_grad_norm=1.0):
    """Run one training epoch with dual-batch loading and gradient clipping."""
    cbf_net.train()
    epoch_metrics = {k: 0.0 for k in [
        'L_safe', 'L_unsafe', 'L_decrease', 'L_total',
        'safe_accuracy', 'unsafe_accuracy', 'violation_rate',
        'mean_B_safe', 'mean_B_unsafe'
    ]}
    num_batches = 0

    # Dual-batch: iterate both loaders simultaneously
    trans_iter = iter(trans_loader)
    for z, obs, label in label_loader:
        # Get transition batch (cycle if shorter)
        try:
            trans_batch = next(trans_iter)
        except StopIteration:
            trans_iter = iter(trans_loader)
            trans_batch = next(trans_iter)

        z_k, z_nom, obs_trans = trans_batch[0], trans_batch[1], trans_batch[2]

        z, obs, label = z.to(device), obs.to(device), label.to(device)
        z_k, z_nom, obs_trans = z_k.to(device), z_nom.to(device), obs_trans.to(device)

        optimizer.zero_grad()
        loss, metrics = compute_cbf_loss(
            cbf_net, z, obs, label, z_k, z_nom, obs_trans,
            lambda_s, lambda_u, lambda_d, alpha, delta_t, safety_margin
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(cbf_net.parameters(), max_grad_norm)
        optimizer.step()

        for k, v in metrics.items():
            epoch_metrics[k] += v
        num_batches += 1

    # Average over batches
    for k in epoch_metrics:
        epoch_metrics[k] /= max(num_batches, 1)

    return epoch_metrics


def validate(cbf_net, label_loader, trans_loader, device,
             lambda_s, lambda_u, lambda_d, alpha, delta_t, safety_margin):
    """Run validation (no gradient)."""
    cbf_net.eval()
    epoch_metrics = {k: 0.0 for k in [
        'L_safe', 'L_unsafe', 'L_decrease', 'L_total',
        'safe_accuracy', 'unsafe_accuracy', 'violation_rate',
        'mean_B_safe', 'mean_B_unsafe'
    ]}
    num_batches = 0

    trans_iter = iter(trans_loader)
    with torch.no_grad():
        for z, obs, label in label_loader:
            try:
                trans_batch = next(trans_iter)
            except StopIteration:
                trans_iter = iter(trans_loader)
                trans_batch = next(trans_iter)

            z_k, z_nom, obs_trans = trans_batch[0], trans_batch[1], trans_batch[2]

            z, obs, label = z.to(device), obs.to(device), label.to(device)
            z_k, z_nom, obs_trans = z_k.to(device), z_nom.to(device), obs_trans.to(device)

            _, metrics = compute_cbf_loss(
                cbf_net, z, obs, label, z_k, z_nom, obs_trans,
                lambda_s, lambda_u, lambda_d, alpha, delta_t, safety_margin
            )

            for k, v in metrics.items():
                epoch_metrics[k] += v
            num_batches += 1

    for k in epoch_metrics:
        epoch_metrics[k] /= max(num_batches, 1)

    return epoch_metrics


def main():
    parser = argparse.ArgumentParser(description='Train CBF barrier network')
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=cfg.SEED)
    parser.add_argument('--epochs', type=int, default=2000,
                        help='Training epochs (default: 2000 for thorough training)')
    parser.add_argument('--batch_size', type=int, default=cfg.CBF_BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=cfg.CBF_LR)
    parser.add_argument('--lambda_safe', type=float, default=cfg.LAMBDA_SAFE)
    parser.add_argument('--lambda_unsafe', type=float, default=cfg.LAMBDA_UNSAFE)
    parser.add_argument('--lambda_decrease', type=float, default=cfg.LAMBDA_DECREASE)
    parser.add_argument('--alpha', type=float, default=cfg.CBF_ALPHA)
    parser.add_argument('--delta_t', type=float, default=cfg.CBF_DELTA_T)
    parser.add_argument('--safety_margin', type=float, default=cfg.SAFETY_MARGIN)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Override checkpoint save directory (for ablation studies)')
    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logging.info(f"Using device: {device}")

    # Setup save directory
    if args.save_dir:
        save_dir = args.save_dir
    else:
        save_dir = cfg.CBF_SNAPSHOT_DIR
    os.makedirs(save_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(save_dir, 'barrier_net_best.pt')

    os.makedirs(cfg.CBF_SNAPSHOT_DIR, exist_ok=True)

    # =========================================================================
    # Load datasets
    # =========================================================================
    logging.info("Loading datasets...")

    train_label_dataset = CBFStateLabelDataset(cfg.STATE_LABELS_TRAIN)
    val_label_dataset = CBFStateLabelDataset(cfg.STATE_LABELS_VAL)
    train_trans_dataset = CBFTransitionDataset(cfg.TRANSITIONS_TRAIN)
    val_trans_dataset = CBFTransitionDataset(cfg.TRANSITIONS_VAL)

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
    # Initialize model
    # =========================================================================
    cbf_net = BarrierNet(
        latent_dim=cfg.LATENT_DIM,
        obs_dim=cfg.OBS_DIM,
        hidden_units=cfg.CBF_HIDDEN_UNITS,
        num_hidden=cfg.CBF_NUM_HIDDEN
    )
    cbf_net.to(device)
    optimizer = optim.Adam(cbf_net.parameters(), lr=args.lr)

    # Cosine annealing LR scheduler for smooth convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    num_params = sum(p.numel() for p in cbf_net.parameters())
    logging.info(f"BarrierNet: {num_params:,} parameters")

    # =========================================================================
    # Training loop
    # =========================================================================
    logging.info(f"\n{'=' * 60}")
    logging.info(f"Training CBF for {args.epochs} epochs")
    logging.info(f"  λ_safe={args.lambda_safe}, λ_unsafe={args.lambda_unsafe}, "
                 f"λ_decrease={args.lambda_decrease}")
    logging.info(f"  α={args.alpha}, Δt={args.delta_t}, margin={args.safety_margin}")
    logging.info(f"{'=' * 60}\n")

    best_score = -float('inf')
    best_epoch = 0

    # Optional: TensorBoard
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=cfg.CBF_TENSORBOARD_DIR)
        use_tb = True
    except ImportError:
        use_tb = False
        logging.info("TensorBoard not available, skipping.")

    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(
            cbf_net, optimizer, train_label_loader, train_trans_loader, device,
            args.lambda_safe, args.lambda_unsafe, args.lambda_decrease,
            args.alpha, args.delta_t, args.safety_margin,
            max_grad_norm=1.0
        )

        # Step learning rate scheduler
        scheduler.step()

        # Validate
        val_metrics = validate(
            cbf_net, val_label_loader, val_trans_loader, device,
            args.lambda_safe, args.lambda_unsafe, args.lambda_decrease,
            args.alpha, args.delta_t, args.safety_margin
        )

        # Log
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

        # TensorBoard logging
        if use_tb:
            for k, v in train_metrics.items():
                writer.add_scalar(f'train/{k}', v, epoch)
            for k, v in val_metrics.items():
                writer.add_scalar(f'val/{k}', v, epoch)

        # Checkpoint selection: composite score
        # Higher = better: safe_acc + unsafe_acc + (1 - violation_rate)
        score = (val_metrics['safe_accuracy'] +
                 val_metrics['unsafe_accuracy'] +
                 (1.0 - val_metrics['violation_rate']))

        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': cbf_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'score': score,
                'config': {
                    'alpha': args.alpha,
                    'delta_t': args.delta_t,
                    'lambda_safe': args.lambda_safe,
                    'lambda_unsafe': args.lambda_unsafe,
                    'lambda_decrease': args.lambda_decrease,
                    'safety_margin': args.safety_margin,
                    'lr': args.lr,
                    'hidden_units': cfg.CBF_HIDDEN_UNITS,
                    'num_hidden': cfg.CBF_NUM_HIDDEN,
                    'latent_dim': cfg.LATENT_DIM,
                    'obs_dim': cfg.OBS_DIM,
                }
            }, best_checkpoint_path)
            logging.info(f"  ★ New best model saved (score={score:.4f})")

        # Periodic checkpoint
        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(save_dir, f'barrier_net_epoch_{epoch:04d}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': cbf_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
            }, ckpt_path)

    if use_tb:
        writer.close()

    logging.info(f"\n{'=' * 60}")
    logging.info(f"Training complete!")
    logging.info(f"Best model: epoch {best_epoch}, score={best_score:.4f}")
    logging.info(f"Saved to: {best_checkpoint_path}")
    logging.info(f"{'=' * 60}")


if __name__ == '__main__':
    main()
