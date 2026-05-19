"""
Fixed-Obstacle Classifier Training — C(z) without obs input.

Binary classifier: C(z) → logit → sigmoid → P(collision)
Trained with BCE loss on (z, label) pairs.

Analogous to cbf_fixed_train.py but for the classifier approach.
"""

from __future__ import print_function
import argparse
import logging
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings('ignore', category=FutureWarning, module='torch')

import classifier_fixed_config as clcfg

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


# =============================================================================
# Fixed Collision Classifier C(z) — no obs input
# =============================================================================
class FixedCollisionClassifier(nn.Module):
    """
    Binary collision classifier C(z) for a fixed obstacle.
    Input: z (latent_dim), Output: logit (scalar).
    P(collision) = sigmoid(logit).
    """

    def __init__(self, latent_dim=7, hidden_units=2048, num_hidden=4):
        super(FixedCollisionClassifier, self).__init__()
        self.latent_dim = latent_dim

        self.fc_in = nn.Linear(latent_dim, hidden_units)
        self.fc_hidden = nn.ModuleList(
            [nn.Linear(hidden_units, hidden_units) for _ in range(num_hidden - 1)]
        )
        self.fc_out = nn.Linear(hidden_units, 1)

    def forward(self, z):
        """Returns logit (before sigmoid)."""
        h = F.elu(self.fc_in(z.view(-1, self.latent_dim)))
        for fc in self.fc_hidden:
            h = F.elu(fc(h))
        return self.fc_out(h).view(-1)


# =============================================================================
# Dataset
# =============================================================================
class ClassifierDataset(Dataset):
    def __init__(self, data_path):
        data = torch.load(data_path, weights_only=False)
        self.z = data['z'].float()
        self.label = data['label'].float()
        self.num_safe = (self.label == 0).sum().item()
        self.num_unsafe = (self.label == 1).sum().item()

    def __len__(self):
        return len(self.z)

    def __getitem__(self, idx):
        return self.z[idx], self.label[idx]

    def get_stats(self):
        return {'total': len(self.z), 'safe': self.num_safe, 'unsafe': self.num_unsafe}


# =============================================================================
# Training
# =============================================================================
def train_epoch(model, optimizer, loader, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for z, label in loader:
        z, label = z.to(device), label.to(device)

        optimizer.zero_grad()
        logit = model(z)
        loss = F.binary_cross_entropy_with_logits(logit, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * z.size(0)

        pred = (torch.sigmoid(logit) >= 0.5).float()
        correct += (pred == label).sum().item()
        total += z.size(0)

    return total_loss / total, correct / total


def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    safe_correct = 0
    safe_total = 0
    unsafe_correct = 0
    unsafe_total = 0

    with torch.no_grad():
        for z, label in loader:
            z, label = z.to(device), label.to(device)

            logit = model(z)
            loss = F.binary_cross_entropy_with_logits(logit, label)
            total_loss += loss.item() * z.size(0)

            pred = (torch.sigmoid(logit) >= 0.5).float()
            correct += (pred == label).sum().item()
            total += z.size(0)

            safe_mask = (label == 0)
            unsafe_mask = (label == 1)
            if safe_mask.sum() > 0:
                safe_correct += (pred[safe_mask] == 0).sum().item()
                safe_total += safe_mask.sum().item()
            if unsafe_mask.sum() > 0:
                unsafe_correct += (pred[unsafe_mask] == 1).sum().item()
                unsafe_total += unsafe_mask.sum().item()

    return {
        'loss': total_loss / total,
        'accuracy': correct / total,
        'safe_accuracy': safe_correct / max(safe_total, 1),
        'unsafe_accuracy': unsafe_correct / max(unsafe_total, 1),
    }


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Train fixed-obstacle classifier C(z)')
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=clcfg.SEED)
    parser.add_argument('--epochs', type=int, default=clcfg.EPOCHS)
    parser.add_argument('--batch_size', type=int, default=clcfg.BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=clcfg.LR)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--log_interval', type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logging.info(f"Using device: {device}")

    os.makedirs(clcfg.SNAPSHOT_DIR, exist_ok=True)

    # Load datasets
    logging.info("Loading datasets...")
    train_ds = ClassifierDataset(clcfg.TRAIN_DATA)
    val_ds = ClassifierDataset(clcfg.VAL_DATA)

    logging.info(f"Train: {train_ds.get_stats()}")
    logging.info(f"Val:   {val_ds.get_stats()}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Model
    model = FixedCollisionClassifier(
        latent_dim=clcfg.LATENT_DIM,
        hidden_units=clcfg.HIDDEN_UNITS,
        num_hidden=clcfg.NUM_HIDDEN
    )
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    param_count = sum(p.numel() for p in model.parameters())
    logging.info(f"FixedCollisionClassifier: {param_count:,} parameters")
    logging.info(f"  Input: z({clcfg.LATENT_DIM})")
    logging.info(f"  Obstacle: {clcfg.FIXED_OBSTACLE}")

    best_val_acc = 0.0
    best_epoch = 0

    logging.info(f"\n{'=' * 60}")
    logging.info(f"Starting C(z) training for {args.epochs} epochs")
    logging.info(f"{'=' * 60}\n")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, optimizer, train_loader, device)
        val_metrics = validate(model, val_loader, device)

        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_metrics['accuracy'],
            }, clcfg.BEST_CHECKPOINT)

        if epoch % args.log_interval == 0:
            logging.info(
                f"Epoch {epoch}/{args.epochs} | "
                f"Train Loss={train_loss:.4f} Acc={train_acc*100:.1f}% | "
                f"Val Loss={val_metrics['loss']:.4f} "
                f"Acc={val_metrics['accuracy']*100:.1f}% "
                f"SafeAcc={val_metrics['safe_accuracy']*100:.1f}% "
                f"UnsafeAcc={val_metrics['unsafe_accuracy']*100:.1f}%"
            )

        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_accuracy': val_metrics['accuracy'],
            }, os.path.join(clcfg.SNAPSHOT_DIR, f'classifier_epoch{epoch}.pt'))

    logging.info(f"\n{'=' * 60}")
    logging.info(f"Training complete! Best: epoch {best_epoch}, "
                 f"val_acc={best_val_acc*100:.1f}%")
    logging.info(f"Saved to: {clcfg.BEST_CHECKPOINT}")
    logging.info(f"{'=' * 60}")


if __name__ == '__main__':
    main()
