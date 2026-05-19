"""
CBF Datasets — PyTorch Dataset classes for dual-batch CBF training.

Two dataset types:
    1. CBFStateLabelDataset:  (z, obs, label) for safe/unsafe sign losses
    2. CBFTransitionDataset:  (z_k, z_nom, obs) for CBF decrease condition
"""

import torch
from torch.utils.data import Dataset


class CBFStateLabelDataset(Dataset):
    """
    Dataset of latent states with safety labels.

    Each sample: (z, obs, label)
        z:     (latent_dim,) encoded robot state
        obs:   (obs_dim,)    obstacle parameters [x, y, h, r]
        label: scalar        0 = safe, 1 = unsafe

    Loaded from .pt file produced by cbf_generate_state_data.py.
    """

    def __init__(self, data_path):
        """
        Args:
            data_path: path to .pt file containing dict with keys 'z', 'obs', 'label'
        """
        data = torch.load(data_path, weights_only=False)
        self.z = data['z'].float()          # (N, latent_dim)
        self.obs = data['obs'].float()      # (N, obs_dim)
        self.label = data['label'].float()  # (N,)

        # Precompute masks for convenience
        self.safe_mask = (self.label == 0)
        self.unsafe_mask = (self.label == 1)

        self.num_safe = self.safe_mask.sum().item()
        self.num_unsafe = self.unsafe_mask.sum().item()

    def __len__(self):
        return len(self.z)

    def __getitem__(self, idx):
        return self.z[idx], self.obs[idx], self.label[idx]

    def get_stats(self):
        """Return dataset statistics."""
        return {
            'total': len(self.z),
            'num_safe': self.num_safe,
            'num_unsafe': self.num_unsafe,
            'safe_ratio': self.num_safe / len(self.z) if len(self.z) > 0 else 0,
        }


class CBFTransitionDataset(Dataset):
    """
    Dataset of transition pairs for CBF decrease condition training.

    Each sample: (z_k, z_nom, obs)
        z_k:   (latent_dim,) latent state before optimizer step
        z_nom: (latent_dim,) latent state after Goal+Prior optimizer step
        obs:   (obs_dim,)    obstacle parameters [x, y, h, r]

    Loaded from .pt file produced by cbf_generate_transition_data.py.
    """

    def __init__(self, data_path):
        """
        Args:
            data_path: path to .pt file containing dict with keys 'z_k', 'z_nom', 'obs'
        """
        data = torch.load(data_path, weights_only=False)
        self.z_k = data['z_k'].float()        # (M, latent_dim)
        self.z_nom = data['z_nom'].float()     # (M, latent_dim)
        self.obs = data['obs'].float()         # (M, obs_dim)

        # Optional: safety labels for z_k and z_nom (if available)
        self.safe_k = data.get('safe_k', None)
        self.safe_nom = data.get('safe_nom', None)

    def __len__(self):
        return len(self.z_k)

    def __getitem__(self, idx):
        item = (self.z_k[idx], self.z_nom[idx], self.obs[idx])
        if self.safe_k is not None:
            item = item + (self.safe_k[idx],)
        if self.safe_nom is not None:
            item = item + (self.safe_nom[idx],)
        return item

    def get_stats(self):
        """Return dataset statistics."""
        stats = {
            'total_transitions': len(self.z_k),
        }
        if self.safe_k is not None:
            stats['safe_k_count'] = (self.safe_k == 1).sum().item()
            stats['unsafe_k_count'] = (self.safe_k == 0).sum().item()
        return stats
