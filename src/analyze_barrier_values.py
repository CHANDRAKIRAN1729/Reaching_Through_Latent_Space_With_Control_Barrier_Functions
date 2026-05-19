"""Analyze B value distribution from current barrier net to determine safety margin γ."""
import torch
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cbf_model import BarrierNet
from cbf_dataset import CBFStateLabelDataset, CBFTransitionDataset
import cbf_config as cfg

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
cbf_net = BarrierNet(cfg.LATENT_DIM, cfg.OBS_DIM, cfg.CBF_HIDDEN_UNITS, cfg.CBF_NUM_HIDDEN).to(device)
ckpt = torch.load(cfg.CBF_BEST_CHECKPOINT, weights_only=False)
cbf_net.load_state_dict(ckpt['model_state_dict'])
cbf_net.eval()
print(f"Loaded checkpoint epoch {ckpt.get('epoch', '?')}")

# Analyze state-label data
for split, path in [('train', cfg.STATE_LABELS_TRAIN), ('val', cfg.STATE_LABELS_VAL)]:
    ds = CBFStateLabelDataset(path)
    z_all, obs_all, labels = ds.z.to(device), ds.obs.to(device), ds.label.to(device)
    
    with torch.no_grad():
        B_all = cbf_net(z_all, obs_all).cpu().numpy()
    
    labels_np = labels.cpu().numpy()
    B_safe = B_all[labels_np == 0]
    B_unsafe = B_all[labels_np == 1]
    
    print(f"\n{'='*60}")
    print(f"STATE-LABEL {split.upper()} — {len(B_all)} samples")
    print(f"{'='*60}")
    
    print(f"\n  SAFE ({len(B_safe)} samples):")
    print(f"    Mean:   {B_safe.mean():.4f}")
    print(f"    Std:    {B_safe.std():.4f}")
    print(f"    Min:    {B_safe.min():.4f}")
    print(f"    Max:    {B_safe.max():.4f}")
    for p in [1, 5, 10, 25, 50]:
        print(f"    P{p:02d}:    {np.percentile(B_safe, p):.4f}")
    
    print(f"\n  UNSAFE ({len(B_unsafe)} samples):")
    print(f"    Mean:   {B_unsafe.mean():.4f}")
    print(f"    Std:    {B_unsafe.std():.4f}")
    print(f"    Min:    {B_unsafe.min():.4f}")
    print(f"    Max:    {B_unsafe.max():.4f}")
    for p in [50, 75, 90, 95, 99]:
        print(f"    P{p:02d}:    {np.percentile(B_unsafe, p):.4f}")
    
    # Overlap analysis
    print(f"\n  OVERLAP ANALYSIS:")
    print(f"    Safe with B < 0:      {(B_safe < 0).sum()}/{len(B_safe)} ({(B_safe < 0).mean()*100:.1f}%)")
    print(f"    Unsafe with B > 0:    {(B_unsafe > 0).sum()}/{len(B_unsafe)} ({(B_unsafe > 0).mean()*100:.1f}%)")
    for gamma in [0.0, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0]:
        safe_ok = (B_safe >= gamma).mean() * 100
        unsafe_ok = (B_unsafe <= gamma).mean() * 100
        print(f"    γ={gamma:.1f}: safe B≥γ={safe_ok:.1f}%, unsafe B≤γ={unsafe_ok:.1f}%, both={min(safe_ok,unsafe_ok):.1f}%")

# Analyze transition data
for split, path in [('train', cfg.TRANSITIONS_TRAIN)]:
    ds = CBFTransitionDataset(path)
    if ds.safe_k is None:
        print(f"\n  Transition {split}: no safe_k labels")
        continue
    
    z_k, obs_tr = ds.z_k.to(device), ds.obs.to(device)
    safe_k = ds.safe_k.numpy()
    
    with torch.no_grad():
        B_traj = cbf_net(z_k, obs_tr).cpu().numpy()
    
    B_traj_safe = B_traj[safe_k == 1]
    B_traj_unsafe = B_traj[safe_k == 0]
    
    print(f"\n{'='*60}")
    print(f"TRAJECTORY {split.upper()} — {len(B_traj)} samples")
    print(f"{'='*60}")
    
    print(f"\n  SAFE ({len(B_traj_safe)} samples):")
    print(f"    Mean: {B_traj_safe.mean():.4f}, Std: {B_traj_safe.std():.4f}")
    print(f"    Min:  {B_traj_safe.min():.4f}, P05: {np.percentile(B_traj_safe, 5):.4f}")
    
    print(f"\n  UNSAFE ({len(B_traj_unsafe)} samples):")
    print(f"    Mean: {B_traj_unsafe.mean():.4f}, Std: {B_traj_unsafe.std():.4f}")
    print(f"    Max:  {B_traj_unsafe.max():.4f}, P95: {np.percentile(B_traj_unsafe, 95):.4f}")
    
    for gamma in [0.0, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0]:
        safe_ok = (B_traj_safe >= gamma).mean() * 100
        unsafe_ok = (B_traj_unsafe <= gamma).mean() * 100
        print(f"    γ={gamma:.1f}: safe B≥γ={safe_ok:.1f}%, unsafe B≤γ={unsafe_ok:.1f}%")
