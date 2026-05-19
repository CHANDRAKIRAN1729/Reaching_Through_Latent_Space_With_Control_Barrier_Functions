"""
Fixed-Obstacle Classifier Configuration.

Paths and hyperparameters for the fixed-obstacle collision classifier C(z).
Uses the same obstacle/goal as cbf_fixed_config.py for direct comparison.
"""

import os
import cbf_config as cfg
import cbf_fixed_config as fcfg

# =============================================================================
# Fixed Scenario (same as CBF fixed experiment)
# =============================================================================
FIXED_OBSTACLE = fcfg.FIXED_OBSTACLE        # [0.5, 0.0, 0.75, 0.1]
FIXED_GOAL_Q_RAD = fcfg.FIXED_GOAL_Q_RAD    # [0.5, -0.5, 0.0, -2.0, 0.0, 2.0, 0.8]

# =============================================================================
# Paths
# =============================================================================
DATA_DIR = os.path.join(cfg.PROJECT_ROOT, 'classifier_fixed_data')
TRAIN_DATA = os.path.join(DATA_DIR, 'classifier_train.pt')
VAL_DATA = os.path.join(DATA_DIR, 'classifier_val.pt')

SNAPSHOT_DIR = os.path.join(cfg.PROJECT_ROOT, 'model_params/panda_10k/classifier_fixed_snapshots')
BEST_CHECKPOINT = os.path.join(SNAPSHOT_DIR, 'classifier_best.pt')

# =============================================================================
# Data Generation
# =============================================================================
NUM_SAMPLES = 1150000       # random robot configs to sample
VAL_SPLIT = 0.1

# =============================================================================
# Architecture — C(z) without obs input
# =============================================================================
LATENT_DIM = cfg.LATENT_DIM  # 7
HIDDEN_UNITS = 2048
NUM_HIDDEN = 4

# =============================================================================
# Training
# =============================================================================
BATCH_SIZE = 512
EPOCHS = 1000
LR = 1e-4
SEED = cfg.SEED
SUCCESS_THRESHOLD = cfg.SUCCESS_THRESHOLD
