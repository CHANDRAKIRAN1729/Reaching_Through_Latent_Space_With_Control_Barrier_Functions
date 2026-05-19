"""
Multi-Position CBF Configuration.

3-4 obstacle positions with FIXED height and radius.
BarrierNet input: B(z, x, y) — conditioned on obstacle (x,y) only.
"""

import os
import cbf_config as cfg

# =============================================================================
# Fixed Obstacle Shape (same for all positions)
# =============================================================================
FIXED_H = 0.75
FIXED_R = 0.1

# Multiple obstacle positions [x, y]
OBSTACLE_POSITIONS = [
    [0.4, -0.2],
    [0.5, 0.0],
    [0.6, 0.2],
    [0.5, 0.3],
]

# Full obstacle descriptors (for collision checking)
OBSTACLES_FULL = [[x, y, FIXED_H, FIXED_R] for x, y in OBSTACLE_POSITIONS]

# Fixed goal (same as single-obstacle experiment)
FIXED_GOAL_Q_RAD = [0.5, -0.5, 0.0, -2.0, 0.0, 2.0, 0.8]

# =============================================================================
# Paths
# =============================================================================
MULTIPOS_DATA_DIR = os.path.join(cfg.PROJECT_ROOT, 'cbf_multipos_data')
MULTIPOS_STATE_LABELS_TRAIN = os.path.join(MULTIPOS_DATA_DIR, 'state_labels_train.pt')
MULTIPOS_STATE_LABELS_VAL = os.path.join(MULTIPOS_DATA_DIR, 'state_labels_val.pt')
MULTIPOS_TRANSITIONS_TRAIN = os.path.join(MULTIPOS_DATA_DIR, 'transitions_train.pt')
MULTIPOS_TRANSITIONS_VAL = os.path.join(MULTIPOS_DATA_DIR, 'transitions_val.pt')

MULTIPOS_SNAPSHOT_DIR = os.path.join(cfg.PROJECT_ROOT, 'model_params/panda_10k/cbf_multipos_snapshots')
MULTIPOS_BEST_CHECKPOINT = os.path.join(MULTIPOS_SNAPSHOT_DIR, 'barrier_net_best.pt')

# =============================================================================
# Architecture
# =============================================================================
LATENT_DIM = cfg.LATENT_DIM       # 7
OBS_XY_DIM = 2                    # only (x, y) — h and r are fixed
CBF_HIDDEN_UNITS = cfg.CBF_HIDDEN_UNITS
CBF_NUM_HIDDEN = cfg.CBF_NUM_HIDDEN

# =============================================================================
# Training
# =============================================================================
NUM_SCENARIOS = 5000    # per obstacle position → 20K total
MAX_STEPS = 300
PLANNING_LR = cfg.TRANSITION_PLANNING_LR
LAMBDA_PRIOR = cfg.TRANSITION_LAMBDA_PRIOR
CBF_LR = cfg.CBF_LR
CBF_BATCH_SIZE = cfg.CBF_BATCH_SIZE
CBF_EPOCHS = 5000
LAMBDA_SAFE = cfg.LAMBDA_SAFE
LAMBDA_UNSAFE = cfg.LAMBDA_UNSAFE
LAMBDA_DECREASE = cfg.LAMBDA_DECREASE
CBF_ALPHA = cfg.CBF_ALPHA
CBF_DELTA_T = cfg.CBF_DELTA_T
SAFETY_MARGIN = cfg.SAFETY_MARGIN
SUCCESS_THRESHOLD = cfg.SUCCESS_THRESHOLD
SEED = cfg.SEED
