"""
Fixed-Obstacle CBF Configuration.

Debugging experiment: ONE fixed obstacle, ONE fixed goal.
Separate data/model paths so existing data is untouched.
"""

import os
import cbf_config as cfg

# =============================================================================
# Fixed Scenario Definition
# =============================================================================
# Obstacle at center-front of workspace [x, y, h, r]
FIXED_OBSTACLE = [0.5, 0.0, 0.75, 0.1]

# Fixed goal: a target joint configuration in the middle of the workspace.
# These are joint angles in radians for the Panda robot.
# This config places the end-effector behind the obstacle.
FIXED_GOAL_Q_RAD = [0.5, -0.5, 0.0, -2.0, 0.0, 2.0, 0.8]

# =============================================================================
# Paths (separate from main CBF data to avoid overwriting)
# =============================================================================
FIXED_DATA_DIR = os.path.join(cfg.PROJECT_ROOT, 'cbf_fixed_data')
FIXED_STATE_LABELS_TRAIN = os.path.join(FIXED_DATA_DIR, 'state_labels_train.pt')
FIXED_STATE_LABELS_VAL = os.path.join(FIXED_DATA_DIR, 'state_labels_val.pt')
FIXED_TRANSITIONS_TRAIN = os.path.join(FIXED_DATA_DIR, 'transitions_train.pt')
FIXED_TRANSITIONS_VAL = os.path.join(FIXED_DATA_DIR, 'transitions_val.pt')

FIXED_SNAPSHOT_DIR = os.path.join(cfg.PROJECT_ROOT, 'model_params/panda_10k/cbf_fixed_snapshots')
FIXED_BEST_CHECKPOINT = os.path.join(FIXED_SNAPSHOT_DIR, 'barrier_net_best.pt')
FIXED_TENSORBOARD_DIR = os.path.join(cfg.PROJECT_ROOT, 'model_params/panda_10k/runs_cbf_fixed')

# =============================================================================
# Training Hyperparameters (inherit from main config, override if needed)
# =============================================================================
NUM_SCENARIOS = 5000       # fewer scenarios needed for single obstacle
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
SEED = cfg.SEED

# BarrierNet architecture (obs_dim removed)
LATENT_DIM = cfg.LATENT_DIM
CBF_HIDDEN_UNITS = cfg.CBF_HIDDEN_UNITS
CBF_NUM_HIDDEN = cfg.CBF_NUM_HIDDEN
SUCCESS_THRESHOLD = cfg.SUCCESS_THRESHOLD
