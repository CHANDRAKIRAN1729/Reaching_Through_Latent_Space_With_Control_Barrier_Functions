"""
CBF Configuration — Centralized paths and hyperparameters.

All frozen baseline assets and tunable CBF parameters live here.
Every CBF script imports from this module.
"""

import os

# =============================================================================
# Project root (relative to src/)
# =============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# Frozen Baseline Assets (DO NOT CHANGE)
# =============================================================================
VAE_CHECKPOINT = os.path.join(PROJECT_ROOT, 'model_params/panda_10k/model.ckpt-016000.pt')
VAE_SNAPSHOT = os.path.join(PROJECT_ROOT, 'model_params/panda_10k/snapshots/model.ckpt-015350.pt')
CLASSIFIER_CHECKPOINT = os.path.join(PROJECT_ROOT, 'model_params/panda_10k/snapshots_obs/model.ckpt-015350-000230.pt')
VAE_CONFIG = os.path.join(PROJECT_ROOT, 'model_params/panda_10k/20260117_225605704-runcmd.json')
CLASSIFIER_CONFIG = os.path.join(PROJECT_ROOT, 'model_params/panda_10k/20260117_231724659-runcmd.json')
TEST_SCENES = os.path.join(PROJECT_ROOT, 'model_params/panda_10k/test_scenes.json')
DATA_PATH = os.path.join(PROJECT_ROOT, 'data')

# =============================================================================
# VAE Architecture (from frozen config — DO NOT CHANGE)
# =============================================================================
INPUT_DIM = 10          # 7 joint angles + 3 EE position
LATENT_DIM = 7          # latent space dimensionality
UNITS_PER_LAYER = 2048  # hidden layer width
NUM_HIDDEN_LAYERS = 4   # number of hidden layers
OBS_DIM = 4             # obstacle: x, y, h, r

# =============================================================================
# CBF Data Paths
# =============================================================================
CBF_DATA_DIR = os.path.join(PROJECT_ROOT, 'cbf_data')
STATE_LABELS_TRAIN = os.path.join(CBF_DATA_DIR, 'state_labels_train.pt')
STATE_LABELS_VAL = os.path.join(CBF_DATA_DIR, 'state_labels_val.pt')
STATE_LABELS_TEST = os.path.join(CBF_DATA_DIR, 'state_labels_test.pt')
TRANSITIONS_TRAIN = os.path.join(CBF_DATA_DIR, 'transitions_train.pt')
TRANSITIONS_VAL = os.path.join(CBF_DATA_DIR, 'transitions_val.pt')

# =============================================================================
# CBF Model Paths
# =============================================================================
CBF_SNAPSHOT_DIR = os.path.join(PROJECT_ROOT, 'model_params/panda_10k/cbf_snapshots')
CBF_BEST_CHECKPOINT = os.path.join(CBF_SNAPSHOT_DIR, 'barrier_net_best.pt')
CBF_TENSORBOARD_DIR = os.path.join(PROJECT_ROOT, 'model_params/panda_10k/runs_cbf')

# =============================================================================
# CBF Network Architecture
# =============================================================================
CBF_HIDDEN_UNITS = 2048     # same width as classifier
CBF_NUM_HIDDEN = 4          # same depth as classifier

# =============================================================================
# CBF Training Hyperparameters
# =============================================================================
CBF_LR = 1e-4               # Adam learning rate
CBF_BATCH_SIZE = 4096        # batch size per DataLoader
CBF_EPOCHS = 5000             # training epochs
LAMBDA_SAFE = 1.0            # weight for safe sign loss
LAMBDA_UNSAFE = 1.0          # weight for unsafe sign loss
LAMBDA_DECREASE = 1.0        # weight for CBF decrease condition
CBF_ALPHA = 1.0              # barrier decay rate — always 1
CBF_DELTA_T = 1.0            # time step
SAFETY_MARGIN = 1.0          # margin γ: forces B ≥ γ for safe, B ≤ -γ for unsafe

# =============================================================================
# Transition Data Generation
# =============================================================================
NUM_TRANSITION_SCENARIOS = 15000  # number of planning scenarios to run (~1M transitions)
TRANSITION_MAX_STEPS = 300        # max optimization steps per scenario
TRANSITION_PLANNING_LR = 0.03    # must match PLANNING_LR (Adam lr)
TRANSITION_LAMBDA_PRIOR = 0.01   # must match LAMBDA_PRIOR

# =============================================================================
# CBF Planning Hyperparameters (inference)
# =============================================================================
PLANNING_LR = 0.03            # nominal planner LR (matches baseline Adam lr)
LAMBDA_PRIOR = 0.01           # prior loss weight (matches baseline default)
MAX_STEPS = 300               # maximum planning steps
SUCCESS_THRESHOLD = 0.01      # 1cm goal-reaching threshold

# =============================================================================
# Reproducibility
# =============================================================================
SEED = 42
