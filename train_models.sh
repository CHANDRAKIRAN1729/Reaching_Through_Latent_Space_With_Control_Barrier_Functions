#!/bin/bash
# Script to train VAE and VAE Obstacle Classifier models
# Usage: ./train_models.sh

set -e  # Exit on error

echo "==================================="
echo "RTLS Model Training Script"
echo "==================================="
echo ""

# Navigate to source directory
cd "$(dirname "$0")/src"

echo "Step 1: Training VAE model..."
echo "-----------------------------------"
python train_vae.py --c ../config/vae_config/panda_10k.yaml

echo ""
echo "Step 1 completed successfully!"
echo ""
echo "Checking for VAE checkpoint files..."
MODEL_DIR="../model_params/panda_10k"

# Find the latest checkpoint
LATEST_CKPT=$(ls -t ${MODEL_DIR}/snapshots/model.ckpt-*.pt 2>/dev/null | head -1)
LATEST_RUNCMD=$(ls -t ${MODEL_DIR}/*-runcmd.json 2>/dev/null | head -1)

if [ -z "$LATEST_CKPT" ] || [ -z "$LATEST_RUNCMD" ]; then
    echo "ERROR: Could not find VAE checkpoint files!"
    exit 1
fi

echo "Found checkpoint: $LATEST_CKPT"
echo "Found runcmd: $LATEST_RUNCMD"

# Update the VAE obstacle config with actual checkpoint paths
echo ""
echo "Updating VAE obstacle classifier config with checkpoint paths..."

# Use sed to update the paths in place
sed -i "s|vae_run_cmd_path.*|vae_run_cmd_path           : '${LATEST_RUNCMD}'|" ../config/vae_obs_config/panda_10k.yaml
sed -i "s|pretrained_checkpoint_path.*|pretrained_checkpoint_path : '${LATEST_CKPT}'|" ../config/vae_obs_config/panda_10k.yaml

echo "Config updated successfully!"
echo ""

echo "Step 2: Training VAE Obstacle Classifier..."
echo "-----------------------------------"
python train_vae_obs.py --c ../config/vae_obs_config/panda_10k.yaml

echo ""
echo "==================================="
echo "All training completed successfully!"
echo "==================================="
echo ""
echo "Model files saved in: ${MODEL_DIR}"
echo ""
