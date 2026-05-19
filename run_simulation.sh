#!/bin/bash

# MoveIt Simulation Script for "Reaching Through Latent Space"
# Uses Phase 2 ablation-optimized evaluation parameters
#
# This script loads pre-generated test scenes and runs simulation
# with the optimal parameters found in Phase 2 ablation (80.3% success).
#
# USAGE:
#   ./run_simulation.sh                          # Load test_scenes.json
#   ./run_simulation.sh --generate_scenes        # Generate new scenes

# Parse command line arguments
GENERATE_SCENES=false
SCENES_FILE="../model_params/panda_10k/test_scenes.json"

while [[ $# -gt 0 ]]; do
    case $1 in
        --generate_scenes)
            GENERATE_SCENES=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--generate_scenes]"
            exit 1
            ;;
    esac
done

# Navigate to source directory
cd "$(dirname "$0")/src"

echo "============================================"
echo "MoveIt Simulation (Phase 2 Optimized)"
echo "============================================"
echo ""
if [ "$GENERATE_SCENES" = true ]; then
    echo "MODE: Generating new validated scenes"
else
    echo "MODE: Loading pre-generated scenes from test_scenes.json"
fi
echo ""
echo "Phase 2 Optimal Parameters:"
echo "  - Planning LR: 0.15"
echo "  - Lambda prior: 0.7"
echo "  - Lambda collision: 0.5"
echo "  - Temperature: 3.0"
echo "  - Alpha GECO: 0.008"
echo "  - Tau prior goal: 6.0"
echo "  - Tau obs goal: 2.0"
echo "  - Alpha MA prior: 0.8"
echo "  - Alpha MA obs: 0.8"
echo ""
echo "Output:"
echo "  - Results: simulation_results_42_1000scenarios.json"
echo "  - Trajectories: trajectories_optimized.json"
echo "============================================"
echo ""

# Build the command
CMD="python -u simulate_in_moveit.py \
    --checkpoint ../model_params/panda_10k/model.ckpt-016000.pt \
    --checkpoint_obs ../model_params/panda_10k/snapshots_obs/model.ckpt-015350-000230.pt \
    --config ../model_params/panda_10k/20260117_225605704-runcmd.json \
    --config_obs ../model_params/panda_10k/20260117_231724659-runcmd.json \
    --data_path ../data \
    --num_scenarios 1000 \
    --num_obstacles 1 \
    --success_threshold 0.01 \
    --max_steps 300 \
    --planning_lr 0.15 \
    --lambda_prior 0.7 \
    --lambda_collision 0.5 \
    --temperature 3.0 \
    --use_geco \
    --alpha_geco 0.008 \
    --tau_prior_goal 6.0 \
    --tau_obs_goal 2.0 \
    --alpha_ma_prior 0.8 \
    --alpha_ma_obs 0.8 \
    --collision_padding 0.001 \
    --interpolation_steps 50 \
    --animation_speed 0.06 \
    --seed 42 \
    --export_trajectory ../model_params/panda_10k/trajectories_optimized.json"

# Add scene loading or saving based on mode
if [ "$GENERATE_SCENES" = true ]; then
    CMD="$CMD --save_scenes $SCENES_FILE"
else
    CMD="$CMD --load_scenes $SCENES_FILE"
fi

# Execute
eval "$CMD" 2>&1

echo ""
echo "============================================"
echo "Simulation complete!"
echo "Results: model_params/panda_10k/simulation_results_42_*scenarios.json"
echo "============================================"
