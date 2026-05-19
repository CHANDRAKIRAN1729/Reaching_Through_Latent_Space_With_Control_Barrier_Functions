#!/bin/bash

# New Evaluation Script — Two-Stage Pipeline (Author-Recommended)
#
# Stage 1: Planning with learned classifier (differentiable, for gradients)
# Stage 2: Validation with Robo3D.check_for_collision (geometric ground truth)
#
# This matches the author's email:
#   "Instantiate a Robo3D with Panda as the definition and
#    use the function check_for_collision."

cd "src"

# Check if validated scenes exist
if [ ! -f "../model_params/panda_10k/test_scenes.json" ]; then
    echo "WARNING: test_scenes.json not found."
    echo "Will generate new random scenarios instead."
    SCENE_ARG=""
else
    SCENE_ARG="--load_scenes ../model_params/panda_10k/test_scenes.json"
fi

echo "============================================"
echo "New Evaluation (Two-Stage Pipeline)"
echo "============================================"
echo ""
echo "Stage 1: Planning with learned classifier"
echo "  -> Differentiable L_collision for gradient descent"
echo "Stage 2: Validation with Robo3D.check_for_collision"
echo "  -> Geometric capsule-based ground truth"
echo ""
echo "Configuration:"
echo "  - 10mm success threshold"
echo "  - Planning LR: 0.1 (higher for goal-reaching)"
echo "  - Lambda prior: 0.07 (calibrated to model)"
echo "  - Lambda collision: 0.11 (calibrated to model)"
echo "  - Max steps: 300"
echo "  - Temperature: 1.0"
echo "  - GECO adaptive weighting ENABLED"
echo "  - tau_prior: 12.0, tau_obs: 7.0 (matched to model loss magnitudes)"
echo ""
echo "============================================"
echo ""

python evaluate_planning.py \
    --checkpoint ../model_params/panda_10k/model.ckpt-016000.pt \
    --checkpoint_obs ../model_params/panda_10k/snapshots_obs/model.ckpt-015350-000230.pt \
    --config ../model_params/panda_10k/20260117_225605704-runcmd.json \
    --config_obs ../model_params/panda_10k/20260117_231724659-runcmd.json \
    --data_path ../data \
    $SCENE_ARG \
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
    --output ../model_params/panda_10k/evaluation_results.json 2>&1 | \
    grep -E '(Using device|Loading|loaded|Starting|Evaluating|GECO|Lambda|Progress:|======|PATH PLANNING|Configuration:|Results:|Success|Goal|Collision|Planning Time|Path Length|Comparison|saved|Robo3D|Stage|two-stage)'

echo ""
echo "============================================"
echo "Evaluation complete!"
echo "Results saved to: model_params/panda_10k/evaluation_results.json"
echo "============================================"