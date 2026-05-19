#!/bin/bash
# =============================================================================
# CBF Fixed-Obstacle Alpha Ablation Study
#
# Sweeps cbf_alpha from 0.0 to 1.0 in steps of 0.05 (21 values)
# All other parameters held constant.
# =============================================================================

set -e

NUM_PROBLEMS=500
RESULTS_DIR="../cbf_fixed_ablation_results"
mkdir -p "$RESULTS_DIR"

ALPHAS=(0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00)

echo "============================================================"
echo "CBF Fixed-Obstacle Alpha Ablation"
echo "  Alpha values: ${ALPHAS[*]}"
echo "  Problems per run: $NUM_PROBLEMS"
echo "  Results dir: $RESULTS_DIR"
echo "============================================================"
echo ""

SUMMARY_FILE="$RESULTS_DIR/alpha_ablation_summary.csv"
echo "alpha,success_rate,goal_rate,collision_free_rate" > "$SUMMARY_FILE"

for ALPHA in "${ALPHAS[@]}"; do
    echo "---------- Running alpha=$ALPHA ----------"

    LOG_FILE="$RESULTS_DIR/alpha_${ALPHA}.log"

    python cbf_fixed_evaluate.py \
        --num_problems "$NUM_PROBLEMS" \
        --cbf_alpha "$ALPHA" \
        2>&1 | tee "$LOG_FILE"

    # Extract rates from the last few lines of output
    SUCCESS=$(grep "Success Rate:" "$LOG_FILE" | tail -1 | grep -oP '[\d.]+%' | tr -d '%')
    GOAL=$(grep "Goal Reached:" "$LOG_FILE" | tail -1 | grep -oP '[\d.]+%' | tr -d '%')
    CF=$(grep "Collision-free:" "$LOG_FILE" | tail -1 | grep -oP '[\d.]+%' | tr -d '%')

    echo "$ALPHA,$SUCCESS,$GOAL,$CF" >> "$SUMMARY_FILE"
    echo "  → Success=$SUCCESS% Goal=$GOAL% CF=$CF%"
    echo ""
done

echo "============================================================"
echo "Ablation complete! Summary saved to: $SUMMARY_FILE"
echo "============================================================"
cat "$SUMMARY_FILE"
