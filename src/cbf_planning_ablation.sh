#!/bin/bash
# =============================================================================
# CBF Evaluate Planning Alpha Ablation Study
#
# Sweeps cbf_alpha from 0.0 to 1.0 in steps of 0.05 (21 values)
# Uses cbf_evaluate_planning.py (random obstacles, random goals)
# =============================================================================

set -e

NUM_PROBLEMS=500
RESULTS_DIR="../cbf_planning_ablation_results"
mkdir -p "$RESULTS_DIR"

ALPHAS=(0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00)

echo "============================================================"
echo "CBF Evaluate Planning Alpha Ablation"
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

    python cbf_evaluate_planning.py \
        --num_problems "$NUM_PROBLEMS" \
        --cbf_alpha "$ALPHA" \
        2>&1 | tee "$LOG_FILE"

    # Extract rates from the log
    SUCCESS=$(grep -i "Success Rate:" "$LOG_FILE" | tail -1 | grep -oP '[\d.]+%' | head -1 | tr -d '%')
    GOAL=$(grep -i "Goal Reached:" "$LOG_FILE" | tail -1 | grep -oP '[\d.]+%' | head -1 | tr -d '%')
    CF=$(grep -i "Collision-free:" "$LOG_FILE" | tail -1 | grep -oP '[\d.]+%' | head -1 | tr -d '%')

    echo "$ALPHA,$SUCCESS,$GOAL,$CF" >> "$SUMMARY_FILE"
    echo "  → Success=$SUCCESS% Goal=$GOAL% CF=$CF%"
    echo ""
done

echo "============================================================"
echo "Ablation complete! Summary saved to: $SUMMARY_FILE"
echo "============================================================"
cat "$SUMMARY_FILE"
