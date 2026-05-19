#!/bin/bash
# =============================================================================
# Fixed-Obstacle Classifier Ablation: λ_prior × λ_collision grid sweep
#
# 5 × 7 = 35 combinations, 500 problems each
# =============================================================================

set -e

NUM_PROBLEMS=500
RESULTS_DIR="../classifier_fixed_ablation_results"
mkdir -p "$RESULTS_DIR"

LAMBDA_PRIORS=(0.001 0.005 0.01 0.05 0.1)
LAMBDA_COLLISIONS=(0.1 0.25 0.5 1.0 2.0 5.0 10.0)

echo "============================================================"
echo "Classifier Fixed-Obstacle Ablation"
echo "  λ_prior:     ${LAMBDA_PRIORS[*]}"
echo "  λ_collision:  ${LAMBDA_COLLISIONS[*]}"
echo "  Problems per run: $NUM_PROBLEMS"
echo "  Total runs: $(( ${#LAMBDA_PRIORS[@]} * ${#LAMBDA_COLLISIONS[@]} ))"
echo "  Results dir: $RESULTS_DIR"
echo "============================================================"
echo ""

SUMMARY_FILE="$RESULTS_DIR/ablation_summary.csv"
echo "lambda_prior,lambda_collision,success_rate,goal_rate,collision_free_rate" > "$SUMMARY_FILE"

RUN=0
TOTAL=$(( ${#LAMBDA_PRIORS[@]} * ${#LAMBDA_COLLISIONS[@]} ))

for LP in "${LAMBDA_PRIORS[@]}"; do
    for LC in "${LAMBDA_COLLISIONS[@]}"; do
        RUN=$((RUN + 1))
        echo "---------- Run $RUN/$TOTAL: λ_prior=$LP, λ_collision=$LC ----------"

        LOG_FILE="$RESULTS_DIR/lp_${LP}_lc_${LC}.log"

        python classifier_fixed_evaluate.py \
            --num_problems "$NUM_PROBLEMS" \
            --lambda_prior "$LP" \
            --lambda_collision "$LC" \
            2>&1 | tee "$LOG_FILE"

        # Extract rates
        SUCCESS=$(grep -i "Success Rate:" "$LOG_FILE" | tail -1 | grep -oP '[\d.]+%' | head -1 | tr -d '%')
        GOAL=$(grep -i "Goal Reached:" "$LOG_FILE" | tail -1 | grep -oP '[\d.]+%' | head -1 | tr -d '%')
        CF=$(grep -i "Collision-free:" "$LOG_FILE" | tail -1 | grep -oP '[\d.]+%' | head -1 | tr -d '%')

        echo "$LP,$LC,$SUCCESS,$GOAL,$CF" >> "$SUMMARY_FILE"
        echo "  → Success=$SUCCESS% Goal=$GOAL% CF=$CF%"
        echo ""
    done
done

echo "============================================================"
echo "Ablation complete! Summary saved to: $SUMMARY_FILE"
echo "============================================================"
echo ""
echo "Results (sorted by success rate):"
head -1 "$SUMMARY_FILE"
tail -n +2 "$SUMMARY_FILE" | sort -t',' -k3 -rn
