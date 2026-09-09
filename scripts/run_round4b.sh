# Round 4b: alpha 5 did not bind (harm fell to 0.079: the reward model's own
# preference for refusing over a harmful answer is ~6 points, so a bonus of 5 only
# makes *safe* compliance attractive). alpha 10 exceeds that gap.
run results/llm_r4/a10 ab grpo          0 $S05 --compliance-bonus 10
run results/llm_r4/a10 ab seldonian_lag 0 $S05 --compliance-bonus 10
echo "=== $(date '+%F %T') ROUND4B DONE"
