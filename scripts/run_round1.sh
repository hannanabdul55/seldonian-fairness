#!/usr/bin/env bash
# Round 1 pilot: Tasks A+B on Qwen2.5-0.5B-Instruct, 4 methods x 3 seeds, sequential.
# Skips any run whose result.json already exists, so it is safe to re-launch.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
OUT=${OUT:-results/llm}
STEPS=${STEPS:-200}
for seed in 0 1 2; do
  for method in reference grpo composite seldonian; do
    dir="$OUT/ab/$method/seed$seed"
    if [ -f "$dir/result.json" ]; then echo "skip $method seed$seed"; continue; fi
    echo "=== $(date '+%F %T') start $method seed$seed"
    $PY scripts/run_llm_rl.py --task ab --method "$method" --seed "$seed" --steps "$STEPS" \
        --out "$OUT" --quiet > "$OUT/logs/${method}_seed${seed}.log" 2>&1
    echo "=== $(date '+%F %T') end   $method seed$seed exit=$?"
  done
done
echo "=== $(date '+%F %T') ROUND1 DONE"
