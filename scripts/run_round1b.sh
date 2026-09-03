#!/usr/bin/env bash
# Round 1 addendum: Lagrangian candidate selection (seldonian_lag) for seeds 0-2.
# Waits for run_round1.sh to finish so the GPU is never shared. Safe to re-launch.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
OUT=${OUT:-results/llm}
STEPS=${STEPS:-200}
while pgrep -f scripts/run_round1.sh > /dev/null; do sleep 30; done
for seed in 0 1 2; do
  method=seldonian_lag
  dir="$OUT/ab/$method/seed$seed"
  if [ -f "$dir/result.json" ]; then echo "skip $method seed$seed"; continue; fi
  echo "=== $(date '+%F %T') start $method seed$seed"
  $PY scripts/run_llm_rl.py --task ab --method "$method" --seed "$seed" --steps "$STEPS" \
      --lam0 2 --eta 20 --predict-every 20 --out "$OUT" --quiet \
      > "$OUT/logs/${method}_seed${seed}.log" 2>&1
  echo "=== $(date '+%F %T') end   $method seed$seed exit=$?"
done
echo "=== $(date '+%F %T') ROUND1B DONE"
