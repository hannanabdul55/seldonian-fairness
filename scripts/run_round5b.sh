#!/usr/bin/env bash
# Round 5b: does the Seldonian layer work on the brevity task? Three questions,
# in order of information value:
#   1. repeatability: seeds 1 and 2 at bonus 16 (reference, grpo, seldonian_lag)
#   2. attribution: composite arms at seed 0, bonus 16, fixed penalty 16 and 32
#   3. the marginal regime: bonus 8 at seed 0 (grpo, composite, seldonian_lag),
#      where candidates land near the threshold and the safety test has to decide
# Same settings as run_round5.sh. Holds the shared GPU lock per run; skips done runs.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
LOCK=/tmp/claude-gpu.lock
S05=Qwen/Qwen2.5-0.5B-Instruct
COMMON="--task brevity --n 3000 --steps 150 --group-size 4 --predict-every 30 --predict-n 768 --lam-max 50 --steps-per-generation 4 --bound clopper_pearson --word-cap 120 --long-margin 0.05"

run() {  # run <out> <method> <seed> [extra args...]
  local out=$1 method=$2 seed=$3; shift 3
  local dir="$out/brevity/$method/seed$seed"
  if [ -f "$dir/result.json" ]; then echo "skip $out $method seed$seed"; return; fi
  mkdir -p "$out/logs"
  echo "=== $(date '+%F %T') start $out brevity $method seed$seed $*"
  flock "$LOCK" bash -c "echo 'seldonian-fairness-c5b pid=$$ $out/brevity/$method/seed$seed started $(date '+%T')' > $LOCK.info; \
    $PY scripts/run_llm_rl.py --method $method --seed $seed --model $S05 --out $out --quiet $COMMON $* > $out/logs/brevity_${method}_seed${seed}.log 2>&1"
  local rc=$?
  echo "=== $(date '+%F %T') end   $out brevity $method seed$seed exit=$rc"
  if [ $rc -ne 0 ]; then echo "=== $(date '+%F %T') ABORT queue: run failed (see $out/logs/brevity_${method}_seed${seed}.log)"; exit $rc; fi
}

# 1. repeatability at bonus 16
for seed in 1 2; do
  run results/llm_r5/v16 reference     $seed
  run results/llm_r5/v16 grpo          $seed --long-bonus 16
  run results/llm_r5/v16 seldonian_lag $seed --long-bonus 16
done
# 2. attribution: fixed-penalty composite at seed 0 (penalty 16 cancels the bonus
#    exactly; 32 dominates it). Separate out dirs so both keep the seed-0 thresholds.
for lam in 16 32; do
  mkdir -p results/llm_r5/v16_comp$lam/brevity
  cp -n results/llm_r5/v16/brevity/reference_rates_seed0.json results/llm_r5/v16_comp$lam/brevity/
  run results/llm_r5/v16_comp$lam composite 0 --long-bonus 16 --lam $lam
done
# 3. marginal regime: bonus 8 at seed 0
mkdir -p results/llm_r5/v8/brevity
cp -n results/llm_r5/v16/brevity/reference_rates_seed0.json results/llm_r5/v8/brevity/
run results/llm_r5/v8 grpo          0 --long-bonus 8
run results/llm_r5/v8 seldonian_lag 0 --long-bonus 8
run results/llm_r5/v8 composite     0 --long-bonus 8 --lam 16
echo "=== $(date '+%F %T') ROUND5B DONE"
