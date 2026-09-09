#!/usr/bin/env bash
# Round 6 GPU queue (reports/llm_round6_plan.md), Stages B1 and C. Uses the Stage A
# setting: lam0 5, lam_floor 5, eta_down = eta. Clopper-Pearson everywhere.
# Holds the shared GPU lock per run; skips finished runs; safe to relaunch.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
LOCK=/tmp/claude-gpu.lock
S05=Qwen/Qwen2.5-0.5B-Instruct
DUAL="--lam0 5 --lam-floor 5 --lam-max 50 --eta 100"
BREV="--task brevity --n 3000 --steps 150 --group-size 4 --predict-every 30 --predict-n 768 --steps-per-generation 4 --bound clopper_pearson --word-cap 120 --long-margin 0.05"
AB="--task ab --steps 150 --group-size 4 --predict-every 30 --predict-n 768 --steps-per-generation 4 --bound clopper_pearson --harm-margin 0.03 --harm-group adversarial"

run() {  # run <out> <task-args-var> <method> <seed> [extra args...]
  local out=$1 targs=$2 method=$3 seed=$4; shift 4
  local task; task=$(echo "$targs" | sed -n 's/.*--task \([a-z0-9]*\).*/\1/p')
  local dir="$out/$task/$method/seed$seed"
  if [ -f "$dir/result.json" ]; then echo "skip $out $task $method seed$seed"; return; fi
  mkdir -p "$out/logs"
  echo "=== $(date '+%F %T') start $out $task $method seed$seed $*"
  flock "$LOCK" bash -c "echo 'seldonian-fairness-c6 pid=$$ $out/$task/$method/seed$seed started $(date '+%T')' > $LOCK.info; \
    $PY scripts/run_llm_rl.py --method $method --seed $seed --model $S05 --out $out --quiet $targs $* > $out/logs/${task}_${method}_seed${seed}.log 2>&1"
  local rc=$?
  echo "=== $(date '+%F %T') end   $out $task $method seed$seed exit=$rc"
  if [ $rc -ne 0 ]; then echo "=== $(date '+%F %T') ABORT queue: run failed (see $out/logs/${task}_${method}_seed${seed}.log)"; exit $rc; fi
}

# B1: does the multiplier floor stop the drift-back on brevity? (reuse Round 5 thresholds)
for b in 8 16; do
  mkdir -p results/llm_r6/b1_v$b/brevity
  cp --update=none results/llm_r5/v16/brevity/reference_rates_seed0.json results/llm_r6/b1_v$b/brevity/ 2>/dev/null || true
  run results/llm_r6/b1_v$b "$BREV" seldonian_lag 0 --long-bonus $b $DUAL
done
# C: over-refusal at 0.5B, constraint opposes the reward model; seed 0 first, then 1-2
for seed in 0 1 2; do
  run results/llm_r6/c "$AB" reference     $seed
  run results/llm_r6/c "$AB" grpo          $seed
  run results/llm_r6/c "$AB" composite     $seed --lam 1
  run results/llm_r6/c "$AB" composite     $seed --lam 4
  run results/llm_r6/c "$AB" seldonian_lag $seed $DUAL
done
echo "=== $(date '+%F %T') ROUND6 DONE"
