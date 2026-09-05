#!/usr/bin/env bash
# Round 3 queue (replaces the 1.5B seeds 1-2 of the previous queue):
#   1. finish the GSM8K control (seldonian_lag) at 0.5B
#   2. 0.5B Lagrangian Seldonian with the corrected settings (eta 100, predicted bound at
#      the effective size with inflation 1, 1,024 prediction prompts), 3 seeds -> results/llm_r1c
#   3. 1.5B reward-pressure pair (beta 0): grpo, then seldonian_lag -> results/llm_r2p
# Waits on the shared GPU lock, skips finished runs, aborts on failure.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
LOCK=/tmp/claude-gpu.lock
S05=Qwen/Qwen2.5-0.5B-Instruct
S15=Qwen/Qwen2.5-1.5B-Instruct

run() {  # run <out> <task> <method> <seed> <model> [extra args...]
  local out=$1 task=$2 method=$3 seed=$4 model=$5; shift 5
  local dir="$out/$task/$method/seed$seed"
  if [ -f "$dir/result.json" ]; then echo "skip $out $task $method seed$seed"; return; fi
  mkdir -p "$out/logs"
  echo "=== $(date '+%F %T') start $out $task $method seed$seed ($model)"
  flock "$LOCK" bash -c "echo 'seldonian-fairness-c3 pid=$$ $out/$task/$method/seed$seed started $(date '+%T')' > $LOCK.info; \
    $PY scripts/run_llm_rl.py --task $task --method $method --seed $seed --model $model \
      --steps-per-generation 4 --out $out --quiet $* > $out/logs/${task}_${method}_seed${seed}.log 2>&1"
  local rc=$?
  echo "=== $(date '+%F %T') end   $out $task $method seed$seed exit=$rc"
  if [ $rc -ne 0 ]; then echo "=== $(date '+%F %T') ABORT queue: run failed (see $out/logs/${task}_${method}_seed${seed}.log)"; exit $rc; fi
}

run results/llm_r2 gsm8k seldonian_lag 0 $S05 --acc-margin 0.05
for seed in 0 1 2; do run results/llm_r1c ab seldonian_lag $seed $S05 --harm-margin 0.03; done
run results/llm_r2p ab grpo          0 $S15 --harm-margin 0.03 --beta 0.0
run results/llm_r2p ab seldonian_lag 0 $S15 --harm-margin 0.03 --beta 0.0
echo "=== $(date '+%F %T') ROUND3 DONE"
