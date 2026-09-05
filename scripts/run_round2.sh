#!/usr/bin/env bash
# Round 1b + Round 2 core queue, in priority order. Each run holds the shared GPU lock
# (/tmp/claude-gpu.lock) and releases it between runs so other sessions can interleave.
# Skips runs that already have result.json; safe to re-launch.
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
  flock "$LOCK" bash -c "echo 'seldonian-fairness-c3 pid=$$ $task/$method/seed$seed started $(date '+%T')' > $LOCK.info; \
    $PY scripts/run_llm_rl.py --task $task --method $method --seed $seed --model $model \
      --steps-per-generation 4 --out $out --quiet $* > $out/logs/${task}_${method}_seed${seed}.log 2>&1"
  local rc=$?
  echo "=== $(date '+%F %T') end   $out $task $method seed$seed exit=$rc"
  if [ $rc -ne 0 ]; then echo "=== $(date '+%F %T') ABORT queue: run failed (see $out/logs/${task}_${method}_seed${seed}.log)"; exit $rc; fi
}

# Round 1b: fixes at 0.5B, Lagrangian only, 3 seeds (gate G3 re-test)
for seed in 0 1 2; do run results/llm_r1b ab seldonian_lag $seed $S05 --harm-margin 0.03; done
# Round 2 core: 1.5B, seed 0
run results/llm_r2 ab reference     0 $S15 --harm-margin 0.03
run results/llm_r2 ab seldonian_lag 0 $S15 --harm-margin 0.03
run results/llm_r2 ab grpo          0 $S15 --harm-margin 0.03
# Task D control at 0.5B (verifiable reward, no-regression floor)
for method in reference grpo seldonian_lag; do
  run results/llm_r2 gsm8k $method 0 $S05 --acc-margin 0.05
done
# Round 2 seeds 1-2 at 1.5B, if time allows
for seed in 1 2; do
  run results/llm_r2 ab reference     $seed $S15 --harm-margin 0.03
  run results/llm_r2 ab seldonian_lag $seed $S15 --harm-margin 0.03
  run results/llm_r2 ab grpo          $seed $S15 --harm-margin 0.03
done
echo "=== $(date '+%F %T') ROUND2 DONE"
