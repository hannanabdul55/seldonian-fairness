#!/usr/bin/env bash
# Round 4: reward-pressure task at 0.5B. The base reward gets a compliance bonus
# alpha*(1 - refusal) on adversarial prompts so the harm constraint binds by
# construction; alpha is the pressure knob. Thresholds are the Round 1c seed-0
# reference rates (copied into each out dir) so every arm shares them.
# Holds the shared GPU lock per run; skips runs that already have result.json.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
LOCK=/tmp/claude-gpu.lock
S05=Qwen/Qwen2.5-0.5B-Instruct
COMMON="--steps 150 --group-size 4 --predict-every 30 --predict-n 768 --harm-margin 0.03 --lam-max 50 --steps-per-generation 4"

run() {  # run <out> <task> <method> <seed> <model> [extra args...]
  local out=$1 task=$2 method=$3 seed=$4 model=$5; shift 5
  local dir="$out/$task/$method/seed$seed"
  if [ -f "$dir/result.json" ]; then echo "skip $out $task $method seed$seed"; return; fi
  mkdir -p "$out/logs"
  echo "=== $(date '+%F %T') start $out $task $method seed$seed ($model) $*"
  flock "$LOCK" bash -c "echo 'seldonian-fairness-c4 pid=$$ $out/$task/$method/seed$seed started $(date '+%T')' > $LOCK.info; \
    $PY scripts/run_llm_rl.py --task $task --method $method --seed $seed --model $model \
      --out $out --quiet $COMMON $* > $out/logs/${task}_${method}_seed${seed}.log 2>&1"
  local rc=$?
  echo "=== $(date '+%F %T') end   $out $task $method seed$seed exit=$rc"
  if [ $rc -ne 0 ]; then echo "=== $(date '+%F %T') ABORT queue: run failed (see $out/logs/${task}_${method}_seed${seed}.log)"; exit $rc; fi
}

run results/llm_r4/a5   ab reference     0 $S05
run results/llm_r4/a5   ab grpo          0 $S05 --compliance-bonus 5
run results/llm_r4/a5   ab seldonian_lag 0 $S05 --compliance-bonus 5
run results/llm_r4/a2.5 ab grpo          0 $S05 --compliance-bonus 2.5
run results/llm_r4/a2.5 ab seldonian_lag 0 $S05 --compliance-bonus 2.5
echo "=== $(date '+%F %T') ROUND4 DONE"
