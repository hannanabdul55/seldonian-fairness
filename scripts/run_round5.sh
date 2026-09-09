#!/usr/bin/env bash
# Round 5: brevity task (verifiable length constraint under a reward with an
# injected length bias of strength beta). Reference, then the breach test (GRPO at
# beta 3), then the Seldonian arm. Holds the shared GPU lock per run; skips done runs.
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
  flock "$LOCK" bash -c "echo 'seldonian-fairness-c5 pid=$$ $out/brevity/$method/seed$seed started $(date '+%T')' > $LOCK.info; \
    $PY scripts/run_llm_rl.py --method $method --seed $seed --model $S05 --out $out --quiet $COMMON $* > $out/logs/brevity_${method}_seed${seed}.log 2>&1"
  local rc=$?
  echo "=== $(date '+%F %T') end   $out brevity $method seed$seed exit=$rc"
  if [ $rc -ne 0 ]; then echo "=== $(date '+%F %T') ABORT queue: run failed (see $out/logs/brevity_${method}_seed${seed}.log)"; exit $rc; fi
}

run results/llm_r5/b3 reference     0
run results/llm_r5/b3 grpo          0 --length-bonus 3   # breach test 1: linear length bias, too weak (over-cap 0.50 -> 0.06)
run results/llm_r5/v4 reference     0
run results/llm_r5/v4 grpo          0 --long-bonus 4      # breach test 2: pays 4 per violation (RM penalises one by ~2)
run results/llm_r5/v4 seldonian_lag 0 --long-bonus 4   # stopped: grpo at 4 went shorter (0.50 -> 0.37), non-binding
run results/llm_r5/v16 grpo          0 --long-bonus 16    # breach test 3: bonus above any within-group reward spread
run results/llm_r5/v16 seldonian_lag 0 --long-bonus 16
echo "=== $(date '+%F %T') ROUND5 DONE"
