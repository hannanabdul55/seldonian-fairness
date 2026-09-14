#!/usr/bin/env bash
# Round 6 follow-up: the DiscrimEval Seldonian arm with the betting-mixture bound on the
# probability feature's parity constraint (Bentkus was 0.087 wide against a 0.05
# threshold and could never pass). Runs after run_round6b.sh; skips finished runs.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
LOCK=/tmp/claude-gpu.lock
S05=Qwen/Qwen2.5-0.5B-Instruct
DUAL="--lam0 5 --lam-floor 5 --lam-max 50 --eta 100"
DISC="--task discrim --n 3000 --steps 150 --group-size 4 --predict-every 30 --predict-n 768 --steps-per-generation 4 --bound clopper_pearson --decision-feature prob --attribute race --groups white Black"
run() {
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
out=results/llm_r6/d8e; mkdir -p $out/discrim
cp --update=none results/llm_r6/d/discrim/reference_rates_seed0.json $out/discrim/ 2>/dev/null || true
cp -rn results/llm_r6/d/discrim/reference $out/discrim/ 2>/dev/null || true
run $out "$DISC" seldonian_lag 0 --bias-bonus 8 --bias-mode differential $DUAL
echo "=== $(date '+%F %T') ROUND6C DONE"
