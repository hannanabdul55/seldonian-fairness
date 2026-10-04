#!/bin/bash
# Run AgentDojo against the local server (native tool calls through the OpenAI-compatible provider):
# one process per (suite, half of the user tasks), 8 in parallel.
# SPLIT_KEY=D_s restricts to the safety-split user tasks (results/spikes/021/split.json).
#   ./bench.sh <model-id> <logdir> [--attack important_instructions] [-ut ...]   # extra args go to the benchmark
MODEL_ID=$1; LOGDIR=$2; shift 2
VENV=/mnt/d/seldonian-runs/020/vllm-venv
export LOCAL_LLM_PORT=8000 OPENAI_COMPATIBLE_BASE_URL=http://localhost:8000/v1 OPENAI_COMPATIBLE_API_KEY=EMPTY
export PYTHONPATH=/home/hannanabdul/seldonian-fairness/.planning/spikes/021-agentdojo-seldonian-training:$PYTHONPATH
ATTACK_TAG=none; for x in "$@"; do [ "$x" = "--attack" ] && ATTACK_TAG=attacked; done
cd /mnt/d/seldonian-runs/020/agentdojo
pids=()
for suite in banking slack travel workspace; do
  tasks=$("$VENV/bin/python" - "$suite" "${SPLIT_KEY:-}" <<'PY'
import json, os, sys
from agentdojo.task_suite.load_suites import get_suite
s = get_suite("v1.2.2", sys.argv[1]); ids = sorted(s.user_tasks)
if sys.argv[2]:   # restrict to one side of the user-task split (stage C: D_s)
    keep = set(json.load(open("/home/hannanabdul/seldonian-fairness/results/spikes/021/split.json"))[sys.argv[2]])
    ids = [i for i in ids if f"{s.name}/{i}" in keep]
print(" ".join(ids))
PY
)
  read -ra arr <<< "$tasks"; n=${#arr[@]}; half=$(( (n + 1) / 2 ))
  for part in 0 1; do
    if [ $part -eq 0 ]; then sub=("${arr[@]:0:$half}"); else sub=("${arr[@]:$half}"); fi
    [ ${#sub[@]} -eq 0 ] && continue
    ut=(); for t in "${sub[@]}"; do ut+=(-ut "$t"); done
    "$VENV/bin/python" -m agentdojo.scripts.benchmark --model OPENAI_COMPATIBLE --model-id "$MODEL_ID" --benchmark-version v1.2.2 \
      --logdir "$LOGDIR" -s "$suite" "${ut[@]}" -ml patch021 "$@" > "$LOGDIR/bench_${ATTACK_TAG}_${suite}_${part}.log" 2>&1 &
    pids+=($!)
  done
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "bench done rc=$rc"; exit $rc
