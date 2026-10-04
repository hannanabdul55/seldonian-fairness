#!/bin/bash
# stage A body: serve the model, run the attacked grid and the no-attack grid, analyse, build prefixes.
#   ./stage_A.sh <model> <tag> [extra vllm args]
MODEL=$1; TAG=$2; shift 2
LOGDIR=/mnt/d/seldonian-runs/021/runs_$TAG
VENV=/mnt/d/seldonian-runs/020/vllm-venv
mkdir -p "$LOGDIR"
PID=$(./serve.sh "$MODEL" "$@") || exit 1
echo "vllm pid $PID up $(date -Is)"
t0=$(date +%s)
./bench.sh "$MODEL" "$LOGDIR" --attack important_instructions
./bench.sh "$MODEL" "$LOGDIR"
echo "bench took $(( $(date +%s) - t0 )) s"
PIPE=$(ls "$LOGDIR" | grep -v "\.log$" | head -1)
"$VENV/bin/python" stageA.py --logdir "$LOGDIR" --pipeline "$PIPE" --tag "$TAG" | tee "stageA_$TAG.md"
"$VENV/bin/python" prefixes021.py --logdir "$LOGDIR" --pipeline "$PIPE" --tag "$TAG" --model "$MODEL" | tee -a "stageA_$TAG.md"
kill -INT "$PID" 2>/dev/null; sleep 5
echo "DONE stage A $TAG"
