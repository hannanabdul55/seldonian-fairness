#!/bin/bash
# stage C body: the trained adapter (and, if asked, the base) through the harness on the D_s user tasks.
#   ./stage_C.sh <base-model> <adapter-dir> <tag>
MODEL=$1; ADAPTER=$2; TAG=$3
LOGDIR=/mnt/d/seldonian-runs/021/runs_$TAG
mkdir -p "$LOGDIR"
PID=$(./serve.sh "$MODEL" --enable-lora --max-lora-rank 32 --lora-modules "trained=$ADAPTER") || exit 1
echo "vllm pid $PID up $(date -Is)"
t0=$(date +%s)
SPLIT_KEY=D_s ./bench.sh trained "$LOGDIR" --attack important_instructions
SPLIT_KEY=D_s ./bench.sh trained "$LOGDIR"
echo "bench took $(( $(date +%s) - t0 )) s"
kill -INT "$PID" 2>/dev/null; sleep 5
PIPE=$(ls "$LOGDIR" | grep -v "\.log$" | head -1)
/mnt/d/seldonian-runs/020/vllm-venv/bin/python stageA.py --logdir "$LOGDIR" --pipeline "$PIPE" --tag "$TAG" | tee "stageC_$TAG.md"
echo "DONE stage C $TAG"
