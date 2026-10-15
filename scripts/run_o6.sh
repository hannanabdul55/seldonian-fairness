#!/bin/bash
# Holds the shared GPU lock for paper item O6 (gpu-lock convention).
#   scripts/run_o6.sh train --pilot      3 steps: seconds a step and peak GPU memory (cap 30 min)
#   scripts/run_o6.sh train [--restart]  200 steps, adapters at 100 and 200 (cap 4 h)
#   scripts/run_o6.sh generate           8 reference + 16 x step 100 + 16 x step 200 a prompt (cap 1.5 h)
#   scripts/run_o6.sh judge              Qwen3Guard-4B on every response (cap 1 h)
#   scripts/run_o6.sh all                train (TRAIN_CAP, default 6 h), generate (1.5 h), judge (1 h) under one hold of the lock
# CAP (seconds) overrides a single stage's cap. Training saves the step-100 adapter the moment it reaches
# it, so a cut at the cap leaves step 100; generate and judge append and skip what is done, so a cut
# stage resumes. Big outputs go to /mnt/d/seldonian-runs/o6/, small ones to results/paper/o6/<stage>.log.
cd "$(dirname "$0")/.."
STAGE=$1; shift
case "$STAGE" in
  train) if [[ " $* " == *" --pilot "* ]]; then CAP=${CAP:-1800}; else CAP=${CAP:-14400}; fi;;
  generate) CAP=${CAP:-5400};;
  judge) CAP=${CAP:-3600};;
  all) [ $# -eq 0 ] || { echo "all takes no arguments" >&2; exit 2; };;
  *) echo "stage: train [--pilot|--restart], generate, judge or all" >&2; exit 2;;
esac
root_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
if [ "$root_gb" -lt 4 ]; then echo "not starting: ${root_gb} GB free on /" >&2; exit 1; fi
[ -f results/paper/o6/pools.json ] && [ -f results/paper/o6/train.json ] || { echo "run the pools stage first" >&2; exit 1; }
mkdir -p /mnt/d/seldonian-runs/o6 || { echo "cannot write /mnt/d/seldonian-runs/o6" >&2; exit 1; }
export STAGE CAP HF_HOME=${HF_HOME:-/mnt/d/hf-cache} PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
exec flock /tmp/claude-gpu.lock bash -c '
  trap ": > /tmp/claude-gpu.lock.info" EXIT
  run() {                                         # stage cap [args]
    local st=$1 cap=$2; shift 2
    local gb; gb=$(df -BG --output=avail / | tail -1 | tr -dc "0-9")
    if [ "$gb" -lt 4 ]; then echo "not starting $st: ${gb} GB free on /" >&2; return 1; fi
    local tag=$st; [[ " $* " == *" --pilot "* ]] && tag=pilot
    echo "owner=seldonian-fairness O6 $tag pid=$$ start=$(date -Is) cap=${cap}s" > /tmp/claude-gpu.lock.info
    timeout --signal=INT --kill-after=120 "$cap" .venv/bin/python scripts/o6_moved.py --stage "$st" "$@" 2>&1 \
      | tee -a "/mnt/d/seldonian-runs/o6/$tag.log"
    local rc=${PIPESTATUS[0]}
    [ "$rc" -eq 124 ] && echo "$st stopped at the CAP of $cap s" >&2
    cp "/mnt/d/seldonian-runs/o6/$tag.log" "results/paper/o6/$tag.log" 2>/dev/null
    return $rc
  }
  if [ "$STAGE" = all ]; then
    run train "${TRAIN_CAP:-21600}"; rc=$?
    if [ "$rc" -ne 0 ] && [ "$rc" -ne 124 ]; then exit $rc; fi   # a cut at the cap still leaves step 100
    run generate 5400 || exit $?
    run judge 3600; exit $?
  fi
  run "$STAGE" "$CAP" "$@"' _ "$@"
