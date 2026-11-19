#!/bin/bash
# Holds the shared GPU lock for paper item Q2 (gpu-lock convention): one new rubric compiled and scored on the full responses of
# four 500-item sets, (2,000 forward passes, about 10 minutes), cap 40 min. Appends and skips what is
# done, so a cut run resumes. Log in results/paper/rubric7/score.log.
cd "$(dirname "$0")/.."
CAP=${CAP:-2400}
root_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
if [ "$root_gb" -lt 4 ]; then echo "not starting: ${root_gb} GB free on /" >&2; exit 1; fi
mkdir -p results/paper/rubric7
export CAP HF_HOME=${HF_HOME:-/mnt/d/hf-cache}
exec flock /tmp/claude-gpu.lock bash -c '
  trap ": > /tmp/claude-gpu.lock.info" EXIT
  echo "owner=seldonian-fairness Q2 rubric7 score pid=$$ start=$(date -Is) cap=${CAP}s" > /tmp/claude-gpu.lock.info
  timeout --signal=INT --kill-after=120 "$CAP" .venv/bin/python scripts/rubric7_judge.py score 2>&1 \
    | grep --line-buffered -v "Loading weights" | tee -a results/paper/rubric7/score.log
  rc=${PIPESTATUS[0]}
  [ "$rc" -eq 124 ] && echo "score stopped at the CAP of $CAP s" >&2
  exit $rc'
