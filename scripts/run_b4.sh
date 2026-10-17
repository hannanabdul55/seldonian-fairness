#!/bin/bash
# Holds the shared GPU lock for paper item B4 (gpu-lock convention): WildGuard scores on spike 017's refusal sets
# and the refusal sheet (4,940 items, about 20 minutes), cap 1.5 h. Appends and skips what is done, so a cut run
# resumes. Log in results/paper/b4/score.log.
cd "$(dirname "$0")/.."
CAP=${CAP:-5400}
root_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
if [ "$root_gb" -lt 4 ]; then echo "not starting: ${root_gb} GB free on /" >&2; exit 1; fi
mkdir -p results/paper/b4
export CAP HF_HOME=${HF_HOME:-/mnt/d/hf-cache}
exec flock /tmp/claude-gpu.lock bash -c '
  trap ": > /tmp/claude-gpu.lock.info" EXIT
  echo "owner=seldonian-fairness B4 score pid=$$ start=$(date -Is) cap=${CAP}s" > /tmp/claude-gpu.lock.info
  timeout --signal=INT --kill-after=120 "$CAP" .venv/bin/python scripts/b4_wildguard.py score 2>&1 \
    | grep --line-buffered -v "Loading weights" | tee -a results/paper/b4/score.log
  rc=${PIPESTATUS[0]}
  [ "$rc" -eq 124 ] && echo "score stopped at the CAP of $CAP s" >&2
  exit $rc'
