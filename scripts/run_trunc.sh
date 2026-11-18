#!/bin/bash
# Holds the shared GPU lock for paper item Q1 (gpu-lock convention): the compiled judge on first-sentence cuts of
# four 500-item sets, six wordings (12,000 forward passes, about 20 minutes), cap 1 h. Appends and skips what is
# done, so a cut run resumes. Log in results/paper/trunc/score.log.
cd "$(dirname "$0")/.."
CAP=${CAP:-3600}
root_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
if [ "$root_gb" -lt 4 ]; then echo "not starting: ${root_gb} GB free on /" >&2; exit 1; fi
mkdir -p results/paper/trunc
export CAP HF_HOME=${HF_HOME:-/mnt/d/hf-cache}
exec flock /tmp/claude-gpu.lock bash -c '
  trap ": > /tmp/claude-gpu.lock.info" EXIT
  echo "owner=seldonian-fairness Q1 truncation score pid=$$ start=$(date -Is) cap=${CAP}s" > /tmp/claude-gpu.lock.info
  timeout --signal=INT --kill-after=120 "$CAP" .venv/bin/python scripts/trunc_judge.py score 2>&1 \
    | grep --line-buffered -v "Loading weights" | tee -a results/paper/trunc/score.log
  rc=${PIPESTATUS[0]}
  [ "$rc" -eq 124 ] && echo "score stopped at the CAP of $CAP s" >&2
  exit $rc'
