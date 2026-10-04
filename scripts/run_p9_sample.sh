#!/bin/bash
# Holds the shared GPU lock for P9's sampling (gpu-lock convention). Args go to p9_sample.py.
# CAP (seconds, default 2 h) kills the pass at the budget; the sampler appends as it goes and
# skips finished prompts, so a killed pass resumes.
cd "$(dirname "$0")/.."
CAP=${CAP:-7200}
root_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
if [ "$root_gb" -lt 4 ]; then echo "not starting: ${root_gb} GB free on /" >&2; exit 1; fi
export CAP
exec flock /tmp/claude-gpu.lock bash -c '
  echo "owner=seldonian-fairness p9-sample pid=$$ start=$(date -Is)" > /tmp/claude-gpu.lock.info
  timeout --signal=INT --kill-after=120 "$CAP" .venv/bin/python scripts/p9_sample.py "$@"
  rc=$?
  [ "$rc" -eq 124 ] && echo "stopped at the CAP of $CAP s" >&2
  : > /tmp/claude-gpu.lock.info
  exit $rc' _ "$@"
