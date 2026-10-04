#!/bin/bash
# Holds the shared GPU lock for a guard scoring pass (gpu-lock convention). Args go to
# guard_refusal_score.py. CAP (seconds, default 1 h) kills the pass at the budget; the
# scorer appends as it goes and skips finished ids, so a killed pass resumes.
cd "$(dirname "$0")/.."
CAP=${CAP:-3600}
root_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
if [ "$root_gb" -lt 4 ]; then echo "not starting: ${root_gb} GB free on /" >&2; exit 1; fi
export CAP
exec flock /tmp/claude-gpu.lock bash -c '
  echo "owner=seldonian-fairness guard-score pid=$$ start=$(date -Is)" > /tmp/claude-gpu.lock.info
  timeout --signal=INT --kill-after=120 "$CAP" .venv/bin/python scripts/guard_refusal_score.py "$@"
  rc=$?
  [ "$rc" -eq 124 ] && echo "stopped at the CAP of $CAP s" >&2
  : > /tmp/claude-gpu.lock.info
  exit $rc' _ "$@"
