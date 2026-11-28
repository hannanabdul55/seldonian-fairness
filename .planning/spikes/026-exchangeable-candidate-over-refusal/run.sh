#!/bin/bash
# Holds the shared GPU lock for spike 026 (gpu-lock convention). Args go to gen026.py.
# CAP (seconds, default 6 h) kills the stage at the budget (CONVENTIONS 2026-10-01; 014 ran
# 12.8 h without one). A killed run has no safety test; its live log keeps the predicted tests.
cd "$(dirname "$0")"
CAP=${CAP:-21600}
root_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
if [ "$root_gb" -lt 4 ]; then echo "not starting: ${root_gb} GB free on /" >&2; exit 1; fi
export CAP
exec flock /tmp/claude-gpu.lock bash -c '
  echo "owner=seldonian-fairness spike026 $* pid=$$ start=$(date -Is)" > /tmp/claude-gpu.lock.info
  timeout --signal=INT --kill-after=120 "$CAP" ../../../.venv/bin/python gen026.py "$@"
  rc=$?
  [ "$rc" -eq 124 ] && echo "stopped at the CAP of $CAP s" >&2
  : > /tmp/claude-gpu.lock.info
  ../../../scripts/backup_offdisk.sh
  exit $rc' _ "$@"
