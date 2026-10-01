#!/bin/bash
# Holds the shared GPU lock for spike 014 (gpu-lock convention). Args go to gen014.py.
# CAP (seconds, default 5 h) kills the stage at the budget: gen014.py writes each
# checkpoint's pool samples as it reaches them, so a kill leaves the earlier ones
# (CONVENTIONS 2026-10-01; this run itself was launched without it and ran 12.8 h).
cd "$(dirname "$0")"
CAP=${CAP:-18000}
root_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
if [ "$root_gb" -lt 4 ]; then echo "not starting: ${root_gb} GB free on /" >&2; exit 1; fi
export CAP
exec flock /tmp/claude-gpu.lock bash -c '
  echo "owner=seldonian-fairness spike014 $* pid=$$ start=$(date -Is)" > /tmp/claude-gpu.lock.info
  timeout --signal=INT --kill-after=120 "$CAP" ../../../.venv/bin/python gen014.py "$@"
  rc=$?
  [ "$rc" -eq 124 ] && echo "stopped at the CAP of $CAP s" >&2
  : > /tmp/claude-gpu.lock.info
  ../../../scripts/backup_offdisk.sh
  exit $rc' _ "$@"
