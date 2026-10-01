#!/bin/bash
# Holds the shared GPU lock for spike 014 (gpu-lock convention). Args go to gen014.py.
cd "$(dirname "$0")"
root_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
if [ "$root_gb" -lt 4 ]; then echo "not starting: ${root_gb} GB free on /" >&2; exit 1; fi
exec flock /tmp/claude-gpu.lock bash -c '
  echo "owner=seldonian-fairness spike014 $* pid=$$ start=$(date -Is)" > /tmp/claude-gpu.lock.info
  ../../../.venv/bin/python gen014.py "$@"
  rc=$?
  : > /tmp/claude-gpu.lock.info
  ../../../scripts/backup_offdisk.sh
  exit $rc' _ "$@"
