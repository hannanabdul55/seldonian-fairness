#!/bin/bash
# Holds the shared GPU lock for spike 009 (see gpu-lock convention). Args go to transfer.py.
cd "$(dirname "$0")"
root_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
if [ "$root_gb" -lt 4 ]; then echo "not starting: ${root_gb} GB free on /" >&2; exit 1; fi
exec flock /tmp/claude-gpu.lock bash -c '
  echo "owner=seldonian-fairness spike009 granite transfer pid=$$ start=$(date -Is) eta=~2h" > /tmp/claude-gpu.lock.info
  ../../../.venv/bin/python transfer.py "$@"
  rc=$?
  : > /tmp/claude-gpu.lock.info
  ../../../scripts/backup_offdisk.sh
  exit $rc' _ "$@"
