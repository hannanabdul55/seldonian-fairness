#!/bin/bash
# Holds the shared GPU lock for spike 016 (gpu-lock convention). Interactive: type a requirement, see the compiled constraint.
cd "$(dirname "$0")"
root_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
if [ "$root_gb" -lt 4 ]; then echo "not starting: ${root_gb} GB free on /" >&2; exit 1; fi
exec flock /tmp/claude-gpu.lock bash -c '
  echo "owner=seldonian-fairness spike016 $* pid=$$ start=$(date -Is)" > /tmp/claude-gpu.lock.info
  ../../../.venv/bin/python try016.py "$@"
  rc=$?
  : > /tmp/claude-gpu.lock.info
  ../../../scripts/backup_offdisk.sh
  exit $rc' _ "$@"
