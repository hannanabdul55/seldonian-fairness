#!/bin/bash
# Holds the shared GPU lock for spike 017's judge-only pass (gpu-lock convention).
cd "$(dirname "$0")"
root_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
if [ "$root_gb" -lt 4 ]; then echo "not starting: ${root_gb} GB free on /" >&2; exit 1; fi
exec flock /tmp/claude-gpu.lock bash -c '
  echo "owner=seldonian-fairness spike017 judge-only pass, about 40 min $* pid=$$ start=$(date -Is)" > /tmp/claude-gpu.lock.info
  ../../../.venv/bin/python score017.py "$@"
  rc=$?
  : > /tmp/claude-gpu.lock.info
  ../../../scripts/backup_offdisk.sh
  exit $rc' _ "$@"
