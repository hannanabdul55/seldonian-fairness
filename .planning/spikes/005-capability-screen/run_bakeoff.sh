#!/bin/bash
# Holds the shared GPU lock for the judge bake-off (see gpu-lock convention).
cd "$(dirname "$0")"
root_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
hf_gb=$(df -BG --output=avail "${HF_HOME:-$HOME}" | tail -1 | tr -dc '0-9')
if [ "$root_gb" -lt 4 ] || [ "$hf_gb" -lt 20 ]; then
    echo "not starting: ${root_gb} GB free on /, ${hf_gb} GB free under HF_HOME" >&2; exit 1
fi
exec flock /tmp/claude-gpu.lock bash -c '
  echo "owner=seldonian-fairness spike005 judge bake-off pid=$$ start=$(date -Is) eta=~1h" > /tmp/claude-gpu.lock.info
  ../../../.venv/bin/python judge_bakeoff.py judge "$@"
  rc=$?
  : > /tmp/claude-gpu.lock.info
  ../../../.venv/bin/python judge_bakeoff.py analyze
  ../../../scripts/backup_offdisk.sh
  exit $rc' _ "$@"
