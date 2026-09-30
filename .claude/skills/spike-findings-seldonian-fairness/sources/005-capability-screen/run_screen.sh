#!/bin/bash
# Holds the shared GPU lock for the capability screen (see gpu-lock convention).
cd "$(dirname "$0")"
# refuse to start on a nearly full disk: a full WSL root goes read-only (2026-09-18)
root_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
hf_gb=$(df -BG --output=avail "${HF_HOME:-$HOME}" | tail -1 | tr -dc '0-9')
if [ "$root_gb" -lt 4 ] || [ "$hf_gb" -lt 20 ]; then
    echo "not starting: ${root_gb} GB free on /, ${hf_gb} GB free under HF_HOME" >&2; exit 1
fi
exec flock /tmp/claude-gpu.lock bash -c '
  echo "owner=seldonian-fairness spike005 capability screen pid=$$ start=$(date -Is) eta=~5-6h (round 3: 7-14B in 4-bit)" > /tmp/claude-gpu.lock.info
  ../../../.venv/bin/python screen.py --models Qwen/Qwen2.5-7B-Instruct ibm-granite/granite-3.3-8b-instruct Qwen/Qwen3-8B Qwen/Qwen2.5-14B-Instruct
  rc=$?
  : > /tmp/claude-gpu.lock.info
  ../../../scripts/backup_offdisk.sh
  exit $rc'
