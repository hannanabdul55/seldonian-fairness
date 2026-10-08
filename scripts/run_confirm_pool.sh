#!/bin/bash
# Holds the shared GPU lock for the confirmation pool (plan step R5; gpu-lock convention).
#   scripts/run_confirm_pool.sh generate|judge [args for confirm_pool.py]
# CAP (seconds) kills the stage at its budget: 3 h for generate, 2 h for judge, 5 GPU-hours in all.
# Both stages append as they go and skip what is done, so a killed stage resumes.
cd "$(dirname "$0")/.."
STAGE=$1; shift
case "$STAGE" in generate) CAP=${CAP:-10800};; judge) CAP=${CAP:-7200};; *) echo "stage: generate or judge" >&2; exit 2;; esac
root_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
if [ "$root_gb" -lt 4 ]; then echo "not starting: ${root_gb} GB free on /" >&2; exit 1; fi
export CAP STAGE HF_HOME=${HF_HOME:-/mnt/d/hf-cache}
exec flock /tmp/claude-gpu.lock bash -c '
  echo "owner=seldonian-fairness r5-confirm-$STAGE pid=$$ start=$(date -Is) cap=${CAP}s" > /tmp/claude-gpu.lock.info
  timeout --signal=INT --kill-after=120 "$CAP" .venv/bin/python scripts/confirm_pool.py --stage "$STAGE" "$@"
  rc=$?
  [ "$rc" -eq 124 ] && echo "stopped at the CAP of $CAP s" >&2
  : > /tmp/claude-gpu.lock.info
  exit $rc' _ "$@"
