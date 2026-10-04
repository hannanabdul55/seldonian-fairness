#!/bin/bash
# Spike 021 stages under the shared GPU lock with a budget timeout (CONVENTIONS 2026-10-01).
#   ./run.sh A <model> <model-id> <tag> [--lora <name>=<path>]      # serve + attacked run + none run + stageA.py
#   CAP (seconds) defaults to 4 h per stage; the design's total is 12 GPU-hours.
STAGE=$1; shift
CAP=${CAP:-14400}
cd "$(dirname "$0")"
root_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
if [ "$root_gb" -lt 3 ]; then echo "not starting: ${root_gb} GB free on /" >&2; exit 1; fi
export CAP STAGE
exec flock /tmp/claude-gpu.lock bash -c '
  echo "owner=seldonian-fairness spike021 stage $STAGE $* pid=$$ start=$(date -Is)" > /tmp/claude-gpu.lock.info
  timeout --signal=INT --kill-after=120 "$CAP" ./stage_"$STAGE".sh "$@"
  rc=$?
  [ "$rc" -eq 124 ] && echo "stopped at the CAP of $CAP s" >&2
  pkill -INT -f "[v]llm serve" 2>/dev/null; sleep 5   # bracket: never match this shell (gpu-lock memory)
  : > /tmp/claude-gpu.lock.info
  ../../../scripts/backup_offdisk.sh
  exit $rc' _ "$@"
