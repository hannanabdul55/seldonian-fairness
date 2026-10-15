#!/bin/bash
# Holds the shared GPU lock for spike 017's judge-only pass (gpu-lock convention).
# CAP (seconds, default 2 h) kills the pass at the budget, as 014's run.sh does: score017.py
# appends its rows in chunks and resumes from them, so a kill loses at most one chunk.
cd "$(dirname "$0")"
CAP=${CAP:-7200}
root_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
if [ "$root_gb" -lt 4 ]; then echo "not starting: ${root_gb} GB free on /" >&2; exit 1; fi
export CAP
exec flock /tmp/claude-gpu.lock bash -c '
  echo "owner=seldonian-fairness spike017 judge-only pass, about 40 min $* pid=$$ start=$(date -Is)" > /tmp/claude-gpu.lock.info
  timeout --signal=INT --kill-after=120 "$CAP" ../../../.venv/bin/python score017.py "$@"
  rc=$?
  [ "$rc" -eq 124 ] && echo "stopped at the CAP of $CAP s" >&2
  : > /tmp/claude-gpu.lock.info
  ../../../scripts/backup_offdisk.sh
  exit $rc' _ "$@"
