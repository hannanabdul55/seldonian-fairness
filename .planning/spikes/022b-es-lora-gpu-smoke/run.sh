#!/bin/bash
# Holds the shared GPU lock for spike 022b (gpu-lock convention) and caps the run with a
# timeout (CAP seconds, default 40 minutes). smoke.py saves its results after every phase.
cd "$(dirname "$0")"
CAP=${CAP:-2400}
root_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
if [ "$root_gb" -lt 4 ]; then echo "not starting: ${root_gb} GB free on /" >&2; exit 1; fi
mkdir -p ../../../results/spikes/022b /mnt/d/seldonian-runs/022b
export CAP
exec flock /tmp/claude-gpu.lock bash -c '
  echo "owner=seldonian-fairness spike022b pid=$$ start=$(date -Is)" > /tmp/claude-gpu.lock.info
  timeout --signal=INT --kill-after=120 "$CAP" ../../../.venv/bin/python smoke.py "$@"
  rc=$?
  [ "$rc" -eq 124 ] && echo "stopped at the CAP of $CAP s" >&2
  : > /tmp/claude-gpu.lock.info
  ../../../scripts/backup_offdisk.sh
  exit $rc' _ "$@"
