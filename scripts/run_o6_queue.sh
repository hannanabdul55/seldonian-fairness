#!/bin/bash
# Paper item O6: the pilot (3 steps), then the full run only if the pilot fits: peak reserved GPU
# memory under 11.0 GiB (spike 014 slowed threefold when it spilled at 11.98) and a projection of
# at most 3.5 h for 200 steps. Both wait on the shared GPU lock (scripts/run_o6.sh).
cd "$(dirname "$0")/.."
LOG=/mnt/d/seldonian-runs/o6/queue.log
mkdir -p /mnt/d/seldonian-runs/o6
say() { echo "=== $(date '+%F %T %Z') $*" | tee -a "$LOG"; }
say "O6 queue: pilot (waits for the GPU lock)"
scripts/run_o6.sh train --pilot || { say "STOP: pilot failed (rc $?)"; exit 1; }
M=/mnt/d/seldonian-runs/o6/pilot/pilot_meta.json
read -r PEAK HOURS < <(python3 -c "
import json, numpy as np; m = json.load(open('$M')); t = m['sec_per_step'][1:] or m['sec_per_step']
print(round(m['peak_reserved_gib'], 2), round(200 * float(np.mean(t)) / 3600, 2))")
say "pilot: peak reserved ${PEAK} GiB, projection ${HOURS} h for 200 steps"
python3 -c "import sys; sys.exit(0 if $PEAK < 11.0 and $HOURS <= 3.5 else 1)" \
  || { say "STOP: the pilot does not fit (needs < 11.0 GiB and <= 3.5 h); the full run is left to a decision"; exit 1; }
say "full run: train, generate, judge"
scripts/run_o6.sh all; rc=$?
say "full run ended (rc $rc)"
[ "$rc" -eq 0 ] && OMP_NUM_THREADS=1 .venv/bin/python scripts/o6_moved.py --stage analyse >> "$LOG" 2>&1 \
  && .venv/bin/python scripts/o6_moved.py --stage report >> "$LOG" 2>&1 && say "O6 analysed and reported"
