#!/bin/bash
# E6 of the harmful-twin proposal, under the shared GPU lock (gpu-lock convention).
# Registration: .planning/harm-twin/REGISTRATION-E1-E5.md, section "2026-10-21 — NEW EXPERIMENT E6".
# Budget: 12 h hard cap as a timeout on the reference stage; the pool and partial reference are saved
# first and the stage resumes (skips prompts already on disk). Run outputs go to D:; only small reports
# and JSON land under results/twin/e6 in the repo. Log in results/twin/e6/run.log.
#
#   STAGE=reference scripts/run_twin_e6.sh   # m0 responses x pool prompts from frozen p, WildGuard-labelled; features
#   STAGE=analyse   scripts/run_twin_e6.sh   # fit reward model, simulate certificate, score E6a-E6d (CPU)
#
# A smoke test is a different entry point (scripts/twin_e6.py smoke ...) and prints timings and key
# names only; it never runs here.
set -u
cd "$(dirname "$0")/.."

STAGE=${STAGE:-reference}
CAP=${CAP:-43200}                       # 12 hours, the registered E6 reference budget
RUNROOT=${RUNROOT:-/mnt/d/seldonian-runs/twin/e6}
OUT=results/twin/e6
export HF_HOME=${HF_HOME:-/mnt/d/hf-cache}

mkdir -p "$OUT" "$RUNROOT"

# free-space preflight on both volumes (storage-safeguards)
root_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
d_gb=$(df -BG --output=avail /mnt/d | tail -1 | tr -dc '0-9')
if [ "${root_gb:-0}" -lt 3 ]; then echo "not starting: ${root_gb} GB free on /" >&2; exit 1; fi
if [ "${d_gb:-0}" -lt 20 ]; then echo "not starting: ${d_gb} GB free on /mnt/d" >&2; exit 1; fi

# the analyse stage is CPU-only and needs no GPU lock
if [ "$STAGE" = "analyse" ]; then
  exec .venv/bin/python -I scripts/twin_e6.py analyse --runroot "$RUNROOT" --out "$OUT" \
    2>&1 | tee -a "$OUT/run.log"
fi

export STAGE CAP RUNROOT OUT
exec flock /tmp/claude-gpu.lock bash -c '
  trap ": > /tmp/claude-gpu.lock.info" EXIT
  echo "owner=seldonian-fairness twin-E6 stage=${STAGE} pid=$$ start=$(date -Is) cap=${CAP}s" \
    > /tmp/claude-gpu.lock.info
  timeout --signal=INT --kill-after=120 "$CAP" \
    .venv/bin/python -I scripts/twin_e6.py "$STAGE" --runroot "$RUNROOT" --out "$OUT" 2>&1 \
    | grep --line-buffered -v "Loading checkpoint shards" | tee -a "$OUT/run.log"
  rc=${PIPESTATUS[0]}
  [ "$rc" -eq 124 ] && echo "E6 stage ${STAGE} stopped at the CAP of ${CAP}s" >&2
  exit $rc'
