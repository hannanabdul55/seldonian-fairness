#!/bin/bash
# E1 of the harmful-twin proposal, under the shared GPU lock (gpu-lock convention).
# Registration: .planning/harm-twin/REGISTRATION-E1-E5.md. Budget: 8 h hard cap as a timeout,
# checkpoints saved before the cap. Run outputs go to D: (the WSL root has < 5 GB free); only
# small reports and JSON summaries land under results/twin/ in the repo. Appends and skips what
# is already on disk, so a cut run resumes. Log in results/twin/e1/run.log.
#
#   STAGE=bruteforce scripts/run_twin_e1.sh   # 20,000 responses x 200 prompts from frozen p, judged
#   STAGE=warmstart   scripts/run_twin_e1.sh   # SFT the twin adapter on unsafe PKU responses
#   STAGE=train       scripts/run_twin_e1.sh   # 300 GRPO steps, logging every 10
#   STAGE=analyse     scripts/run_twin_e1.sh   # E1a-E1d scored against the registration (CPU)
#   STAGE=all         scripts/run_twin_e1.sh   # the four in order
#
# A smoke test is a different entry point (scripts/twin_e1.py smoke ...) and prints timings and
# key names only; it never runs here.
set -u
cd "$(dirname "$0")/.."

STAGE=${STAGE:-all}
CAP=${CAP:-28800}                       # 8 hours, the registered E1 budget
RUNROOT=${RUNROOT:-/mnt/d/seldonian-runs/twin/e1}
OUT=results/twin/e1
export HF_HOME=${HF_HOME:-/mnt/d/hf-cache}

mkdir -p "$OUT" "$RUNROOT"

# free-space preflight on both volumes (storage-safeguards): the repo root and the D: run tree
root_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
d_gb=$(df -BG --output=avail /mnt/d | tail -1 | tr -dc '0-9')
if [ "${root_gb:-0}" -lt 3 ]; then echo "not starting: ${root_gb} GB free on /" >&2; exit 1; fi
if [ "${d_gb:-0}" -lt 20 ]; then echo "not starting: ${d_gb} GB free on /mnt/d" >&2; exit 1; fi

# the analyse stage is CPU-only and needs no GPU lock
if [ "$STAGE" = "analyse" ]; then
  exec .venv/bin/python -I scripts/twin_e1.py analyse --runroot "$RUNROOT" --out "$OUT" \
    2>&1 | tee -a "$OUT/run.log"
fi

export STAGE CAP RUNROOT OUT
exec flock /tmp/claude-gpu.lock bash -c '
  trap ": > /tmp/claude-gpu.lock.info" EXIT
  echo "owner=seldonian-fairness twin-E1 stage=${STAGE} pid=$$ start=$(date -Is) cap=${CAP}s" \
    > /tmp/claude-gpu.lock.info
  timeout --signal=INT --kill-after=120 "$CAP" \
    .venv/bin/python -I scripts/twin_e1.py "$STAGE" --runroot "$RUNROOT" --out "$OUT" 2>&1 \
    | grep --line-buffered -v "Loading checkpoint shards" | tee -a "$OUT/run.log"
  rc=${PIPESTATUS[0]}
  [ "$rc" -eq 124 ] && echo "E1 stage ${STAGE} stopped at the CAP of ${CAP}s" >&2
  exit $rc'
