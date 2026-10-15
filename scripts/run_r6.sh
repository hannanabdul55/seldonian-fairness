#!/usr/bin/env bash
# Paper step R6 (.planning/paper-certification/PLAN.md): two more seeds (1, 2) of spike 014's
# constrained Lagrangian GRPO run, judged by the guard as 014 did, then scored with spike 017's
# E8 judge pass and analysed by transfer017.py, so that section 9.2 rests on three runs.
#
# Order: 014 train s1, 014 judge s1, 014 train s2, 014 judge s2, 017 scoring of the four new
# sets, transfer017.py (CPU). Every GPU stage goes through its spike's own run.sh, which takes
# /tmp/claude-gpu.lock for that stage only, so other sessions can interleave between stages.
# This script never touches the lock itself.
#
#   scripts/run_r6.sh                  # the queue; log in /mnt/d/seldonian-runs/r6/
#   DRY=1 scripts/run_r6.sh            # print what a launch would do now, run nothing
#
# Re-launching after a crash resumes:
#   - train: a seed whose pool samples are complete (500 prompts at step 100 and at step 200 in
#     results/spikes/014/gen_s<N>.jsonl) is skipped. gen014.py cannot resume training (TRL runs
#     with save_strategy "no" and starts again from step 0 with the same seed), while its pool
#     sampling skips prompts already in gen_s<N>.jsonl; a restart on a partial file would mix
#     the step-100 samples of a dead trajectory with the step-200 samples of a new one. So a
#     partial gen_s<N>.jsonl (and judged_s<N>.jsonl, if any) is moved to
#     /mnt/d/seldonian-runs/r6/aside-<time>/ and the run directory /mnt/d/seldonian-runs/014/s<N>
#     is renamed, before the seed is trained again from the start. Nothing is deleted.
#   - judge: gen014.py --stage judge appends and skips rows already judged; it is skipped when
#     judged_s<N>.jsonl has as many rows as gen_s<N>.jsonl.
#   - scoring: score017.py appends and skips (set, wording, id) rows already scored.
#   - transfer017.py is deterministic (rng 17) and always rerun.
# A train stage stopped by its CAP with the pool samples incomplete stops the queue (a retry
# is another 16 h and is the user's call). One stopped by the CAP after the step-200 samples
# (during the final safety test) has every input of 9.2: the queue goes on and says
# run_s<N>.json is missing.
#
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True changes how the CUDA caching allocator maps
# memory, not what is computed: no code on these paths reads free memory or catches an
# out-of-memory error to change a batch (seldonian/llm, gen013/gen014, rubric015), and TRL's
# auto_find_batch_size is off. It may avoid the spill to host memory that tripled s0's step time.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
S014="$REPO/.planning/spikes/014-pushed-label-stratification"
S017="$REPO/.planning/spikes/017-calibration-carrying-certificate"
OUT014="$REPO/results/spikes/014"
RUNS014=/mnt/d/seldonian-runs/014
LOGDIR=/mnt/d/seldonian-runs/r6
SEEDS=(1 2)
TRAIN_CAP=57600          # 16 h; s0 took 45,982 s (12.8 h) end to end, and a cut run cannot resume
JUDGE_CAP=3600           # s0: 976 s for 8,000 responses
SCORE_CAP=3600           # E8: 475 s for 6,000 passes; here 12,000
DRY=${DRY:-0}

export HF_HOME=/mnt/d/hf-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$LOGDIR"
QLOG="$LOGDIR/queue.log"

say() { echo "=== $(date '+%F %T %Z') $*" | tee -a "$QLOG"; }

disk_ok() {
  local gb
  gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
  if [ "$gb" -lt 4 ]; then say "STOP: ${gb} GB free on / (needs 4)"; return 1; fi
}

# rows of gen_s<N>.jsonl at a step (the number of pool prompts sampled there)
gen_rows() {
  [ -f "$OUT014/gen_s$1.jsonl" ] || { echo 0; return; }
  python3 -c "import json,sys; print(sum(json.loads(l)['step'] == $2 for l in open(sys.argv[1])))" "$OUT014/gen_s$1.jsonl"
}
lines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }
train_done() { [ "$(gen_rows "$1" 100)" -eq 500 ] && [ "$(gen_rows "$1" 200)" -eq 500 ]; }
judge_done() { train_done "$1" && [ "$(lines "$OUT014/judged_s$1.jsonl")" -eq "$(lines "$OUT014/gen_s$1.jsonl")" ]; }

move_aside() {   # a partial seed: keep its files, out of the way of a fresh start
  local n=$1 dst="$LOGDIR/aside-$(date +%Y%m%d-%H%M%S)-s$1"
  say "seed $n: partial outputs (step 100: $(gen_rows "$n" 100), step 200: $(gen_rows "$n" 200) of 500); moving them to $dst"
  [ "$DRY" = 1 ] && return 0
  mkdir -p "$dst"
  for f in gen judged run safety; do
    for g in "$OUT014/${f}_s$n".json*; do [ -e "$g" ] && mv "$g" "$dst/"; done
  done
  [ -d "$RUNS014/s$n" ] && mv "$RUNS014/s$n" "$RUNS014/s$n.aside-$(date +%Y%m%d-%H%M%S)"
  return 0
}

stage() {   # <log name> <cap> <dir> <args...>: one spike stage through its own run.sh (GPU lock inside)
  local name=$1 cap=$2 dir=$3; shift 3
  disk_ok || return 1
  say "start $name: CAP=$cap $dir/run.sh $*"
  [ "$DRY" = 1 ] && return 0
  CAP=$cap "$dir/run.sh" "$@" >> "$LOGDIR/$name.log" 2>&1
  local rc=$?
  say "end $name (rc $rc)"
  return $rc
}

say "R6 queue start (pid $$, DRY=$DRY, HF_HOME=$HF_HOME, PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF)"
disk_ok || exit 1
grep -q '"verdict": *"OPEN' "$S014/gate.json" || { say "STOP: spike 014's gate.json is not OPEN"; exit 1; }
[ -d "$HF_HOME/hub" ] || { say "STOP: $HF_HOME/hub missing (is D: mounted?)"; exit 1; }

for n in "${SEEDS[@]}"; do
  if train_done "$n"; then
    say "seed $n: pool samples complete, train skipped$([ -f "$OUT014/run_s$n.json" ] || echo ' (run_s'$n'.json missing: final safety test did not finish)')"
  else
    if [ -e "$OUT014/gen_s$n.jsonl" ] || [ -e "$OUT014/judged_s$n.jsonl" ] || [ -d "$RUNS014/s$n" ]; then
      move_aside "$n"
    fi
    stage "train_s$n" "$TRAIN_CAP" "$S014" --stage train --steps 200 --cand 12 --seed "$n"
    rc=$?
    if [ "$DRY" != 1 ] && ! train_done "$n"; then
      say "STOP: seed $n train ended (rc $rc) without complete pool samples; see $LOGDIR/train_s$n.log"
      exit 1
    fi
  fi
  if judge_done "$n"; then
    say "seed $n: judged, skipped"
  else
    stage "judge_s$n" "$JUDGE_CAP" "$S014" --stage judge --seed "$n"
    if [ "$DRY" != 1 ] && ! judge_done "$n"; then say "STOP: seed $n judge incomplete"; exit 1; fi
  fi
done

SETS=ref_c1_014s1_100,ref_c1_014s1_200,ref_c1_014s2_100,ref_c1_014s2_200
stage score_r6 "$SCORE_CAP" "$S017" --sets "$SETS" || { say "STOP: scoring did not finish (it resumes on relaunch)"; exit 1; }

say "transfer017.py (CPU)"
if [ "$DRY" != 1 ]; then
  (cd "$S017" && CUDA_VISIBLE_DEVICES= ../../../.venv/bin/python transfer017.py) > "$LOGDIR/transfer.log" 2>&1 \
    || { say "STOP: transfer017.py failed, see $LOGDIR/transfer.log"; exit 1; }
  grep -q "^## Constrained runs, one row per seed and step" "$S017/transfer.md" \
    || { say "STOP: transfer.md has no per-seed section (a seed's scores incomplete?)"; exit 1; }
fi
say "R6 queue done: $S017/transfer.md (section 'Constrained runs'), transfer_seeds.json"
