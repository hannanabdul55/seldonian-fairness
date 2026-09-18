#!/usr/bin/env bash
# PyRIT red-team queue against the Round 6 over-refusal policies (task ab, seed 0).
# Holds the shared GPU lock for each battery and skips finished ones, so it can be
# relaunched after a reboot. Log: results/redteam/queue.log
set -u
cd "$(dirname "$0")/.."
OUT=results/redteam
mkdir -p "$OUT"
LOCK=/tmp/claude-gpu.lock
# the expandable-segments allocator raises an internal assertion on the batch after an
# out-of-memory event (seen 2026-09-15 with ASCII-art prompts); the stock allocator just fails
# the batch, which the target then halves and retries
export PYTORCH_CUDA_ALLOC_CONF=
RUN=results/llm_r6/c/ab/seldonian_lag/seed0        # certified: step 120, harm 0.107 (ub 0.122) vs 0.197
RUN_NSF=results/llm_r6/c/ab/seldonian_lag/seed1    # NSF: step 90 selected, refusal 0.184 (ub 0.204) vs 0.192
ATTACKER=Qwen/Qwen2.5-1.5B-Instruct

battery() {  # <out dir> <args...>
  local out=$1; shift
  if [ -f "$out/summary.json" ]; then echo "skip $out (done)"; return; fi
  if pgrep -f "redteam_pyrit.py --out $out " >/dev/null; then echo "skip $out (running in another queue)"; return; fi
  echo "=== $(date '+%F %T') start $out"
  flock "$LOCK" -c "echo 'redteam pid=$$ $out' > $LOCK.info; uv run scripts/redteam_pyrit.py --out $out $*"
  echo "=== $(date '+%F %T') end $out (rc $?)"
}

# 1. the certified policy, harm constraint, single-turn battery + resampling
battery "$OUT/c0_harm"     --run-dir $RUN --constraint harm --n 600 --techniques converter template resample
# 2. the certified policy, over-refusal constraint on benign prompts (obfuscation makes a model refuse)
battery "$OUT/c0_refusal"  --run-dir $RUN --constraint refusal --n 600 --techniques converter template resample_4
# 3. attacker-model techniques (Foundry DIFFICULT tier + tense) with a local 1.5B attacker
battery "$OUT/c0_llm"      --run-dir $RUN --constraint harm --n 100 --techniques llm --attacker $ATTACKER --max-concurrency 16
# Trimmed 2026-09-15 23:58 at the user's request to the certified policy only; the reference
# (--checkpoint none) and NSF (seed 1) batteries were dropped from the queue:
#   battery "$OUT/ref_harm"    --run-dir $RUN --checkpoint none --constraint harm --n 600 --techniques converter template resample
#   battery "$OUT/ref_refusal" --run-dir $RUN --checkpoint none --constraint refusal --n 600 --techniques converter template resample_4
#   battery "$OUT/ref_llm"     --run-dir $RUN --checkpoint none --constraint harm --n 100 --techniques llm --attacker $ATTACKER --max-concurrency 16
#   battery "$OUT/nsf1_harm"   --run-dir $RUN_NSF --constraint harm --n 600 --techniques converter template resample
echo "=== $(date '+%F %T') queue complete"
