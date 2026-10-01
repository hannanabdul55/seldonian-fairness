---
spike: 014
idea: rerandomized-split
name: pushed-label-stratification
type: standard
validates: "Given the over-refusal label driven by LagrangianReward (not a side effect), when per-prompt rates are sampled at the reference and trained checkpoints, then we measure rate compression and rho decay and whether 013's 2.4x stratification gain survives (bandit first, GPU only if it does)"
verdict: PENDING
related: [013, 012, 017]
tags: [stratification, safety-set, lagrangian, over-refusal, cpu, gpu]
---

# Spike 014: stratified safety sets when training pushes the label

The design, hypotheses and gate were fixed in `DESIGN.md` and approved by the user on
2026-09-30 before anything ran. This README records what each stage found.

## How to Run
    cd .planning/spikes/014-pushed-label-stratification
    OMP_NUM_THREADS=1 ../../../.venv/bin/python bandit014.py --seeds 300 --out bandit.json   # stage 1, ~1 h
    ../../../.venv/bin/python summarise014.py bandit.json   # bandit.md, gate.json (bandit.json is xz in results/spikes/014/)

## Pre-registration note (stage 1, written before the sweep)
DESIGN.md section 3 holds the hypotheses. Seen before the sweep: a 12-run smoke test of the
harness (2 seeds, 6 cells), in which a threshold 3 points below the reference moved the
bandit's rate down by 6-9 points, ICC_cand came out *above* ICC_ref (0.39-0.43 against
0.27-0.31 on icc26; 0.70 against 0.63 on one icc55 placebo run) and rho was 0.78-0.90. If
that pattern holds over the sweep, H1 (compression) is refuted in the bandit: pushing the
label spreads the per-prompt rates instead of compressing them, and the gain would survive
for a reason 013's formula already contains. The gate in DESIGN.md is unchanged.

## Investigation Trail
1. `bandit014.py` wraps 013's `inloop.py` pieces (strata, blocking, bounds) and re-implements
   its `run` with the five knobs at 013's defaults. The unconstrained control draws its
   safety episodes with the policy's own one-shot `_safetyTest`, so every arm's labels come
   from the same code path. Bounds are computed at delta 0.05 and 0.1 from the same labels.
2. **The reward-model check ran before the gate** (`gen014.py --stage rmcheck`, 26 s of GPU, no
   training): it settles whether Round 6's reward model can push Granite's refusal at all,
   the risk DESIGN.md section 7 names. On the pool's 4,000 reference responses it scores a
   refusal 1.43 points above an answer (0.63 sd; the same on XSTest and OR-Bench prompts).
   Within a prompt that has both kinds among its 8 samples (110 prompts) the gap is 0.31 and
   the refusal wins in 55% of them, which is the pressure GRPO's group advantage would see:
   present, weaker than the pooled gap. `results/spikes/014/rmcheck.json`. The model was
   re-downloaded to the D: cache (Apache-2.0, not gated; used since Round 1).
3. **Stage 1 ran**: 120 cells x 300 seeds, 36,000 in-loop runs, 61 minutes on 6 workers
   (`bandit.json`, `bandit.md`, `gate.json`). The gate opened. Stage 2's pilot was queued on
   the GPU lock at 16:49 PT behind another session's job.

## Stage 1 results (bandit)
- **H1 is refuted in the bandit.** Under the Lagrangian the candidate's per-prompt rates
  spread instead of compressing: ICC_cand / ICC_ref is 1.20-1.36 in every constrained cell
  and 1.44-1.51 in the unconstrained control. rho is set by the environment, not the
  pressure: 0.68-0.70 at ICC_ref 0.26, 0.84-0.87 at 0.51, 0.98 at 0.75, flat from pressure
  0.5 to 4. The pressure knob barely matters because the multiplier answers it: the rate
  lands 5-7 points below the reference whatever the pressure when the threshold is below
  it, and 1-2 points below when the threshold is above it. In the control the rate rises
  2 to 12 points with pressure.
- **At ICC_ref 0.75 the label cannot be pushed down** (moved 1-2 points, thresholds below
  the reference feasible in 1-13% of runs): with a shared risk direction the per-prompt rate
  is the prompt's, not the action's. Those cells are excluded from the gate by its
  feasibility clause and are the reason rho stays 0.98 there.
- **H2 holds.** 013's formula at the measured ICC_cand and rho predicts the realised ESS
  with a median absolute error of 0.097 over 56 cells (asked: 0.15) and Spearman 0.97
  (asked: 0.8). The naive pre-flight (ICC_ref, rho 0.8) is off by 0.40 in the median pushed
  cell, but it *under*-predicts everywhere (0 of 44 cells over by 0.3), because ICC_cand
  rose and rho stayed above 0.8.
- **Validity held.** Stratified `b1w` misses at most 0.113 at delta 0.1 and 0.063 at 0.05
  (random-split `b1w`: 0.12 and about 0.06); the placebo strata give ESS 1.01-1.02 with
  misses 0.04-0.06.
- **The gate.** Matched cells (rate moved 3-13 points, feasible in at least half the runs):
  ESS 1.15-1.17 at ICC_ref 0.26 and 1.77-1.79 at 0.51; on 013's real C1 reference labels
  (ICC_ref 0.72) the predicted ESS is 1.70-2.60, all above 1.2. **Gate open.**
