---
spike: 014
idea: rerandomized-split
name: pushed-label-stratification
type: standard
validates: "Given the over-refusal label driven by LagrangianReward (not a side effect), when per-prompt rates are sampled at the reference and trained checkpoints, then we measure rate compression and rho decay and whether 013's 2.4x stratification gain survives (bandit first, GPU only if it does)"
verdict: VALIDATED
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
    ./run.sh --stage rmcheck; ./run.sh --stage pilot         # stage 2 checks (GPU lock)
    ./run.sh --stage train --steps 200 --cand 12             # stage 2 run; samples the pool at steps 100 (K 4) and 200 (K 12)
    ./run.sh --stage judge                                   # Qwen3Guard-4B -> results/spikes/014/judged_s0.jsonl
    OMP_NUM_THREADS=1 ../../../.venv/bin/python plasmode014.py --reps 5000 --out plasmode.json   # stage 3, 3 min
    ../../../.venv/bin/python analyse014.py --tag s0 --plasmode plasmode.json                    # results.md, stage3.json
    ../../../.venv/bin/python ../013-stratified-safety-set/summarise_plasmode.py plasmode.json --key env   # the full k x H x n_s table

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
4. **Stage 2 pilot** (20 steps, 100 pool prompts, 4 samples each; `pilot.log`): the full stack
   fits (policy, Qwen3Guard-4B refusal judge in 4-bit, the 0.6B reward model), 60 s a
   training step with the judge scoring every group, 3.2 pool generations a second, and the
   policy's own safety test on the 100 prompts gave refusal 0.08 against a reference of 0.186
   after 20 steps: the multiplier at its floor of 5 pushes the label down from step 1, as
   010 found in the bandit. The design's pilot stop (a move under 2 points) does not fire.
   At that speed the full plan (K 16 at two checkpoints) is about 5.4 GPU-hours, over the
   cap, so the fallback applies: K = 12 at step 200 (the plasmode checkpoint) and 4 at step
   100 (trajectory only). Estimated 4.5 h plus 20 minutes of judging. Launched 17:45 PT.
   A first launch failed at once on a missing `prompt_id` in the safety-set records
   (fixed with `make_record`).
5. **Stage 2 ran to completion but over its budget.** `train.log`: 200 steps, the first 20 at
   62 s a step as in the pilot, then 2.7-4 minutes a step from step 30 on (an abrupt jump in `step_time` from 62 to 165 s
   with the completion length unchanged at about 115 tokens; the lock was held throughout;
   the GPU sat at 11.98 of 12 GB, so the Windows driver's spill to host memory is the likely
   cause, not diagnosed), so training alone took 11.5 h against the
   design's 3.4 h estimate, plus 73 minutes of pool sampling (2,000 generations at step 100,
   6,000 at step 200, 1.8 a second). 12.8 GPU-hours against the 5-hour cap in DESIGN.md
   section 5: the cap was passed at about 22:45 PT with the run at step 100, and it was
   left to finish rather than lose the step-200 checkpoint. The run itself did what C4 asked:
   the predicted test oscillated between 0.10 and 0.26 with the multiplier between 5 (floor)
   and 12; the returned policy is step 175 (best feasible predicted test) and the safety
   test on D_s passed at 0.168 (upper 0.198, threshold 0.206). Judging queued on the lock at
   06:35 PT on 2026-10-01 behind another session's job.
6. **Judging and the plasmode.** 8,000 pool responses judged in 16 minutes
   (`judged_s0.jsonl`); `plasmode014.py` ran 013's real plasmode (random ties) on 014's two
   checkpoints and, for the like-for-like comparison, on 013's own side-effect checkpoints
   at the same steps against the same reference labels; `analyse014.py` wrote `results.md`
   and `stage3.json`.

## Stage 2-3 results (Granite over-refusal under the Lagrangian)

Full tables in `results.md`. Reference: 013's 8 samples per C1 prompt, refusal 0.186,
ICC_ref 0.72; threshold 0.206; the policy's safety test on D_s passed at 0.168.

- **The push was real during training and two-sided.** The predicted test swung between
  0.10 and 0.26 (multiplier 5-12; the reward model pulls refusal up, the multiplier pushes
  it down). At the sampled checkpoints the pool rate was 0.151 at step 100 (3.4 points
  below the reference, Round 6's constrained-arm size) and 0.182 at step 200 (back at the
  reference, net of the two forces). The 13-point move of Round 6's unconstrained arm was not
  reached; the bandit's cells covered moves of 2-9 points.
- **H1 (compression) is refuted on real data as in the bandit.** ICC_cand / ICC_ref 1.03 at
  step 100 and 1.01 at step 200; rho (disattenuated) 0.91 and 0.92, against 1.00 / 0.99 for
  013's side-effect checkpoints on the same prompts. Pushing the label cost 0.08 of
  persistence and no spread.
- **The gain survives: realised ESS of 013's rule (S2, k 8, H 8, n_s 200, `b1w`) 2.11 at
  step 100 and 2.21 at step 200 at delta 0.1 (2.05 / 2.16 at delta 0.05), with `b1w` valid
  (miss 0.040-0.064 at delta 0.1, 0.011-0.023 at 0.05).** Like for like, 013's side-effect
  checkpoints give 2.42-2.49: the push costs about 0.2-0.3 of ESS.
- **H3 holds.** The bandit cells at the real compression (ratio 1.01 +- 0.15: the icc75
  cells) have ESS 3.35 with stratified miss 0.07-0.09.
- **H4 holds.** The bandit-calibrated prediction (rho 0.87 from the gate's nearest cells,
  ratio 1.21, at the real ICC_ref) is 2.00 against the realised 2.21 at delta 0.1 (error
  0.21) and 2.16 at 0.05 (0.16), both inside the 0.3 asked. The formula at the *measured*
  real moderators is 2.26 (error 0.04-0.09), as H2 found in the bandit; the naive pre-flight
  (rho 0.8) gives 1.72, under by 0.44-0.49.
- **Go rule met (H2 and H4): `013/preflight.py --pushed`** uses rho interpolated by ICC_ref
  from the bandit's table (0.69 at 0.26, 0.87 at 0.51, 0.98 at 0.75) with ICC_cand = ICC_ref.
  On this pool it predicts 2.56 against the realised 2.21 (13% over; the measured rho was
  0.92 where the table says 0.96), an upper estimate as 013's other predictions are. 013's
  use-case map gains the row "over-refusal pushed by the Lagrangian: ESS 2.1-2.2, use it",
  and its "a label the training does not target" condition is dropped.

**Verdict: VALIDATED.** 013's stratification gain survives a label the Lagrangian targets
directly: no compression, rho 0.92, ESS 2.1-2.2 against 2.4-2.5 for the same label as a
side effect, bounds valid, and the formula at the measured moderators predicts it. The
narrowing: one model, one pool, one label, net rate moves of at most 3 points on real data
(2-9 in the bandit); a push that lands the rate far from the reference was not seen. The
budget deviation (12.8 GPU-hours against the 5-hour cap, item 5) is a process failure, not
a design one: the cap was watched, not enforced, and `run.sh` should carry a `timeout`
with the checkpoint saved before it (CONVENTIONS).

### Scored expectations
| hypothesis | asked | found | result |
|---|---|---|---|
| H1 compression | ICC ratio and rho fall with pressure; rho < 0.7 at the strongest feasible pressure | ratio 1.0-1.5, rho set by ICC_ref (0.69-0.98), flat in pressure; real ratio 1.01, rho 0.92 | refuted |
| H2 prediction | formula at measured moderators within 0.15 (median), Spearman >= 0.8; naive over by > 0.3 | 0.097, 0.97; naive *under* by 0.40 | holds (direction of the naive error reversed) |
| H3 survival | ESS >= 1.2 and valid at the matched compression | 3.35, miss 0.07-0.09 | holds |
| H4 real data | realised within 0.3 of the bandit-calibrated prediction, `b1w` valid | 2.21 vs 2.00 (0.21), miss 0.064 | holds |

## Files
`DESIGN.md` (approved design), `bandit014.py`, `summarise014.py` (`bandit.md`, `gate.json`),
`gen014.py` + `run.sh` (stages rmcheck / pilot / train / judge), `plasmode014.py`
(`plasmode.json`), `analyse014.py` (`results.md`, `stage3.json`), logs `pilot.log`,
`train.log`, `judge.log`. Data in `results/spikes/014/`: `bandit.json.xz`, `rmcheck.json`,
`gen_pilot.jsonl`, `run_pilot.json`, `safety_pilot.jsonl`, `gen_s0.jsonl.xz`, `run_s0.json`,
`safety_s0.jsonl`, `judged_s0.jsonl`.
