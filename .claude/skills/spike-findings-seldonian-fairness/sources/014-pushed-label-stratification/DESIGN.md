# Spike 014 design: stratified safety sets when training pushes the label

Status: **DRAFT for review, nothing run.** Queued by the user on 2026-09-29 (/gsd-explore),
scoped to over-refusal only. Follows spike 013 (reference-rate strata + `b1w`: ESS 2.4 on
Granite over-refusal, VALIDATED for labels the training does not target) and 017 (whose
one transfer that held was 200 steps of side-effect training on the same prompts).

## 1. The one question

> When the Lagrangian drives the over-refusal label directly (the reward pushes refusal one
> way, the multiplier the other), do per-prompt rates compress and the persistence `rho`
> decay enough to erase spike 013's stratification gain, and does 013's pre-flight number,
> evaluated at the `rho` measured under pressure, still predict what survives?

Held fixed from 013: 8 equal rank strata of an 8-sample reference rate with random
tie-breaking, proportional allocation, one response per safety prompt, the `b1w` bound,
delta 0.05 and 0.1, and the candidate being a real checkpoint with the split drawn after
training. The label is over-refusal (Qwen3Guard-4B's refusal field on XSTest + OR-Bench
prompts), rate 0.17-0.20 on the reference, ICC_ref 0.72.

**Not tested here:** harm (rare rates; the bound, not the strata, is the problem there:
`SEED-260929-nq7`), Neyman allocation, re-drawing the strata from the *candidate's* rate
(not implementable without leaking the candidate), shift between pool and deployment.

## 2. Why the gain might not survive

013's pre-flight number is

    G = ICC_cand x rho^2 x rel(k) x c_H,     ESS = 1 / (1 - G)

with `rho = corr(p_ref(x), p_cand(x))`. Direct pressure on the label changes two of the
four factors: rates compress toward the threshold (ICC_cand falls) and the prompts that
move are the ones the multiplier can move, not the ones the reference found hardest (`rho`
falls). 013 measured `rho` 0.8-1.0 only where the label was a side effect. 013's own bandit
stage did train the constrained label under the Lagrangian, at one pressure, and found
`rho` 0.44-0.98 rising with ICC_ref, with the gain surviving at ICC_ref >= 0.55 (ESS 1.3-2.8).
What it did not vary is how hard the label is pushed.

## 3. Hypotheses and decision rule (fixed before any data)

- **H1, compression.** Under the Lagrangian, `ICC_cand / ICC_ref` and `rho` both fall as
  pressure rises; at the strongest pressure that still yields a feasible candidate,
  `rho < 0.7` in the bandit.
- **H2, prediction.** Across bandit cells, `1 / (1 - G)` with `G` evaluated at the *measured*
  `ICC_cand` and `rho` predicts the realised ESS to within 0.15 in the median cell and ranks
  the cells (Spearman >= 0.8). The pre-flight run the naive way (reference ICC, `rho` = 0.8)
  over-predicts the pushed cells by more than 0.3 where H1 holds.
- **H3, survival (the gate).** The gain survives, `ESS >= 1.2` with `b1w` valid (miss <=
  delta + 2 MC se), in the bandit cell that matches the real run's compression
  (`ICC_cand / ICC_ref` within 0.15 of what the Granite run shows at its final checkpoint;
  see stage order: the real ICC_cand is measured only if the gate opens on the bandit's
  own prediction first).
- **H4, real data.** On Granite over-refusal under the Lagrangian (013's C4), realised ESS
  at step 200 is within 0.3 of the bandit-calibrated prediction, and `b1w` stays valid.

**Gate to the GPU stage:** run stage 2 only if, at the bandit's measured `rho` and
compression for a pressure that moves the rate by 3-13 points (Round 6's constrained and
unconstrained over-refusal runs moved it by 3 and 13), the predicted ESS on 013's real C1 reference labels is
>= 1.2. If the bandit says the gain is gone, record that, mark the pre-flight's
pushed-label rule from the bandit alone, and stop.

**Go** (change `preflight.py`'s verdict for labels the training targets, using the
pushed-label `rho`): H2 holds and H4 holds. **Stop, and narrow 013's use-case map to
side-effect labels only:** H3 or H4 fails.

## 4. Stage 1: the bandit, in the loop (CPU, about 1 hour)

`013/inloop.py` unchanged in its bounds and rules (`random`, `strat_ref`, `placebo`; `b1w`
pooled and stratified, pool and population targets), extended by keyword options with
neutral defaults, as 004's lab was:

| factor | levels | what it sets |
|---|---|---|
| `HeteroEnv` | icc26, icc55, icc75 | ICC_ref 0.26 / 0.55 / 0.75 (013's levels; icc05 dropped, predicted null) |
| pressure | 0.5, 1, 2, 4 | the reward's weight on the violating action (013's bandit ran at 1) |
| dual step `eta` | 100 (013), 400 | how hard the multiplier answers a predicted violation |
| threshold | reference rate minus 0.03 / 0.06 | how far the label must move |
| steps | 200, 400 | more training, more compression |
| control | `method = grpo` (no constraint) | the side-effect case, 013's setting |

500 seeds a cell, `k = 8`, `H = 8`, `n_s` 200, paired streams. The primary cells are
icc55 and icc75 at pressure 2 and 4 with `eta` 400; everything else is exploratory.

Outputs per cell: ICC_ref, ICC_cand, `rho`, rate moved, feasibility and solution rate, miss
rate of each bound against the pool and population targets, width, realised ESS, `G` at
the measured moderators, `G` the naive way, and the two predictions' errors.

## 5. Stage 2: Granite over-refusal under the Lagrangian (GPU, at most 5 hours)

013's C4, as designed there: Granite-3.3-2B, 013's C1 pool (500 prompts, the 8 covariate
samples per prompt already on disk in `results/spikes/013/`), 009's GRPO recipe with
`LagrangianReward` constraining the refusal judge to `reference rate + 0.02` while the
base reward is the reward model Round 6's stage C used, the one the constraint opposes
(there, at 0.5B, unconstrained GRPO moved refusal from 0.105 to 0.241 and the Seldonian arm
held it at 0.133). Checkpoints at steps 100 and 200 (about 75 minutes of
training at 22 s/step), then `K = 16` responses per prompt per checkpoint, judged
(about 16k generations, 1.5-2 h at 013's rate), split 8 / 8 into truth and evaluation
halves. `013/real_plasmode.py` then draws 5,000 safety sets per arm and factor cell.

Budget cap: 5 GPU-hours, `run.sh` with the free-disk check and the shared lock, TRL scratch
in `/mnt/d/seldonian-runs/014`, rows in `results/spikes/014/` (xz if large). Fallback if the
pilot rate is slow: `K = 12`, step 200 only.

Measured: everything in section 4 on real data, plus the trajectory of the rate and of
ICC_cand across 0 / 100 / 200 (the covariate samples give step 0 for free).

## 6. Stage 3: analysis and the pre-flight rule (CPU, about 1 hour)

- The bandit's `rho(pressure, compression)` table, and which of the two predictions
  (measured moderators, naive) tracks the realised ESS.
- If the gain survives on real data: `preflight.py` gets a `--pushed` option that uses the
  bandit's `rho` at the planned rate move, and 013's use-case map row for "a constraint the
  training pushes hard" changes from "don't use it" to what was measured.
- If it does not: the row stays, with the measured `rho` and ESS as its evidence.

## 7. Threats to validity

- **The bandit's pressure is not Granite's.** The gate uses compression (`ICC_cand /
  ICC_ref`, rate moved) to match cells, not the pressure knob itself.
- **One real run, one model, one pool.** H4 is a single point; a miss of 0.3 would not
  distinguish noise from a wrong formula. It is a check on direction and magnitude, not a
  calibration.
- **Round 6's reward model has only been run against a 0.5B policy.** Whether it moves
  Granite-3.3-2B's refusal rate at all is unknown; a 20-step pilot (about 10 minutes)
  measures the move before the full run, and stage 2 stops if the rate moves under 2 points.
- **Judge = Qwen3Guard-4B's refusal field**, as in 013 and 017. 017 found the compiled
  judge's recall differs by prompt source; the guard's own error by source is unmeasured.
- **Training and the split are decoupled** (as in 013 stage 3); the bandit stage runs in
  the loop and checks nothing is lost.
- **The two-phase term** applies if the certificate is for the population the pool was
  drawn from; both targets are reported, as in 013.

## 8. Order, budget, stop points

| stage | what | cost | stop if |
|---|---|---|---|
| 1 | bandit pressure sweep | CPU about 1 h | predicted ESS < 1.2 at the matched compression (gate) |
| 2 | Granite C1 Lagrangian run, two checkpoints, generation, judging | GPU <= 5 h, `flock`, overnight or a free window | - |
| 3 | plasmode, prediction check, pre-flight rule | CPU about 1 h | - |

Nothing runs before the user approves this document. Deviations are reported in the
README. No weekday 9-5 PT commits.

## 9. Deliverables

`014-pushed-label-stratification/`: README (verdict, pre-registered expectations scored),
`results.md` (bandit cells; real-data cells if stage 2 ran), the `rho(pressure)` table,
the changed or unchanged `preflight.py` rule, and MANIFEST requirement lines.
