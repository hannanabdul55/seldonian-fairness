# O2 registration: the recommended design inside the training loop

Written 2026-10-14, before any run of `scripts/inloop_wald.py`. This file and the script are
committed and pushed together before the run; the pushed commit is the timestamp. Open item O2
of PLAN.md section 10 (blind review W6).

## Why

Section 8.1 recommends 8 equal rank strata of an 8-sample reference rate with the stratified
Wald-t limit (`b1`), and the same limit with the sampled-pool term for a claim about the prompt
source. Section 7.2's in-loop check ran 4 strata, 500 runs a cell, and only delta 0.10; there
the Wald-t source-rate limit was over its level in one of four cells (icc05, 0.132). The
recommended configuration has not been run inside a training loop.

## What is fixed

- **Harness.** `bandit014.run` (spike 014; the real `SeldonianLLMPolicy`, Lagrangian GRPO on
  `HeteroEnv`) at spike 013's in-loop cell: pressure 1, eta 100, margin +0.03, 200 steps,
  method `seldonian_lag`, the policy's own safety test at delta 0.10 (so the candidate is
  chosen at delta 0.10 for both deltas scored).
- **Cells.** The four environments of spike 013 (`icc05`, `icc26`, `icc55`, `icc75`), each under
  two rules: `strat_ref` (8 equal rank strata of the 8-sample reference rate, random ties,
  proportional allocation, safety set 40% of the 1,000-prompt pool) and `random`.
- **Runs.** 8,000 a cell, seeds 100000-107999 (spikes 013 and 014 used 0-499), 64,000 runs.
- **Scoring.** Every bound at delta 0.05 and 0.10 on the same safety labels. A miss is truth
  above the bound over all runs (not conditioned on passing). Pool claims against the pool's
  rate, source claims against the 20,000-context population's rate. A cell is over its level
  when its miss exceeds delta by more than 2 Monte Carlo SE, unresolved between delta and that.
  At 8,000 runs a one-point excess is called over with power 0.83 at delta 0.10 and 0.98 at
  0.05.
- **Gain.** (mean width of the pooled Wilson bound on the random split / mean width of the
  Wald-t limit on the strata)^2, width = bound minus the arm's estimate, by environment and
  delta.

## Predictions

Scored by `inloop_wald.py report` as committed.

- **Q1.** The Wald-t limit on 8 strata, for the pool's rate: over in at most 1 of its 8 cells.
- **Q2.** The Wald-t limit on 8 strata with the sampled-pool term, for the source rate: over in
  at most 2 of its 8 cells.
- **Q3.** The same at icc05 and delta 0.10: over (the 4-strata cell was).
- **Q4.** The pooled Wilson bound on a random split, for the source rate: over in at most 2 of
  its 8 cells.
- **Q5.** The gain of the Wald-t source-rate limit at delta 0.10 is at least 1.6 at icc75.
- **Q6.** The same gain at icc05 is between 0.9 and 1.15.

Reported without a prediction: `b1w` on the strata (pool and source), the policy's t-test on
both rules, the pool-rate pooled Wilson cell, the solution rate, rho.

## Seen before writing this

Spike 013's 4-strata in-loop table (`.planning/spikes/013-stratified-safety-set/inloop.md`) and
the paper's Table 4. Spike 014's `bandit.json.xz` already holds 8-strata `b1` scores in other
cells (eta 400 and other margins, no icc05, 300 runs a cell); they were not summarised and were
not read for this registration.

## Reading

Q1 and Q2 stand behind recommending the Wald-t limit inside a loop; a refuted one is reported
as a limit of the recommendation. Q3 tests whether the one over cell repeats at 8 strata. The
bandit is synthetic; no language model is in this check.

## Not allowed after the run

Changing cells, seeds, run count, the cell rule or the scoring code. A forced deviation is
reported as one.
