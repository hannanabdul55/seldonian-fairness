# R5b registration: a mid-rate confirmation pool

Written 2026-10-08, after step R5's results and before any response on the new prompts was
generated or judged. This file, `scripts/confirm_mid.py`, `scripts/run_confirm_mid.sh`, the
pool (`results/paper/confirm/mid/pools.json`) and the preview described below are committed
and pushed together before the GPU run; the pushed commit is the timestamp.

## Why

Step R5's three pools gave refusal rates of 12%, 82% and 98%. No label fell between 18% and
65%, the range no earlier cell tested either, and R5 changed the paper's recommendation for
exactly that range: `b1w` only at rates well under one half, otherwise the stratified Wald-t
limit, with the sampled-pool term for a claim about the prompt source. That recommendation has
not been tested on prompts that played no part in it.

## How this differs from R5, stated before the run

- **The rate is aimed at.** No slice of OR-Bench sits in the range for this policy (by
  category, R5's pools run from 0% to 22%, 46% to 89% and 87% to 100%). So the pool is a fixed
  mixture of two sources whose rates R5 measured, and its rate is expected near 46%. R5 chose
  no label to land in a class; this pool does.
- **The gain is flattered.** The reference rate sorts the two sources into separate strata, so
  part of what the strata gain is the gain of knowing the source. The report states how much
  (the variance ratio of two strata by source against the 8 reference-rate strata).
- **The predictions were written after a preview.** `confirm_mid.py --stage preview` ran the
  whole analysis on a re-mix of R5's responses (207 prompts of its K1, 193 of its K2), to test
  the code and to set expectations. That is seen data and confirms nothing;
  `results/paper/confirm/mid/preview.md` holds it. Two predictions were changed by it, and both
  changes are recorded under the predictions.

## What is fixed

**Prompts.** One pool, K4, of 400 OR-Bench prompts (Cui et al., 2025): all 193 prompts of the
hard-1K set that appear in no results file of this repository, in none of the benign-prompt
loader's draws at the sizes and seeds earlier runs used, and in none of R5's pools; and 207
benign prompts of the 80K set under the same exclusions and not in the hard-1K set (35,778
unused), drawn with seed 20261009. The order is shuffled with the same generator.

**Policies, labels, strata, draws.** As R5, through the same code: Granite-3.3-2B; 8 responses
a prompt of the untrained base for the strata; 16 of spike 014's returned adapter (step 175);
128 new tokens, temperature 1; 9,600 responses. Qwen3Guard-4B, 4-bit, refusal and safety
fields. 8 equal rank strata of the 8-sample reference rate, random ties, proportional
allocation, safety sets of 100 and 200. `scripts/replacement_check.py`'s `job` (5,000 stored
draws, 40,000 with replacement, the pass the predictions use) and `scripts/twophase_check.py`'s
`job` (10,000 replications), both unchanged. Deltas 0.05 and 0.10. A cell is over its level
when its miss exceeds delta by more than two Monte Carlo standard errors. A label is rare
under 5%, near one above 95%, mid between.

**Budget.** At most 1.5 GPU-hours, enforced by a timeout in `scripts/run_confirm_mid.sh`
(45 minutes for generation, 45 for judging); both stages append and resume. If the cap cuts a
stage short, the analysis runs on the prompts complete in every role and the report says how
many.

## Predictions

Scored by `confirm_mid.py`'s `report`, as coded before the run. M1 to M8 are about the refusal
label and are untested if it is not a mid-rate label. The preview's value is given with each.

- **M1.** With a large pool the stratified Wald-t limit is over its level in no cell (two
  sizes, two deltas). Preview: 0 of 4.
- **M2.** With a large pool the bootstrap-t StratPPI limit is over its level in at most one of
  its four cells, and its miss exceeds the level by no more than 0.01 in any. Preview: 1 of 4
  over, largest excess 0.003. *Changed by the preview:* the first draft said "in no cell", as
  R5's P1 did; R5 refuted that at 82% and the preview at 48% (0.103 at delta 0.10).
- **M3a.** R5's rule for `b1w` with a large pool, unchanged: over in no cell at a rate under
  0.45, over in at least one of its four cells above 0.55.
- **M3b.** If the rate is between 0.45 and 0.55, where M3a says nothing, the miss of `b1w` is
  within 0.01 of its level in every cell. Preview: rate 0.478, excess 0.001 to 0.002, all four
  cells unresolved. *Added after the preview.*
- **M4.** The gain of `b1w` with a large pool (ESS against the pooled Wilson bound on a random
  draw, from the truth to the limit, delta 0.05) is within 20% of the pre-flight's prediction
  at both safety-set sizes. Preview: 2.81 and 3.04 against 2.93.
- **M5.** The gain of the Wald-t limit with a large pool is at least 1.7 at both sizes (80% of
  the preview's smaller value). Preview: 2.12 and 2.63.
- **M6.** For the prompt source, the Wald-t limit with the sampled-pool term is over in no
  cell. Preview: 0 of 4.
- **M7.** For the prompt source, `b1w` without the term and the bootstrap-t StratPPI limit are
  each over in at least one of the two cells at delta 0.05. Preview: 2 of 2 each.
- **M8.** For the prompt source, the gain of the Wald-t limit with the term is above 1 and
  within 15% of the cap `1 / (1 - G + G n_s / N)` at both sizes (G from the Wald-t limit's
  large-pool gain). Preview: 1.49 against 1.66, 1.35 against 1.45.
- **M9.** On a rare label the strata gain nothing (ESS of `b1w` between 0.9 and 1.15) and
  Clopper-Pearson on a random draw is over in no cell. Preview: 1.00 and 1.08, 0 of 4.

Reported without a prediction: `b1w` with the sampled-pool term for the prompt source (R5
refuted its general prediction), and whether the refusal rate lands between 18% and 65%.

## Reading

Every prediction is reported, kept or refuted, with its counts. M1, M5, M6 and M8 stand behind
the recommendation of the Wald-t limit at a mid rate; a refuted one changes it. M3 places where
`b1w` stops being safe. The pool is one model, one trained policy and two sources of one
benchmark; its prompts are new, its sources are not. A kept prediction here is weaker evidence
than one kept in R5, because the preview on the same two sources came first.

## Not allowed after the run

Changing the pool, labels, class limits, checks, sizes, seeds or the scoring code. A forced
deviation (a crash, the cap) is reported as one, with what was done instead.
