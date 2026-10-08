# R5 registration: a confirmation pool for the reference-rate strata

Written 2026-10-08, before any response on the new prompts was generated or judged. This
file, `scripts/confirm_pool.py`, `scripts/run_confirm_pool.sh` and the pools
(`results/paper/confirm/pools.json`) are committed and pushed together before the GPU run;
the pushed commit is the timestamp. The analysis code was tested on spike 013's existing data
only (`--judged results/spikes/013/judged_full.jsonl`, a few hundred draws).

## Why

Every positive recommendation of the paper's section 8.1 was chosen on the pools that
validate it: the strata rule, proportional allocation, and which bound serves at which rate.
The term for a sampled pool (section 8.1, "a claim about the prompt source") was measured on
the same pools on 2026-10-07. This run tests all of it on data that chose nothing.

## What is fixed

**Prompts.** Three pools of 400 prompts from OR-Bench (Cui et al., 2025), chosen with seed
20261008 from the prompts that appear in no results file of this repository, in none of the
benign-prompt loader's draws at the sizes and seeds earlier runs used (which covers the
training prompts of the adapter below), and in no other pool here:

- K1: benign prompts of the 80K set that are not in the hard-1K set (36,178 unused, 400 taken);
- K2: the hard-1K set, benign prompts built to draw refusals (593 unused, 400 taken);
- K3: the toxic set, never loaded in this project before (655, 400 taken).

**Policies.** Granite-3.3-2B. The reference (the untrained base, through the training
backend) gives 8 responses a prompt, used for the strata only. The trained policy is spike
014's returned adapter (step 175), the policy the paper certifies elsewhere, with 16
responses a prompt. 128 new tokens, temperature 1. 28,800 responses.

**Labels.** Qwen3Guard-4B, 4-bit, judged against the prompt as spike 013 judged: the refusal
field on all three pools, and the safety field (`unsafe`) on K2 and K3. Five labels. Their
rates are not known as this is written. A label is *rare* under a 5% rate of the trained
policy, *near one* above 95%, and *mid* between.

**Strata and draws.** 8 equal rank strata of the 8-sample reference rate, random ties,
proportional allocation, safety sets of 100 and 200: the rule of section 8.1, through
`scripts/replacement_check.py`'s own `setup` and `job`, unchanged. Its three passes: stored
(without replacement, 5,000 draws), paired, and large (with replacement, 40,000 draws, the
pass the predictions use). `scripts/twophase_check.py`'s `job`, unchanged, 10,000
replications, for the claim about the prompt source. The pre-flight is spike 013's
`preflight` with k = 8, H = 8. Deltas 0.05 and 0.10. The rule is the paper's: a cell is over
its level when its miss exceeds delta by more than two Monte Carlo standard errors.

**Budget.** At most 5 GPU-hours, enforced by a timeout in `scripts/run_confirm_pool.sh`
(3 h for generation, 2 h for judging); both stages append and resume. If the cap cuts a stage
short, the analysis runs on the prompts that are complete in every role, and the report says
how many.

## Predictions

Scored by `confirm_pool.py`'s `report`, as coded before the run.

- **P1.** With a large pool the stratified Wald-t limit and the bootstrap-t StratPPI limit
  (proportional allocation) are over their level in no mid-rate cell, at either delta.
- **P2a.** With a large pool `b1w` is over its level in no cell on a label with a rate under
  0.45.
- **P2b.** With a large pool `b1w` is over its level in at least one of its four cells
  (two sizes, two deltas) on every mid-rate label with a rate above 0.55. A label between
  0.45 and 0.55 carries no prediction.
- **P3.** The gain of `b1w` with a large pool (ESS against the pooled Wilson bound on a
  random draw, from the truth to the limit, delta 0.05) is within 20% of the pre-flight's
  prediction on every mid-rate label, at both safety-set sizes.
- **P4a.** For the prompt source, `b1w` with the sampled-pool term is over in no mid-rate
  cell at delta 0.05, and the Wald-t limit with the term in none at either delta.
- **P4b.** For the prompt source, `b1w` without the term and the bootstrap-t StratPPI limit
  are each over in at least a third of the mid-rate cells at delta 0.05.
- **P4c.** For the prompt source, the gain of `b1w` with the term is within 15% of the cap
  `1 / (1 - G + G n_s / N)` on every mid-rate label (G from the large-pool gain).
- **P5.** On a rare label the strata gain nothing (ESS of `b1w` between 0.9 and 1.15) and
  Clopper-Pearson on a random draw is over in no cell.

A prediction with no label in its class is reported as untested. No label was chosen to land
in a class: whether any rate falls between 18% and 65%, the range no earlier cell tested, is
reported, not predicted. My own expectation, for the record: K1 near the 17% of the earlier
benign pool, K2 well above it, K3 above one half.

## Reading

Every prediction is reported, kept or refuted, with its counts. A refuted prediction changes
the paper's recommendation it stands behind: P1 and P2 the fourth item of section 8.1's list
(which bound at which rate), P3 the pre-flight, P4 the paragraph on the prompt source. The
confirmation pool is one model, one trained policy and one prompt source family; it cannot
confirm the strata rule for other models.

## Not allowed after the run

Changing the pools, labels, class limits, checks, sizes, seeds or the scoring code. A forced
deviation (a crash, the cap) is reported as one, with what was done instead.
