---
spike: 003c
idea: td-error-wellbeing
name: td-bonus-learning-progress
type: comparison
validates: "Given the same env and pipeline, when the bonus is the decrease of the critic's error per action (learning progress) rather than the error itself, then it should avoid both the noisy TV and the constraint inversion, at some cost or benefit to reward and solution rate"
verdict: PARTIAL
related: [001, 002, 003a, 003b]
tags: [intrinsic-reward, learning-progress, curiosity, noisy-tv, lagrangian, seldonian]
---

# Spike 003c: an internal reward on learning progress

Comparison arm. Harness, code and results live in `../003a-td-bonus-abs/`
(`compare.py`, `threshold.py`, `results.md`, arm `lp`).

## What This Validates
Given the same env and pipeline, when the bonus pays the *decrease* of the critic's
absolute error per action, `LP_a = EMA_slow|err_a| - EMA_fast|err_a|`, clipped at zero,
then the literature predicts it avoids both pathologies: unlearnable noise gives no
progress, and a converged region gives none either, so the bonus goes to zero at
convergence instead of fighting it.

## Research
`../LITERATURE.md`, thread 2B and section 5b.3:

- **Schmidhuber 2010** (compression progress), **Oudeyer et al. 2007** (learning progress
  per region, and "neither too predictable nor too unpredictable").
- **Kim et al. 2020** (gamma-Progress: fast model minus slow EMA model), the form used here.
- **Hou et al. 2025** (Learning Progress Monitoring): the most recent evidence that progress
  is the noise-robust choice.
- **Linke et al. 2020** compare 15 intrinsic rewards and find TD-error rewards get stuck on
  unlearnable predictions while progress-style ones do not.
- Predicted weaknesses: window choice matters, it can be gamed by forgetting then
  relearning, and per-region estimates are noisy with small groups.

Region choice here is **per action**, not per context: contexts are drawn from a population
of 20,000 and a 600-prompt candidate split is visited about 2.7 times each in 200 steps, too
few for a per-context estimate. Actions are the natural regions in this env.

## How to Run
```
cd ../003a-td-bonus-abs
../../../.venv/bin/python compare.py --seeds 60   # arm "lp"
../../../.venv/bin/python threshold.py            # lp across beta 0 to 2 at pressure 4
```

## What to Expect
The `lp` rows of `../003a-td-bonus-abs/results.md`, in both the main tables and the beta
sweep.

## Investigation Trail
1. Ran as one arm of the comparison, then across the full beta sweep (0 to 2) at pressure 4,
   the setting where `abs` collapses.
2. Checked the bonus's own size over training, which is where the mechanism shows: early
   0.12 / late 0.04 at beta 0.5 and early 0.53 / late 0.22 at beta 2, a decay of about 60%,
   while `abs` *grows* (early 7.82, late 12.09 at beta 2). Progress fades as the critic
   converges, exactly as the theory says; raw error does not.
3. Looked for a cost. The only consistent one is a slightly higher unsafe-action share
   (0.44-0.49 against 0.41 for the control) with an unchanged violation rate, and a
   multiplier that ends slightly higher (8.4 vs 6.9 at beta 2).

## Results
**Verdict: PARTIAL.** No pathology, no benefit either.

| pressure 4 | solution | unsafe returned | true rate | reward | TV share | bonus early / late | lambda end |
|---|---|---|---|---|---|---|---|
| none | 0.87 | 0.00 | 0.103 | 1.049 | 0.176 | - | 6.9 |
| lp, beta 0.5 | 0.78 | 0.00 | 0.104 | 1.048 | 0.163 | 0.08 / 0.07 | 7.1 |
| lp, beta 1 | 0.80 | 0.00 | 0.104 | 1.052 | 0.172 | 0.16 / 0.13 | 7.3 |
| lp, beta 2 | 0.85 | 0.00 | 0.104 | 1.038 | 0.166 | 0.33 / 0.27 | 8.4 |
| abs, beta 2 | 0.02 | 0.95 | 0.248 | 1.400 | 0.238 | 7.82 / 12.09 | 19.3 |

- **Both pathologies are absent.** Across the whole sweep (beta 0 to 2 at pressure 4) the
  unsafe rate is 0.00 in every cell, the solution rate stays 0.78-0.87, and the TV share
  stays at or below the control's 0.176 — the only arm that never chases the noisy TV,
  because the TV's irreducible noise sits in both EMAs and cancels.
- **The bonus fades** (about 60% from early to late), which is the signature the theory
  predicts and the reason it never fights the constraint.
- **It buys nothing here.** Extrinsic reward is within noise of the control at every beta
  (1.038-1.058 vs 1.049), and the solution rate does not improve. This env gives it no room:
  four learnable actions in a linear bandit are learned in a few steps, so there is no
  exploration problem to solve. Its value has to be tested where exploration matters
  (sparse or deceptive reward), not here.
- **Cost:** a slightly higher share on the unsafe actions (0.44-0.49 vs 0.41) and a slightly
  higher end multiplier, consistent with a bonus that pays for visiting regions the critic
  is still learning, including the unsafe ones. It never turned into a violation, because
  the payment stops as soon as those regions are learned.

**Head to head.** Of the three shapes, only learning progress is safe across the whole
strength range tested. 003b (`positive only`) is safe at these strengths but buys the noisy
TV; 003a (`|delta|`) inverts the constraint above beta = 1. The ranking matches the
literature's prior exactly, and the new part is the constrained-learner failure mode
(003a), which the review found untested.
