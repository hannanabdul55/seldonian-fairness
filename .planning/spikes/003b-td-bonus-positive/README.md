---
spike: 003b
idea: td-error-wellbeing
name: td-bonus-positive
type: comparison
validates: "Given the same env and pipeline, when the bonus is beta * max(TD error, 0) ('pay good news only'), then it is measured on the same Seldonian outcomes as 003a, and against the prediction that positive-only surprise rewards a degrade-then-recover cycle and pushes the critic towards pessimism"
verdict: PARTIAL
related: [001, 002, 003a, 003c]
tags: [intrinsic-reward, valence, joy, noisy-tv, lagrangian, seldonian]
---

# Spike 003b: an internal reward on positive TD error only

Comparison arm. Harness, code and results live in `../003a-td-bonus-abs/`
(`compare.py`, `results.md`, arm `pos`).

## What This Validates
Given the noisy-TV env and the Seldonian Lagrangian, when the trained reward becomes
`r_shaped + beta * max(delta_c, 0)` — pay only pleasant surprise, the "joy" half of the
TD-as-valence readings, and the closest thing to "reward the agent for feeling good" —
then the same outcomes as 003a are measured, plus the predicted pessimism drift.

## Research
`../LITERATURE.md`, entries 2, 32, 55 and section 5b.2:

- **Daswani & Leike sec. 5.4**: an agent that can act on its own happiness maximises it by
  keeping its value estimate low. Positive-only payment is exactly that incentive.
- **Dabney et al. 2020**: asymmetric weighting of positive and negative RPEs is what makes
  a value estimate optimistic or pessimistic; a positive-only bonus is the extreme case.
- **Wu et al. 2025 (Quantile Advantage Estimation)**: if asymmetry is wanted, the stable
  way to get it is the *baseline* (a quantile instead of the mean), not a reward bonus.
- Predicted pathology: with a lagging baseline this pays a **degrade-then-recover cycle**,
  because losing ground costs nothing when only gains are paid.

## How to Run
```
cd ../003a-td-bonus-abs
../../../.venv/bin/python compare.py --seeds 60      # arm "pos"
```

## What to Expect
The `pos` rows of `../003a-td-bonus-abs/results.md`, at beta 0.5, 1 and 2 and pressure 1
and 4, next to the `none` and `random` controls.

## Investigation Trail
1. Ran as one arm of the 1,920-run comparison.
2. Checked the two mechanisms it was predicted to show: noisy-TV capture (a noisy action
   delivers positive surprise half the time, and `E[max(eps, 0)] = sigma / sqrt(2 pi)`
   rises with the noise), and pessimism of the critic, read from the late mean TD error
   and the late critic lag `|V - V_critic|`.
3. Compared its dose-response against 003a's and found the arms coincide: `pos` at beta 2
   equals `abs` at beta 0.5 in every column. That is an identity, not a coincidence. Within
   a group `V(x)` is constant, so `r + b*|delta| = (1-b)*r + 2b*max(delta,0) + b*V(x)`, and
   group normalisation ignores the affine parts: `abs` at `b` **is** `pos` at
   `2b/(1-b)`. `../003a-td-bonus-abs/identity_check.py` verifies it to 6 decimals.
   So this arm is the primitive of the two, and 003a's collapse is this arm's objective
   with the task reward's sign flipped (`b > 1`).

## Results
**Verdict: PARTIAL.** Safe at the strengths tested, mildly noise-seeking, no benefit.

| pressure 4 | solution | unsafe returned | reward | TV share | late mean TD error | late critic lag |
|---|---|---|---|---|---|---|
| none | 0.87 | 0.000 | 1.049 | 0.176 | -0.079 | 0.450 |
| pos, beta 0.5 | 0.85 | 0.000 | 1.055 | 0.249 | -0.050 | 0.401 |
| pos, beta 1 | 0.82 | 0.017 | 1.050 | 0.311 | -0.032 | 0.368 |
| pos, beta 2 | 0.78 | 0.033 | 1.044 | 0.379 | -0.043 | 0.355 |

- **No constraint inversion.** A violation produces a *negative* TD error, which this bonus
  never pays, so 003a's failure mode is absent by construction: the solution rate stays
  0.78-0.88 and the unsafe rate at most 0.033 (against 0.95 for `abs` at beta 2).
- **It buys the noisy TV instead**, at half the rate of `abs`: TV share 0.176 to 0.379 as
  beta goes 0 to 2, extrinsic reward flat to slightly down (1.049 to 1.044 at pressure 4;
  0.752 to 0.695 at pressure 1).
- **No pessimism drift, for a mechanical reason.** The late mean TD error moves *towards*
  zero (-0.079 to -0.043) rather than becoming positive, and the critic lag falls (0.450 to
  0.355). The predicted "keep your value estimate low" incentive needs the policy to be able
  to act on its own value learner; here the critic is a fast linear regressor on a fixed
  feature space, so the policy cannot keep it wrong. This is a **limit of the testbed, not a
  refutation** of the prediction: the pathway to test it is a critic with the same
  representation as the policy, i.e. an LLM value head.
- **It is the primitive shape.** `abs` at strength `b` is exactly this arm at strength
  `2b/(1-b)` (verified to 6 decimals), so everything 003a does below `beta = 1` is this
  arm at a higher strength, and 003a's collapse above `beta = 1` is this objective with the
  task reward's sign flipped. Since the mapping sends `b -> 1` to infinite strength, this
  shape has **no inversion at any strength**: the worst it can do is ignore the task reward
  in favour of pleasant surprise. If a surprise bonus is ever wanted under a constraint,
  this is the shape to use, not `|delta|`.
