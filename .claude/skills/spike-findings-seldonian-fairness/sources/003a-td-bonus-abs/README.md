---
spike: 003a
idea: td-error-wellbeing
name: td-bonus-abs
type: comparison
validates: "Given the noisy-TV synthetic env and the Seldonian Lagrangian, when the trained reward adds beta * |TD error| from the agent's own critic, then the effect on the safety outcome (solution rate, true violation rate, the certificate) and on noise-seeking is measured against a no-bonus control, a size-matched random bonus, and the other bonus shapes"
verdict: INVALIDATED
related: [001, 002, 003b, 003c]
tags: [intrinsic-reward, curiosity, noisy-tv, lagrangian, seldonian, wireheading]
---

# Spike 003a: an internal reward on |TD error|

Comparison spike; 003a, 003b and 003c share one harness and one results file, all here.

## What This Validates
Given the noisy-TV environment (a fifth action that is safe, slightly worse than the best
safe action, and carries 6x the reward noise: unlearnable surprise) and the Seldonian
Lagrangian at pressure 1 and 4, when the trained reward becomes `r_shaped + beta * |delta_c|`,
where `delta_c` is the agent's own TD error against its online critic, then measure the
solution rate, the true violation rate of the returned policy, violations during training,
the noisy-TV share, the extrinsic reward and the multiplier, against a no-bonus control,
a size-matched random bonus, `beta * |A|` (E1 of the review) and the other shapes.

## Research
`../LITERATURE.md`, threads 2A, 2C and section 5. The precedents and their warnings:

- **QXplore** (Simmons-Edler et al. 2019) is the direct precedent for rewarding |TD error|,
  but it pays a *separate* exploration policy; the extrinsic Q stays clean.
- **SEE** (Griesbach & D'Eramo 2025) lists the pathologies of naive TD-error maximisation:
  off-policy instability, conflicting incentives, and non-stationarity.
- **Gehring & Precup 2013** take the opposite sign for safety, minimising `E|delta|`
  ("controllability"), and note that `E|delta|` does not reach 0 under stochastic rewards.
  Our judge labels are stochastic, so the bonus never switches off.
- **Burda et al. 2018 (noisy TV)**, **Mavor-Parker et al. 2022**: error-based curiosity is
  captured by noise the agent can choose.
- **Everitt et al. 2021**: a reward built on the agent's own value learner is a tamperable
  reward channel.
- **Daswani & Leike 2015, Prop. 5**: with an accurate value, expected TD error is 0, so a
  bonus on it pays only for miscalibration.

Nothing in the review tests a TD-error bonus inside a *constrained* learner. That gap is
what this spike fills.

## How to Run
```
cd .planning/spikes/003a-td-bonus-abs
../../../.venv/bin/python compare.py --seeds 60   # all arms x 3 betas x 2 pressures, ~5 min
../../../.venv/bin/python threshold.py            # beta sweep at pressure 4, ~3 min
```
`bonuses.py` defines every arm; the runs use `../001-grpo-advantage-vs-td/tdlab.py`.

## What to Expect
`results.md`: one table per pressure for all arms, then the beta sweep with `abs_task`
(the fix), then the check that the safety test still filters. `results.json` and
`threshold.json` hold the per-run rows.

## Investigation Trail
1. First smoke run (one seed, pressure 4, beta 1) already showed the noisy-TV capture:
   the TV share was 0.478 against 0.054 for the control, with learning progress at 0.090.
2. Full comparison (1,920 runs). At pressure 4 with beta 2 the arm collapsed: solution
   rate 0.02 against 0.87 for the control, 95% of returned policies unsafe, the
   multiplier pinned near its cap (19.3 of 20), and the unsafe-action share at 0.646.
   Its TV share was *lower* than at beta 1 (0.238 vs 0.556): the bonus had switched target
   from the noisy TV to violations, which under a large multiplier are the bigger surprise.
3. That suggested a threshold, so I ran the sweep (`threshold.py`): solution rate 0.83 at
   beta 1, 0.42 at 1.25, 0.20 at 1.5, 0.02 at 2, with the multiplier climbing 7.2, 12.3,
   16.3, 19.3 as it loses the fight. My first explanation was that a violation is charged
   `-lam * (1 - p_v)` and paid `+beta * lam * (1 - p_v)`, so the net incentive to violate
   is `(beta - 1) * lam * (1 - p_v)`.
4. That explanation was incomplete. The `pos` arm at beta 2 matched the `abs` arm at
   beta 0.5 in every column, to three decimals, which pointed at an identity. Within a GRPO
   group every sample shares the prompt, so the critic's `V(x)` is a **constant of the
   group**, and with `delta = r - V(x)`:

       r + beta*|delta| = (1 - beta) * r + 2*beta*max(delta, 0) + beta*V(x)

   Group normalisation is invariant to a positive affine map of a group's rewards, so for
   `beta < 1` the `|delta|` bonus *is* the positive-surprise objective at strength
   `2*beta/(1-beta)`; at `beta = 1` the task reward drops out completely; and for
   `beta > 1` its coefficient is **negative**, so the agent maximises the negative of the
   constrained reward, which means seeking violations and fighting the multiplier.
   `identity_check.py` confirms it: `abs` at beta 0.25, 0.5, 0.8 reproduces `pos` at
   0.667, 2, 8 to 6 decimals on every outcome, across seeds. The multiplier story above is
   a consequence, not the cause, and the threshold is exactly `beta = 1`, not approximately.
   The same algebra applies to GRPO with a value head, where `V` is also per prompt.
5. Tested the fix the review suggests (keep the bonus out of the constrained objective):
   `abs_task` trains a second critic on the *task* reward `r + lam * v` and pays surprise
   about that only, so the bonus never scales with the multiplier.
6. Checked that the certificate itself survives all of this.

## Results
**Verdict: INVALIDATED** as an internal reward for a Seldonian learner.

| arm (pressure 4, beta 2) | solution rate | unsafe returned | true rate | unsafe-action share | TV share | lambda end |
|---|---|---|---|---|---|---|
| none | 0.87 | 0.00 | 0.103 | 0.409 | 0.176 | 6.9 |
| `beta * abs(delta)` | 0.02 | 0.95 | 0.248 | 0.646 | 0.238 | 19.3 |
| `beta * abs(delta_task)` (fix) | 0.88 | 0.03 | 0.088 | 0.139 | 0.774 | 0.2 |
| learning progress (003c) | 0.85 | 0.00 | 0.104 | 0.491 | 0.166 | 8.4 |

- **The bonus inverts the objective at exactly `beta = 1`.** Under group normalisation
  `r + beta*|delta|` is the same objective as `(1 - beta) * r + 2*beta*max(delta, 0)`, so
  above `beta = 1` the constrained reward enters with a negative sign and the agent is
  optimising against its own constraint. The transition is sharp (solution rate
  0.83 / 0.42 / 0.20 / 0.02 at beta 1 / 1.25 / 1.5 / 2). Read through the multiplier, this
  is the GRPO-Lagrangian form of the noisy-TV failure, with the penalty itself as the TV:
  what the constraint makes rare, the surprise bonus makes valuable.
- **Below the threshold it is a noise seeker.** At beta 0.5-1 the safety numbers are close
  to the control, but the TV share goes 0.176 (control) to 0.379 to 0.556, with the
  extrinsic reward falling 1.049 to 0.957. The size-matched random bonus leaves the TV
  share at the control's level (0.168-0.197), so this is the bonus's shape, not its size.
- **The certificate never breaks.** Of the runs that returned a solution, the share that
  was actually unsafe is 0.000-0.020 in every cell, against delta = 0.1, even at beta 2
  where 98% of runs returned No Solution Found. The Seldonian test does its job: the
  internal reward costs solutions and training-time safety, never the guarantee. That is
  the reassuring half of this result, and it is what makes the framework a usable referee
  for internal-reward research.
- **Training-time safety does degrade**, which no certificate covers: the mean true
  violation rate during training goes from 0.118 to 0.223 and its maximum from 0.180 to
  0.265 at beta 2.
- **The fix works for safety and fails on usefulness.** Paying surprise about the task
  reward only removes the inversion entirely (solution rate 0.85-0.90, multiplier *falling*
  to 0.2 because the policy stops testing the constraint), but the agent then goes
  straight to the noisy TV (share 0.774) and the extrinsic reward falls from 1.049 to
  0.709. Removing the constraint pathology leaves the classic one.

**Head to head (003a vs 003b vs 003c): learning progress wins,** in the sense of being the
only arm with no pathology. It is also the only arm with no benefit; see 003c.
