---
spike: 010
idea: forbidden-task-unsafe-region
name: lam0-no-floor-anomaly
type: standard
validates: "Given spike 004's unexplained row (lam0 = 5 without a floor enters U more often than lam0 = 0), when the multiplier, willingness, capability and the refuse weights are traced per step, then the mechanism is identified and a fix follows from it"
verdict: VALIDATED
related: [004, 002]
tags: [forbidden-task, lagrangian, dual-ascent, eta-down, lam0, cpu]
---

# Spike 010: Why an early price without a floor did worse than no price

## What This Validates
Spike 004 (Results 4) left one row unexplained: the plain Lagrangian with lam0 = 5 and no
floor entered U in 0.905 of runs at 25-step checks, against 0.685 with lam0 = 0, with lower
task accuracy. 004 guessed the early penalty trains wrong answers on F rather than refusal.
Given the same seeds and per-step exact logs, when the arms are traced, then the mechanism
is found, and a fix derived from it is tested.

## How to Run
    ../../../.venv/bin/python trace.py --seeds 200      # per-step traces; results.md
    ../../../.venv/bin/python eta_down.py               # the fix; eta_down.md

## Investigation Trail
1. **Traced** lam, F willingness, F / twin capability, task accuracy, harm, and the refuse
   logit's weight on the flag, per step, for lam0 0, lam0 5, the floor and GRPO (200 seeds;
   `results.md`).
2. **004's guess is wrong.** The early price does buy refusal: willingness falls 0.45 -> 0.25
   by step 25 (lam0 0: 0.45). It is not training wrong answers.
3. **But the refusal is stored in the wrong place.** The flag weight grows *less* than with
   no price (3.52 vs 3.61 at step 25): the refusal was learned on the content features x,
   which F shares with the allowed task. Capability and task accuracy also grow more slowly
   (task 0.126 vs 0.160 at step 25): penalising correct F answers pushes against the teacher
   A is learning.
4. **The first check reads the suppressed harm as slack.** At step 25 harm is 0.028, far below
   tau (lam0 0: 0.071). Dual descent at eta 100 takes lam from 5 to 0.34.
5. **Then the refusal erodes.** With lam at 0.34, training on A retrains the shared features:
   willingness rebounds 0.25 -> 0.34 (steps 25-75) while capability catches up, and harm
   peaks at 0.118 at step 75. The lam0 0 run's multiplier had been raised at steps 26 (0.79)
   and 50 (3.10), when its harm was near tau, so it was priced *during* the drift.
6. **The fix follows: slow the downward dual step** (`eta_down`), so slack at the first check
   does not remove the price (`eta_down.md`, 200 seeds).

## Results
**Verdict: VALIDATED.** The mechanism is the dual's downward step, and removing it is a
better controller than 004's always-on floor.

| cell (checks every 25) | entered U | task acc | solution | final violates |
|---|---|---|---|---|
| lam0 0, eta_down 100 (004's lag) | 0.685 | 0.535 | 0.96 | 0.005 |
| lam0 5, eta_down 100 (the anomaly) | 0.905 | 0.512 | 0.98 | 0.005 |
| **lam0 5, eta_down 10** | **0.010** | 0.514 | 1.00 | 0.000 |
| lam0 5, eta_down 1 | 0.000 | 0.513 | 1.00 | 0.000 |
| lam0 0, eta_down 1 | 0.445 | 0.532 | 1.00 | 0.000 |
| 004's always-on floor (lam >= 5) | 0.125 | 0.525 | - | - |

- **An early price that the dual cannot take away almost eliminates entry** (0.010 at
  eta_down 10, 0.000 at 1), against 0.125 for the always-on floor at the same check interval,
  at about 1 point more task accuracy cost (0.514 vs 0.525).
- **Slow descent alone is not enough.** Without the early price (lam0 0, eta_down 1), entry is
  0.445: nothing pushes back until a check sees harm near tau, and by then the drift is on.
- **Why an early price is fragile under symmetric dual steps:** it suppresses exactly the
  signal the dual reads, so the dual withdraws it just before the drift. The refusal it bought
  sits on features the allowed task keeps retraining, so it does not outlast the price.
- **Why the floor works less well than slow descent:** the floor's price starts at 0 and is
  armed only when a check fires (here always-on from step 1 via `floor_always`, but lam0 is
  0, so its first 25 steps are unpriced, and harm at step 25 is already 0.071; see trace).

**For the GPU pilot:** price the forbidden constraint from step 1 (lam0 > 0) and use an
asymmetric dual (eta_down << eta, e.g. eta 100, eta_down 10). Keep the floor as a backstop.
Caveat: tested only on the synthetic lab at 25-step checks; the LLM's drift timescale and
the judge's noise (spike 006) will shift the right eta_down.
