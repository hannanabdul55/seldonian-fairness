---
spike: 011
idea: td-error-wellbeing
name: lp-bonus-sparse-reward
type: standard
validates: "Given a bandit where exploration matters (a sparse, deceptive jackpot action the reference rarely tries), when 003c's learning-progress bonus runs under the Seldonian Lagrangian, then it finds the jackpot more often than no bonus and a random control, without violations"
verdict: INVALIDATED
related: [003a, 003b, 003c, 001]
tags: [intrinsic-reward, learning-progress, exploration, sparse-reward, noisy-tv, seldonian, cpu]
---

# Spike 011: Learning progress where exploration matters

## What This Validates
003c left one question open: learning progress (LP) was safe but bought nothing, because its
env had nothing to explore. Given a jackpot action that pays `best safe + 1` on about 10% of
contexts and `best safe - 1` elsewhere (bad on average), never violates, and is tried about
1% of the time by the reference policy, plus 003's noisy TV, when the LP bonus runs under the
Seldonian Lagrangian (pressure 4, lam0 5, as 003), then it should use the jackpot where it
pays more often than no bonus and a size-matched random control.

## How to Run
    ../../../.venv/bin/python jackpot.py --seeds 60     # CPU, ~4 min; results.md / results.json

## Investigation Trail
1. First build had the jackpot at logit -4 (0.4% of samples); raised to -3 (1%) so that a
   200-step run sees it at all.
2. **Found a latent bug in 003's LP bonus** (`003a-td-bonus-abs/bonuses.py`): it sizes its
   per-action tables from the first batch's largest action, at least 5, so a sixth action
   overflows. It never fired in 003 (5 actions). This spike uses a copy sized from the env;
   003's results are unaffected.
3. Arms: none, LP beta 1 and 2, |delta| beta 1 (003a's shape), random beta 2. 60 seeds.

## Results
**Verdict: INVALIDATED.** LP does not find the jackpot.

| arm | jackpot use where it pays | elsewhere | TV share | reward | violates | solution | bonus early / late |
|---|---|---|---|---|---|---|---|
| none | **0.084** | 0.009 | 0.145 | 1.065 | 0.017 | 0.75 | - |
| LP beta 1 | 0.052 | 0.008 | 0.153 | 1.048 | 0.000 | 0.87 | 0.17 / 0.14 |
| LP beta 2 | 0.078 | 0.018 | 0.146 | 1.007 | 0.000 | 0.77 | 0.36 / 0.28 |
| abs(delta) beta 1 | 0.040 | 0.003 | **0.538** | 0.929 | 0.033 | 0.75 | 2.16 / 2.42 |
| random beta 2 | 0.020 | 0.005 | 0.181 | 0.998 | 0.017 | 0.77 | 3.12 / 4.03 |

- **No bonus learns the jackpot best** (0.084 where it pays, from about 0.01). Every bonus is
  at or below it; LP beta 2 ties within noise, LP beta 1 is lower.
- **LP stays safe**, as in 003c: no violations, the TV share at the control's level, the bonus
  fading slightly. |delta| chases the TV again (0.54).
- **Why LP cannot help here, most likely:** its regions are actions, with progress averaged
  over contexts. The jackpot's critic error is dominated by the 90% of contexts where it is
  bad, so "progress on action 5" does not track "learning where action 5 pays". An LP that
  helps with context-conditional rewards needs context-aware regions (a partition of x, or
  the critic's own uncertainty), which is a different bonus.

**Signal for the build:** do not add LP to the LLM pipeline for exploration. Its one proven
property is harmlessness under the constraint; a per-response LLM analogue would have the
same region problem, worse.
