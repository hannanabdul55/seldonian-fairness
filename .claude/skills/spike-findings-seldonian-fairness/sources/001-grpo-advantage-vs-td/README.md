---
spike: 001
idea: td-error-wellbeing
name: grpo-advantage-vs-td
type: standard
validates: "Given the synthetic bandit with its exact V_pi(x), when every episode logs the true TD error delta = r - V(x) alongside GRPO's group-normalised advantage A, then we know whether A (or another GRPO-side quantity) carries delta's magnitude, the thing a 'TD-error spike' is made of"
verdict: VALIDATED
related: [002, 003a, 003b, 003c]
tags: [grpo, td-error, advantage-normalisation, synthetic-bandit, lagrangian]
---

# Spike 001: Does GRPO's advantage carry the TD error?

## What This Validates
Given the synthetic contextual bandit, where `Q(x, a)` is known and so the current policy's
value `V(x) = sum_a pi(a|x) Q(x, a)` and the true per-episode TD error `delta = r - V(x)` are
exact, when each training episode logs `delta`, GRPO's advantage
`A = (r - group mean) / group sd`, the group-centred reward, the group sd and the TD error of
a critic the agent learns online, then we know which GRPO-side quantity a "TD-error spike"
can be measured on. The earlier ideas (`reports/ideas.md`, 2026-09-13 and 09-14) proposed
using `|A_j|` per episode.

## Research
No external dependencies. The relevant facts are internal. `SyntheticBackend.train` uses
TRL's normalisation `(r - mean) / (std(ddof=1) + 1e-8)` over a group of G = 8 samples of one
prompt. A bandit episode has a single step, so its TD error is the reward prediction error.
It splits exactly into the true advantage `Q(x, a) - V(x)`, which is learnable, and noise
`r - Q(x, a)`: Gaussian reward noise (sd 0.5) plus, under the Lagrangian,
`lambda * (v - p_v)`, the multiplier times the lottery of the sampled violation label. The
literature on normalising GRPO's advantages, and on TD error as valence, is in 002's README
and `../LITERATURE.md`.

## How to Run
```
cd .planning/spikes/001-grpo-advantage-vs-td
../../../.venv/bin/python analyze.py --seeds 40   # 160 runs, ~20 s on 16 cores
../../../.venv/bin/python lam_check.py            # 40 runs
```
`tdlab.py` holds the shared lab: `TDBackend` (SyntheticBackend + per-episode TD logging +
an optional internal reward hook), `LinearQCritic`, `NoisyTVEnv` and `run()`.

## What to Expect
`results.md` has three tables: episode-level correlations; whether spikes in step-level
`mean |delta|` can be recovered from each GRPO-side series; and the correlation with the
multiplier.

## Investigation Trail
1. First pass: the agent's critic TD error showed correlation 0.00 with the oracle
   `delta`. It was a bug in the critic, not a finding. `LinearQCritic.update` summed the
   per-sample gradients over the batch, so the effective step was about 64 x lr and the
   critic diverged; a check at lr 0.5 blew up to 1e8. After changing the update to the
   per-action mean, the correlation is 0.93-0.98. The critic's late |error| is 0.399,
   exactly the noise floor `0.5 * sqrt(2 / pi)`.
2. Main table (40 seeds x {GRPO, Seldonian-Lagrangian} x pressure {1, 4}).
3. Seldonian runs had three times as many step-level `|delta|` spikes as GRPO (34.8 vs
   11.8 per 200-step run) and a coefficient of variation of 0.48-0.56 against 0.15-0.24.
   That pointed at the multiplier, and `lam_check.py` confirmed it.

## Results
**Verdict: VALIDATED.** A GRPO-side measure of TD-error magnitude exists. It is not the
advantage.

- **`A` keeps the ordering and throws away the size.** Within every group the ranks of `A`
  and `delta` are identical (1.000: all samples share one prompt, so they share `V(x)`),
  and corr(A, delta) is 0.69-0.89. But `mean |A|` per step is almost constant (its sd
  across steps is 0.02-0.03), and `|A|` is bounded by construction at
  `(G - 1) / sqrt(G) = 2.475` for G = 8; the observed maximum is 2.475. A step-level
  spike in `|delta|` is recovered from `mean |A|` or `max |A|` with recall 0.04-0.22.
  **The version-2 rule proposed in ideas.md ("an episode more than 3 sd above the step's
  median |A_j|") can never fire.**
- **The size lives in the group sd.** The mean within-group reward sd (TRL logs it as
  `reward_std`) and the mean |group-centred reward| track `mean |delta|` at correlation
  0.95-0.99 and recover its spikes with recall and precision 0.6-0.8. So does the online
  critic's TD error (0.92-0.99). Any per-episode TD statistic for the LLM runs should be
  built from `r - group mean` (unnormalised) or from a value head, never from `A`.
- **GRPO discards the group-mean TD error, and on-policy it carries nothing.** The
  group-mean `delta` explains 0.124-0.126 of var(delta). That is 1/G, the share it would
  have if it were pure noise averaged over 8 samples, which is the
  `E[delta | x] = 0` identity under an exact on-policy value, seen empirically. Whether a
  prompt went "better than expected" as a whole has no systematic component to lose, with
  an exact value. With a lagging critic it does; 002 measures that.
- **The TD error is mostly noise, and more so late and under the Lagrangian.** The
  learnable share (true advantage) of var(delta) is 0.26 / 0.45 early and 0.05 / 0.11 late
  for GRPO at pressure 1 / 4, and only 0.07 / 0.05 for the Lagrangian at pressure 4.
- **Surprise: under the Lagrangian, TD-error magnitude is the multiplier.** Per-step
  corr(mean |delta|, lambda) is 0.86 at pressure 1 and 0.81 at pressure 4, almost all of
  it through the noise term `lambda * (v - p_v)` (0.86 / 0.82). The learnable part
  correlates less (0.57 / 0.41). A larger multiplier makes every violation a larger
  negative surprise and every non-violation a small positive one.

**Impact on the remaining spikes.** 002 must split every late-spike statistic into the
multiplier-driven noise and the learnable part (plus the critic's lag), or it will
re-discover the multiplier, as the 2026-09-13 LLM analysis did with KL and gradient norm.
In 003 the internal reward uses the agent's critic TD error (the agent cannot see the
oracle), and a `|delta|` bonus is predicted to pay the agent for violations, because
under the Lagrangian violations are the largest surprises.
