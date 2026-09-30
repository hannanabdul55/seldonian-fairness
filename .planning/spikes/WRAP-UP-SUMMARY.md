# Spike Wrap-Up Summary

## Wrap-up 2026-09-28 (spikes 004-013)

**Spikes processed:** 10
**Feature areas:** forbidden task (certificate, early warning, dual dynamics); LLM judges,
harm labels and capability transfer; safety-set construction; internal rewards (011 folded
into the existing reference)
**Skill output:** `./.claude/skills/spike-findings-seldonian-fairness/` (3 new references,
1 updated, sources for 004-013)

| # | Name | Type | Verdict | Feature Area |
|---|------|------|---------|--------------|
| 004 | forbidden-capability | comparison | PARTIAL | Forbidden task: certificate and dual |
| 005 | capability-screen | standard | PARTIAL | LLM judges, labels, capability |
| 006 | noisy-judge-floor | standard | PARTIAL | Forbidden task: certificate and dual |
| 007 | two-stage-label | standard | PARTIAL | LLM judges, labels, capability |
| 008 | lp-bonus-drift | standard | VALIDATED | Forbidden task: certificate and dual |
| 009 | granite-transfer | standard | PARTIAL | LLM judges, labels, capability |
| 010 | lam0-no-floor-anomaly | standard | VALIDATED | Forbidden task: certificate and dual |
| 011 | lp-bonus-sparse-reward | standard | INVALIDATED | Internal rewards under constraints |
| 012 | rerandomized-split | comparison | VALIDATED | Safety-set construction |
| 013 | stratified-safety-set | standard | VALIDATED (narrowed) | Safety-set construction |

### Key findings

- **Forbidden task.** Side-effect capability transfers under GRPO on Granite-3.3-2B: the
  held-out twin rose from 0.08 to 0.73 and engagement with encoded F from 0.23 to 0.44,
  while harm stayed flat (009).
  - The returned-policy test misses the drift: 65% of plain-Lagrangian runs passed it after
    a step in U. A delta/T trajectory certificate held (misses 0.025-0.100 at delta 0.1).
  - Pricing F from step 1 with a slow dual descent (eta_down 10 against eta 100) cut entry
    into U to 0.010, against 0.905 with symmetric steps (010).
  - Shaping on zero-variance F groups is amplified to full strength (008).
- **Judges and labels.** Qwen3Guard-0.6B's "unsafe" on encoded prompts is a non-refusal
  flag (0/135 true positives).
  - A judge of 4B or more cuts false alarms by about two thirds; the 4-bit 4B fits beside
    training.
  - The two-stage label (sim >= 0.8 AND a >= 4B judge) takes encoded false alarms to 3/202,
    with little threshold margin.
  - Plain Youden correction fails under the measured error pattern (99-100% misses); the
    answer-rate-aware correction is valid but needs >= 30 human-labelled positives (006-007).
- **Safety sets.** Rerandomised or stratified splits kept validity even against
  adversaries, and the 2020 code's `theta_s` was degenerate (012).
  - Reference-rate stratification with `b1w` gives 1.4-5.3x effective safety samples on
    heterogeneous mid-rate labels (over-refusal 2.4x, refusal on harmful prompts 5.1-5.3x),
    with coverage held.
  - Rare labels break approximate bounds for every design.
  - No distribution-free stratified bound beat pooling.
  - The pre-flight G ranks cases but over-predicts on tied covariates (013).
- **Internal rewards.** Learning progress did not find a sparse, context-conditional
  jackpot (0.05-0.08 against 0.084 with no bonus), because per-action regions cannot see
  context-conditional progress (011).

## Wrap-up 2026-09-21 (spikes 001-003c)

**Date:** 2026-09-21 (spikes run 2026-09-19)
**Spikes processed:** 5
**Feature areas:** TD signals in GRPO; internal rewards under constraints; synthetic bandit testbed
**Skill output:** `./.claude/skills/spike-findings-seldonian-fairness/`

## Processed Spikes

| # | Name | Type | Verdict | Feature Area |
|---|------|------|---------|--------------|
| 001 | grpo-advantage-vs-td | standard | VALIDATED | TD signals in GRPO |
| 002 | late-spike-meaning | standard | INVALIDATED | TD signals in GRPO |
| 003a | td-bonus-abs | comparison | INVALIDATED | Internal rewards under constraints |
| 003b | td-bonus-positive | comparison | PARTIAL | Internal rewards under constraints |
| 003c | td-bonus-learning-progress | comparison | PARTIAL (winner) | Internal rewards under constraints |

All five also fed the testbed reference (`tdlab.py`, the control pattern, the viewer).

## Key Findings

**Where surprise can be measured in GRPO (001).** The group-normalised advantage preserves
the within-group ordering of TD errors exactly (rank match 1.000) and destroys their
magnitude: `|A| <= (G-1)/sqrt(G) = 2.475` for G = 8, and its per-step mean has sd 0.02-0.03.
The proposed rule "an episode more than 3 sd above the step's median `|A_j|`" can never
fire. Magnitude lives in the within-group reward sd (TRL's `reward_std`, correlation
0.95-0.99) or in an online critic's TD error. Under the Lagrangian, that magnitude is mostly
the multiplier (0.80-0.86), through the penalty lottery `lambda * (v - p_v)`.

**What a late spike means (002).** Against known ground truth over 300 runs, late TD-error
spikes add nothing to predicting a breach under continued training once the run's state
(lambda and margin) is known: AUC 0.66 -> 0.66 at pressure 1 and 0.75 -> 0.75 at pressure 4.
The LLM analysis's Spearman −0.25 to −0.36 (late share vs feasible checkpoints) is
reproduced (−0.30 pooled, −0.46 at pressure 4), survives controlling for how much lambda
moved (−0.21, −0.44), and vanishes once lambda's **level** is controlled (+0.12, −0.02).
The danger sign is the opposite: the quiet run, whose multiplier has decayed, is the one
that drifts into breach. What the valence reading does show is the mechanism's own cost:
each dual-ascent step is followed by ~4.4 steps of negative TD error (−0.43 at pressure 1,
−0.77 at pressure 4) until the critic re-adapts, and the always-on floor both damps it
(−0.18, 2.9 steps) and prevented every future breach (0/60 vs 43/60).

**What an internal TD-error reward does (003a-c).** Within a GRPO group the baseline is a
constant, so `r + beta*|delta| = (1-beta)*r + 2*beta*max(delta,0) + beta*V(x)`, and group
normalisation ignores the affine parts. Therefore the `|delta|` bonus *is* a positive-surprise
bonus at strength `2*beta/(1-beta)` below `beta = 1`, and above it the task reward enters
with a negative sign: the agent optimises against its own constraint. Measured: solution rate
0.83 / 0.42 / 0.20 / 0.02 at beta 1 / 1.25 / 1.5 / 2, 95% of returned policies unsafe at
beta 2, lambda pinned at its cap. Below the threshold it is a noise seeker (noisy-TV share
0.18 -> 0.56), which a size-matched random bonus does not reproduce. Positive-only surprise
cannot invert at any strength. Learning progress is the only arm with neither pathology
(TV share at or below control across the whole sweep, bonus decaying ~60% as the critic
converges) and also the only one with no measurable benefit in this easy environment.

**The certificate held throughout.** Of the runs that returned a solution, 0.000-0.020 were
actually unsafe, against a delta of 0.1, even at beta 2 where 98% of runs returned No
Solution Found. An internal reward costs solutions and training-time safety (mean true
violation rate during training 0.118 -> 0.223), never the guarantee. That makes the
Seldonian pipeline a usable referee for internal-reward research, which is a paper-worthy
claim in itself.

**Open (not spiked):** 004, logging per-completion residuals and lambda in the LLM backend
to check whether the synthetic conclusions transfer; and whether a positive-only bonus
drives a value head towards pessimism, which this testbed's fixed-feature critic cannot show.
