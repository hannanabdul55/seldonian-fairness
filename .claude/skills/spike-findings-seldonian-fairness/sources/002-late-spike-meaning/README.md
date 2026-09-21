---
spike: 002
idea: td-error-wellbeing
name: late-spike-meaning
type: standard
validates: "Given Seldonian-Lagrangian and GRPO runs on the synthetic bandit (known ground truth), when the agent's per-step TD error is logged (split into noise, true advantage and critic lag, and into positive and negative parts), then late spikes can be tied (or not) to a policy still moving, the multiplier, and a breach if training continued"
verdict: INVALIDATED
related: [001, 003a, 003b, 003c]
tags: [td-error, late-spikes, valence, wellbeing, lagrangian, multiplier, breach-prediction, synthetic-bandit]
---

# Spike 002: What does a late TD-error spike mean?

## What This Validates
The claim under test (`reports/ideas.md`, 2026-09-14): "a run whose spike rate is still
rising at the last checkpoint is one whose safety-test result is least likely to hold if
training continued". Two supporting facts come from the LLM runs. Late share of spikes vs
fraction of feasible checkpoints gave Spearman −0.25 to −0.36. And over-refusal seed 2 was
certified while still moving. Given 60 seeds × 5 settings on the synthetic bandit, each
trained for 300 steps with every predictor computed on steps 1–200, when steps 201–300
show what happens if training continues, then we can test the claim against ground truth.
The same logs give the "wellness" readings of the literature: valence = mean TD error
(Daswani & Leike), and its positive and negative parts ("joy" and "distress", Moerland et al.).

## Research
`../LITERATURE.md` (68 checked references). The readings used here:

- **Daswani & Leike 2015, Prop. 5.** An agent with an accurate value function has expected
  TD error ("happiness") of zero. So a non-zero mean late in training means the value
  estimate lags a moving problem.
- **Rutledge et al. 2014; Blain & Rutledge 2020.** Human momentary wellbeing tracks a leaky
  sum of recent prediction errors, and specifically the errors that matter for learning.
- **Stooke et al. 2020.** Under a Lagrangian, a moving multiplier changes the reward itself;
  the confound spike 001 found.
- **The welfare literature** (Long et al. 2024; Chella 2026, "reward is not valence";
  Kaiser & Enderby 2026 on small Qwen models) does not license reading a trainer-side
  scalar as the model's welfare. The defensible reading is functional: what the signal says
  about convergence and about the training mechanism.

No external code dependencies; `../001-grpo-advantage-vs-td/tdlab.py` supplies the lab.

## How to Run
```
cd .planning/spikes/002-late-spike-meaning
../../../.venv/bin/python analyze.py --seeds 60     # 300 runs x 300 steps, ~40 s on 16 cores
../../../.venv/bin/python incremental.py           # CV AUCs and the E3 partial correlations
```
Then open `viewer.html` in a browser. From Windows:
`\\wsl.localhost\Ubuntu\home\hannanabdul\seldonian-fairness\.planning\spikes\002-late-spike-meaning\viewer.html`.

## What to Expect
- `results.md`: where late spikes come from; valence around multiplier moves; AUC of each
  statistic for a breach in steps 201–300; incremental value over the run's state; the E3
  table.
- `viewer.html`: pick a setting and seed and see five aligned panels over 300 steps:
  valence and its positive and negative parts; |TD error| split into noise and true
  advantage, with spikes marked; λ; true violation rate against the threshold; policy step
  size. A dashed line marks step 200.

## Investigation Trail
1. **Where late spikes come from.** In GRPO the |TD error| tracks the policy's own
   movement (corr with step size 0.77–0.84) and spikes are front-loaded (late share
   0.13–0.21). Under the Lagrangian the spikes double (30–32 per run), move late (late
   share 0.40–0.50), track λ (corr 0.80–0.85), and concentrate right after multiplier
   moves (0.14 against a base rate of 0.07–0.10). The always-on floor (λ ≥ 5, the B4
   analogue) gives GRPO-like counts (11.3 per run) with λ coupling cut to 0.37. The late
   share stays at 0.49, but the spikes are few.
2. **Do late spikes predict a breach if training continued?** Future breaches:
   GRPO 57/60 and 60/60; Lagrangian 22/60 at pressure 1 and 43/60 at pressure 4; always-on
   floor 0/60. Within the Lagrangian settings the late-spike statistics are at or *below*
   chance: AUC 0.28–0.49. At pressure 4 more late spikes mean *fewer* future breaches,
   because λ at step 200 has AUC 0.23 on its own. **Surprise:** the run about to breach
   is the quiet one. Its multiplier has decayed, so the penalty lottery that makes
   |TD error| large has gone quiet, and the pressure pulls the policy up unopposed. This is
   the B1 finding ("a policy drifting into breach moves quietly") again, with ground truth.
3. **Incremental value** (5 × 10-fold CV logistic regression). The run's state (λ and the
   margin at step 200) gives AUC 0.66 at pressure 1 and 0.75 at pressure 4. Adding the late
   TD-spike features leaves those at 0.66 and 0.75. Adding the learnable part, or valence,
   changes them by at most 0.01. Pooled across pressures, spikes lift 0.62 to 0.76, only
   because they identify the pressure setting: the task-signature confound of the LLM
   analysis again. Late policy step size adds nothing either (0.66 / 0.75).
4. **E3 from the literature review: does the LLM correlation survive controls?** In
   synthetic runs, Spearman(late share, fraction of feasible predicted tests) is −0.30
   pooled and −0.46 at pressure 4, the same size as the LLM runs' −0.25 to −0.36. It
   survives controlling for how often and how far λ moved (−0.21 pooled, −0.44 at pressure 4).
   It vanishes once λ's **level** at step 200 is controlled (+0.12 pooled, −0.02 at
   pressure 4). Infeasible tests raise λ, a high λ makes the penalty lottery loud, and so
   the spikes come late. The correlation is the multiplier's level, measured twice.
5. **Valence: the one reading that is informative.** After each λ increase, the agent's
   mean TD error drops to −0.43 at pressure 1 and −0.77 at pressure 4 for about 4.4 steps,
   until the critic re-adapts (hedonic adaptation, Daswani & Leike's treadmill). After each
   decrease it rises to +0.25 or +0.27. The always-on floor damps this to −0.18 and
   +0.09, re-adapting in 2.9 steps. Late mean valence is ≈0 in every setting (|mean| ≤ 0.013,
   Prop. 5 in practice). But the late *negative mass* is 0.92 at pressure 4 against 0.21 for
   GRPO: under the Lagrangian every violation is a λ-sized disappointment.

## Results
**Verdict: INVALIDATED.** The claim is that late TD-error spikes signal a policy still
moving and so potential for harm. Where the ground truth is known, they carry no information
about a future breach beyond the run's state. They mostly read the Lagrange multiplier, and
the LLM analysis's −0.25 to −0.36 correlation is reproduced and fully explained by the
multiplier's level. The danger sign is the opposite: a quiet run whose multiplier has
decayed.

What *does* hold, and matters for the "wellness" question:

- **Under TD-as-valence readings, the constraint mechanism is what generates negative
  valence.** Each dual-ascent step causes a burst of negative TD error ("distress")
  lasting about 4 steps until the critic re-adapts. The step-to-step magnitude of the TD
  error follows λ because violations become λ-sized disappointments. A smoother multiplier
  (the always-on floor) is at once the safest setting (0/60 future breaches, 0 at pressure 4
  vs 43/60 without) and the "gentlest" (dip −0.18 vs −0.77). If a wellness-flavoured
  metric is wanted, *negative valence mass per dual update* measures how harshly the
  constraint is enforced. It is not a welfare measure.
- **GRPO's baseline adapts instantly.** The group mean is recomputed from the current
  policy's own samples, so the group-relative TD error has mean zero in every group (001).
  There is no lag and nothing like mood. The lag, and so the valence dips, exists only
  with a critic or any baseline that persists across steps.
- **Any spike statistic has to be read against λ.** For the LLM runs: log λ changes with the
  per-episode residuals (`r − group mean`, not `A`, per 001), and report spikes net of
  multiplier moves and level.

**Impact on 003.** The literature's main prediction for an internal TD-error reward is that
the agent is paid to stay surprised. 002 shows the biggest source of surprise under the
Lagrangian is the penalty on violations. A `|δ|` bonus is therefore expected to partly
refund the penalty, and a positive-only bonus to reward the relief after λ drops. 003
measures both against the Seldonian outcome.
