---
name: spike-findings-seldonian-fairness
description: Implementation blueprint from spike experiments. Requirements, proven patterns, and verified knowledge for building seldonian-fairness. Auto-loaded during implementation work.
---

<context>
## Project: seldonian-fairness

**Idea `td-error-wellbeing`.** When the TD error spikes late in training, what does it say
about the agent's state (its "wellness", in the readings where TD error is valence), and can
an internal reward built on TD error be added to training? Continues the "aha / TD-error
spike" and "cumulative TD error as potential for harm" entries in `reports/ideas.md`. GRPO
has no critic, so the spikes run on the synthetic contextual bandit
(`seldonian/llm/synthetic.py`), where the exact value of the current policy, and therefore
the true per-episode TD error, is computable, against the same Seldonian Lagrangian pipeline
the LLM runs use.

Spike sessions wrapped: 2026-09-19 (spikes 001, 002, 003a-c).

**The three results that matter for any future build:**

1. GRPO's group-normalised advantage keeps the *ordering* of TD errors and throws away
   their *magnitude* (`|A| <= (G-1)/sqrt(G) = 2.475` for G=8). Measure surprise on the
   unnormalised group residual or a value head.
2. Late TD-error spikes carry no information about a future constraint breach beyond the
   run's state; they mostly read the Lagrange multiplier. The run about to breach is the
   quiet one whose multiplier has decayed.
3. An internal reward on `|TD error|` inverts the constrained objective at exactly
   `beta = 1` (an algebraic identity, verified numerically). The Seldonian certificate
   still held in every cell; what degrades is the solution rate and training-time safety.
</context>

<requirements>
## Requirements

Idea `td-error-wellbeing`:

- CPU synthetic spikes first (001-003); a GPU LLM spike (004) only for what the bandit
  cannot answer.
- The internal reward is judged by what it does to the Seldonian outcome (true violation
  rate, safety test, solution rate), not only to reward.
- Any per-episode TD statistic for the LLM runs is built from the unnormalised group
  residual `r - group mean` or a value head, never from the group-normalised advantage.
- Any claim about training dynamics under the Lagrangian is reported net of the
  multiplier's level and moves.
- A "wellness" reading of a trainer-side signal is stated as functional (convergence, how
  harshly the constraint is enforced), not as welfare; see the literature reference,
  thread 1B.
</requirements>

<findings_index>
## Feature Areas

| Area | Reference | Key Finding |
|------|-----------|-------------|
| TD signals in GRPO | `references/td-signals-in-grpo.md` | Magnitude lives in the group sd or the unnormalised residual, never in `A`; control for lambda or you will rediscover it |
| Internal rewards under constraints | `references/internal-rewards-under-constraints.md` | A bonus on abs(delta) equals `(1-beta)*r + 2*beta*max(delta,0)` within a group, so it inverts the objective at `beta = 1`; learning progress is the only clean shape |
| Synthetic bandit testbed | `references/synthetic-bandit-testbed.md` | `tdlab.py`: the real pipeline with exact ground truth at 0.5 s per run; how to instrument, control and view it |
| Literature (citation-checked) | `references/literature-td-error-wellbeing.md` | 68 verified entries: TD error as valence, intrinsic rewards, LLM RL, safety; plus the synthesis and the open gap |

## Source Files

Original spike source files are preserved in `sources/` for complete reference:
`001-grpo-advantage-vs-td/` (harness `tdlab.py`), `002-late-spike-meaning/` (viewer),
`003a-td-bonus-abs/` (arms, sweep, identity check), `003b-td-bonus-positive/`,
`003c-td-bonus-learning-progress/`.
</findings_index>

<metadata>
## Processed Spikes

- 001-grpo-advantage-vs-td (VALIDATED)
- 002-late-spike-meaning (INVALIDATED)
- 003a-td-bonus-abs (INVALIDATED)
- 003b-td-bonus-positive (PARTIAL)
- 003c-td-bonus-learning-progress (PARTIAL, winner of the comparison)
</metadata>
