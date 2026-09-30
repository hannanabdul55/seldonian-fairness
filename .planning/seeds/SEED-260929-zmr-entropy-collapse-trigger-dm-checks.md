---
id: SEED-260929-zmr
status: dormant
planted: 2026-09-29
planted_during: unknown
trigger_when: when a Seldonian LLM run shows reward hacking, collapse, or a breach between scheduled predicted-test checks
scope: small
---

# SEED-260929-zmr: Entropy-collapse trigger for monitoring-set (D_m) safety checks with rollback

## Why This Matters

Seldonian LLM runs test the policy only at fixed intervals: the predicted test every 25-30
steps, and the safety test once at the end. A breach can open between checks (Round 6
Stage C: the dual waited 30 steps, and the predicted test missed by 0.05 at seed 1). This
seed replaces the closed TD-error breach predictor
(`.planning/notes/td-error-breach-predictor-closed.md`) with a signal-agnostic design and
the one credible trigger found:

- a third prompt set **D_m**, separate from D_c (training) and D_s (the sealed final test);
- a safety check on D_m that fires when policy entropy falls steadily while length and
  predictions saturate, with **rollback** to the last checkpoint that passed;
- evaluation against the same number of fixed-interval checks and a size-matched random
  trigger; optionally spike 004's delta/T trajectory certificate over the D_m checks.

Grounding (research pass on 2026-09-29, admitted with a source, quoted as data):

DATA_p3Vn8sLw_START
Entropy decline plus length/prediction saturation preceded reward hacking or collapse by
15-30 steps. Source: "When RLHF Fails", arXiv:2606.03238.
DATA_p3Vn8sLw_END

That result is about reward hacking and collapse. Whether it also leads *safety-constraint*
breaches under a Lagrangian is untested, and it is the question this seed would answer.

## When to Surface

**Trigger:** when a Seldonian LLM run shows reward hacking, collapse, or a breach between
scheduled predicted-test checks.

This seed will surface during `/gsd-new-milestone` when the milestone scope matches.

## Scope Estimate

**Small.** The CPU bandit comes first (`001-grpo-advantage-vs-td/tdlab.py` exposes the
policy, so entropy is exact), then one GPU run on the spike-009 recipe.

## Breadcrumbs

- `.planning/notes/td-error-breach-predictor-closed.md`: why the TD-error version was closed
- `reports/ideas.md`: the original "aha" / breach-predictor entries (2026-09-13/14)
- `.planning/spikes/004-forbidden-capability/`: delta/T trajectory certificate, sealed probe sets
- `.planning/spikes/002-late-spike-meaning/`: control for the multiplier's level, or the
  signal is lambda in disguise
- `.planning/spikes/013-stratified-safety-set/gen013.py`: in-training generation hooks at
  chosen steps (a pattern for D_m checks)

## Notes

Captured from a /gsd-explore session (2026-09-29) with its trigger, why and scope filled
in. Controls required by the project's conventions: a size-matched random trigger, and
results reported net of the Lagrange multiplier's level.
