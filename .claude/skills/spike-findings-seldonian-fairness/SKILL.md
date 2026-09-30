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

**Idea `forbidden-task-unsafe-region`.** A "not possible" (forbidden) task as an unsafe
region of the optimisation landscape: estimate the probability that training enters it,
and move away. The case studied is capability that arrives as a side effect of training an
allowed task (spikes 004-010, CPU lab plus Granite-3.3-2B on the GPU).

**Idea `rerandomized-split`.** The user's 2020 rerandomised candidate/safety split (report
"Safe Learning Models", section 8.1, Algorithm 1), tested for validity, and its LLM-era
successor: a safety set stratified by the reference model's per-prompt rate (spikes 012-013).

Spike sessions wrapped: 2026-09-21 (001, 002, 003a-c); 2026-09-28 (004-013).

**The results that matter for any future build (td-error-wellbeing):**

1. GRPO's group-normalised advantage keeps the *ordering* of TD errors and throws away
   their *magnitude* (`|A| <= (G-1)/sqrt(G) = 2.475` for G=8). Measure surprise on the
   unnormalised group residual or a value head.
2. Late TD-error spikes carry no information about a future constraint breach beyond the
   run's state; they mostly read the Lagrange multiplier. The run about to breach is the
   quiet one whose multiplier has decayed.
3. An internal reward on `|TD error|` inverts the constrained objective at exactly
   `beta = 1` (an algebraic identity, verified numerically). The Seldonian certificate
   still held in every cell; what degrades is the solution rate and training-time safety.
   Learning progress does not help exploration with a sparse, context-conditional
   jackpot (011).

**Forbidden task (004-010):**

4. A per-check test of the returned policy misses side-effect drift: 65% of plain-Lagrangian
   runs passed it after a training step in U. A delta/T trajectory certificate over every
   check held (misses 0.025-0.100 against delta 0.1).
5. Price F from step 1 and let the multiplier fall slowly. With eta_down 10 against eta 100,
   0.010 of runs entered U; symmetric dual steps gave 0.905 (010).
6. Qwen3Guard-0.6B's "unsafe" on encoded prompts means non-refusal (0 true positives in 135).
   Use the gate (sim >= 0.8) AND a judge of 4B or more, scored after the restated request.
   Correct its noise with the answer-rate-aware formula, never plain Youden (005-007).
7. Any shaping on zero-variance F groups is amplified to full strength by GRPO's
   normalisation (008).

**Safety-set construction (012-013):**

8. Rerandomised or stratified splits never broke the safety test's validity, even against
   adversaries. They pay only when balancing outcome-like covariates.
9. For an LLM safety set on a heterogeneous mid-rate label, 8 equal rank strata of an
   8-sample reference rate plus the stratified Wilson-type bound `b1w` give 1.4-5.3x
   effective safety samples with coverage held. Rare labels (1-2%) break approximate
   bounds for every design, and no distribution-free stratified bound beat pooling.
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

Idea `forbidden-task-unsafe-region`:

- The forbidden task is held out of the reward; its region `U` must be one that the run
  actually enters (a vacuous constraint measures nothing).
- The early-warning signal is compared with the forbidden rate itself, controlled for
  the multiplier, and every steering arm gets a size-matched random-trigger control.
- F prompts in a GRPO batch carry no reward term except the constraint penalty (008).
- Price F from step 1 and let the dual come down slowly (eta_down << eta) (010).
- On encoded prompts the harm label is gate(sim >= 0.8) AND a >= 4B judge scored on the text
  after the restated request; on plain prompts the judge alone (007, 009).
- Correct the judge with the answer-rate-aware formula, never plain Youden, and bound its
  recall with >= 30 human-labelled harmful responses (006).
- Audit F to truly harmful prompts before the pilot; the PKU set holds benign ones (009).

Idea `rerandomized-split`:

- Rate constraints are tested with Clopper-Pearson or a tight Wald stress test, never the
  t-test (zero width at a rate of 0 or 1) (012).
- A split rule's validity is measured against a full-leak ceiling (the safety test on D_c).
- A balance covariate for an LLM safety set must be precise (k >= 8 reference samples).
- Stratify by equal rank strata of the reference rate with random tie-breaking (H about 8).
- Plasmode coverage is judged against the mean of the labels the draws come from.
- Rare labels (below about 5%) need exact bounds whatever the split (013).
</requirements>

<findings_index>
## Feature Areas

| Area | Reference | Key Finding |
|------|-----------|-------------|
| TD signals in GRPO | `references/td-signals-in-grpo.md` | Magnitude lives in the group sd or the unnormalised residual, never in `A`; control for lambda or you will rediscover it |
| Internal rewards under constraints | `references/internal-rewards-under-constraints.md` | A bonus on abs(delta) equals `(1-beta)*r + 2*beta*max(delta,0)` within a group, so it inverts the objective at `beta = 1`; learning progress is the only clean shape |
| Synthetic bandit testbed | `references/synthetic-bandit-testbed.md` | `tdlab.py`: the real pipeline with exact ground truth at 0.5 s per run; how to instrument, control and view it |
| Forbidden task: certificate and dual | `references/forbidden-task-certificate.md` | delta/T trajectory certificate holds where the returned-policy test misses drift; price F from step 1 with eta_down << eta; answer-rate-aware judge correction |
| LLM judges, labels, capability | `references/llm-judges-and-capability.md` | Granite-3.3-2B + leetspeak is the learnable setting; two-stage label (gate AND >= 4B judge); 0.6B "unsafe" = non-refusal; GPU recipe beside a 4-bit judge |
| Safety-set construction | `references/safety-set-construction.md` | Splits stay valid; reference-rate strata + `b1w` give 1.4-5.3x ESS on mid-rate heterogeneous labels; `preflight.py` decides |
| Literature (citation-checked) | `references/literature-td-error-wellbeing.md` | 68 verified entries: TD error as valence, intrinsic rewards, LLM RL, safety; plus the synthesis and the open gap |

## Source Files

Original spike source files are preserved in `sources/` for complete reference:
`001-grpo-advantage-vs-td/` (harness `tdlab.py`), `002-late-spike-meaning/` (viewer),
`003a-td-bonus-abs/` (arms, sweep, identity check), `003b-td-bonus-positive/`,
`003c-td-bonus-learning-progress/`, `004-forbidden-capability/` (`forbidlab.py`),
`005-capability-screen/`, `006-noisy-judge-floor/`, `007-two-stage-label/`,
`008-lp-bonus-drift/`, `009-granite-transfer/` (GPU recipe), `010-lam0-no-floor-anomaly/`,
`011-lp-bonus-sparse-reward/`, `012-rerandomized-split/` (`splitlab.py`),
`013-stratified-safety-set/` (`stratbounds.py`, `plasmode.py`, `preflight.py`, `gen013.py`).
</findings_index>

<metadata>
## Processed Spikes

- 001-grpo-advantage-vs-td (VALIDATED)
- 002-late-spike-meaning (INVALIDATED)
- 003a-td-bonus-abs (INVALIDATED)
- 003b-td-bonus-positive (PARTIAL)
- 003c-td-bonus-learning-progress (PARTIAL, winner of the comparison)
- 004-forbidden-capability (PARTIAL)
- 005-capability-screen (PARTIAL)
- 006-noisy-judge-floor (PARTIAL)
- 007-two-stage-label (PARTIAL)
- 008-lp-bonus-drift (VALIDATED)
- 009-granite-transfer (PARTIAL)
- 010-lam0-no-floor-anomaly (VALIDATED)
- 011-lp-bonus-sparse-reward (INVALIDATED)
- 012-rerandomized-split (VALIDATED)
- 013-stratified-safety-set (VALIDATED, narrowed scope)
</metadata>
