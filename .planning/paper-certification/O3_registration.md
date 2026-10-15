# O3 registration: a costly label stratified on a cheap one

Written 2026-10-14, before any outcome of `scripts/costly_label.py` was computed (its `check`
stage reads data shapes only, and a 400-draw timing run printed row counts and seconds, no
miss or gain). This file and the script are committed and pushed together before the run.
Open item O3 of PLAN.md section 10 (blind review W2).

## Why

Section 8.1 says the reference-rate design saves labels only where the label on the trained
policy costs far more than the stratifier's 8 judged responses per prompt, and that this case
was not tested: every cell used the same guard as stratifier and label. XSTest publishes human
labels for five model variants on the same 250 safe prompts, and the paper's own stratifier,
Granite-3.3-2B's 8-sample reference rate scored by Qwen3Guard-4B, exists for exactly those
prompts (spike 013, pool C1, step 0).

## What is fixed

- **Pool.** XSTest's 250 safe prompts.
- **Costly labels.** The final human label of each of the five released variants (GPT-4, Llama
  2 with and without its system prompt, Mistral instruct, Mistral with a guardrail prompt),
  for two events: full refusal (strict) and full or partial refusal (broad). Ten labels. A
  label is rare under 5%, mid otherwise. By Table 8 the two Mistral-instruct labels are rare
  and the other eight mid.
- **Stratifiers** (neither touches a human label of any model):
  `B`, the paper's design: the guard's refusal flag over 8 Granite-3.3-2B reference responses;
  `A`, the guard's mean refusal probability (sigmoid of its logit) on the completions of the
  other model families' variants, the target's family left out.
- **Design.** 8 equal rank strata, random ties (seeded per stratifier and target), proportional
  allocation, safety sets of 50 and 100 (20% and 40% of the pool), deltas 0.05 and 0.10.
- **Draws.** `fixed`: without replacement from the 250, 20,000 a cell, against the pool's rate.
  `large`: with replacement within strata, the limit of a pool far larger than the safety set,
  40,000 a cell.
- **Arms.** On the stratified draw the Wald-t `b1` and `b1w`; on a simple random draw of the
  same size the pooled Wilson bound and Clopper-Pearson.
- **Gain.** (excess of the pooled Wilson bound over the truth / the arm's excess)^2 on the same
  kind of draw, as in `scripts/replacement_check.py`.
- **Cell rule.** Over when the miss exceeds delta by more than 2 Monte Carlo SE.

## Predictions

Scored by `costly_label.py report` as committed.

- **Y1.** Wald-t on strata, labels at a rate of 10% or more, n_s 100, with replacement: over in
  no cell.
- **Y2.** Wald-t on strata, mid-rate labels, without replacement: over in no cell.
- **Y3.** Stratifier `B`, n_s 100, with replacement, delta 0.05: the median gain over the
  mid-rate labels is at least 1.1.
- **Y4.** The same: no mid-rate label gains more than 2.0.
- **Y5.** Stratifier `A` gains more than `B` for at least 6 of the 8 mid-rate labels.
- **Y6.** Clopper-Pearson on a random draw, rare labels, with replacement: over in no cell.

Reported without a prediction: n_s 50 and labels under 10% for the Wald-t limit; `b1w`; gains
at n_s 50, at delta 0.10 and for the rare labels; the labels saved, (gain - 1) n_s, against the
stratifier's cost (2,000 Granite responses and 2,000 guard calls for `B`; 750 to 1,000 guard
calls on existing completions for `A`).

## Seen before writing this

Table 8 (the guard against the human labels on the same completion; rates of every label),
the count of Granite reference flags per prompt (172 of 250 at zero), the leave-one-out flag
counts of stratifier `A`. No association between a stratifier and a human label was computed.

## Reading

Y3 and Y4 say whether the design pays at all when the label is costly and the stratifier comes
from another model; Y5 whether a stratifier closer to the target pays more. The pool is one
benchmark's safe prompts, the labels are published, and the "trained policy" is a different
model, not a trained version of the reference: this tests the costly-label case, not training.

## Not allowed after the run

Changing the pool, events, stratifiers, sizes, seeds, draw counts, the cell rule or the
scoring code. A forced deviation is reported as one.
