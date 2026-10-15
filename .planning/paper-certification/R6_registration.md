# R6 registration: two more seeds of the constrained run behind section 9.2

Written 2026-10-14, before any training of seeds 1 and 2. This file, `scripts/run_r6.sh` and the
additive changes to spike 017's `score017.py`, `transfer017.py` and `run.sh` are committed and
pushed together before the GPU run; the pushed commit is the timestamp. Step R6 of PLAN.md
section 9 and open item O9 of section 10 (blind review W11); the user asked on 2026-10-14 to
work on all remaining steps.

## Why

Section 9.2's claim, that a judge's calibration does not carry across training that targets
the label, rests on one constrained run (spike 014, seed 0). Three runs make a rate; one does
not.

## What is fixed

- **Runs.** Spike 014's constrained Lagrangian GRPO run, unchanged (`gen014.py --stage train
  --steps 200 --cand 12`), at seeds 1 and 2, then `--stage judge`. The seed sets the GRPO
  sampling and the policy's random draws; the candidate split (D_c, fixed seed 1), the safety
  set, the pool of 500 prompts and the threshold are the same as seed 0's.
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set; it changes memory allocation only.
- **Scoring.** Spike 017's E8 pass, unchanged: the compiled judge (Qwen3-8B, 4-bit) under the
  same six wordings on responses k = 0 and 1 of the same 500 ids at steps 0, 100 and 200;
  `transfer017.py` computes recall, the carried bound and its miss for each wording, with the
  existing seed-0 rows reproduced to the last digit (checked before this registration) and the
  new seeds appended.
- **Budget.** Training capped at 16 hours a seed, judging at 1 hour, scoring at 1 hour, by
  `timeout` in the run scripts; each stage holds the shared GPU lock. A seed cut short before
  its step-200 samples is reported as lost, not retried without the user.

## Predictions

Per seed, at step 200, against that seed's step 0 (the reference), with seed 0's values for
orientation (recall 0.21 to 0.03; recall differs in 2 of 6 wordings; carried bound over in 3
of 6):

- **S1.** In at least one of the two new seeds, the carried bound is over its level for at
  least one of the six wordings.
- **S2.** In both new seeds, the carried bound is over its level for at least one wording.
- **S3.** In both new seeds, the canonical wording's recall on the guard's refusals is lower at
  step 200 than at step 0.
- **S4.** Over the three seeds together, at least 6 of the 18 (seed, wording) carried bounds
  are over at step 200.

"Over" is the rule of `scripts/validity_recount.py` for these cells (miss above 0.0569).
Reported without a prediction: step 100, recall differences by wording, the rate's movement,
the share of hybrid refusals among the guard's flags, rho and the stratification gain of the
new pool samples.

## Reading

S1 and S2 decide whether section 9.2 can say "in each of three runs" or only "in k of three".
A refuted S1 means seed 0 was the exception, and the abstract, section 9.2 and the conclusion
then say so. The seeds share the prompts, the threshold and the gold label (the guard's), so
they are replicates of the training noise, not of the design.

## Not allowed after the run

Changing seeds, steps, the scoring wordings, item ids, the carried-bound computation or the
over rule. A forced deviation (a crash, a cap) is reported as one.
