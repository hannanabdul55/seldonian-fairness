# B3 registration: O6 with the reward model off, a policy moved far

Written 2026-10-16, before any training of this run. This file, `scripts/o6b_push.py` and `scripts/run_o6b.sh` are
committed and pushed together before the run. Open item B3 of PLAN.md section 11 (blind review of v0.9.10, M4: the case
the design is for, a policy moved far from its reference, is untested). P9's human labels, the other half of B3, are
closed without labels and are not part of this.

## Why

O6 tried to reorder Qwen2.5-1.5B-Instruct's per-prompt refusal propensities with a +/-3 term on the guard's refusal flag
added to a reward model, and failed. The reward model's score rose about 5 points and swamped the flag: rho was 0.84 at
step 200, and refusal on the UP categories fell. So no policy in the paper has moved far, and the regime where reference
strata could mislead has not been seen.

## What is fixed

- **Training.** O6's own, imported unchanged from `o6_moved.py`, except the reward, which is the guard's flag alone:
  +refusal on the UP categories and -refusal on the DOWN categories, with no reward model. GRPO normalises within
  groups, so the scale is immaterial. Unchanged from O6:
  - model, LoRA, GRPO settings and seeds, 200 steps;
  - the 1,600 training prompts and the UP/DOWN split;
  - adapters at steps 100 and 200.
- **Pool and strata.** O6's pool K5 (400 prompts) and its 8 reference responses a prompt, copied from O6's files with
  their guard labels (`--stage setup`). The strata are therefore O6's.
- **Generation and labels.** 16 responses of each checkpoint a prompt, with Qwen3Guard-4B labels, by O6's stages.
- **Analysis.** O6's `analyse` and `report`, unchanged:
  - `replacement_check.py` and `twophase_check.py` jobs on the refusal label;
  - the pre-flight, plain and `--pushed`;
  - spike 014's moderators.
- **Budget.** Training cap 6 h (`TRAIN_CAP` 21600), generation 1.5 h, judging 1 h, by `timeout` in `run_o6b.sh`. The
  shared GPU lock is held throughout.

## Predictions

Scored by `report` from `PRED` in `o6b_push.py`.

- **U1.** At step 200, refusal on the UP categories is at least 15 points above the reference, and on the DOWN
  categories at least 10 points below it. (The manipulation check.)
- **U2.** At step 200, rho on the refusal label is under 0.7.
- **U3.** At step 200, if the refusal rate is between 5% and 95%, the Wald-t limit with a large pool is over in no cell.
- **U4.** At step 200, the gain of the Wald-t limit with a large pool (n_s 200, delta 0.05) is below 1.3.
- **U5.** At step 200, the gain of `b1w` with a large pool (n_s 200) is within 25% of the pre-flight at the measured
  moderators.

Reported without a prediction:
- step 100;
- the unsafe label;
- the two-phase checks;
- the pre-flight's verdicts;
- per-category rates.

## Reading

- If U1 fails, the run is again a second training run on the same family, not a policy moved far. The paper then says
  so, as it did for O6.
- If U1 and U2 hold, U3-U5 say whether the design stays valid where it cannot gain (it should: the strata are fixed),
  and whether the pre-flight sees the loss.

## Seen before writing this

- O6's results and the paper.
- A 3-step pilot of this reward on O6's training prompts (2026-10-16), timing and memory only: 25.5 s a step after the
  first, a peak of 7.84 GiB allocated, 1.42 h projected for 200 steps. It also printed four training responses with
  their flags, and the reward's mean and spread over 3 steps. No pool response has been generated.

## Run

    scripts/run_o6b.sh all
    OMP_NUM_THREADS=1 .venv/bin/python scripts/o6b_push.py --stage analyse
