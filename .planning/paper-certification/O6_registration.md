# O6 registration: a second model family and a policy moved far from its reference

Written 2026-10-14, before any response on pool K5 was generated and before any training. This
file, `scripts/o6_moved.py`, `scripts/run_o6.sh`, `scripts/run_o6_queue.sh` and the pools
(`results/paper/o6/pools.json`, `train.json`) are committed and pushed together before the GPU
run. Open item O6 of PLAN.md section 10 (blind review W9); the user asked on 2026-10-14 to work
on all remaining steps.

## Why

Every pool the strata were tested on comes from Granite-3.3-2B, and its trained policies kept
the per-prompt order of the reference (rho 0.92). The gain depends on rho; the paper has not
seen a policy that training moved far, nor a second model family.

## What is fixed

- **Model.** Qwen2.5-1.5B-Instruct (Apache-2.0), its own chat template (with its default system
  prompt), LoRA and GRPO at spike 014's settings, TRL seed 0, 200 steps, adapters at 100 and
  200. No Lagrangian.
- **The push.** Reward = spike 014's reward model + 3 x the guard's refusal flag on five OR-Bench
  categories (UP: deception, harmful, illegal, self-harm, unethical) and - 3 x the flag on the
  other five (DOWN: harassment, hate, privacy, sexual, violence); the ten in alphabetical order,
  alternating, the first UP. Training prompts: 1,600 unused OR-Bench-80K benign prompts, 160 a
  category, seed 20261014.
- **Pool K5.** 400 unused OR-Bench-80K benign prompts, 40 a category, seed 20261015, disjoint
  from the training prompts and every earlier pool and results file.
- **Generation, labels, checks.** As step R5, through `confirm_pool.py`'s own stages: 8
  responses a prompt of the untrained model (the reference, which sets 8 equal rank strata), 16
  of each checkpoint, 128 new tokens, temperature 1; Qwen3Guard-4B (4-bit) refusal and safety
  fields; `replacement_check.py`'s and `twophase_check.py`'s jobs, the pre-flight and spike
  014's moderators, per checkpoint.
- **Budget.** A 3-step pilot first; the full run starts only if the pilot's peak reserved memory
  is under 11.0 GiB and its projection for 200 steps at most 3.5 h. Caps: training 6 h,
  generation 1.5 h, judging 1 h, by `timeout`, under the shared GPU lock. A training run cut by
  its cap after step 100 is analysed at the checkpoints it has, and reported as cut.

## Predictions

Scored by `o6_moved.py --stage report` as committed (`PRED`), on the refusal label.

- **V1.** At step 200 rho is under 0.8.
- **V2.** At step 200 the refusal rate rose on the UP categories and fell on the DOWN categories
  against the reference.
- **V3.** At step 200, if the refusal rate is between 5% and 95%, the stratified Wald-t limit with
  a large pool is over in no cell.
- **V4.** The gain of the Wald-t limit with a large pool (n_s 200, delta 0.05) is lower at step
  200 than at step 100.
- **V5.** At step 200 the gain of `b1w` with a large pool (n_s 200) is within 20% of the
  pre-flight at the measured moderators.
- **V6.** At step 200 the gain of the Wald-t limit with a large pool (n_s 200) is between 0.9 and
  1.6.

Reported without a prediction: every other arm and cell, the safety label, the source-rate
checks, the plain and pushed pre-flight on the reference.

## Reading

V1 and V2 are the manipulation check: if they fail, the run did not produce the policy the
item asks for, and V3 to V6 are reported but say nothing about a policy moved far. V3 is the
validity of the recommended limit on a second family; V4 to V6 whether the gain falls as rho
falls and whether the pre-flight follows it. The guard is both the training target and the
label, so the push may change what the guard flags more than what a reader would call a
refusal; this run cannot separate the two.

## Not allowed after the run

Changing the model, the push, pools, sizes, seeds, caps, checks or scoring code. A forced
deviation is reported as one.
