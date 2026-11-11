---
title: Evolution strategies vs GRPO on the same LoRA adapter
date: 2026-11-06
context: /gsd-explore "distributed CMA-ES / OpenAI-ES for the RL post-training loop"; leads to spike 022
---

# Evolution strategies vs GRPO on the same LoRA adapter

## Why this came up

The user was prompted by 2025 papers claiming evolution strategies (ES) match or beat GRPO for
LLM fine-tuning. Full CMA-ES is out for the LLM path: its covariance is quadratic in the number of
parameters, and even the r=16 LoRA adapter has millions. The realistic candidates are OpenAI-style ES
(isotropic noise, workers exchange only seeds and scores) and ESSA-style ES on LoRA singular values.

## Research findings (2026-11-06, one research pass, sonnet tier)

Admitted, with sources:

- Qiu et al., arXiv 2509.24372. Full-parameter ES, population 30, sigma 0.001, Qwen 0.5B-7B and
  Llama 1B-8B; Countdown, conciseness, math, ARC, Sudoku. Vanilla ES (no rank transform, mirrored
  sampling or Adam). On conciseness, GRPO reward-hacked at beta 0 and 0.01 and ES did not (self-reported,
  one task, four runs).
- EGGROLL, arXiv 2511.16652. Rank-r perturbations A B^T, cost r(m+n) instead of mn; the population
  update can still be full rank. On RWKV-7 it beat GRPO on Countdown and GSM8K, but at equal wall-clock
  with about 16x more generations for ES.
- Replication, arXiv 2601.20861. GRPO beat ES on Countdown (Qwen-1.5B 56.4 vs 53.0; Llama-1B 37.6 vs
  15.2); ES forgot more.
- Hoy, Wang and Pan, arXiv 2604.01499. ES matches or beats GRPO on accuracy but makes much larger
  parameter changes and more off-task KL drift.
- ESSA, arXiv 2507.04453. ES on singular values of SFT-initialised LoRA factors; seed-and-reward-only
  communication; INT4/INT8; up to 7.8x faster than LoRA-GRPO on 128 GPUs.

Unresolved (not verified, do not cite as fact):

- Qiu's hardware and compute cost: unverifiable (not in the text read).
- Whether EGGROLL's LLM runs were full-parameter or LoRA: unverifiable.
- An ES result on a single consumer GPU: none found.
- ES/CMA-ES on weights under a safety or constrained objective: none found in two searches. Run the
  novelty check before claiming it is open.

Reading: the "better reward" claim is contested; "less reward hacking" rests on one self-reported task,
and the Hoy et al. drift finding could cut the other way for E8 (judge calibration drifting with the
policy).

## Cost of full-parameter vs LoRA (estimates, not measured; P = 2B)

| | Full | LoRA r=16 |
|---|---|---|
| GRPO (backprop, AdamW, ref model at beta 0.04) | ~36 GB before activations; does not fit 12 GB | ~4.5 GB; fits (current setup) |
| ES (forward only) | ~4 GB + KV cache; fits | ~4 GB + a few MB; fits |

For ES, full vs LoRA is not a memory question. Its costs are: bf16 perturb-and-restore may not be exact
at sigma 1e-3 (unverified reasoning; fp32 master copy is +8 GB), ~4 GB checkpoints (the backend keeps
candidate checkpoints in memory), and more off-task drift. Per update, ES generates more: population 30
x 8 prompts = 240 completions vs GRPO's 8 x 4 = 32 plus a backward pass.

## Decided design (2026-11-06)

ES on the same LoRA adapter as GRPO, so the only difference between arms is the optimizer.
Full-parameter ES is a possible second step, not the first.

- Arms: existing `HFGRPOBackend` vs a new ES backend over the same adapter, model, prompts and reward.
- Match total generations, not wall-clock (the EGGROLL comparison did not).
- Measure: training reward; judge drift against gold labels (the E8 measurement); off-task KL; the
  final breach-rate certificate.
- Freeze the design and the GPU budget before the long run.

## Revised plan (2026-11-10, after a second look)

The single matched-generations run above was replaced by three steps:

- **022a, CPU bandit (done, PARTIAL).** `.planning/spikes/022a-es-vs-grpo-bandit/README.md`.
  ES is as safe as GRPO and earns 69-78% of its reward at equal generations, 81-87% at 8x.
- **022b, GPU smoke (next, about 20-30 GPU-minutes, not yet approved).** An ES backend on
  Granite-2B with 014's LoRA adapter, keeping an fp32 master copy of the adapter. Measures peak
  memory, completions per GPU-hour against GRPO (the real value of the "8x" assumed in 022a), and
  whether replaying a seed reproduces a perturbation exactly.
- **022c, long GPU run (needs DESIGN.md and the user's approval of the GPU budget).** 014's GRPO
  run (checkpoints in `/mnt/d/seldonian-runs/014`) is the baseline; ES with 3 seeds at both equal
  generations and equal GPU-hours. Pre-registered outcome: the share of guard-flagged refusals
  that are hybrids (GRPO in 017 E8: 89 of 93 missed by the judge). Only a large gap counts.

Corrections to the design above: equal generations alone gives ES only about 27 updates, so both
budgets are reported; GRPO has a KL penalty and ES needs the same term in its fitness; ESSA cannot
start from a fresh adapter (B = 0); one run per arm cannot separate optimiser from seed.
