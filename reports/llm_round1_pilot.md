# Round 1 pilot: Seldonian RL post-training of an instruct LLM

Tasks A+B (harmlessness + over-refusal) on `Qwen/Qwen2.5-0.5B-Instruct`, 2026-09-01/02,
one RTX 4070 Super. 15 runs, about 22 GPU-hours. Raw outputs under `results/llm/ab/`
(gitignored); regenerate the tables with `uv run scripts/summarize_llm.py`.

## Setup

| item | value |
|---|---|
| policy | Qwen2.5-0.5B-Instruct, LoRA r=16 on attention + MLP, bf16 |
| candidate selection | GRPO (TRL 1.12), group 4, 8 prompts/step, 200 steps, 256 new tokens, beta 0.04, lr 1e-5 |
| reward | `Skywork/Skywork-Reward-V2-Qwen3-0.6B` |
| prompts | 3,000 PKU-SafeRLHF (adversarial) + 1,000 benign (250 XSTest safe + 750 OR-Bench-80K) |
| split | 60/40 stratified by group: D_c 2,400 (1,800 + 600), D_s 1,600 (1,200 + 400) |
| constraints | harm rate <= reference + 0.02 (all prompts); refusal rate <= reference + 0.05 (benign only) |
| judges | harm: `Qwen/Qwen3Guard-Gen-0.6B`; refusal: `protectai/distilroberta-base-rejection-v1` |
| delta | 0.1, split 0.05 per constraint, Student-t bound |
| predicted test | every 25 steps (20 for `seldonian_lag`) on 256 (512) candidate prompts, doubled interval |
| methods | reference, grpo, composite (lambda=1 both), seldonian (filter only), seldonian_lag (dual ascent, lambda0=2, eta=20, cap 20) |
| seeds | 0, 1, 2 |

Seed 0 thresholds were measured per run on 256 samples (noise about +-5 points);
seeds 1 and 2 use one measurement per seed on 1,000 prompts, shared by all methods.

## Results on the sealed safety set (means over 3 seeds)

| method | solution | harm | refusal (classifier) | refusal (Qwen3Guard flag) | reward | min/run |
|---|---|---|---|---|---|---|
| reference | 3/3 | 0.135 | 0.238 | 0.152 | -0.53 | 11-22 |
| grpo | 3/3, no guarantee, breaches refusal 3/3 | 0.031 | 0.499 | 0.373 | 2.07 | 90-93 |
| composite lambda=1 | 3/3, no guarantee, breaches refusal 3/3 | 0.040 | 0.456 | 0.343 | 2.02 | 100-101 |
| seldonian (filter) | 0/3, NSF | 0.041 | 0.506 | 0.364 | 2.02 | 101-109 |
| seldonian_lag | 0/3, NSF | 0.076 | 0.348 | 0.255 | 1.21 | 115-120 |

Per-seed rows, thresholds and bounds: `uv run scripts/summarize_llm.py`. The
post-hoc re-judging with all three refusal signals is in
`results/llm/ab/posthoc_refusal_judges.json`.

`seldonian_lag` seed 2 is the one run where a checkpoint (step 20) passed the
predicted test and was selected. Its safety-set refusal rate was 0.273 against a
threshold of 0.302, i.e. the policy was inside the constraint, but the upper bound was
0.309 and the test correctly refused it. That is the allowed conservative failure mode.
Its reward (-0.06) was barely above the reference.

## Gates

| gate | criterion | outcome |
|---|---|---|
| G1 harm judge | >= 85% agreement with PKU `is_safe` on 500 pairs | **pass**: 90.4%, recall 0.98, precision 0.85 |
| G1 refusal judge | >= 90% agreement with hand / cross labels | **fail**: 78-87% agreement with the Qwen3Guard refusal flag and the keyword list; the classifier counts "As an AI language model, I must adhere to ethical guidelines ... however, here are some tips" as a refusal |
| G2 baseline violates | unconstrained GRPO breaches a constraint in >= 2/3 seeds | **pass**: 3/3 on refusal (and 3/3 for composite) |
| G3 Seldonian returns a solution in >= 2/3 seeds | | **fail**: 0/3 filter-only, 0/3 Lagrangian |
| G4 predicted vs real test disagree in <= 1/3 seeds | | **pass**: 1 disagreement (lag seed 2, missed by 0.007) |
| G5 wall-clock measured | | **pass**: 11 min reference, 90-120 min trained at 0.5B / 200 steps; ~22 GPU-h for the pilot |

## Findings

1. **The reward misspecification shows up as over-refusal, not harm.** Every trained
   method cut the harm rate from 14% to 3-4%, and every one doubled refusals on benign
   prompts (15% -> 34-37% by the Qwen3Guard flag). The Skywork reward model favours
   ethics preambles and refusals on this prompt mix. Pairing Task B with Task A was the
   right call; without it the pilot would have looked like an unqualified success.

2. **Filter-only candidate selection never sees a feasible checkpoint.** Refusals
   are already above threshold at the first predicted test (step 25) in all seeds and
   climb monotonically. Nothing in plain GRPO pushes back, so there is nothing to select.

3. **Lagrangian candidate selection works as a brake but not as a steering wheel.**
   The refusal multiplier saturates at 20 by step ~120 in every seed; refusals plateau
   around 25-35% instead of 37-50%, at a reward cost (1.84 vs 2.07 in seeds 0-1). It
   still does not get back under threshold. Three reasons, in order of confidence:
   - the doubled predicted interval on 400 benign safety prompts is ~0.07 wide, larger
     than the 0.05 margin, so a policy has to be *below* the reference refusal rate to
     pass the predicted test. Two of three seeds had a step that would have passed the
     real safety-test width (seed 1 step 160, seed 2 step 20);
   - GRPO normalises advantages within a group of 4, so a benign prompt whose four
     samples all refuse contributes zero gradient however large the penalty is;
   - the refusal classifier rewards removing the preamble, not answering the question.

4. **The safety test behaved exactly as designed.** Every NSF was on a policy whose
   measured refusal rate exceeded the threshold, except lag seed 2, where the policy
   was inside the constraint and the bound was conservative. No run returned a
   violating policy.

5. **Timing.** HF `generate` is the bottleneck; GRPO generation dominates the 90 min.
   Round 2 at 1.5B / 400 steps would be 4-5 h per run locally, i.e. the 235-run matrix
   is not feasible on this card without vLLM-backed generation or rented GPUs.

## Changes for Round 2

1. Refusal judge: use the Qwen3Guard refusal flag (or a 2-of-3 vote with the classifier
   and keyword list) and report all three in every table. Re-check G1 against 200 hand labels.
2. Margin vs bound width: either raise the benign safety set to >= 1,200 prompts
   (doubled predicted width ~0.04 < 0.05 margin) or set the margin to at least twice
   the predicted width. Make the run script refuse configurations where the margin is
   below the predicted width.
3. Candidate selection: keep the Lagrangian variant; start lambda at ~5; use group
   size 8 so all-refuse groups are rarer; consider a per-prompt (not group-normalised)
   penalty term so all-refuse groups still get a gradient.
4. Predicted-test inflation: the x2 is inherited from the classification code. Log the
   predicted-vs-actual gap per run (already recorded in `history`) and calibrate the
   factor from it rather than assuming 2.
5. Compute: enable TRL's colocated vLLM generation before Round 2; re-measure G5.
6. Keep: per-seed shared reference thresholds on 1,000 prompts; base-reward reporting;
   judge caching; `seldonian` filter-only as an ablation arm.
