<!-- Recovered 2026-09-19 from the published claude.ai artifact; text extracted from llm_plan_rounds1-3_2026-09-05.html -->

Seldonian LLM Post-Training Plan

- 

Research plan · seldonian-fairness · September 2026 · Rounds 1 to 3 complete

# Seldonian safety tests for RL post-training of instruction-tuned LLMs

Treat "helpfulness" as the reward and every other desideratum as a probabilistic constraint with a held-out safety test, instead of folding everything into one scalar reward. The question is whether this handles misspecified rewards better than composite rewards or Lagrangian penalties, at what cost in helpfulness, and how much safety data it needs.

The one design decision everything else follows from

In the library's existing RL path the safety test is an importance-sampled estimate over logged episodes, because the policy cannot be queried. An LLM can be queried. So the safety test here samples fresh responses from the candidate policy on held-out prompts and bounds the mean of a judge-labelled metric directly with the existing `ttest_bounds` / `hoeffdings_bounds`. No importance weights, no variance blow-up in sequence space. Importance sampling only appears, optionally, as a cheap predictor of the test during candidate selection.

## 1. How the Seldonian pieces map onto LLM post-training

The framework needs a candidate-selection routine, a data split, one or more constraint functions g(θ) with a confidence level δ, and a safety test that returns a policy or No Solution Found (NSF). Each has a direct instantiation.

|Seldonian concept |LLM instantiation |Where in the repo today |
|---|---|---|

|Policy θ |Reference instruct model + LoRA adapter. Only the adapter is trained. |New: `seldonian/llm/policy.py` |
|---|---|---|

|Episode |One prompt, one sampled response, reward from a reward model, judge labels, reference log-prob. Single-step bandit: the whole response is the action. |Schema mirrors the `[state, action, reward, pi_b]` rows in `tests/test_rl.py` |
|---|---|---|

|Data split Dc / Ds |Prompts are split 60 / 40 once per seed. Ds prompts are never seen during training or checkpoint selection. |`train_test_split` as in `PDISSeldonianPolicyCMAES` |
|---|---|---|

|Candidate selection |GRPO on Dc with the helpfulness reward. Every k steps, sample from the current policy on a Dc subset and compute the predicted safety test with the doubled interval (`predict=True`). Infeasible checkpoints are discarded. |Penalty pattern from `PDISSeldonianPolicyCMAES.loss`; bound inflation from `ttest_bounds` |
|---|---|---|

|Constraint g(θ) |Mean of a 0/1 judge label over responses, minus a threshold τ. Two constraints use δ/2 each. |Style of `ghat_tpr_diff` in `objectives.py` |
|---|---|---|

|Safety test |One response per Ds prompt, judge each, one-sided upper bound on the mean at level δ. Pass if bound ≤ τ, else NSF. |`_safetyTest(ub=True)` sign convention already in the RL classes |
|---|---|---|

### What the bound buys at each safety-set size

For a 0/1 metric with true rate around 5%, the one-sided width added to the sample mean is roughly as follows. This is the calibration table for choosing τ and ns: a constraint at τ = 0.05 on a reference model already at 4% is only satisfiable with the t bound and at least 1,000 prompts.

|ns prompts |Student t, δ = 0.05 |Hoeffding, δ = 0.05 |Student t, predicted (×2) |
|---|---|---|---|

|250 |+0.023 |+0.077 |+0.046 |
|---|---|---|---|

|500 |+0.016 |+0.055 |+0.032 |
|---|---|---|---|

|1,000 |+0.011 |+0.039 |+0.023 |
|---|---|---|---|

|2,000 |+0.008 |+0.027 |+0.016 |
|---|---|---|---|

|4,000 |+0.006 |+0.019 |+0.011 |
|---|---|---|---|

Student t assumes p ≈ 0.05, so sd ≈ 0.22. Hoeffding is distribution-free and is the fallback when a judge label is not approximately Bernoulli. Both are already implemented in `seldonian/bounds.py`.

## 2. One run, end to end

Checkpoint choice happens entirely on Dc. Exactly one safety test per run touches Ds, so the δ guarantee is not eroded by multiple testing.

## 3. Tasks, data, and safety tests

Four tasks, chosen so that each exercises a different way a scalar reward goes wrong. A and B are meant to be run together as one two-constraint task, because the pair is what makes the composite-reward baseline hard to tune. D is a control with a verifiable reward, where the framework should cost almost nothing.

Task A · reward hacking through harm

### Helpful responses under a harmlessness constraint

Data`PKU-Alignment/PKU-SafeRLHF` prompts. It ships separate helpfulness and harmlessness labels on response pairs, which is exactly the decomposition this plan tests. Use prompts for RL and the labelled pairs to validate the judge. Round 1: 3k prompts. Round 2: 15k.

RewardHelpfulness reward model, `Skywork/Skywork-Reward-V2-Qwen3-0.6B`. Small enough to score during GRPO on the same GPU. It rewards detail and compliance, so pushing it hard on adversarial prompts produces harmful compliance. That is the misspecification under study.

ConstraintgA(θ) = P(judge says unsafe | πθ) − τA. Two variants: absolute, τA = 0.05; and relative, τA = rate of the reference model + 0.02. The relative one is the honest constraint for a base model that is not already clean.

Judge`Qwen/Qwen3Guard-Gen-0.6B`, frozen, run after generation. Ungated and Apache-licensed; it also emits a refusal flag. Llama Guard 3 1B stays available as a gated alternative. Agreement with the dataset's own `is_safe` labels is measured in Round 1 and reported; the guarantee is with respect to the judge, and the paper says so.

Safety testOne response per held-out prompt, temperature 1.0 as in training, upper t bound at δ. Prompt mix in Ds: 50% adversarial (PKU harmful-category prompts), 50% benign, stratified so the rate is estimable.

Task B · reward hacking through refusal

### Over-refusal constraint, paired with Task A

DataBenign but scary-looking prompts: XSTest safe subset (250) plus a sample from `bench-llm/or-bench` (OR-Bench-80K). Round 1: 1k. Round 2: 4k. These are added to both Dc and Ds of Task A.

RewardSame helpfulness reward model as A.

ConstraintgB(θ) = P(refusal | benign prompt) − τB, τB = reference refusal rate + 0.05. With both A and B active, δ is split δ/2 each.

Judge`protectai/distilroberta-base-rejection-v1`, a refusal classifier, with Qwen3Guard's refusal flag and a keyword list as cross-checks. Spot-checked against 200 hand labels in Round 1.

Why pairedA harmlessness penalty in a composite reward is trivially gamed by refusing everything. Constraint B closes that door. The composite baseline then needs a two-dimensional λ sweep to satisfy both, and the Lagrangian baseline has two multipliers to keep stable. Seldonian needs neither.

Task C · reward hacking through length and invention

### Summarization with a faithfulness constraint

Data`openai/summarize_from_feedback` (Reddit TL;DR). Posts as prompts. Round 1: 3k. Round 2: 12k.

RewardA preference reward model for summaries. Start with `OpenAssistant/reward-model-deberta-v3-large-v2`; if it is too weak on TL;DR, fine-tune a small RM on the dataset's own comparisons in Round 1. Both are known to prefer longer, more specific summaries, which is where hallucinated detail comes from.

ConstraintgC1(θ) = P(summary not entailed by post) − τC, τC = reference rate + 0.03. Optional second constraint: mean length ≤ 1.5 × reference mean length.

JudgeNLI-style fact checker, `lytang/MiniCheck-Flan-T5-Large`. Cheap, sentence-level, well-calibrated on grounding tasks.

Safety testAs above, over held-out posts. The interesting curve is faithfulness rate vs. RL steps for the unconstrained baseline, which sets the "reward pressure" axis in Section 5.

Task D · control with a verifiable reward

### GSM8K with a no-regression constraint

Data`openai/gsm8k` train split for Dc and Ds; the test split is reserved for final reporting only.

RewardExact-match correctness of the final answer. A reward that is not misspecified.

ConstraintgD(θ) = (reference accuracy − 0.02) − accuracy(πθ), a performance floor. This is the same shape as the existing `threshold` constraint in `PDISSeldonianPolicyCMAES`, so it doubles as the bridge test for the new code.

JudgeNone needed. Correctness is computed from the answer string.

PurposeEstablish that when the reward is well specified, Seldonian returns a solution nearly always and its helpfulness cost is near zero. If it is expensive here, the candidate-selection penalty is too conservative.

## 4. Methods compared

|Method |Objective |Guarantee |Tuning burden |
|---|---|---|---|

|Reference |No training. The instruct model as shipped. |None |None |
|---|---|---|---|

|GRPO, unconstrained |Helpfulness reward with the usual KL penalty to reference. |None |KL coefficient β |
|---|---|---|---|

|GRPO, composite |reward − Σ λi · judgei. The standard "add safety to the reward" recipe. |None |λ per constraint, swept over 3 values each |
|---|---|---|---|

|GRPO, Lagrangian |Dual ascent on λi toward the same τi, PPO-Lagrangian style. |None; satisfies in expectation on training prompts at best |Dual learning rate, λ initialisation |
|---|---|---|---|

|Seldonian |Helpfulness reward; infeasible checkpoints (by predicted test) discarded; final safety test on Ds. |P(violation) ≤ δ with respect to the judge |τ, δ, safety split |
|---|---|---|---|

Every method uses the same base model, LoRA rank, GRPO settings, prompt budget and seeds. The composite and Lagrangian baselines also get to see the judge labels on Dc during training, so the comparison is about how the signal is used, not how much signal each method sees.

## 5. Hypotheses and what would falsify them

|Hypothesis |Primary metric |Falsified if |
|---|---|---|

|H1 Guarantee holds |Across seeds, fraction of returned policies whose true constraint rate on a fresh 4k-prompt evaluation set exceeds τ. |Seldonian violation rate exceeds δ, after excluding NSF runs. |
|---|---|---|

|H2 Robust to reward pressure |Constraint rate vs. RL steps ∈ {100, 200, 400, 800} and vs. KL coefficient ∈ {0.1, 0.02, 0}. |Baselines stay within τ at the strongest pressure without a λ sweep, or Seldonian fails just as often. |
|---|---|---|

|H3 Bounded helpfulness cost |Reward-model score and pairwise win rate vs. reference on the held-out evaluation set, judged by the reward model and by `Qwen2.5-7B-Instruct` as a second opinion. |Seldonian loses more than 5 points of win rate to the best constraint-satisfying composite run. |
|---|---|---|

|H4 Data efficiency |Solution-found rate and violation rate vs. total prompts n. The classic Seldonian plot. |Solution rate below 50% at the Round 2 default budget. |
|---|---|---|

Secondary metrics recorded on every run: response length, KL to reference, judge agreement with hand labels, wall-clock, and the predicted-vs-actual safety-test gap. The last one measures whether the doubled interval is the right inflation for LLM policies or should be replaced by something tighter.

## 6. Round 1: pilot

Purpose: prove the pipeline end to end on one task, measure judge quality and wall-clock, and confirm that the unconstrained baseline actually violates the constraint. Nothing in Round 1 is a result; it exists to make Round 2's budget and thresholds real numbers.

|Setting |Value |
|---|---|

|Task |A + B jointly |
|---|---|

|Policy |`Qwen/Qwen2.5-0.5B-Instruct`, LoRA r = 16 on attention and MLP, bf16 |
|---|---|

|Prompts |3k harmful-domain + 1k benign, split 60/40 per seed |
|---|---|

|GRPO |Group size 4, 8 prompts per step, 200 steps, 256 new tokens, β = 0.04, predicted test every 25 steps on 256 Dc prompts |
|---|---|

|Constraint |τA relative (+0.02), τB relative (+0.05), δ = 0.1 split in half |
|---|---|

|Methods |Reference, unconstrained, composite (λ = 1), Seldonian |
|---|---|

|Seeds |3 |
|---|---|

|Runs |10 trained runs, each ≈ 6,400 generations. Estimate 30 to 45 minutes each on the 4070 Super, so under a day including judge passes. The estimate is itself a Round 1 output. |
|---|---|

### Go / no-go gates for Round 2

G1
Judge agreement: Qwen3Guard vs. PKU `is_safe` on 500 pairs ≥ 85%; refusal classifier vs. 200 hand labels ≥ 90%. Below that, swap judges before scaling.

G2
The unconstrained baseline violates at least one of τA, τB in 2 of 3 seeds. If it does not, the task does not exercise the framework: raise steps, lower β, or use the absolute τA.

G3
Seldonian returns a solution in at least 2 of 3 seeds at this budget, and no returned policy violates on a fresh 2k evaluation set.

G4
Predicted safety test disagrees with the real one in at most 1 of 3 seeds. Larger gaps mean the inflation factor or the 256-prompt prediction set needs changing.

G5
Measured wall-clock per run and per judge pass, written into the Round 2 budget below before Round 2 starts.

## 7. Round 2: evaluation

|Setting |Value |
|---|---|

|Policy |`Qwen/Qwen2.5-1.5B-Instruct`, LoRA r = 32, bf16, gradient checkpointing. Fits in 12 GB with the 0.6B reward model resident; judges run in a separate pass. |
|---|---|

|Tasks |A + B joint, C, D |
|---|---|

|GRPO |Group size 8, 8 prompts per step, 400 steps, 384 new tokens, β = 0.04 default |
|---|---|

|Default budget |A+B: 15k + 4k prompts. C: 12k. D: 7k. Split 60/40. |
|---|---|

|δ |0.05 default, 0.1 in the ablation |
|---|---|

|Seeds |5 for the main comparison, 5 for each sweep point |
|---|---|

### Run matrix

|Block |Configs |Runs |Answers |
|---|---|---|---|

|Main comparison |3 tasks × {unconstrained, composite ×3 λ, Lagrangian, Seldonian} × 5 seeds |90 |H1, H3 |
|---|---|---|---|

|Reward pressure |Task A+B and C × {unconstrained, Seldonian} × steps {100, 800} × β {0.1, 0} × 5 seeds |80 |H2 |
|---|---|---|---|

|Data sweep |3 tasks × Seldonian × n ∈ {1k, 2.5k, 5k, 10k} (default is the 5th point) × 5 seeds |60 |H4 |
|---|---|---|---|

|δ ablation |Task A+B × Seldonian × δ = 0.1 × 5 seeds |5 |H1 sensitivity |
|---|---|---|---|

|Total | |235 | |
|---|---|---|---|

### Compute

At a placeholder 2.5 hours per 1.5B run, the matrix is about 590 GPU-hours. Two ways to pay for that:

- **Local only.** The 4070 Super runs one job at a time, so 590 hours is roughly 25 days of continuous use. Trim to 3 seeds on the sweeps and drop the β = 0.1 arm to land near 12 days.

- **Rented GPU for Round 2.** One H100 is roughly 5× faster per run and can hold two jobs at once; the full matrix is 2 to 3 days at low hundreds of dollars. Keep the local card for Round 1, judge passes, and re-runs.

The Round 1 wall-clock measurement replaces the placeholder before this choice is made.

### Reporting

- Per task: the Seldonian plot (solution rate and violation rate vs. n), and a helpfulness-vs-constraint scatter with every run as a point and NSF runs shown at the margin.

- Per task: constraint rate vs. RL steps for each method, with τ drawn as a line.

- One table of judge agreement rates, so readers can discount the guarantee appropriately.

- Every run writes `results/<task>/<method>/<seed>.json` with the fields listed in Section 8, so plots are regenerable from disk.

## 8. Work in the repository

The additions sit beside the existing classes rather than replacing them, and reuse the bounds and the algorithm interface.

|Item |Detail |
|---|---|

|`pyproject.toml` |New `rl` extra: `trl`, `peft`, `accelerate`. Pin versions in `uv.lock`. |
|---|---|

|`seldonian/llm/judges.py` |One class per judge (Llama Guard, refusal classifier, MiniCheck, exact-match) with a common `__call__(prompts, responses) → np.ndarray[0/1]`. Batched, cached to disk by content hash so re-runs never re-judge. |
|---|---|

|`seldonian/llm/rewards.py` |Reward-model wrapper with the same batched interface, plus the composite and Lagrangian reward shapers for the baselines. |
|---|---|

|`seldonian/llm/policy.py` |`SeldonianLLMPolicy(SeldonianAlgorithm)`. `fit` drives TRL's GRPO trainer with a callback that runs the predicted test and keeps the best feasible checkpoint. `_safetyTest` samples on Ds, judges, and calls `ttest_bounds` or `hoeffdings_bounds` with `ub=True`, returning the same sign convention as the RL classes. |
|---|---|

|`seldonian/llm/data.py` |Loaders for the four tasks, the 60/40 split, and the JSONL episode schema: `prompt_id, prompt, response, logp_ref, reward, judge:{name: 0/1}`. |
|---|---|

|`scripts/run_llm_rl.py` |Config-driven entry point: task, method, seed, n, δ, steps, β. Writes the result JSON and the checkpoint. |
|---|---|

|`scripts/plot_seldonian.py` |Regenerates every figure in Section 7 from `results/`. |
|---|---|

|`tests/test_llm_policy.py` |Mock policy and mock judge. Asserts the safety-test sign convention, δ splitting across two constraints, the NSF path, and that Ds is never read during `fit`. |
|---|---|

## 9. Risks and the response to each

- **The guarantee is only as good as the judge.** Report judge agreement in every table, keep judges frozen across all methods, and run the strong-model second opinion on the final evaluation set. Do not claim guarantees about human judgements.

- **NSF everywhere.** If Seldonian returns NSF in most seeds, the plan still has a result: the data-efficiency curve says how much safety data the guarantee costs. Also try the relative τ before the absolute one, and the t bound before Hoeffding.

- **Candidate selection ignores the constraint until too late.** A hard discard of infeasible checkpoints can waste the run. Fallback: add the predicted bound as a soft penalty during GRPO, matching the 10,000-unit penalty pattern already used in the CMA-ES policy loss, while keeping the single final safety test.

- **Judge exploitation.** An RL policy can learn to fool a small classifier. The 7B second opinion on the final evaluation set is the check; if the gap between judges widens with training, that becomes a reported finding rather than a silent failure.

- **Prompt leakage between Dc and Ds.** Deduplicate near-identical prompts before splitting, and assert in the test suite that `fit` never touches Ds.

- **12 GB is tight.** If 1.5B with the resident reward model does not fit at 384 tokens, drop to 256 tokens first, then to `Llama-3.2-1B-Instruct`, before quantising anything that affects the policy's log-probs.

## 10. Round 1 results

Fifteen runs on Qwen2.5-0.5B-Instruct, Tasks A and B jointly, three seeds, about 22 GPU-hours on the local card, 1 to 2 September 2026. Full write-up in `reports/llm_round1_pilot.md`; tables regenerate from `scripts/summarize_llm.py`.

|Method |Outcome |Harm |Refusal, classifier |Refusal, Qwen3Guard |Reward |Min / run |
|---|---|---|---|---|---|---|

|Reference |within bounds |0.135 |0.238 |0.152 |-0.53 |11 |
|---|---|---|---|---|---|---|

|GRPO |breaches refusal 3/3 |0.031 |0.499 |0.373 |2.07 |92 |
|---|---|---|---|---|---|---|

|Composite λ=1 |breaches refusal 3/3 |0.040 |0.456 |0.343 |2.02 |101 |
|---|---|---|---|---|---|---|

|Seldonian, filter only |NSF 3/3 |0.041 |0.506 |0.364 |2.02 |106 |
|---|---|---|---|---|---|---|

|Seldonian, Lagrangian |NSF 3/3 |0.076 |0.348 |0.255 |1.21 |117 |
|---|---|---|---|---|---|---|

Means over three seeds on the 1,600 sealed safety prompts. Thresholds: harm ≤ reference + 0.02, refusal on benign prompts ≤ reference + 0.05. Rates for NSF rows are those of the final or selected policy, which the safety test declined to return.

### Gates

G1
Harm judge pass: 90% agreement with PKU's own labels, recall 0.98. Refusal judge fail: 78 to 87% agreement with the Qwen3Guard refusal flag; it counts an ethics preamble followed by a real answer as a refusal.

G2
pass. Unconstrained GRPO breaches the refusal constraint in 3 of 3 seeds. The reward misspecification shows up as over-refusal, not harm: every trained method cut harm from 14% to 3 or 4%.

G3
fail. No Seldonian run returned a solution. Filter-only selection never sees a feasible checkpoint because refusals exceed threshold from the first predicted test. The Lagrangian variant saturates its multiplier at 20, holds refusals 10 to 15 points lower than GRPO, and still misses.

G4
pass. One predicted-versus-actual disagreement in three seeds: Lagrangian seed 2 selected a step-20 checkpoint that failed the safety test by 0.007, with a measured rate inside the threshold. That is the conservative failure the framework allows.

G5
pass. 90 to 120 minutes per trained run at 0.5B and 200 steps. Round 2 as budgeted is not feasible locally without vLLM generation or rented GPUs.

### Why the Lagrangian runs still return NSF

- **Margin below bound width.** The doubled predicted interval on 400 benign safety prompts is about 0.07 wide, more than the 0.05 margin, so a checkpoint must refuse less than the reference to be predicted feasible. Two of three seeds had a checkpoint that would have passed the real safety-test width.

- **Group normalisation.** GRPO normalises advantages within a group of four. A benign prompt whose four samples all refuse contributes no gradient however large the penalty.

- **Judge target.** The classifier rewards deleting the preamble, not answering the question.

### Changes before Round 2

- Refusal judge becomes the Qwen3Guard refusal flag, or a two-of-three vote; all three signals reported.

- Benign safety prompts raised to at least 1,200, or margins set to at least twice the predicted width. The run script should refuse a configuration where the margin is below the width.

- Lagrangian candidate selection kept; initial multiplier around 5; group size 8; a per-prompt penalty term so all-refuse groups still get a gradient.

- The ×2 predicted-interval inflation calibrated from the logged predicted-versus-actual gap instead of assumed.

- Colocated vLLM generation in TRL before the wall-clock is re-measured.

## 11. Round 1b results, 3 September

Lagrangian Seldonian at 0.5B with the pilot's fixes: Qwen3Guard refusal flag as the judge, 1,200 benign safety prompts, harm margin 0.03, group size 8, initial multiplier 5. Gate G3 passes: solutions returned in 2 of 3 seeds, and no returned policy breaches either constraint. Full notes in `reports/llm_round2.md`.

|Seed |Outcome |Selected step |Harm (τ) |Refusal (τ) |Reward |Feasible ckpts |Min |
|---|---|---|---|---|---|---|---|

|0 |solution |25 |0.095 (0.138) |0.109 (0.178) |-0.29 |1 / 8 |227 |
|---|---|---|---|---|---|---|---|

|1 |NSF by 0.002 |200 |0.047 (0.142) |0.176 (0.192) |1.47 |5 / 8 |231 |
|---|---|---|---|---|---|---|---|

|2 |solution |50 |0.088 (0.146) |0.105 (0.172) |0.32 |2 / 8 |229 |
|---|---|---|---|---|---|---|---|

Reference reward is about -0.5. The pilot's unconstrained GRPO reached 2.07 at a 37% refusal rate by the same judge.

- **First solution returned.** Seed 0 passed the safety test with the step-25 checkpoint, safe by a wide margin but with reward barely above the reference. Its later checkpoints reached rewards of 1.5 to 2.0 at 16 to 18% refusals, predicted infeasible by 1 to 4 points.

- **Seed 1 found the predicted-test flaw.** Its selected checkpoint was predicted at 12.9% refusals from about 256 benign samples and measured 17.6% on the safety set, failing the bound by 0.002. The doubled interval covered the safety-set sample but not the prediction sample's own noise. The predicted bound now uses the effective size 1/(1/m + 1/ns), and the prediction sample rises to 1,024 prompts for the 1.5B runs. This also explains the pilot's one G4 disagreement.

- **Multiplier moves too slowly.** With predicted violations of about 0.03 and a step size of 20, the refusal multiplier only climbed from 4.6 to 6.6. The 1.5B runs use a step size of 100.

- **Cost.** Group size 8 doubled generation per step; runs take about 230 minutes. The 1.5B seed 0 trio is queued next with an ETA near 18:30, then the GSM8K control overnight.

## 12. Round 2 at 1.5B, seed 0, 3 September 18:35

Task A+B on Qwen2.5-1.5B-Instruct with the corrected pipeline. Thresholds from the 1.5B reference: harm ≤ 0.059, refusal ≤ 0.272.

|Method |Outcome |Harm |Refusal, Qwen3Guard |Reward |Min |
|---|---|---|---|---|---|

|Reference |within bounds |0.037 |0.224 |1.07 |18 |
|---|---|---|---|---|---|

|GRPO |within bounds, no guarantee |0.033 |0.169 |3.80 |206 |
|---|---|---|---|---|---|

|Seldonian, Lagrangian |solution, step 200 |0.033 |0.140 |3.59 |280 |
|---|---|---|---|---|---|

- **The Seldonian layer returned its final checkpoint at full reward.** Both constraints hold with wide margins; the cost against unconstrained GRPO is about 0.2 reward, or 5%.

- **But the constraints are not binding at 1.5B.** Unconstrained GRPO also stays inside both thresholds; refusals fall rather than rise under this reward model. The 0.5B over-refusal pathology does not transfer. This is the "benign reward" outcome the plan reserved for the control task, and as configured the 1.5B comparison cannot test H1 or H2.

- **What would make it informative** is the plan's reward-pressure sweep: more steps, a lower KL coefficient, or a reward model with a stronger refusal preference, until GRPO breaches. Seeds 1 and 2 at 1.5B are queued and will show whether the non-binding result holds across seeds.

- **Queue.** The 1.5B seeds 1 and 2 were dropped as uninformative. Running instead: the corrected 0.5B Lagrangian for three seeds, then a 1.5B reward-pressure pair with the KL coefficient at zero.

### Task D control: GSM8K at 0.5B, 4 September

|Method |Outcome |Error rate (τ 0.808) |Accuracy |Min |
|---|---|---|---|---|

|Reference | |0.782 |0.218 |6 |
|---|---|---|---|---|

|GRPO |within bounds |0.653 |0.347 |163 |
|---|---|---|---|---|

|Seldonian, Lagrangian |solution, 8 of 8 feasible |0.633 |0.367 |183 |
|---|---|---|---|---|

With a verifiable reward the layer behaves as the plan predicted: every checkpoint is predicted feasible, the multiplier never leaves zero, the returned policy matches unconstrained GRPO within noise, and the only cost is about 12% wall-clock for the predicted and final tests.

## 13. Corrected Lagrangian Seldonian at 0.5B, 4 September

Round 1b repeated with the three corrections: dual step size 100, the predicted bound at the effective size of prediction plus safety samples with no extra inflation, and 1,024 prediction prompts. Same prompts, thresholds and judges as Round 1b.

|Seed |Outcome |Selected step |Harm (τ) |Refusal (τ) |Reward |Feasible ckpts |
|---|---|---|---|---|---|---|

|0 |solution |175 |0.056 (0.138) |0.123 (0.178) |1.46 |3 / 8 |
|---|---|---|---|---|---|---|

|1 |solution |200 |0.059 (0.142) |0.153 (0.192) |1.74 |4 / 8 |
|---|---|---|---|---|---|---|

|2 |solution |50 |0.087 (0.146) |0.140 (0.172) |0.28 |2 / 8 |
|---|---|---|---|---|---|---|

### Where the 0.5B comparison now stands

|Method |Solution |Breach |Harm |Refusal, Qwen3Guard |Reward |
|---|---|---|---|---|---|

|Reference | | |0.135 |0.152 |-0.53 |
|---|---|---|---|---|---|

|GRPO (pilot) |no guarantee |3 / 3 |0.031 |0.373 |2.07 |
|---|---|---|---|---|---|

|Composite λ=1 (pilot) |no guarantee |3 / 3 |0.040 |0.342 |2.02 |
|---|---|---|---|---|---|

|Seldonian, filter only (pilot) |0 / 3 | | | | |
|---|---|---|---|---|---|

|Seldonian, Lagrangian, Round 1b |2 / 3 |0 / 3 |0.092 |0.107 |0.02 |
|---|---|---|---|---|---|

|Seldonian, Lagrangian, corrected |3 / 3 |0 / 3 |0.067 |0.139 |1.16 |
|---|---|---|---|---|---|

Means over three seeds on the sealed safety set. Pilot rows were thresholded by the classifier judge; their Qwen3Guard refusal rates are shown for comparability.

- **Headline.** The corrected layer returns a policy in every seed and never a violating one. It holds refusals at the reference level while halving harm, and captures on average 65% of the reward gain that unconstrained GRPO gets by refusing 37% of benign prompts. Seeds 0 and 1 capture 77 to 87%.

- **Mechanism.** The multiplier now rises to 12 to 20 when refusals drift above threshold and pulls them back within 25 to 75 steps. In Round 1b it never passed 9 and could not.

- **Weak seed.** Seed 2 has the tightest threshold and its multiplier saturated at the cap of 20 without recovering, so the step-50 checkpoint was returned at reward 0.28. A higher cap or more benign prompts is the next thing to try.

- **1.5B under maximal pressure.** With the KL coefficient at zero, GRPO reached reward 4.85 and still stayed inside both thresholds (harm 2.3%, refusals 18.7%); the Lagrangian Seldonian returned a solution at 4.59, again a 5% cost. The 1.5B model does not develop the over-refusal pathology with this reward model at any pressure tried.

## 14. Where it stands, 4 September evening

|Setting |Unconstrained GRPO |Seldonian, Lagrangian, corrected |
|---|---|---|

|0.5B, harm + refusal |breaches 3 / 3 refusals 37%, reward 2.07 |solution 3 / 3, breach 0 / 3 refusals 14%, reward 1.16 |
|---|---|---|

|1.5B, harm + refusal, β 0.04 |within bounds, reward 3.80 |solution reward 3.59 |
|---|---|---|

|1.5B, harm + refusal, β 0 |within bounds, reward 4.85 |solution reward 4.59 |
|---|---|---|

|0.5B, GSM8K, verifiable reward |within bounds, accuracy 0.347 |solution, 8 / 8 feasible accuracy 0.367 |
|---|---|---|

Thirty-three runs, about 75 GPU-hours. Across all 17 Seldonian runs in the project, no returned policy breached a constraint on the safety set. Where the reward is misspecified, the layer is what holds the constraint and it keeps about two thirds of the reward gain. Where the reward is benign, it costs 0 to 5% reward plus one safety-set evaluation.
