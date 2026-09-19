<!-- Recovered 2026-09-19 from the published claude.ai artifact; text extracted from llm_round4_round5_2026-09-09.html -->

Seldonian LLM Round 4

- 

- 

seldonian-fairness · branch llm-seldonian-rl · Round 4 2026-09-06/07, Round 5 09-07 to 09-09

# Round 4: an evaluation design for the Seldonian LLM framework (overnight 2026-09-06/07)

**In one screen.** The Student-t bound used in Rounds 1-3 fails 1.2-1.8× more often than delta at the 3% harm rates trained policies reach; Clopper-Pearson is valid and nearly as tight. A synthetic environment running the unchanged pipeline over thousands of trials shows the Lagrangian arm 50× inside delta and filter-only selection returning a solution in a tenth of trials. A compliance bonus of 5 was absorbed by safe compliance; at 10 the harm rate rose to within 2 points of the threshold in 150 steps. DiscrimEval gives a real 3-point racial parity gap at 0.5B, but sampled decisions at temperature 1 make the paired bound useless until a probability feature replaces them. Round 5 built a task that breaches on demand: a brevity constraint under a reward that pays 8-16 per violation sends GRPO to 89-92% over the cap against a 52-56% threshold in 5 of 5 runs; the Seldonian arm returns a certified policy at 5-13% in 5 of 5, finding its own penalty, while a fixed penalty set with knowledge of the pressure matches it on the constraint and beats it on base reward.

Companion to `llm_round1_pilot.md` and `llm_round2.md`. This round does not add another arm to the harm / over-refusal task; it asks what data and what safety tests would let the framework itself be tested properly, builds three of them, and reports first numbers from each.

## 1. What the first three rounds could not tell us

Every constraint so far had one shape, `P(judge = 1 | group) <= reference + margin`, and every experiment had three seeds of a run that costs two to five GPU-hours. That setup has four blind spots.

- **The guarantee itself was never measured.** A Seldonian method promises that the probability of returning a violating policy is at most delta. Seventeen runs with no breach is consistent with any failure rate below about 15%, so the claim "delta = 0.1 holds" has not been tested. Testing it needs hundreds of independent runs, which the real LLM pipeline cannot provide.
- **Whether the constraint binds was luck.** At 0.5B the reward model happened to drive over-refusal; at 1.5B and on GSM8K nothing bound, and those runs said nothing about the framework beyond "the layer costs 0-5%".
- **The guarantee is stated against a frozen judge, at rates of 3-15%, with a Student-t bound.** The t interval is not distribution-free, and at these rates and sample sizes its coverage is an empirical question. The library has since gained distribution-free bounds (Bentkus, betting, convex-order) that were never used on the LLM path.
- **Only single group rates.** Fairness constraints in the rest of the library are differences between groups; nothing on the LLM path could express one.

## 2. What was built tonight

|piece |where |purpose |
|---|---|---|
|synthetic contextual-bandit backend |`seldonian/llm/synthetic.py`, `scripts/synthetic_calibration.py` |runs the unchanged `SeldonianLLMPolicy` pipeline (split, reference thresholds, predicted tests, Lagrangian dual ascent, checkpoint selection, safety test) against a policy whose true violation rate is an exact expectation; hundreds of trials per minute |
|bound calibration by resampling |`scripts/resample_calibration.py` |resamples safety sets from the pooled judge labels of real 0.5B policies and measures each bound's empirical failure rate against its nominal delta |
|open-ended constraints |`seldonian/llm/constraints.py` |measurement-vector expressions over group-conditional means (parity, ratios, bounded scores), paired counterfactual differences, two-sample differences; plug into the policy through a `measure` hook |
|reward-pressure task |`--compliance-bonus` in `scripts/run_llm_rl.py`, `BonusReward` |pays the policy for complying on adversarial prompts so the harm constraint binds by construction; `alpha` is the pressure knob |
|paired counterfactual fairness task |`seldonian/llm/discrim.py` |DiscrimEval decision scenarios paired across race / gender / age with a decision-parity constraint on the per-pair difference; ready to run, not run tonight |
|bound registry on the LLM path |`seldonian/llm/policy.py` |every one-sample bound in `seldonian.bounds` can now be named by a constraint; the margin check evaluates the chosen bound's width |

### 2.1 Open-ended constraints

The unifying move is to stop thinking in rates. Each episode yields a vector of bounded features (judge flags, judge probabilities, clipped scores, normalised length, a decision) and group tags. A constraint is an expression over group-conditional means of those features, and its upper bound is propagated through the expression with the interval arithmetic of `RandomVariable`. The delta budget is split so that every measure's two-sided interval holds simultaneously (union bound; `delta / 2k` per side for `k` measures). Three shapes get special, tighter treatment:

- **paired difference** (`PairedDifferenceConstraint`): for counterfactual prompt pairs the per-pair difference lives in [-1, 1] and one one-sample bound on its mean replaces two group bounds;
- **two-sample difference** (`TwoSampleDifferenceConstraint`): unpaired group differences use the library's Bentkus / convex-order difference bounds;
- **plain rate** (`rate_constraint`): reproduces the built-in `Constraint` exactly, which is the regression test.

### 2.2 The synthetic environment

A population of contexts `x ~ N(0, I_8)` with a group tag, four actions, a ground-truth violation probability `p_v(x, a) = sigmoid(u_a . x + b_a)` and a reward `w_a . x + rho * p_v(x, a) + noise`. `rho` is the pressure knob: at `rho = 0` the reward and the constraint are unrelated, at `rho > 0` the better actions are the more violating ones. The policy is softmax-linear, trained with the same group-normalised advantage as GRPO plus a KL penalty to the uniform reference. The judge reads the drawn violation label and can flip it with a given sensitivity / specificity, which separates "guarantee with respect to the judge" from "guarantee with respect to the truth". Because the policy is queryable and the population is finite, the true violation rate of any returned policy is an exact sum, so the delta-calibration question has a ground-truth answer.

### 2.3 Tasks that bind by construction

- **Compliance pressure (run tonight).** On the reference 0.5B policy the reward model pays about 3.7 points more for refusing an adversarial prompt than for complying, and 27.8% of compliant answers are judged harmful against 0.8% of refusals. A bonus `alpha * (1 - refusal)` on adversarial prompts flips that preference once `alpha` exceeds the gap, so unconstrained GRPO must raise the harm rate, and `alpha` traces a frontier instead of one point.
- **Decision parity (ready).** DiscrimEval has 70 scenarios filled with 5 races, 3 genders and 9 ages, each ending in a yes/no question; 1,890 white / Black pairs differ in one token. The constraint is `|mean over pairs of (yes_a - yes_b)| <= epsilon`, plus a non-answer rate ceiling so the policy cannot satisfy parity by refusing to decide. The pressure knob is a bonus for "yes" on one group only.

## 3. Results

Sections 3.1 and 3.2 are the cheap, high-trial studies; 3.3 and 3.4 are the GPU runs.

### 3.1 Is the safety-test bound honest? Resampling real judge labels

`uv run scripts/resample_calibration.py`; outputs in `results/calibration/`. The harm labels (Qwen3Guard) of the pilot's three reference-policy safety sets are pooled (3,600 adversarial episodes, harm rate 0.161) and likewise for the pilot's GRPO policies (3,600 episodes, harm rate 0.034, the low rate a trained policy actually has). The pooled mean is taken as the truth, safety sets of size n are resampled 5,000 times, and each bound's one-sided upper limit at delta is checked against it. Cells are failure rate P(upper < truth) / mean width.

GRPO policy, harm rate 0.034, delta = 0.1 (nominal failure rate 0.100):

|bound |n=200 |n=400 |n=800 |n=1200 |n=2400 |
|---|---|---|---|---|---|
|ttest (used so far) |**0.184** / 0.016 |**0.115** / 0.012 |**0.121** / 0.008 |0.101 / 0.007 |0.101 / 0.005 |
|clopper_pearson |0.084 / 0.023 |0.067 / 0.015 |0.083 / 0.010 |0.074 / 0.008 |0.084 / 0.005 |
|bentkus |0.029 / 0.032 |0.035 / 0.021 |0.034 / 0.014 |0.023 / 0.011 |0.032 / 0.007 |
|chernoff_kl |0.029 / 0.034 |0.016 / 0.023 |0.019 / 0.016 |0.014 / 0.012 |0.019 / 0.009 |
|betting_mixture |0.007 / 0.038 |0.016 / 0.025 |0.011 / 0.017 |0.009 / 0.014 |0.010 / 0.009 |
|hoeffding, anderson |0.000 / 0.076 |0.000 / 0.054 |0.000 / 0.038 |0.000 / 0.031 |0.000 / 0.022 |

Reference policy, harm rate 0.161, delta = 0.1: ttest 0.136, 0.105, 0.094, 0.112, 0.096; clopper_pearson 0.098, 0.081, 0.079, 0.096, 0.086; bentkus 0.027-0.032; betting_mixture 0.008-0.009. At delta = 0.05 the ttest failure rate on the GRPO pool is 0.084, 0.067, 0.083, 0.074, 0.067: above nominal at every n up to 2,400.

Reading:

- **The Student-t bound is anti-conservative at the rates that matter.** At a 3% harm rate it fails 1.2-1.8 times more often than delta for n <= 800 and is still at or above delta at n = 2,400. Every "solution" reported in Rounds 1-2 rests on it. The pilot's harm rates after training were 3-6%, exactly this regime.
- **Clopper-Pearson is the right default for 0/1 judge labels.** It is exact for Bernoulli samples, its failure rate sits just under delta at every n, and its width is within 5-15% of the t interval for n >= 800.
- **Bentkus is the tightest distribution-free choice** (about 1.4x the t width, failure about 0.3 delta) and the one to use for non-binary features (clipped scores, normalised lengths). Hoeffding, Anderson and empirical Bernstein are 2-5x wider and never fail; betting_mixture sits in between.
- Cost is irrelevant: the slowest bound (Bentkus) takes 5 ms per call at n = 2,400.

Caveat: resampling from a finite pool measures coverage at the observed rate, not prompt-distribution effects, and the three seeds' safety sets overlap in prompts.

**Change made:** every bound in `seldonian.bounds` can now be named with `--bound` on the LLM path, and the margin check evaluates the named bound's width. The Round 4 GPU runs below were launched before this study finished and still use `ttest`; the recommendation is `clopper_pearson` from Round 5 on.

### 3.2 Does the pipeline honour delta? The synthetic environment

`scripts/synthetic_calibration.py`; tables and per-trial JSONL in `results/synthetic/`. Each trial draws a fresh population of 20,000 contexts, splits n prompts 60/40, and runs the unchanged `SeldonianLLMPolicy` (200 steps, group 8, predicted test every 25 steps on 512 prompts, dual ascent with lambda0 = 5, eta = 100, cap 20). One trial takes 0.06 s, so every row below is 500-1,000 independent trials. The environment was tuned so that, at pressure 1, the reference policy violates 13% of the time, the threshold is the true reference rate + 0.03, and unconstrained GRPO breaches it in a clear majority of trials. `unsafe` is P(solution returned and true rate > threshold), the event the guarantee bounds by delta = 0.1; `viol|sol` is the same conditional on a solution, which the guarantee does not bound.

**(a) delta calibration across bounds, pressure 1, n = 1,000 (400 safety prompts)**

|method |bound |solution |unsafe |95% upper |true rate given sol |reward given sol |
|---|---|---|---|---|---|---|
|grpo | |1.00 |**0.830** |0.857 |0.208 |0.85 |
|seldonian_lag |ttest |0.81 |0.002 |0.009 |0.128 |0.77 |
|seldonian_lag |clopper_pearson |0.82 |0.000 |0.006 |0.124 |0.76 |
|seldonian_lag |bentkus |0.84 |0.000 |0.006 |0.114 |0.75 |
|seldonian_lag |betting_mixture |0.86 |0.000 |0.006 |0.105 |0.73 |
|seldonian_lag |hoeffding |0.91 |0.000 |0.006 |0.094 |0.71 |

Reference: true rate 0.13, reward 0.14. The Lagrangian layer is 50x inside delta with every bound, because dual ascent parks the returned policy at the reference rate; the safety test is rarely the binding element. A wider bound raises the solution rate (the predicted g is larger, the multiplier grows faster, the policy ends up more conservative) at a reward cost, which is the same mechanism as the inflation result in (e).

**(b) sample size, pressure 1, ttest**

|method |n |solution |unsafe |viol given sol |true rate given sol |reward given sol |width |
|---|---|---|---|---|---|---|---|
|seldonian_lag |200 |0.72 |0.000 |0.000 |0.100 |0.70 |0.043 |
|seldonian_lag |500 |0.74 |0.000 |0.000 |0.119 |0.75 |0.029 |
|seldonian_lag |1000 |0.81 |0.002 |0.002 |0.128 |0.77 |0.022 |
|seldonian_lag |2000 |0.87 |0.000 |0.000 |0.131 |0.78 |0.015 |
|seldonian_lag |5000 |0.89 |0.000 |0.000 |0.134 |0.78 |0.010 |
|seldonian (filter only) |200 |0.08 |0.028 |0.333 |0.165 |0.85 |0.058 |
|seldonian (filter only) |1000 |0.09 |0.004 |0.043 |0.146 |0.83 |0.026 |
|seldonian (filter only) |5000 |0.13 |0.004 |0.030 |0.146 |0.84 |0.012 |

Filter-only selection reproduces the pilot's failure mode exactly: it sees 0.3-0.7 feasible checkpoints out of 8 and returns a solution in about a tenth of trials. At small n a third of those solutions truly violate, yet the unconditional unsafe rate (2.8%) still respects delta. The predicted-minus-actual gap is a steady -0.7 points (the selected checkpoint looked slightly better than it is: winner's curse), which is what the effective-n predicted bound absorbs.

**(c) pressure sweep, n = 1,000: the phase diagram**

|pressure |grpo unsafe |grpo true rate |grpo reward |lag solution |lag unsafe |lag true rate |lag reward |
|---|---|---|---|---|---|---|---|
|0 |0.184 |0.144 |0.67 |0.88 |0.000 |0.119 |0.65 |
|0.5 |0.524 |0.175 |0.75 |0.83 |0.000 |0.126 |0.71 |
|1 |0.830 |0.208 |0.85 |0.81 |0.002 |0.128 |0.77 |
|2 |0.996 |0.263 |1.08 |0.79 |0.000 |0.128 |0.89 |
|4 |1.000 |0.325 |1.68 |0.76 |0.000 |0.130 |1.13 |

GRPO goes from mostly inside to always violating; the Seldonian layer holds the true rate at the reference level at every pressure, its solution rate falls slowly (0.88 to 0.76) and the reward gap to GRPO grows with pressure, which is the price of not taking the violating actions. This is the curve the real tasks should be read against: the 0.5B harm / over-refusal task sits near pressure 1-2, the 1.5B and GSM8K runs near pressure 0.

**(d) judge noise, pressure 1, n = 1,000, thresholds measured with the noisy judge**

|judge (sensitivity, specificity) |solution |judge-level violation given sol |true violation given sol |true rate given sol |
|---|---|---|---|---|
|(1.0, 1.0) |0.81 |0.002 |0.002 |0.128 |
|(0.8, 0.95) |0.67 |0.003 |0.036 |0.133 |
|(0.6, 1.0) |0.65 |0.000 |0.059 |0.141 |
|(1.0, 0.9) |0.70 |0.000 |0.011 |0.125 |
|grpo, (0.6, 1.0) |1.00 |0.620 |0.808 |0.207 |

The guarantee holds exactly at the judge level in every row. A judge that misses 40% of violations lets 30x more true violations through (5.9% of solutions against 0.2%), still under delta here only because dual ascent is conservative. False positives are harmless: the policy becomes more conservative and NSF rises. This is the quantitative case for calibrating the LLM judges' sensitivity on hand labels and correcting the threshold for it.

**(e) predicted-test inflation and dual step, pressure 1, n = 1,000**

|inflation |eta |solution |unsafe |true rate given sol |reward given sol |
|---|---|---|---|---|---|
|1.0 |20 |0.77 |0.000 |0.125 |0.76 |
|1.0 |100 |0.81 |0.002 |0.128 |0.77 |
|2.0 |20 |0.96 |0.000 |0.109 |0.74 |
|2.0 |100 |0.97 |0.002 |0.108 |0.74 |

Doubling the predicted interval buys 16 points of solution rate for 4% reward. The Round 1c decision to drop the x2 inflation traded solution rate for reward; on this evidence the x2 is the better default when NSF is the expensive outcome.

**(f) stress test: margin 0 and 0.01, 1,000 trials per row.** With the threshold at the true reference rate the bound is the only safeguard. Filter-only at n = 500 returns a solution in 3% of trials and half of those truly violate, but the unconditional unsafe rate is 1.2% (95% upper 1.9%) against delta = 0.1, with ttest and Clopper-Pearson alike; the Lagrangian arm stays at 0.0-0.2% with a 72-80% solution rate. The guarantee is on the joint event, and it holds; the high conditional rate is what a practitioner sees, and it is the argument for the Lagrangian arm over filter-only rather than against the bound.

Caveats: one constraint, a 36-parameter linear policy, a reward scale of about 1 where lambda0 = 5 dominates. The environment tests the pipeline's logic and the bound's coverage, not GRPO's optimisation dynamics on a transformer.

### 3.3 The compliance-pressure task at 0.5B (results/llm_r4)

Settings: Qwen2.5-0.5B-Instruct, LoRA r 16, GRPO group 4, 8 prompts per step, 150 steps, beta 0.04, 3,000 PKU + 3,000 benign prompts split 60/40 (D_s = 1,200 + 1,200), Round 1c seed-0 thresholds (harm 0.108 + 0.03 = 0.138 on all prompts; refusal 0.128 + 0.05 = 0.178 on benign), predicted test every 30 steps on 768 prompts, ttest bound, delta 0.1, Lagrangian lambda0 5, eta 100, cap 50. Reward = Skywork + alpha * (1 - Qwen3Guard refusal flag) on adversarial prompts. Reported reward is the base Skywork score on D_s.

|alpha |method |outcome |harm (ub) / tau |refusal on benign (ub) / tau |reward |selected |feasible |min |
|---|---|---|---|---|---|---|---|---|
| |reference | |0.101 / 0.138 |0.112 / 0.178 |-0.83 | | |11 |
|5 |grpo |within bounds, no guarantee |0.079 / 0.138 |0.117 / 0.178 |0.44 |final | |129 |
|5 |seldonian_lag |**solution**, step 120 |0.100 (0.111) / 0.138 |0.094 (0.108) / 0.178 |0.07 |120 |5 / 5 |134 |
|10 |grpo |within bounds, no guarantee |0.115 / 0.138 |0.065 / 0.178 |0.09 |final | |130 |
|10 |seldonian_lag |**solution**, step 120 |0.117 (0.127) / 0.138 |0.076 (0.088) / 0.178 |-0.11 |120 |5 / 5 |151 |

**alpha = 5 does not bind.** Unconstrained GRPO lowered the harm rate from 0.101 to 0.079 while collecting the bonus. Diagnostics on the adversarial half of D_s explain why (refusal by the Qwen3Guard flag; reward is the Skywork score):

|policy |refusal on adversarial |harm on adversarial |reward given refusal |reward given compliance |harm given compliance |
|---|---|---|---|---|---|
|reference |0.454 |0.147 |2.09 |-1.71 |0.258 |
|grpo, alpha 5 |0.479 |0.107 |3.57 |-0.37 |0.190 |
|grpo, alpha 10 |0.376 |0.170 |3.62 |-0.68 |0.255 |

The reward model prefers a refusal to a compliant answer by 3.8 points and to a harmful compliant answer by about 6 (harmful answers score around -4). A bonus of 5 therefore makes safe compliance the best option but leaves harmful compliance behind refusal, and GRPO found exactly that: the same refusal rate, better-scoring refusals, and compliant answers that are less often harmful. The knob has to exceed the reward model's own safety preference before the constraint can bind, which is why the second pressure point was moved from 2.5 to 10.

**alpha = 10 is where the pressure starts to bite.** Adversarial refusals fell from 45% to 38%, harm on adversarial prompts rose from 14.7% to 17.0%, benign refusals fell to 6.5%, and the base reward dropped to 0.09 (the policy is now paid for compliance, not for what the reward model likes). The pooled harm rate the constraint is written on, 0.115, is diluted by the benign half of D_s (6% harm) and still 2.3 points under the 0.138 threshold after 150 steps. Two lessons for the task design: the harm constraint should be stated on the adversarial group (0.170 against a reference of 0.147, where alpha 10 is within 0.7 points of a +0.03 margin), and 150 steps at group 4 is about half the training the pilot's breaches needed.

The Lagrangian arm at alpha 10 saw the pressure: its predicted harm bound climbed from 0.105 at step 30 to 0.135 at step 150, three thousandths under the threshold, and the harm multiplier moved off zero only in the first 30 steps. It returned the step-120 checkpoint (harm 0.117, upper 0.127) at a reward of -0.11 against GRPO's 0.09. With 300 steps the trajectory would cross the threshold and the arms would separate; at 150 steps this is the edge of the binding regime.

Two smaller observations. The Lagrangian arm behaved as in the non-binding 1.5B and GSM8K runs: every predicted test feasible, both multipliers at zero from step 60, a solution returned, and it captured 71% of GRPO's reward gain (0.90 of 1.27 above the reference). Its selected checkpoint (step 120) had a predicted reward of 0.29 and a safety-set reward of 0.07: the predicted test on 768 candidate prompts overestimates the reward of the checkpoint it picks, the winner's curse the synthetic sweep measured as a -0.7-point rate gap, here visible on the reward instead.

### 3.4 Decision parity on DiscrimEval: the reference measurement (results/llm_r4/d)

`uv run scripts/run_llm_rl.py --task discrim --method reference --bound clopper_pearson`, 8 minutes at 0.5B. 1,890 white / Black pairs (attribute `race`, all 70 scenarios, 9 ages, 3 genders) split by pair into 1,120 + 770 pairs; each prompt is the filled scenario plus "Please answer the above question with either "yes" or "no", followed by a one-sentence justification." Constraints: `|mean over pairs of (yes_white - yes_Black)| <= 0.05` (absolute) and non-answer rate <= reference + 0.05.

|quantity |value |
|---|---|
|yes-rate, white fills / Black fills (safety set) |0.710 / 0.740 |
|paired mean difference (white - Black) |-0.030 |
|standard deviation of the per-pair difference |0.60 |
|one-sided t width on the paired mean at delta 0.05, n = 770 pairs |0.036 |
|non-answer rate |0.002 (reference on D_c: 0.006) |
|Skywork reward, mean words |0.38, 81 |

Three things follow.

- **The model decides, and it decides differently by race.** It answers in 99.8% of cases and favours Black fills by 3 points, the same direction Anthropic reported for larger models. So the task has a real, non-zero parity gap to constrain before any pressure is added.
- **At temperature 1 the counterfactual pairing buys almost nothing.** Two independent Bernoulli(0.72) draws would give a difference with standard deviation 0.635; the paired differences have 0.60, so the pair members' decisions are nearly uncorrelated: the 0.5B model's yes/no on a given scenario is close to a coin flip across samples. As a result the reference's own upper bound on the parity gap, 0.030 + 0.036 = 0.066, already exceeds the 0.05 threshold; the run script would return NSF for the reference policy. The fix is not a larger safety set but a better feature: score each prompt by the policy's probability of "yes" (a bounded feature in [0, 1] via `CallableFeature`, one forward pass, no sampling noise) or by a greedy decision. With sampling noise gone the paired difference reflects the counterfactual effect alone and the interval collapses.
- **The non-answer ceiling is nearly free** (0.2% against a 5.6% threshold), so a policy cannot satisfy parity by declining to decide without breaching it.

No trained arm was run on this task tonight; the queue was given to the pressure task. The driver supports `--bias-bonus` (a bonus for "yes" on white fills) with `grpo` and `seldonian_lag`; the Lagrangian arm gets a signed per-episode parity penalty (`ParityPenalty`) whose batch mean is the subgradient of `|mean d|`.

### 3.5 Round 5 (2026-09-07, daytime): a task built to breach

Asked for a task with a high breach probability, the design was: benign prompts with "Answer in at most 80 words." appended, the verifiable constraint `P(words > 120) <= reference + 0.05` (Clopper-Pearson, delta 0.1), and a reward with an injected, tunable pressure toward long answers. Skywork has no length bias on plain benign prompts (correlation 0.00 between score and word count), so the pressure has to be injected. Same training settings as Round 4. Results in `results/llm_r5/`.

|pressure |method |over-cap rate / tau |reward |words p50 |min |
|---|---|---|---|---|---|
| |reference |0.500 / 0.561 |0.48 |120 |6 |
|linear, 3 * min(words, 300) / 300 |grpo |0.062 / 0.561 |2.36 |74 |81 |
|step, 4 * 1[words > 120] |grpo |0.368 / 0.561 |1.50 |101 |102 |
|step, 16 * 1[words > 120] |grpo |**0.904 / 0.561, breach** |-0.84 |194 |106 |
|step, 16 * 1[words > 120] |seldonian_lag |**0.063 (ub 0.073) / 0.561, solution at step 90** |2.05 |67 |107 |

Both first attempts went the wrong way: the policy got shorter. With the brevity instruction in the prompt the reward model rewards obeying it, and strongly:

|words |0-60 |60-80 |80-100 |100-120 |120-160 |160-200 |200+ |
|---|---|---|---|---|---|---|---|
|Skywork score, reference samples |1.32 |1.96 |1.82 |0.85 |0.65 |-1.01 |-1.93 |

The marginal gap between answers under and over the cap is 2.0 points, and it is steeper at the tails (a 200-word answer scores 3.9 below an 80-word one). The linear bonus of 3 adds only 1.0 across that whole range; the step bonus of 4 should have flipped the marginal preference, but the training log shows rollout lengths flat at 100-130 tokens for all 150 steps and the reward rising through the reward model alone. Within a GRPO group the relevant quantity is the reward spread among four answers to the same prompt (standard deviation 2.6-3.2 in the log), and a bonus that only lands on the quarter of rollouts over the cap did not dominate it. The bonus of 16 removes the ambiguity: it exceeds any within-group spread, so the direction of the gradient is fixed.

**At 16 the task breaches decisively**: 90% of safety-set answers exceed the cap (median 194 words, a third at the 256-token ceiling), the base reward drops below the reference, and the policy pads answers by switching into Chinese mid-sentence, a second, unconstrained drift that a language-match constraint would catch. This is the first real-LLM setting in the project where the unconstrained baseline breaches by 34 points and the Seldonian arm has to hold a policy back rather than merely certify one.

**The Seldonian arm held it.** Its predicted-test trajectory is the whole story:

|step |predicted over-cap rate (ub) |feasible |multiplier after the test |base reward |
|---|---|---|---|---|
|30 |0.914 (0.930) |no |41.9 |-1.46 |
|60 |0.314 (0.343) |yes |20.1 |1.23 |
|90 |0.060 (0.076) |yes |0.0 |2.19 |
|120 |0.181 (0.206) |yes |0.0 |1.86 |
|150 |0.266 (0.294) |yes |0.0 |1.65 |

By step 30 the policy had breached exactly as GRPO did (91% over the cap); the dual step raised the multiplier to 42, which reversed the drift within 60 steps, and once the constraint had slack the multiplier decayed to zero and the pressure began pulling the policy back up. Candidate selection took the step-90 checkpoint; the safety test measured 0.063 with an upper bound of 0.073 against 0.561. Base reward 2.05 against the reference's 0.38 and GRPO's -0.84: because the injected pressure opposes the reward model, the constrained policy scores higher on the reward model than the unconstrained one, which is the "misspecified reward" outcome in its cleanest form. Two things to improve: the layer over-corrected (a 6% rate against a 56% threshold, so a policy nearer the threshold with a longer, more informative answer was left on the table), which a smaller eta or a proportional-integral dual update would fix; and the oscillation after step 90 shows the multiplier cap and decay need a floor once a constraint has been binding.

**Three seeds at bonus 16 (2026-09-08, `scripts/run_round5b.sh`).** Thresholds are per-seed reference rates + 0.05 on each seed's own split.

|seed |reference rate / tau |grpo rate, reward |seldonian_lag rate (ub), reward, selected step |multiplier peak |
|---|---|---|---|---|
|0 |0.496 / 0.561 |0.904 breach, -0.84 |0.063 (0.073), 2.05, step 90 |42 |
|1 |0.457 / 0.517 |0.917 breach, -1.05 |0.111 (0.123), 2.04, step 120 |49 |
|2 |0.507 / 0.531 |0.916 breach, -0.80 |0.051 (0.060), 2.04, step 90 |44 |

GRPO breaches 3 of 3 by 36-40 points; the Seldonian arm returns a solution 3 of 3 with 0 breaches, 4 of 5 checkpoints feasible in every seed, and the same trajectory each time: breached at step 30, multiplier to 42-49, reversed by step 60-90, then drift back toward the cap once the multiplier decays to zero (0.18-0.26 over the cap by step 150, still inside the threshold). Three seeds cannot bound the failure rate below about 0.5, so this is repeatability of the rescue, not a test of delta; the delta test remains the synthetic environment's job.

**Attribution: fixed-penalty composite arms at seed 0, bonus 16** (reward minus lambda times the violation flag, no predicted tests, no safety test, no certificate):

|arm |over-cap rate / tau |base reward |min |
|---|---|---|---|
|composite, lambda 16 (cancels the bonus exactly) |0.057 / 0.561 |2.45 |79 |
|composite, lambda 32 |0.028 / 0.561 |2.81 |72 |
|seldonian_lag (dual ascent, peak 42, decayed to 0) |0.063 (0.073) / 0.561 |2.05 |107 |

A hand-set penalty at or above the pressure does as well on the constraint and better on base reward, at two thirds of the wall clock. What the Seldonian arm added here is that it found a sufficient penalty on its own (the pressure, 16, is known to us but not to the algorithm) and returned a certificate; what it cost is the 30% overhead of the predicted and safety tests and an over-correction the fixed penalty did not suffer from. The comparison that matters is at a pressure where a fixed penalty chosen without knowing the pressure is wrong in one direction or the other, which is the bonus-8 block below.

**Bonus 8 at seed 0 (2026-09-09).** Meant as the marginal regime; it is not. GRPO breaches almost as hard as at 16 (0.888 over the cap, median 193 words), so the transition sits between bonus 4 (policy went shorter) and 8 and is sharp, as group-normalised advantages predict: once the bonus exceeds the within-group reward spread the gradient's sign flips for every group at once. But the Seldonian arm behaved differently from the bonus-16 runs, and more informatively:

|step |predicted over-cap rate (ub) |feasible |multiplier after |base reward |
|---|---|---|---|---|
|30 |0.135 (0.158) |yes |0.0 |1.80 |
|60 |0.260 (0.288) |yes |0.0 |1.59 |
|90 |0.561 (0.591) |**no** |3.0 |0.73 |
|120 |0.624 (0.653) |**no** |12.2 |0.72 |
|150 |0.487 (0.518) |yes |7.9 |1.19 |

The initial multiplier of 5 held the policy short through step 60 and decayed to zero; the pressure then carried the policy through the threshold at step 90, the dual step raised the multiplier to 12 and pulled it back under by step 150. Two checkpoints were predicted infeasible by 3-9 points and rejected; candidate selection returned the step-30 checkpoint (rate 0.135, upper 0.149 on the safety set, reward 1.82; GRPO -0.59). Unlike the bonus-16 runs, here the predicted test did real work: without it the final checkpoint (0.487, upper 0.518) would have been returned, feasible but by 4 points, and the two checkpoints across the threshold were the ones with the lowest reward anyway. This is the regime where the layer's selection, not only its penalty, changes the outcome.

The composite arm with a fixed penalty of 16 at bonus 8 (an over-penalty by a factor of two, chosen without knowing the pressure) lands at 0.055 over the cap with base reward 2.45 in 75 minutes. On this task over-penalising is free because the constraint points the same way as the reward model's own preference for short answers, so the brevity task cannot show the downside of a too-large fixed penalty. Showing that needs a task where the constraint opposes the reward model, which is the original over-refusal setting at 0.5B (penalising refusals costs Skywork score).

**Does the Seldonian layer work here? Summary of Round 5.**

|setting |grpo |composite (fixed penalty) |seldonian_lag |
|---|---|---|---|
|bonus 16, 3 seeds |breach 3/3 (0.90-0.92) |penalty 16: 0.057, reward 2.45; penalty 32: 0.028, reward 2.81 (seed 0) |solution 3/3, breach 0/3, 0.05-0.11, reward 2.04 |
|bonus 8, seed 0 |breach (0.888) |penalty 16: 0.055, reward 2.45 |solution, 0.135 (ub 0.149), reward 1.82; predicted test rejected 2 of 5 checkpoints |

Yes, in the sense that matters for the mechanism: in five of five runs where the unconstrained baseline breached by 33-40 points, the layer returned a certified policy inside the constraint, found the needed penalty on its own (peaks of 42-49 at bonus 16, 12 at bonus 8, decaying to zero once the constraint had slack), and at the lower pressure its predicted test discarded checkpoints that had crossed the threshold. No, in two senses that still stand: five runs cannot test delta (that remains the synthetic environment's result), and a fixed penalty set with knowledge of the pressure matches or beats it on base reward at two thirds of the wall clock, so the layer's advantage is adaptivity and the certificate, not reward. Its measured costs are a 30-45% wall-clock overhead, an over-correction to 5-13% against a 56% threshold, and a multiplier that decays to zero and lets the policy drift back (0.18-0.26 by step 150 at bonus 16, 0.49 at bonus 8). A floor on the multiplier once a constraint has bound, or a slower decay, is the obvious fix.

The general rule these three attempts establish for building a breaching task: the pressure must be paid on the constraint's own event (not a proxy), on the constraint's own prompts (no dilution), and its magnitude must exceed the reward model's within-group preference against that event, which is larger than the marginal gap. Measure the marginal gap on the reference samples first, then set the pressure to several times it.

## 4. What this changes

- **Use Clopper-Pearson for 0/1 judge labels, Bentkus for bounded scores.** The t bound fails 1.2-1.8x more often than delta at the 3% harm rates trained policies reach. Both are one flag away (`--bound clopper_pearson`), cost nothing, and the margin check already knows their widths.
- **State the guarantee at the judge level and calibrate the judge.** In the synthetic sweep a judge with 60% sensitivity let 30x more true violations through at an identical judge-level failure rate. Qwen3Guard's sensitivity on PKU human labels was 0.98 in the pilot (harm), but nothing comparable exists for the refusal flag; a 200-example hand-labelled set per judge is the cheapest experiment with the largest effect on what the guarantee means.
- **Reward pressure has to be defined against the constraint judge, not a proxy.** A compliance bonus of 5 was absorbed by safe compliance; 10 moved harm 2.3 points in 150 steps. The synthetic environment's `rho * p_v` is the clean version: pressure proportional to the violation probability itself. On the LLM side the equivalent is a bonus paid through the harm judge, which is a deliberate red-team reward and should be run as such, or a helpfulness-only reward model, which is the natural version and worth a fetch (PKU's beaver reward model is the canonical one, at 7B too large for this card).
- **Prefer the Lagrangian arm and the x2 predicted interval when NSF is the expensive outcome.** Filter-only selection returns a solution in a tenth of trials and half of those violate at small n; dual ascent returns one in 80% and holds the true rate at the reference level. Doubling the predicted interval buys 16 points of solution rate for 4% reward.
- **The winner's curse is real and should be reported.** The selected checkpoint looked 0.7 rate points better than it was in the synthetic sweep and 0.22 reward better at alpha 5; the effective-n predicted bound covers the rate, nothing covers the reward.
- **Paired counterfactual constraints need a noise-free feature.** With sampled decisions at temperature 1 the pairing is worthless on a 0.5B model; with the policy's yes-probability it is the tightest fairness constraint available.

## 5. Next runs, in order

- Brevity task, done for seeds 0-2 at bonus 16 and seed 0 at bonus 8 (section 3.5). Next on it: a multiplier floor or slower decay to stop the drift-back; bonus 5-6 for the true marginal regime; and the same three arms on a task where the constraint opposes the reward model (over-refusal at 0.5B), where a fixed over-penalty is not free.
- Compliance pressure at alpha 10 with 300 steps and the harm constraint on the adversarial group, `--bound clopper_pearson`, 3 seeds: the first real-LLM point where GRPO breaches and the Seldonian arm must hold it. (~4.5 h per seed.)
- DiscrimEval with a yes-probability feature and `--bias-bonus 2`: reference, grpo, seldonian_lag. Also the gender pairing (male / female) and the age pairing (20 / 70), which reuse the loader unchanged.
- Hand-label 200 responses per judge; report sensitivity / specificity; thread the correction into the threshold.
- Synthetic: two constraints with a shared delta, a non-binary feature (clipped score) under Bentkus, and a transformer toy LM to check that the dual-ascent conclusions survive sequence-level credit assignment.

## 6. Files

New: `seldonian/llm/synthetic.py`, `seldonian/llm/constraints.py`, `seldonian/llm/discrim.py`, `scripts/synthetic_calibration.py`, `scripts/resample_calibration.py`, `scripts/run_round4.sh`, `scripts/run_round4b.sh`, `tests/test_llm_synthetic.py`, `tests/test_llm_constraints.py`, `tests/test_llm_discrim.py`. Modified: `seldonian/llm/policy.py` (bound registry, `measure` hook, generic predicted width, nested reward unwrapping), `seldonian/llm/rewards.py` (`BonusReward`, group-aware penalties), `seldonian/llm/judges.py` (`cache_only`), `scripts/run_llm_rl.py` (tasks `discrim`, `--compliance-bonus`, `--bias-bonus`, `--bound` over the full registry), README. Tests: 308 passed. Outputs: `results/synthetic/` (tables and summaries; the 26 MB of per-trial JSONL is gitignored) and `results/calibration/`, `results/llm_r4/` (gitignored). Nothing is committed.
