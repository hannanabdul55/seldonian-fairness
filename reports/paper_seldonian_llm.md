# Seldonian post-training of language models: certified constraints on judge-measured behaviour under reinforcement learning

Draft v0.3, 2026-09-11. Branch `llm-seldonian-rl`. This is a working paper written
to make the whole setup legible end to end; every number is traceable to a
`result.json` under `results/` or to a table under `results/synthetic/` and
`results/calibration/`, and every design decision to a file in `seldonian/llm/`.

## Abstract

Reinforcement-learning post-training of language models optimises a learned reward
and, in doing so, routinely moves behaviour the reward does not measure: a safety
reward model drives over-refusal, a length-biased reward drives verbosity, a
compliance reward drives harm. We adapt the Seldonian framework (Thomas et al.,
2019) to this setting. Prompts are split into a candidate set and a sealed safety
set; the policy is trained on the candidate set by GRPO with a *predicted* safety
test every few steps and a Lagrangian penalty driven by that prediction; the
selected checkpoint is then measured exactly once on the safety set with a
one-sided confidence bound on each constrained quantity, and the algorithm
returns "no solution found" rather than a policy it cannot certify. Constraints
are stated on a frozen judge (a guard model, a verifiable check, a paired
counterfactual difference), so the guarantee is a statement about the judge, and
we say so throughout. Across five rounds on a 0.5B and a 1.5B instruct model we
find: (i) the Student-t bound used by the original framework fails 1.2-1.8 times
more often than its nominal delta at the 3% violation rates trained policies
reach, and Clopper-Pearson is the correct default for binary judges; (ii) in a
synthetic environment that runs the unchanged pipeline against an exactly
computable ground truth, the Lagrangian variant respects delta by a factor of
fifty and holds the true violation rate at the reference level across a full
pressure sweep, while filter-only candidate selection returns a solution in a
tenth of trials; (iii) on a verifiable brevity constraint under an injected reward pressure,
unconstrained GRPO breaches by 33-40 points in 5 of 5 runs and the Seldonian arm
returns a certified policy in 5 of 5, finding its own penalty and, at lower
pressure, discarding checkpoints that had crossed the threshold; with a floor on
the multiplier it returns a certified policy in 10 of 10 seeds with every predicted
checkpoint feasible (solution rate at least 0.74 at delta 0.05);
(iv) a fixed penalty chosen with knowledge of the pressure matches it on the
constraint and beats it on reward, so the layer's contribution is adaptivity and
the certificate, not reward; (v) on the natural over-refusal task, where the
penalty size is not known in advance, a too-small penalty breaches in 3 of 3
seeds, a penalty of the right average size breaches in 1 of 3 at the same reward
as the Seldonian arm, and the Seldonian arm breaches in 0 of 3 while returning a
certified policy in 2 of 3. We document the failure modes we hit on the way,
because they are the content: pressure through a proxy is absorbed, pooled
constraints are diluted, sampled decisions make paired constraints useless, and
a multiplier that decays to zero lets the policy drift back.

## 1. Introduction

A language model is post-trained by sampling responses to prompts, scoring them
with a reward, and updating the policy toward higher scores. The reward is
almost never the thing one actually cares about. It is a learned proxy (a
preference reward model), a verifiable check on part of the output (an exact
match on a final number), or a hand-built combination. Optimising the proxy
moves whatever correlates with it in the training distribution, whether or not
that was intended. The three cases that recur in this project are:

- a preference reward model trained on safety data prefers refusals and ethics
  preambles, so a policy trained on it doubles its refusal rate on benign
  prompts (our pilot, section 6.1);
- a compliance bonus meant to raise harm was instead satisfied by safe
  compliance, because the reward model's own preference against harmful text
  was larger than the bonus (section 6.4);
- a per-violation bonus above the reward model's within-group preference sends
  90% of answers over a length cap and, incidentally, into a different language
  (section 6.5).

The Seldonian framework offers a different contract. The user states a
constraint as a function `g` of the returned policy and a tolerance `delta`. The
algorithm may return a policy or the token "no solution found" (NSF), and it
guarantees that the probability of returning a policy with `g > 0` is at most
`delta`, where the probability is over the data the algorithm sees. The
guarantee is achieved by two mechanisms: the training data is used only to
*propose* a policy (candidate selection), and a held-out safety set is used
exactly once to *test* it with a high-confidence upper bound on `g`. What the
algorithm does during candidate selection cannot affect the validity of the
guarantee, only the probability that a solution is found.

This paper describes how we instantiated that contract for GRPO post-training
of small instruct models, what it took to test the contract itself rather than
merely apply it, and what we found. The framing question throughout is "does
the Seldonian layer change the outcome, and can we tell?" That question has
three parts, and each needed its own instrument:

1. **Is the safety test honest?** The bound has to hold at the rates and sample
   sizes that occur. We resampled real judge labels to measure the failure rate
   of eight bounds (section 6.2).
2. **Does the whole pipeline honour delta?** A guarantee over repeated runs
   cannot be checked with three seeds of a two-hour run. We built a synthetic
   environment that runs the unchanged pipeline thousands of times against an
   exactly computable ground truth (section 6.3).
3. **Does the constraint bind?** If the unconstrained baseline never breaches,
   the layer has nothing to do and the experiment says nothing. We built tasks
   where the baseline breaches by construction, and learned what that takes
   (sections 6.4, 6.5).

## 2. Background

### 2.1 The Seldonian framework

Let `D` be a dataset drawn from a distribution, `a` an algorithm that maps `D`
to a policy `theta` or to NSF, and `g_i(theta)` real-valued constraint
functions, with `g_i(theta) <= 0` meaning the constraint is satisfied. `a` is
Seldonian for `(g, delta)` if

    P( g_i(a(D)) > 0 for some i, and a(D) != NSF ) <= delta.

Note what is and is not promised. The event is joint: "a policy is returned and
it violates". Nothing is promised about how often NSF is returned (that is the
algorithm's quality, not its safety), and nothing is promised conditional on a
policy being returned. A method that always returns NSF is trivially Seldonian;
the design problem is to return a good policy as often as possible while keeping
the joint event rare.

The standard construction (Thomas et al., 2019) partitions `D` into a candidate
set `D_c` and a safety set `D_s`. Candidate selection uses `D_c` to pick a
`theta_c`, typically by optimising the objective subject to a *predicted*
version of the safety test. The safety test computes, for each `i`, a
`1 - delta_i` one-sided upper confidence bound `U_i` on `g_i(theta_c)` from
`D_s` and returns `theta_c` if every `U_i <= 0`, else NSF. If each bound has
its nominal coverage and the `delta_i` sum to `delta`, the union bound gives the
guarantee. Because `theta_c` is chosen without looking at `D_s`, the `D_s`
samples are independent of `theta_c` and the bound applies to it as to any
fixed policy; the multiple candidates examined during selection do not cost
anything, because only one of them is ever tested.

The predicted safety test is a heuristic: it applies the same bound to `D_c`
data (or a subset) with an inflated interval, so that candidates likely to fail
the real test are not selected. It has no bearing on validity; it only trades
solution rate against objective value.

### 2.2 RL post-training with GRPO

Group Relative Policy Optimisation samples `G` responses per prompt, scores
each with the reward, and forms advantages by normalising within the group:
`A_j = (r_j - mean_k r_k) / (std_k r_k + eps)`. The policy gradient is the
usual token-level clipped surrogate weighted by `A_j`, plus a KL penalty to the
reference policy with coefficient `beta`. Two properties matter for what
follows. First, the update depends on the *ranking* of responses within a
prompt, so a reward term only has an effect when it changes that ranking; a
bonus paid on an event that occurs in one of four samples changes the ranking
only in that group, and a bonus smaller than the within-group spread of the
base reward may change nothing. Second, a group in which all responses share
the same reward has zero advantage and contributes no gradient, so a penalty
cannot pull a prompt out of a behaviour that all its samples exhibit.

### 2.3 Why the two combine naturally

The policy is queryable. That removes the hardest part of Seldonian RL, which
is off-policy estimation of `g(theta)` from behaviour data with importance
weights. Here `g_i(theta)` is a mean over the prompt distribution of a bounded
judge applied to a fresh sample from `theta`, so a direct Monte Carlo estimate
on `D_s` is unbiased and its samples are i.i.d. The safety test is a
one-sample confidence bound on a mean of bounded variables, the simplest case
the bounds literature has.

## 3. Method

The implementation is `seldonian/llm/`: `policy.py` (the Seldonian logic,
model-free, unit-tested with a mock backend), `backend.py` (TRL GRPO with a
LoRA adapter), `judges.py` (constraint judges with a disk cache), `rewards.py`
(reward models and shaping), `constraints.py` (open-ended constraints),
`data.py` and `discrim.py` (task loaders), `synthetic.py` (the synthetic
environment). `scripts/run_llm_rl.py` is the driver; one invocation is one
(task, method, seed) run.

### 3.1 Data and the split

A *prompt record* is a prompt string with a group tag (`adversarial`,
`benign`, `math`, or a demographic value) and, for verifiable tasks, a
reference answer. Prompts are de-duplicated on a whitespace- and
case-normalised key, then split 60/40 into `D_c` and `D_s`, stratified by
group, with a fixed seed; the split is written next to the results so every
method in a comparison uses the same prompts on each side. Paired tasks split by
pair so both members land on the same side. `D_s` is sealed during training:
the policy object refuses to sample it while `fit` is running and counts safety
tests so a run cannot spend it twice.

### 3.2 Constraints and judges

A constraint is `g(theta) = E[ f(x, y) | x in group ] - tau`, where `f` is a
*judge*: a frozen function of (prompt, response) with values in a known bounded
range, `1` meaning the constrained event happened. Judges used:

| judge | type | used for |
|---|---|---|
| `Qwen/Qwen3Guard-Gen-0.6B`, Safety field = Unsafe | model | harm rate |
| `Qwen/Qwen3Guard-Gen-0.6B`, Refusal field = Yes | model | refusal rate |
| `protectai/distilroberta-base-rejection-v1` | model | refusal (pilot only; counts ethics preambles as refusals, replaced) |
| exact match of the final number | code | GSM8K error rate |
| `words > cap` | code | brevity over-cap rate |
| first yes/no in the response | code | DiscrimEval decision, abstention |

Every judge is cached on disk keyed by a hash of (judge name, prompt, response,
reference), so re-judging is free and a second method on the same samples never
re-runs a model. The guarantee is stated with respect to the judge. Where a
human label exists we report agreement (Qwen3Guard harm vs PKU `is_safe`: 90%
agreement, recall 0.98, precision 0.85 on 500 pairs) but we do not claim the
guarantee transfers; section 6.3(d) quantifies how much a judge's missed
violations cost.

Thresholds are *relative* by default: the reference (untrained) policy is
measured once per (task, seed) on 1,000 candidate prompts, the file is shared
by every method, and `tau = reference rate + margin`. A margin below the width
the predicted test will add cannot be satisfied by a policy at the reference
rate, so the driver refuses such a configuration unless overridden.

**Open-ended constraints** (`constraints.py`) generalise the single rate. Each
episode yields a vector of bounded features; a constraint is an expression over
group-conditional means of those features, and its bound is propagated through
the expression by interval arithmetic (`RandomVariable`), with the delta budget
split so all measures' two-sided intervals hold simultaneously. Three shapes
have dedicated, tighter forms: the paired counterfactual difference (a
one-sample bound on per-pair differences in `[-1, 1]`), the two-sample
difference (Bentkus or convex-order difference bounds), and the plain rate
(which reproduces the built-in constraint exactly, the regression test). A
constraint object exposes `measure(...)` and the policy calls it; the built-in
rate constraint and the expression constraints are interchangeable.

### 3.3 The safety test and its bounds

For constraint `i` with `n_i` safety prompts in its group, the test samples one
response per prompt at temperature 1, judges them, and computes a one-sided
upper bound `U_i` at level `delta_i = delta / k` for `k` constraints. The
policy is returned iff every `U_i <= tau_i`.

The bound is a named function from `seldonian.bounds`, selectable per
constraint. Those used or evaluated:

| bound | valid for | form |
|---|---|---|
| Student-t (`ttest`) | approximately, for near-normal sample means | `mean + s / sqrt(n) * t_{1-delta, n-1}` |
| Clopper-Pearson | exactly, for Bernoulli samples | `Beta^{-1}(1 - delta; k + 1, n - k)` |
| Bentkus | any distribution on `[0, 1]` | `e` times the binomial tail with the same mean |
| Hoeffding, Anderson, empirical Bernstein, Chernoff-KL | any distribution on `[0, 1]` | standard |
| betting mixture | any distribution on `[0, 1]` | fixed-`n` hedged capital, order-invariant |

Rounds 1-4 used the t bound with `delta = 0.1` split as 0.05 per constraint;
Round 5 used Clopper-Pearson at `delta = 0.1` for one constraint. Section 6.2
is the study that motivated the switch.

### 3.4 Candidate selection

Training runs GRPO on `D_c` for a fixed number of steps. Every `predict_every`
steps a *predicted safety test* is run: a group-stratified subset of `m`
candidate prompts is sampled, one response each, judged, and the bound
computed. The variance that matters is that of the difference between the
predicted rate (from `m` samples) and the eventual safety-set rate (from `n_s`
samples), so the bound is evaluated at the effective size
`n_eff = 1 / (1/m + 1/n_s)`, and an optional inflation factor multiplies the
deviation from the point estimate (Round 1 inherited a factor of 2 from the
classification code; Rounds 1c-5 used 1.0 at the effective size). A checkpoint
is *feasible* if every predicted `U_i <= tau_i`. Among feasible checkpoints
the one with the highest predicted mean base reward is kept; if none is
feasible the final checkpoint is tested anyway (and will usually fail).

Two variants of candidate selection were compared:

- **filter only** (`seldonian`): plain GRPO on the base reward; the predicted
  test only selects among checkpoints.
- **Lagrangian** (`seldonian_lag`): the reward fed to GRPO is
  `r - sum_i lambda_i * f_i` (each penalty restricted to its constraint's
  group), and after every predicted test each multiplier is updated by dual
  ascent on the predicted constraint value, `lambda_i <- clip(lambda_i + eta *
  (U_i - tau_i), 0, lambda_max)`. The predicted upper bound, not the point
  estimate, drives the multiplier, so the penalty rises before the point
  estimate crosses the threshold. The final safety test is unchanged, so this
  heuristic affects only which checkpoints exist to be selected.

Baselines: `reference` (no training), `grpo` (unconstrained), `composite`
(GRPO on `r - sum_i lambda_i f_i` with fixed `lambda_i`, no tests, no
certificate).

### 3.5 What is reported

For every run: the constraint rates on `D_s`, the upper bounds and thresholds,
whether a solution was returned, the selected step, the number of feasible
predicted tests, the *base* reward on `D_s` (shaped rewards unwrap to the
underlying reward model so methods are comparable; when a pressure bonus is
part of the training reward, the base is the reward model without the bonus),
mean response length, wall clock, and the full predicted-test history with
multipliers. For the pressure tasks we also report the objective GRPO actually
maximised, because the base reward alone can mislead (section 6.5).

### 3.6 The synthetic environment

`synthetic.py` implements the `PolicyBackend` interface with a contextual
bandit so that `SeldonianLLMPolicy` runs unchanged: a population of 20,000
contexts `x ~ N(0, I_8)` with a group tag; four actions; ground-truth
violation probability `p_v(x, a) = sigmoid(u_a . x + b_a)` with biases chosen so
the uniform reference policy violates 13% of the time and two actions are
"unsafe"; reward `w_a . x + rho * p_v(x, a) + N(0, 0.5^2)`. `rho` is the
pressure: at 0 reward and violation are unrelated, at `rho > 0` the better
actions are the more violating ones. The policy is softmax-linear, trained by
the same group-normalised advantage as GRPO with Adam and a KL penalty to the
uniform reference. A "response" is the string `a=<action>;v=<label>`, the judge
reads `v` (optionally flipping it with a given sensitivity and specificity),
and the reward object parses `a`. The true violation rate of any policy is the
exact expectation over the population, so the delta-calibration question has a
ground-truth answer. A full trial (200 steps, group 8, predicted tests every 25
steps on 512 prompts, safety test) takes 0.06 s.

## 4. Tasks

| task | prompts | reward | constraints | pressure knob |
|---|---|---|---|---|
| A+B, harm and over-refusal | 3,000 PKU-SafeRLHF (adversarial) + 3,000 XSTest-safe / OR-Bench (benign); pilot used 1,000 benign | Skywork-Reward-V2-Qwen3-0.6B | harm <= ref + 0.03 (all prompts); refusal on benign <= ref + 0.05 | none (natural), or `--compliance-bonus alpha`: `+ alpha * (1 - refusal)` on adversarial |
| D, GSM8K control | 3,000 GSM8K train | exact match | error <= ref + 0.05 | none |
| brevity | 3,000 benign prompts + "Answer in at most 80 words." | Skywork | `P(words > 120) <= ref + 0.05` | `--long-bonus b`: `+ b * 1[words > 120]`; `--length-bonus b`: `+ b * min(words, 300) / 300` |
| DiscrimEval parity | 1,890 white / Black pairs (70 scenarios x 9 ages x 3 genders), question + "Please answer yes or no, followed by a one-sentence justification." | Skywork | `|mean over pairs (yes_a - yes_b)| <= 0.05`; abstain <= ref + 0.05 | `--bias-bonus b`: `+ b * 1[yes]` on group a |

The pressure knobs are the point. A task tests the framework only if the
unconstrained baseline breaches, and section 6.4 shows that whether it does is
not something one can assume. A knob that scales the conflict between the
objective and the constraint lets one trace the whole frontier from
non-binding to always-breaching, and lets the real-LLM results be read against
the synthetic phase diagram of section 6.3(c).

## 5. Experimental setup

| item | value |
|---|---|
| policy | `Qwen/Qwen2.5-0.5B-Instruct` (Rounds 1, 1b, 1c, 4, 5), `Qwen/Qwen2.5-1.5B-Instruct` (Round 2) |
| adapter | LoRA r 16, alpha 32, dropout 0.05 on q, k, v, o, gate, up, down; bf16; gradient checkpointing |
| GRPO (TRL 1.12) | group 4 (pilot, Rounds 4-5) or 8 (Rounds 1b-2); 8 prompts per step; 150 (Rounds 4-5) or 200 steps; lr 1e-5; beta 0.04 (0 in one pair); temperature 1; 256 new tokens; `steps_per_generation` 4 from Round 1b |
| predicted test | every 25 (Rounds 1-2) or 30 (Rounds 4-5) steps on 256 (pilot), 512, 1,024 (1c) or 768 (4-5) candidate prompts |
| Lagrangian | lambda0 2 (pilot) then 5; eta 20 (pilot, 1b) then 100; cap 20 then 50 |
| delta | 0.1 |
| hardware | one RTX 4070 Super, 12 GB, shared under a file lock; HF `generate` for sampling (vLLM not installed) |
| wall clock | reference 4-11 min; trained arm 80-150 min at group 4 / 150 steps, 230 min at group 8 / 200 steps, 4.5 h at 1.5B |

About 75 GPU-hours went into Rounds 1-3 (33 runs), 12 into Round 4, and 30 into
Round 5 (18 runs).

## 6. Results

### 6.1 Rounds 1-3: the natural over-refusal task

The pilot (0.5B, 3 seeds, DistilRoBERTa refusal judge) found that every trained
method cut the harm rate from 14% to 3-4% and *doubled* refusals on benign
prompts (15% to 34-37% by the Qwen3Guard flag): the reward model prefers
refusals and preambles on this prompt mix. Unconstrained GRPO breached the
refusal constraint 3 of 3. Filter-only selection never saw a feasible
checkpoint and returned NSF 3 of 3. The first Lagrangian configuration also
returned NSF 3 of 3: its multiplier saturated at the cap of 20 and slowed but
did not reverse the drift, for three reasons diagnosed in the pilot report: the
predicted interval on 400 benign safety prompts was wider than the margin, so a
checkpoint had to be *below* the reference to pass; the refusal classifier
rewarded removing the preamble rather than answering; and all-refuse groups
give zero gradient.

Round 1b fixed the judge (Qwen3Guard refusal flag), tripled the benign set,
raised the margin to at least the predicted width, and found 2 of 3 solutions at
low reward; the third missed by 0.002 because the predicted bound had ignored
the prediction sample's own variance, the origin of the effective-`n` rule.
Round 1c (eta 100, effective-`n` bound, 1,024 prediction prompts) returned a
solution 3 of 3 with 0 breaches, held refusals at the reference level (14% vs
15%), halved harm, and captured on average 65% of the reward gain GRPO obtained
by refusing 37% of benign prompts.

At 1.5B and on GSM8K the constraints did not bind: GRPO stayed inside both
thresholds even with the KL penalty removed, and the Seldonian arm returned a
solution at a cost of 0-5% reward. Those runs bound the "cost of safety" on a
benign reward but say nothing about the framework, which is what motivated
sections 6.3-6.5.

| setting | unconstrained GRPO | Seldonian Lagrangian (corrected) |
|---|---|---|
| 0.5B, A+B | breaches refusal 3/3; refusals 37%; reward 2.07 | solution 3/3, breach 0/3; refusals 14%; reward 1.16 |
| 1.5B, A+B, beta 0.04 / 0 | within bounds; reward 3.80 / 4.85 | solution; reward 3.59 / 4.59 |
| 0.5B, GSM8K | within bounds; accuracy 0.347 | solution, 8/8 feasible; accuracy 0.367 |

### 6.2 Is the bound honest? Resampling real labels

`scripts/resample_calibration.py`. The Qwen3Guard harm labels of the pilot's
three reference safety sets are pooled (3,600 adversarial episodes, rate 0.161)
and likewise for the pilot's three GRPO policies (3,600 episodes, rate 0.034,
the low rate a trained policy has); the pooled mean is taken as truth, safety
sets of size `n` are resampled 5,000 times, and each bound's failure rate
`P(U < truth)` at nominal `delta` is recorded with its mean width.

GRPO pool, rate 0.034, `delta = 0.1`:

| bound | n=200 | n=400 | n=800 | n=1200 | n=2400 |
|---|---|---|---|---|---|
| Student-t | **0.184** / 0.016 | **0.115** / 0.012 | **0.121** / 0.008 | 0.101 / 0.007 | 0.101 / 0.005 |
| Clopper-Pearson | 0.084 / 0.023 | 0.067 / 0.015 | 0.083 / 0.010 | 0.074 / 0.008 | 0.084 / 0.005 |
| Bentkus | 0.029 / 0.032 | 0.035 / 0.021 | 0.034 / 0.014 | 0.023 / 0.011 | 0.032 / 0.007 |
| betting mixture | 0.007 / 0.038 | 0.016 / 0.025 | 0.011 / 0.017 | 0.009 / 0.014 | 0.010 / 0.009 |
| Hoeffding, Anderson | 0.000 / 0.076 | 0.000 / 0.054 | 0.000 / 0.038 | 0.000 / 0.031 | 0.000 / 0.022 |

At `delta = 0.05` the t bound fails at 0.084, 0.067, 0.083, 0.074, 0.067: above
nominal at every `n`. On the reference pool (rate 0.16) it is close to nominal.
The t interval is a large-sample approximation whose sample standard deviation
is itself noisy at low rates and small `n`; Clopper-Pearson is exact for
Bernoulli labels and within 5-15% of the t width for `n >= 800`; Bentkus is the
tightest distribution-free bound (1.4x the t width, failing at 0.3 delta) and
the right one for non-binary features. Every "solution" in Rounds 1-4 rests on
the t bound, and the trained policies' harm rates (3-6%) are exactly the regime
where it is anti-conservative. Nothing in those rounds was close enough to a
threshold for this to have flipped an outcome, but it is why Round 5 and
everything after use Clopper-Pearson.

### 6.3 Does the pipeline honour delta? The synthetic environment

`scripts/synthetic_calibration.py`; every row is 500-1,000 independent trials
with a fresh population and split; `n` is the total number of prompts (40% to
the safety set); threshold is the true reference rate + 0.03 unless stated;
`unsafe` is `P(solution returned and true rate > tau)`, the quantity delta
bounds.

**(a) Delta calibration across bounds, pressure 1, n = 1,000.**

| method | bound | solution | unsafe | true rate given solution | reward given solution |
|---|---|---|---|---|---|
| grpo | | 1.00 | 0.830 | 0.208 | 0.85 |
| seldonian_lag | t | 0.81 | 0.002 | 0.128 | 0.77 |
| seldonian_lag | Clopper-Pearson | 0.82 | 0.000 | 0.124 | 0.76 |
| seldonian_lag | Bentkus | 0.84 | 0.000 | 0.114 | 0.75 |
| seldonian_lag | Hoeffding | 0.91 | 0.000 | 0.094 | 0.71 |

The Lagrangian arm is fifty times inside delta with every bound, because dual
ascent parks the policy at the reference rate and the safety test is rarely the
binding element. A wider bound *raises* the solution rate: the predicted `g` is
larger, the multiplier grows faster, the policy ends more conservative, at a
reward cost.

**(b) Sample size, pressure 1, t bound.**

| method | n | solution | unsafe | violation given solution | true rate given solution | reward |
|---|---|---|---|---|---|---|
| seldonian_lag | 200 | 0.72 | 0.000 | 0.000 | 0.100 | 0.70 |
| seldonian_lag | 1000 | 0.81 | 0.002 | 0.002 | 0.128 | 0.77 |
| seldonian_lag | 5000 | 0.89 | 0.000 | 0.000 | 0.134 | 0.78 |
| filter only | 200 | 0.08 | 0.028 | 0.333 | 0.165 | 0.85 |
| filter only | 1000 | 0.09 | 0.004 | 0.043 | 0.146 | 0.83 |
| filter only | 5000 | 0.13 | 0.004 | 0.030 | 0.146 | 0.84 |

Filter-only reproduces the pilot: 0.3-0.7 feasible checkpoints of 8, a solution
in a tenth of trials, and at small `n` a third of those solutions truly
violate; the unconditional unsafe rate still respects delta, which is the
guarantee and also its limitation.

**(c) Pressure sweep, n = 1,000: the phase diagram.**

| pressure | grpo unsafe | grpo true rate | lag solution | lag unsafe | lag true rate | lag reward / grpo reward |
|---|---|---|---|---|---|---|
| 0 | 0.184 | 0.144 | 0.88 | 0.000 | 0.119 | 0.65 / 0.67 |
| 0.5 | 0.524 | 0.175 | 0.83 | 0.000 | 0.126 | 0.71 / 0.75 |
| 1 | 0.830 | 0.208 | 0.81 | 0.002 | 0.128 | 0.77 / 0.85 |
| 2 | 0.996 | 0.263 | 0.79 | 0.000 | 0.128 | 0.89 / 1.08 |
| 4 | 1.000 | 0.325 | 0.76 | 0.000 | 0.130 | 1.13 / 1.68 |

The layer holds the true rate at the reference level at every pressure; the
solution rate falls slowly and the reward gap grows, which is the price of not
taking the violating actions. The 0.5B over-refusal task sits near pressure
1-2 on this diagram, the 1.5B and GSM8K runs near 0, the brevity task at bonus
16 beyond 4.

**(d) Judge noise, pressure 1, n = 1,000, thresholds measured with the noisy
judge.**

| judge (sensitivity, specificity) | solution | judge-level violation given solution | true violation given solution |
|---|---|---|---|
| (1.0, 1.0) | 0.81 | 0.002 | 0.002 |
| (0.8, 0.95) | 0.67 | 0.003 | 0.036 |
| (0.6, 1.0) | 0.65 | 0.000 | 0.059 |
| (1.0, 0.9) | 0.70 | 0.000 | 0.011 |

The guarantee holds exactly at the judge level in every row. A judge that
misses 40% of violations lets thirty times more true violations through at an
identical judge-level failure rate. False positives are harmless (the policy
becomes more conservative; NSF rises). This is the case for measuring each
judge's sensitivity on hand labels and correcting the threshold for it.

**(e) Predicted-test inflation and dual step, pressure 1.** Doubling the
predicted interval raises the solution rate from 0.81 to 0.97 for a 4% reward
loss; eta 20 vs 100 changes the solution rate by 4 points. **(f) Stress at
margin 0** (threshold equal to the true reference rate, so the bound is the
only safeguard): filter-only at n = 500 returns a solution in 3% of trials and
half of those violate, but the joint unsafe rate is 1.2% (95% upper 1.9%)
against delta 0.1; the Lagrangian arm stays at 0.0-0.2% with a 72-80% solution
rate.

### 6.4 Making the constraint bind: the compliance-pressure task

Round 4 tried to make the harm constraint bind by paying a bonus `alpha * (1 -
refusal)` on adversarial prompts (0.5B, group 4, 150 steps, t bound, Round 1c
seed-0 thresholds: harm 0.108 + 0.03, refusal 0.128 + 0.05).

| alpha | method | harm (ub) / tau | benign refusal (ub) / tau | base reward | outcome |
|---|---|---|---|---|---|
| | reference | 0.101 / 0.138 | 0.112 / 0.178 | -0.83 | |
| 5 | grpo | 0.079 / 0.138 | 0.117 / 0.178 | 0.44 | within bounds |
| 5 | seldonian_lag | 0.100 (0.111) / 0.138 | 0.094 (0.108) / 0.178 | 0.07 | solution, step 120, 5/5 feasible |
| 10 | grpo | 0.115 / 0.138 | 0.065 / 0.178 | 0.09 | within bounds |
| 10 | seldonian_lag | 0.117 (0.127) / 0.138 | 0.076 (0.088) / 0.178 | -0.11 | solution, step 120, 5/5 feasible |

Neither pressure bound. The diagnostics on the adversarial half of `D_s`
explain why: the reward model pays 3.8 points more for a refusal than for a
compliant answer and about 6 more than for a *harmful* one, so a bonus of 5
makes safe compliance the best option and leaves harmful compliance behind
refusal. GRPO found exactly that: an unchanged refusal rate, better-scoring
refusals, compliant answers less often harmful (harm given compliance 0.19
against 0.26). At 10 adversarial refusals fell from 45% to 38% and harm on
adversarial prompts rose from 14.7% to 17.0%, but the constraint was written on
all prompts and the benign half (6% harm) diluted it to 0.115 against 0.138.
Two design rules came out of this: pay the pressure on the constraint's own
event, not a proxy that can be satisfied another way; and state the constraint
on the prompts the pressure acts on.

### 6.5 A task that breaches on demand: brevity

Benign prompts with an explicit 80-word instruction, the verifiable constraint
`P(words > 120) <= reference + 0.05` (Clopper-Pearson, delta 0.1, 1,200 safety
prompts), and an injected pressure. Skywork has no length bias on plain benign
prompts, but *with* the instruction it rewards obeying it: on reference samples
an answer under the cap scores 2.0 points above one over it, and a 200-word
answer 3.9 below an 80-word one.

**Finding the pressure.** A linear length bonus of 3 made GRPO *shorter* (over-cap
rate 0.50 to 0.06); a step bonus of 4 per violation also made it shorter (to
0.37), with rollout lengths flat for all 150 steps and the reward rising through
the reward model alone. Within a GRPO group the relevant quantity is the spread
of rewards among four answers to the same prompt (standard deviation 2.6-3.2 in
the training log); a bonus that lands on the quarter of rollouts over the cap did
not dominate it. At 8 and 16 it does, and the transition is sharp, as
group-normalised advantages predict: once the bonus exceeds the within-group
spread, the sign of the gradient flips for every group at once.

**Bonus 16, three seeds.** Thresholds are per-seed reference + 0.05.

| seed | reference / tau | grpo rate, base reward | seldonian_lag rate (ub), base reward, selected step | multiplier peak |
|---|---|---|---|---|
| 0 | 0.496 / 0.561 | 0.904 breach, -0.84 | 0.063 (0.073), 2.05, step 90 | 42 |
| 1 | 0.457 / 0.517 | 0.917 breach, -1.05 | 0.111 (0.123), 2.04, step 120 | 49 |
| 2 | 0.507 / 0.531 | 0.916 breach, -0.80 | 0.051 (0.060), 2.04, step 90 | 44 |

The trajectory is the same in every seed (seed 0 shown):

| step | predicted over-cap rate (ub) | feasible | multiplier after | base reward |
|---|---|---|---|---|
| 30 | 0.914 (0.930) | no | 41.9 | -1.46 |
| 60 | 0.314 (0.343) | yes | 20.1 | 1.23 |
| 90 | 0.060 (0.076) | yes | 0.0 | 2.19 |
| 120 | 0.181 (0.206) | yes | 0.0 | 1.86 |
| 150 | 0.266 (0.294) | yes | 0.0 | 1.65 |

By step 30 the Seldonian arm had breached exactly as GRPO did; the dual step
raised the multiplier past 40, the drift reversed within 60 steps, the
multiplier decayed to zero once the constraint had slack, and the pressure began
pulling the policy back toward the cap. Selection took the step-90 checkpoint.
Under length pressure the policy also padded answers by switching into Chinese
mid-sentence, a second drift a language-match constraint would catch.

**Bonus 8, seed 0**, meant as the marginal regime; GRPO still breached (0.888).
The Seldonian trajectory differed:

| step | predicted rate (ub) | feasible | multiplier after | base reward |
|---|---|---|---|---|
| 30 | 0.135 (0.158) | yes | 0.0 | 1.80 |
| 60 | 0.260 (0.288) | yes | 0.0 | 1.59 |
| 90 | 0.561 (0.591) | **no** | 3.0 | 0.73 |
| 120 | 0.624 (0.653) | **no** | 12.2 | 0.72 |
| 150 | 0.487 (0.518) | yes | 7.9 | 1.19 |

The initial multiplier of 5 held the policy short through step 60 and decayed to
zero; the pressure then carried it across the threshold; two checkpoints were
predicted infeasible and rejected; the dual step pulled it back by step 150;
selection returned step 30 (safety-set rate 0.135, upper 0.149, base reward
1.82). Here the predicted test did real work: without it the final checkpoint
would have been returned, feasible by 4 points.

**Attribution: fixed-penalty composite arms, seed 0.**

| pressure | arm | over-cap rate / tau | base reward | min |
|---|---|---|---|---|
| 16 | composite, penalty 16 (cancels the bonus exactly) | 0.057 / 0.561 | 2.45 | 79 |
| 16 | composite, penalty 32 | 0.028 / 0.561 | 2.81 | 72 |
| 16 | seldonian_lag | 0.063 (0.073) / 0.561 | 2.05 | 107 |
| 8 | composite, penalty 16 (twice the pressure) | 0.055 / 0.561 | 2.45 | 75 |
| 8 | seldonian_lag | 0.135 (0.149) / 0.561 | 1.82 | 116 |

A fixed penalty at or above the pressure does as well on the constraint and
better on base reward at two thirds of the wall clock. On this task
over-penalising is free because the constraint points the same way as the
reward model's own preference for short answers, so the brevity task cannot
show the downside of a too-large fixed penalty; that requires a task where the
constraint opposes the reward model (the over-refusal task), which is Stage C of
the Round 6 plan.

**The fixed-penalty frontier** (Round 6 B2, seed 0; `results/llm_r6/b2_*`):

| bonus | penalty 2 | 4 | 8 | 16 | 32 | Seldonian |
|---|---|---|---|---|---|---|
| 8, over-cap rate | 0.815 | 0.313 | 0.066 | 0.055 | 0.061 | 0.146 (floor 5 always-on) |
| 8, base reward | -0.17 | 1.72 | 2.42 | 2.45 | 2.41 | 2.37 |
| 16, over-cap rate | | 0.927 | 0.860 | 0.057 | 0.028 | 0.063 (peak multiplier 42) |
| 16, base reward | | -0.98 | -0.51 | 2.45 | 2.81 | 2.05 |

The transition from breach to compliance sits at a quarter to a half of the bonus
(between 2 and 4 at bonus 8, between 8 and 16 at bonus 16), because the reward
model's own preference for short answers carries the rest; above it the frontier
is flat, since over-penalising costs nothing on this task. The penalty that holds
at bonus 8 is useless at bonus 16. The Seldonian arm found a multiplier of 5 at
bonus 8 and 42 at bonus 16 from one configuration.

**Objective actually optimised.** On the training reward (Skywork plus 16 per
violation) GRPO scores about 13.6 and the Seldonian policy about 3.1. The
higher *base* reward of the constrained policy is a property of an adversarial
bonus, not a free lunch.

### 6.6 DiscrimEval: the reference measurement

At 0.5B the model answers in 99.8% of cases, says yes to 71.0% of white fills
and 74.0% of Black fills, and the paired mean difference is -0.030 (Black
favoured), the direction Anthropic reported for larger models. But the
per-pair differences have standard deviation 0.60 against 0.635 for two
independent Bernoulli(0.72) draws: at temperature 1 the model's yes/no on a
scenario is close to a coin flip across samples, the pair members are nearly
uncorrelated, and the reference's own upper bound on the parity gap (0.030 +
0.036 = 0.066) already exceeds a 0.05 threshold. The fix is a noise-free feature,
the policy's probability of "yes" from one forward pass, which Stage 0 of the
Round 6 plan adds. No trained arm was run on this task.

### 6.7 Fixing the dual dynamics in the synthetic environment

The drift-back seen in every Lagrangian trajectory (section 6.5) has a simple
cause: the multiplier decays to zero once the constraint has slack, and the
pressure is still there. Two knobs were added to `LagrangianReward`: a floor the
multiplier cannot fall below once its constraint has been predicted infeasible at
least once, and a separate step size for the decay direction. A sweep of 500
trials per setting at pressures 1 and 4 (`results/synthetic/g_dynamics_*.md`)
measured the *drift*, the true violation rate at the last checkpoint minus the
minimum over checkpoints, alongside the usual quantities.

| pressure | starting multiplier, floor | solution | unsafe | true rate given solution | reward | drift |
|---|---|---|---|---|---|---|
| 1 | 5, 0 (Rounds 1-5) | 0.81 | 0.002 | 0.128 | 0.77 | 0.065 |
| 1 | 5, 5 | 0.88 | 0.000 | 0.120 | 0.75 | 0.012 |
| 1 | frozen after first bind | 1.00 | 0.000 | 0.067 | 0.65 | 0.006 |
| 4 | 5, 0 | 0.76 | 0.000 | 0.130 | 1.13 | 0.031 |
| 4 | 5, 5 | 0.81 | 0.000 | 0.128 | 1.12 | 0.008 |
| 4 | frozen after first bind | 0.94 | 0.000 | 0.114 | 1.07 | 0.011 |

A slower decay alone does not remove the drift, freezing the multiplier removes
it at a 16% reward cost, and a floor equal to the starting value removes most of
it (4-5x less drift) while *raising* the solution rate by 5-7 points at a 1-3%
reward cost, because one more checkpoint per run stays feasible. That setting
is the default for the next round; its confirmation on the brevity task, where
the floor of 5 is below the pressure of 8-16, is the first GPU experiment of the
Round 6 plan.

### 6.8 The floor on the brevity task

The Round 6 gate ran the floored multiplier (start 5, floor 5) at bonus 8 and 16,
seed 0, against the Round 5 trajectories (`results/llm_r6/b1_v8`, `b1_v16`).

| bonus | setting | predicted rate at steps 30 / 60 / 90 / 120 / 150 | multiplier | selected | test rate (ub) | base reward | drift |
|---|---|---|---|---|---|---|---|
| 8 | Round 5, no floor | 0.135 / 0.260 / 0.561 / 0.624 / 0.487 | 0 / 0 / 3.0 / 12.2 / 7.9 | 30 | 0.135 (0.149) | 1.82 | +0.352 |
| 8 | floor 5 | 0.130 / 0.383 / 0.699 / 0.387 / 0.337 | 0 / 0 / 16.6 / 5.0 / 5.0 | 30 | 0.140 (0.154) | 1.74 | +0.207 |
| 16 | Round 5, no floor | 0.914 / 0.314 / 0.060 / 0.181 / 0.266 | 41.9 / 20.1 / 0 / 0 / 0 | 90 | 0.063 (0.073) | 2.05 | +0.206 |
| 16 | floor 5 | 0.901 / 0.311 / 0.073 / 0.211 / 0.279 | 40.7 / 18.7 / 5.0 / 5.0 / 5.0 | 90 | 0.095 (0.107) | 2.05 | +0.206 |

Two things the synthetic environment did not show. First, the floor as
implemented arms only after a constraint has been predicted infeasible, and at
bonus 8 that is too late: the starting multiplier of 5 decays to zero at the first
prediction (a rate of 0.13 against a threshold of 0.56 gives `g = -0.43`, and a
step of 100 takes the multiplier to zero), so the policy climbs unpenalised from
0.13 to 0.70 between steps 30 and 90. The floor then holds the last two
checkpoints under the threshold (0.39, 0.34 against 0.56) instead of on it, which
halves the drift. Second, at bonus 16 a floor of 5 against a pressure of 16 is no
floor at all: the trajectory is unchanged to the second decimal. What the floor
needs to be is a fraction of the pressure, which is unknown in advance but is what
the multiplier itself measures at its peak (41 at bonus 16, 17 at bonus 8); a
ratchet floor set from the peak is the untested next version. Both runs return the
same checkpoint at the same reward as without the floor, so the drift-back is dual
ascent behaving as designed (the constrained optimum sits at the threshold) and
candidate selection makes it harmless for the guarantee; its cost is feasible
checkpoints, which the ten-seed solution rate measures.

A floor applied from the first update (`--lam-floor-always`) was swept in the
synthetic environment (`results/synthetic/g_dynamics_floor_always.md`, 500 trials
per row): at pressure 4 it raises the solution rate from 0.82 to 0.91 at a 3%
reward cost (floor 5) and from 0.87 to 0.98 at 10% (floor 10); at pressure 1 it
over-corrects (true rate 0.067 against a threshold of 0.16, reward down 12%),
because a multiplier of 5 outweighs a pressure of 1. The brevity stages, whose
pressure is well above 5, use it; the over-refusal stage, near pressure 1-2,
keeps the armed floor.

On the real task the always-on floor is decisive at bonus 8 (`results/llm_r6/b1a_v8`):
predicted rates 0.105 / 0.122 / 0.122 / 0.174 / 0.154 against a threshold of 0.561,
the multiplier at 5 at every update, 5 of 5 checkpoints feasible, drift +0.048,
and the final checkpoint returned at a base reward of 2.37, against 1.82 (Round 5)
and 1.74 (armed floor), both of which had to fall back to the step-30 checkpoint.
The multiplier never rose: at this pressure a constant penalty of 5 is enough, and
the dual step is insurance against a pressure it does not know.

**Ten seeds** (Round 6 B4; `results/llm_r6/b4_v8`, `b1a_v8`): at bonus 8 with the
always-on floor of 5, seeds 0-9 return a certified policy 10 times in 10, breach
0 times, and have every one of their 50 predicted checkpoints feasible; over-cap
rate 0.156 (sd 0.016) against thresholds of 0.50-0.56, base reward 2.17 (sd 0.13),
selected step 90-150. The one-sided Clopper-Pearson lower limit on the solution
rate is 0.74 at delta 0.05, the first solution-rate number in this work with an
interval. The gap between the selected checkpoint's predicted rate and its
safety-set rate is +0.003 on average (sd 0.018, largest 0.027) against margins of
0.35 or more, so the winner's curse had no room to act. The multiplier never left
the floor, so these ten runs are a fixed penalty of 5 plus a certificate; the dual
step's value is on the tasks where 5 is the wrong number (bonus 16, over-refusal).

### 6.9 Judge calibration (provisional)

The guarantee is stated in the judge's labels. With sensitivity `s` and
specificity `p`, the judge-level rate `q` and the true rate `r` are related by
`q = (s + p - 1) r + (1 - p)`. Because the same judge measures the reference
policy, a relative constraint `r <= r_ref + m` is exactly `q <= q_ref + J m` with
`J = s + p - 1`: the offset cancels and the margin scales by Youden's index
(`seldonian/llm/calibration.py`; `run_llm_rl.py --judge-calibration`). Lower
confidence limits on `s` and `p` make the correction conservative.

Violations are rare, so the labelling sample is stratified by the judge's own
flag: 100 flagged and 100 cleared responses per judge, drawn from the safety-set
responses of every 0.5B over-refusal run (34,687 adversarial and 22,757 benign;
flag prevalence 0.088 and 0.172). That estimates the predictive values directly;
sensitivity and specificity follow from them and the prevalence, and their lower
limits from Clopper-Pearson limits on the predictive values. A language model
produced a first pass of labels; the human labels are pending, and six harm rows
with weapons content were left for the human.

| judge | PPV | NPV | sensitivity | specificity | J | margin, judge-level |
|---|---|---|---|---|---|---|
| harm (Qwen3Guard unsafe) | 0.52 (>= 0.43) | 0.99 (>= 0.95) | 0.83 (>= 0.47) | 0.96 (>= 0.95) | 0.79 (>= 0.41) | 0.030 -> 0.024 (point) |
| refusal (Qwen3Guard refusal) | 0.69 (>= 0.61) | 1.00 (>= 0.97) | 1.00 (>= 0.81) | 0.94 (>= 0.92) | 0.94 (>= 0.73) | 0.050 -> 0.047 (point) |

Provisional as they are, the labels say three things. Both judges miss almost
nothing the labeller counts: the failure direction of the guarantee, a true
violation the judge clears, is rare. Both over-flag by the labeller's stricter
definition (half of the harm flags are dark fiction, generic caution, or
rambling without actionable help; a third of the refusal flags are a disclaimer
followed by a real answer), which is a definition gap the human labels have to
settle, and one that makes the judge conservative rather than unsafe. And the
sensitivity lower limit is set by the cleared stratum: one miss in 98 gives an
NPV of at least 0.95, which at an 8.8% flag prevalence allows a sensitivity as low
as 0.47. Pinning it above 0.8 needs about 500 cleared-stratum labels, not 100.
At the current sample the lower-limit correction (margin 0.030 to 0.012) is below
the predicted-test width, so only the point-estimate correction is usable.

### 6.10 The over-refusal task with fixed-penalty and Seldonian arms

Stage C of Round 6 ran the natural over-refusal task at 0.5B with the Round 5
machinery (group 4, 150 steps, Clopper-Pearson, delta 0.1, the harm constraint on
the adversarial prompts only, margins harm 0.045 and refusal 0.05, the armed floor
of 5) over three seeds and four trained arms (`results/llm_r6/c`, `c_l4`). The
constraint here opposes the reward model, which prefers refusals, so a penalty
that is too large should cost reward, which brevity could not show.

| arm | breaches | mean harm | mean benign refusal | mean base reward | per-seed reward |
|---|---|---|---|---|---|
| reference | | 0.158 | 0.110 | -0.80 | |
| grpo | 3 of 3 | 0.069 | 0.260 | 0.63 | 0.47 / 0.77 / 0.65 |
| composite, penalty 1 | 3 of 3 | 0.072 | 0.219 | 0.72 | 0.60 / 0.75 / 0.82 |
| composite, penalty 4 | 1 of 3 | 0.089 | 0.172 | 0.43 | 0.46 / 0.45 / 0.37 |
| seldonian_lag | 0 of 3 (solution 2 of 3) | 0.099 | 0.153 | 0.37 | 0.32 / 0.36 (NSF) / 0.43 |

Thresholds on benign refusal were 0.178 / 0.192 / 0.172 by seed. Unconstrained
GRPO more than doubles refusals in every seed. A penalty of 1 is no penalty: it
breaches every seed and scores above GRPO, a mild regulariser. A penalty of 4 is
the right size on average and lands within one or two points of the threshold in
every seed (0.151, 0.182, 0.183): inside twice, a breach once, at a reward within
0.06 of the Seldonian arm's. The Seldonian arm never breaches and certifies the
two policies it returns.

The three Seldonian trajectories are three different stories. At seed 0 the
refusal multiplier bound at step 60, the floor held it at 5, and the step-120
checkpoint passed (0.133, upper 0.151). At seed 1 the selected step-90 checkpoint
predicted 0.133 and measured 0.184 (upper 0.204 against 0.192): NSF. That gap is
the winner's curse on a prediction sample of 384 benign prompts (standard deviation
0.018) chosen as the best of the feasible ones, and the multiplier, climbing 1-2
per update from the floor, never drove the policy under the threshold in the
second half of the run. At seed 2 no checkpoint was predicted feasible (upper
bounds 0.176-0.207 against 0.172), the multiplier climbed to 14.6, the final
checkpoint was tested and passed (0.142, upper 0.160): the predicted test was
pessimistic by the same 0.01-0.02 it had been optimistic at seed 1. The harm
multiplier decayed to zero at the first or second prediction in every seed, since
training lowers harm.

The attribution this supports: on a task where the penalty size is not known in
advance, a fixed penalty at the right average size buys the same reward as the
Seldonian layer and a one-in-three chance of a silent breach; the layer's addition
is that it never returns the breaching policy, and it says so. Its cost is the
NSF, which is a candidate-selection weakness (the prediction sample) rather than
a property of the bound, and which a larger prediction sample or a faster dual
step on this task would reduce.

## 7. Analysis

**What the guarantee is about.** It is the joint event "a policy is returned
and it violates", against a frozen judge, on the safety-set prompt distribution.
It is not a conditional statement, not a statement about true labels, and not
a statement about other prompt distributions. The synthetic stress test shows
the difference concretely: filter-only selection at margin 0 returned a
violating policy in half of its rare solutions and still respected delta
because it rarely returned anything. A practitioner sees the conditional rate,
and it is the Lagrangian variant, not the bound, that makes it small.

**Where the safety test does work.** When the dual step drives the policy far
inside the constraint (bonus 16), the safety test is a formality: predicted
and actual agree and nothing is close. The test earns its keep in the marginal
regime (bonus 8), where checkpoints straddle the threshold and the predicted
test discards the ones over it. Experiments aiming to show the value of the
test, as opposed to the penalty, should be run there.

**The winner's curse.** Selecting the best-looking checkpoint biases its
predicted quantities. On the synthetic environment the selected checkpoint's
predicted rate was 0.7 points below its safety-set rate on average; on the
compliance task the predicted reward of the selected checkpoint was 0.29 against
0.07 measured. The effective-`n` bound covers the rate; nothing covers the
reward, which should be reported from the safety set only.

**Dual dynamics.** Three things recur in every Lagrangian trajectory: the
initial multiplier over-corrects (rates of 5-13% against thresholds near 55%);
the multiplier decays to zero once the constraint has slack and the policy
drifts back (0.18-0.26 over the cap by step 150 at bonus 16, 0.49 at bonus 8);
and the peak multiplier scales with the pressure (42-49 at 16, 12 at 8). A
floor on the multiplier after a constraint has bound is the remedy section 6.7
selects in the synthetic environment; on the real task (6.8) it halves the drift
at bonus 8 and does nothing at bonus 16, because a fixed floor is only a floor
when it is a fraction of the pressure. The drift-back itself is not a failure of
the guarantee, which candidate selection protects; it is a cost in feasible
checkpoints.

**A moving landscape, and why that is allowed.** While the multiplier moves
there is no fixed objective for the policy, so "the optimum of the training
reward" is not defined at any moment. Primal-dual methods do not look for a
minimum of a stationary landscape; they look for a saddle point of the
Lagrangian `L(theta, lambda) = reward(theta) - sum_i lambda_i (rate_i(theta) -
tau_i)`, a maximum in `theta` and a minimum in `lambda >= 0`. For a convex
problem that saddle exists and ascent-descent converges to it, and at the saddle
the landscape stops moving: `lambda*` is the price of each binding constraint
and `theta*` is the optimum of that fixed penalty, so the moving landscape is a
route to a fixed one whose penalty was not known in advance. Nothing here is
convex. The primal step is a few GRPO updates on a nonconvex LoRA policy, the
dual step is discrete, large (eta 100 on a confidence bound) and thirty steps
apart, and the trajectories show what that produces: overshoot (rate 0.91 at
step 30, multiplier to 42, rate 0.06 by step 90), decay, drift-back. Those are
the oscillations of a two-timescale game that is not converging, the standard
failure of a fast dual against a nonconvex primal. The design does not need it
to converge. Every checkpoint at every predicted test is a fixed policy that is
evaluated as such, the best feasible one is kept, and the safety test checks it
once; a wandering optimiser costs solution rate and reward, never safety. The
Round 6 runs read cleanly in this light. The always-on floor at bonus 8 (6.8)
worked because it froze the landscape: the multiplier sat at 5 for the whole
run, the policy optimised a fixed penalty, and the final checkpoint was the
best one (5 of 5 feasible, reward 2.37), where the moving penalty of the earlier
runs (0 to 17 to 5) left the step-30 checkpoint to be rescued by selection.
Over-refusal seed 1 (6.10) is the other side: a dual step too slow for the
pressure (5 to 7.4), the policy hovering at the threshold, 2 of 5 checkpoints
feasible, NSF. The useful schedule is therefore not adaptive against fixed but
two-phase: let the dual step find the price, then hold it and let the primal
settle, which is what averaging the dual iterates, a decaying dual step, or a
ratchet floor set from the peak multiplier all do; the synthetic "frozen after
first bind" row (6.7) is the crude version and removes the drift at a reward
cost because it freezes too high. Even with a fixed multiplier there is no
global optimum to certify, since GRPO on a 0.5B adapter finds a local one; the
Seldonian contract was built for that situation, asking nothing of the
optimiser and only honesty of the safety test.

**Hardening the judge.** The guarantee is a statement about a frozen judge,
and that is the paper's weakest point: the judge is a 0.6B guard model whose
agreement with human judgement is, so far, measured by another language model.
Three things harden it, in order of how much they change the claim. First, the
guarantee transfers to human labels once the judge's sensitivity `s` and
specificity `p` against them are known (6.9): a constraint on the true rate is
exactly a constraint on the judge-level rate with the margin scaled by
`J = s + p - 1`, conservative when lower confidence limits are used. The
labelling is a one-off cost, not a per-run one, and the arithmetic says what it
costs: because the failure direction is a true violation the judge clears, the
sensitivity lower limit is set by the cleared stratum, and pinning it above 0.8
at an 8-9% flag prevalence needs about 500 cleared responses per judge with a
handful of misses, plus 100-200 flagged ones for the predictive value; roughly
1,400 labels for both judges, double-annotated, on the order of 70
annotator-hours. Second, the Seldonian split lets humans replace the judge where
it matters: the judge is called thousands of times during candidate selection,
where it only affects which checkpoint is proposed, but the safety test runs once
per returned policy on a fixed set of 1,200 responses per constraint. Human
labels on that one set put the certificate in human terms and take the judge out
of the guarantee entirely, at about 2,400 labels per certified policy, a cost for
a deployment candidate rather than for every seed; a stratified human subsample
with the correction above on the rest cuts it by 5-10x. Third, the definition
has to be fixed before labels are bought: the provisional pass says both judges
over-flag by a stricter definition (dark fiction and generic caution without
actionable help; a disclaimer followed by a real answer), which is harmless for
safety but means the labelling guideline decides what the guarantee is about. A
second, independently trained judge with humans routed only to disagreements is
the human-in-the-loop form of the same step.

**The cost of safety** depends entirely on the pressure. At pressure 0 (1.5B,
GSM8K) it is 0-5% reward and one safety-set evaluation. At pressure 1-2 (0.5B
over-refusal) it is 35% of the reward gain the baseline obtained by violating.
At pressure beyond 4 (brevity) the constrained policy scores lower on the
optimised objective by a factor of four and higher on the base reward, because
the objective was adversarial to it. There is no single number.

**Attribution.** Given the pressure, a fixed penalty set at or above it does as
well or better on brevity (6.5). Where the penalty size is not known in advance
(6.10), a penalty of 1 breaches every seed and a penalty of 4, the right size on
average, breaches one seed in three at the Seldonian arm's reward; the layer's
contributions are that it finds the penalty without being told the pressure
(peaks of 42-49 at bonus 16, 12-17 at 8, 5-15 on over-refusal, one
configuration), never returns the breaching policy, and returns a certificate.
Its cost is the NSF rate, one in three on over-refusal, which traces to the
prediction sample rather than the bound.

## 8. Limitations

- Real-LLM comparisons have one to three seeds except the ten-seed brevity block
  (6.8); the delta claim rests on the synthetic environment, which has one
  constraint and a 36-parameter linear policy.
- The guarantee is with respect to judges. Their calibration (6.9) rests on
  machine labels until the hand labels are in, and its sensitivity lower limits
  are loose at 100 cleared labels per judge.
- Rounds 1-4 used the t bound, shown in 6.2 to be anti-conservative at the
  rates trained policies reach.
- Reward pressure is injected. The natural case (a helpfulness-only reward
  model) was not run because the canonical one is too large for the card.
- Generation is the bottleneck (HF `generate`, 80-150 minutes per arm), which
  is what limits seeds; vLLM would roughly halve it.
- Response length is capped at 256 new tokens, which truncates the brevity
  task's upper tail and made a pure length-ceiling task impossible.
- The DiscrimEval task has a reference measurement only.
- The over-refusal prediction sample (384 benign prompts) is the weak link: it
  produced one optimistic and one pessimistic miss of 0.01-0.05 in three seeds.

## 9. Next steps

`reports/llm_round6_plan.md` lays out six stages. Done: the code changes, the
synthetic dual-dynamics sweeps (6.7, 6.8), the floor's confirmation on brevity
(6.8) and the calibration tooling with provisional labels (6.9). Done as well: the
over-refusal task with fixed-penalty and Seldonian arms over three seeds (6.10).
Also done: the always-on floor on brevity and the ten-seed solution rate (6.8),
the fixed-penalty frontier and the marginal regime (6.5), and DiscrimEval with the
probability feature (6.6: the feature collapses the point estimate, the pressure
cannot open a gap at 0.5B, and the parity bound has to be variance-adaptive). Off the GPU: the human labels (with about 500
cleared-stratum responses per judge, after a written labelling guideline; see
"Hardening the judge" in section 7), a human-labelled safety set for one returned
policy, and a ratchet floor set from the peak multiplier.

## Appendix A. Glossary

| term | meaning here |
|---|---|
| `D_c`, `D_s` | candidate (training + predicted tests) and safety (one final test) prompt sets, 60/40 |
| judge | frozen function of (prompt, response) in a bounded range; 1 = the constrained event |
| `g_i`, `tau_i` | constraint value `E[judge_i | group] - tau_i`; positive = violated |
| reference rate | the untrained policy's rate on 1,000 candidate prompts, measured once per (task, seed) |
| margin | `tau - reference rate` for relative constraints |
| predicted test | the safety-test bound applied to `m` candidate prompts at `n_eff = 1/(1/m + 1/n_s)` |
| feasible checkpoint | every predicted upper bound at or under its threshold |
| solution / NSF | safety test passed and the policy returned / "no solution found" |
| breach (baseline) | a baseline's safety-set rate above the threshold; carries no guarantee either way |
| base reward | the underlying reward model's mean score on `D_s`, without any bonus or penalty |
| pressure | a term added to the reward that pays for the constrained event (`alpha`, `b`, `rho`) |
| multiplier | the Lagrangian penalty weight, updated by dual ascent on the predicted `g` |
| composite | fixed-penalty baseline, no tests, no certificate |
| unsafe (synthetic) | `P(solution and true g > 0)`, the quantity delta bounds |

## Appendix B. Reproduction

```
uv sync --extra rl
# bound calibration on cached labels (no GPU)
uv run scripts/resample_calibration.py
# synthetic sweeps (CPU)
uv run scripts/synthetic_calibration.py --tag c_pressure --pressure 0 0.5 1 2 4 --n 1000 --method grpo seldonian_lag --trials 500
# brevity task, one seed, three arms (GPU, ~4 h)
uv run scripts/run_llm_rl.py --task brevity --method reference     --seed 0 --bound clopper_pearson --out results/llm_r5/v16
uv run scripts/run_llm_rl.py --task brevity --method grpo          --seed 0 --bound clopper_pearson --out results/llm_r5/v16 --long-bonus 16 --steps 150 --group-size 4
uv run scripts/run_llm_rl.py --task brevity --method seldonian_lag --seed 0 --bound clopper_pearson --out results/llm_r5/v16 --long-bonus 16 --steps 150 --group-size 4 --lam-max 50
uv run scripts/summarize_llm.py --out results/llm_r5/v16 --task brevity
# judge calibration sheets from cached labels (no GPU), then analysis once "label" is filled
uv run scripts/judge_calibration.py sample --per-stratum 100
uv run scripts/judge_calibration.py analyze
```

Queue scripts with the exact settings of every round: `scripts/run_round1.sh`
through `scripts/run_round6b.sh`. Reports per round: `reports/llm_round1_pilot.md`,
`reports/llm_round2.md`, `reports/llm_round4_evaluation_design.md`.
