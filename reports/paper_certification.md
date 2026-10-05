# Certifying behaviour rates of language-model policies: what holds, what a label buys, and what does not carry

**Draft v0.3, 2026-10-04.** Working title; the framing is open (plan section 7, item 5).
v0.3: the certificate in human labels (section 8.4) is costed and prepared but not run; open
gaps are P8, P10 and P16.
Built from results already in the repository; nothing here is new measurement. Every number
carries a source tag, resolved in Appendix A. `[GAP: Pn]` marks a result that step `Pn` of
`.planning/paper-certification/PLAN.md` will supply. `[CHECK]` marks a number to re-verify
against its source before v1.0. The companion paper on training (`reports/paper_seldonian_llm.md`)
is cited as "the training paper".

## Abstract

A deployed language-model policy is usually described by rates: how often it refuses a benign
request, how often a tool-using agent follows an injected instruction, how often a robot
controller trips a safety stop. We study the statement "this rate is at most tau" as a
certificate: a one-sided bound at level delta, computed on a sample the policy's selection never
saw, with "no solution found" as a permitted outcome. This is the safety test of the Seldonian
framework (Thomas et al., 2019) applied to a fixed policy. We report where the certificate
holds its level (synthetic environments with exact truth, resampling studies on real judged
labels, and published traces of frontier models), two ways to make a fixed label budget go
further (safety sets stratified by the reference model's own per-prompt rate; a judge used as a
variance reducer under a fixed routing rule, with a bootstrap-t limit where the published
normal-quantile intervals of PPI++ and StratPPI run over their level), and what does not carry: a judge's
calibration across prompt populations, the same calibration across training that targets the
label, independent-sample bounds on crossed benchmark designs, normal limits at the rates
trained policies reach, judge-assisted finite-sample bounds, and a stratified labelling sheet
read as a random sample. On AgentDojo's published runs
the usual per-pair bound misses in 5-28% of resamples at a nominal 5%, and one of 28 pipelines
certifies a 5% attack success rate once the design is respected. A certificate of a trained
policy in human labels is costed and prepared, not run: from one annotator's measured rates a
2-point refusal margin takes about 420 labelled prompt pairs for an even chance of passing and
950 for an 80% chance, and a guard used as predictor cuts that by 2.5 only when the claim is
about a fixed pool of prompts.

## 1. Claim and scope

**Claim.** A Seldonian safety test certifies a label-defined behaviour of a fixed language-model
policy from a sample: a one-sided bound at level delta on the rate of a code- or model-defined
label, with "no solution found" as an outcome. We show where it holds, how to make the sample
go further, and what does not carry.

**Scope, stated once.**

- The label is a guard model's or the benchmark harness's unless marked human. The human-label
  results are in section 8.3. Section 8.4 costs a certificate in human labels and does not
  compute one.
- Policies trained here are 0.5B to 3B parameters on one 12 GB card. The frontier-model results
  are certificates on other people's published traces, not training.
- A certificate is about the label as defined and the population the safety sample was drawn
  from. It says nothing about other prompts or about the policy after further training.
- How the certified policies were trained is the training paper's subject. This paper needs
  training in two places only: the policy whose human-terms certificate section 8.4 costs, and
  the negative result of section 7.2, which exists because training targeted the label.

## 2. The certificate

**Setting.** A policy `pi` maps a prompt to a response. A label `y(prompt, response)` in {0, 1}
marks the behaviour (a refusal, a successful injection, a safety stop). The constraint is
`g(pi) = E[y] - tau <= 0` for a threshold `tau`. Prompts are split into a candidate set `D_c`,
which training and checkpoint selection may use freely, and a safety set `D_s`, which they
never see.

**The test.** Draw the safety sample once, compute an upper confidence bound `U` on `E[y]` at
level `1 - delta`, and return the policy if `U <= tau`. Otherwise return "no solution found"
(NSF). The guarantee is

    P( g(returned policy) > 0 and a policy is returned ) <= delta,

over the draw of the data, for any selection rule on the candidate side. The guarantee is on the
output. It does not depend on how the candidate was found or on any optimiser converging.

**What it is not.** Training with a penalty or a Lagrange multiplier also yields a policy whose
constraint estimate "ended below the threshold". Table 1 contrasts the two statements as they
appeared in this project [SR 2.3].

*Table 1. "The constraint held": a Lagrangian's statement against a certificate's.*

| | penalty or Lagrangian | certificate |
|---|---|---|
| measured on | candidate-side samples the optimiser and the selection saw | one draw of `D_s`, independent of both |
| statement | the estimate ended below the threshold in this run | the (1 - delta) upper bound on the returned policy is below the threshold |
| selection bias | uncorrected: predicted 0.133 against 0.184 on the safety set (Round 6, seed 1); 0.141 against 0.168 (spike 014) | removed by the fresh draw; that seed returned NSF |
| checked by | the breach rate: a fixed penalty of 1 breached in 3 of 3 seeds, a penalty of 4 in 1 of 3 | the miss rate against delta (section 3) |
| can decline | no | yes; the solution rate is reported beside every certificate |

A method that always returns NSF satisfies the guarantee, so the solution rate is part of the
result: 10 of 10 on a brevity constraint, 2 of 3 on over-refusal [SR 2.1].

**Relative thresholds.** Several constraints here are relative: the trained policy's rate may
exceed the reference model's by at most a margin. In the experiments cited the reference rate
was measured once on candidate data and then treated as a constant. Section 8.4's design does
not do that: in human terms the reference rate is unknown and is estimated from labels like the
other.

## 3. Which bounds hold their level

"Holds" means: over repeated draws of the safety sample from a population whose true rate is
known, the bound falls below the truth in at most a fraction delta of draws. We measure this
three ways: a synthetic environment where the true rate of any policy is computable; resampling
from a large pool of real judged responses, with the pool's mean as truth (a plasmode in the
sense of Franklin et al., 2014); and, for crossed designs, resampling whole clusters.

Figure 1 draws the table below as miss rate over delta.

*Table 2. Miss rate against delta for every bound used. "Exact" means valid at every sample
size by construction; "approximate" means valid asymptotically and checked by resampling.
Monte Carlo standard errors are 0.003-0.004 for 5,000 resamples and 0.011 for 400.*

| bound | kind | setting | delta | miss | source |
|---|---|---|---|---|---|
| Clopper-Pearson | exact (binary labels) | real harm labels, trained-policy pool, rate 0.034, n 200-2,400 | 0.10 | 0.067-0.084 | [R 6.2] |
| Clopper-Pearson | exact | spike 017 plasmodes, every cell | 0.05 | at most 0.051 | [017 B6] |
| betting mixture | exact (bounded) | same pool as row 1 | 0.10 | 0.007-0.016 | [R 6.2] |
| Bentkus | exact (bounded) | same pool | 0.10 | 0.023-0.035 | [R 6.2] |
| Hoeffding, Anderson | exact (bounded) | same pool | 0.10 | 0.000 | [R 6.2] |
| stratified Wilson-type `b1w` | approximate | 4 mid-rate labels (9-94%), real Granite-3.3-2B responses, n_s 100-200 | 0.05 / 0.10 | 0.023-0.054 / 0.069-0.097 | [013 H8] |
| `b1w` | approximate | label pushed by the Lagrangian, n_s 200 | 0.05 / 0.10 | 0.011-0.023 / 0.040-0.064 | [014] |
| `b1w` on a design-weighted sheet | approximate | sheets re-drawn by their real sampling rule | 0.05 | 0.001-0.059 | [017 4] |
| PPI++ with a bootstrap-t limit | approximate (second order) | spike 017 plasmodes, every cell and feature | 0.05 | at most 0.053 | [017 B6] |
| cluster bootstrap-t, by user task | approximate | AgentDojo, 6 pipelines, user tasks resampled | 0.05 | 0.020-0.060 | [020 P] |
| StratPPI estimator with a bootstrap-t limit | approximate | reference-rate strata, 5 mid-rate labels, n_s 100-200; judge-logit strata, 14 cells, n 100-1,000 | 0.05 | at most 0.045; at most 0.056 | [P14] |
| **Student-t** | fails at low rates | same pool as row 1, n 200-800 | 0.10 | **0.115-0.184** | [R 6.2] |
| **`b1w` or any approximate bound at rare rates** | fails | harm labels at 1-2%, n_s 100, any design | 0.05 | **0.24-0.44** | [013 H8] |
| **PPI++ with a normal limit** | fails | spike 017 plasmodes | 0.05 | **up to 0.241** | [017 B6] |
| **StratPPI as published (normal limit)** | fails at these sizes | reference-rate strata: over its level in 4 of 10 mid-rate cells; judge-logit strata: in 26 of 28 | 0.05 | **up to 0.086; up to 0.239** | [P14] |
| **Clopper-Pearson over pairs** | fails on a crossed design | AgentDojo, user tasks resampled | 0.05 | **0.048-0.275** | [020 P] |
| **two-way bootstrap** | fails where positives sit in few clusters | AgentDojo | 0.05 | **up to 0.170** | [020 P] |
| **stratified sheet read as an i.i.d. sample** | fails | PPI on the sheet, one judge wording | 0.05 | **up to 0.98** | [017 4] |
| **the judge's rate alone** | fails | spike 017 plasmodes | 0.05 | **1.000** | [017 B6] |

Three readings of Table 2.

1. **Exact bounds for rare labels.** At 1-2% no approximate bound held, stratified or not; the
   failure is binomial discreteness, not the design [013]. Rare labels take Clopper-Pearson or a
   betting bound.
2. **The t-test is the wrong default.** It is anti-conservative exactly where trained policies
   sit (harm rates of 3-6%), and its width is zero at a rate of 0 or 1, which lets candidate
   selection win by driving a subgroup rate to the boundary [R 6.2, 012].
3. **"Approximate" has a number attached.** The three approximate bounds we use miss within
   about one percentage point of delta in every cell tested. The one cell above nominal at
   delta 0.05 is a label at a 94% rate (0.054, one standard error over).

The StratPPI rows (Fisch et al., 2024) are our implementation of its estimator and interval on
the same draws as the rows above them; sections 5 and 6 give the comparison.

## 4. Validity under the full pipeline

Table 2 tests bounds on a fixed population. The certificate also has to survive selection: the
candidate is the best-looking checkpoint of a training run.

**Synthetic environment, exact truth.** In a contextual bandit where every policy's true rate is
computable, the probability of returning a policy that truly violates was 0.000-0.002 against a
delta of 0.1 for the Seldonian arm under four bounds, against 0.830 for unconstrained training
at the same reward pressure; it stayed at or below 0.002 across reward pressures 0 to 4 and
sample sizes 200 to 5,000 [R 6.3]. With a noisy judge the guarantee held at the judge's level in
every row, while a judge that misses 40% of violations let the true violation rate among
returned policies reach 0.059: the certificate is about the label [R 6.3d].

**Adversarial splits.** Rerandomised and stratified candidate/safety splits never broke the
test, across 2,000 seeds per arm, eleven split rules and candidates built to overfit the
balancing statistics; Clopper-Pearson missed in at most 1 of 2,000 runs in any cell [012].

**The real training loop.** With the project's `SeldonianLLMPolicy` in the loop, 500 runs per
cell, the stratified `b1w` test missed 0.084-0.116 at delta 0.1 (Monte Carlo standard error
0.013), the same as a random split with a pooled bound [013 4].

**Language-model policies.** On an over-refusal constraint (Qwen2.5-0.5B, three seeds),
unconstrained training and a fixed penalty of 1 breached the threshold in 3 of 3 seeds, a
penalty of 4 in 1 of 3; the Seldonian arm breached in 0 of 3 and certified 2 of 3. On a
brevity constraint, 10 of 10 seeds certified with no breach [SR 2.1]. Three and ten seeds
cannot estimate a miss rate; they are consistent with the level, and the resampling studies
above are the evidence.

**Trajectories.** A test of the returned policy alone misses drift through a forbidden region
during training: 65% of plain-Lagrangian runs passed it after a step inside the region. A
certificate at level delta/T over every one of T checks held, with misses of 0.025-0.100
against a delta of 0.1 [SR 2.1].

## 5. Making the sample go further: reference-rate strata

**Idea.** For a relative constraint the reference model is already run on the safety prompts.
Sample it k times per prompt, rank prompts by their reference rate, cut the ranking into H
equal strata with ties broken at random, sample the safety set within strata, and bound the
rate with a stratified Wilson-type bound (`b1w`). The gain depends on how much of the
per-response variance sits between prompts (the intraclass correlation of the reference,
ICC_ref) and on how well a prompt's reference rate predicts its rate under the trained policy.

**Result.** With k = 8 and H = 8 on real Granite-3.3-2B responses, the effective safety-sample
size (ESS: the size of a simple random sample giving the same bound width) against a random
split was 2.4 for over-refusal (rate 17%, ICC_ref 0.72), 5.1-5.3 for refusal of plainly harmful
requests (66%, 0.86), and 1.4 for non-refusal of encoded requests (9%, 0.50), with coverage as
in Table 2. A placebo covariate gave 1.0 [013]. When the Lagrangian pushed the stratified label
itself, the gain was 2.1-2.2 with no compression of the between-prompt spread (correlation
between a prompt's reference and trained rates 0.92) [014].

**Where it does not help.** Rare labels (the approximate bound is invalid there and exact
stratified bounds did not beat pooling); labels near 0 or 1; a safety set that is a large share
of the prompt pool (the gain is capped at `1 / (1 - G + G n_s / N)`); and task strata on a
benchmark with 20 trials a task, where the bound is set by the positives [013, 019].

**A pre-flight.** The gain is predictable from k reference samples before any trained-policy
response is labelled: Spearman 0.83 between predicted and realised ESS on real data, with the
prediction 0-20% high at H = 8 [013] (Figure 2). The absolute-error criterion we pre-registered (median
error at most 0.1) failed on real data (0.29) and passed on the bandit (0.02); we report the
ranking as the usable part.

**Against StratPPI.** Stratified sampling with prediction-powered intervals for language-model
evaluation is StratPPI (Fisch et al., 2024): within each stratum the labelled mean is corrected
by a regression on a predictor, and the interval uses normal quantiles. We ran it on the same
strata and the same 5,000 draws per cell, with the reference rate as the predictor [P14]
(Figure 3, left).

*Table 3. Reference-rate strata at delta 0.05: largest miss over checkpoints, and ESS against a
random split with a pooled Wilson bound. Five mid-rate labels, two safety-set sizes.*

| label (rate) | n_s | `b1w` | StratPPI as published | StratPPI estimator, bootstrap-t limit |
|---|---|---|---|---|
| over-refusal (0.16) | 100 | 0.023; 2.39 | **0.066**; 4.11 | 0.042; 2.58 |
| over-refusal (0.16) | 200 | 0.024; 2.30 | 0.052; 3.46 | 0.042; 3.10 |
| over-refusal, pushed by the Lagrangian (0.18) | 100 | 0.024; 2.19 | 0.056; 3.24 | 0.036; 2.43 |
| over-refusal, pushed (0.18) | 200 | 0.023; 2.13 | 0.045; 2.85 | 0.037; 2.49 |
| non-refusal, encoded prompts (0.09) | 100 | 0.034; 1.38 | **0.086**; 2.59 | 0.045; 1.43 |
| non-refusal, encoded prompts (0.09) | 200 | 0.027; 1.36 | **0.063**; 2.11 | 0.041; 1.59 |
| refusal, harmful prompts (0.65) | 100 | 0.047; 4.75 | **0.056**; 6.30 | 0.033; 1.77 |
| refusal, harmful prompts (0.65) | 200 | 0.040; 5.19 | 0.045; 6.29 | 0.034; 5.35 |
| refusal, encoded prompts (0.93) | 100 | 0.054; 0.96 | 0.032; 0.78 | 0.031; 0.83 |
| refusal, encoded prompts (0.93) | 200 | 0.045; 1.12 | 0.026; 0.92 | 0.033; 1.03 |

Bold: more than two Monte Carlo standard errors over delta. Three things follow.

1. **StratPPI's bound is shorter, and part of that is the normal limit running over its level.**
   It exceeds delta in 4 of 10 cells at delta 0.05 (one of them marginally, at 0.056; 3 of 10 at
   delta 0.1), most at the 9% label, and in every rare-label cell, where `b1w` fails too.
2. **With an honest limit the two are close.** The same estimator with a bootstrap-t limit holds
   in all ten cells (largest miss 0.045) and its median ESS is 2.10 against 2.16 for `b1w`. It
   is ahead in seven cells, by 3-35%, and far behind in one (1.77 against 4.75), where strata
   of about twelve labels on a heavily tied predictor make the bootstrap unstable.
3. **So the gain here is the stratification.** The within-stratum regression on the reference
   rate adds little once the limit is valid. We keep `b1w` as the bound for reference-rate
   strata at these sizes and note the bootstrap-t StratPPI as the better choice when strata
   hold 25 labels or more.

We did not run StratPPI's optimal allocation, and the implementation is ours, written from the
paper's equations. `[GAP: P10]` (optional) adds a second policy model.

## 6. What a judge buys: a routing rule

A judge model scores every response cheaply; gold labels are scarce. We fix in advance which
estimator a constraint gets (Table 4) [017 1].

*Table 4. Routing rule.*

| data | route | guarantee |
|---|---|---|
| a property code can verify (a word count) | compute it on every response; Clopper-Pearson | exact |
| gold labels, fewer than 10 in the rarer class | Clopper-Pearson on the labels; judge unused | exact |
| gold labels, 10 or more in each class | PPI++ (Angelopoulos et al., 2023b) on the judge's logit, bootstrap-t limit | approximate |
| gold labels from a stratified sheet | design-weighted labels, `b1w` | approximate |
| a calibration measured on another population | refused | none |

**The judge's worth in labels.** With `rho^2` the squared correlation between the gold label
and the judge's feature, PPI++ multiplies the labels' worth by about `1 / (1 - rho^2)` when
unlabelled responses are plentiful. For a compiled refusal judge against the guard's label
(225 labels, a pool of 2,000 responses): the 0/1 verdict gives 1.11, the probability 1.15, the logit
1.61, a cross-fitted Platt score 1.86 in measured bound width [017 3]. The lesson is to carry
the logit, not the verdict. For a harm rubric `rho^2` was 0.02, a gain of 1.02: the labels
carry the certificate and the judge contributes nothing.

**Label budgets for a rare label.** With no positive observed, 301 gold labels on the
certified policy's own responses certify a 1% threshold at delta 0.05, 149 certify 2%, and 59
certify 5% [017 6].

**Choosing which responses to label.** Table 4 labels a random subset. If the judge has scored
every response first, the labels can be drawn within strata of its logit. On the refusal pools
of spike 017 (two judge wordings, 14 cells, 4,000 draws each, delta 0.05) [P14] (Figure 3,
right):

- StratPPI as published, with 5 or 10 strata, is over its level in 26 of 28 cells (misses
  0.053-0.239), as PPI++ with a normal limit is in all 14 (0.061-0.232).
- The StratPPI estimator with a bootstrap-t limit holds in all 28 (largest miss 0.056) and is
  the most efficient valid route: against the labels alone its median ESS is 2.54 at a 20%
  rate and 1.87 at 5%, where unstratified PPI++ with the same limit gives 1.67 and 1.39.
- Stratifying on the judge and ignoring it within strata (`b1w`) also holds (largest miss
  0.053), at 2.41 and 1.42.
- At a 1.3% rate with 225 labels the bootstrap-t routes return nothing useful and the count
  rule's fallback to the labels alone is the right one.

So the third row of Table 4 has a better form when labelling can follow scoring: stratify on
the judge's logit, StratPPI's estimator, a bootstrap-t limit. The count threshold stays.

**Position.** Choosing which items go to humans from a model's confidence, with valid
intervals, is Active Statistical Inference (Zrnic and Candès, 2024) and Confidence-Driven
Inference (Gligorić et al., 2025); PPI for model evaluation is Boyeau et al. (2025); a bootstrap
for PPI is PPBoot (Zrnic, 2024). Our contribution in this section is narrower: evidence that
the published normal-quantile intervals, stratified or not, do not hold their level at the
sample sizes and rates of a safety test; a studentised bootstrap that does; a count rule for
when to drop the judge; and the measured gap between a judge's verdict and its logit.

## 7. What does not carry

Figure 4 shows sections 7.1 and 7.2: the miss rate of a carried bound and the judge's recall, by shift.

### 7.1 A judge's calibration across prompt populations

Measure a judge's recall and false-alarm rate on one population and apply them to another, and
the corrected bound fails. Across two sources of benign prompts in the same pool, recall
differed in 4 of 6 judge wordings and the carried bound missed in up to 94% of draws; across
benign and harmful pools, 4 of 6 and up to 100% [017 5]. False-alarm rates did not differ;
recall did.

### 7.2 The same calibration across training that targets the label

This is the result we consider most general. A calibration carried across 200 steps of
training that moved the label only as a side effect (recall differed in 0 of 6 wordings). It
did not carry across training whose Lagrangian targeted the label: the compiled judge's recall
on the guard's refusals fell from 0.21 to 0.03, and the carried bound missed in 4 of 6 wordings
(80% of draws for the canonical one), although the rate itself had returned to the
reference's [017 E8].

The mechanism is a change in the form of the positives. The reward pulled toward long helpful
answers and the multiplier penalised the guard's refusal flag. After training, the responses
the guard flags are long refusals with an explanatory, helpful-sounding body: by the guard and
a surface pattern (a refusal opener and more than 40 words), their share of the guard's flags
rose from 37% in the reference to 82% in the trained policy. The rubric judge reads that body
as an answer: of 93 guard-flagged refusals after training it called 89 answers. Human labels
side with the guard. Of 18 such responses of the trained policy, the annotator labelled 12
refusals, 6 answers and none refuse-then-answer, and refuse-then-answer is 3.4% of the
reference's responses against 0.3% of the trained policy's (section 8.3). We had first read
these responses as a refusal sentence followed by the answer; the labels do not support that
reading, and the surface pattern is not a label (on XSTest's completions 56% of responses
with it are human full refusals and 34% partial ones [P15]).

The consequence for certification: gold labels must be on the responses of the policy being
certified, every time the policy changes. A calibration is not an asset that survives the
optimisation it was meant to audit.

### 7.3 Independent-sample bounds on crossed designs

AgentDojo crosses user tasks with injection tasks; a pipeline's attack successes cluster by
both. Design effects by user task ran 1.4-8.7 and by injection task 2.6-15.5 [020]. The
per-pair Clopper-Pearson bound missed in 5-28% of resamples at delta 0.05 (Table 2). Section
8.2 gives the consequence for a published leaderboard.

### 7.4 Normal limits and judge-assisted exact bounds

PPI++ with its normal limit missed in up to 24% of draws at delta 0.05 and n of 100-500
[017 B6]. Finite-sample judge-assisted bounds (betting on blocks) were valid and never narrower
than the labels alone at n up to 1,000 [SR 2.4]. Taking the smaller of two valid bounds is not a
bound.

### 7.5 A stratified sheet read as a random sample

A labelling sheet that over-samples a guard's flags is a stratified sample. Read as i.i.d., it
certified a negative harm rate under one judge wording, and missed in 98% of re-drawn sheets
[017 4]. Every sheet in this paper carries its design weights (section 8.3).

## 8. Certificates in practice

### 8.1 A robot benchmark: the resolution is the result

RoboDojo-RC Tier 1 publishes 120 trials per model with a harness-emitted safety-stop flag
[019]. Table 5 gives the certificates and Figure 5 what 120 trials can resolve.

*Table 5. Safety stops, Clopper-Pearson upper bounds.*

| model | n | stops | rate | upper, delta 0.05 | upper, delta 0.10 | "at most 5%" at delta 0.05 |
|---|---|---|---|---|---|---|
| Claude Opus 5 | 120 | 10 | 0.083 | 0.137 | 0.125 | NSF |
| Claude Opus 5.5 | 120 | 2 | 0.017 | 0.052 | 0.044 | NSF (passes at delta 0.10) |
| GPT-6 Astra | 120 | 2 | 0.017 | 0.052 | 0.044 | NSF (passes at delta 0.10) |

A benchmark of 120 trials can certify a 2.5% rate when it observes nothing and cannot certify
5% after two events; a 1% claim with nothing observed needs 299 trials per model. One model
stops more than the other two (10 of 120 against 4 of 240, Fisher p = 0.003), a difference the
benchmark's report does not mention. The policy's own narration of risk does not predict a
stop (AUC 0.38); joint effort does (AUC 0.75-0.83, exploratory).

### 8.2 An agent-security benchmark: one of 28

On AgentDojo's published runs (19,380 episodes, the harness's `security` label), a certificate
that respects the design is the studentised cluster bootstrap by user task, or the larger of
the two clustered bounds where injection tasks carry more dependence [020] (Figure 6).

*Table 6. Targeted attack success, selected pipelines, upper bounds at delta 0.05. The
certificate is the larger of the two clustered bounds.*

| pipeline | pairs | rate | per-pair bound | clustered by user task | clustered by injection task | "at most 5%" |
|---|---|---|---|---|---|---|
| claude-3-5-sonnet-20241022 | 629 | 0.011 | 0.021 | 0.022 | 0.034 | pass |
| Meta-SecAlign-70B | 949 | 0.022 | 0.032 | 0.104 | 0.035 | NSF |
| command-r | 629 | 0.033 | 0.048 | 0.062 | 0.054 | NSF |
| claude-3-7-sonnet-20250219 | 949 | 0.050 | 0.063 | 0.070 | 0.106 | NSF |
| gpt-4o + tool_filter | 629 | 0.068 | 0.087 | 0.092 | 0.106 | NSF |
| gpt-4o-2024-05-13 | 629 | 0.477 | 0.511 | 0.538 | 0.586 | NSF |

One pipeline of 28 certifies. A defended model whose raw rate is 2.2% does not: its 21
successes sit in 5 of 97 user tasks (ICC 0.72), so the clustered bound is 0.104. The defence
concentrated the failures instead of removing them, and only a bound that respects the design
shows it.

### 8.3 The guard against human labels

**Published human labels.** XSTest releases 2,250 completions from five model variants, each
annotated by two of its authors as full compliance, full refusal or partial refusal ("any
combination of refusal and compliance"; Röttger et al., 2024). We scored every completion with
Qwen3Guard-4B, the guard behind every refusal number above, and compared its flag with the
final human label on the 250 safe prompts per model [P15] (Figure 7).

*Table 7. The guard's refusal flag against XSTest's human labels, safe prompts. Recall and
false-alarm rate against full refusal alone (strict) and against full or partial refusal
(broad).*

| model | humans: full / partial refusal | guard flags | strict: recall, false alarms | broad: recall, false alarms |
|---|---|---|---|---|
| GPT-4 | 0.064 / 0.020 | 0.088 | 1.00, 0.03 | 0.90, 0.01 |
| Llama 2, original system prompt | 0.380 / 0.216 | 0.636 | 0.96, 0.44 | 0.94, 0.19 |
| Llama 2, no system prompt | 0.140 / 0.156 | 0.304 | 0.94, 0.20 | 0.85, 0.07 |
| Mistral instruct | 0.008 / 0.008 | 0.036 | 0.50, 0.03 | 0.50, 0.03 |
| Mistral with a guardrail prompt | 0.096 / 0.092 | 0.168 | 0.75, 0.11 | 0.70, 0.04 |
| pooled (1,250) | 0.138 / 0.098 | 0.246 | 0.92 (0.88-0.96), 0.14 | 0.87 (0.82-0.91), 0.05 |

Three readings.

1. **The guard's flag is a broad one.** It marks 92% of human full refusals (159 of 172), 80% of
   partial refusals (98 of 123) and 5% of full compliance (51 of 955). Its agreement with the
   broad human label (Cohen's kappa 0.81) is close to two humans' agreement on the three classes
   (0.88); with the strict label it is 0.59 against the humans' 0.90.
2. **What it is worth as a predictor.** The squared correlation of the human label with the
   guard's logit is 0.70 for the broad event and 0.46 (95% interval 0.39-0.53) for the strict
   one. By section 6 that is roughly a threefold and a twofold gain in labels when unlabelled
   responses are plentiful. The logit beats the 0/1 flag for the strict event (0.46 against
   0.40) and for the broad one (0.70 against 0.65).
3. **Its error rates belong to a model.** The false-alarm rate against the strict label runs
   from 0.03 (GPT-4) to 0.44 (Llama 2 with its original system prompt), following each model's
   share of partial refusals; recall runs from 0.75 to 1.00 among the four models with more
   than two refusals. This is section 7.1 on other people's models and labels: a calibration
   measured on one of these models would not serve another.

A consequence for section 8.4: the certified event there is strict refusal, and the guard that
shaped the training measures the broad one. The guard can still reduce variance (reading 2);
it cannot stand in for the label.

**Our own policies.** One annotator (an author) labelled 220 responses to benign prompts: a
stratified sample of the reference model's and the trained policy's responses, read with its
design weights, under a guideline that follows XSTest's scheme with refuse-then-answer as its
own label [P6]. Figure 8 and Table 8 give the result.

*Table 8. Human labels on our own policies (one annotator, 220 responses; conservative 95%
intervals in brackets).*

| | reference model | policy trained under the constraint |
|---|---|---|
| the guard's refusal flag, share of the pool | 17.9% | 17.4% |
| human label: refuses | 10.7% [4.7, 21.5] | 10.6% [5.2, 21.2] |
| human label: refuse-then-answer | 3.4% [0.4, 15.5] | 0.3% [0.0, 11.0] |
| guard's false-alarm rate against "refuses" | 8.1% | 7.7% |
| guard's precision | 0.59 | 0.60 |
| guard's recall: estimate, conservative lower limit | 0.99, 0.44 | 0.98, 0.45 |
| rho^2 of the human label with the guard's logit | 0.61 | 0.60 |

In human terms the two policies refuse equally often (difference -0.1 points, prompt-clustered
standard error 3.9), and about 8 points less often than the guard's flag says. The guard missed
almost nothing the annotator called a refusal, but the sample cannot rule misses out: none was
found among 72 and 70 responses in the large guard-negative strata, which bounds recall below
only at 0.44. As a predictor the guard is worth more than a halving of the labels
(`rho^2` 0.6), in line with the published-label result above.

One annotator means no agreement statistic. What stands in for it is thin and we say so: the
guideline's classes are XSTest's, whose authors report agreement of 0.9 on them; three
responses that appeared twice on the annotator's sheets received the same label; and two
recurring cases (a redirect that says where to find the answer; "I cannot answer" followed by
the false premise) were labelled both ways and are the main source of label noise. `[GAP: P16]`
adds the annotator's agreement with XSTest's two-annotator labels on 60 of its completions.

`[GAP: P8]` The same for harm, on a sheet aimed at 30 or more human positives.

### 8.4 What a certificate in human labels costs

The constraint of the trained policy is relative: its refusal rate may exceed the reference's
by at most a margin of 0.02. Section 8.3 puts the two human-terms rates at 10.7% and 10.6%. With
a true difference near zero, whether a sample certifies the margin is a question about the
label budget. We report that budget and the margin a fixed sample would be expected to
certify. We prepared the measurement and did not run it.

*Table 9. Prompt pairs needed to certify a margin on the difference of the two strict refusal
rates at delta 0.05, from the rates of section 8.3 [P9 budget]. A design calculation: normal-type
limits, pairing correlation 0.66 (the guard's flags on the same prompts), guard `rho^2` 0.6. The
two guard columns differ in what the claim covers: the pool's own 490 prompts, where guard-only
responses pin down the guard's mean, or new prompts from the same source, where the pool's
unlabelled prompts are all the guard has.*

| margin | chance of certifying | labels alone, unpaired | labels alone, paired by prompt | with the guard, pool rate | with the guard, new prompts |
|---|---|---|---|---|---|
| 0.02 | 50% | 1,226 | 416 | 167 | 339 |
| 0.02 | 80% | 2,801 | 950 | 380 | not within 490 prompts |
| 0.03 | 80% | 1,266 | 429 | 172 | 362 |
| 0.05 | 80% | 462 | 157 | 63 | 78 |

Pairing the two policies on the same prompts cuts the budget threefold. What the guard adds
depends on the claim. For the rate on the pool's own prompts, a guard-only response costs GPU
seconds, the guard's mean can be measured as closely as wanted, and the guard cuts the budget
again by 2.5: a certificate that would take 2,800 unpaired pairs takes 380. For new prompts from
the same source, the guard's mean is known only through the pool's unlabelled prompts. With 300
of 490 labelled it removes 23% of the variance, not 60%, and the 2-point margin at an 80% chance
is out of reach inside the pool. A predictor removes label noise. It does not remove the
uncertainty of a small prompt set.

**Prepared, not run.** We drew 300 prompt pairs of fresh responses (256 tokens) from the
returned policy of one training run and from its reference, with 8 further responses per policy
on each of the 490 pool prompts that only the guard reads, and scored all 8,820 responses with
the guard [P9 data]. The analysis was fixed before any label: four upper limits at delta 0.05
on the difference, each computed once on the full sample. (a) Labels alone, the betting bound
(exact). (a') Labels alone, bootstrap-t (approximate, the like-for-like comparator). (b1) With
the guard, for the pool rate, PPI++ with a bootstrap-t limit. (b2) With the guard, for new
prompts, the same with the other 190 prompts' pairs as the unlabelled data.

A check of the whole design on the guard's real scores, with synthetic labels drawn from the
guard's logit at the rates of section 8.3, puts the miss rates of the four limits at 0.001,
0.040, 0.045 and 0.042 for the pool rate against a level of 0.05, and their mean certified
margins at 0.065, 0.037, 0.032 and 0.035 [P9 design check]. The exact bound is about twice as
wide as the approximate ones: with some 30 discordant pairs in 300, exactness is the larger
cost and the guard's gain is the smaller one. The synthetic labels disagree with the guard
independently across the two policies, which weakens both the pairing and the guard, so these
margins are on the wide side and those of Table 9 on the narrow side.

On either set of figures a sample of 300 pairs is expected to certify a margin of 2.4 to 6.5
points with labels alone, not the 2-point target. The 600 labels were not collected. The
samples, the guard's scores, the blind sheet and the analysis script are in the repository, so
the certificate can be computed from 600 labels with no design choice left open. One property
of the sheet would limit that measurement: the reference writes longer responses than the
trained policy (57% against 32% of the sheet's items run to the token limit), so length is a
weak cue to the policy.

## 9. Limits

- **Labels.** Outside section 8.3, and the budget of section 8.4 computed from it, every
  refusal and harm number is relative to a guard model's field. The human refusal labels are
  one annotator's, an author's, with no measured agreement; the published XSTest labels are
  the independent check.
- **No certificate of a trained policy in human labels.** Section 8.4 costs one and prepares
  it; the labels were not collected. Every certificate of a trained policy in this paper is in
  a guard model's terms.
- **Scale.** Trained policies are 0.5B-3B on one consumer card; the frontier evidence is
  certificates on published traces.
- **One run.** The policy that section 8.4's samples come from is one training run with one
  seed.
- **Benchmarks are not deployments.** Sections 8.1-8.2 certify a rate over a benchmark's task
  distribution. The clustered bound treats user tasks as sampled from a population of tasks
  like them; nothing is claimed about tasks unlike them.
- **Approximate bounds.** `b1w`, the bootstrap-t limits and the cluster bootstrap are checked
  by resampling, not proved at finite n. Table 2 is the evidence, with its Monte Carlo error.
- **Provenance.** The per-prompt data of the rounds cited from the training paper did not
  survive a disk loss; those numbers trace to that paper's tables and cannot be regenerated
  without retraining (Appendix A).

## 10. Related work

**Seldonian algorithms.** Thomas et al. (2019) define the framework: a candidate/safety split,
a high-confidence test, and NSF. We use its safety test unchanged and study what it certifies
when the constrained quantity is a judged property of generated text.

**Error bars for evaluations.** Miller (2024) sets out standard errors for language-model
evaluations, including clustered ones. Bowyer et al. (2025) show that normal-approximation
intervals fail at small n and recommend alternatives. Our Table 2 agrees and adds the
one-sided, selection-robust setting and the failure on crossed designs (section 7.3), which
clustered standard errors in one dimension do not cover.

**Risk control.** Risk-controlling prediction sets (Bates et al., 2021), Learn-then-Test
(Angelopoulos et al., 2025) and conformal risk control (Angelopoulos et al., 2024) calibrate a
post-processing parameter on held-out data so that a risk is controlled with high probability
or in expectation. They share the finite-sample, distribution-free aim and the possibility of
finding no valid setting. They control a wrapper around a fixed model; we certify a policy
that training has changed, on a label a judge defines. Khosravi and Huo (2026) give
anytime-valid selective risk control at deployment for models trained with verifiable
rewards; their guarantee is online and per decision, ours is a pre-deployment statement about
a rate.

**Prediction-powered inference.** Angelopoulos et al. (2023a, 2023b) combine a few gold labels
with many model predictions; Boyeau et al. (2025) apply it to model evaluation; Fisch et al.
(2024) add stratification; Zrnic and Candès (2024) and Gligorić et al. (2025) choose which
items to label; Zrnic (2024) gives a bootstrap; Csillag et al. (2025) give e-value versions.
Sections 5 and 6 sit inside this line. What we add is evidence on when the intervals hold at
the sample sizes and rates of a safety test (the normal-quantile intervals of PPI++ and
StratPPI do not; their estimators with a bootstrap-t limit do), a stratifier that a relative
constraint supplies for free, and the negative of section 7.2: the predictions' relation to
the label is not stable under training that targets the label.

**Betting bounds.** Waudby-Smith and Ramdas (2024) give the betting confidence sequences we use
for bounded means and for the sequential test of section 8.4; the stratified constructions
tried in section 5 follow Spertus and Stark (2022).

**Benchmarks and labels.** XSTest (Röttger et al., 2024) and OR-Bench (Cui et al., 2025) supply
the benign prompts; XSTest's three-class annotation scheme is the basis of our refusal
guideline. AgentDojo (Debenedetti et al., 2024) supplies the crossed design of section 8.2.
Qwen3Guard (Qwen Team, 2025) is the guard.

## References

Verified against the publisher or arXiv page on 2026-10-04; BibTeX in
`reports/paper_certification.bib`.

- Angelopoulos, Bates, Fannjiang, Jordan, Zrnic (2023a). Prediction-powered inference. *Science* 382(6671).
- Angelopoulos, Duchi, Zrnic (2023b). PPI++: Efficient prediction-powered inference. arXiv:2311.01453.
- Angelopoulos, Bates, Fisch, Lei, Schuster (2024). Conformal risk control. ICLR 2024.
- Angelopoulos, Bates, Candès, Jordan, Lei (2025). Learn then test: calibrating predictive algorithms to achieve risk control. *Annals of Applied Statistics* 19(2).
- Bates, Angelopoulos, Lei, Malik, Jordan (2021). Distribution-free, risk-controlling prediction sets. *Journal of the ACM* 68(6).
- Bowyer, Aitchison, Ivanova (2025). Position: Don't use the CLT in LLM evals with fewer than a few hundred datapoints. ICML 2025. arXiv:2503.01747.
- Boyeau, Angelopoulos, Yosef, Malik, Jordan (2025). AutoEval done right: using synthetic data for model evaluation. ICML 2025. `[CHECK]` venue and arXiv id.
- Csillag, Struchiner, Goedert (2025). Prediction-powered e-values. arXiv:2502.04294.
- Cui, Chiang, Stoica, Hsieh (2025). OR-Bench: an over-refusal benchmark for large language models. ICML 2025 (PMLR 267). arXiv:2405.20947.
- Debenedetti, Zhang, Balunovic, Beurer-Kellner, Fischer, Tramèr (2024). AgentDojo: a dynamic environment to evaluate prompt injection attacks and defenses for LLM agents. NeurIPS 2024 Datasets and Benchmarks.
- Fisch, Maynez, Hofer, Dhingra, Globerson, Cohen (2024). Stratified prediction-powered inference for effective hybrid evaluation of language models. NeurIPS 2024. arXiv:2406.04291.
- Franklin, Schneeweiss, Polinski, Rassen (2014). Plasmode simulation for the evaluation of pharmacoepidemiologic methods in complex healthcare databases. *Computational Statistics & Data Analysis* 72.
- Gligorić, Zrnic, Lee, Candès, Jurafsky (2025). Can unconfident LLM annotations be used for confident conclusions? NAACL 2025. arXiv:2408.15204.
- Khosravi, Huo (2026). Conformal selective acting: anytime-valid risk control for RLVR-trained LLMs. arXiv:2605.20270.
- Miller (2024). Adding error bars to evals: a statistical approach to language model evaluations. arXiv:2411.00640.
- Qwen Team (2025). Qwen3Guard technical report. arXiv:2510.14276.
- Röttger, Kirk, Vidgen, Attanasio, Bianchi, Hovy (2024). XSTest: a test suite for identifying exaggerated safety behaviours in large language models. NAACL 2024. arXiv:2308.01263. `[CHECK]` author list.
- Spertus, Stark (2022). Sweeter than SUITE: supermartingale stratified union-intersection tests of elections. arXiv:2207.03379.
- Thomas, Castro da Silva, Barto, Giguere, Brun, Brunskill (2019). Preventing undesirable behavior of intelligent machines. *Science* 366(6468).
- Waudby-Smith, Ramdas (2024). Estimating means of bounded random variables by betting. *JRSS-B* 86(1).
- Zrnic (2024). A note on the prediction-powered bootstrap. arXiv:2405.18379.
- Zrnic, Candès (2024). Active statistical inference. ICML 2024 (PMLR 235).

## Appendix A. Where each number comes from

| tag | file | regenerable |
|---|---|---|
| [SR 2.1], [SR 2.3], [SR 2.4] | `reports/state_2026-10-01.md`, sections 2.1, 2.3, 2.4 (a digest of the training paper and spikes 004-017) | see the rows below |
| [R 6.2], [R 6.3] | `reports/paper_seldonian_llm.md`, sections 6.2 and 6.3 | 6.3 yes (`scripts/synthetic_calibration.py`, CPU); 6.2 no (cached labels lost 2026-09-19) |
| [012] | `.planning/spikes/012-rerandomized-split/README.md` | yes, CPU |
| [013], [013 4] | `.planning/spikes/013-stratified-safety-set/README.md` (item 4 of its trail for the in-loop run) | yes; responses in `results/spikes/013/` |
| [013 H8] | `.planning/spikes/013-stratified-safety-set/validity_H8.md` | yes |
| [014] | `.planning/spikes/014-pushed-label-stratification/README.md` | yes; adapters on the D: drive |
| [017 n], [017 B6], [017 E8] | `.planning/spikes/017-calibration-carrying-certificate/README.md` (Results n, E8 addendum) and `results.md` (B6) | yes |
| [019] | `.planning/spikes/019-external-trace-certificate/README.md` | yes; transcripts on the D: drive |
| [P6] | `results/labels/refusal/analysis.md` (`scripts/refusal_labels.py analyze`; labels in `labels_ah.jsonl`, design in `design.json`) | yes |
| [P9 budget] | `results/paper/p9_budget.md` (`scripts/p9_budget.py`) | yes |
| [P9 design check] | `results/labels/p9/design_check.md` (`scripts/p9_certificate.py check`) | yes |
| [P9 data] | `results/labels/p9/` (`scripts/p9_sample.py`, `p9_sheet_build.py`; analysis fixed in `p9_certificate.py`) | the sheet and scores yes; the samples need the GPU |
| [P14] | `results/paper/stratppi.md`, `stratppi.json` (`scripts/stratppi_baseline.py`; 5,000 draws per cell in part A, 4,000 in part B) | yes, CPU, about 5 minutes |
| [P15] | `results/labels/xstest/analysis.md` (`scripts/xstest_guard.py`; guard scores and human labels in `guard_scores.jsonl`) | yes; XSTest's completions are fetched from its repository and kept outside this one |
| [020], [020 P] | `.planning/spikes/020-agentdojo-injection-certificate/README.md` and `plasmode.md` | yes |

**Open checks before v1.0.**

1. `[013 H8]` against the 013 README: the README says the mid-rate misses are at most 0.093
   (delta 0.1) and 0.047 (delta 0.05); the table has one label at 0.097 and 0.054 (refusal on
   encoded prompts, rate 94%). Table 2 quotes the table.
2. `[R 6.2]`: the sentence giving the t-bound's misses at delta 0.05 repeats the
   Clopper-Pearson row digit for digit, and the source data are gone. Table 2 uses only the
   delta 0.1 table.
3. The 37% and 82% shares in section 7.2 are counts by the guard's flag and a surface pattern
   (`results/labels/refusal/design.json`), not human labels; the text says so.
4. Round 6 seed-level numbers in Table 1 trace to the state report only.
5. Solution rates for the certificates of section 8 are not defined (fixed published traces);
   say so where NSF is reported.

## Appendix B. Figures

`scripts/paper_figures.py` writes each figure to `reports/figs/` as a PDF, a PNG and a CSV of
every number drawn.

| figure | file | section |
|---|---|---|
| 1. Which bounds hold their level | `fig1_validity` | 3 |
| 2. Gain from reference-rate strata against the reference's ICC | `fig2_strata_ess` | 5 |
| 3. StratPPI as published and with a bootstrap-t limit | `fig3_stratppi` | 5, 6 |
| 4. A carried calibration: miss rate and recall by shift | `fig4_carrying` | 7 |
| 5. What a 120-trial benchmark can certify | `fig5_robodojo` | 8.1 |
| 6. AgentDojo: per-pair and clustered bounds, 28 pipelines | `fig6_agentdojo` | 8.2 |
| 7. What the guard flags, by human class, on XSTest | `fig7_guard_xstest` | 8.3 |
| 8. Guard and human refusal rates on our two policies | `fig8_refusal_sheet` | 8.3 |

