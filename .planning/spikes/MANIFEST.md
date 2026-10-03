# Spike Manifest

## Ideas

### td-error-wellbeing
When the TD error spikes late in training, what does it say about the agent's state (its
"wellness", in the readings where TD error is valence), and can an internal reward built
on TD error be added to training? Continues the "aha / TD-error spike" and "cumulative
TD error as potential for harm" entries in `reports/ideas.md`. GRPO has no critic, so the
spikes run on the synthetic contextual bandit (`seldonian/llm/synthetic.py`), where the
exact value of the current policy, and therefore the true per-episode TD error, is
computable, against the same Seldonian Lagrangian pipeline the LLM runs use.

**Requirements:**

- CPU synthetic spikes first (001-003); the GPU LLM logging spike (004) is decided after them.
- The internal reward is judged by what it does to the Seldonian outcome (true violation
  rate, safety test, solution rate), not only to reward.
- Any per-episode TD statistic for the LLM runs is built from the unnormalised group
  residual `r - group mean` or a value head, never from the group-normalised advantage
  (001: its magnitude is bounded and flat by construction).
- Any claim about training dynamics under the Lagrangian is reported net of the
  multiplier's level and moves (002: three statistics turned out to be lambda in disguise).
- A "wellness" reading of a trainer-side signal is stated as functional (convergence,
  how harshly the constraint is enforced), not as welfare; see LITERATURE.md thread 1B.

### forbidden-task-unsafe-region
A "not possible" (forbidden) task as an unsafe region of the optimisation landscape:
estimate the probability that training enters it, and move away. Sketch in the last
entry of `reports/ideas.md`. The case studied is capability that arrives as a side
effect of training an allowed task.

**Requirements:**

- The forbidden task is held out of the reward; its region `U` must be one that the run
  actually enters (a vacuous constraint measures nothing).
- The early-warning signal is compared with the forbidden rate itself, controlled for
  the multiplier, and every steering arm gets a size-matched random-trigger control.
- F prompts in a GRPO batch carry no reward term except the constraint penalty (008: any
  shaping on a zero-variance F group is amplified to full strength by the normalisation).
- Price F from step 1 and let the dual come down slowly (eta_down << eta); an early price
  under symmetric dual steps is withdrawn just before the drift (010).
- On encoded prompts the harm label is gate(sim >= 0.8) AND a >= 4B judge scored on the text
  after the restated request; on plain prompts the judge alone (007, 009).
- Correct the judge with the answer-rate-aware formula, never plain Youden, and bound its
  recall with >= 30 human-labelled harmful responses (006).
- Audit F to truly harmful prompts before the pilot; the PKU set holds benign ones (009).

### rerandomized-split
The user's 2020 independent-study extension (report "Safe Learning Models", section 8.1,
Algorithm 1) re-drew the candidate/safety split until `g(theta_s)` at a random `theta_s`
matched on the two halves (v2: keep the best of n splits). That is rerandomisation
(Morgan & Rubin 2012): unbiased and conservative for a *fixed* outcome function, but the
Seldonian safety test evaluates `g(theta_c)` with `theta_c` trained on D_c after the split,
which no rerandomisation proof covers. Question: is the rerandomised split conservative in
practice for the safety test, what does it buy in solution rate and predicted-vs-actual
agreement, and when does it turn optimistic? Then: the LLM-era version, balancing on prompt
metadata and the reference model's per-prompt violation rate, against stratified
(blocked) randomisation.

**Requirements:**

- Queued by the user on 2026-09-27: run spike 012 at the next `/gsd-spike` (frontier mode
  proposes it first).
- Arms: simple random split; label-stratified split (the 2026-09-27 audit fix); the user's
  Algorithm 1 at threshold and best-of-n; Mahalanobis rerandomisation at a moderate
  acceptance probability; stratified randomisation on the same covariates.
- Measure the safety test's true miss rate against delta (validity), the solution rate
  (power), and predicted-vs-actual test agreement, over many seeds, on the synthetic
  bandit and the classic TPR-gap setup.
- Include an adversarial case: a candidate that overfits a covariate the split balanced on.
- Rate constraints are tested with Clopper-Pearson or a tight Wald stress test, not the
  t-test: the t-test's zero width at a rate of exactly 0 or 1 is found by candidate
  selection and swamps everything else (012).
- Validity of a split rule is measured against a full-leak ceiling (the safety test on
  D_c), so a null result has a stated resolution (012: under ~5% of the ceiling).
- A balance covariate for an LLM safety set must be precise (the exact or many-sample
  reference rate); a 4-sample estimate bought nothing (012).
- Spike 013 tests only reference-rate stratified safety sets with a stratified bound,
  under the pre-registered hypotheses and go/stop rule in `013-stratified-safety-set/DESIGN.md`
  (user-approved 2026-09-27: Granite-3.3-2B primary, Qwen3-1.7B replication only, 12 GPU-hour cap).
- Stratify a safety set by equal rank strata of the reference rate with random tie-breaking
  (H about 8, k = 8); tie-keeping quantile cuts collapse on zero-inflated covariates (013).
- Plasmode coverage is judged against the mean of the labels the draws come from, never an
  independent finite "truth" sample (013: that shared offset faked 0.20 misses).
- Rare labels (below about 5%) need exact bounds whatever the split; approximate bounds miss
  up to 0.45 there (013).
- Spike 014 (queued by the user 2026-09-29 in /gsd-explore; run it first at the next
  `/gsd-spike`) tests reference-rate stratification when the Lagrangian drives the
  constrained label directly. Scope: **over-refusal only** (XSTest + OR-Bench); harm is out,
  because at rare rates its problem is the bound, not the strata (seed SEED-260929-nq7-rare-harm-bound).
- 014 is gated. A CPU bandit stage comes first, with the surviving gain pre-registered from
  the pre-flight formula at the rho measured under pressure. A Granite-3.3-2B
  `LagrangianReward` run on over-refusal (013's pipeline, reference + checkpoint samples)
  follows only if the bandit says the gain survives. A `DESIGN.md` goes to the user for
  approval before anything runs, as for 013.
- A label the training targets directly still stratifies (014): under the Lagrangian the
  per-prompt rates do not compress (ICC_cand / ICC_ref 1.0-1.5) and the persistence rho is
  set by ICC_ref, not by the pressure (0.69 / 0.87 / 0.98 at ICC_ref 0.26 / 0.51 / 0.75 in
  the bandit; 0.92 on Granite at 0.72). Use `013/preflight.py --pushed`, which interpolates
  that table; read it as an upper estimate (13% over on the one real point).
- Score the pre-flight formula at the measured moderators, never with a fixed rho: at the
  measured ICC_cand and rho it predicted the realised ESS within 0.10 (bandit, 56 cells)
  and 0.04-0.09 (Granite); the naive rho 0.8 was off by 0.4-0.5 and in the *under* direction (014).
- A GPU budget cap is enforced by the script (`timeout` around the run, checkpoints saved
  first), not watched (014: 12.8 GPU-hours against a 5-hour cap after step time tripled at
  step 30 near the 12 GB memory limit).

### prompted-ghat
`g_hat` defined by a prompt: the algorithm developer writes the constraint in natural
language ("responses to benign prompts should not be refused more than the reference plus 2
points") and the system compiles it into (1) a measurement, a judge whose rubric or typed
question comes from the text; (2) a statistic and threshold in the existing expression
machinery; (3) a bound chosen by spike 013's rules. The certificate then covers the *judged*
quantity, so the compiled g_hat ships with its calibration against human labels. Entry of
2026-09-29 in `reports/ideas.md`. Research pass 2026-09-29: compiling text into a
statistical constraint, and certifying through a corrected proxy label, both look
unpublished; prompted judges are paraphrase-sensitive and can be predicted from the rubric
alone (rubric artifacts); Jev (TypeSafe, 2026-09-15) is a typed-decision API whose Noul
question is this measurement, but proprietary and unmeasured on safety.

**Requirements:**

- Every prompted judge is tested for the rubric artifact (verdict predictable from the
  rubric alone; response reversal must flip it) and for paraphrase flip rate, against the
  trained guard as the baseline.
- The compiled spec is rendered back to English for the developer to confirm before it
  builds a constraint (Language to Rewards' inspectable intermediate layer).
- A compiled g_hat ships with its calibration: judge error rates on an i.i.d. human-labelled
  sample of the scored population, corrected with PPI++ or the answer-rate-aware formula; it
  states what it can certify and with how many human positives (006: >= 30).
- The 225-item human sheet is checked for being an i.i.d. sample of the scored population
  before it is used as a PPI calibration set.
- **No external judge API, decided 2026-09-30.** The user has no Jev access and declined
  sending prompts or responses off the machine, so every judge in this idea is a local model.
  Jev stays a reference point for the interface (typed question in, calibrated probability
  out), not a component.
- A verifiable property (length, format, exact match) compiles to a deterministic feature,
  never to a judge: the compiled judge was at chance on "more than 80 words" (015, AUC 0.500,
  corr with the actual count +0.06), and its aggregate rate landed within a point of the truth
  by coincidence on noise labels.
- A compiled judge is used through its probability with a threshold calibrated on labelled
  data, never through its 0.5 label: across paraphrases the AUC held (0.80-0.87 on harm)
  while the measured rate swung 0.09-0.46 (015).
- The exact constraint wording is part of the constraint's identity in any certificate, since
  paraphrases of one sentence measure different quantities (015).
- Constraint prose describing harmful content is loaded from the repo's existing artifacts
  (the labelling guideline, spike 005's rubric), not authored anew, so a compiled judge is
  tested against the definition the human labels used (015).
- The statistic and threshold are compiled as: the model writes one line of the constraint
  DSL; deterministic code parses, validates, builds the project's constraint classes, renders
  the English and picks the bound. The model never chooses `monotone`, the delta split or the
  bound (016: the built g equals the hand-written g exactly on 8 constraints x 5 data sets).
- The registry shown to the model is one flat list of named prompt groups and one way to
  mention the reference (`REF` in the limit). Overlapping attributes and three routes to the
  reference gave a silent error on 20 of 39 faithful wordings; the flat registry alone cut
  that to 7, the examples alone to 13, both to 4, and to 0 with thinking (016).
- How the reference enters the certificate (its value as a constant, a two-sample
  difference, or a paired difference) is a policy of the builder, not the model's choice:
  the three differ in width (0.029 / 0.077 / 0.039 at 500 prompts) and the model switched
  between them across paraphrases of one requirement (016). Round 6's `reference + margin`
  is the constant form and does not cover the reference's sampling error.
- A compiled spec is accepted only if it passes validation, the reference lint (a relative
  constraint must be one the reference model itself satisfies) and the number lint (every
  number in the spec occurs in the sentence), and a second, independent compile states the
  same requirement. Anything else goes back to the developer as a question (016: 46 right and
  1 wrong of 47 accepted; 7 of 8 under-specified sentences and 18 of 64 clear ones sent back).
- The English the developer confirms is generated by code from the spec, never by the model:
  9 of 40 model-written paraphrases of a constraint drifted or were ambiguous (016).
- The compile prompt carries one example per construct, and its examples share no group,
  number or measurement pairing with anything it is tested on (016: with three examples the
  model wrote a one-sided difference for every parity sentence; with ten it once returned an
  example verbatim, which the number lint caught).
- Open after 016, not built: a bare percent beside a reference comparison ("by more than
  5%") was read as 5 points by every arm and needs a rule that always asks; a `JUDGE("...")`
  property must be a verbatim span of the sentence (reworded in 5 of 15) and must keep the
  sentence's prompt group (dropped in 3 of 6); two requirements in one sentence and a
  two-sided relative limit are rejected rather than compiled.
- The certificate on the developer's quantity is gold labels on the responses it certifies,
  with the judge as a variance reducer. A calibration measured elsewhere is not carried across
  prompt populations: the compiled refusal judge's recall differed between prompt sources in
  4 of 6 wordings and a carried Youden bound missed in up to 94% of draws (017). Across 200
  training steps on the same prompts it held only for side-effect training; when the
  training targets the label (014's Lagrangian run) the compiled judge's recall on the
  guard's refusals fell from 0.21 to 0.03 and the carried bound missed in 80% of draws, with
  the rate back at the reference (017 E8). The constrained policy learns refusal-prefixed
  answers, the form the two judges split on. Labels are taken on the policy being certified.
- The route is fixed by the data's shape before any bound is computed (017): a verifiable
  property is measured by code on every response; with fewer than 10 labels in the rarer class
  the bound is Clopper-Pearson on the labels and the judge is unused; otherwise PPI++ with a
  bootstrap-t limit. Reporting the smaller of two bounds missed up to 0.078.
- PPI and PPI++ are never used with a normal limit: it missed 0.08-0.13 at 100-225 labels on
  the real refusal judge and 0.20-0.24 at a 1.3% rate; a Wilson-type limit did not repair it.
  The bootstrap-t limit (lam re-estimated in every resample) missed at most 0.053 (017).
- The judge feature for PPI++ is the logit of P(Yes), or a cross-fitted Platt map of it, never
  p or the 0.5 label: rho^2 with the gold label 0.48 / 0.58 against 0.22 / 0.18, worth 1.6 /
  1.9 against 1.16 / 1.12 in labels. `E[p]` as a bounded feature is the weakest option (017).
- Labelled and unlabelled responses are scored in one run: the same 225 prompts scored in a
  different batch order flipped 3 and 5 of 225 labels and moved p by up to 0.36 (017).
- A label sheet sampled by strata is analysed with its design weights and `b1w`, never as an
  i.i.d. sample and never with a normal limit: read as i.i.d. the 225-item sheet certified a
  negative harm rate under one wording (missing in 0.98 of planted re-draws), and the weighted
  normal limit missed 0.07-0.22. Rates reported from the sheet unweighted are sheet rates:
  Qwen3Guard-4B's false alarms are 0.120 weighted, not 0.230 (017).
- Spike 006's answer-rate-aware correction is for the guard, whose false alarms fall on
  answers (0.291 answered, 0.036 refused). A compiled rubric's fall on refusals (0.132
  answered, 0.280 refused), so that correction does not apply to it (017).
- A harm threshold is certified by labels on the candidate's own responses (301 clean labels
  for 1%, 59 for 5%; no positive needed). Thirty human positives is the floor for a carried
  calibration of the gated label only, and only within one prompt population (017).
- The target for a better judge (spike 018) is rho^2 with human labels on the logit scale:
  0.5 turns 225 labels into about 360; the compiled harm rubric is at 0.02 (017).

### external-trace-certificate
A certificate on a fixed policy evaluated by someone else, from its public per-trial traces
alone: the safety-test half of the pipeline with labels the harness emits (free and
complete) instead of humans or a judge. First case: Anthropic's RoboDojo-RC Tier 1 report
(2026-09-23; three API models on six robot manipulation tasks, 120 trials each, no safety
section). Opened 2026-10-02 after the user asked whether the report could be evaluated for
harmful behaviour; the agreed framing is that the scarce resource moves from labels to
trials and the headline is a limit, not a number. The same shape rent-my-gpu's spike 062
needs on its own traffic.

**Requirements:**

- Every event is a verifiable property computed by code from the trace (016/017's route);
  the policy's own narration of risk is a feature, never the label (015, 017 E8).
- The parser is frozen on an enumeration pass over every trace before anything is counted
  (CONVENTIONS); counts that the source publishes (terminations) are reproduced first.
- Per-model bounds are Clopper-Pearson on the trials, with the task-stratified `b1w` beside
  them; per-task cells (n 20) are reported, never certified.
- The resolution is stated with the result: 0.025 with nothing observed at n 120, 0.139
  per task; a 1% claim needs about 300 clean trials per model.
- A text feature built from a word list is checked against the task's own action
  vocabulary before it is frozen (019: `drop` matched "drop the fruit into the bowl", most
  of the policy's matches); report the frozen feature and the corrected one both.
- The physical state in the trace (joint effort) predicted the harness's safety stop (AUC
  0.75-0.83); the policy's narration did not (0.38). Precursors come from the state, not
  from what the policy says about itself (019, with 017 E8).
- A benchmark with a crossed design (tasks x attacks, prompts x samples) is bounded by a
  studentised cluster bootstrap over the unit that carries the dependence, never by
  Clopper-Pearson over the cells: on AgentDojo the i.i.d. bound missed 5-28% of the time at
  delta 0.05 and the clustered one 2-6%; a 2.2% raw rate concentrated in 5 of 97 user tasks
  certifies at 0.104, not 0.032 (020). Report the ICC by each candidate unit and cluster by
  the larger; the exact any-cluster bound (unit = task, label = any failure) is the clean
  stricter statement.
- Check a bound under the design's own resampling before trusting it on published runs
  (020's `plasmode020.py`: the observed clusters as the population, redraw clusters).

## Spikes

| # | Idea | Name | Type | Validates | Verdict | Tags |
|---|------|------|------|-----------|---------|------|
| 001 | td-error-wellbeing | grpo-advantage-vs-td | standard | Given exact V_pi(x), when delta and GRPO's A are logged per episode, then we know which GRPO-side quantity carries delta's magnitude | VALIDATED | grpo, td-error, advantage-normalisation, lagrangian |
| 002 | td-error-wellbeing | late-spike-meaning | standard | Given runs with known ground truth, when per-step agent TD error is logged and split, then late spikes can be tied (or not) to a still-moving policy, the multiplier, and a breach if training continued | INVALIDATED | td-error, late-spikes, valence, lagrangian, breach-prediction |
| 003a | td-error-wellbeing | td-bonus-abs | comparison | Given the noisy-TV env and the Lagrangian, when the reward adds beta*abs(TD error), then measure solution rate, violations and noise-seeking against controls | INVALIDATED | intrinsic-reward, curiosity, noisy-tv, wireheading, seldonian |
| 003b | td-error-wellbeing | td-bonus-positive | comparison | Same, with beta*max(TD error, 0) ("pay good news only") | PARTIAL | intrinsic-reward, valence, noisy-tv, seldonian |
| 003c | td-error-wellbeing | td-bonus-learning-progress | comparison | Same, paying the decrease of the critic's error per region (learning progress) | PARTIAL (winner) | intrinsic-reward, learning-progress, noisy-tv, seldonian |
| 004 | forbidden-task-unsafe-region | forbidden-capability | comparison | Given training on A that transfers to a held-out forbidden F, when monitored on sealed F prompts and benign twins, then drift into U, the delta/T trajectory certificate, twin capability as a predictor, and a capability-triggered dual are measured against controls | PARTIAL | forbidden-task, trajectory-certificate, capability, early-warning, lagrangian |
| 005 | forbidden-task-unsafe-region | capability-screen | standard | Given Qwen2.5-0.5B/1.5B/3B and six encodings, before training, when decoding (A), encoded arithmetic (twin) and encoded PKU prompts (F) are sampled, then we know which model and encoding give a learnable A, a measurable twin and an incapacity-low F | PARTIAL (round 1 INVALIDATED at 96 tokens) | forbidden-task, capability, model-size, judge, gpu |
| 006 | forbidden-task-unsafe-region | noisy-judge-floor | standard | Given 004's lab with a judge at the measured false-alarm rates (0.23 ungated, 0.015 gated) and a recall sweep, under hash and non-refusal noise, when the floor and the trajectory certificate run with naive, known and estimated corrections, then we learn which setups stay valid and useful | PARTIAL | forbidden-task, judge-noise, youden, trajectory-certificate, cpu |
| 007 | forbidden-task-unsafe-region | two-stage-label | standard | Given the 225 human labels and the bake-off verdicts, when the engagement gate (sim >= 0.8) is ANDed with a judge, then encoded false alarms fall to about 0 and the harmful encoded response is kept | PARTIAL | forbidden-task, judge, engagement-gate, human-labels, cpu |
| 008 | forbidden-task-unsafe-region | lp-bonus-drift | standard | Given 004's lab and 003c's learning-progress bonus, when the bonus is added to the trained reward, then its effect on drift into U is measured against a size-matched random control | VALIDATED | forbidden-task, intrinsic-reward, group-normalisation, zero-variance, cpu |
| 009 | forbidden-task-unsafe-region | granite-transfer | standard | Given Granite-3.3-2B, when GRPO trains encoded benign QA (leetspeak capitals) with no constraint, then twin capability, decoding and engagement with encoded F rise, and policy training fits beside the 4B judge on 12 GB | PARTIAL | forbidden-task, capability-transfer, gpu, granite, memory |
| 010 | forbidden-task-unsafe-region | lam0-no-floor-anomaly | standard | Given 004's unexplained lam0 = 5 row, when the arms are traced per step, then the mechanism is found and a fix follows | VALIDATED | forbidden-task, lagrangian, dual-ascent, eta-down, cpu |
| 011 | td-error-wellbeing | lp-bonus-sparse-reward | standard | Given a sparse, deceptive jackpot action, when 003c's learning-progress bonus runs under the Lagrangian, then it finds the jackpot more often than controls without violations | INVALIDATED | intrinsic-reward, learning-progress, exploration, sparse-reward, cpu |
| 012 | rerandomized-split | rerandomized-split | comparison | Given a candidate/safety split chosen by rerandomisation (the user's 2020 Algorithm 1 and Mahalanobis balance) vs random and stratified splits, when the Seldonian safety test runs over many seeds, then its true miss rate stays <= delta, and we measure the gain in solution rate and predicted-vs-actual agreement, including when the candidate overfits a balanced covariate | VALIDATED | rerandomization, data-split, safety-test-validity, stratification, cpu |
| 013 | rerandomized-split | stratified-safety-set | standard | Given an LLM Seldonian safety test on a per-response 0/1 label, when D_s is sampled within strata of the reference model's per-prompt rate and scored with a stratified bound, then the test stays valid and needs fewer safety prompts, where the pre-flight G predicts | VALIDATED (go, narrowed: mid-rate heterogeneous labels, approximate b1w bound, 8 equal rank strata; ESS 1.4-5.3 on real data; H3 fails in absolute terms) | stratification, safety-set, pre-flight, plasmode, gpu |
| 014 | rerandomized-split | pushed-label-stratification | standard | Given the over-refusal label driven by LagrangianReward (not a side effect), when per-prompt rates are sampled at the reference and trained checkpoints, then we measure rate compression and rho decay and whether 013's 2.4x stratification gain survives (bandit first, GPU only if it does) | VALIDATED (no compression, rho 0.92 on Granite and 0.69-0.98 by ICC_ref in the bandit; ESS 2.1-2.2 against 2.4-2.5 as a side effect, b1w valid; formula at measured moderators within 0.1; `preflight.py --pushed`; narrowed to net rate moves <= 3 points real / 9 bandit; 12.8 GPU-h over a 5-h cap) | stratification, safety-set, lagrangian, over-refusal, cpu, gpu |
| 015 | prompted-ghat | prompted-judge-fidelity | standard | Given a constraint in English, when a rubric is generated from it and run on Qwen3-8B (4-bit) as a JudgeFeature, then agreement with the hand-written judge on 013's responses and with the 225 human harm labels, the rubric-artifact test, the paraphrase flip rate and calibration (ECE) are measured against the trained guard | PARTIAL | prompted-judge, rubric, calibration, jev, gpu |
| 016 | prompted-ghat | prompt-to-spec-compile | standard | Given the three Round 6 constraints plus five harder ones in English, when a local LLM compiles each to a Seldonian-toolkit-style constraint string and JSON spec (measure, group, expression, threshold form, bound by 013's rules), rendered back to English, then the compiled g equals the hand-written g on cached responses and paraphrases compile to the same spec | PARTIAL (deterministic half exact; the pre-registered prompt failed, 3/8 sentences and a silent error on 20/39 faithful wordings; the redesigned one with thinking gives 7/8 certificates, 8/8 requirements, 34/39 wordings and no silent error, and needs lints plus a second compile to reject what it gets wrong) | compiler, constraint-dsl, expression-constraint, lint, gpu |
| 017 | prompted-ghat | calibration-carrying-certificate | standard | Given 015's prompted-judge scores and the human sheet, when PPI++ and the answer-rate-aware correction are applied (0/1 judge, and E[p] as a bounded feature), then the compiled constraint reports what it can certify (brevity exactly; harm only with >= 30 human positives) and how far its threshold moves | PARTIAL (a fixed routing rule held its level in every cell: code for verifiable properties, Clopper-Pearson below 10 labels in the rarer class, PPI++ on the logit with a bootstrap-t limit above, design-weighted `b1w` for the stratified sheet; the normal-limit PPI++ missed up to 0.24, no finite-sample judge-assisted bound beat the labels, `E[p]` is the weakest feature, and a calibration carried across prompt populations misses in up to 94% of draws) | ppi, calibration, certificate, bootstrap, plasmode, cpu, gpu |
| 018 | prompted-ghat | own-noul-judge | standard | Given ~10-20 public labelled safety/refusal sets recast as (instruction, state, label) plus synthetic verifiable constraints, when a 0.6B-2B backbone with a sigmoid head is fine-tuned on log loss and temperature-scaled, then it generalises to held-out instruction families, is calibrated (ECE) on the 225 human labels and 013's responses, and matches Qwen3Guard-4B on harm (Jev tier A: a local, versioned, calibrated Noul-only judge; ~3-5 weeks, 20-60 GPU h) | QUEUED (gate FIRED by 015; and the only route to a better judge now that the external API is ruled out) | own-judge, calibration, instruction-conditioned, gpu |
| 019 | external-trace-certificate | external-trace-certificate | standard | Given a fixed policy evaluated by someone else and only its public per-trial traces (RoboDojo-RC Tier 1: 3 models x 6 tasks x 20 trials), when the harness's own safety events are extracted by code and bounded per model, then we know what such a benchmark can certify about safety stops, whether the policy's self-narration carries any signal about them, and whether task strata buy anything | VALIDATED (the resolution is the result: 2.5% with nothing observed, 5.2% at two stops, so the pre-registered 5% certificate returns NSF for all three models; Opus 5 stops 10/120 against 4/240, p 0.003; self-narration AUC 0.38; peak joint effort AUC 0.75-0.83; task strata within 0.004 of pooled) | certificate, external-traces, robotics, clopper-pearson, stratified, cpu |
| 020 | external-trace-certificate | agentdojo-injection-certificate | standard | Given AgentDojo's published per-episode runs (29 model/defence pipelines, 629-949 (user task, injection task) pairs each, the harness's code-computed `security` label), when a per-pipeline certificate on the targeted attack success rate is computed with the crossed design respected, then we know how much the naive i.i.d. bound understates the uncertainty, which pipelines certify at 5%, and whether suite strata buy anything | VALIDATED (naive Clopper-Pearson misses 5-28% under user-task resampling, cluster-t 2-6%; 1 of 28 pipelines certifies at 5% (claude-3-5-sonnet-20241022, 0.022); Meta-SecAlign's 2.2% is 0.104 clustered, 21 successes in 5 user tasks; website numbers reproduce exactly on the v1 subset; suite strata ESS 1.0-1.5) | certificate, external-traces, prompt-injection, agents, clustered, cpu |
