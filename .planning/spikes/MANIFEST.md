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
