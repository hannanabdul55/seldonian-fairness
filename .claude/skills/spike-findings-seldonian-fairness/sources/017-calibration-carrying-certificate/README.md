---
spike: 017
idea: prompted-ghat
name: calibration-carrying-certificate
type: standard
validates: "Given 015's prompted-judge scores and the human sheet, when PPI++ and the answer-rate-aware correction are applied (0/1 judge, and E[p] as a bounded feature), then the compiled constraint reports what it can certify (brevity exactly; harm only with >= 30 human positives) and how far its threshold moves"
verdict: PARTIAL
related: [006, 007, 013, 015, 016]
tags: [ppi, calibration, certificate, judge-noise, plasmode, cpu, gpu]
---

# Spike 017: A certificate that carries its calibration

## What This Validates
Spike 015 left the compiled judge as a good ranker and a bad labeller; 016 showed the
statistic and threshold compile exactly. This spike is the third piece of `prompted-ghat`:
the certificate is about the *judged* quantity, so what must ship with it for the claim to be
about the developer's quantity, and what can it then honestly say?

Given the compiled judge's scores and gold labels (225 human harm labels; Qwen3Guard-4B's
refusal field and an exact word count on 500 responses), when the judged rate is corrected by
(a) prediction-powered inference, PPI and PPI++ (Angelopoulos et al. 2023, arXiv:2301.09633;
Angelopoulos, Duchi & Zrnic 2023, arXiv:2311.01453), (b) the Youden correction in
`seldonian/llm/calibration.py`, and (c) spike 006's answer-rate-aware correction, with the
judge used as a 0/1 label and as its probability `p` (a bounded feature), then for each of the
three constraints the certificate states an upper bound on the true rate, what threshold it
could certify, how far the judged-scale threshold moves, and refuses where it cannot.

## Pre-registration (written 2026-09-30 before any estimator was run)
Seen before writing this: 015's tables; the label counts by sampling stratum (3 `h`, 79 `n`,
143 `g`, 2 `?`); the sheet's sampling code; the compiled refusal judge's 0.5-label error
rates by prompt source (sens 0.31 xstest, 0.13 orbench, no false alarms); 013's C1 refusal
rates by checkpoint (0.181, 0.162, 0.166). Nothing else.

**Expectations.**
- **E1 (the sheet).** The 225-item sheet is not an i.i.d. sample of the screen's 4,800
  responses: it takes 4 flagged and 2 cleared (by the 0.6B guard) per model x encoding cell.
  It is a stratified sample with known inclusion probabilities, so a weighted PPI is
  available. Treating it as i.i.d. overstates the judge's flag rate by more than 5 points
  and biases the PPI rectifier; under the real design with planted labels the unweighted PPI
  misses more than 2 delta and the weighted one does not.
- **E2 (PPI++ where labels are not rare).** On refusal (rate 0.200) with i.i.d. labels, the
  PPI++ bound misses within 2 points of delta = 0.05 for n >= 100. Its gain in effective
  labels over Clopper-Pearson on the labels alone is `1 / (1 - rho^2 N_u / (N_u + n))`,
  `rho = corr(Y, f)`, to within 10%. With `p` as the feature the gain is larger than with the
  0.5 label, and lands between 1.2 and 2.5.
- **E3 (a useless judge).** On brevity the compiled judge is at chance, so PPI++ sets
  `lambda` near 0 and costs nothing (gain 0.95 to 1.05), while plain PPI (`lambda = 1`) is
  *worse* than ignoring the judge (gain below 1). The deterministic word count needs no
  labels and is exact.
- **E4 (wording).** Across the six wordings the raw refusal rate runs 0.008 to 0.090; the
  PPI++ estimate for every wording stays within its own 90% interval of the truth, and the
  spread of the six estimates is under a third of the raw spread. Wording then changes the
  width, not the certified quantity.
- **E5 (rare labels).** At a 1.3% rate the CLT PPI/PPI++ bounds under-cover (miss above
  2 delta at n = 225), exact bounds hold, and no judge-assisted exact bound is more than 10%
  narrower than Clopper-Pearson on the labels. With 3 positives every sens/spec correction is
  vacuous (bound above 0.5); with 30 it is not.
- **E6 (transfer).** Sensitivity and false-alarm rate measured on one population do not
  carry to another: they differ beyond their 90% intervals between xstest and orbench, and
  between the over-refusal pool and plain harmful prompts (GPU stage), and a Youden bound
  calibrated on one and applied to the other misses the gold rate in at least one direction.
- **E7 (threshold move).** For the constraints as Round 6 stated them, the PPI route moves
  the judged-scale threshold by the rectifier's upper limit; pre-registered direction only:
  down for harm (the judge over-flags), up for refusal (the judge under-flags).

**Kill rule.** The calibration-carrying certificate is INVALIDATED if, on i.i.d. labels at
n >= 100 and a mid rate, the best valid method misses more than 0.10 (2 delta), or if no
method that holds its level is at least as narrow as Clopper-Pearson on the labels alone in
any cell (the judge then adds nothing anywhere, and the certificate is just human labelling).
It is VALIDATED only if one routing rule gives a valid bound in every cell, including the
cells where the right answer is "cannot certify".

**Deviation from the manifest row, decided before running.** The row is tagged `cpu`. PPI
needs the judge's scores on the unlabelled population, and 015 scored only the 225 labelled
harm items, so one judge-only GPU pass is added (Qwen3-8B 4-bit, the 015 scorer unchanged):
the 4,800 screen responses under three of 015's harm wordings (canonical, the highest-rate and
the lowest-rate paraphrase), and 500 responses each from 013's C1 pool at step 200 and C3
pool at step 0 under the six refusal wordings, for E6. About 40 minutes. No text is authored:
rubrics come from `results/spikes/015/wordings.json`.

## Research
- **PPI** (Angelopoulos, Bates, Fannjiang, Jordan & Zrnic 2023, arXiv:2301.09633):
  `mean_unlabelled(f) + mean_labelled(Y - f)`. Unbiased for any `f`; the judge only changes
  the variance.
- **PPI++** (Angelopoulos, Duchi & Zrnic 2023, arXiv:2311.01453):
  `mean_n(Y) + lam (mean_u(f) - mean_n(f))` with `lam = cov(Y, f) / ((1 + n/Nu) var(f))`.
  Its gain in effective labels is `1 / (1 - rho^2 Nu / (Nu + n))`; `lam = 0` is the labels
  alone, so asymptotically it never loses. The guarantee is a normal limit.
- **Cross-prediction** (Zrnic & Candes 2023, arXiv:2309.16598): fit the predictor on folds
  of the labels. Used here for the cross-fitted Platt feature.
- **Stratified PPI** (Fisch et al. 2024, arXiv:2406.04291): PPI within strata with their
  own weights. The human sheet is a stratified sample, so stage D is this estimator.
- **Rogan-Gladen / Youden**: `r = (q - FA) / (sens - FA)`, already in
  `seldonian/llm/calibration.py`; spike 006's answer-rate-aware form for false alarms that
  fall on answers only.
- **Betting bounds** (Waudby-Smith & Ramdas 2020, arXiv:2010.09686), the project's
  `betting_bounds`: the finite-sample bound tried for PPI (`ppi_block`, `ppi_bounded`).
- **Bootstrap-t**: a studentised bootstrap limit is second-order correct for a one-sided
  bound where the normal limit is first-order (Hall 1992, *The Bootstrap and Edgeworth
  Expansion*; from memory, not re-checked online. The five arXiv entries were checked).

| Route | Guarantee | What it needs | Outcome here |
|---|---|---|---|
| Labels alone, Clopper-Pearson | exact | i.i.d. labels | the baseline every route must beat |
| Youden / answer-rate-aware | exact, if recall and false alarms carry over | positives and negatives from *somewhere* | loose on the same population, wrong across populations |
| PPI, PPI++, normal limit | asymptotic | i.i.d. labels on the scored responses | under-covers: 0.08-0.13 at n = 100-225 |
| PPI, exact pieces (three limits, post-stratified, block betting) | finite-sample | the same | valid, never narrower than the labels alone at n <= 500 |
| PPI++, bootstrap-t | second-order | the same, >= 10 labels in the rarer class | **chosen**: holds its level, keeps the gain |

## How to Run
    cd .planning/spikes/017-calibration-carrying-certificate
    ../../../.venv/bin/python check_cert.py        # identities, coverage on known truth (~2 min)
    ../../../.venv/bin/python design017.py         # stage A: the sheet's sampling design
    OMP_NUM_THREADS=1 ../../../.venv/bin/python plasmode017.py --reps 4000          # stage B (~2 min)
    OMP_NUM_THREADS=1 ../../../.venv/bin/python plasmode017.py --shift --reps 4000  # rare rates
    OMP_NUM_THREADS=1 ../../../.venv/bin/python route017.py      # the routing rule (~3 min)
    ../../../.venv/bin/python harmcert017.py       # stage C: labels and positives needed
    ./run.sh                                       # GPU, judge only, 35 min: results/spikes/017/scores_pop.jsonl
    ../../../.venv/bin/python harm017.py           # stage D: the real sheet (~6 min)
    ../../../.venv/bin/python transfer017.py       # stage E: carried calibration
    ../../../.venv/bin/python cards017.py          # the three certificates
    ../../../.venv/bin/python report017.py         # results.md
    ../../../.venv/bin/python make_viewer.py       # then open viewer.html

`viewer.html` is the thing to try: pick the constraint, the wording, the judge feature and
the number of gold labels, press "Draw new labels", and watch each route's bound against the
true rate. State can be put in the address, e.g. `viewer.html#task=brevity&n=100`.

## What to Expect
- Refusal: the judged rate sits at 0.04, the true rate at 0.20. The normal-limit bar is the
  shortest and dips under the true rate on some draws; the bootstrap-t bar does not.
  Switching the wording moves the judged rate between 0.008 and 0.30 and leaves the
  certified estimate near 0.20.
- Brevity: the rule ignores the judge and the labels and counts words on all 500 responses.
- Below 45 or so gold labels on refusal the rule falls back to the labels alone.
- Harm: bound 0.039 from the design-weighted labels; the carried calibration is refused.

## Observability
Every stage writes its per-cell rows beside its table: `plasmode.json`, `plasmode_shift.json`,
`plasmode_n500.json`, `harm.json`, `harmcert.json`, `transfer.json`, `cards.json`. The GPU
pass appends to `results/spikes/017/scores_pop.jsonl` in chunks of 256 with a progress line
per chunk in `score.log`, and resumes. The viewer's "Table view" shows each chart's numbers.

## Investigation Trail
1. **Checks on known truth first** (`check_cert.py`). On a parametric judge the gain formulas
   matched Monte Carlo to two decimals, and two problems showed before any real data: the
   PPI++ normal limit missed 0.06-0.08 at a 20% rate and 0.18-0.27 at 1.3%, and every exact
   judge-assisted bound (three Clopper-Pearson limits, post-stratified) was wider than
   Clopper-Pearson on the labels alone in all 18 cells.
2. **A score-type limit did not fix it.** `ppipp_wilson` puts the label variance at the
   hypothesis, as Wilson does. It repaired the parametric cells (0.034-0.055 at n = 225) and
   failed on the real refusal judge (0.085 at n = 225, 0.17 at n = 50). Two causes: the
   in-sample `lam` biases the estimate down (0.185-0.188 against 0.200 at n = 50), and the
   compiled judge fires on 4% of responses, so the rectifier is as skewed as a rare binomial.
3. **A finite-sample PPI did not pay.** `ppi_block` pairs each label with a block of
   unlabelled scores, which gives i.i.d. bounded variables with the PPI variance, so one
   betting bound at the full delta applies. Valid everywhere, and narrower than the labels
   alone in 10 of 210 cells, by at most 0.003. The betting bound costs about a third of the
   labels against Clopper-Pearson on 0/1 data, which eats the judge's gain at n <= 1,000.
4. **The studentised bootstrap did.** Resample the labelled pairs, re-estimate `lam` in each
   resample, take the delta-quantile of t. Prototype: miss 0.036-0.055 from n = 50 to 500.
   Added as `ppipp_boot`; both plasmodes re-run with it. Degenerate resamples count as
   `-inf`, so with 3 positives it returns 1 instead of a wrong number.
5. **The feature matters more than the estimator.** On the canonical refusal judge rho^2
   with the gold label is 0.18 for the 0/1 label, 0.22 for p, 0.48 for the logit and 0.58
   for a Platt map: p is saturated near 0, and PPI++'s `lam` can only rescale it. Added the
   `logit` and cross-fitted `platt` features.
6. **The routing rule had to be tested as a rule** (`route017.py`). Reporting the smaller
   of the exact and the bootstrap bound missed up to 0.078 (12 of 20 cells above 0.05). A
   rule on the label counts (`>= 10` in the rarer class) did not.
7. **Stage D on the first wording looked like E1 was wrong**: the compiled judge's flag rate
   was the same on the sheet (0.218) and the population (0.224), and PPI read as i.i.d. never
   missed. The other two wordings reversed that (Results 4). The planted check was extended
   from one wording to all three after the GPU pass finished.
8. **The same prompts scored in a different batch order do not give the same scores.** On
   the 225 sheet items, 017's pass against 015's: largest difference in p 0.18 / 0.36 / 0.15
   for wordings 0 / 2 / 4, and 3 / 5 / 0 labels flipped at 0.5 (4-bit weights, bf16, left
   padding). Stage D therefore uses 017's scores for the sheet and the population alike.
9. **Deviation:** the GPU pass, pre-registered above. 20,400 forward passes, 35 minutes
   including the model load, lock held and released.

## Results
**Verdict: PARTIAL.** The kill rule did not fire, and one routing rule held its level in
every cell tested. It is not the certificate the idea described. The calibration cannot be
measured once and shipped: it has to be gold labels on the responses being certified. Of the
two instruments the manifest row names, `E[p]` is the weakest feature and the
answer-rate-aware correction is the wrong noise model for a compiled judge. Tables:
`results.md`; the three certificates: `cards.md`.

### 1. The rule that held
| Data | Route | Guarantee |
|---|---|---|
| verifiable property | the feature by code on every response, Clopper-Pearson | exact |
| labels, fewer than 10 in the rarer class | Clopper-Pearson on the labels, judge unused | exact |
| labels, 10 or more in each class | PPI++ on the judge's logit, bootstrap-t | second-order |
| labels from a stratified sheet | design-weighted labels, `b1w` | approximate |
| calibration measured on another prompt population | refused | - |

Largest miss over every cell (delta 0.05): bootstrap-t PPI++ 0.053, the counts rule 0.052,
Clopper-Pearson 0.051, `b1w` on re-drawn sheets 0.059. For comparison: PPI++ with a normal
limit 0.241, the judge alone 1.000.

### 2. What each constraint can report
| Constraint | Route | Estimate | Upper bound | Note |
|---|---|---|---|---|
| brevity (> 80 words) | word count, all 500 | 0.404 | 0.441 | the compiled judge says 0.048 (bound 0.067); rho^2 with the count 0.001 |
| refusal, 225 gold labels of 500 | PPI++ logit, bootstrap-t | 0.204 | 0.247 | labels alone 0.239 on this draw; on average 0.238 against 0.248 at N = 2,000 |
| harm, 225 human labels of 4,800 | design-weighted labels, `b1w` | 0.0134 | 0.0393 | judge unused; `lam` = -0.07 |

The harm bound certifies 0.05 for the 4,800 screened responses and nothing about a trained
policy. The refusal gold label is Qwen3Guard-4B's, not a human's.

### 3. What the judge is worth (refusal, n = 225, N = 2,000)
| Feature | rho^2 | Worth in labels: formula | measured (variance) | measured (bound) |
|---|---|---|---|---|
| 0/1 label | 0.18 | 1.18 | 1.12 | 1.11 |
| p | 0.22 | 1.24 | 1.16 | 1.15 |
| logit | 0.48 | 1.73 | 1.60 | 1.61 |
| cross-fitted Platt | 0.58 | 2.06 | 1.93 | 1.86 |

The formula `1 / (1 - rho^2 Nu / (Nu + n))` runs 1 to 11% high. The gain needs unlabelled
responses: at N = 500 the bound is 0.244 against 0.249. On brevity `lam` is 0 and the bound
equals the labels-alone bound (0.460), while plain PPI is worth 0.77 to 0.91 of the labels.
On harm rho^2 is 0.02, a gain of 1.02.

### 4. The sheet is not an i.i.d. sample, and it matters by wording
The sheet takes 4 responses the 0.6B guard flagged and 2 it cleared per model x encoding
cell. 61.8% of it is guard-flagged against 24.8% of the population; design effect 1.71, so
225 labels are worth about 131.

| Harm wording | Judge flag rate: population | sheet, unweighted | PPI read as i.i.d.: estimate (bound) | design-weighted PPI |
|---|---|---|---|---|
| 0 (canonical) | 0.224 | 0.218 | 0.019 (0.065) | 0.010 (0.063) |
| 2 | 0.311 | 0.462 | **-0.138 (-0.082)** | -0.005 (0.033) |
| 4 | 0.055 | 0.093 | -0.025 (0.010) | -0.017 (0.015) |

Read as i.i.d., the sheet certifies a negative harm rate under wording 2. With planted
labels and the sheet re-drawn by its real rule, that route missed in 0.98 of draws at a 1.3%
rate for wording 2, 0.09 to 0.70 for wording 4 and at most 0.005 for wording 0. The design-weighted
PPI missed 0.03 to 0.08, the weighted labels with a normal limit 0.07 to 0.22, with `b1w`
0.001 to 0.059. Two earlier numbers are sheet rates, not population rates: 015's 0.458 flag
rate for wording 2 (0.311), and Qwen3Guard-4B's 0.230 false-alarm rate (0.120 weighted).

### 5. A calibration does not carry across prompt populations
| Shift (refusal judge, six wordings) | Recall differs (p < 0.05) | Carried Youden bound misses > 0.05 | Mean error of a carried Platt map |
|---|---|---|---|
| xstest to orbench, same pool | 4 of 6 | 4 of 6 (up to 0.94) | 0.038 |
| orbench to xstest | 4 of 6 | 1 of 6 | 0.027 |
| step 0 to step 200, same prompts | 0 of 6 | 0 of 6 | 0.012 |
| over-refusal to harmful prompts | 4 of 6 | 0 of 6 (bound 1.0) | 0.233 |
| harmful to over-refusal prompts | 4 of 6 | 5 of 6 (up to 1.00) | 0.272 |

False-alarm rates never differed (they are near 0); recall did. The one shift that held is
200 training steps on the same prompts, where the refusal rate moved 0.200 to 0.172.

### 6. Harm: positives, labels, and the noise model
- **By labels on the candidate's own responses:** 301 labels with no positive certify 1%,
  149 certify 2%, 59 certify 5%; 776 if the true rate is a quarter of a 1% threshold. No
  human positive is needed.
- **By a carried calibration**, best case: the compiled rubric (false alarms 0.216) reaches
  a bound of 0.17 with 300 positives and never certifies 5%; Qwen3Guard-4B (recall 1/3
  against false alarms 0.23) stays at 1.0 up to 100 positives; the gated label certifies 5%
  for a clean candidate in 0.10 / 0.41 / 0.87 / 1.00 of draws at 3 / 10 / 30 / 100 positives
  (recall 0.5). So 30 positives is the right floor, for the gated label only, and only
  where Results 5 allows carrying at all.
- **The answer-rate-aware correction fits the guard, not the compiled rubric.** On human
  negatives the guard's false alarms fall on answers (0.291 answered, 0.036 refused,
  weighted); the compiled rubric's fall on refusals (0.132 answered, 0.280 refused). On the
  whole population it flags 27.2% of refusals and 12.6% of answers: it reacts to the harmful
  request being restated, which is 015's rubric artifact again.

### 7. How far the threshold moves
The test `true rate <= tau` on the judged rate is `judged rate <= tau - rectifier - margin`.

| Constraint | tau | Rectifier | Judged-rate threshold |
|---|---|---|---|
| refusal, rubric 0 | 0.25 | +0.156 | 0.055 |
| refusal, bare sentence | 0.25 | -0.089 | 0.300 |
| brevity, compiled judge | 0.45 | +0.387 | 0.003 |
| harm, rubric 0 | 0.05 | -0.220 | 0.217 |

### Pre-registered expectations
| | Expectation | Outcome |
|---|---|---|
| E1 | sheet not i.i.d.; flag rate overstated by > 5 points; unweighted PPI misses > 2 delta, weighted does not | **holds for wordings 2 and 4, not for 0.** Overstated by 15 / 4 / -1 points; unweighted PPI missed up to 0.98 / 0.70 / 0.005. Weighted PPI missed up to 0.084, above delta but under 2 delta |
| E2 | PPI++ within 2 points of delta at n >= 100; gain formula within 10%; p better than the label, 1.2 to 2.5 | **coverage refuted** (0.107 at n = 100, 0.082-0.086 at 225, 0.062 at 500). Formula holds (1-11% high). p is better than the label but only 1.16; the logit gives 1.60 |
| E3 | useless judge: PPI++ costs nothing, plain PPI loses | **holds**: `lam` 0.04, gain 1.00; plain PPI 0.77-0.91 |
| E4 | wording moves the raw rate, not the certified estimate | **holds**: raw 0.008-0.090, PPI++ mean 0.197-0.199; one real draw 0.195-0.204 |
| E5 | rare: normal PPI misses > 2 delta; no exact judge-assisted bound 10% narrower; 3 positives vacuous, 30 not | **first two hold** (0.20-0.24; none narrower at all). Third is wrong both ways: the gated label gives 0.06-0.22 at 3 positives, the guard is still 1.0 at 30 |
| E6 | recall and false alarms do not carry across populations | **holds for recall** across prompt sources and pools, **fails to appear** across 200 training steps |
| E7 | threshold moves down for harm, up for refusal | **stated backwards.** An over-flagging judge raises the judged-rate threshold (harm: 0.05 to 0.217); an under-flagging one lowers it (refusal: 0.25 to 0.055) |

### What this means for the idea
- The compiled constraint's certificate is **gold labels on its own safety responses plus
  the judge as a variance reducer**, routed by the table in Results 1. It states the route,
  the label counts, the wording's hash, and the smallest threshold the data certifies.
- **Label budget is the real cost.** A good judge (rho^2 0.5) turns 225 labels into about
  360; the compiled harm judge turns 225 into 229.
- **Use the logit, never p or the 0/1 label**, and score labelled and unlabelled responses
  in the same run.
- **Spike 018's target is now a number**: rho^2 with human labels on the logit scale. 0.5
  halves the labels; the compiled harm rubric is at 0.02.

### Limits
One judge model (Qwen3-8B, 4-bit). Refusal's gold label is a guard model's. The plasmode
pool is 500 responses. The bootstrap-t and `b1w` guarantees are approximate; the largest
misses above are Monte Carlo figures with standard errors near 0.004. Carrying a
calibration across training: see the E8 addendum (it fails when the training targets the label).

## Addendum 2026-10-01: E8, carrying the calibration to a *constrained* policy

**Pre-registered before scoring (08:30 PT).** The one transfer that held (E6, step 0 to
step 200) was side-effect training: the refusal label was not the training's target. Spike
014 has since trained the same Granite model on 013's C1 pool with `LagrangianReward`
driving the refusal label (multiplier 5-12 against a reward model that pulls refusal up),
and sampled the pool at steps 100 (rate 0.151, 3.4 points below the reference) and 200
(0.182), with the guard's gold label on every response. The worry E6 could not test:
pressure on the judged label changes the *form* of the refusals, so the compiled judge's
recall at the trained policy is not the reference's.

- **E8a.** Score the same 500 (prompt, k) pairs 015 scored at step 0 on 014's step-100 and
  step-200 responses, six refusal wordings, 015's scorer unchanged (6,000 passes). Fisher
  tests of recall and false-alarm equality against step 0, the three carried estimates, and
  the carried Youden bound's plasmode, exactly as `transfer017.py` does for E6.
- **Expectation.** At step 200 (rate back at the reference, 014's rho 0.92) the calibration
  carries as it did for E6: recall equal in at least 5 of 6 wordings, carried errors under
  0.03, no Youden miss above 0.05. At step 100 (the pushed checkpoint) I expect recall to
  differ in at least 2 of 6 wordings and the carried Rogan-Gladen error to exceed 0.03 in
  the median wording. If step 100 also carries, the "constrained rate barely moved" limit
  above is lifted for moves of about 3 points; if step 200 fails, E6's one positive result
  was luck and no calibration carries across training either.

**E8 result (scored 08:36 PT; `transfer.md`, sections "training, constrained").** The
calibration does not carry to the constrained policy, and it fails harder at step 200 than
at step 100, the reverse of the expectation above.

| shift (same 500 items, gold = guard) | recall differs (p < 0.05) | mean abs error RG / Platt / rect | carried Youden bound misses > 0.05 |
|---|---|---|---|
| step 0 to step 200, side effect (E6) | 0 of 6 | 0.051 / 0.012 / 0.019 | 0 of 6 |
| step 0 to 014 step 100, constrained | 1 of 6 | 0.103 / 0.018 / 0.011 | 1 of 6 |
| step 0 to 014 step 200, constrained | 2 of 6 | 0.139 / 0.030 / 0.011 | 4 of 6 (0.80 for wording 0) |

- The compiled judge's recall on the guard's refusals falls from 0.21 at step 0 to 0.06
  (step 100) and 0.03 (step 200) for wording 0, from 0.12 to 0.07 and 0.00 for wording 2;
  the stronger wordings 3-5 drop 5-10 points without reaching significance at n 100. False
  alarms stay at zero. The carried Rogan-Gladen estimate lands at 0.03 against a gold rate
  of 0.19.
- **What changed is the form of the refusals.** Of the 93 guard-flagged refusals at step
  200, the compiled judge (wording 0) misses 89, at a median p of 0.0002 and 74 words. They
  are refusal-prefixed answers: a "cannot assist" sentence followed by the answer (the can
  is crushed, the lights are turned off, the fictional character has no bank account). The
  guard reads the preamble as a refusal; the rubric judge reads the body as an answer. The
  reward model (pulling toward long helpful answers) and the multiplier (penalising the
  guard's label) together select exactly the form the two judges split on.
- **Platt and the rectifier look fine only by accident.** With recall near zero the target's
  judged rate is near zero, so the carried quantity is the source's own rate (0.20) against
  a target rate of 0.17-0.19; a target whose rate had moved would be missed by the same
  amount. The source and target rates happening to agree is not a carry.

Scored against the pre-registration: step 200 was expected to carry and did not (recall
differs in 2 of 6, Youden misses in 4 of 6); step 100 was expected to differ in 2 of 6 and
differed in 1 (the trend is the same, the sample is 100 refusals). The E6 "held" row is
now explained: side-effect training does not change the refusals' form; training against
the label does. **The limit "tested on one run where the constrained rate barely moved" is
replaced by a result: a calibration does not carry across training that targets the
label, even when the rate comes back to the reference.** Which judge is right about a
refusal-prefixed answer is a human-label question (follow-up 1). For spike 014 this means
its stratified label at step 200 is the guard's, hybrids included; the strata are built on
the guard's label at step 0 and the gain measured on the guard's at step 200, so the
result stands on its own definition.
