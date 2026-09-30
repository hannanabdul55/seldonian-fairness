# Spike 013 design: reference-rate stratified safety sets

Status: **DRAFT for review, nothing run.** Follows spike 012 (rerandomised splits: valid,
power only from outcome-like covariates) and its follow-up (real LLM rates: ICC 0.2-0.7,
reference rates persist about 0.6 through GRPO).

## 1. The one question

> For an LLM Seldonian safety test on a per-response 0/1 constraint, does building the
> safety set by stratified sampling on the *reference model's per-prompt rate*, paired with
> a stratified bound, keep the test valid and cut the safety-set size needed for a given
> power, and in which constraint/pool settings?

Everything else is held fixed: the candidate policy, the judge, delta, one response per
safety prompt, proportional allocation.

**Not tested here** (each needs its own spike): rerandomisation (settled in 012), Neyman
allocation, several responses per safety prompt, stratifying on anything computed from the
candidate, shift between the pool and deployment, tabular classifiers (012 part A: design
covariates buy nothing there, and per-row indicators carry no sampling noise to remove).

## 2. The method, exactly

Given a prompt pool `P` (N prompts) and a reference policy `pi_ref`, before any training:

1. **Covariate.** Sample `k` responses per prompt from `pi_ref` and judge them:
   `r_ref(x)` = the fraction flagged. These samples are never reused for any test.
2. **Strata.** Cut `r_ref` into `H` strata at its quantiles (ties: keep all the prompts
   with the same `r_ref` value in one stratum; merge any stratum below `n_min` expected
   safety prompts into its neighbour).
3. **Split.** Within each stratum, send a random fraction `f` of prompts to `D_s`
   (proportional allocation); the rest go to `D_c`. The rule depends only on `r_ref`, so
   `D_s` is a stratified random sample and is independent of training given the strata.
4. **Safety test.** At the candidate, one response per `D_s` prompt, judged:
   `mu_st = sum_h W_h ybar_h`, with `W_h` = stratum share of the target distribution, and an
   upper bound that uses the within-stratum variances (section 5). Pass if `UB <= tau`.

`W_h` from the pool is exact when the target *is* the pool. When the pool is itself a
sample of the deployment distribution, this is two-phase sampling for stratification
(Cochran, *Sampling Techniques*, 1977, ch. 12). The bound must then add the term for the
pool's own composition, `sum_h W_h (mu_h - mu)^2 / N`. It is small when `N >> n_s`, but it
is not zero.

## 3. Why it can help, and the pre-flight number

For one response per prompt, the variance of a response label is
`p(1 - p) = within-prompt + between-prompt`, with `ICC = between / total`. Stratification
removes the part of the *candidate's* between-prompt variance that the strata explain:

    G  ~=  ICC_cand  x  rho^2  x  rel(k)  x  c_H
    effective-sample-size gain  ESS = 1 / (1 - G)

- `rho = corr(p_ref(x), p_cand(x))`: persistence of per-prompt rates through training.
- `rel(k) = k ICC_ref / (1 + (k - 1) ICC_ref)`: reliability of a `k`-sample covariate.
- `c_H`: the share a coarse `H`-stratum cut keeps. For a normal covariate this is 2/pi = 0.64
  at H = 2, and about 0.88 at H = 4 and 0.96 at H = 8. Stage 0 measures it for skewed rates.

Worked from the data already on disk (Granite-3.3-2B refusal ICC 0.45, Qwen3-1.7B 0.64;
`rho^2` 0.36-0.8 from 009; `k = 8`; `H = 4`):

| model / label | ICC | rel(8) | G | ESS gain |
|---|---|---|---|---|
| Qwen3-1.7B refusal | 0.64 | 0.93 | 0.19-0.42 | 1.23-1.72 |
| Granite-2B refusal | 0.45 | 0.87 | 0.12-0.28 | 1.14-1.39 |
| any label, ICC 0.05 (bandit default) | 0.05 | 0.30 | < 0.02 | ~1.0 |

**The experiment's main output is whether `G` predicts the realised gain.** If it does,
anyone can decide *before training*, from reference samples plus an assumed `rho`, whether
stratification is worth it.

## 4. Hypotheses and decision rule (fixed before any data)

- **H1, validity.** In every cell, the stratified arms' coverage failure `P(true > UB)` is at
  most `delta + 2 MC se`. For the approximate bound (B1) this holds only where every
  stratum has `n_h >= 20`; cells below that are reported but don't count.
- **H2, where it works.** ESS gain >= 1.2 in at least one *real* pool/label predicted to
  work (C1 or C2), with `k = 8`. Gain <= 1.05 in the predicted-null cells (C3, placebo).
- **H3, predictability.** Across all cells, realised ESS gain vs `1 / (1 - G)`: absolute
  error <= 0.1 in the median cell, and the ordering of cells preserved (Spearman >= 0.8).
- **H4, covariate precision.** Gain rises with `k`; at `k = 1-2` it is under half the `k = 8`
  gain.

**Go** (build it into `seldonian/llm/` as an option with a pre-flight check): H1 holds, and
H2 holds with the distribution-free bound B2. It is also a go with B1 alone if B1's coverage
holds in those cells. **Stop** (record the negative and close the idea) if H2 fails in both
C1 and C2, or H1 fails for B2.

## 5. Arms and bounds (these and no others)

| arm | split | bound | role |
|---|---|---|---|
| R | simple random | project `ttest` and `betting_mixture` on the pooled mean | baseline |
| S1 | stratified on `r_ref` | same pooled bounds as R | composition effect only (012 bandit's mechanism) |
| **S2** | stratified on `r_ref` | stratified B1 and B2 | **the method** |
| M | stratified on metadata cells (source/encoding) | stratified B1 and B2 | the cheap alternative (metadata explained < 15%) |
| P | stratified on a *permuted* `r_ref` | stratified B1 and B2 | placebo: must give no gain and stay valid |
| O | stratified on the candidate's own `p_cand(x)` | stratified B1 and B2 | oracle ceiling (not implementable) |

Factors: `k` in {1, 2, 4, 8}, `H` in {2, 4, 8}, `n_s` in {100, 200, 400}, delta in {0.05, 0.1}.

- **B1, stratified t:** `mu_st + t_{1-delta, df} * sqrt(sum_h W_h^2 s_h^2 / n_h)`, with
  Satterthwaite df. It is approximate, like the project's `ttest`.
- **B2, distribution-free stratified bound:** the stage-0 research picks it. Candidates are
  a stratified betting/supermartingale test (the risk-limiting-audit literature, e.g.
  Spertus & Stark on stratified union-intersection tests) and an empirical-Bernstein bound
  with per-stratum variance bounds plus a union over strata. **Citations are to be verified
  before use.** A range-only bound (Hoeffding) is excluded by construction: its width
  ignores variance, so stratification gains nothing with it (proportional allocation gives
  the same width).

## 6. Data setup

### Stage 0: synthetic, in the loop (CPU, about 1 h)
The 012 bandit with the real `SeldonianLLMPolicy`. It is the only stage that retrains on
`D_c`, so it checks that decoupling training from the split (stage 3) loses nothing. Grid:
`u_scale` giving ICC in {0.05, 0.2, 0.4, 0.6} x training strength (steps, lr) giving `rho`
from about 0.4 to 1 x `k` x `H`, 1000 seeds a cell. It also checks the two-phase term: pool
drawn from a 20k population, target = population. Outputs: validity of B1/B2, `c_H` for
skewed rates, a first test of H3. **Gate:** B2 valid and H3 within +-0.1, or stop here.

### Stages 1-3: real models and prompts (plasmode resampling)
Real generations and real judge labels, a known finite-pool truth, and many simulated
safety sets drawn from them. The candidates are real trained checkpoints; only the draw of
`D_s` is simulated.

**Pools** (`N = 1000` each, fixed after the stage-1 pilot):

| cond. | pool | label | predicted | why |
|---|---|---|---|---|
| C1 | over-refusal: XSTest safe + OR-Bench-80K (`load_benign_prompts`) | refusal (Qwen3Guard refusal field) | **works** | borderline prompts: high ICC, mid rates |
| C2 | PKU prompts in the 005/009 encodings | non-refusal on encoded (`unsafe`), and the gated two-stage harm label (sim >= 0.8 AND 4B judge, 007) | **works** for non-refusal, unsure for gated | side-effect label not trained directly: `rho` about 0.6 in 009 |
| C3 | plain PKU prompts | gated harm | **null** | rates 0-3%: little variance to remove |
| C4 (optional) | C1 pool | refusal, under a Lagrangian run constraining it | **shrinks** | training pushes the label: rates compress, `rho` decays |
| Placebo | C1 | refusal | **null** | arm P |

**Model and checkpoints.** Granite-3.3-2B (the 009 pipeline fits 12 GB beside the 4B judge).
Candidates: the reference itself (`rho = 1`, positive control) and 009-recipe GRPO
checkpoints at steps 100 and 200. 009 kept none, so this needs one training run with
checkpoint saving, about 75 min at 009's 22 s/step. Qwen3-1.7B, highest ICC, is added for
C1 only if Granite C1 is positive.

**Generations per prompt:** `k_max = 8` covariate samples from the reference, plus `K = 16`
per candidate. Each candidate's K responses split 8 / 8 into a *truth* half (per-prompt and
pool truth) and an *evaluation* half (drawn for `D_s`). The reference as candidate gets its
own 16, separate from its 8 covariate samples. Total about 3 pools x 1000 x (8 + 3 x 16) =
170k generations plus judging. The stage-1 pilot measures throughput and shrinks this to fit
a cap of **12 GPU-hours** (fallback: N = 600, K = 12).

**Resampling (stage 3, CPU).** For each pool x label x candidate x arm x factor cell, draw
5000 safety sets of `n_s` prompts, one evaluation response each, and compute each bound.
Arms use the same random streams (paired). The truth is the pool mean of per-prompt
truth-half rates, with sd about 0.004 at N = 1000 x 8. The truth and evaluation halves are
swapped as a sensitivity check.

## 7. Measurements

- **Validity:** coverage failure `P(truth > UB)`; miss rate `P(pass and truth > tau)` at
  `tau = truth + Delta`, Delta in {-0.02, 0}.
- **Power:** mean bound width, ESS gain = (width_R / width_arm)^2 at equal `n_s`, and the
  pass-rate curve over Delta in {0.01, 0.02, 0.04, 0.06}.
- **Moderators, measured per cell:** ICC_ref, ICC_cand, rho (from the truth halves), rel(k),
  realised composition-error sd, smallest stratum size, and `G` against the realised gain (H3).
- **MC precision:** 5000 draws give coverage se about 0.004 at 0.1. The primary cells are
  the ones named in section 4; everything else is exploratory.

## 8. Use cases, narrowed (prior; the experiment confirms or strikes each)

**Where it should help.** All of these must hold: `G >= 0.17` (ESS >= 1.2) and a
variance-adaptive bound.
1. **Refusal / over-refusal constraints on mixed pools** (benign prompts that sound
   harmful). High ICC, mid-range rates, and the label usually isn't the training target.
2. **Side-effect constraints on held-out prompt pools**, e.g. the forbidden-capability
   setting: engagement with encoded harmful prompts while training something else. Rates
   persist (009: about 0.6).
3. **Mixed-source red-team pools** (many attack types or encodings), where success rates
   per prompt differ by orders of magnitude, but only when the per-prompt reference rate is
   used. Metadata strata alone capture little (< 15%).
4. **Expensive safety labels** (human, or a large judge), where `n_s` is small and a 20-40%
   ESS gain is worth `8 x N` cheap reference samples.

**Where it should not help (don't use it):**
- rare-event harm (rates below about 3%): there is little absolute variance, and
  distribution-free bounds are dominated by rarity;
- a constraint the training pushes hard (C4 tests this): rates compress, persistence decays;
- homogeneous pools (single template or source): low ICC;
- range-only bounds (Hoeffding): zero gain by construction;
- cheap labels: a bigger `D_s` is simpler;
- small pools, where strata fall below `n_min` (about 20 safety prompts each);
- tabular fairness constraints (out of scope, see section 1).

## 9. Order, budget, stop points

| stage | what | cost | stop if |
|---|---|---|---|
| 0 | bandit grid + bound research (B2) | CPU, about 1-2 h | no valid B2, or H3 off by > 0.1 |
| 1 | pilot: 100 prompts per pool, reference only | GPU about 1 h | ICC_ref < 0.15 in both C1 and C2 |
| 2 | 009-recipe training with checkpoints + full generation and judging | GPU <= 12 h, overnight, `flock` | - |
| 3 | resampling analysis, use-case table, go/no-go | CPU about 1 h | - |
| 4 (opt.) | C4 Lagrangian run, Qwen3-1.7B on C1 | GPU about 6 h | only if stage 3 is a go |

GPU stages follow the conventions: `run.sh` with a free-disk check,
`flock /tmp/claude-gpu.lock`, TRL scratch in `/mnt/d/seldonian-runs/013`, per-response rows
in `results/spikes/013/` (xz if large). No weekday 9-5 PT commits.

## 10. Threats to validity

- **Judge ≠ harm.** Encoded "unsafe" measures non-refusal (005). Prompt-specific judge
  errors raise the ICC, so a gain is real *for the judged constraint*, not evidence about
  harm. C2 reports the gated label separately.
- **Training decoupled from the split** (stage 3 draws `D_s` after training on a separate
  set). Stage 0 checks this in the loop, and 012 found no leak even from adversarial balance.
- **Truth noise:** truth-half sd about 0.004 against bound widths of about 0.04-0.08; the
  half-swap sensitivity check covers it.
- **Pool ≠ deployment:** the certificate is for the pool's distribution (plus the
  two-phase term). Shift is out of scope.
- **One model family in stages 1-3:** C1 on Qwen3-1.7B is the planned replication.

## 11. Deliverables

`013-stratified-safety-set/`: README (verdict), `results.md` with the arm x cell tables, a
**use-case map** (each condition: `G`, realised ESS gain, validity, verdict), and a
`preflight.py` that computes `G` from reference samples for a new pool, which would become
the library's pre-flight check on a go.
