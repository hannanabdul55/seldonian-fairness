---
spike: 013
idea: rerandomized-split
name: stratified-safety-set
type: standard
validates: "Given an LLM Seldonian safety test on a per-response 0/1 constraint, when the safety set is sampled within strata of the reference model's per-prompt rate and scored with a stratified bound, then the test stays valid and needs fewer safety prompts for the same power, in settings the pre-flight number G predicts"
verdict: VALIDATED
related: [012, 009, 005, 007]
tags: [stratification, safety-set, pre-flight, plasmode, bounds, gpu, cpu]
---

# Spike 013: Reference-rate stratified safety sets

The design, hypotheses and go/stop rule were fixed before any data: `DESIGN.md`. This
README records what each stage found. All four stages ran. **Verdict: go, in a narrowed
scope** (Results, below).

## How to Run
    cd .planning/spikes/013-stratified-safety-set
    ../../../.venv/bin/python check_bounds.py          # B2 constructions: coverage, width
    ../../../.venv/bin/python check_b1.py              # approximate bounds: coverage grid
    ../../../.venv/bin/python inloop.py --seeds 500 --out inloop.json      # stage 0a (~6 min)
    ../../../.venv/bin/python summarise_inloop.py inloop.json
    ../../../.venv/bin/python bandit_plasmode.py --seeds 10 --reps 1000 --out bandit_plasmode.json
    ../../../.venv/bin/python summarise_plasmode.py bandit_plasmode.json
    ./run.sh --stage pilot --n 100 && ./run.sh --stage judge --pilot       # stage 1 (GPU)
    ./run.sh --stage generate --n 500 --cov 8 --cand 16 && ./run.sh --stage judge   # stage 2
    ../../../.venv/bin/python real_plasmode.py --reps 5000 --out real_plasmode.json  # stage 3

## Research: the bounds
- **B1 (approximate).** A stratified Wald-t (`b1`) and a stratified Wilson-type score bound
  (`b1w`, variance at the hypothesis, every stratum shifted equally). `b1w` is the primary:
  at one stratum it is the Wilson bound.
- **B2 (distribution-free).** A union-intersection betting test after Spertus & Stark
  (2022, "Sweeter than SUITE", arXiv:2207.03379) and Spertus, Sridhar & Stark (2024,
  "Sequential stratified inference for the mean", arXiv:2409.06680). Three constructions
  were tried, all valid by construction and in simulation:
  1. a common bet mixture;
  2. per-stratum bets fixed by the hypothesised mean;
  3. predictable (aGRAPA-style) bets, interleaved across strata with shrinkage to the
     pooled running mean.
- **Rejected by construction:** empirical-Bernstein (its constant makes it about 1.5x wider
  than Clopper-Pearson, a 2.3x ESS penalty) and Hoeffding (range-only, so stratification
  cannot help).

## Investigation Trail
1. **B2 is valid but gives no gain** (`check_bounds.md`). All three constructions miss 0 times.
   The stratified version was never narrower than its own pooled version (ESS 0.35-1.0).
   The one exception is a stratum near 0 (1.29). The composite null has to be rejected
   for every way the mean could be distributed across strata, and at n_s <= 400 that price
   cancels the variance saved. **Recorded negative: no distribution-free stratified bound
   here beats pooling.** Under DESIGN.md section 4, a go now needs B1 with its coverage
   holding.
2. **B1's gain matches theory** (`check_b1.md`, 4000 reps a cell). With strata that
   differ, the ESS gain is 1.55-1.67; with flat strata it is 1.0-1.09. The Wald version
   undercovers at low rates (0.154 at delta 0.1), exactly as the pooled Wald does. The
   Wilson-type `b1w` fixes most of that. Its coverage sits at delta in spread, flat and mid
   profiles, equal to its pooled version everywhere. In rare-rate profiles (0-7%), pooled
   and stratified alike overshoot in some cells (up to 0.122 at delta 0.05). That is
   binomial discreteness, and it matches the use case the design already excludes.
3. **The bandit's heterogeneity was capped** at ICC 0.22: the uniform reference averages
   four independent per-action violation curves. `heteroenv.py` adds a violation direction
   shared across actions, which reaches ICC 0.05-0.75 and covers the real range.
4. **In-loop (stage 0a, `inloop.md`)**: the real `SeldonianLLMPolicy`, 500 runs per cell.
   - Validity: `b1w` stratified misses 0.084-0.116 at delta 0.1, the same as the random
     rule's pooled bounds and the project's t-test.
   - ESS against a random split with a pooled bound:
     - pool target: 1.03 / 1.16 / 1.71 / 2.80, from the lowest to the highest ICC;
     - population target: 1.02 / 1.09 / 1.32 / 1.61.
   - The solution rate rose from 0.51 to 0.71 (population target) or 0.78 (pool target) at
     the highest ICC.
   - Placebo and metadata strata: exactly 1.00.
5. **The two-phase cap.** With the pool itself a sample from the population, the gain
   becomes `1 / (1 - G + G n_s / N)`. This matched the population-target ESS to 0.01
   (`n_s / N = 0.4` here). The gain is capped whenever the safety set is a large fraction
   of the prompt pool.
6. **H3 and H4 pass on the bandit** (stage 0b, `plasmode_bandit.md`: 288 cells, 1000
   resamples each, trained and reference candidates).
   - Realised vs predicted ESS: median absolute error 0.021, 90th percentile 0.176,
     Spearman 0.988.
   - The gain grows with k: 1.09 at k = 1 and 1.21 at k = 8 (ICC 0.26, H = 4).
   - The gain grows with H: 1.57 at H = 2, 1.93 at H = 4, 2.04 at H = 8 (ICC 0.55).
   - At the highest ICC the method reaches 3.02 against the oracle's 3.32.
   - Stratified misses 0.073-0.086 at delta 0.1.
7. **Stage 1 pilot** (Granite-3.3-2B reference, 100 prompts x 8 samples per pool,
   `pilot_icc.md`).
   - Throughput: 5.8 generations/s at batch 128. Batch 256 spills past the 12 GB card
     (0.5/s).
   - Judge (Qwen3Guard-4B, 4-bit): 7.1 responses/s.
   - ICC_ref:
     - C1 refusal 0.73;
     - C2 non-refusal 0.44, refusal 0.41, gated harm 0.33;
     - C3 refusal 0.91.
   - Harm rates are rare, as predicted: C3 1.6%, C2 gated 2.6%.
   - Encoding (C2's metadata) explains 13-25% of the variance.
   - **Stop rule passed.** C3 refusal was added as a label.
   - The pilot's C1 is XSTest only (100 < 250); the full pool mixes XSTest and OR-Bench.
   - A missing Triton guard crashed the first judging run; it was fixed in `gen013.py`.
8. **Stage 2 sizing.** Measured rates put the design's 170k generations at about 20 h. The
   run is 500 prompts per pool, 8 covariate + 16 candidate samples, at steps 0, 100 and
   200: 84k generations, about 9 h with training and judging, inside the 12 h cap.
   Safety-set sizes are 100 and 200; 400 would be 80% of the pool.

9. **Stage 2** ran 12.5 h (33.4k s generation and training, 11.5k s judging): **0.5 h over
   the 12 h cap**. The encoded pool generated at 2.3/s in the full run, against 5.2/s in the
   pilot. Training worked: capital exact-match reward went from 0.06 to 0.84. Per-prompt safety
   behaviour barely moved (rho 0.80-1.00, `preflight_real.md`), as expected for side-effect
   labels.
10. **Stage 3: a flaw in the pre-registered coverage truth.** The draws come from each
    candidate's evaluation half, but the design compared bounds with the *truth* half's pool
    mean. With 8 samples a prompt, that target is off by a fixed +-0.004, which every
    replicate shares, and it inflated misses. The worst hit was the narrowest arm: S2 on C1 at
    n_s 200 missed 0.20. Judged against the evaluation half's own mean (what the draws come
    from), the same cells are 0.054. Both tables are in `validity_real.md`; the corrected one
    is the valid measure.
11. **Stage 3: the strata rule matters with tied covariates.** 73-97% of prompts have an
    8-sample reference rate of exactly 0 or 1. The design's rule (ties kept, quantile cuts,
    strata merged below 20 expected safety prompts) collapses to 1-2 strata, and often to
    ESS 1.00. Random tie-breaking into equal rank strata (what stage 0 used) and value
    strata (exploratory) were run beside it (`ess_real.md`). **Equal rank strata with random
    ties at H = 8 are best and most consistent**, and their coverage holds (`validity_H8.md`).
12. **H3 fails as pre-registered** (`plasmode_real.md`). The median |realised - predicted|
    ESS is 0.29 (pre-registered <= 0.1), the 90th percentile 0.53. Spearman is 0.83, so the
    ordering half passes. G over-predicts because coarse strata over a tied covariate keep
    less than `c_H`. At H = 8 the gap is small: C1 2.40 against 2.89, C3 refusal 5.13-5.33
    against 5.43, C2 non-refusal 1.42 against 1.48. The half-swap check agrees
    (`plasmode_real_swap.md`).

## Results
**Verdict: VALIDATED, as a go in a narrowed scope.** It passes the DESIGN.md section 4 rule
through its B1 clause: no distribution-free stratified bound helped (step 1), so the go
rests on the Wilson-type `b1w`, whose coverage holds in the cells that show the gain.

| hypothesis | result |
|---|---|
| H1 validity | **holds at mid rates.** S2 (8 strata, k 8) misses <= 0.093 at delta 0.1 and <= 0.047 at 0.05 on every label with rate 9-94%, beside R's 0.066-0.114. **Fails at rare rates for every arm, the random-split baseline included**: C3 harm (1%) misses 0.44 at n_s 100, C2 gated harm (2%) 0.24. That is the approximate bound, not the strata. |
| H2 where it works | **holds.** ESS vs a random split (step 200, n_s 100/200, H 8): C1 over-refusal 2.40/2.42; C3 refusal 5.13/5.33; C2 encoded non-refusal 1.42/1.43. Pass rate at tau = truth + 0.02 (random split -> S2 H 8, n_s 100 / 200): C1 0.19 -> 0.26 / 0.23 -> 0.41; C3 refusal 0.18 -> 0.35 / 0.20 -> 0.56; C2 non-refusal 0.27 -> 0.26 (a small *loss* at n_s 100, despite narrower mean width) / 0.29 -> 0.38. Null cells: C3 harm 0.97-1.03, placebo 0.98-1.03, C2 gated 1.00-1.07. |
| H3 predictability | **fails in absolute terms** (median error 0.29), **passes in ranking** (Spearman 0.83); over-prediction is 0-20% at H = 8 and more at H = 2/4. On the bandit, with no ties, it passed (0.02, 0.99). |
| H4 covariate precision | **holds.** ESS grows with k (C1, H 8: 1.61 at k 1 -> 2.40 at k 8; C3 refusal 2.92 -> 5.13). k = 1 captures under half of the k = 8 gain. |

### Use-case map (real Granite-3.3-2B data, 500-prompt pools)

| condition | label, rate | ICC_ref | realised ESS (H 8, k 8) | valid? | verdict |
|---|---|---|---|---|---|
| C1 over-refusal (XSTest + OR-Bench) | refusal, 17% | 0.72 | 2.4 | yes | **use it** |
| C3 plain harmful requests | refusal, 66% | 0.86 | 5.1-5.3 | yes | **use it** |
| C2 encoded harmful requests | non-refusal, 9% | 0.50 | 1.4 | yes | use it; encoding strata alone give 1.2 |
| C2 encoded harmful requests | refusal, 94% | 0.47 | 1.05-1.18 | yes | not worth it (rate near 1, few non-refusing prompts) |
| C2 encoded, gated harm | 2% | 0.50 | 1.0-1.07 | **no, for any design** | don't; rare-rate labels need an exact bound, and exact bounds gain nothing from strata |
| C3 plain harm | 1% | 0.33 | 1.0 | **no, for any design** | don't (same reason) |
| placebo covariate | - | - | 1.0 | yes | the negative control behaves |

**Where to use it:**
- per-response 0/1 labels with rates of roughly 5-90%, high prompt-level ICC, and a label
  the training does not target directly (rho stayed 0.8-1.0 here);
- with an approximate bound (`b1w`);
- 8 equal rank strata of an 8-sample reference rate, random tie-breaking;
- a prompt pool much larger than the safety set, or the two-phase cap
  `1 / (1 - G + G n_s / N)` eats the gain.

**Where not to use it:**
- rare labels (the approximate bound is invalid there, stratified or not);
- labels near 0 or 1;
- when an exact small-sample guarantee is required (no distribution-free stratified bound
  here beat pooling);
- when labels are cheap (enlarge D_s instead).

**Not tested:** a label that training pushes directly (the design's optional C4; rho may
decay); the Qwen3-1.7B replication (the stage-2 overrun left no budget); a population
target on real data (the plasmode certifies the pool).

**Pre-flight:** `preflight.py` computes G from k reference samples and gives the predicted
gain for rho 1 and 0.8, with the two-phase cap and the verdict above. Read the prediction as
optimistic by up to about 20% at H = 8, and trust its ranking.

**Deviations from DESIGN.md, all reported:**
- the stage-2 GPU budget was overrun (12.5 h against 12 h);
- the coverage truth was redefined (step 10);
- the strata rule was replaced (step 11);
- n_s 400 was dropped (N = 500);
- the Qwen3-1.7B and C4 stages were not run.
