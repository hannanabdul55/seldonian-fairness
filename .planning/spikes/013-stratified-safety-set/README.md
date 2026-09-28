---
spike: 013
idea: rerandomized-split
name: stratified-safety-set
type: standard
validates: "Given an LLM Seldonian safety test on a per-response 0/1 constraint, when the safety set is sampled within strata of the reference model's per-prompt rate and scored with a stratified bound, then the test stays valid and needs fewer safety prompts for the same power, in settings the pre-flight number G predicts"
verdict: PENDING
related: [012, 009, 005, 007]
tags: [stratification, safety-set, pre-flight, plasmode, bounds, gpu, cpu]
---

# Spike 013: Reference-rate stratified safety sets

The design, hypotheses and go/stop rule were fixed before any data: `DESIGN.md`. This
README records what each stage found. **Stages 0 and 1 are done. Stage 2 (GPU generation) is
running, and stage 3 (real-data plasmode) follows it.**

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

## Results
Pending stage 3.
