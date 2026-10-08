*An independent source check of the passages rewritten in v0.9.3, by a fresh agent shown none of the earlier reviews (2026-10-07). It checked the text as it stood before the 40,000-draw pass; what was done about each finding is in `reports/b1w_fix_and_audit_2026-10-06.md`, section 7.6.*

# Source check of `reports/paper_certification.md` v0.9.3 (the passages changed by the `b1w` fix)

Every number in the listed passages traces to a file, and all 28 rows of Table 4 recount correctly. There are two high findings, both about the text and not the data, and eight medium ones. No repository file was edited; my scripts and outputs are in `/tmp/claude-1000/-home-hannanabdul-seldonian-fairness/7d6bd08c-7147-49c9-b474-eff8af75b2a9/scratchpad/`.

## 1. Findings, most serious first

### High

**H1. Section 6.3 defines both Wilson-type bounds with the trivial root that was the bug.**
- Text: "the smallest `m >= p_hat` with `m - p_hat >= z sqrt(m (1 - m) / n)` … so its width is not zero when no positive is observed", and the same "smallest `m >= mu_hat`" for `b1w`.
- Source: at `p_hat = 0`, `m = 0` satisfies the inequality as 0 >= 0, so the printed definition returns 0. That contradicts the next sentence and is the pre-fix behaviour. The same holds for `b1w` whenever every stratum is all-zero or all-one. The code now excludes that point (`stratbounds.py`, `ok[0] = mu >= 1.0`).
- Correction: write "the smallest `m > p_hat`" (the upper root), with U = 1 at `p_hat = 1`, in both definitions. The `b1w` docstring has the same wording.

**H2. Table A1's caption explains a gap by the wrong cause, and the spike-013 ESS figures are at delta 0.10.**
- Text: "whose ESS is measured from the estimate and not from the truth (section 6.4), so the two sets of ESS differ too (5.13 against 4.74 for refusal of harmful prompts at n_s 100)", under a caption that says delta 0.05.
- Source (`real_plasmode_randomties.json.xz`, C3:refusal, step 200, n_s 100, k 8, H 8): the estimate-based ESS is 4.79 at delta 0.05 and 5.13 at delta 0.10. The truth-based figure in `stratppi.json` is 4.74. So the definition accounts for 4.79 against 4.74, and delta accounts for the rest.
- The 013 README figures (2.40/2.42, 5.13/5.33, 1.42/1.43) all reproduce at delta 0.10 only. At delta 0.05 they are 2.30/2.35, 4.79/5.14 and 1.39/1.41.
- Consequence for the abstract (medium): "1.4 to 5.3" is a delta 0.10 range quoted beside "a 5% level". At delta 0.05 it is 1.4 to 5.1.
- Correction: say in the caption that section 8.1's figures are at delta 0.10, and either quote "1.4 to 5.1" in the abstract or state the delta.

### Medium

**M1. "5 of 28" and "1.5 points" depend on the 5,000-draw resolution.**
- Text: abstract "runs up to 1.5 points over a 5% level"; 7.1 "over its level in 5 of 28 mid-rate cells at delta 0.05 (largest miss 0.065)"; sections 8.1 and 11 repeat the 5 of 28.
- The stored file does give 5 / 5 / 18 and 0.0654. I re-simulated the with-replacement stratified draw independently, as per-stratum binomials at the stratum rates with fresh seeds, at 40,000 draws a cell.

  | `b1w`, with replacement | stored, 5,000 draws | independent, 40,000 draws |
  |---|---|---|
  | delta 0.05, cells over of 28 | 5 | 8 |
  | delta 0.05, largest miss | 0.065 | 0.061 (se 0.0012) |
  | delta 0.10, cells over of 28 | 2 | 6 |
  | delta 0.10, largest miss | 0.118 | 0.111 |

- At 40,000 draws the 0.0654 cell reads 0.0575. Of the 12 cells at 65% and above, 8 are over and 2 unresolved; the 16 cells at 9-18% stay at 0.016-0.033.
- The 1.5 points sits on the 93% label, outside the abstract's "rates of 9% to 66%". Within that range the largest is 0.057 (stored) or 0.056 (40,000).
- `b1w` on the 1.4% label at n_s 200 is 0.0546 stored (unresolved) and 0.057 at 40,000 (over). The text names only the pooled bound there.
- Correction: "over its level on the two labels at 65% and above, by about one point", and note that the count grows with the number of draws.

**M2. The guidance in 8.1 item 4 goes beyond its cells and contradicts 7.1.**
- Text: "`b1w` suits … a rate between about 10% and one half; elsewhere the bootstrap-t StratPPI limit is the one that kept its level."
- No real cell lies between 18.5% and 65%. In the i.i.d. synthetic grid (`check_b1.md`, rerun at four decimals) `b1w` is over in 3 of the 24 cells with truth 0.30-0.45, all at n_s 100: 0.0573 at delta 0.05 (truth 0.35, threshold 0.0569), and 0.1143 and 0.1115 at delta 0.10.
- "The one that kept its level" is not right: the stratified Wald-t `b1` is over in 0 of 80 cells across both designs, both deltas and rare labels, as 7.1 itself says. Its median ESS with replacement is 2.01, against 1.93 for the bootstrap-t StratPPI.
- Correction: give the tested range (9-18%), cite the exact Wilson asymmetry as the reason for "below one half", and name both limits that kept their level.

**M3. `replacement_check.py`: the StratPPI arms keep the finite pool's N_h.**
- The script passes `pool_stats` with N_h of about 62. So with replacement the estimator still shrinks lambda by 1 / (1 + n_h / N_h), adds `lam^2 var(f) / N_h` to the variance, and the bootstrap perturbs a predictor mean that is known exactly. In the large-pool limit these terms vanish.
- I replayed the script's seeds (160 stored cells reproduced with difference 0.0) and reran with N_h set to 1e12.

  | arm, mid-rate cells, with replacement | N_h as stored | N_h unbounded |
  |---|---|---|
  | bootstrap-t, delta 0.05 | 0 / 1 / 27, max 0.0506 | 0 / 2 / 26, max 0.0518 |
  | bootstrap-t, delta 0.10 | 0 / 5 / 23, max 0.1036 | 0 / 6 / 22, max 0.1056 |
  | normal limit, delta 0.05 | 13 / 8 / 7 | 18 / 4 / 6 |
  | bootstrap-t, median ESS | 1.93 | 1.90 |

- "Over in none" survives. Table 4 row 12 and the 1.93 in 8.1 are not quite the stated limit.
- Correction: rerun those arms with N_h unbounded, or say that they keep the 500-prompt pool's predictor statistics.

**M4. "A little over their level" is a selective reading of `wilson_exact.md`.**
- Text: the heading of reading 1, and the quoted 0.069, 0.068, 0.042 and 0.056.
- Those numbers are right, but the same file gives suprema at n 100, delta 0.05 of 0.085 for rates of 80-95%, 0.121 for 95-99%, and 0.200 just above U(n-1) (p = 0.9978, which is 1 - p^100). At delta 0.10 they are 0.146, 0.181 and 0.259.
- The paper's own 95.35% checkpoint is in the 95-99% band. The exact pooled miss at its 93.35% cell (n_s 100) is 0.0686.
- "A step function" is loose: the miss jumps at each attainable limit and falls between them.
- Correction: quote the suprema for the high bands and restrict "a little" to rates below one half.

**M5. The synthetic i.i.d. grid is under-reported.**
- Text: the Wald-t limit "is over its level on synthetic strata with rates near zero".
- `check_b1.md` (4,000 draws): `b1` is also over at strata rates of 0.2-0.5 (0.0578 at delta 0.05 and 0.1153 at delta 0.10, n_s 100) and at a flat 0.3 (0.1143).
- The paper says nothing of `b1w` on that grid. It is over in 5 of 36 cells: 0.1350 (truth 2%), 0.1168 (5%), 0.1143 (30%), 0.1115 (45%) at delta 0.10, and 0.0573 (35%) at delta 0.05.
- Correction: report the `b1w` count beside the with-replacement rows, and drop "with rates near zero".

**M6. Section 11, "Our own code", claims more than the test file shows.**
- Text: "Each bound we use now has a test at its end points … against a closed form or the vacuous value."
- `tests/test_bound_endpoints.py` passes with 37 tests and 1 strict xfail. The xfail is the paired bootstrap-t limits of section 10.4 (`pool_upper`, `new_prompts_upper`). I ran them: with one -1 in 50 differences both return 0.0036, where the betting bound gives 0.076.
- The carried Youden estimator is pinned at 0.0 on a negative estimate. The library bounds and the stratified bootstrap-t are tested at zero positives only.
- The list of "bounds that still return 0" omits these.
- Correction: add the paired limits and the Youden estimator as known exceptions.

**M7. Section 8.1 lists the same condition as both a weakness and a strength.**
- "Where it does not help" includes "a safety set that is a large share of the prompt pool"; item 4 says "`b1w` suits a safety set that is a large share of its pool".
- The first is about a claim on the population the pool was sampled from; the second is about the pool's own rate. Neither sentence states its target.
- Correction: name the target in both.

**M8. "1 of 322" and "over its level in no cell" are a selected subset.**
- The 322 recounts correctly (1 over, 23 unresolved, largest 0.0595). Stored `b1w` cells outside it are over:
  - 8 of 576 mid-rate cells at other (k, H) in the 013 file, all on the 65%-and-above labels at n_s 100 (largest 0.0598 and 0.1166);
  - 1 of 78 in the strata sweep (H 16, n_s 100: 0.058);
  - 1 of 28 with the `heur10` allocation on judge strata (0.065);
  - the rare-label row's `b1w` cell at delta 0.10 (0.126), left out because the row is bold.
- Correction: say that the count is for k 8, H 8 and proportional allocation, and give these.

### Low

- **L1.** Appendix A, "It is also over in every one of the four rare-label cells, where `b1w` is over in none", follows a delta 0.1 sentence. It is true at delta 0.05; at delta 0.10 `b1w` is over in 1 of the 4 (0.126).
- **L2.** "4 mid-rate labels (9-94%)" in Table 4 row 6: the truths run from 0.093 to 0.9535, so 9-95%, as the caption itself says.
- **L3.** The same arm is "9 of 26" in 8.1 item 1 and 9 / 5 / 14 of 28 in Table 4. Appendix A sets "none of ten" against counts out of 26.
- **L4.** "Ahead of `b1w` in seven cells, by 3-35%, and far behind in one" omits the two 93%-label cells, where it is behind by 14% and 8%.
- **L5.** The bug is described as "when the sample held no positive". It acted whenever every stratum was all-zero or all-one, including mixed cases such as [0, 0, 25, 25].
- **L6.** `b1w` at one stratum is the Wilson limit rounded up by at most 0.00025. In one control cell this decides the result: C2:unsafe, step 100, n_s 200, delta 0.10, truth 0.1025. The closed-form limit at 15/200 is 0.10248 and the grid value is 0.10252, so the exact miss is 0.119 against 0.076 (simulated 0.0772). With the closed form, Table 4 row 17 at delta 0.10 would read 6 / 9 / 13.
- **L7.** "It tends to `exp(-z^2)` at any n" should read "is close to it at any n and tends to it as n grows". The `wilson_exact.py` docstring says 0.193 where the value is 0.1935; the paper's 0.194 is right.
- **L8.** "Which is how often a valid bound lands in that class here" is true of Clopper-Pearson, whose exact miss at these rates is 0.029-0.049 (expected 2.25 unresolved, observed 2). A bound sitting at delta would be unresolved in about half the cells.
- **L9.** Section 6.3's reason for "heuristic" (clipping) is true but incomplete: a uniform shift is not the restricted maximum-likelihood estimate even when nothing is clipped.
- **L10.** Section 4.3: the code also divides by the group's standard deviation, so a penalty's strength is not proportional to lambda. The Lagrangian sentence matches the standard result if "satisfies" is read as "violation bounded by a term that vanishes".
- **L11.** The in-loop 0.116 comes from `b1w` with the two-phase N term and 4 strata, which the section 6.3 formula does not show.

## 2. Verified and found correct

- **Table 4**: all 28 rows (ranges, counts, draws per cell) and the caption, recounted from `real_plasmode_randomties.json.xz`, 014 `plasmode.json`, 017 `plasmode.json`, `plasmode_shift.json` and `harm.json`, `stratppi.json`, `stratppi_validate.json`, `agentdojo_recheck.json`, `replacement_check.json` and the training paper's table.
- **Below the table**: 0.0595 and 0.0485 pool to 0.054, which is unresolved; the cell is not over after a correction for 18 cells (p 0.0039 times 18); 322 is 48 + 4 + 18 + 168 + 28 + 56 with 1 over and 23 unresolved.
- **Reading 1**: 0 of 24 rare cells over at delta 0.05; two at 0.126 on the 1.9% label; every exact Wilson number (reproduced to four decimals by a separate root-finding enumeration); the pooled counts of 6 of 28 and 1 of 12 (0.065); Clopper-Pearson 0 / 2 / 26.
- **Readings 2 and 3**: the Student-t cells (3 / 2 / 0), 0.99^100 = 0.366, the Wald-t limit over in none in either design, and 0.116 unresolved at 500 runs.
- **Section 8.1**: the gain of 1.01-1.09 at rare rates (delta 0.05); 0.96 and 1.12; items 1 to 4 (4 of 10, 0.045, 2.10 against 2.16, 2.6-2.7 against 1.8-2.2, 2.20 and 1.93, and the six ESS pairs).
- **Appendix A**: every Table A1 row and bold mark; predictor constant in 5 to 7 of 8 strata (recomputed from the pools); 3 of 10 at delta 0.1; the allocation counts; the sweep; 1.97 against 2.10; Wald-t 0.036 and 2.06.
- **Sections 4.3, 6.2 and 6.4**:
  - Advantages are (r - group mean) / (group std + eps) in TRL 1.12.0 and in `synthetic.py`.
  - N is 500 and the safety set is 20-40% of the pool.
  - The truth equals the weighted stratum rates in all 20 pools.
  - The plasmode checkpoints were trained on unrelated prompts, and the in-loop check trains on the candidate split.
- **Section 11**: 5 of 28; the two-way zero-success share is at most 5.8% and removing those draws leaves its counts unchanged. `validity_recount.py`, run with output redirected to scratch, passes all its assertions against the current text and reproduces the stored report byte for byte.
- **`replacement_check.py`**:
  - The with-replacement draw is i.i.d. within stratum and scored against the right truth.
  - My independent simulation agrees with the stored cells (standardised differences: mean 0.19, sd 1.02, largest 2.52 over 112 cells).
  - Clopper-Pearson is exact on its arm.
  - The binomial control is sound for the random-draw arm.
  - The control does not cover the stratified path or the StratPPI arms.
- **`wilson_exact.py`**: the enumeration, the suprema, the limit and the band means are all correct.

## 3. Not checked, and why

- **[R 6.2] rows**: only the printed table exists; the draws are lost.
- **Not rerun**: `stratppi_validate.py`, the 013, 014 and 017 harnesses and `agentdojo_recheck.py`. I recounted their stored files and confirmed the 013 and 014 files agree cell for cell with `replacement_check.json`.
- **TRL version**: I checked the installed 1.12.0, not whatever the cited runs used.
- **No file to check against**: the Lagrangian sentence and section 5's "must return something" sentence. Both read as correct.
- **Numbers the recount script does not assert**: the exact-Wilson numbers, the ESS figures of 8.1 item 4, and the 1.01-1.09 gain. I checked them directly.
- **Stale background report**: `reports/b1w_fix_and_audit_2026-10-06.md` item 12 gives with-replacement counts (4 / 5 / 15 of 24; pooled 3 / 6 / 19; StratPPI 14 / 7 / 7) that differ from the current result file (5 / 5 / 14; 6 / 3 / 19; 13 / 8 / 7). The paper uses the current file.
