# Source check B: sections 5, 6.4, 8 and Appendix D of `reports/paper_certification_clean.md`

Checked against the result files, the two registrations, the scoring scripts and the git history. Nothing in the repository was edited; all recomputation was on CPU from stored files (scripts in the scratchpad, `a1.py` to `a7.py`).

**Verdict.** No HIGH finding. Both registered runs are reported as scored, and the numbers I traced agree with their sources except in M1 (strata count), M2 (a Spearman and a median taken from different deltas) and L16 (a rounding). There are 11 MEDIUM findings (dropped conditions, too-wide scope, like-with-unlike) and 18 LOW, the last two of which are notes.

**The registered runs hold up.**
- **Order.** R5: registration, `confirm_pool.py` and pools committed 00:09:14 and pushed 00:09:16 on 2026-10-08 (`9fb817d`); generation ended 01:14 after 3,855 s; `confirm.json` committed 02:25 (`fa5d82b`). R5b: registration, `confirm_mid.py`, pool and preview committed 17:01:34 and pushed 17:01:37 (`d56feb6`); preview files written 14:15; generation ended 17:24; `confirm.json` committed 18:05 (`31b5e60`).
- **Scoring code.** `confirm_pool.py`, `confirm_mid.py` and both run scripts have one commit each. `replacement_check.py` and `twophase_check.py` were last changed in `56721e6` (2026-10-07), before both runs. The working tree has no uncommitted change to any of them.
- **Predictions.** I compared P1 to P5 (eight items) and M1 to M9 (ten items) with the scoring code and the paper: the wording, the outcomes (6 kept, 2 refuted; 9 kept, 0 refuted, 1 untested) and every quoted count match.

Line numbers are those of the clean copy.

---

## MEDIUM

### M1. The registered strata rule is described as weaker than the stored run shows (lines 785-787)

> "That rule (ties kept together, cuts at quantiles, small strata merged) collapsed to one or two strata here, because 73-97% of prompts have an 8-sample reference rate of exactly 0 or 1, and often gave no gain."

**Source.** `results/spikes/013/real_plasmode.json.xz` (ties = keep, the registered rule), arm S2, k 8, H 8, field `n_strata`; adopted rule from `real_plasmode_randomties.json.xz`.

- Strata: 1 on the three labels where the adopted rule also gains nothing (2% gated, 94% refusal, 1% unsafe). On the three featured labels it is 2 at n_s 100 and **3 at n_s 200**.
- Gain on the featured labels, delta 0.05, range over three checkpoints:

| label | registered rule, n_s 100 / 200 | adopted rule, n_s 100 / 200 |
|---|---|---|
| over-refusal (17%) | 1.62-1.74 / 2.26-2.33 | 2.30-2.38 / 2.35-2.45 |
| encoded non-refusal (9%) | 1.16-1.20 / 1.31-1.38 | 1.39-1.50 / 1.41-1.52 |
| harmful refusal (66%) | 3.02-3.79 / 5.10-5.60 | 4.33-4.79 / 4.59-5.14 |

- Under the registered rule `b1w` was over its level on the 66% label in 2 of 6 cells at delta 0.05 (0.073, 0.092) and 4 of 6 at 0.10 (up to 0.131).

So "one or two" is wrong at n_s 200, and "often gave no gain" is true only on the labels where nothing gains. The 73-97% figure is right (recomputed: 73% to 97%).

**Suggested.** "That rule (...) collapsed to one stratum on three labels and to two or three on the others. On the three labels featured above it gave 1.2 to 2.3 at 9% and 17%, and 3.0 to 5.6 at 66%, where `b1w` was then over its level in 2 of 6 cells at delta 0.05 (0.073 and 0.092)."

### M2. Pre-flight: "0-20% high at H = 8" drops three conditions, and the ranking criterion was registered too (lines 826-831)

> "Spearman 0.83 between predicted and realised ESS on real data, with the prediction 0-20% high at H = 8 (Figure 1). ... The absolute-error criterion we pre-registered (median error at most 0.1) failed on real data (0.29) and passed on the bandit (0.02); we report the ranking as the usable part."

**Source.** `real_plasmode_randomties.json.xz`, S2, k 8, H 8, `pf_ess_pred` against realised ESS.

- 0-20% holds only at the last checkpoint, at delta 0.10, on the three featured labels (+20%, +4%, +2% to +6%).
- Over all three checkpoints and both deltas the same labels run +13% to +30% (17%), +4% to +14% (9%) and +2% to +28% (66%).
- On the 94% label it is +18% to +98%; on the two rare labels +33% to +44%.
- Spearman and median error over the 432 cells: 0.842 and 0.293 at delta 0.10; 0.830 and 0.313 at delta 0.05. The paper pairs the Spearman of one delta with the median of the other.
- `DESIGN.md` (`9e1b7b4`, line 78-79) registered both halves: "absolute error <= 0.1 in the median cell, and the ordering of cells preserved (Spearman >= 0.8)". Under the registered strata rule the Spearman is 0.708 (`plasmode_real.md` line 212; recomputed 0.707-0.708). The ranking passes only under the rule adopted afterwards.

**Suggested.** "Spearman 0.84 between predicted and realised ESS over 432 cells at delta 0.10 (0.83 at 0.05), under the adopted strata rule; under the registered rule it is 0.71, below the 0.8 we had registered. At H = 8 the prediction is 2% to 20% high on the three labels at 9% to 66% at the last checkpoint and delta 0.10, up to 30% high over all checkpoints and both deltas, and 18% to 98% high on the 94% label and the rare labels. Both registered criteria (median error at most 0.1; Spearman at least 0.8) failed under the registered rule (0.28; 0.71)."

### M3. "`b1w` is over its level in no cell" is true of the 28 mid-rate cells only (lines 858-859)

> "For the pool's own rate, with a safety set of 20-40% of the pool, `b1w` is over its level in no cell."

**Source.** `results/paper/replacement_check.md`, Counts table, stored pass.

- Mid-rate: 0 / 3 / 25 at delta 0.05 (three unresolved, 0.052-0.054, all on the 93-95% label at n_s 100); 0 / 0 / 28 at 0.10.
- Rare: 0 / 0 / 12 at 0.05; **1 / 0 / 11 at 0.10** (C2:gated, checkpoint 200, n_s 100: 0.126).

`validity_recount.py` line 889 asserts the mid-rate groups only.

**Suggested.** "For the pool's own rate, with a safety set of 20-40% of the pool, `b1w` is over its level in none of the 28 mid-rate cells at either delta (3 unresolved at delta 0.05, on the 93% label), and in 1 of 12 rare-label cells at delta 0.10 (0.126)."

### M4. "Gains on new prompts from one source are 1.5 to 1.6" takes half its range from a limit that failed (lines 936-938)

> "The gains on new prompts from one source are 1.5 to 1.6, at the low end of the 1.5 to 5.0 that the same measure gives at rates of 9% to 66% on the pools that chose the rule"

**Source.** `results/paper/confirm/confirm.md`, sections 2 and 3.

- The four values are `b1w` with a large pool: 1.53 and 1.62 at 12%, 1.49 and 1.62 at 82%.
- At 82% `b1w` is over its level in all 4 cells. Section 6.4 says width is compared at equal validity.
- The limit that held at 82%, Wald-t, gives 0.86 and 1.13.
- The 5.0 end of the older range is the 66% label, where `b1w` is over in 2 of 6 cells with a large pool (`replacement_check.md`, by-label table).
- The two ranges are like with like in design (`validity_recount.py` lines 1015-1016: large pool, truth to limit, delta 0.05). The gap is validity, and the 98% pool (0.55, 0.63) is also "one source".

**Suggested.** "On new prompts from one source, a limit that kept its level gained 1.5 to 1.7 at 12% (`b1w` 1.53 and 1.62; Wald-t 1.47 and 1.73) and 0.86 and 1.13 at 82% (Wald-t; `b1w`'s 1.49 and 1.62 there come from a limit over its level in all four cells). The same measure gives 1.5 to 5.0 for `b1w` on the pools that chose the rule, the upper end at 66%, where `b1w` is also over its level with a large pool."

### M5. "This paper needs training in two places only" is contradicted by section 8.1 (lines 353-355)

> "This paper needs training in two places only: the policy whose human-terms certificate section 10.4 costs, and the negative result of section 9.2, which exists because training targeted the label."

**Source.** Section 8.1 itself.

- The strata gains are measured on trained checkpoints (spike 013, steps 100 and 200).
- The 2.1-2.2 is on spike 014's pushed-label run.
- Both confirmation runs use spike 014's step-175 adapter (`scripts/confirm_pool.py`, `ADAPTER`).
- The pre-flight's `ICC_cand` and `rho` are properties of a trained policy.
- The two places named are one run (spike 014), which is also the confirmation's policy.

**Suggested.** "This paper trains no policy of its own beyond two runs made for other purposes: the side-effect run whose checkpoints section 8.1 resamples, and the constrained run that supplies section 8.1's pushed label and confirmation policy, section 9.2's negative result and the policy section 10.4 costs."

### M6. Appendix D, two-way bounds row: the registration reached the remote in the same push as its results (line 1648)

> "a registration committed with its script on 2026-10-07 (`cb50927`), before the script was run on the benchmark's tables"

**Source.** `git reflog show origin/llm-seldonian-rl --date=iso`, plus file times.

- `cb50927` committed 20:26:06.
- `results/paper/agentdojo_twoway.json` written 20:43:20.
- The first push containing `cb50927` is the push of `56721e6` at 21:08:31, the commit that adds the results.

The local commit time is consistent with the stated order. But the appendix's own test is "a commit that was made, and pushed to a remote, before the result existed", and this row does not meet it. Rows 6 and 7 do and say "pushed"; rows 2 to 4 say "no independent timestamp"; this row says neither. (The reflog is this machine's record; a push from elsewhere would not show.)

**Suggested.** Append: "; the commit was not pushed until its results were (`56721e6`, 42 minutes later), so the order rests on the local commit time".

### M7. "The within-stratum regression adds little once the limit is valid" holds at n_s 100 with 8 strata only (lines 854-857)

> "**So the gain here is the stratification.** The within-stratum regression on the reference rate adds little once the limit is valid. At a safety set of 200 or with 4 strata the bootstrap-t StratPPI is the narrower, with a median ESS of 2.6-2.7 against 1.8-2.2 for `b1w`"

**Source.** `results/paper/stratppi_validate.md` section 5, median ESS over 13 cells, bootstrap-t StratPPI against `b1w`:

| n_s, H | bootstrap-t StratPPI | `b1w` |
|---|---|---|
| 100, 4 | 2.65 | 1.88 |
| 200, 4 | 2.63 | 1.83 |
| 200, 8 | 2.70 | 2.13 |
| 200, 16 | 2.63 | 2.23 |
| 100, 8 | 1.77 | 2.19 |
| 100, 16 | 1.37 | 2.35 |

In four of the six blocks the regression adds 0.4 to 0.8 of ESS (18% to 44%). The numbers in the second sentence are right; the first sentence states the n_s 100, H 8 result as general. All of it is on the stored design at delta 0.05.

**Suggested.** "**At a safety set of 100 with 8 strata the gain is the stratification:** the regression adds nothing there once the limit is valid (2.10 against 2.16). At a safety set of 200, or with 4 strata, it adds 0.4 to 0.8 of ESS (median 2.6-2.7 against 1.8-2.2 ...)."

### M8. Section 8.2's closing rule overstates `b1w` between 20 and 45 labels a stratum (lines 1004, 1015-1018)

> "It is the narrowest such route with 5 or 10 strata at the 20% and 5% rates" ... "about 45 or more per stratum. Below that, `b1w` on the same strata is narrower."

**Source.** `results/paper/stratppi_validate.md` section 5, part B, and `results/paper/stratppi.md` part B.

- 5-11 labels a stratum: `b1w` narrowest in 10 of 10 cells.
- 20-25 labels a stratum: `b1w` narrowest in 4 of 10 and bootstrap-t StratPPI in 6 of 10; medians 2.29 against 2.23.
- 45-50: bootstrap-t StratPPI in 10 of 12.
- "Narrowest with 5 or 10 strata" is true of medians, not cells: at the 20% rate with 100 labels `b1w` is narrower in all four cells (K 5: 2.33 against 2.32 and 2.25 against 2.13; K 10: 2.57 against 2.48 and 2.40 against 2.11).

The quoted medians (2.54, 1.87, 2.47, 1.76, 2.41, 1.42, 2.24, 1.26, 1.67, 1.39) all check. This recommendation is read off the same 28 to 42 cells that validate it, at delta 0.05, on one refusal judge in two wordings; the text says so for the count threshold only, and section 8.2 has no confirmation run.

**Suggested.** "... about 45 or more per stratum. With about ten or fewer, `b1w` on the same strata is narrower in every cell; at 20 to 25 the two are level (`b1w` narrowest in 4 of 10 cells). Like the threshold, this is read off the cells it describes."

### M9. The change of coverage truth is disclosed without its size (lines 787-789; Appendix D line 1644)

> "and the truth used for coverage was changed to the mean of the half of the responses the draws come from."

**Source.** `.planning/spikes/013-stratified-safety-set/validity_real.md`, second table (the registered truth) against the first, both at k 8 and H 4, largest miss over checkpoints; spike 013 README step 10.

- Under the registered truth the stratified arm missed up to 0.201 at delta 0.10 (over-refusal, n_s 200); 0.054 after the change.
- The random-split baseline missed 0.136-0.176 on the same label against 0.066-0.075.

The change is defensible (the registered truth carried a fixed offset shared by every draw). A reader cannot tell that under the registered analysis both the method and its baseline were over their level on over-refusal. Appendix D's first row lists two registered items as not kept and omits this third.

**Suggested.** Add: "Under the registered truth the stratified arm and the random-split baseline were both over their level on over-refusal (up to 0.20 and 0.14 to 0.18 at delta 0.10, with 4 strata), because that truth carried a fixed offset of about 0.004 shared by all draws." Add "and the coverage truth" to the Appendix D row.

### M10. Table 5 states no rate range, and its rows were tested at 1.3% to 20% and delta 0.05 only (lines 968-976)

> "| gold labels, 10 or more in each class | PPI++ ... bootstrap-t limit | approximate |" and "| gold labels from a stratified sheet | design-weighted labels, `b1w` | approximate |"

**Source.**
- Row 3: `.planning/spikes/017-calibration-carrying-certificate/route.md`. 20 cells, rates 0.013 to 0.2, delta 0.05, logit of one refusal judge in two wordings. The text gives this scope; the table does not.
- Row 4: `harm.md` section 3. Planted rates 0.013, 0.05 and 0.2 on one sheet design, 18 cells, `b1w` 0.000 to 0.059, one cell over (0.0595 per `validity_recount.py` line 741).

Section 8.1 concludes that `b1w` is "supported only at rates under one half" and is over its level on rare labels in some cells. Row 4 routes any stratified sheet to `b1w` with no rate condition.

The routing sentences themselves check: threshold 10, largest miss 0.052, two cells above delta, none over at 3,000 draws (band ends at 0.058); the smaller-of-two rule up to 0.078. They are not in `validity_recount.py`; I checked them by hand.

**Suggested.** Caption: "Rows 3 and 4 were resampled at rates of 1.3% to 20% and delta 0.05 only (row 4 on one sheet design, 1 of 18 cells over, 0.0595). Section 8.1's finding that `b1w` is over its level above one half applies to row 4."

### M11. The pushed-label result drops its conditions (lines 780-782)

> "When the Lagrangian pushed the stratified label itself, the gain was 2.1-2.2 with no compression of the between-prompt spread (correlation between a prompt's reference and trained rates 0.92)."

**Source.** `.planning/spikes/014-pushed-label-stratification/results.md` and README.

- 2.11 (step 100) and 2.21 (step 200) are at n_s 200 and delta 0.10 only; 2.05 and 2.16 at delta 0.05.
- The pool rate had moved 3.4 points at step 100 and 0.4 points at step 200, where the 0.92 is measured.
- The README's own narrowing: "net rate moves of at most 3 points on real data ...; a push that lands the rate far from the reference was not seen".
- On the new pools the same adapter's measured `rho` is 0.84-0.85 (`confirm.json`, `pf_rho`, K1 and K2).

**Suggested.** "... the gain was 2.1-2.2 at n_s 200 and delta 0.10 (2.05-2.16 at 0.05), with no compression of the between-prompt spread (correlation 0.91-0.92), on one run whose rate ended within 0.4 to 3.4 points of the reference; a push that moves the rate far was not seen."

---

## LOW

**L1. Lines 955-956**, "At 46% it kept 75% to 86% of `b1w`'s gain; at 82% it gave none or lost."
- The Wald-t gain at 82% with a large pool and n_s 200 is 1.13 (`confirm.md` section 3), a gain.
- The baseline at 82%, n_s 100, delta 0.05, pooled Wilson, is itself over its level (0.064 large pool, 0.062 source). The paper says this for the 46% pool and not here.
- Against Clopper-Pearson, which held, the Wald-t limit is 1.06 and 1.32 (0.863 / 0.816, 1.135 / 0.863, from `confirm.json`).
- Suggested: "at 82% it gave 0.86 and 1.13 with a large pool and lost for the source (0.75 and 0.89), against a pooled Wilson baseline that is itself over its level at n_s 100".

**L2. Lines 942-944**, "it predicts 2.67 at the 98% rate ... and 2.47 on a rare label".
- These follow the sentence on the reference-only tool, but they are the measured-quantities version (`confirm.md` section 1; asserted at `validity_recount.py` line 1014). The reference-only tool at `rho = 1` gives 2.65 and 2.02.
- The shipped tool's verdict on both labels is "NO: rate outside 5-95%" (`preflight.py`, `verdict`), so "not usable" applies to the formula's number, not to the tool.
- Suggested: "Outside the mid range the formula's number is not usable even with the measured quantities: it gives 2.67 ...; the tool's own verdict refuses a reference rate outside 5% to 95%."
- The three predictions 2.58, 4.12 and 2.22 are attributed correctly and I reproduced them. "Within 20%" is true for the measured version at all three rates and both sizes: ratios 0.84 and 0.89 (12%), 1.00 and 1.06 (46%), 0.90 and 0.98 (82%).

**L3. Lines 555-557**, "our tool takes `ICC_cand = ICC_ref` and `rho` of 1 or 0.8); section 8.1 measures both readings."
- Section 8.1 reports `rho = 1` only.
- Recomputed with `preflight.assess(..., rho=0.8)`: 1.64, 1.94 and 1.54 for the 12%, 46% and 82% pools. Realised gains are 1% to 7% under it at 12%, 3% under to 5% over at 82%, and 43% to 53% above it at 46%.
- Either report it or write "(... `rho` of 1 or 0.8; section 8.1 reports the `rho` = 1 reading)".

**L4. Line 947**, "With 4 strata it was over in 1 cell of 13". The sweep has 26 mid-rate cells with 4 strata (13 at each size); the one over is at n_s 100, and at n_s 200 it is 0 of 13 (`stratppi_validate.md` section 5). Suggested: "1 cell of 26 (0.062 at n_s 100 ...)".

**L5. Lines 943-944**, "where the strata lose under every limit (0.55 and 0.63 for `b1w`)". True for the large pool at delta 0.05 against pooled Wilson (Wald-t 0.11 and 0.26; bootstrap-t StratPPI 0.75 and 0.95). On the stored design at n_s 200 the bootstrap-t StratPPI is at 1.05 (delta 0.05) and 1.26 (0.10) and is not over its level (`confirm.json`, draw `stored`). Add "with a large pool".

**L6. Line 840**, "Three things follow." Four numbered items follow.

**L7. Line 780**, "A placebo covariate gave 1.0." It is 1.00-1.04 on five labels and 0.83-1.02 on the 93-95% label (`real_plasmode_randomties.json.xz`, arm P). Add "on the mid-rate labels".

**L8. Lines 819-820**, "exact stratified bounds did not beat pooling".
- `check_bounds.md` is a synthetic grid, not these pools.
- In its low-rate profile ("H2 low") the stratified exact bound beats its own pooled version (ESS 1.29 at both sizes); the README notes the exception.
- Whether it beats pooled Clopper-Pearson there is not in the file.
- Suggested: "on a synthetic grid, exact stratified bounds beat their pooled form in one low-rate profile only (1.29) and were wider than Clopper-Pearson elsewhere".

**L9. Lines 810-812 and 895-896**, "Section 6.4's cap gives ...: the prediction is met ..." and "within 15% of the cap". The cap's G is backed out of the realised large-pool gain (`twophase_check.py` report; registration P4c: "G from the large-pool gain"). It is a consistency check, not a prediction available in advance. Add "with G taken from the measured large-pool gain".

**L10. Lines 525-535** (section 6.4).
- "a simple random sample with the pooled bound of the same family": every gain quoted for the Wald-t and bootstrap-t StratPPI limits is against the pooled Wilson bound (`replacement_check.py` report; `confirm_pool.py` `ess`).
- The truth-to-limit measure is said to apply to "Table A1 and the medians of sections 8.1 and 8.2"; it also applies to every prompt-source and confirmation gain.

**L11. Lines 338-340**, "for 27 of 28 pipelines of an agent-security benchmark". True for the bound clustered by user task with injection tasks fixed. Under the one bound that covers new injection tasks no pipeline certifies (line 1189). The robot clause gives threshold and delta; this one gives neither. The list also omits the brevity constraint, where 10 of 10 seeds certified (line 755).

**L12. Appendix D line 1650**, "set the floor of a third (the Wald-t limit's gain, raised from 1.5 to 1.7)".
- No committed file holds 1.5: the registration and `confirm_mid.py` (`WALD_GAIN = 1.7`) have one commit each. The only record is a `PLAN.md` note written after the results ("a placeholder of 1.5").
- On the count: the registration names two changes (M2, M3b) and gives M5's floor as "80% of the preview's smaller value"; the paper's three is the fuller count and does not understate.
- Not listed by either: M8 moved from `b1w`'s gain (R5's P4c) to the Wald-t limit's and added "above 1"; R5's prediction on `b1w` with the term was dropped to "reported without a prediction".
- The paper's blanket statement that all ten were written after the re-mix covers these.

**L13. Appendix D line 1646**, "a note dated before scoring". The note is stamped 08:30 PT. `score_e8.log` ends at 475 s and the score file was written 08:36:12, so the pass began about 08:28. The note was written while scoring ran and before any result existed. Suggested: "a note dated while the scoring pass ran, before its output". "No independent timestamp" is already stated.

**L14. Appendix D line 1651**, "amended once ... before any label".
- The rules also had two revisions on 2026-10-04: version 2 (`9e45371`) replaced rules 3 to 5 of the first rules (`f1582e5`) after section 10.3's labels showed the difference near zero; amendment 2.1 came with `e076974`.
- "Before any label" should read "before any label of this sheet".
- "Design check unchanged": the stored `design_check.md` dates from `e076974`; I did not rerun it (it writes into the repository).

**L15. Lines 859 and 813.** "over in 7 of 28" is at delta 0.05 (6 of 28 at 0.10); "over in 17 of the 28 cells ... in 21" is at delta 0.05 (18 and 23 at 0.10). Neither sentence restates the delta.

**L16. Line 868**, "1.5 and 5.5 (bootstrap-t StratPPI)". The file has 1.55 and 5.47.

**L17. Lines 916-917 and 938-939**, the mixed pool: "part of the gain is the gain of knowing the source"; "in large part".
- Supported: 1.93 and 2.90 are in `mid/confirm.md` and are computed on the new pool.
- By variance removed, the source alone accounts for 74% of what the 8 strata remove ((1 - 1/1.93) / (1 - 1/2.90)); by gain above 1, for 49%.
- The remaining factor, 2.90 / 1.93 = 1.50, equals the within-source gain of 1.5 to 1.6 on the single-source pools.
- The ratios are a variance calculation on the trained policy's 16 responses; no two-strata bound was resampled. Giving the share would make "in large part" checkable.

**L18. Line 784.** "The rule was chosen on these pools" is said for the strata rule, the bound-by-rate rule (lines 869-871) and the routing threshold (lines 964-966), and the confirmation covers 8.1. It is not said for the 8.2 closing recommendation (see M8).

---

## Universal claims verified against the whole result file

- **Lines 798-800.** `b1w` with the term: none of 28 mid-rate cells over at delta 0.05 (4 unresolved, 0.052), one at 0.10 (0.112, C2:refusal at 93%); Wald-t with the term over in none at either (one unresolved, 0.0504). On the rare labels the Wald-t limit is also 0 of 12 at both. (`twophase_check.json`, all cells.)
- **Lines 801-802.** Rare labels: 1 of 12 and 3 of 12, with or without the term.
- **Lines 859-861.** With replacement `b1w` over in 7 of 28 at delta 0.05, all at truth 0.654 or above; bootstrap-t StratPPI and Wald-t over in none of 28 at either delta, in any of the three passes, and none of the 12 rare cells.
- **Lines 864-866.** `b1w` at or under delta in all 16 cells at 9-18% at both deltas; no label between 18.5% and 65.4% on those pools.
- **Lines 852, 1003-1008.** Bootstrap-t StratPPI over in none of ten (0.045) and none of 28 (5 unresolved, 0.056); `b1w` on judge strata none of 28 (1 unresolved, 0.053); 26 of 28 and 14 of 14 for the normal limits; PPBoot 7 and 12.
- **Lines 886-903.** Every count and miss of the first confirmation, including the cells the paper does not feature.
  - Clopper-Pearson is over in none of the 8 rare-label cells (large pool) and none for the source.
  - `b1w` at 98% is over in none of its 4 cells in any pass.
  - The Wald-t limit is over in none of the 8 mid-rate cells in the stored, paired and large passes, and none of 8 for the source.
  - Unfeatured: the Wald-t limit with the term is over on one rare-label cell for the source (K2 unsafe, n_s 200, delta 0.10: 0.110); the bootstrap-t StratPPI limit is over at 82%, n_s 200 at both deltas in the 5,000-draw paired pass (0.056, 0.111) and not in the registered 40,000-draw pass (0.049, 0.098).
- **Lines 905-934.** Every count and miss of the mid-rate pool; "all nine that could be tested already held on the re-mix" (`preview.md`: nine kept, M3a no label).
- **Lines 945-947.** "With the 8 strata of the rule the stratified Wald-t limit is the only stratified limit that was over its level in no mid-rate cell on real responses". Checked in:
  - spike 013's random-ties file, H 8, every k, three checkpoints, both deltas (0 of 192 mid-rate cells);
  - `replacement_check.json`, all three passes;
  - `twophase_check.json`;
  - `stratppi_validate.json`, H 8 and 16;
  - both confirmation files and the preview.
  - It needs both qualifiers: with 4 strata one cell is over, and on a rare label one source cell is over.
- **Lines 951-953.** `b1w` over at truths from 0.654 to 0.954 ("65% to 95%"); at 46% misses 0.044, 0.045, 0.096, 0.096.
- **Lines 978-982.** Routing rule: 20 cells, largest 0.052, two above delta, none over; smaller-of-two up to 0.078.
- **Lines 992-994.** 299, 149 and 59 labels (`ln 0.05 / ln(1 - tau)`).
- **Lines 907-908.** By-category rates 0-22%, 46-89%, 87-100%; three categories inside the gap with 47, 20 and 17 prompts (recomputed from `judged.jsonl`).
- **Section 5, lines 338-340.** 1 of 3 seeds, all three robot models at 5% and delta 0.05, 27 of 28 pipelines: each agrees with sections 7.2, 10.1 and 10.2 as written.
- **Appendix D.** All eight hashes exist with the stated dates and contents.
  - `9e1b7b4` was pushed at 18:25:53, before `1afe911` at 19:46 (pilot files written 19:00-19:29).
  - Rows 2 to 4 are single commits, as the table says.
  - "No label has been collected": `results/labels/p9/` holds no label file.

## Claims I could not trace to a stored source

- **"raised from 1.5 to 1.7"** (Appendix D): a post-hoc plan note only (L12).
- **"with the design check unchanged"** (Appendix D): rests on the commit message of `a5bd04b` and `PLAN.md`; not rerun.
- **"that no earlier pool, training set or result had used"** (lines 882-883): the exclusion scans 165 result files and re-derives the benign-prompt loader's draws at 10 seeds and 7 sizes, because earlier episode files are partly lost (`confirm_pool.py`, `used_before`). It is a reconstruction, not a stored list of training prompts.
- **"pushed"** in Appendix D rows 6 and 7: consistent with this machine's reflog of the remote (00:09:16 and 17:01:37); not checkable from the repository alone.
- **2.58, 4.12, 2.22 and the by-category rates**: no results file prints them; `validity_recount.py` recomputes and asserts them, and I reproduced both.
- **"An ESS of 2.4 is worth 140 labels at n_s 100"** and **"enlarging the safety set is the better use of the budget"**: arithmetic and argument, no result file.
- **"which is the asymmetry of the Wilson limit"** (line 866): `results/paper/wilson_exact.md` (band table: mean miss under delta below one half, over above) supports it for the pooled limit; nothing isolates it for the stratified `b1w`.

## What `validity_recount.py` does not assert in this part

It asserts numbers and counts, and that quoted sentences are in the draft word for word. It does not assert:
- the scope of M3 (mid-rate groups only);
- the strata counts or gains of the registered rule (M1);
- the 0-20% range or the Spearman (M2);
- the routing-rule sentences of 8.2;
- "the only stratified limit" beyond the new pools (its lines 1121-1123 cover the three new pools; I checked the rest);
- any of Appendix D.
