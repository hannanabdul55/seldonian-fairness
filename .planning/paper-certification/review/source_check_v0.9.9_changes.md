# Source check of the v0.9.9 corrections

Draft: `reports/paper_certification.md` (working tree, v0.9.9) against the diff `draft_v099.diff`.
Nothing in the repository was edited. `scripts/validity_recount.py` was read, not run.

**Result.** No new number is wrong against its source. 26 findings: no HIGH, 12 MEDIUM, 14 LOW.
The MEDIUM ones are all a number or a universal word placed in a wider scope than its source
has. Three of them are in the sentences most likely to be quoted: the abstract's "24% in the
worst" (M1) and the new list "The rule, in one place" (M6, M7).
Two unchanged sentences of section 10.2 still say "an earlier draft" (D1 below).

How it was checked: Table 4's rows 32, 33, 35 and 36, section 8.1's strata-rule and pre-flight
numbers, the confirmation gains and the Clopper-Pearson extremes were recomputed from the stored
JSON files; the rest was read against the `.md` result files named in each entry. Groups 6 and 8
were checked by a forked copy of this checker; its findings are marked (fork).

Abbreviations: `rc` = `results/paper/replacement_check.md`, `tp` = `results/paper/twophase_check.md`,
`sv` = `results/paper/stratppi_validate.md`, `c1` = `results/paper/confirm/confirm.md` (+ `.json`),
`c2` = `results/paper/confirm/mid/confirm.md` (+ `.json`), `013/` = `.planning/spikes/013-stratified-safety-set/`.

## MEDIUM: a dropped condition or too-wide scope

**M1. Lines 73-77 (abstract).**
- Sentence: "At the sample sizes of a safety test, in our implementation, they were wrong more often than promised in most of our tests with a judge's score, and in 24% of trials in the worst (a 1.3% rate with 225 labels)."
- Source: 0.241 is PPI++ with a normal limit (017 `results.md`, rubric, rate 0.013, n 225) and 0.239 is StratPPI's normal limit with proportional allocation (`results/paper/stratppi.md` part B, same cell). Table 4's own row for StratPPI's normal limit under the paper's allocations has 0.219 (oracle) and 0.90 (heuristic; `validity_recount.md` line 55: 0.9025, rubric judge, 5% rate, 1,000 labels). "Most" holds either way (126 of 168, 26 of 28, 28 of 28, 24 of 28).
- Problem: "in the worst" is true only with labels allocated in proportion to stratum size.
- Replacement: "... in most of our tests with a judge's score: in 24% of trials in the worst when labels were spread in proportion to group size (a 1.3% rate with 225 labels), and in 90% under one of the allocation rules published with the second."

**M2. Lines 793-794 (section 7.1, reading 3).**
- Sentence: "The Wald-t limit's record is clean on real responses with 8 strata and is not clean on synthetic ones."
- Source: `c1` section 4: the Wald-t limit with the sampled-pool term is over at K2:unsafe, n_s 200, delta 0.10 (0.110 at 10,000 replications, band ends at 0.106). The draft says so itself at lines 1014-1015 and 1074.
- What holds: no mid-rate cell over with 8 strata in any file (stored 0 of 56, with replacement 0 of 56, source 0 of 56, confirmation 0 of 24, and 0 of 48 in spike 013's own file at k = 8, H = 8).
- Replacement: "On real responses with 8 strata the Wald-t limit was over in no mid-rate cell and, with the sampled-pool term, in one rare-label cell (section 8.1); on synthetic data it is over in 10 of 36 cells."

**M3. Line 884 (section 8.1, Result).**
- Sentence: "A placebo covariate gave 1.00 to 1.04 on every label but the one at 93% (0.83 to 1.02) [013]."
- Source: `results/spikes/013/real_plasmode_randomties.json.xz`. The placebo arm exists only at k = 8, H = 4 (72 rows a bound). Its ESS over both deltas, both sizes and three checkpoints is 1.00 to 1.04 on five labels and 0.83 to 1.02 on C2:refusal. The numbers are right.
- Problem: the paragraph opens "With k = 8 and H = 8".
- Replacement: "A placebo covariate, run with 4 strata, gave 1.00 to 1.04 on every label but the one at 93% (0.83 to 1.02) [013]."

**M4. Lines 934-937 (section 8.1, Where it does not help).**
- Sentence: "labels near 0 or 1 (at a 93% rate 0.88 and 0.93 as a median over checkpoints, 0.96 and 1.12 at the last one, Table A1)".
- Source: `tp` section 2, C2:refusal. 0.88 and 0.93 are `b1w` with the sampled-pool term for the prompt source. For the pool's own rate the median over checkpoints is 0.96 and 1.11 (stored) and 0.95 and 1.11 (large pool).
- Problem: the two pairs differ by the claim, not by median against last checkpoint.
- Replacement: "labels near 0 or 1 (at a 93% rate 0.96 and 1.12 for the pool's own rate, Table A1, and 0.88 and 0.93 for the prompt source)".

**M5. Lines 968-970 (section 8.1, Against StratPPI, item 3).**
- Sentence: "At a safety set of 100 with 8 strata the gain is the stratification. The within-stratum regression on the reference rate adds nothing there once the limit is valid (2.10 against 2.16)."
- Source: 2.10 against 2.16 is the median over the ten cells of both sizes (`results/paper/stratppi.md`, summary). At n_s 100 with 8 strata the sweep gives 1.77 against 2.19 (`sv` section 5, 13 cells); at n_s 200 with 8 strata it gives 2.70 against 2.13.
- The rest of item 3 is right: differences of medians 0.77, 0.80, 0.57 and 0.40; 2.63 to 2.70 against 1.83 to 2.23.
- Replacement: "... adds nothing there once the limit is valid (median ESS 1.77 against 2.19 for `b1w` over the 13 cells of the sweep)."

**M6. Lines 1095-1096 (The rule, in one place, second bullet).**
- Sentence: "the stratified Wald-t limit; `b1w` is narrower and kept its level only at rates under one half".
- "only": for the pool's own rate with a safety set of 20-40% of the pool, `b1w` is over in none of the 28 mid-rate cells, the 66% and 93% labels included (lines 973-975; `rc`). On the 82% confirmation pool in the stored design it is not over either (0.052 and 0.100, unresolved; `c1` json). It is over above one half once the pool is large.
- "narrower": not at n_s 200 on the low-rate labels. Wald-t against `b1w`: 1.73 against 1.62 at 12% (`c1`), 1.49 against 1.36 at 9%, 2.32 against 2.30 at 17%, 2.17 against 2.13 on the pushed label (`results/paper/stratppi.md`). `b1w` is narrower at n_s 100 and at 46%.
- Replacement: "the stratified Wald-t limit; `b1w` is narrower at a safety set of 100 and at 46%, and once the pool is much larger than the safety set it kept its level only at rates under one half".

**M7. Lines 1090-1098 (The rule, in one place, as a whole).** Does it follow from the section? The first bullet does. The two mid-rate bullets drop what the section found above about 80%.
- Source: under the Wald-t limit the strata gave 0.86 and 1.13 at 82% with a large pool and 0.75 and 0.89 for the source (lines 1082-1083), and at 93% 0.43 and 0.65 (`rc`, last table) and 0.40 and 0.57 for the source (`tp` section 2). The rule's cut is 95%.
- Replacement, as a clause after the third bullet: "Above about 80% the strata gained nothing or lost under this limit (0.75 to 1.13 at 82%, 0.4 to 0.65 at 93%)."

**M8. Lines 1564-1566 (section 11, Where the strata pay).**
- Sentence: "for the one trained policy the correlation between reference and trained rates was 0.84 to 0.92."
- Source: `c1`/`c2` json, `pf_rho`: refusal 0.849 (12%), 0.844 (82%), 0.943 (98%), 0.901 (46%); safety field 0.833, 0.970, 0.782. Spike 014 `results.md`: 0.91 and 0.92 (corrected for sampling noise; observed 0.85 and 0.88).
- Problem: 0.84 to 0.92 is the mid-rate refusal labels only; over all its labels the range is 0.78 to 0.97. There is also a second trained policy, spike 013's run (0.80 to 1.00, `013/preflight_real.md`), which section 5 mentions.
- Replacement: "for the policy of the confirmation pools the correlation between reference and trained rates, corrected for sampling noise, was 0.84 to 0.92 on the mid-rate refusal labels and 0.78 to 0.97 over all its labels."

**M9. Line 315 (Table 2).**
- Cell: "yes; the solution rate is reported for every trained policy".
- Source: the draft reports a solution rate only for the seeded arms (lines 357-358 and 857-858: 2 of 3, 10 of 10). The Granite policies are one run each (line 1567).
- Replacement: "yes; where training was repeated over seeds the solution rate is reported (section 7.2)".

**M10. Lines 1276-1277 (section 10.1). (fork)**
- Sentence: "The tightest exact limits we have on their mean, Bentkus's and the betting mixture, are 0.44 to 0.48 at delta 0.05 (Table 6 gives the latter)".
- Source: `results/paper/robodojo_sampling.md` section 4: Bentkus 0.552 / 0.440 / 0.440, betting mixture 0.480 / 0.441 / 0.442. The range quoted is the betting mixture's; Bentkus's is 0.44 to 0.55.
- Replacement: "The two tightest exact limits we have on their mean are the betting mixture, 0.44 to 0.48 at delta 0.05 (Table 6), and Bentkus's, 0.44 to 0.55, and"

**M11. Lines 1248-1250 (section 9.5). (fork)**
- Sentence: "The sheet's labels alone, read as i.i.d., missed in under 1% of sheets, because over-sampling the flagged responses overstates the rate: the reading is wrong in the safe direction for an upper bound [017 4]."
- Source: 017 `harm.md` and `harm.json`. Under 1% is right (largest 0.0083 over 18 planted cells). The overstatement belongs to the planting (harm only among answered responses, which the sheet over-samples). On the real 225 labels the two readings give the same rate (0.0133 and 0.0134) and the i.i.d. limit is the lower one (0.034 against 0.039).
- Replacement: "On those planted sheets the labels alone, read as i.i.d., missed in under 1% of draws: the sheet over-samples the answered responses that carry the planted harm, so the reading overstates the rate there. On the real labels the two readings give the same rate (1.3%), and the i.i.d. limit is the lower one (0.034 against 0.039) [017 4]."

**M12. Lines 1157-1159 (section 8.2, unchanged), contradicted by the new lines 1167-1169. (fork)**
- Old sentence: "With 20 strata, and wherever strata hold about ten labels or fewer, `b1w` is the narrower."
- Source: `results/paper/stratppi_validate.json`, part B. With 20 strata `b1w` is narrower in the 8 cells at 5 to 11 labels a stratum and the bootstrap-t in all 6 at 25 to 50; `b1w` leads at 20 strata only by median. The new sentence (10 of 10; 4 of 10) is the one the data supports.
- Replacement: "With 20 strata `b1w` has the larger median ESS (2.59 against 2.51 at the 20% rate, 1.70 against 1.64 at 5%), and it is the narrower in every cell where strata hold about ten labels or fewer."

## LOW: wording or rounding

**L1. Line 72 (abstract).** "the standard one was wrong in at most 5.2% of ours".
- True for every pass the paper quotes (largest 0.0515, spike 017's plasmodes; 0.0514 `tp`; 0.0508 section 8.2; 0.0497 `rc` large pass; 0.046 on the confirmation pools).
- One stored pass is higher: the paired pass of `results/paper/replacement_check.json` has Clopper-Pearson at 0.0528 (C2:unsafe, step 100, n_s 100, 5,000 draws with replacement); the same cell at 40,000 draws is 0.0497.
- Replacement: "was wrong in at most 5.3% of ours", or keep 5.2% and say "of the trials we report".

**L2. Line 87 (abstract).** "for rare behaviours (under 4%) it gained nothing".
- The subject is the approximate formula. Its ESS on the three rare labels is 0.86 to 1.20 (`c1`/`c2` json: 0.99 and 1.20, 0.86 and 1.08, 0.88 and 1.12); 1.00 to 1.11 is `b1w`'s. Rates 3.2%, 1.6% and 1.9% are right.
- Replacement: "for rare behaviours (under 4%) it gained nothing on balance (0.9 to 1.2)".

**L3. Lines 896-897.** "`b1w` was over its level under it in 2 of 6 cells (0.073 and 0.092)".
- Right at delta 0.05 (`real_plasmode.json.xz`, C3:refusal, n_s 100, steps 100 and 200). At delta 0.10 it is 4 of 6 (0.112 to 0.131).
- Replacement: "... in 2 of 6 cells at delta 0.05 (0.073 and 0.092) and 4 of 6 at 0.10".

**L4. Lines 899-902.** "(misses up to 0.20, and 0.14 to 0.18, at delta 0.10)".
- Reproduced from `results/spikes/013/real_plasmode_truthhalf.json.xz` (k 8, H 4): stratified 0.104 and 0.201, baseline 0.176 and 0.136, each the largest over three checkpoints at n_s 100 and 200. Per cell the baseline runs 0.064 to 0.176 (over in 4 of 6) and the stratified arm 0.053 to 0.201 (over in 2 of 6).
- Replacement: "(largest miss over checkpoints 0.20 for the stratified arm and 0.14 and 0.18 for the baseline, at delta 0.10)".

**L5. Lines 1058-1060.** "a limit that kept its level gained 1.5 to 1.7 at 12% (`b1w` 1.53 and 1.62, the Wald-t limit 1.47 and 1.73)".
- The bootstrap-t StratPPI limit also kept its level at 12% (0.048, 0.099, 0.048, 0.097) and gained 1.55 and 1.88 (`c1` section 3).
- Replacement: add "; the bootstrap-t StratPPI limit, which also held there, 1.55 and 1.88".

**L6. Lines 593-594 with 1068-1069.** "where a baseline is itself over its level we say so".
- Kept at 82%, at 46% and for the source on the first pools. Not kept at 98%: the pooled Wilson baseline is over at n_s 100 and delta 0.05 (0.080 large pool, 0.077 source; `c1`), and "0.55 and 0.63 for `b1w`" is quoted without it.
- Replacement at line 1069: "(0.55 and 0.63 for `b1w`, against a baseline itself over its level at a safety set of 100)".

**L7. Lines 703-705 (Table 4 caption).** "is in Table A1 (largest miss 0.024)".
- 0.024 is at delta 0.05, Table A1's delta; at 0.10 the same cells give 0.067 (`results/paper/stratppi.md`). Table 4's row has both deltas.
- Also: the caption gives 5,000 draws for "[013]" and 4,000 for "the synthetic grid", whose two rows are tagged [013].
- Replacement: "(largest miss 0.024 at delta 0.05)"; tag the two grid rows "[013 grid]" or say "5,000 for [013]'s real responses".

**L8. Line 1852 (Table D1, P2a).** "kept: 0 of 4 cells at 12%".
- The registered sentence is "on a label with a rate under 0.45", with no mid-rate limit; the scorer counted mid-rate labels only (`c1` section 5: "0 of 4 cells over on 1 such labels"). With the two rare labels it is 0 of 12 (`c1` section 2). Kept either way.
- Replacement: "kept: 0 of 4 cells at 12% (0 of 12 with the two rare labels)".

**L9. Lines 1649-1651 (section 12).** "their estimators were not over on the pools we began with and were over in 2 of 16 cells on new prompts".
- On the new prompts only StratPPI's estimator was run with a bootstrap-t limit. The 16 are the four refusal labels with a large pool; the three rare labels add 12 cells with none over.
- Replacement: "... and StratPPI's was over in 2 of the 16 refusal-label cells on new prompts, one of them at a 98% rate".

**L10. Line 1678 (conclusion).** "It gained nothing at rare or extreme rates".
- At 82% it lost for the source (0.75 and 0.89) and at 98% it lost under every limit; 82% is neither rare nor extreme.
- Replacement: "It lost at 82% and 98% and gained nothing at rare rates".

**L11. Lines 1244-1246 (section 9.5). (fork)** "certified a negative harm rate under one judge wording".
- `harm.json`: wording 2 is negative with every feature; wording 4 is negative with the logit (-0.004).
- Replacement: "... under one judge wording with every feature (and under a second with the logit, by 0.004)".

**L12. Lines 1112-1113 (Table 5 caption). (fork)** "Rows 3 and 4 were resampled at rates of 1.3% to 20% and delta 0.05 only".
- 52 of the 168 cells behind row 3 are a word-count label at a 40% rate (017 `plasmode.json`, confirmed: rate 0.404, 299 rows).
- Replacement: "Row 3 was resampled at delta 0.05 only, at rates of 1.3% to 20% for a refusal label and at 40% for a word count the judge does not predict; row 4 at delta 0.05 and planted rates of 1.3% to 20%, on one sheet design (1 of 18 cells over, 0.0595)."

**L13. Lines 1515-1516 (section 10.4). (fork)** "after the labels of section 10.3, whose rates it uses: four upper limits".
- The limits use no rate from section 10.3; the budget and the design check do (`scripts/p9_certificate.py`, `scripts/p9_budget.py`).
- Replacement: "... after the labels of section 10.3, whose rates set its sample size and its design check: four upper limits".

**L14. Line 658 (Table 3, judge-routing row).** "100 to 1,000 responses". 017 `plasmode.json` holds 92 rows at n = 50, which are among Table 4's 168 PPI++ cells. Replacement: "50 to 1,000 responses", or say the n = 50 cells are in Table 4 only.

## Sentences about the paper's own drafts

No corrected sentence reads as a description of an earlier version. "The recommendation changes" (line 1078) and "became the recommendation after the first confirmation pool" (line 1590) describe the order of the work; the Appendix D rows describe registrations that were replaced. Both are allowed.

**D1. Two unchanged sentences of section 10.2 do. (fork)**
- Line 1334: "the larger of the two clustered bounds, which an earlier draft used here (16)". Replacement: "and the larger of the two clustered bounds (16) [AgentDojo recheck]."
- Line 1385: "over 30 seeds, and an earlier draft reported 0.104". Replacement: "over 30 seeds (0.104 at the seed first run)."

## Checked and found correct

**1. Abstract.**
- "24% ... (a 1.3% rate with 225 labels)": the cell is right (see M1 for the scope).
- "most of our tests with a judge's score": 126 of 168, 26 of 28, 14 of 14.
- "ran slightly over on new prompts": 0.105 at delta 0.10 (82%) and 0.052 at 0.05 (98%).
- "wrong no more often than promised at 12%, 46% and 82%": every Wald-t cell is at or under delta, large pool and source (largest 0.039 / 0.083 and 0.047 / 0.095).
- "1.5 to 1.7 at 12% and 2.1 to 2.6 at 46%": 1.47, 1.73; 2.09, 2.56.
- "At 82% it gained little or lost, at 98% it lost": 0.86, 1.13 (0.75, 0.89 for the source); 0.11, 0.26.
- "A sharper formula gave up to 3.0 but ran over its promise at 82%": 2.97; 4 of 4 cells over.
- "1.3 to 1.5 at the first two rates": 1.31, 1.32 (recomputed from `c1` json); 1.49, 1.34.
- "Fifteen of eighteen ... two failed and one could not be tested; ten were written after a trial run": 6 + 9, 2, 1; the ten of the second registration.
- "on simulated data it was wrong too often": 10 of 36 grid cells; 1 of 4 in-loop.
- "27 of 28" and "at most one of the 28 certifies": user-task bootstrap 1, pigeonhole bootstrap-t 0 (`results/paper/agentdojo_twoway.md`).
- "up to 99% of trials": 0.989.
- "about 420 ... the exact one is about twice as wide": 416 pairs (`results/paper/p9_budget.md`); margins 0.065 against 0.032 to 0.037.

**2. Table 4 and section 7.1.**
- Row 32 (Wald-t, stored): 0.0074-0.0364 / 0.0274-0.0804, 0 / 0 / 28 twice, upper limits 0.042 and 0.088. Recomputed from `replacement_check.json`.
- Row 33 (with replacement): 0.0066-0.0376 / 0.0257-0.0857, 0 / 0 / 28 twice, 0.040 and 0.088. Recomputed.
- Row 34 (sampled-pool term): 0.007-0.050 / 0.031-0.095, 0 / 1 / 27 and 0 / 0 / 28 (`tp`).
- Row 35 (Wald-t, grid): 5 / 5 / 8 at each delta, 0-0.111 / 0.086-0.147, 4 and 4 after correction. Recounted from `013/check_b1.md`.
- Row 36 (`b1w`, grid): 1 / 2 / 15 and 4 / 5 / 9, 0-0.057 / 0.086-0.135, 0 and 2 after correction. Recounted.
- Caption: 10,000 and 4,000 draws; 20 plantings by 200 sheets; the pushed label's n_s 100 cells.
- "1 of 378 cells": the seven rows are 48 + 4 + 18 + 168 + 28 + 56 + 56 = 378, with 1 over, 24 unresolved (3 + 0 + 1 + 13 + 2 + 5 + 0) and a largest miss of 0.0595. What they share is described correctly: both deltas come from one set of draws (`scripts/stratppi_baseline.py`), and the reference-rate cells of the three rows use spike 013's seeds (identical misses in `013/validity_H8.md` and `stratppi.md`).
- "5 cells and ... 10" of 36; in-loop 0.116 unresolved and 0.132 over (band ends at 0.1268; `013/inloop.md`).
- Reading 1: one harm label, 1.9%, n_s 100, both at 0.126; the pattern sentence agrees with `results/paper/wilson_exact.md`.
- Redrawn paragraph: 1.4% checkpoint, n_s 200, 0.055 (0.0555) and 0.064; 3 of 12 at 0.10, up to 0.138 and 0.142.

**3. Table D1.** All eighteen rows against the two registrations and `c1`/`c2`. Each wording is a fair short form and each outcome's numbers match (P2a: see L8). Caption (40,000 draws, 10,000 replications, cell, delta 0.05 gains, 5% to 95%) matches. Commits `9fb817d` and `d56feb6` are the two registrations.

**4. Section 8.1.**
- Strata rule: one stratum on three labels, two or three on the others at H = 8; 73-97%; "1.2 to 2.3 against 1.4 to 2.5" (1.16-2.33 against 1.39-2.45, and lower in all 12 cells). Recomputed from both `.xz` files.
- Pre-flight: Spearman 0.842 and 0.830 over 432 cells, 0.708 under the registered rule; median error 0.293 and 0.021; the two registered criteria (`013/DESIGN.md` H3); 2% to 20%, up to 30%, 18% to 98%. Recomputed.
- Against StratPPI item 3: "0.4 to 0.8", "2.6-2.7 against 1.8-2.2" (`sv` section 5; see M5 for the parenthesis).
- Item 4: 0 of 28 at either delta, 3 unresolved on C2:refusal, 1 of 12 rare at 0.10 (0.126); 7 and 6 of 28; medians 2.16, 1.92, 1.99; 2.6 and 3.6; 1.55 and 5.47.
- Pushed label: 2.1-2.2 (2.05-2.16), 0.92, 0.4 to 3.4 points (014 `results.md`).
- Confirmation bullets: every count and miss, including 0.110 at delta 0.10 and n_s 200, at most 0.039 / 0.082 and 0.047 / 0.088, and gains 1.31, 1.32, 0.75, 0.89.
- "about three quarters": (1 - 1/1.93) / (1 - 1/2.90) = 0.74.
- Three things follow: 1.5 to 5.0 (1.47 to 4.98, `tp`); 75% to 86%; 1.06 and 1.32 against Clopper-Pearson (recomputed); "1 cell of 26" with 4 strata (0.0618, C2:unsafe, n_s 100); "the only stratified limit ... over in no mid-rate cell"; the tool refuses rates outside 5% to 95% (`013/preflight.py` line 51).

**5. Section 7.2.**
- Adversarial splits: eleven rules, 2,000 seeds, a true-positive-rate difference, Clopper-Pearson 1 in 2,000 in one cell, "too conservative to see a leak", tight Wald test at delta 0.05 with a largest uncovered rate of 0.0550 (012 `README.md`, `splitlab.py`, `results.md`).
- Training loop: 4 strata, 500 runs, Wald-t with the sampled-pool term 0.090-0.132, one of four over.
- Synthetic sweeps and seed counts against the training paper (sections 6.3 and 6.10): Student-t in the size sweep, 40% to the safety set, 3 of 3, 3 of 3, 1 of 3, 0 of 3, 2 of 3; seed 1 returned NSF.

**6. Sections 8.2, 9 and 10 (fork).** 9.1 (99.6%; bound of 1 in at least 98% of draws; median 1 for four wordings, recomputed); 9.2 (Fisher p, counts, recall 17/18 and 16/18, 0.99 and 0.98 weighted); 9.4 (10 of 84, at most 0.003, recomputed); 9.5 (98%, 0-99%, 7 of 18); 10.1 (three shapes; AUC 0.38, 0.31 / 0.79 / 0.30 on 10 / 2 / 2 stops, 0.01 to 0.37, under 0.2 notes, 0.42 to 0.49); 10.2 (task counts summing to 19,380; 9% and 16%; under 1%); 10.3 (kappa 0.917; -0.00051; GPT-4 0.71 against 0.67 and 0.76 against 0.68; 0.46); 10.4 (0.045; 0.051, 0.052, 0.063; margins); 8.2 (five rates; rho^2 0.02; 10 of 10 and 4 of 10; `b1w` ahead at 20% with 100 labels); Appendix A and B (5 to 7 of 8 strata; none of 28; "called 90 answers", recomputed).

**7. Sections 4 to 6, 11, 12, 13.**
- Worked case: 84 of 500, Clopper-Pearson 0.198 at delta 0.05, threshold 0.206, step 175.
- Section 5: 27 of 28 at 5% and delta 0.05 with injection tasks fixed; one training run supplies 9.2, 10.4, the pushed label and the confirmation policy.
- Section 6.4's two new sentences on how `w` is measured.
- Section 11: "a second implementation ... in its general form" (`sv` section 1); the `overheat` reading (019 `results.md`); Wald-t "10 of 36 cells", "one cell of four", not run in a loop with 8 strata or a language model; harm sheet of 225 with 3 harmful.
- Section 12: 26 of 28, 27, none; 71 or 97 by 27 to 35; "2 of 16 ... one of them at a 98% rate" (see L9).
- Section 13: "1.3 to 1.5 ... at rates of 12% and 46%"; "no exact bound was over its level on data sampled as it assumes".

**8. Appendix D rows (fork).** Every hash, date and order holds in `git log` and the remote reflog: `9e1b7b4` before `1afe911`; `3f89e98` with the note at 08:30 and the scoring log ending 08:36; `cb50927` absent from the remote until `56721e6`, 42 minutes later; `d56feb6` before generation, with M2, M3b, M5's floor (0.8 x 2.12), M8 and the unpredicted `b1w` term as the row says; `f1582e5` replaced by `9e45371` and `e076974` after section 10.3's labels; `a5bd04b`.

## Could not check

- The 42 cells and the "3 of 42" of the adversarial-split paragraph were taken from `validity_recount.md`; I confirmed the rules, delta and the largest value, not the count.
- "10 of 10 seeds certified and none ended over its threshold" (brevity): the training paper's section 6.5 was not re-read; the change is wording only.
- Bonferroni counts were recomputed for the two grid rows only; rows 32 to 34 have no cell over.
- "Three procedures ... (Appendix C)" in section 7.1 was not traced beyond the sections it names.
- (fork) "The paper gives none" (Fisch et al.'s floor) and the many-clusters remark on Cameron, Gelbach and Miller: no stored copy of either paper.
- (fork) `a5bd04b` "with the design check unchanged", that `cb50927`'s script had not been run before its commit, and the 08:30 time of the 9.2 note rest on the records' own statements.
- (fork) "partly mechanical" for the effort AUC is an interpretation in the 019 README.

## Two things outside the changed text

- `013/preflight.py` still answers "YES: ... use b1w" (lines 57-59). Line 1070 cites the tool; lines 1095-1098 now recommend the Wald-t limit.
- Section 9.2 (unchanged): by the standard of the new 9.1 sentence, the side-effect run's "carried" bound returned 1 in 90% and 77% of draws under wordings 1 and 2 (fork, recomputed).
