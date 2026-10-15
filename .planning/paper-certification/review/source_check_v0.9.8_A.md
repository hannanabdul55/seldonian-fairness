# Source check A: Abstract, sections 1 to 7, Appendix A

Manuscript: `reports/paper_certification_clean.md` (line numbers are the clean copy's). Nothing in the repository was edited.

**How it was checked.** Table 4 (31 rows) against `results/paper/validity_recount.md` section (b), and Figure 1's CSV (38 rows) against the same: every miss range, count and upper limit agrees. Independently of the recount script I recounted from raw files: Table 4 rows 6 and 16 and the "8 of 528" sentence (`results/spikes/013/real_plasmode_randomties.json.xz`), all 30 entries of Table A1 (`results/paper/stratppi.json`), and every cell of both confirmation runs (`results/paper/confirm/confirm.json`, `confirm/mid/confirm.json`). I did not run `scripts/validity_recount.py`, because it rewrites files in the repository.

**Registration dates.** The abstract's "registered in advance" holds. R5: commit `9fb817d` 2026-10-08 00:09:14 -0700, pushed 00:09:16 (reflog of `origin/llm-seldonian-rl`); generation finished 01:14 after about 65 minutes of sampling, so it started after the push; results committed 02:25 (`fa5d82b`). R5b: commit `d56feb6` 17:01:34, pushed 17:01:37; K4 generation finished 17:24 after about 21 minutes; results 18:05 (`31b5e60`). The re-mix preview (14:15) precedes the R5b registration, which the abstract discloses ("ten were written after a trial run").

---

## HIGH

### H1. Table 4 does not hold every bound the paper uses: the stratified Wald-t limit has no row
- **Line 606:** "Table 4 applies that rule to every bound we use." Also lines 321-322: "with every bound we use measured under one rule (sections 7 and 8, Table 4)".
- **Source.** Table 4 and `reports/figs/fig1_validity.csv` have no row for the stratified Wald-t limit (`b1`). It is the limit section 8.1 recommends ("the limit to use is the Wald-t one with 8 strata", lines 953-954) and the abstract's "approximate formula" (lines 27-28). The recount already classes it (`validity_recount.md` section (a)), and its record is mixed:
  - stored cells: 0 / 0 / 28 at both deltas;
  - redrawn with replacement: 0 / 0 / 28 mid-rate and 0 / 0 / 12 rare, at both deltas;
  - prompt source, with the term: 0 / 1 / 27 at delta 0.05;
  - synthetic i.i.d. grid: 5 / 5 / 8 at each delta (10 of 36 over, largest 0.111 and 0.147);
  - training loop: one of four cells over (M2 below);
  - confirmation run, prompt source, rare label K2:unsafe, n_s 200, delta 0.10: 0.1099, over (`confirm.json`, `twophase`);
  - with 4 strata: 1 of 13 over (0.062).
- **Why it matters.** A reader takes Table 4 and Figure 1 as the inventory of what holds. The limit behind the abstract's headline is missing, and its failures are spread over three sections.
- **Suggested.** Add rows for the Wald-t limit (stored, redrawn, synthetic grid), or write: "Table 4 applies that rule to every bound we use except the stratified Wald-t limit, whose cells are in section 8.1 and in the third reading below."

---

## MEDIUM

### M1. Section 7.2, adversarial splits: the two sentences count different events
- **Lines 740-742:** "Clopper-Pearson missed in at most 1 of 2,000 runs in any cell. A tight Wald test on the same runs was unresolved above delta in 3 of 42 cells, with a largest miss of 0.055 at delta 0.05."
- **Source.** `.planning/spikes/012-rerandomized-split/splitlab.py` lines 447-460. The Clopper-Pearson count is `miss_proj = (ub_proj <= TAU and gap_true > TAU)`: the run passed and truly violated. The Wald count is `uncovered = (gap_true > ub)`: the bound fell below the truth, which is section 6.1's miss. The first event is a subset of the second kind, so "1 in 2,000" against "0.055" is not like for like. The README (trail item 4) also says: "Clopper-Pearson is too conservative to see a leak: no true violation in any cell", which is why the Wald test was the instrument. The setup is a TPR-difference constraint on a classifier, not a rate.
- **Suggested.** "With the project's Clopper-Pearson test, a policy that passed and truly violated was returned in at most 1 of 2,000 runs in any cell; that test is too conservative here to show a leak. The sensitive check is a tight Wald test on the same runs, whose bound fell below the truth in at most 0.055 of runs at delta 0.05 (unresolved above delta in 3 of 42 cells)."

### M2. Section 7.2, training loop: the recommended limit's cell over its level is left out
- **Lines 746-749:** "the stratified `b1w` test [...] missed 0.084-0.116 at delta 0.1 (Monte Carlo standard error 0.013): three cells at or under delta and one unresolved above it."
- **Source.** `.planning/spikes/013-stratified-safety-set/inloop.md`, rule `strat_ref`, bound `b1_strat_pop` (the stratified Wald-t, population target): 0.132, 0.090, 0.096, 0.098. At 500 runs the band ends at 0.1268, so the first cell is over. The recount lists these cells as "not quoted in the paper". The sentence is correct for `b1w`.
- **Suggested addition.** "The stratified Wald-t limit missed 0.090-0.132 in the same runs, over its level in one of the four cells (0.132 against a band that ends at 0.127)."

### M3. Abstract: "gained little or nothing" covers losses, and the 98% label is not mentioned
- **Lines 30-32:** "It gained little or nothing at 82% or for rare behaviours (under 4%), where exact methods remain the ones to use."
- **Source.** `confirm.json`, large pool, delta 0.05, Wald-t (the abstract's formula):
  - 82%: 0.86 and 1.13; for the prompt source 0.75 and 0.89. Three of four values are below 1.
  - 98% (K3:refusal, not in the abstract): Wald-t 0.11 and 0.26, `b1w` 0.55 and 0.63. Section 8.1 says so (lines 943-944).
  - rare labels: 0.86 to 1.20.
- A second point: "where exact methods remain the ones to use" reads as covering 82%. Section 8.1 (lines 953-957) recommends the Wald-t limit near or above one half and the exact limit only for rare or near-one labels.
- **Suggested.** "At 82% it gained nothing or lost (0.75 to 1.13 of a random sample), and at 98% it lost heavily; for rare behaviours (under 4%) it gained nothing, and there exact methods remain the ones to use."

### M4. Abstract: the judge-assisted shortcuts' "promise" is asymptotic, and 24% is the worst cell
- **Lines 19-21:** "Two published shortcuts that let an automatic judge stand in for some labels did not keep that promise in our implementation: in our main tests they were wrong as often as 24% of the time."
- **Source.** 0.2412 (PPI++, `[017 B6]`) and 0.2387 (StratPPI, `stratppi.md` part B) are both the cell at a 1.3% rate with 225 labels, about 3 positives. Pooled misses are 0.078 and 0.098. At a 20% rate the misses are 0.053-0.112. `stratppi_validate.md` line 276: "It is a finite-sample, one-sided result at rates of 1-20% and strata of 6-200 labels. The paper claims asymptotic coverage [...] nothing here contradicts a statement the paper makes." Section 11 (line 1400) keeps that condition; the abstract drops it. Section 9.4 (line 1083) gives the cell.
- **Suggested.** "Two published shortcuts that let an automatic judge stand in for some labels are guaranteed only for large samples. At safety-test sizes (100 to 1,000 labels), in our implementation, they were wrong more often than 5% in most of our tests with a judge's score, and in 24% of trials in the worst one (a 1.3% rate with 225 labels)."

### M5. Abstract: the 420 pairs are for a normal-type limit with labels alone
- **Lines 48-50:** "if training left the refusal rate unchanged, about 420 labelled prompt pairs would give an even chance of certifying that it rose by no more than 2 points."
- **Source.** `results/paper/p9_budget.md`: 416 is the column "labels alone, paired by prompt", margin 0.02, 50%, "One-sided test at delta 0.05, normal-type limit", pairing correlation 0.66 taken from the guard's flags. The same file: "The betting bound that P9 uses for labels alone is wider than the normal-type limit". Appendix C (lines 1625-1626): the exact bound "is about twice as wide". So 420 belongs to an approximate limit of the kind the paper warns against. A limit twice as wide needs roughly four times the pairs by square-root scaling; that figure is my inference and is in no file. Section 10.4 states "normal-type limits" in Table 10's caption.
- **Suggested.** "[...] about 420 labelled prompt pairs would give an even chance of certifying that it rose by no more than 2 points with an approximate limit; the exact limit is about twice as wide."

### M6. Abstract and Table 4 row 28: the sheet failure is shown for the judge-assisted estimate only
- **Lines 44-45:** "And a labelling sheet that deliberately over-samples some responses cannot be read as a random sample." Table 4 row 28 (line 653): "stratified sheet read as an i.i.d. sample | fails | PPI on the sheet".
- **Source.** `.planning/spikes/017-calibration-carrying-certificate/harm.md` section 3. The route that fails is "PPI as i.i.d." (7 of 18 over, up to 0.989). The route "sheet as i.i.d." (labels alone, Clopper-Pearson) missed 0.000-0.008 in all 18 cells. The recount marks that route "not quoted in the paper". In this study the labels-alone reading is conservative; no file shows it missing.
- **Suggested (abstract).** "And a labelling sheet that deliberately over-samples some responses cannot be read as a random sample when a judge is used to extend it: the estimate then missed in up to 99% of trials."
- **Suggested (row 28).** Rename to "PPI on a stratified sheet read as an i.i.d. sample".

### M7. Section 7.1, redrawn with replacement: the rare-label count is at delta 0.05 only
- **Lines 715-716:** "Each of the two is over in 1 of the 12 rare-label cells, the 1.4% label at n_s 200 (0.055 and 0.064)."
- **Source.** `results/paper/replacement_check.md`, large pass, rare labels: `b1w` 1 / 0 / 11 at delta 0.05 and 3 / 0 / 9 at 0.10 (largest 0.138); pooled Wilson 1 / 0 / 11 and 3 / 0 / 9 (largest 0.142). The delta is carried over from two sentences earlier, and the 0.10 counts appear nowhere in section 7.1.
- **Suggested.** "Each of the two is over in 1 of the 12 rare-label cells at delta 0.05, the 1.4% label at n_s 200 (0.055 and 0.064), and in 3 of 12 at delta 0.10 (up to 0.138 and 0.142)."

### M8. Table 3 lists human harm labels as gold where the validity study's labels are planted
- **Line 586:** "judge routing (8.2) | 500 real responses; a harm sheet of 225 from 4,800 | gold: guard's refusal field, a word count, 225 human harm labels; judge: a rubric | 100 to 1,000 responses | pool mean | 0.05".
- **Source.** Section 6.2 (lines 424-425): "The labels in this study are planted on the real responses at a known rate; they are not an annotator's." `harm.md`: the 225 human labels (3 positive) give one real estimate with no known truth; the validity cells use planted labels and the planted rate as the truth. The row's "pool mean" is wrong for the sheet. The working draft's open check 9 names this conflict; it is still in the clean copy.
- **Suggested.** Gold: "guard's refusal field, a word count; on the harm sheet, labels planted at 1.3%, 5% and 20%". Truth: "pool mean; planted rate for the sheet".

---

## LOW

### L1. Section 7.1: the sheet cell's 4,000 draws are 20 plantings of 200 sheets
- **Lines 660-661:** "`b1w` on a design-weighted sheet, 0.0595 at 4,000 draws." Caption, line 613: "Draws per cell are 4,000 or 5,000".
- **Source.** `harm.md` section 3: "re-drawn 4,000 times (20 plantings x 200 draws)"; recount caution (2) and section (e): each planting has its own truth, per-planting counts were not stored, and the binomial standard error treats them as 4,000 independent draws. This affects rows 8 and 28.
- **Suggested.** Add: "(20 plantings of the labels by 200 sheets each; the standard error treats them as 4,000 independent draws)".

### L2. Appendix A mixes two cell definitions in one comparison
- **Lines 1568-1570:** "the estimator is over its level in 6 of 26 cells under the oracle allocation and in 11 under the heuristic, against none of ten under proportional allocation."
- **Source.** 26 are cells by checkpoint; ten are largest-over-checkpoints cells. By checkpoint, proportional allocation is 0 of 28 (Table 4 row 11).
- **Suggested.** "[...] against none of 28 under proportional allocation."

### L3. First reading against the exact paragraph on rates under one half
- **Lines 671-672:** "a small excess at rates under one half, a larger one above". **Lines 706-707:** "Averaged over rates, the miss is under delta below one half and over it above."
- **Source.** `results/paper/wilson_exact.md`, n 100, delta 0.05: mean 0.042 over 5-20% (over delta at 26% of rates, largest 0.063); mean 0.056 over 80-95%. Redrawn, `b1w` is over in none of 16 cells under one half.
- **Suggested (reading 1).** "under one half the miss is below delta on average and above it at some rates (up to 0.063 at n 100); above one half it is above delta on average".

### L4. One label under two names
- **Line 671:** "the 1.9% label at n_s 100". **Line 716:** "the 1.4% label at n_s 200".
- **Source.** Both are `C2:gated`: 0.019 at step 200 and 0.0138 at step 0 (`replacement_check.md`). A reader will count two labels.
- **Suggested.** Name the label once, with its range (1.4% to 2.4% across checkpoints), and give the checkpoint each time.

### L5. Table 2: "beside every certificate"
- **Line 253:** "yes; the solution rate is reported beside every certificate".
- **Source.** Section 5, lines 340-342: "a published trace has one sample and one answer"; draft open check 5 says the same.
- **Suggested.** "yes; the solution rate is reported for every trained policy".

### L6. Section 6.6: one Table 2 entry used the 4B guard
- **Lines 594-596:** "except in the results cited from the companion paper (Table 2, [...]), which used the 0.6B model."
- **Source.** Table 2's "0.141 against 0.168" and section 4.4's worked case are spike 014's run, trained and tested with Qwen3Guard-4B (`.planning/spikes/014-pushed-label-stratification/gen014.py` line 132).
- **Suggested.** "(the companion paper's entries of Table 2, [...])".

### L7. Section 5: "in two places only"
- **Lines 353-355:** "This paper needs training in two places only: the policy whose human-terms certificate section 10.4 costs, and the negative result of section 9.2".
- **Source.** Section 6.2 (lines 410-411): the per-prompt pools hold "further responses of each trained checkpoint" (steps 100 and 200 of two runs). `R5_registration.md`: the confirmation pools sample "spike 014's returned adapter (step 175)". Sections 7.1 and 8.1 therefore also rest on trained policies.
- **Suggested.** "Trained checkpoints are resampled as fixed policies in sections 7.1 and 8.1. How a policy was trained matters to the result in two places only: [...]"

### L8. Abstract: the exact-method sentence rests on Clopper-Pearson alone, and "rules out" is one interval
- **Lines 17-19:** "Standard exact methods [...] were wrong in at most 5.2%, and the test rules out anything above 5.9%."
- **Source.** `validity_recount.py` lines 837-841: groups `017_classical` (42 cells) and `rep_cp_mid_0.05` (28 cells), Clopper-Pearson at delta 0.05 only; 0.0515 and 0.0588. The betting, Bentkus and Hoeffding bounds were run at delta 0.10 only (Table 4 rows 3-5). 5.9% is the largest of 70 per-cell 95% upper ends, uncorrected.
- The sentence does hold on the other Clopper-Pearson cells at delta 0.05: two-phase 0.0500, rare redraw 0.0471, both confirmation runs at most 0.0462.
- **Suggested.** "The standard exact method for yes/no labels (Clopper-Pearson) [...] was wrong in at most 5.2%, and a 95% interval for its worst case ends at 5.9%."

### L9. Section 6.4 does not say the confirmation gains are measured from the truth
- **Lines 533-535:** "In the comparisons with StratPPI (Table A1 and the medians of sections 8.1 and 8.2) `w` is measured from the truth to the limit".
- **Source.** The abstract's gains (1.5 to 1.7 and so on) use the `excess` field, truth to limit (`validity_recount.py` line 996; R5 registration, P3: "from the truth to the limit"). They are not comparisons with StratPPI.
- **Suggested.** Add "and in the confirmation runs of section 8.1" to the list.

### L10. Section 7.2: the sample-size sweep used the Student-t bound
- **Lines 731-733:** "In two one-way sweeps of that variant it stayed at or below 0.002, across reward pressures 0 to 4 at 1,000 prompts and across sample sizes 200 to 5,000 at pressure 1."
- **Source.** `reports/paper_seldonian_llm.md` section 6.3(b): "Sample size, pressure 1, t bound"; n is total prompts with 40% in the safety set, so safety sets of 80 to 2,000. Table (c) states no bound and no trial count per row ("every row is 500-1,000 independent trials"); the recount leaves tables (a), (b) and (d) unclassed for that reason.
- **Suggested.** Add "(with the Student-t bound; 500 to 1,000 runs a row)".

### L11. Table 4 row 9: "approximate (second order)"
- **Line 634.** No source or condition is given, and section 6.3 says only "It is approximate." Second-order accuracy of a bootstrap-t limit needs a non-lattice condition that 0/1 labels with a 0/1 judge feature (`f01`, one of the four features in these 168 cells) do not meet.
- **Suggested.** "approximate".

### L12. Appendix A: "independent rewrite"
- **Lines 1535-1536:** "Our implementation agrees with an independent rewrite of their Algorithm 1 to rounding error".
- **Source.** `stratppi_validate.md` line 25: "(a) Algorithm 1 written a second time in its general M-estimator form", by the same authors.
- **Suggested.** "agrees with a second implementation of their Algorithm 1 that we wrote in its general form".

### L13. Section 7.1: "on the same draws as the rows above them"
- **Lines 720-721.**
- **Source.** True for the reference-rate cells (spike 013's seeds, replay difference 0). The judge-logit cells draw within strata of the judge's logit in `scripts/stratppi_baseline.py` part B (28 cells); the PPI++ rows above (168 cells) are spike 017's own plasmodes. Same pools, different draws.
- **Suggested.** "on the same pools as the rows above them, and for the reference-rate strata on the same draws".

### L14. Figure 1 CSV: mislabelled column
- `reports/figs/fig1_validity.csv`: the column headed `source` holds the upper-limit strings (for example "0.092 (0)"), not a source tag. The numbers are right.

### L15. "Breach" outside the abstract
- "Breach rate" appears only in the abstract (lines 11, 14, 28), as intended. **Lines 753-755** use "breached the threshold" and "no breach" for a training run that ended over its threshold, a different sense from the abstract's definition.
- **Suggested.** "ended over the threshold" in section 7.2.

### L16. Table 2: "that seed" has no clear antecedent in the clean copy
- **Line 251:** "removed by the fresh draw; that seed returned NSF".
- **Source.** The working draft names "Round 6, seed 1", which the clean copy replaces by "one run of the companion paper". The nearest "seed" is then "another seed", which was returned, not refused (`paper_seldonian_llm.md` lines 795-803).
- **Suggested.** "the first of those runs returned NSF".

### L17. Table 4 row 7 has n_s 200 only
- **Line 632.** `stratppi.json` holds the pushed label at n_s 100 as well (`b1w` largest 0.024 at delta 0.05, not over), and Table A1 prints it. No class changes; the 322 of the third reading would be 326. This is the draft's open check 10.

---

## Universal claims checked and holding

- Line 18, "at most 5.2%": holds on every Clopper-Pearson cell at delta 0.05 I could find (see L8 for scope).
- Lines 27-28, "wrong no more often than promised at breach rates of 12%, 46% and 82%": all 24 Wald-t cells on the three refusal labels are at or under delta (not merely not over), at both deltas, both sizes, large pool and prompt source.
- Lines 34-35, "Fifteen of eighteen [...] two failed and one could not be tested; ten were written after a trial run": P1 and P4a refuted, M3a untested, 10 in R5b.
- Lines 36-37, "one small model, one benchmark's prompts": Granite-3.3-2B; OR-Bench 80K, hard-1K and toxic (`pools_meta.json`).
- Lines 42-44, "27 of 28" and "at most one of the 28": recount item 19 (scheme (a) 27 / 0 / 1; schemes (b) and (c) 28 of 28; one pipeline certifies under the user-task bound, none under the pigeonhole bootstrap-t).
- Line 376, "One rule is applied to every bound": true of every row of Table 4.
- Lines 658-659, "no cell stays over its level after the correction" and "0.067 [...] 0.106": hold.
- Lines 659-663, exactly one valid-row cell over, 0.0595; the other run 0.0485; together 0.054, unresolved: hold. The "9 settings run twice" is right: `b1w` ignores the judge feature.
- Lines 663-665, "Every row in bold has at least one cell over, and every one but the quadrature bound keeps one after the correction" and "Most also have cells that are not over": hold for all 17 bold rows.
- Lines 669-671, neither bound over in any of 24 rare-label cells at delta 0.05; two at 0.10, both 0.126: hold in the raw file.
- Lines 679-682, Student-t above delta "at 96% or more of 79 rates [...] at both deltas", Clopper-Pearson "at none": hold (`binomial_rows.md`; minimum 96%).
- Lines 683-695, 1 of 322 over and 24 unresolved; "8 of 528 [...] all at n_s 100 on the two labels at 65% and above"; 7 of 28 redrawn, by at most 1.2 points; bootstrap-t StratPPI and Wald-t over in none; 5 and 10 of 36: all hold. The 528 covers both deltas.
- Line 709, "Clopper-Pearson's miss probability never exceeds delta": holds (largest 0.0500 and 0.1000).
- Lines 712-717, 7 of 28 all at 65% and above, none of 16 at 9-18%, 6 of 28 with two at 15-18%, Clopper-Pearson "over or unresolved in none": hold in the 40,000-draw pass.
- Lines 729-736, 0.000-0.002 under four bounds against 0.830; filter-only 0.028 and 0.004; "held at the judge's level in every row"; 0.059: match the companion paper's tables (a) to (d).
- Lines 761-763, "at or under delta in all 10 cells", 0.025-0.100, resolves only above 0.14: hold. The 65% and 50% match spike 004's README.
- Appendix A: Table A1, all 30 entries, with 0.0558 and 0.0564 either side of 0.0562; "3 of 10" at delta 0.1; "every one of the four rare-label cells"; 9 and 11 of 26; 6 and 11 of 26; the 3-35%, 8% and 14% comparisons; the sweep (1.77 at H 8 and 1.60 at H 16 for n_s 100; absent at H 4 and at n_s 200); Wald-t "all ten cells (largest miss 0.036, median ESS 2.06)"; 1.97 against 2.10; 26 of 28 under each library's conventions, 28 and 24; 13 of 14 from 0.056 to 0.187; 14 of 28; "5 to 7 of the 8 strata"; the whole 0.90 paragraph (992 and 2 labels, 19%, 1.0-1.5%, 91%, 59%, 68% to 85%): all hold.
- Formulas of sections 4.3, 6.3 and 6.4 match the code (`seldonian/llm/rewards.py`, `stratbounds.py`, `cert017.py`, `scripts/agentdojo_twoway.py`, `preflight.py`). The arithmetic holds: 299 / 149 / 59 labels; 37%; over from 0.057; zero-count limits 2.6%, 1.3%, 1.6%; `exp(-z^2)` 0.067 and 0.194.

## Claims I could not trace to a stored source

- **Table 2, line 251, "erred the other way, by 0.01-0.02".** Only the companion paper's sentence supports it (line 803). `results/llm_r6` is not in the working tree. The sentence beside it compares point estimates (0.133 against 0.184, a gap of 0.05), while 0.01-0.02 looks like a gap in upper bounds; the companion's limitations section says "0.01-0.05". The draft's open check 4 covers this.
- **Section 7.2, synthetic environment.** Trial counts per row for tables (a), (b) and (d) are not stored, and neither is the bound used in the pressure sweep.
- **Line 219, Thomas et al. (2019) "use the safety-set size with `kappa = 2`".** No local copy; the companion paper says only that Round 1 "inherited a factor of 2 from the classification code".
- **Appendix A.** "StratPPI's authors have released no code" rests on a search recorded in `stratppi_validate.md`. Two descriptions of Fisch et al. have no local source: "the paper asks only for a positive share" (`stratppi_validate.md` line 107 says instead "The paper gives no floor and its rule can return no label for a stratum") and "the heuristic becomes too aggressive at larger n when the judge's confidence is not calibrated".
- **Line 73, Bowyer et al. (2025) on normal approximations at small rates.** Not checked against the paper.
- **Table 4 rows 1, 3, 4, 5 and 15.** The draws are lost. The printed rates are consistent with an exact enumeration at a pool count of 124 (`binomial_rows.md`), which the caption says.
