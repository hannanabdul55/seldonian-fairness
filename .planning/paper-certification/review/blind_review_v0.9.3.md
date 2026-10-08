*A blind review of `reports/paper_certification_clean.md` (v0.9.3, after the audit's corrections), by a fresh agent that read only the manuscript and its figures (2026-10-07). Verdict: weak accept, mean 3.83 of 5. The inconsistencies of its section 4 that were plain errors were corrected afterwards; the rest are in the draft's open checks.*

# Review: "Certifying behaviour rates of language-model policies: what holds, what a label buys, and what does not carry"

## 1. Summary

The paper takes the Seldonian safety test (a one-sided bound at level delta on a judged behaviour rate, with "no solution found" as an outcome) as a stand-alone certificate for a fixed language-model policy. It audits that certificate against known truths: a synthetic bandit, resampled real responses, and published benchmark traces. It reports that the normal-quantile limits of PPI++, StratPPI and PPBoot run over their level at safety-test sizes, that a bootstrap-t limit with proportional allocation repairs them, and that stratifying on the reference model's per-prompt rate multiplies the effective sample by 1.4 to 5.3. It reports three failures: carried judge calibrations, per-pair bounds on AgentDojo's crossed design, and a stratified sheet read as i.i.d. A human-label certificate of a trained policy is costed and prepared, not run.

## 2. Strengths

- **One stated validity rule for every bound** (section 6.1), favoured or not, with an explicit "unresolved" class.
- **Unusual candour.** The paper reports a code bug that had stood as a finding (7.1 "A correction"), a registered strata rule that was replaced (8.1), a pre-registered criterion that failed (8.1 pre-flight), lost data (11, Appendix D), and a retracted reading of the mechanism (Appendix B).
- **The with-replacement redraw and the exact Wilson enumeration** (7.1) show the authors attacking their own favoured bound.
- **The AgentDojo re-analysis** (9.3, 10.2) is concrete and consequential: four of five per-pair passes do not survive clustering.
- **Table 8 is fully reconstructible.** The per-model counts sum to the pooled counts, and the recall, false-alarm and kappa figures all follow from them.
- **The cost paragraph of 8.1** (4,000 judged reference responses buy about 140 labels) and the "prepared, not run" framing of 10.4 are honest about value.

## 3. Weaknesses

**W1 (major). The top of the headline gain comes from the cell where its bound fails, and applies only to the pool as a finite population.**
- The abstract's 5.1 to 5.3 is the 65-66% harmful-refusal label. That is where `b1w` is over its level once redrawn with replacement (7 of 28 cells, up to 0.062; 7.1, 8.1 item 4).
- Section 6.2 concedes that the without-replacement record is helped by uncorrected finite-population slack.
- The valid alternative in that cell (StratPPI estimator with bootstrap-t, n_s 100) gives 1.77 against 4.74 (Table A1).
- For a claim about the prompt source, 6.4's own cap `1/(1 - G + G n_s/N)` with G about 0.8 and N = 500 gives about 2.8 at n_s 100 and 1.9 at n_s 200. The abstract does not say this.
- Table 5 still routes stratified data to `b1w`.

**W2 (major). Each positive recommendation was chosen on the cells that validate it, with no held-out confirmation.**
- This covers the strata rule and the redefined truth (8.1), proportional allocation and "about 45 per stratum" (8.2), which bound suits which rate (8.1 item 4: "read off the cells it describes"), the grouping behind "1 of 322" (11), and the AgentDojo sampling scheme (10.2, 11).
- Each is disclosed, but the result is that "over in 1 of 322" is an in-sample figure.
- The registrations ("registered in advance", "pre-registered", "wrote down beforehand") cite no registry or timestamp.

**W3 (major). The validity rule under-reports uncertainty and is corrected for multiplicity only where that helps.**
- Replications run from 200 to 40,000, so "not over" means anything from "miss below 0.052" to "miss below 0.14" (7.2).
- "Over in no cell" (abstract) counts unresolved cells as passes (5 of 28 and 12 of 168).
- The paper's only multiplicity correction rescues the one over cell in a favoured row (7.1), while failing rows get none.
- Interval estimates of each miss rate against a stated tolerance would be more informative.

**W4 (major). The evidence base is narrower than the cell counts suggest.**
- Each plasmode family is one pool of about 500 items. The 168 PPI++ cells are 42 draw sets read through four features of one judge, against a guard as gold.
- Section 9.2 is two single-seed runs with a judge whose recall is 0.04-0.54, and the human-label echo is non-significant (p at least 0.18). The abstract and conclusion nonetheless state "training that targets the label" without qualification.
- The Student-t row and four exact rows of Table 4 are "printed rates from the companion paper whose draws were lost", and that paper is not in the reference list.
- The StratPPI and PPBoot failures rest on the authors' reimplementation, including an unexplained 0.90 miss under the heuristic allocation.

**W5 (major). Section 10.1 applies i.i.d. Clopper-Pearson to 120 trials without stating what is sampled.**
- Section 8.1 mentions "a benchmark with 20 trials a task". If that is RoboDojo, the 120 trials are 6 tasks, and assumption A2 (the issue 10.2 turns on) is unaddressed.
- Section 11 says the certificate is "over a benchmark's task distribution".
- RoboDojo-RC has no reference. "Joint effort" and the AUC figures are undefined.

**W6 (major). Section 10.2 concludes that no pre-specified bound holds under two-way sampling without engaging the literature on that problem.**
- Missing: multiway cluster variance (Cameron, Gelbach and Miller), the pigeonhole bootstrap (Owen), and Menzel's two-way bootstrap.
- The quadrature bound that held "in this one check" is essentially the multiway variance and is set aside.
- The clustered bound for Meta-SecAlign ranges from 0.058 to 0.107 across seeds at 4,000 draws. The paper does not give the inner bootstrap size of the validity study or its handling of zero-variance resamples.

**W7 (major, for significance rather than validity). No trained policy is certified in human terms.**
- Section 10.3 shows the guard overstates refusal by about 7 points, with strict false-alarm rates of 0.03-0.44 by model.
- The only human labels on the authors' policies are from one annotator, who is an author.

**W8 (minor). Internal inconsistencies.** These are listed in section 4 below; the stale Figure 1 subtitle is the most visible.

**W9 (minor). Length and focus.** The paper runs to about 16,800 words. Sections 2 to 4 (including GRPO and multiplier details) re-derive background the paper says is not its subject. "No exact bound was over its level" is a theorem being used as a code check, and read literally it conflicts with the "Clopper-Pearson over pairs" row.

## 4. Checks of the statistics

**Verified by recomputation:**
- **Zero-count sample sizes** 299, 149 and 59; and 0.99^100 = 0.37.
- **Rule thresholds:** 0.057 at R = 4,000; the 0.116 in-loop and 0.14 trajectory resolutions; pooled 0.054 "unresolved" at 8,000 draws.
- **Wilson limit:** zero-count limits of 2.6%, 1.3% and 1.6%; misses just above of 0.069 and 0.068; exp(-z^2) of 0.067 and 0.194. My own enumeration at n = 100 reproduces the averages 0.042 and 0.056, the maxima 0.063 and 0.085, and 0.20 near 1.
- **Table 6:** all four Clopper-Pearson limits, 2.5% at zero events, and Fisher p = 0.003 (though the contrast is post hoc).
- **Table 7:** all six per-pair limits, from 7, 21, 21, 47-48, 43 and 300 successes.
- **Section 4.4 worked case:** 0.168 to 0.198 reproduces as Clopper-Pearson on 84 of 500 at delta 0.05.
- **Table 4 row sums:** the 322 / 1 / 23 count, 168 = 119 + 32 + 17, and 528 = 11 x 48.
- **Table 8:** every cell and both kappas, plus the phi-squared values of 0.40 and 0.65.
- **Table 9:** recall and false-alarm rates follow from precision and flag rate.
- **Table 10:** every entry to within 2 pairs under the stated formula; also 23% = 0.6 x 190/490, 8,820 responses, and the 2.4-point margin at 300 pairs.
- **Table A1:** medians 2.16 and 2.10, and "ahead in seven cells by 3-35%".
- **Formulas:** PPI++ variance and ESS, the cluster standard error, the direction of the bootstrap-t limit, and the finite-pool cap are correct as written.

**Failed or inconsistent:**
1. **Figure 1's subtitle** says labels at 1-2% are omitted because "no approximate bound holds there". Section 7.1's correction retracts exactly that.
2. **Table 3** lists "225 human harm labels" as gold for 8.2. Sections 6.2 and 11 say sheet labels are planted and that all harm numbers are the guard's.
3. **Section 8.1 item 1** gives "9 of 26" for StratPPI's normal limit under proportional allocation. Table 4 gives 9 / 5 / 14, which is 28 cells, and its caption says proportional rows count all 28.
4. **`b1w` is checked without replacement in 26 cells** (24 + 2, pushed label at n_s 200 only) but in 28 with replacement. Table A1 reports a pushed-label `b1w` cell at n_s 100 that Table 4 omits.
5. **Section 8.1 item 3** gives medians at n_s 200 of 2.6-2.7 against 1.8-2.2. Table A1's n_s 200 rows give 2.49 against 2.13.
6. **The encoded-refusal label** is 93% (8.1, A1), 94% (Figure 1) and 95% (Table 4 caption). ESS is defined two ways for the same labels (A1 caption).
7. **In Table A1** one 0.056 is bold and another is not, with no extra digit shown.
8. **19,380 episodes** cannot be a sum over 28 pipelines of only 629s and 949s (22/6 gives 19,532; 23/5 gives 19,212). Other sizes must exist and are not stated.
9. **Section 10.4** quotes "about 930" pairs at +0.006. The Table 10 formula gives about 919, and 830 with Appendix C's rates. The 2,100 figure matches.
10. **Section 9.2** gives guard recall of 17 of 18 and 16 of 18. Table 9 gives 0.99 and 0.98 and says no miss was found in the large guard-negative strata. Presumably this is raw against weighted, but the text does not say.
11. **Table 4's "not over at delta 0.05"** rare-rate row is contradicted in the with-replacement text (each bound is over in 1 of 12 cells).

## 5. Questions for the authors

1. What ESS does a valid limit (bootstrap-t or Wald-t) give on the 65% label for a claim about the prompt source rather than the pool?
2. What are the tasks-by-trials structure of RoboDojo-RC and its citation?
3. Which choice in your StratPPI heuristic allocation produces a 0.90 miss? Are there strata with fewer than two labels?
4. In the AgentDojo validity study, what is the inner bootstrap size, and how are zero-variance resamples treated? Is the passing pipeline one of the two unresolved cells?
5. In 7.2, is 0.059 the probability that a returned policy violates? That would still be under delta.
6. Was the 10.3 annotator blind to policy and to the guard's flag?
7. In Appendix B, are the 93 responses all guard-flagged responses, or the surface-pattern subset? 4 of 93 is 0.043, not 0.03.

## 6. Scores

| dimension | score | reason |
|---|---|---|
| evidence relevance | 4 | Known-truth resampling bears directly on the validity claims; the human-terms claim has only a design calculation. |
| falsifiability | 4 | Miss rate against delta under a stated rule, with criteria that visibly failed; "unresolved" softens it. |
| scope calibration | 4 | The body and limits are careful; the abstract's top gain, the unstated population cap and the single-run 9.2 claim over-reach. |
| argument coherence | 3 | Table 5 recommends `b1w` where the paper's own evidence prefers other limits, and a figure contradicts the corrected text. |
| exploration integrity | 5 | Bugs, reversals, failed registrations, lost data and post hoc choices are all reported. |
| methodological rigor | 3 | Single pools, single seeds, in-sample rule selection, no multiplicity control, reimplemented baselines, one author-annotator. |

**Mean: 3.83**

## 7. Recommendation

**Weak accept (borderline). Confidence 3:** I verified the arithmetic and one enumeration, not the simulations.

The changes that would most improve the paper:
1. **Confirm the post hoc rules on a pool that played no part in choosing them.** That means the strata rule, the allocation, and the bound-by-rate choice. Report the ESS under a limit that holds at high rates and for a source-population claim, and revise the abstract's range to match.
2. **Replace the three-class cell count with interval estimates of each miss rate** at a common replication count. Fix the stale Figure 1 and the denominator inconsistencies. Cite, or remove, the lost companion-paper rows.
3. **Cut sections 2 to 4 to a page.** State the sampling unit for RoboDojo, and position 10.2 against the two-way clustering literature, evaluating the quadrature (multiway) bound properly.
