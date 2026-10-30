*A blind review of `reports/paper_certification_clean.md` (v0.9.11), by a fresh agent that read only the manuscript and its seven figures (2026-10-19). Verdict: weak reject, mean 3.58 of 5 (v0.9.10: weak reject 2; v0.9.8: 3.25; v0.9.3: 3.83). The text below is the reviewer's, unedited; what was done about it is in PLAN.md.*

# Blind review: "Certifying behaviour rates of language-model policies: what holds, what a label buys, and what does not carry"

Reviewer expertise: confidence bounds and finite-sample certification, Seldonian / safe RL, LLM evaluation.
Material read: the full clean manuscript (L1-L2428, all appendices) and the seven figures. All line numbers below refer to `paper_certification_clean.md`. I recomputed by hand or in Python the numbers listed under "recomputed" at the end of section 5.

---

## 1. Summary

The paper takes the safety test of a Seldonian algorithm (a one-sided confidence bound at level delta, computed on a held-out sample, with "no solution found" as a permitted answer) and proposes using it on its own as a *certificate* that a language-model policy's rate of some judged behaviour (refusal, harm, attack success, a robot safety stop) is below a threshold. Most of the paper is an empirical audit of that certificate's four assumptions (selection, independence, bound validity at small n, label = quantity of interest). The tests are resampling studies where the truth is known: a synthetic bandit, "plasmodes" that redraw real guard-labelled responses from small Granite-3.3-2B / Qwen2.5 pools, and redraws of AgentDojo's published tables. The findings are as follows. Exact bounds (Clopper-Pearson, betting) hold. Normal-quantile PPI++/StratPPI, Student-t and per-pair bounds on crossed designs do not. A heuristic stratified design (8 strata of the reference model's 8-sample per-prompt rate) with an approximate stratified Wald-t limit gives width-based effective-sample-size gains of about 1.5-2.6 for the pool's own rate at mid rates, 1.3-1.5 for a claim about the prompt source, about 1.2 on a costly human label, and nothing at rare rates, near-one rates or for exact bounds. A rubric-compiled judge's recall/false-alarm calibration does not carry across prompt populations or across training that targets the label, whereas WildGuard's does carry across training. On AgentDojo, one of 28 pipelines certifies a 5% attack rate if injection tasks are taken as fixed, and none does if they are treated as sampled. A human-label certificate of a trained policy is costed (about 420 prompt pairs for an even chance at a 2-point margin) but not run. Of 72 registered predictions, 53 held, 18 failed and one was untestable.

## 2. Scores

| dimension | score (1-5) | justification |
|---|---|---|
| Evidence relevance | 4.0 | Nearly every claim points to a table row, a count or a registered prediction, and the counts I checked recompute (Table 4 row counts, Table 8 marginals, Tables 6/7 CP limits, Table 10 budgets, the 72/53/18/1 tally). Points off: one headline abstract number (5.3%) cannot be found in the body. Figure 1 is cited for a statistic it does not show. Several results (Table 2, LM seeds of 7.2) rest on an unpublished companion paper whose per-draw data were lost (L1887-1896). Certificates of trained policies are in a guard model's terms only. |
| Falsifiability | 4.0 | Validity is defined operationally (L415-443) with one over/unresolved/at-or-under rule applied to every bound, and 72 predictions were written down with scoring code. Points off: most registrations have no independent timestamp, or reached the remote only with the results (L2252-2255). Ten predictions were written after a dry run in which all nine testable ones already held (L1056-1058). Several "predictions" are near-tautologies for exact bounds (X1, W1, Z1, Z2, Y6, the CP halves of P5/M9). |
| Scope calibration | 3.5 | The abstract and limits are unusually hedged ("in our implementation", "for the prompts sampled", "only one training run ... moved a model far"). The ambition is modest and matches the effort. Remaining over-reach: the untraceable 5.3% (L18). An "at most a quarter" that splices a post hoc threshold into a registered result (L37-38). The 90% StratPPI failure is attributed to the published allocation rule although it depends on the authors' own 2-label floor (L2190-2191). "the only stratified limit" (L1091) ignores the exact stratified bounds. The motivating example ("harmful answer to at most 5%", L8-9) sits in the rare-rate regime where the paper's constructive method gains nothing. |
| Argument coherence | 2.5 | There is no single thesis. The title has three clauses, and the paper bundles a bound-validity audit, a stratified-design method, a judge-calibration negative result, two benchmark case studies, a guard-vs-human comparison and an unrun budget calculation. Four ESS variants (L605-616) and a 55-row Table 4 mixing registered and unregistered rows make the main line hard to follow. The ~600-word abstract has a misplaced sentence (L38-40). The rule summary in 8.1.5 has overlapping bullets. Unexplained internal tags ([B6], [O2], [B25 sensitivity] ...) and build notes are left in. |
| Exploration integrity | 4.5 | This is exceptionally candid. It reports refuted predictions, rules chosen on the data (L944-949, L1817-1819), deviations (the seed retry without the registered decision, L2261), the post hoc reinterpretation of run B3 against its registration (L1136-1140), the lost companion data, the implementation bug at end points (L1876-1886) and abstention as part of the bootstrap record (L1537, L1562-1565). Points off only because some of these candid admissions do not reach the abstract's framing, e.g. "72 predictions registered in advance" with little weight given to how weak several registrations are. |
| Methodological rigour | 3.0 | Strong points: a uniform miss-rate rule, Bonferroni and Clopper-Pearson interval ends per row, exact enumeration where possible, redraws with replacement to remove the finite-pool artefact, exact end-point tests. Weak points: validity of the recommended approximate limit is established on a handful of pools from essentially one model and one guard, and it fails on a synthetic i.i.d. grid (10 of 36 cells, row 35). Cells are heavily dependent (L817-821). Sheet draws (20 plantings x 200) are counted as 4,000 independent draws (L700-701), which understates Monte Carlo error. Bootstrap abstentions count as non-misses. The judge-transfer negative result uses a judge with recall 0.04-0.54. Models are 0.5B-2B. |

**Mean: 3.58.**

**Verdict: weak reject.**

The mean of these six epistemic dimensions overstates readiness for a top venue, because significance and presentation are not among them. On significance, the "proposal" is a standard held-out one-sided bound. The constructive contribution is an approximate heuristic whose gains are modest and do not reach the rare-rate regime that safety certification is about. The certificate the paper motivates itself with (a trained policy, in human terms) is not delivered. On presentation, the manuscript is not in a reviewable form for a general ML audience. I would be glad to see a focused version of this work again (see section 7).

## 3. Strengths

1. **One validity rule for every bound, favoured or not** (L428-443). It includes the largest Clopper-Pearson upper end per row and a Bonferroni count (Table 4, L723-779). This is better practice than most of the PPI and LLM-eval literature.
2. **It finds the finite-pool artefact and corrects it.** Stored plasmode draws take 20-50% of a pool with no finite-population correction, so miss rates come out low. Redrawing with replacement exposes `b1w`'s failures at 65%+ (row 17, L849-858). Many plasmode studies miss this artefact.
3. **The exact Wilson-limit analysis is careful and correct** (L840-847). I recomputed the zero-count limits, the exp(-z^2) peaks, and the result that a 1% policy passes a 0.5% threshold 37% of the time.
4. **The crossed-design result on AgentDojo matters in practice and is sound.** Per-pair CP is over its level for 27/28 pipelines (row 24). The paper separates clearly what is certified when injection tasks are fixed versus sampled, and tests multiway, pigeonhole and model-based truths (L1544-1619).
5. **The registered confirmations include real refutations.** Examples are P1, P4a, V1/V2/V6, Q1-Q3/Q5 and Z4 (Appendix D), and they are reported in the main text. The count 72 = 53 + 18 + 1 reconciles exactly with Tables D1-D4.
6. **The guard-vs-XSTest analysis is clean and internally consistent** (Table 8, Figure 6). The per-model marginals recompute exactly from the pooled counts (159/172, 98/123, 51/955). It also shows concretely that error rates belong to a model (FA 0.03-0.44).
7. **Section 11 states the limits plainly**, including the guard-only nature of every trained-policy certificate (L1814-1816) and the provenance loss (L1887-1896).

## 4. Weaknesses

### Critical

**C1. The certificate the paper is motivated by is not delivered, and the paper's own evidence undermines the ones it does deliver.**
- The opening use case (L8-9, L69-74) is a developer certifying the rate of an unwanted behaviour. Every certificate of a trained policy here is in the terms of a guard model (L62, L1814-1816).
- The paper itself shows that this guard's error rates against human strict refusal are model-dependent: false-alarm rate 0.03 to 0.44 across five models (Table 8, L1690-1694).
- On the authors' own policies the guard overstates human strict refusal by about 7 points, with precision 0.59-0.60 (Table 9, L1714-1723).
- The paper also shows that judge calibrations need not carry across training (section 9.2).
- So a guard-terms certificate of a trained policy says little about the human-terms quantity. The human-terms certificate is "prepared, not run" (L1743, L1777-1799), although its cost is 600 labels.
- As it stands, the paper's positive certificates are about labels the paper argues should not be trusted as stand-ins.

**C2. The constructive contribution does not reach the regime that motivates it, and it has no guarantee.**
- The safety-relevant rates in the paper's own examples are a few percent: harm at 3.4% (L803), attack success at 1.1% (L1630), the 5% threshold in the abstract.
- At rare rates the reference-rate strata gain nothing: 1.01-1.11 (L971, L1038, L1076, L1182). The paper's rule sends such labels back to unstratified Clopper-Pearson (L1242-1244).
- At mid rates the gain belongs to an approximate stratified Wald-t limit. That limit has no finite-sample guarantee and is over its level on a synthetic i.i.d. grid in 10 of 36 cells (row 35, L759).
- It was adopted after the first confirmation pool (L1869-1870), and its strata rule was chosen on the pools that measure it (L944-949).
- Exact stratified bounds give no reliable gain (L1230-1233). For the source claim the gain is 1.3-1.5 (L2024-2025). On the only costly-label test it is about 1.2 (L1173-1174), which pays only above roughly 120 judged responses per human label (L1183-1184).
- The "proposal" itself (L357-361) is a held-out one-sided binomial bound, which is not new (cf. Chaudhary et al. 2024/2025, cited at L1919-1924).
- Net: the paper's novelty is the audit, and the audit's constructive arm is narrow.

### Major

**M1. Validity evidence for the recommended approximate limits is thin and correlated.**
- The "approximate bounds we use are over in 1 of 378 cells" (L811-817) rests on cells that are not independent: delta 0.10 cells reuse delta 0.05 draws, 168 PPI++ cells are 42 draw sets, and the StratPPI/Wald-t rows share seeds with `b1w` (L817-821). The grouping was made after the results (L817).
- The real-response evidence for the Wald-t limit comes from Granite-3.3-2B pools plus one Qwen2.5-1.5B family, with one guard's labels. Its failure on the synthetic grid (row 35) shows that its validity depends on the pool's stratum structure. The paper does not characterise which structure that is.
- At R = 4,000 the 2-se rule cannot detect excesses below about 0.007 at delta 0.05. Row 8's 4,000 draws are 20 plantings x 200 sheets "counted as independent draws" (L700-701), which understates the standard error.
- Bootstrap limits that return no limit count as non-misses. Examples: 9-16% of AgentDojo tables (L1533-1537), and more than 5% of tables for 7 pipelines with the pigeonhole bootstrap-t, up to 77% (L1562-1565). The abstention-adjusted miss rate is never reported.

**M2. The registration record is weaker than the "72 predictions registered in advance" headline (L42-44) suggests.**
- Four records have no independent timestamp, or reached the remote only with the results (L2252-2255). The two-way-bounds registration rests on a local commit time (L2255).
- The ten mid-pool predictions were written after a dry run on a re-mix in which all nine testable ones already held, and three were shaped by it (L1056-1058, L2257).
- Several counted predictions cannot fail for an exact bound: X1, W1, Z1, Z2, Y6, and the CP halves of P5 and M9.
- Run B3 is reported "against our registered reading" (L1136-1140, L2266).
- A crashed seed was retried without the explicit decision the registration reserved (L2261).
- Each of these is disclosed. Even so, substantive confirmatory predictions should be counted separately from trivial and post hoc ones.

**M3. The judge-transfer negative result (section 9.2) rests on a weak judge and one set-up, and the proposed mechanism does not account for its size.**
- The compiled judge's recall on the guard's positives is 0.04-0.54 (L1363), and 0.21 at step 0 (L1427). A judge that misses about 80% of positives is a weak test of whether a calibration carries. The positive control (WildGuard) did carry across training but failed across pools (85% miss, L1410-1412).
- Appendix B's composition argument: with the stated per-pattern recall (10% with the pattern, 25% without, L2216-2218) and a pattern share of 82% after training (L2208), recall would fall to about 0.13, not to the observed 0.03 (L1372-1373, L2209-2210).
- So the sentence "what changed with training is how many responses have it" (L2218) is not supported by the paper's own numbers, unless the per-pattern recall also fell.
- Three seeds share prompts, threshold and gold label (L1415-1416). That is one design.

**M4. The single AgentDojo certificate is fragile, and the paper's own rules contradict it.**
- The one passing pipeline has a rate of 1.1% and passes on 7 successes in 6 user tasks (L1640-1641). That rate is in the rare regime where section 8.1.5 sends labels to exact bounds (L1578-1580).
- The exact bound on the pair rate, "the quantity the certificate is about", does not pass (0.060, L1864-1866). Only the exact bound on the user-task average passes, narrowly (0.048, L1594).
- The clustered bootstrap returns no limit in 9% of that pipeline's resampled tables (L1535).
- Under sampled injection tasks no pipeline certifies with the only bound that held (L1566-1568).
- The headline "one of 28" (L1522, L1637) should read "one of 28 under an approximate bound with injection tasks fixed, not confirmed by the exact pair-rate bound".

**M5. Presentation is not at venue standard.**
- The abstract (L8-65) runs to about 600 words of nested qualifications.
- Four ESS variants (L605-616) and two "power ESS" definitions are used.
- Table 4 has 55 rows mixing settings, registration status and bolding rules.
- Unexplained internal tags appear: [B6] L491/L1598, [O2] L899, [O6] L1099/L1107, [B3] L1128, [O3] L1168, [B25] L1191, [B25 sensitivity] L1226, [B4] L1393/L2215, [O4] L1578.
- Build notes and file paths appear: L3-4, L2038-2047, Appendix E, `transfer017.py` at L2310.
- Commit hashes, "the repository" (L1798) and "the companion paper (by the same authors)" (L2108) break anonymity.
- Section 10.1 adds a plain Clopper-Pearson calculation on 120 trials, and it is not clear what it adds to the paper's argument.

**M6. Dependence on an unpublished companion paper with lost data.**
- Table 2's first run, the three-seed over-refusal results (L909-914), the labels-alone row of Table 3 and the 0.6B-guard results trace to a paper the reader cannot see, whose per-prompt data "did not survive a disk loss" (L1887-1896).
- The authors limit the damage. Even so, these results should be removed or clearly fenced off from a submission.

**M7. The human-label budget (section 10.4) is sized on the wrong policy and with an optimistic guard gain.**
- The rates come from step 200, while the sampled policy is step 175 (L1748-1749, L1700-1702).
- Table 10's "with the guard" columns assume an `ESS_pred` of 2.5 (L1769-1770). The authors' own design check on real guard scores gives about 1.4 (L2236-2237).
- So the 167/380-pair figures are optimistic by roughly 1.8x. The text presents them as the guard "cut[ting] the budget again by 2.5".

### Minor

- **m1.** `b1w` is a heuristic the paper names after its code ("bound family 1", L506-507), argues against (L1257-1260), and still spends a large share of the results on.
- **m2.** The bolding rule in Table 4 is inconsistent. Row 17 (`b1w` redrawn with replacement) is bold, but rows 45 and 54 (the same bound and design, registered) are not, although row 54 keeps 4 cells over after correction (L697-698, L741, L769, L778).
- **m3.** The noisy-judge example of section 7.2 does not show a failure. 0.059 is below delta 0.1, and it is a conditional rate, not the joint probability the guarantee bounds (L876-878).
- **m4.** The trajectory check uses 200 runs per cell, which resolve only misses above 0.14 (L919-920). A Bonferroni-over-T procedure reaching a miss of 0.100 = delta is itself surprising and is not explained.
- **m5.** All human labels on the authors' own policies come from one annotator, an author (L1897-1901).
- **m6.** In section 10.1 the model with ten stops ran last, 4-11 hours into the session (L1507-1508). The label is an interpretation of an `overheat` termination (L1465-1466). The source is a non-peer-reviewed web report (L2088).
- **m7.** The StratPPI implementation is the authors' own (L1840-1843, L2126-2130). The 90% failure uses a floor of 2 labels per stratum that the original paper does not specify (L2190-2191). The abstract (L22-23) attributes it to "an allocation rule published with one of them".

## 5. Plain errors

Each entry gives the line(s), what is wrong, and a fix.

1. **L18. Untraceable number.** "Clopper-Pearson ... was wrong in at most 5.3% of ours." No CP cell in the body reaches 0.053. Table 4 gives a maximum of 0.052 (row 2, L726) and 0.050 (row 13, L737), and exact enumeration "never exceeds delta" (L847, L810). The only 0.053/0.057 values for a baseline belong to the pooled Wilson bound (L1074). *Fix:* "at most 5.2%", or cite the cell that gives 5.3%.

2. **L35-38. Not like with like.** "...the other did so in about half of the real-response tests ... and in at most a quarter of those on the costly human label."
   - At the registered threshold, stratified CP certified more often than CP in 19/36 real cells and in **none** of the 32 costly cells (L1208, L1216-1217, L1232-1233).
   - "A quarter" (8/32) comes from the post hoc step-spread sensitivity, and from a different measure, power ESS > 1 (L1225-1229).
   - Taking the registered result for one half of the sentence and the post hoc one for the other is not like with like.
   - *Fix:* "in 19 of 36 real-response tests and in none of the 32 on the costly human label at the registered threshold (8 of 32 under a threshold chosen after the results)."

3. **L38-40. Misplaced sentence.** "The gain is smaller, 1.3 to 1.5 at the first two rates, when the certificate must cover all prompts..." follows the sentence about exact grouped bounds, so "the gain" reads as theirs. It is the Wald-t limit's gain for the prompt source (L1047, L1071). *Fix:* move it to after L34 and name the limit.

4. **L40-41. Missing conditions.** "about 1.2, which pays only where a human label costs more than about 120 judged responses." The break-even holds only for n_s = 100 on a 250-prompt pool with k = 8 (L1182-1184). The 1.2 is a median whose range is 0.97-1.34 (L1173). *Fix:* state the conditions, e.g. "at a safety set of 100 from a 250-prompt pool".

5. **L784-785. Scope of "remaining rows" makes the number wrong.** "In the remaining rows ... the largest miss rate a cell's interval leaves open is 0.067 at delta 0.05 and 0.106 at 0.10." The sentence follows a discussion of rows 37-55. If "remaining" includes the registered rows not listed as over, row 37's upper end is 0.107 at delta 0.10 (L761). *Fix:* say "rows 1-36 not in bold", or change the figure to 0.107.

6. **L1083-1084. Range does not match the table.** "The same measure gives 1.5 to 5.0 for `b1w` at rates of 9% to 66% on the pools that chose the rule." The same measure (`ESS_truth`, delta 0.05) in Table A1 gives 1.36-5.19 for `b1w` at those labels (L2144-2151). The `ESS_med` values are 1.50-5.14 (L961-962). Neither gives "1.5 to 5.0". *Fix:* "1.4 to 5.2".

7. **L1091-1093. Overclaim with "only".** "the stratified Wald-t limit is the only stratified limit that was over its level in no mid-rate cell on Granite's pools." The exact stratified CP and betting bounds were over in none of 116 cells, including the 58 real Granite cells (rows 47-48, L771-772, L1191-1193). The bootstrap-t StratPPI limit was over in none on the earlier pools (row 11) or the 46% pool (L1067). *Fix:* "the only approximate stratified limit".

8. **L1279 (Table 5 caption), and L1257-1258. Universal "and above" contradicted.** The caption says `b1w` can be over its level "at 58% and above on trained policies". L1257-1258 says "at rates of 65% and above on Granite's pools". But L1260 says "at 98% it is over in none of its 4 cells". *Fix:* "at 58% and at 65-82%; not at 98%".

9. **L1009. Range does not match the cited table.** "at a safety set of 200 or with 4 strata it adds 0.4 to 0.8 of ESS (Appendix A)." In Table A1 at n_s 200, bootstrap-t StratPPI minus `b1w` is +0.80, +0.36, +0.23, +0.16 and -0.09 (L2145-2153), a median of about 0.2. The 0.4-0.8 must come from the sweep medians (2.6-2.7 against 1.8-2.2, L2167-2169), which give 0.4-0.9. *Fix:* cite the sweep and give 0.4-0.9, or report the Table A1 range.

10. **L1011-1012. Count does not reconcile.** "`b1w` is over its level in none of the 28 mid-rate cells at either delta." Table 4 counts 24 such cells for 4 mid-rate labels (row 6, L730), plus 2 for the pushed label at n_s 200 (row 7), giving 26 per delta. *Fix:* reconcile the count, or add the missing pushed-label cells to Table 4.

11. **L1249-1250. Untraceable figures.** "`ESS_truth` 0.75 to 1.13 at 82%, 0.43 and 0.65 at 93%" for the Wald-t limit. The 0.43/0.65 at 93% appear nowhere else. Table A1 has no Wald-t column, and the other 93% figures are 0.96/1.12 (`b1w`), 0.83/1.03 (bootstrap-t) and 0.88/0.93 (source). *Fix:* give the source, or add a Wald-t column to Table A1.

12. **L1242-1250. Overlapping rule definitions.** The 8.1.5 rule defines mid-rate as 5-95% (L1242, also L2284), but a later bullet says that above about 80% a random draw serves as well. *Fix:* define the stratify region as about 5-80%.

13. **L1796. Range mixes bounds and truths.** "a sample of 300 pairs is expected to certify a margin of 2.4 to 6.5 points with labels alone." 2.4 comes from the normal-type formula at a true difference of -0.0005. 6.5 is the simulated mean of the exact betting bound at +0.006 (L2227-2229). The range mixes approximate and exact bounds and two different truths. *Fix:* report them separately: approximate 2.4 (formula) / 3.7 (simulation); exact 6.5.

14. **L1767-1770 vs L2236-2237. Text contradicts the paper's own check.** "the guard cuts the budget again by 2.5." The authors' design check on real guard scores measures a guard gain of about 1.4. *Fix:* qualify Table 10's guard columns as upper estimates, or recompute them at 1.4.

15. **L876-878. The example does not show a failure.** "a judge that misses 40% of violations let the true violation rate among returned policies reach 0.059." 0.059 is below delta 0.1, and the guarantee bounds P(returned and violating), not a rate among returned policies. *Fix:* report the joint probability and say whether it exceeds delta.

16. **L2216-2218 vs L1372-1373. Numbers do not support the mechanism.** Appendix B says the compiled judge flagged 10% of patterned and 25% of unpatterned guard flags before training, "so what changed with training is how many responses have it". With 82% patterned after training (L2208), that predicts recall of about 0.13. The observed recall for the canonical wording is 0.03 (seed 0: 3 of 93, L2209-2210). *Fix:* say which wording the 10%/25% refer to, and that per-pattern recall also fell, or soften the claim.

17. **L978-979, L985-987 (Figure 1). Figure does not show what it is cited for.**
    - The text cites Figure 1 for a Spearman of 0.84 between `ESS_pred` and realised `ESS_est` over 432 real cells. The figure instead plots five real labels and synthetic bandit cells against the reference ICC. No predicted-vs-realised comparison is shown.
    - The figure labels the encoded-refusal label "94%"; the text says 93% (L937, L962, Table A1).
    - The figure title "can be ranked in advance" is stronger than the registered result: the registered Spearman criterion failed under the registered rule (0.71, L2279).
    - *Fix:* add the 432-cell scatter, harmonise the label, and qualify the title.

18. **Figure 3 title (L1346-1348). Title overstates.** "A judge's calibration does not carry across prompt populations" overstates the figure. In two of the four population directions the carried bound did not fail (it returned 1 or held; L1357-1361). *Fix:* "did not carry in at least one direction for most wordings".

19. **L1042. Classification not supported at the printed precision.** "in one more at the 98% rate (0.052 at delta 0.05)" is classed as over. At 40,000 draws the over threshold is 0.05 + 2(0.00109) = 0.0522, so 0.052 as printed is unresolved. Related work (L1989) counts the cell as over. *Fix:* print four digits.

20. **L34 (abstract), L1038, L1076. "Nothing" is not literally true.** "for rare behaviours (under 4%) it gained nothing." The reported gains go up to 1.11 (L1038) and 1.08-1.09 (L971, L1076). *Fix:* "at most about 1.1".

21. **L1586-1588. Wording inverts the meaning.** "both are over their level for no pipeline ..., and so is the bound clustered by user task within suites." After a negative, "so is" reads as "is over". *Fix:* "nor is" / "and neither is".

22. **L966-967. Awkward wording.** "So is the bootstrap-t StratPPI limit, in 21." *Fix:* "The bootstrap-t StratPPI limit is over in 21 (up to 0.166)."

23. **L697-698, L741, L769, L778. Bolding inconsistent.** The Table 4 caption defines bold as "a bound we do not use in that setting". Row 17 (`b1w`, redrawn with replacement) is bold, but rows 45 and 54 (the same bound and design) are not. Row 54 has 4 cells over after correction. *Fix:* apply the stated rule, and mark registered status separately.

24. **L721 vs L1550-1551 and Table D3. Inconsistent registration status.** The caption says "Rows 37 to 55 were registered before they were run". Rows 30-31 (the multiway and quadrature bounds, scheme (c)) were also registered, by a local commit. *Fix:* mark rows 30-31 as registered (local timestamp).

25. **L1094. Unexplained cell count.** "With 4 strata it was over in 1 cell of 26." The count of 26 is unexplained; elsewhere the reference-strata cells number 28. *Fix:* explain it.

26. **L1312. Style.** Double parenthetical "(... delta 0.05) (Figure 2, right)". *Fix:* merge them.

27. **L3-4, L2038-2047, Appendix E, L2310. Internal notes and anonymity.** These contain build notes ("Generated from the working draft by `scripts/paper_clean.py`"), reference-checking logs, file paths and a script name (`transfer017.py`). Commit hashes throughout Appendix D, "the repository" (L1798) and "by the same authors" (L2108) break double-blind anonymity. *Fix:* remove them, and move provenance to an anonymised supplement.

28. **L491, L899, L1099, L1107, L1128, L1168, L1191, L1226, L1393, L1578, L1598, L2215. Unexplained tags.** These lines carry internal tags such as [B6], [O2], [O6], [B3], [O3], [B25], [B25 sensitivity], [B4] and [O4]. *Fix:* remove them, or map them to Appendix D rows.

29. **L22-23 (abstract) vs L2190-2191. Attribution.** The 90% miss is attributed to "an allocation rule published with one of them". The cell uses a floor of 2 labels per stratum that "is ours (the paper gives none)". *Fix:* "under the published heuristic allocation with our minimum of 2 labels per stratum (59% with 10)".

30. **L1522 / L1637 (section title "one of 28") and L56 (abstract). Missing qualifier.** The single pass is under an approximate bound. The exact pair-rate bound, on "the quantity the certificate is about", does not pass (L1864-1866). The abstract mentions the exact bound only as "agrees only narrowly". *Fix:* add "under an approximate bound; not under the exact pair-rate bound" where the count is first given.

**Count of plain errors: 30.**

**Recomputed and found consistent:**
- se thresholds: 0.057 at R = 4,000 and 0.0562 at 5,000.
- Wilson zero-count limits and peaks: 2.6%/1.3%, 0.069, exp(-z^2) = 0.067/0.194.
- 0.99^100 = 0.37.
- Label budgets 299/149/59.
- 1 - 0.05^(1/6) = 0.393; 1 - 0.05^(1/120) = 0.025.
- Table 6 CP limits: 0.137/0.125/0.052/0.044.
- Fisher 10/120 vs 4/240: p = 0.003.
- Table 7 per-pair CP limits: 0.021/0.032/0.048/0.063/0.087/0.511.
- 19,380 episodes.
- Seed-0 CP 0.198.
- Table 8 marginals: 159/172, 98/123, 51/955, 308/1250; the per-model counts recompute.
- Table 9 precision and false-alarm rate.
- Design effects 1.2/0.8/2.0.
- Table 10 budgets: 1,226/416/167/339, 2,801/950/380, 1,266/429, 462/157.
- 23% = 0.6 x 190/490.
- 8,820 responses.
- Table A1 medians 2.10/2.16, and the ahead/behind percentages.
- The 378-cell count, with 1 over and 24 unresolved.
- The 64 registered cells.
- The 2/3/1/4 cells over in rows 38/39/46/54.
- The 72/53/18/1 tally across Tables D1-D4.
- 27%-41% pre-flight shortfall; 75%-86% of `b1w`'s gain kept.
- The mixture rate of 46%.
- "Three quarters" of the variance reduction.

## 6. Questions for the authors

1. Which cell gives the abstract's 5.3% for Clopper-Pearson?
2. The human-label certificate needs 600 labels and its analysis is fixed. Why was it not run? Is a second annotator feasible?
3. What property of the real pools makes the stratified Wald-t limit hold there when it fails on 10 of 36 synthetic i.i.d. cells? For example: nearly degenerate strata (73-97% of prompts at reference rate 0 or 1), n_h, or the smoothing p~_h. Can you give a diagnostic a practitioner can check before relying on it?
4. If "no limit" is scored as a miss, or replaced by the vacuous bound and reported as an NSF rate, what are the miss rates of the cluster and pigeonhole bootstrap-t rows? Please report the abstention rate for every bootstrap row of Table 4.
5. For the labelling-sheet rows, what is the Monte Carlo standard error with the 20 plantings treated as clusters?
6. Appendix B: which judge wording do the 10%/25% per-pattern rates refer to, and what are the per-pattern recalls after training?
7. StratPPI heuristic allocation: what happens with a floor proportional to the stratum weight, or the minimum stratum size you consider reasonable? Did you contact Fisch et al. about the reference code?
8. The trajectory check (L916-920) uses a level delta/T over T checks and reaches a miss of 0.100 = delta. Are the T checks nearly independent, or are the per-check bounds anti-conservative?
9. How would you separate in the prediction tally those predictions that are implied by an exact bound's construction (X1, W1, Z1, Z2, Y6) from substantive ones? What is the held/failed count among the substantive predictions only?
10. Can the pre-flight be made useful without sampling the trained policy? From reference samples alone it over-predicts by 27-41% (L1087) and missed the moved-far case (1.55 predicted vs 0.97/1.08 realised).
11. Is there any evidence for the strata beyond 0.5-2B policies and the Qwen3Guard label, for example published samples from larger models with a second guard?
12. For AgentDojo, would you call the claude-3-5-sonnet-20241022 result a certificate, given that your own rule (8.1.5) prescribes an exact bound at 1.1% and the exact pair-rate bound does not pass?

## 7. What would most raise my score

1. **Deliver the motivating certificate.** Run the prepared human-label certificate (600 labels, at least two annotators, with agreement reported) on the returned policy (step 175). Report all four registered limits. This alone would address C1.
2. **Choose one thesis and cut accordingly.** One option is "a validity audit of certificate procedures for LLM behaviour rates": Table 4 plus the crossed-design and judge-transfer results. The other is "reference-rate stratification for costly labels". Move the robot benchmark, the routing rule and the companion-paper material to an appendix or a separate paper. Bring the abstract under about 250 words, use one ESS variant in the main text, and split Table 4 into a main table and a registered-confirmation table.
3. **Make the constructive method either guaranteed or characterised.** Either develop a finite-sample-valid stratified bound that keeps a useful part of the gain, or give a theoretical or diagnostic account of when the stratified Wald-t limit holds, and test it on more models and guards, natively at 18-65% rates and at rare rates near 1-5%.
4. **Strengthen the confirmatory record.** Use externally timestamped registrations (e.g. OSF), separate trivial from substantive predictions, report abstention-adjusted miss rates, and give clustered Monte Carlo errors for the sheet rows.
5. **Fix the plain errors in section 5.** In particular: the abstract's 5.3%, "a quarter", the misplaced gain sentence, "the only stratified limit", "65% and above", the Figure 1 mismatch, and the Appendix B arithmetic. Remove the internal tags, build notes and identifying provenance.
