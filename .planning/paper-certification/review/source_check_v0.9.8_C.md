# Source check C: sections 9 to 13, References, Appendices B, C, E

Manuscript: `reports/paper_certification_clean.md` (line numbers are the clean copy's). Nothing in the repository was edited. Two things were recomputed on CPU from stored files, both in the scratchpad: the carried bound of sections 9.1/9.2 with the clip and the refuse variants (reproduces `transfer.md` to the last digit), and `scripts/bib_check.py` on the bib file (it fetches arXiv and CrossRef itself).

Overall: every cell of Tables 6, 7, 8, 9 and 10 and every row of the figure CSVs fig4 to fig8 matches its source. The findings below are about prose: one false universal statement, an unsupported denominator, and a set of dropped conditions. `scripts/validity_recount.py` asserts the counts of sections 9.1, 9.2 and 10.2, Table 7's last column and the design-check miss rates. It does NOT assert: Tables 6, 8, 9, 10; Table 7's other columns; any number in sections 9.4, 9.5, 10.1, 10.3; the budget numbers of 10.4; Appendix B; the "about 420" of the abstract (and it asserts the abstract's "at most one of the 28" only as a string that must be present, not against the files).

---

## HIGH

### H1. Limits says every harm number outside 10.3 is a guard's; section 9.5 and section 8.2 use 225 human harm labels, whose annotator the paper never names
- Line 1376-1379: "Outside section 10.3, and the budget of section 10.4 computed from it, every refusal and harm number is relative to a guard model's field: Qwen3Guard-4B, 4-bit quantised, or the 0.6B model in the results cited from the companion paper".
- Source: `.planning/spikes/017-calibration-carrying-certificate/harm.md` line 3: "Population 4800 responses, all scored; 225 human labels, 3 positive"; its section 2 gives the "PPI as i.i.d." estimate -0.1378 (bound -0.0822) for wording 2. That is the "certified a negative harm rate" of section 9.5 (line 1091), so that number is in human labels. Table 3 (line 586) itself lists "225 human harm labels" as gold for section 8.2. `.planning/spikes/005-capability-screen/README.md` line 181: "The annotator `ah` labelled all 227 tier-1 responses" (one annotator, the same author as in 10.3). The working draft's own open check 9 lists this contradiction as unreconciled. Line 345-346 ("The human-label results are in section 10.3") has the same gap.
- Also a human-labels point: nowhere does the paper say that the 225 harm labels are one annotator's, an author's, with 3 positives.
- Suggested: "Outside section 10.3, the budget of section 10.4 computed from it, and the harm sheet of sections 8.2 and 9.5 (225 responses labelled by one annotator, an author, 3 of them harmful), every refusal and harm number is relative to a guard model's field: ..."

---

## MEDIUM

### M1. Section 9.4: "10 of 210 cells" is not in the stored results (10 of 84)
- Line 1084-1086: "Finite-sample judge-assisted bounds (betting on blocks) were valid and bought nothing: narrower than the labels alone in 10 of 210 cells, by at most 0.003."
- Source: `plasmode.json` (60 block-PPI cells), `plasmode_shift.json` (24), `plasmode_n500.json` (0) in the spike 017 directory: 84 block-PPI cells in all, 10 narrower than Clopper-Pearson on the same cell, largest gap 0.0026, largest miss 0.044. "210" appears only in the 017 README's prose (trail item 3). 7 of the 10 are on the raw judge, not the compiled one. Draft open check 7 already notes that the denominator was never reproduced.
- Suggested: "... narrower than the labels alone in 10 of the 84 stored cells, by at most 0.003."

### M2. Section 9.1: the direction in which the carried bound "held" is the direction in which it returned no certificate
- Line 1034-1036: "The failure depends on the direction of the carry. Between the two benign sources the bound was over its level for 4 of 6 wordings one way and 1 of 6 the other; between the pools, for 5 of 6 one way and none the other."
- Source: `transfer.md`, "pool: over-refusal -> harmful": median bound 1.000 in all six wordings. Recomputed on the stored draws: the bound is 1 (vacuous) in 98.3% to 100% of draws for every wording in that direction, and in 22% to 97% of draws for the six wordings of OR-Bench to XSTest (medians 1.000, 1.000, 0.642, 1.000, 0.733, 1.000). The counts are right; what is missing is that "none" means "no certificate", not "a carry that worked".
- Suggested: "... between the pools, for 5 of 6 one way and none the other, where the carried bound returned 1 (no certificate) in at least 98% of draws under every wording; in the benign direction with 1 of 6 its median was 1 for four wordings."

### M3. Section 9.5 (and the abstract): what failed was PPI on the sheet read as i.i.d., not the sheet's labels read as i.i.d.
- Line 1090-1093: "Read as i.i.d., it certified a negative harm rate under one judge wording. With labels planted at a 1.3% rate it missed in 98% of re-drawn sheets under that wording, and in 0-99% across wordings and rates (Table 4)." Abstract line 44-45: "a labelling sheet that deliberately over-samples some responses cannot be read as a random sample."
- Source: `harm.md` section 3. The column "PPI as i.i.d." runs 0.000 to 0.989 (7 of 18 cells over, all on wordings 2 and 4; wording 0 at most 0.005). The column "sheet as i.i.d." (the labels alone, no judge) runs 0.000 to 0.008: it never missed, because over-sampling flagged responses overstates the rate. Table 4's row (line 653) does say "PPI on the sheet"; section 9.5 and the abstract do not.
- Suggested (9.5): "Read as i.i.d. inside a prediction-powered estimate, it certified a negative harm rate under one judge wording. With labels planted at a 1.3% rate that estimate missed in 98% of re-drawn sheets under that wording, and in 0-99% across three wordings and three rates (over its level in 7 of 18 cells; Table 4). The sheet's labels alone, read as i.i.d., missed in under 1% of sheets: they overstate the rate." Abstract: "... cannot be read as a random sample when a judge's scores are combined with it."

### M4. Section 10.2: the paper's own certificate also abstains, and the text attributes abstention to tables with no success only
- Line 1160-1163: "that bound is over its level for none (2 unresolved, neither of them the pipeline that certifies below). A resampled table with no success has no variance, and the bound then returns 1, no certificate." Line 1184-1186 applies the abstention caveat to the pigeonhole bootstrap-t only ("a bound that returns no limit cannot miss").
- Source: `results/paper/agentdojo_twoway.json`, `resampling[...]["a"]["t_user"]["no_limit"]` (scheme (a), 4,000 resampled tables): 0.087 for claude-3-5-sonnet-20241022 (the one pipeline that certifies), 0.155 and 0.157 for the two Meta-SecAlign-70B pipelines; `zero_success_resamples` for the same cells is 0.002, 0.004 and 0.006. So the bound clustered by user task returns no limit 25 to 50 times more often than a table with no success occurs (the inner bootstrap's 5% quantile is not finite), and part of "over for none" on exactly the low-rate pipelines is abstention, the caveat the paper makes for the other bound. (The recheck file of seed 20261004 does not store this share; the figures are from the registered run's fresh draws.)
- Suggested, after "no certificate": "It also returns 1 when more than 5% of its own bootstrap draws hold no success: with user tasks resampled it gave no limit in 9% of the tables of the pipeline that certifies and in 16% for the two Meta-SecAlign-70B pipelines, and a bound that returns no limit cannot miss."

### M5. Abstract: "at most one of the 28" holds only for the pairing-aware bounds that kept their level
- Line 43-44: "with methods that allow for the pairing at most one of the 28 certifies an attack success rate under 5%."
- Source: `agentdojo_recheck.md` section 2 and `agentdojo_twoway.md` section 2: the bootstrap clustered by injection task, the pigeonhole bootstrap with a basic limit and the multiway variance with a t quantile each pass 3 pipelines (the two Meta-SecAlign-70B pipelines and claude-3-5-sonnet-20241022). All three allow for the pairing; all three are over their level. Section 10.2 (line 1214-1216) says this for two of them; the abstract's universal does not hold against the whole file.
- Suggested: "... and with the methods that allow for the pairing and kept their promise in our checks, at most one of the 28 certifies ..."

### M6. Section 12 and section 13 summarise section 9.2 without its scope (one run each side, one seed, one judge)
- Line 1473-1474: "and the negative of section 9.2: the predictions' relation to the label is not stable under training that targets the label."
- Line 1489-1491: "three common shortcuts were, in their worst cells by wide margins: a normal quantile at small rates, a per-pair bound on a crossed design, and a judge calibration carried to a policy it was not measured on."
- Source: 017 README, E8 addendum and `transfer.md`: one constrained run (spike 014, seed 0), one side-effect run, one judge model with six wordings, gold = the guard. Carrying "to a policy it was not measured on" was over its level in that one run only (3 of 6 wordings; 0 of 6 in the side-effect run); section 9.1's failures are across prompt populations of the same policy. Section 9.2 itself (line 1043), the abstract (line 40) and the Limits (line 1395-1396) carry the scope; these two sentences do not. Section 5, line 326-327 ("across training that targets the label"), outside my part, has the same gap.
- Suggested (12): "... and the negative of section 9.2: in the one run we examined, the judge's relation to the label did not survive training that targets the label." (13): "... and a judge calibration carried to another prompt population or, in the one training run we examined, to the trained policy."

### M7. Section 13: "no exact bound was over its level" needs its condition
- Line 1488-1489: "Where we could check against a known truth, no exact bound was over its level, and three common shortcuts were".
- Source: `validity_recount.md`, Table 4 row 24: per-pair Clopper-Pearson, an exact bound, over for 27, 28 and 28 of 28 pipelines; the carried bound of section 6.5 is built from exact limits at delta/3 and is over in up to 5 of 6 wordings. The abstract has the condition ("on data sampled as they assume"); the conclusion drops it and then lists the per-pair bound as a shortcut in the same sentence.
- Suggested: "Where we could check against a known truth, no exact bound was over its level on data sampled as it assumes, and ..."

### M8. Abstract: the "about 420 labelled prompt pairs" drops the limit type and the borrowed pairing
- Line 48-50: "if training left the refusal rate unchanged, about 420 labelled prompt pairs would give an even chance of certifying that it rose by no more than 2 points."
- Source: `results/paper/p9_budget.md`, row margin 0.02 / 50% / "labels alone, paired by prompt": 416. Its conditions (`scripts/p9_budget.py`): a normal-type one-sided limit at delta 0.05; the pairing correlation 0.66 taken from the guard's flags, not from human labels; the rates 0.107 and 0.106 from one annotator's 220 labels at step 200 (the policy sampled for the certificate is step 175); the measured difference -0.0005. "Unchanged", "even chance" and "2 points" are carried. Not carried: (i) the limit is approximate. The prepared analysis's exact limit (a) is about twice as wide (`design_check.md`: mean margin 0.065 against 0.037 at 300 pairs), which by 1/sqrt(n) scaling is some thousands of pairs, not 420; (ii) the paper's own design check sits at a difference of +0.006, where the same formula gives about 930 (line 1336-1337); (iii) 420 pairs are 840 labels.
- Suggested: "... if training left the refusal rate unchanged, a design calculation with an approximate limit puts the cost at about 420 prompt pairs (840 labels) for an even chance of certifying that it rose by no more than 2 points; the exact limit we prepared would need several times more."

### M9. Section 10.4: the only validity figure the main text gives for the new-prompts limit is under the scheme it is not for
- Line 1360-1363: "puts the miss rates of the four limits at or under 0.045 for the pool rate against a level of 0.05".
- Source: `results/labels/p9/design_check.md`: pool scheme 0.001, 0.040, 0.045, 0.042; new-prompts scheme 0.001, 0.051, 0.063, 0.052 (se 0.005). Limit (b2) is defined as the claim about new prompts (line 1357-1358); its like-for-like miss is 0.052 (unresolved), and (a') is 0.051. Appendix C gives these; the main text does not, and "the four limits at or under 0.045" reads as a clean pass for all four. Also unstated: the check computes each limit with 300 bootstrap draws, the frozen analysis with 10,000 (`p9_certificate.py`, `BOOTS = 10000`, `check --boots 300`).
- Suggested: "... at or under 0.045 when the 490 prompts are fixed; when the prompts are redrawn, which is the claim of limit (b2), the labels-alone bootstrap-t and (b2) are at 0.051 and 0.052 (unresolved at 2,000 repetitions) and (b1), not built for that case, is at 0.063."

### M10. Section 10.1: "narration does not predict a stop" is a count over a trial, and stopped trials are short
- Line 1146-1151: "The policy's own narration of risk does not predict a stop. ... is 0.38 (90% interval 0.24 to 0.52; 0.42 to 0.49 by model when the word "drop" ... is left out)."
- Source: `.planning/spikes/019-external-trace-certificate/results.md`, exploratory block. (i) The score is a count of notes over the trial; the AUC of the call count against a stop is 0.01, 0.37 and 0.01 by model, so a stopped trial has fewer notes of any kind. An AUC below 0.5 is what truncation gives. (ii) By model under the frozen pattern the AUC is 0.31, 0.79 and 0.30; the 0.79 (Claude Opus 5.5, 2 stops) is not shown, while the by-model range that is shown is the other pattern's. (iii) The strict pattern matches 0.02, 0.03 and 0.15 notes per trial, so its AUC near 0.5 is mostly ties.
- Suggested: "The count of the policy's risk notes does not predict a stop (AUC 0.38, 90% interval 0.24 to 0.52; 0.31, 0.79 and 0.30 by model, on 10, 2 and 2 stops), but the count is over a trial and stopped trials are short (AUC of the call count 0.01 to 0.37), so this does not show that the narration carries no signal. Without the word "drop" the pattern matches under 0.2 notes a trial."
- Same sentence, line 1151-1152 ("The peak joint effort ... does predict one (AUC 0.75-0.83 by model, exploratory)"): the 019 README adds "partly mechanical, since a stop follows sustained torque" (an overheat stop is triggered by effort). The paper keeps "exploratory" and drops that. Suggested addition: "..., exploratory, and partly mechanical: the stop is an overheat)."

### M11. Section 10.3: the guard's kappa on the broad event is compared with the humans' kappa on three classes
- Line 1257-1258: "Its agreement with the broad human label (Cohen's kappa 0.81) is close to two humans' agreement on the three classes (0.88); with the strict label it is 0.59 against the humans' 0.90."
- Source: `results/labels/xstest/analysis.md` gives 0.88 (two humans, three classes), 0.90 (two humans, strict), 0.81 and 0.59 (guard). The strict comparison is like for like. For the broad event the file has no two-human figure; computed from `guard_scores.jsonl` (fields `a1`, `a2`, safe prompts, n 1,250) it is 0.92.
- Suggested: "... (Cohen's kappa 0.81) is below two humans' agreement on the same event (0.92; 0.88 on the three classes); with the strict label it is 0.59 against the humans' 0.90."

### M12. Section 12: the two-way sentence drops the resampling scheme and the cluster counts
- Line 1448-1451: "Section 10.2 runs the first two on a benchmark's published table at rates of 1% to 56%: the variance with a symmetric limit is over its level for most pipelines, and the studentised pigeonhole bootstrap is not."
- Source: `agentdojo_twoway.md` section 1: multiway-t over for 5, 8 and 26 of 28 under schemes (a), (b), (c), so "most" is scheme (c) only. The tables have 71 or 97 user tasks and 27, 30 or 35 injection tasks (`agentdojo_twoway.json`, `real[...]`), a few-cluster setting for Cameron, Gelbach and Miller's asymptotics, with successes in a handful of clusters. Owen's bootstrap as published (no studentising) was over for 27 of 28 (line 1171-1172); the bound that held is the paper's own studentised variant, and it gave no limit in more than 5% of tables for 7 pipelines.
- Suggested: "Section 10.2 runs the first two on a benchmark's published tables (97 user tasks by 27 to 35 injection tasks, rates of 1% to 56%). When both kinds of task are resampled, the variance with a symmetric limit is over its level for 26 of 28 pipelines and the pigeonhole bootstrap with a basic limit for 27; a pigeonhole bootstrap studentised by that variance is over for none, in part by returning no limit."

### M13. Appendix E does not list the scripts behind sections 9.1, 9.2, 9.4, 9.5, 9.3's design effects and most of 10.1
- Line 1655-1656: "The numbers in this paper come from the scripts below, apart from those cited from the companion paper and a few recomputed from the data files."
- Source: the table has no row for `transfer017.py`, `plasmode017.py`, `harm017.py`, `route017.py` (sections 9.1, 9.2, 9.4, 9.5; in `.planning/spikes/017-calibration-carrying-certificate/`), `cert020.py` (the design effects of 9.3, the ICC 0.72 of 10.2), or `extract019.py` / `bounds019.py` (Table 6's first columns, the Fisher p, the narration and effort AUCs). All of these exist and their outputs match; they are just not "the scripts below". Every path the table does name exists (45 of 45 checked with `ls`), and each named script writes the named output with its defaults or an obvious subcommand (`refusal_labels.py analyze`, `p9_certificate.py check`, `judge_on_labels.py analyze`). Not said in the table: `xstest_guard.py`, `judge_on_labels.py score`, `p9_sample.py` and the two confirmation pools need a GPU, and XSTest's completions are fetched from outside the repository. `bib_check.py` exits with "5 entries need a look" (see L9).
- Suggested: add rows for the three spike directories, or change the sentence to "... come from the scripts below and from the spike scripts under `.planning/spikes/017`, `019` and `020`, ...".

---

## LOW

- **L1, line 1123-1124.** "The tightest exact limit we have on their mean, the betting mixture, is 0.44 to 0.48". `robodojo_sampling.md` section 4: Bentkus is 0.440 for the two models with two stops, the betting mixture 0.441 and 0.442. The betting mixture is the tightest for one model of three. Suggested: "The tightest exact limits we have on their mean (Bentkus or the betting mixture) are 0.44 to 0.48".
- **L2, line 1056 against Table 9 line 1293.** "The guard's recall and precision ... (17 of 18 and 16 of 18; 0.59 and 0.60)" are unweighted counts (0.94, 0.89); Table 9 prints the design-weighted recall 0.99 and 0.98. Both are in the sources (`rubric_vs_human.md`, `analysis.md`); the text does not say which is which. Suggested: "(17 of 18 and 16 of 18 by count, 0.99 and 0.98 weighted; precision 0.59 and 0.60)".
- **L3, line 1055.** "lower after training in all six wordings" is true by count (3, 3, 3, 5, 11, 7 of 18 against 0, 0, 0, 1, 6, 6), but wordings 0, 1 and 2 give identical counts and largely the same flagged responses (`rubric_scores.jsonl`: q247 and q274 under all three), and wording 5 differs by one response (p = 1.0). Suggested: add "(three of the six give the same counts, and one differs by a single response)".
- **L4, line 1609 (Appendix B).** "of 93 guard-flagged refusals after training it called 89 answers". With the rule used everywhere else (p > 0.5) the judge flags 3 of the 93 (`scores_pop.jsonl`, set `ref_c1_014_200`, wording 0), which is the recall of 0.03 in section 9.2, so it called 90 answers; 89 needs p >= 0.5 (one response sits at 0.500). Suggested: "it called 90 answers".
- **L5, line 1033-1034.** "up to 100%" is 0.996 (`transfer.md`, harmful to over-refusal, wording 3). Suggested: "up to 99.6%".
- **L6, line 1262-1263.** "The logit beats the 0/1 flag for the strict event (0.46 against 0.40) and for the broad one (0.70 against 0.65)" is the pooled row. By model it reverses for GPT-4 (0.67 against 0.71 strict, 0.68 against 0.76 broad), and on our own reference policy the broad event is a tie (0.73, 0.73; `results/labels/refusal/analysis.md`). Suggested: add "pooled over the five models; for GPT-4 alone the flag is ahead".
- **L7, line 1354.** "The analysis was fixed before any label". No label of this sample exists, and the commits (9e45371, e076974 on 2026-10-04; a5bd04b on 2026-10-07) are before any. But the design was reframed in 9e45371 after the first sheet's 220 human labels on the same pool had been analysed (`labels_ah.jsonl` 14:34, commit 14:58), and its rates come from them. Suggested: "fixed before any label of this sample, and after the first sheet's result".
- **L8, line 1220-1221.** "at 4,000 draws it ranged from 0.058 to 0.107 over 30 seeds" is the recheck's 30 seeds; the registered run's 30 seeds give 0.057 to 0.118 (`agentdojo_twoway.json`, `real["Meta-SecAlign-70B"]["spread"]["t_user"]`). Table 7's 0.062 for the same cell is the recheck's 200,000-draw value (0.0615); the registered run's is 0.061 (0.0607). The verdict does not change.
- **L9, line 1499-1502 and 1682.** "Verified against the publisher or arXiv page on 2026-10-04". No stored output of that check exists. Run today, `bib_check.py` matches title, authors and year for 22 of 27 entries and flags 5: Machcha (a web report, not indexable; I confirmed title, five authors, date and "Inspect Robots 0.58.0" against the stored `results/spikes/019/page.html`); Menzel (the DOI resolves; the mismatch is a non-ASCII hyphen in CrossRef's title, a script artefact); Qwen3Guard (the bib abbreviates; arXiv lists 43 authors, so "and 40 others" is right); Boyeau (arXiv v3 lists five authors, without Li; the reference gives six, which I believe is the ICML version's list but could not confirm); Fisch (the arXiv record's title does not equal the reference's title; I believe the reference gives the NeurIPS title and arXiv an earlier one, unconfirmed).
- **L10, line 1115.** "its miss probability stays under delta at every mean rate from 0.5% to 16%": `robodojo_sampling.md` prints the largest as 0.0500 (even over tasks), and the check covers three shapes of task rates (even, two tasks, one task), not all. Suggested: "stays at or under delta ... for the three shapes we tried".
- **L11, line 1267-1268.** "a calibration measured on one of these models would not serve another" is an inference from recall and false-alarm rates that differ by model (Table 8); no carried bound was computed on XSTest. Suggested: "would not be expected to serve another".
- **L12, line 1302-1303.** "worth more than a halving of the labels (`rho^2` 0.6), in line with the published-label result above": for the same (strict) event the published-label figure is 0.46, "roughly a twofold gain" at best (1.85). "In line with" is loose; 0.6 is higher.

---

## Checks that came out clean

**"Registered" and "fixed in advance" (git).**
- Two-way bounds (line 1175-1176): `R2_registration.md` and `agentdojo_twoway.py` enter in cb50927 (2026-10-07 20:26), results in 56721e6 (21:08); the JSON holds the registered seed 20261008, 4,000 by 4,000. Holds. The registration's expectation for the quadrature bound ("over for none") failed, and the paper reports that.
- Section 9.2's note (line 1061-1063): the quoted words are in the 017 README; note and result share commit 3f89e98, and the paper says "by our own note". Holds as worded. (The same note expected step 200 to carry and step 100 to fail; the paper quotes only the conditional reading, which is accurate.)
- Section 10.4: see L7.

**Scope of section 9.2.** Lines 1043-1050 say "two single runs", "one training run", "one run"; line 1059-1061 lists one seed each, the guard as gold, and the judge's low recall; the abstract (line 40) and Limits (line 1395-1396) carry it. The gaps are M6.

**Human labels.** Section 10.3 and Table 9 say one annotator, an author, 220 responses, no agreement statistic; `labels_ah.jsonl` has 220 rows from handle `ah`, slot 1 of 2, dealt within strata. XSTest: two annotations and a final label per completion, kappa 0.88 on three classes. No sentence presents one annotator as several. The gap is H1.

**Section 10.4 says designed and costed, not run** (lines 1317, 1351, 1367, 1382-1384, 1493-1495; abstract line 48). No P9 label file exists in `results/labels/p9/`. Holds.

**Cited methods inside their conditions.** Cluster bootstrap-t with six clusters: the paper says it is not usable (line 1126-1128). Pooled Clopper-Pearson on trials grouped by task: stated as tasks fixed, trials independent within task, checked by convolution, with Hoeffding (1956) cited for the direction. Clopper-Pearson over pairs on the crossed design: reported as failing. The points on the multiway variance and the pigeonhole bootstrap are M12.

## Universal claims verified as holding

1. Line 1036-1037, "False-alarm rates did not differ": 0 of 6 wordings in each of the seven shifts (`transfer.md`, smallest p 0.112).
2. Line 1044-1045, "recall differed in 0 of 6 wordings" (side-effect run); line 1047-1048, 3 of 6 over (0.08, 0.23, 0.80) and 0.054 marginal; line 1049-1050, 1 of 6 at step 100.
3. Line 1076-1077, per-pair bound over for 27 of 28 (up to 0.268) and for all 28 under the other two schemes.
4. Line 1085, block-betting bounds "valid": largest miss 0.044 over the 84 stored cells.
5. Line 1093, "Every sheet in this paper carries its design weights": the harm sheet (design-weighted, `b1w`), the refusal sheet (weights at analysis time), the P9 sheet (a simple random sample of 300 of 490 prompts, equal weights, no labels).
6. Line 1125, `1 - 0.05^(1/6)` = 0.393; line 1141, 299 trials for 1%; line 1140, 2.5% at zero stops (0.0247) and 0.0515 at two.
7. Line 1127-1128, Student-t below the pooled limit for both two-stop models (0.038 and 0.050 against 0.052); cluster bootstrap-t returns 1 for both at delta 0.05.
8. Line 1161-1162, bound clustered by user task over for none of 28, 2 unresolved, neither the certifying pipeline.
9. Line 1169-1173, 24, 23, 27, 22 and 16 of 28 under scheme (c).
10. Line 1181-1185, multiway-t over for 26 of 28 (up to 0.179); pigeonhole bootstrap-t over for none under any scheme (largest 0.042); no limit in more than 5% of tables for 7 pipelines, 77% for the lowest-rate one.
11. Line 1188-1189, median margin ratio 2.05 and no pipeline certifying under it; line 1190-1191, no limit for the certifying pipeline (6.0% of scheme (c) tables hold no success).
12. Line 1213-1216, one pipeline certifies under the user-task bound and the same one alone under the larger of two; the injection-task bootstrap and the basic pigeonhole bootstrap pass three and are over in some pipelines under all three schemes (3/4/23 and 5/14/27).
13. Line 1222-1223, five pipelines pass the per-pair bound (gemini-1.5-flash-002 at 0.0496) and four do not survive.
14. Line 1299-1302, no guard miss among 72 and 70 responses in the large guard-negative strata; line 1298-1299, one stratum of 18 labels and two of 15 (recomputed from `labels_ah.jsonl` and `key.jsonl`).
15. Line 1307-1308, three responses labelled twice with the same label (`v1_labels_ah.jsonl` against `labels_ah.jsonl`: q359, q229, q189).
16. Line 1426-1428, "refusing instead moves three of the 42 cells ... and changes no cell's class": no stored file holds this; recomputed, it holds (0.133 to 0.089, 0.034 to 0.025, 0.079 to 0.058; the last stays over by 0.001).
17. Line 1416, `b1w` over in 7 of 28 cells redrawn with replacement; line 1403-1404, largest interval end 0.067 (both asserted by the recount script).
18. Appendix C: all eight miss rates, the four margins, 9.5% and 10.1%, +0.006, the guard's gain of 1.4, and 16.9% / 17.2% on 3,920 guard-only responses per policy (recomputed from `guard.jsonl`: 662 and 673 of 3,920).
19. Appendix B: 37% and 82% (258 of 700; 840 of 1,022 in `design.json`), 12 / 6 / 0 of 18, 56% and 34% (88 and 54 of 158).
20. Section 10.4: 8,820 responses, 57% and 32% cut off (170 and 96 of 300), threefold (2.95), 2.5, 23%, "about 930" and "2,100" (931 and 2,129 at +0.0063).
21. References: all 27 bib entries and the companion paper are cited in the text, and every in-text citation has an entry.

## Claims I could not trace to a result file

- Line 1085, "210 cells" (M1).
- Line 1308-1309, "two recurring cases ... were labelled both ways": only a log line in `.planning/paper-certification/PLAN.md` (2026-10-04, P6); the label file's note fields are empty.
- Line 1499, the 2026-10-04 verification of the references: no stored output (L9).
- Line 1458-1461, what Khosravi and Huo (2026) do ("online and per decision"): the script confirms title, authors and year on arXiv; the description of the content I can confirm only from the title.
- Line 1510 and 1515, Boyeau's sixth author and Fisch's title against the publisher's pages (L9).
- Line 1342-1343, "a guard-only response costs GPU seconds": no timing file read; plausible from `guard.log`, not checked.
- The other one-line descriptions of cited work in section 12 (Miller; Bowyer et al.; Bates et al.; the two Angelopoulos risk-control papers; the PPI line; Waudby-Smith and Ramdas; Spertus and Stark; Menzel's validity result when cluster effects are absent) agree with the titles and with what I know of the papers. None looks wrong to me; none was checked against the papers' text.
