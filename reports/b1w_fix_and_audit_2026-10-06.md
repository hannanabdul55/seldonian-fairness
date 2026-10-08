# `b1w` zero-count bug, its effect on the certification paper, and a code audit (2026-10-06)

Scope: the stratified Wilson-type bound `b1w` of spike 013, every result that calls it, and a
read-only audit of the bound implementations, the `seldonian/llm` safety-test path and the
validation harness behind Table 4 of `reports/paper_certification.md`.

Status: the bound is fixed and checked, and every caller that could be rerun on a CPU has been
(section 4). The items left open here on 2026-10-06 (`scripts/stratppi_validate.py`,
`scripts/validity_recount.py`, the paper text) were done on the PC on 2026-10-07, with the
with-replacement check of item 12 and several of the audit's fixes: **section 7**. Sections 1
to 6 are as written on 2026-10-06.

## 1. How it was found

An outside review of the draft by `gpt-6-astra` (`reports/gpt-6-astra-review-2026-10-06.md`)
pointed out that the pooled Wilson upper limit at zero positives is `z^2 / (n + z^2)`, which is
0.0263 at n = 100 and delta 0.05, so the bound cannot miss a true rate of 1-2%. Table 4 reported
misses of 0.076-0.448 there. The reviewer guessed a plug-in Wald interval. That guess was wrong;
the cause is below.

## 2. The bug

`.planning/spikes/013-stratified-safety-set/stratbounds.py`, `b1w`: the bound is the smallest
`m >= mu_hat` with `m - mu_hat >= z sqrt(V(m))`, found on a grid that started at `m = mu_hat`.
When every stratum is all-zero (or all-one), `V(mu_hat) = 0` and the first grid point passes as
`0 >= 0`. The function returned the estimate itself: U = 0 at zero positives.

The pooled Wilson arm is the same function with one stratum (`plasmode.py`, `BOUND_FN`), so the
pooled and stratified rows shared the bug. Only samples with no positive at all (or no negative)
were affected; with any positive the first grid point fails and the true root is returned.

The old code therefore missed whenever no positive was drawn, which has probability
(1 - p)^n (the ordinary misses of a Wilson limit at one positive or more come on top):

| true rate | n = 100 | n = 200 |
|---|---|---|
| 0.8% | 0.448 | 0.201 |
| 1.0% | 0.366 | 0.134 |
| 2.0% | 0.133 | 0.018 |
| 2.54% | 0.076 | 0.006 |

The stored cell `C3:unsafe`, step 0, n_s 100 has truth 0.0077 and miss 0.448 = (1 - 0.0077)^100.

**Fix.** The point `m = mu_hat` is no longer accepted as a root (it is kept only at
`mu_hat = 1`, where the bound is 1). `check_b1.py` now asserts, before its Monte Carlo, that
`b1w` at one stratum equals the closed-form Wilson limit at every count for n in 25, 100, 200,
400 at both deltas (largest gap 0.00023, under the grid step of 0.00025), and that four all-zero
strata give a bound above zero. The assertions fail on the old code.

## 3. Effect on stored results

`real_plasmode.py --reps 5000 --ties random` was rerun with the fixed bound (CPU, 8 minutes on
8 cores) and now replaces `results/spikes/013/real_plasmode_randomties.json.xz` (the old file is
in git history at 3d1714e).

- All 1,200 `b1` cells and all 804 mid-rate `b1w` cells are identical to the stored file, so the
  two files differ only where the bug acted.
- 228 of the 396 rare-label `b1w` cells change. By the paper's rule (over when the miss exceeds
  delta + 2 se) the rare-label `b1w` cells go from 245 over / 10 unresolved / 141 at or under to
  17 / 10 / 369.

The cells quoted in Table 4 (k 8; stratified with 8 strata, pooled with a random split; delta 0.05):

| label | checkpoint | truth | n_s 100, pooled | n_s 100, stratified | n_s 200, pooled | n_s 200, stratified |
|---|---|---|---|---|---|---|
| C2:gated | step 0 | 0.0138 | 0.240 -> 0.000 | 0.238 -> 0.000 | 0.048 (same) | 0.038 (same) |
| C2:gated | step 100 | 0.0235 | 0.076 -> 0.000 | 0.085 -> 0.000 | 0.030 (same) | 0.027 (same) |
| C2:gated | step 200 | 0.0192 | 0.126 -> 0.000 | 0.126 -> 0.000 | 0.013 (same) | 0.012 (same) |
| C3:unsafe | step 0 | 0.0077 | 0.448 -> 0.000 | 0.443 -> 0.000 | 0.178 -> 0.000 | 0.174 -> 0.000 |
| C3:unsafe | step 100 | 0.0105 | 0.329 -> 0.000 | 0.331 -> 0.000 | 0.102 -> 0.000 | 0.092 -> 0.000 |
| C3:unsafe | step 200 | 0.0103 | 0.350 -> 0.000 | 0.341 -> 0.000 | 0.099 -> 0.000 | 0.093 -> 0.000 |

- Table 4 row "`b1w` and the pooled Wilson bound at rare rates", n_s 100: 12 over / 0 / 0
  becomes 0 / 0 / 12.
- Widths at rare rates grow (C3:unsafe step 0, n_s 100: 0.0196 to 0.0315), so any ESS quoted at
  rare rates changes.
- 17 rare-label `b1w` cells remain over, all at delta 0.10 with misses of 0.112-0.139: 16 at
  C2:gated step 200, n_s 100, and one at C3:unsafe step 100, n_s 200. This is the ordinary
  discreteness of a score interval and is the failure the paper can still report.

### Claims in `reports/paper_certification.md` that no longer hold as written

| where | text | status |
|---|---|---|
| section 6.3, "Labels alone" | "It is approximate, and at rates of 1-2% it fails (Table 4)" | false at delta 0.05 |
| Table 4, rare-rate row | fail, 0.076-0.448, 12 / 0 / 0 | becomes 0 / 0 / 12 at n_s 100, delta 0.05 |
| section 7.1, item 1 | "missed in 8-45% of draws ... the failure is binomial discreteness"; "At n_s 200 `b1w` is still over its level for the 1% label" | an artifact; not over at n_s 200 |
| section 7.1, the 322-cell reading | "leaves out the rare labels, where `b1w` fails" | the exclusion loses its reason |
| section 8 | "Rare labels (the approximate bound is invalid there ...)" | needs the delta 0.10 qualification |
| Appendix (StratPPI) | "`b1w` fails in" the rare-label cells | recount after rerun |
| abstract and conclusion | any statement that Wilson-type bounds fail at rare rates | check |

The advice "rare labels take Clopper-Pearson or a betting bound" stays safe, but its evidence is
now the 17 cells at delta 0.10, not the 12 at delta 0.05.

Other callers of `b1w` (results not rerun): spike 013 `inloop.py`; spike 014 `analyse014.py`;
spike 017 `harm017.py` and `cards017.py`; spike 019 `bounds019.py`; spike 020 `cert020.py` and
`plasmode020.py`; `scripts/stratppi_baseline.py`; `scripts/stratppi_validate.py`. A cell changes
only if some draws have no positive in any stratum, so mid-rate cells should be unchanged and
rare-rate and sheet cells may move.

### Does this make the paper's case stronger or weaker?

Mixed. It removes one negative finding, leaves the headline results untouched, and makes the
paper's own bound look better than the draft says.

**Weaker.**

- One exhibit for "normal approximations fail at small rates" is gone. Student's t and the
  normal limits of PPI++ and StratPPI still fail there, so the theme survives, but the Wilson
  bound is no longer an example of it.
- The advice "rare labels take Clopper-Pearson or a betting bound" loses most of its evidence.
  What remains is 17 cells over their level at delta 0.10 (misses of 0.112-0.139) and none at
  delta 0.05.
- A cost in credibility. The paper is an audit of bounds, and its own bound had an untested edge
  case that became a reported finding. A referee found it from the text alone. The correction
  should say so.

**Stronger.**

- `b1w` now holds over the whole range tested at delta 0.05, not only on mid-rate labels. In the
  StratPPI comparison (part A, 40 cells) it goes from 9 cells over to none.
- The comparison with StratPPI improves. Before, both failed on rare labels. Now `b1w` holds
  there and StratPPI's published normal limit still does not (its cells are unchanged by the
  fix).
- The exclusion in the 322-cell summary ("leaves out the rare labels, where `b1w` fails") is no
  longer needed.

**Unchanged.**

- The stratification gain of 1.4 to 5.3 on mid-rate labels: those cells reproduce exactly.
- The normal limits of PPI++ and StratPPI failing, and the bootstrap-t limit repairing them.
- AgentDojo, the carried-calibration negatives, the costing in human labels.
- Stratification still gives no gain on rare labels (C3:unsafe step 0, n_s 100: width 0.0313
  stratified against 0.0315 pooled). "Does not help for rare labels" stays true; the reason
  becomes "no gain", not "invalid".

**The larger threat is a different finding** (section 5.3, item 12). The harness draws 20-40% of
a 500-prompt pool without replacement, which makes every bound look more conservative than it
would be on a large population. Redrawn with replacement, the mid-rate `b1w` row goes from 0
cells over of 24 to 4 of 24. That weakens "holds" for the paper's own bound by more than the
Wilson fix strengthens it, and it is a question of design that the fix does not touch. The
bootstrap-t StratPPI limit stayed at 0 over of 28 under that test, so it may be the more robust
recommendation.

## 4. Reruns (all CPU)

The GPU was used only to generate and judge responses (`gen013.py`, `gen014.py`). Everything
below resamples stored labels. Each rerun used the original seeds, so a cell that does not call
`b1w` should come back identical; that is the check that the rerun is the same experiment.

### 4.1 Done on 2026-10-06 (Apple silicon, 8 cores)

| rerun | arms that do not call `b1w` | what changed | installed |
|---|---|---|---|
| 013 `real_plasmode.py`, default ties | identical (1,200 `b1` cells) | rare-label `b1w`: 244 over / 17 / 135 becomes 16 / 17 / 363; 4 mid-rate cells move by a draw | yes, with `plasmode_real.md` |
| 013 `real_plasmode.py --ties random` | identical | rare-label `b1w`: 245 / 10 / 141 becomes 17 / 10 / 369; mid-rate cells identical | yes, with `plasmode_real_randomties.md`, `plasmode_real_d05.md` |
| 013 `real_plasmode.py --ties value` | identical | rare-label `b1w`: 244 / 17 / 135 becomes 16 / 17 / 363; 6 mid-rate cells move by a draw | yes, with `plasmode_real_value.md` |
| 013 `inloop.py --seeds 500` | identical | nothing: all 8,000 runs equal in every field | not needed |
| 014 `plasmode014.py`, `analyse014.py` | identical | 6 `b1w` widths in the last digits; no miss, no class | yes (`plasmode.json`) |
| 017 `harm017.py`, `cards017.py`, `report017.py` | identical | sheet `b1w`: 6 cells at the 1.3% rate go from 0.001-0.008 to 0.000; range 0.001-0.059 becomes 0.000-0.059; still 1 / 1 / 16 | yes (`harm.json`, `harm.md`, six rows of `results.md`) |
| 019 `bounds019.py` | identical | nothing | not needed |
| 020 `plasmode020.py --reps 400` | identical to the last float digit | nothing | not needed |
| `scripts/stratppi_baseline.py` | identical, parts A and B | part A, delta 0.05, 40 cells: `b1w` 9 / 3 / 28 becomes 0 / 3 / 37 (largest miss 0.443 to 0.054); pooled Wilson 10 / 0 / 30 becomes 1 / 0 / 39 (0.448 to 0.057). Judge-strata `b1w` unchanged (largest miss 0.049 and 0.053) | yes (`results/paper/stratppi.json`, `.md`) |

`cert020.py` reads the raw AgentDojo runs, which are not on this machine. It does not need a
rerun: the bug needs a pipeline with no success, and the fewest is 7.

Not regenerated, and now stale in their rare-label `b1w` cells: `validity_real.md`,
`validity_H8.md` and the README of spike 013 (hand-assembled), and `check_b1.md`.

### 4.2 Could not be completed here

- **`scripts/stratppi_validate.py`.** Ran (task 1 kept from the stored file, since it needs the
  off-repo reference environment). Part A reproduces the stored file exactly outside `b1w`; over
  the whole file 39 `b1w` cells and 9 pooled-Wilson cells go from over to at or under. Part B on the
  raw-logit refusal pool does **not** reproduce on this machine: 130 rows of the StratPPI and
  PPBoot arms differ, by at most 0.011 in miss, and a few change class. The stored file is
  therefore left as it is. Rerun on the PC: `OMP_NUM_THREADS=1 .venv/bin/python
  scripts/stratppi_validate.py` (about 20 minutes).
- **`scripts/validity_recount.py`.** Reads result files that are not in git
  (`.planning/spikes/004-forbidden-capability/results.json` is the first). It also asserts the
  old result (line 688: every rare-label cell at n_s 100 is over) and that its quotations are in
  the paper word for word (line 942), and it hard-codes `verdict="fails"` for the rare rows
  (lines 219, 221). It has to be edited together with the paper text.
- **Two stored files that current code does not reproduce**, with or without the fix:
  `results/spikes/013/real_plasmode_truthhalf.json.xz` and `..._truthhalf_swap.json.xz` (written
  by an earlier harness; they lack the `truth_half` field) and
  `results/spikes/013/bandit_plasmode.json.xz` (`b1` differs too). Left untouched.

## 5. Audit: other blind spots

Three independent read-only audits. Each item was reproduced by the auditor on a concrete input
unless marked otherwise. They have not been fixed.

### 5.1 Could void a guarantee (`seldonian/llm`)

1. **A NaN can pass the safety test.** `seldonian/llm/policy.py:330`: `max(g.values())` drops a
   NaN unless it is the first value. With constraints `[ok, harm]` and a NaN threshold on `harm`,
   `fit()` returned the policy; with the order reversed it returned NSF. A NaN threshold arises
   from `calibrate()` on an empty stratum (`calibration.py:84-92`; `youden()` at line 45 tests
   `j <= 0`, which is false for NaN).
2. **The default bound is Student's t, which returns U = 0 at zero positives.**
   `seldonian/bounds.py:172`; defaults at `policy.py:77`, `constraints.py:190, 240, 261`,
   `scripts/run_llm_rl.py:76`. Same class as the `b1w` bug. Exact pass probability of a violating
   policy at delta 0.05: 0.366 at n 100, p 0.01, tau 0.005; 0.223 at n 299, p 0.005, tau 0.004.
3. **Relative thresholds ignore the sampling error of the reference rate.** `policy.py:294`.
   With reference rate 0.034 measured on 256 samples, margin 0.03, n_s 299 and Clopper-Pearson,
   a policy just above the true threshold passes with probability 0.082 against delta 0.05. The
   paper states this limitation (section 4.2); the code offers no corrected option.
4. **The safety test is not strictly one-shot.** `policy.py:327-336`: the counter increments
   only after sampling and judging succeed, so an exception allows a second test. With the set
   sealed, `evaluate(prompts_s)` and `constraint_values(prompts_s, ...)` still return safety-set
   values (`policy.py:269-276`).
5. `calibration.py:56-60`, `judge_threshold` with lower confidence limits is anti-conservative
   (the threshold decreases in specificity). Not called today.
6. `learned_miller_thomas`, whose coverage is conjectured, is selectable for the safety test
   (`policy.py:25`). No violation found.

### 5.2 Wrong bound on a degenerate input (spikes and scripts)

7. **`scripts/p9_certificate.py:128` (`pool_upper`), and `cert017.py:205` through
   `new_prompts_upper`.** Zero-variance bootstrap resamples go to `-inf` only when `eb < est`.
   For paired differences with a negative estimate they go to `+inf` and leave the lower tail,
   so the limit is tight instead of vacuous. With P(-1) = 0.004 and P(+1) = 0.008 the bootstrap-t
   limit missed 0.113 at delta 0.05 (the betting bound: 0.001). Affects columns (a'), (b1), (b2)
   of the human-terms certificate once it is run; no certificate file exists yet.
8. **`cert017.py:115-116` (`youden`) and `133-134` (`answer_aware`).** A negative
   Rogan-Gladen value is clipped to U = 0, and `certify_carried` then certifies. It happens when
   the carried false-alarm rate contradicts the target's flag rate, which is when the carry-over
   assumption has failed. The bound should refuse or return 1.
9. **`cert020.py:121` (`twoway_t`) and `scripts/agentdojo_recheck.py:159` (`twoway_arr`).** The
   basic bootstrap limit returns 0 at zero successes. No real pipeline is affected (the fewest
   successes is 7).
10. `stratppi_baseline.py:70` and `stratppi_validate.py:175`: a stratum of size 1 gives a NaN
    bound, and `(ub < truth).mean()` scores NaN as no miss. Latent: every allocation floors at 2.
11. `cert020.py:94`, `agentdojo_recheck.py:117`: a single cluster divides by zero (a crash, not a
    wrong value).

### 5.3 The validation harness

12. **Sampling without replacement from a small pool.** `plasmode.py:105,163`, and the same
    draws in `stratppi_baseline.py` and `stratppi_validate.py`: 100-200 prompts are drawn from a
    pool of 500 with truth the pool mean, and no bound has a finite-population correction, so
    misses come out low. Redrawn with replacement within the same strata (5,000 reps):

    | bound, mid-rate labels | delta | stored: over / unresolved / at or under | with replacement |
    |---|---|---|---|
    | `b1w`, 24 cells of spike 013 (Table 4 row 6) | 0.05 | 0 / 3 / 21 | 4 / 5 / 15 |
    | `b1w`, same cells | 0.10 | 0 / 0 / 24 | 3 / 6 / 15 |
    | pooled Wilson comparator, 28 cells | 0.05 | 1 / 0 / 27 | 3 / 6 / 19 |
    | StratPPI, normal limit | 0.05 | 9 / 5 / 14 | 14 / 7 / 7 |
    | StratPPI, bootstrap-t | 0.05 | 0 / 0 / 28 | 0 / 0 / 28 |

    "Holds" for `b1w` on reference-rate strata is a statement about a finite pool sampled at
    20-40%. It is weaker for a safety set that is a small share of the population.
13. **The 322-cell headline counts correlated cells.** The 168 PPI++ bootstrap-t cells are 42
    draw sets times 4 judge features on the same draws and bootstrap seed
    (`plasmode017.py:129-135`); the delta 0.10 `b1w` cells reuse the delta 0.05 draws; the 18
    sheet cells are 9 settings run twice. There are about 170 independent draw sets. The paper
    discloses the last two, not the shared features.
14. **Table 4 is partly hand-carried.** `validity_recount.py:497-532` still builds an older
    19-row table; the AgentDojo, StratPPI-allocation and PPBoot rows come from
    `agentdojo_recheck.json` and `stratppi_validate.json` with no assertion tying them to the
    text. Every count matches today.
15. Plausible, not measured: `ppipp_boot(seed=seed + done)` restarts the stream that drew the
    data (`plasmode017.py:112,135`); `stratppi_baseline.py:156` uses `seed = step + n_s`, which
    collides (0 + 200 = 100 + 100).

### 5.4 Checked and found sound

- Exact enumeration over binomial counts (n in 10, 50, 100, 299; p in 0.005, 0.01, 0.034, 0.05,
  0.5; delta 0.05 and 0.10): Clopper-Pearson, Bentkus, Chernoff-KL and the convex-order bound
  reach at most 0.991 of delta; the betting mixture 0.629; Hoeffding and Anderson 0.215. Only
  Student's t is over (up to 19 times delta).
- No trivial root in the library's root finders at k = 0, k = n, n = 1.
- `cp_upper` and `cp_lower`, `ppi_point`, `ppipp_wilson` (0.0263 at zero positives),
  `ppipp_boot`, `stratppi_point`, `stratppi_boot`, `ppboot`, the cluster bootstrap-t, the
  betting bound on paired differences, the `b2` betting test: correct on the inputs tried.
- Delta is split as delta / k across constraints; the final test uses n = n_s with no inflation;
  checkpoint selection uses candidate prompts only; overlapping prompt ids are rejected.
- All 2,180 classified cells in `results/paper/validity_recount.json` recompute to their stored
  class; stored 013 and StratPPI cells reproduce exactly from the pre-fix code.
- Strata are built from reference labels that are separate from the candidate labels.

## 6. Lessons

- A bound needs unit tests at its endpoints (zero and all positives) against a closed form
  before it enters a coverage study. Three separate functions here return U = 0 at zero
  positives.
- A surprising failure of a standard method is a reason to check a hand calculation before it
  becomes a finding. The check here was one line.
- NaN must fail closed in any pass/fail decision.

## 7. Follow-up on the PC (2026-10-07)

Nothing here needed a GPU. Nothing is committed yet. The user's five decisions of 2026-10-07 on the open points are in the paper plan's log and are reflected in the table of section 7.4; the plan for the review's remaining major points is the plan's section 9.

### 7.1 The two reruns of section 4.2

- **`scripts/stratppi_validate.py`** ran in full, task 1 included (the reference environment is
  on this machine). Every arm other than `b1w` and the pooled Wilson bound is identical to the
  stored file, part B and task 1 as well, so the stored part B does reproduce here. 39 `b1w`
  cells and 9 pooled-Wilson cells go from over to at or under, as section 4.2 predicted; 14
  `b1w` cells of part B (the 1.3% rate) move by at most 0.047 with no change of class. The new
  file is installed.
- **`scripts/validity_recount.py`** was rewritten for the current draft. Its inputs were all on
  this machine. It no longer asserts the old result: the rare-label groups are at both deltas
  (delta 0.05: 0 / 0 / 24 for the two bounds together; delta 0.10: 2 / 1 / 21). It now also
  reads `stratppi_validate.json`, `agentdojo_recheck.json` and `replacement_check.json`, reads
  Table 4 from the draft and asserts all 28 rows against the files (each printed miss rate to
  its printed digits, each count of cells exactly), and asserts 15 groups of count-bearing
  sentences, which must also be in the draft word for word. That closes item 14 of the audit.
  The review of v0.5's wording that the script used to carry is gone; it is in git history.

### 7.2 Item 12, sampling without replacement: measured

`scripts/replacement_check.py` draws each of the 40 cells of part A three ways
(`results/paper/replacement_check.md`): as stored (without replacement, 5,000 draws, the
baseline's seeds; its arms reproduce to the last digit, asserted), with replacement on the
same seeds, and with replacement at 40,000 draws from seeds of its own. With replacement the
pool is unbounded for StratPPI too (its N_h is infinite). The 40,000-draw pass is the one to
quote: at 5,000 draws the largest miss over 28 cells reads too high (0.065 against 0.062) and
cells one point over their level stay unresolved.

| bound, delta 0.05 | labels | stored, 5,000 | with replacement, 5,000 | with replacement, 40,000 | largest miss, 40,000 |
|---|---|---|---|---|---|
| `b1w` | 5 mid-rate, 28 cells | 0 / 3 / 25 | 5 / 5 / 18 | 7 / 1 / 20 | 0.062 |
| `b1w` | 2 rare, 12 cells | 0 / 0 / 12 | 0 / 1 / 11 | 1 / 0 / 11 | 0.055 |
| pooled Wilson | mid-rate | 1 / 0 / 27 | 6 / 3 / 19 | 6 / 0 / 22 | 0.069 |
| Clopper-Pearson on the random draw | mid-rate | 0 / 0 / 28 | 0 / 2 / 26 | 0 / 0 / 28 | 0.050 |
| Wald-t `b1` | mid-rate | 0 / 0 / 28 | 0 / 0 / 28 | 0 / 0 / 28 | 0.038 |
| StratPPI, normal limit | mid-rate | 9 / 5 / 14 | 18 / 4 / 6 | 20 / 1 / 7 | 0.096 |
| StratPPI estimator, bootstrap-t | mid-rate | 0 / 0 / 28 | 0 / 2 / 26 | 0 / 0 / 28 | 0.050 |

- The audit's reading holds, and more strongly at the higher resolution: `b1w` is over its
  level in 7 of 28 mid-rate cells when the pool is large. All seven are on the two labels at
  rates of 65% and above; the 16 cells at 9-18% are at or under delta (largest 0.034). The
  bootstrap-t StratPPI limit and the Wald-t limit are over in none. No label between 18% and
  65% was tested.
- The gains do not depend on the draw: `b1w`'s ESS at n_s 100 is 2.29, 1.38 and 4.73 with
  replacement against 2.39, 1.38 and 4.74 without; the medians over ten cells are 2.16 and
  2.16.
- Control: with replacement the random draw is i.i.d. at the pool's rate, so the miss of the
  pooled Wilson bound and of Clopper-Pearson is a binomial sum. Over 160 cells of the
  40,000-draw pass the simulated miss minus the exact one has mean -0.05 and standard
  deviation 0.94 in standard errors; the largest (2.7) was redrawn twice and came back on the
  exact value.
- On the synthetic i.i.d. grid of spike 013 (`check_b1.md`, now printed to four decimals)
  `b1w` is over in 5 of 36 cells and the Wald-t limit in 10, so neither is clean there.
- In the stored file itself, at the eleven settings of reference samples and strata other
  than the one the paper uses, `b1w` is over in 8 of 528 mid-rate cells, all at n_s 100 on the
  same two labels.

### 7.3 The Wilson limit, exactly

`scripts/wilson_exact.py` enumerates the binomial (`results/paper/wilson_exact.md`). The
one-sided Wilson limit's miss probability is a step function of the true rate, largest just
above each value the limit can take. Just above the zero-count limit it is 0.069 at n 100 and
tends to exp(-z^2): 0.067 at delta 0.05, 0.194 at delta 0.10. Averaged over rates it is under
delta below one half and over it above, which is the pattern of section 7.2. This is the
finding that replaces "Wilson fails at rare rates", and it answers the reviewer's question of
where `b1w` begins to hold: for one stratum there is no such point, only an excess that
shrinks with n.

### 7.4 What was fixed from section 5, and what was not

| item | action |
|---|---|
| 1, NaN passes the safety test | **Fixed.** `policy.py`: a NaN `g` counts as infinite. `calibration.youden` raises on NaN. Tests for both orders of the constraints. |
| 2, Student's t is the default bound | **Changed** (the user's decision, 2026-10-07). A 0/1 rate now defaults to `clopper_pearson` and a bounded score or paired difference to `bentkus`; `run_llm_rl.py --bound` defaults to `clopper_pearson`. The scripts of rounds 1 to 4 pass `--bound ttest`, so their commands reproduce. |
| 3, relative thresholds ignore the reference's sampling error | Not changed. The paper states it (section 4.2). |
| 4, the safety test is not strictly one-shot | **Fixed** for the counter: it moves before D_s is sampled, so a test that raises is spent. `evaluate(prompts_s)` on an unsealed policy still works; not changed. |
| 5, 6 | Not changed (not called; no violation found). |
| 7, paired-difference bootstrap-t with a negative estimate | **Fixed** (the user's decision, 2026-10-07; amendment 2.2 in the paper plan, section 6). A zero-variance resample goes to the lower tail whatever its sign, in `pool_upper` and, through `cert017.ppipp_boot(..., degenerate="low")`, in `new_prompts_upper`. The design check reruns byte-identical. |
| 8, negative Rogan-Gladen clipped to 0 | `certify_carried` now refuses a negative estimate. The estimator in the stored studies is unchanged, so they still reproduce. Measured on the transfer study: refusing moves three of 42 cells, by at most 0.044 in miss, and no cell changes class. |
| 9, two-way bootstrap returns 0 at zero successes | Left as the method under test (the paper lists it as failing) and pinned by a test. Zero-success resamples are at most 0.7%, 1.2% and 5.8% of a cell's draws under the three schemes; taking them out leaves the counts of cells over at 5, 14 and 27. |
| 10, a NaN bound scored as no miss | **Fixed** in both StratPPI harnesses. All three reruns are identical, so no NaN had occurred. |
| 11, a single cluster divides by zero | Not changed (a crash, not a wrong value). |
| 12 | Measured: section 7.2. |
| 13, the 322 cells are correlated | Disclosed in the paper (reading 3 of section 7.1): 42 sets of draws behind the 168 PPI++ cells, shared seeds between rows. |
| 14, Table 4 partly hand-carried | **Closed**: section 7.1. |
| 15, seed reuse | **Measured for the row it could touch, and it does not matter.** Spike 017's two plasmode files were replayed (identical in every arm) and rerun with a bootstrap stream that cannot coincide with the one that drew the data. PPI++ with a bootstrap-t limit stays at 0 / 12 / 156 over its 168 cells; ten cells swap between unresolved and at or under, the mean change in miss is 0.0001 and the largest 0.008 (0.058 at 1,000 draws is the new largest, still unresolved). Not written to the stored files. The collision of `step + n_s` in `stratppi_baseline.py` is on an arm the paper does not quote. `replacement_check.py`'s 40,000-draw pass puts the label in its seeds. |

Lesson 1 of section 6 is now a test file: `tests/test_bound_endpoints.py` checks every bound
the paper uses at zero positives, and at all positives where that can occur, against a closed
form or the vacuous value, and pins the four bounds that return 0 there (Student's t, the two
normal limits, the two-way bootstrap), all of which the paper lists as failing.

### 7.5 The paper (v0.9.3) and the notes around it

- Table 4: the rare-rate row is corrected and four rows are added for the redraw with
  replacement (`b1w`, the pooled Wilson bound, the bootstrap-t StratPPI limit, and
  Clopper-Pearson as the control). Reading 1 of section 7.1 is rewritten around the correction,
  the exact calculation and the redraw; reading 3 says what the 322-cell count does and does
  not cover. Sections 6.2, 6.3, 6.4, 8.1 (a fourth item: which bound to use depends on the
  pool), 11 (two bullets) and Appendix A follow. The abstract carries one clause on `b1w`.
- From the outside review, four statements were corrected as plain errors: section 5's "a
  method that must return something cannot promise anything" (it can, with a known feasible
  fallback), GRPO "ranking" (it is differences, scaled within the group), the Lagrangian
  saddle-point sentence (convex case), and "a stratifier supplied for free".
- Figure 1 is now drawn from the recount's file, one bar a row of Table 4 and delta.
- Regenerated or corrected: `check_b1.md`, `validity_H8.md`, the first table of
  `validity_real.md`, the README of spike 013 (a corrections block, both copies), the
  spike-findings skill and its safety-set reference, and the rare-harm seed.

### 7.6 An independent check of the new text

A fresh agent, shown none of the earlier reviews, checked every number in the rewritten
passages against the files and read the two new scripts. It confirmed all 28 rows of Table 4
and the exact Wilson numbers by its own recomputation, and found two high and eight medium
problems in the text. All are acted on:

- the definitions of the two Wilson-type bounds in section 6.3 said "smallest m >= estimate",
  which admits the trivial root that was the bug (now "m > estimate", in the docstring too);
- the stratification gains of section 8.1 and the abstract (1.4 to 5.3) are at delta 0.10 and
  were quoted beside a 5% level (both deltas are now given: 1.4 to 5.1 at 0.05);
- "5 of 28" and "1.5 points" were artefacts of 5,000 draws (the 40,000-draw pass above);
- StratPPI kept the finite pool's N_h under replacement (now unbounded);
- the guidance on which bound to use went beyond its cells (now stated as tested ranges, with
  the Wald-t limit named beside the bootstrap-t one, and the synthetic grid's counts given);
- "a little over their level" left out the Wilson limit's larger excess at high rates;
- the 322-cell count is for one setting of the strata (the other settings are now given);
- the end-point tests have two recorded exceptions, the frozen paired limits and the clipped
  carried estimate, which the limits section now names.

The audit's own report is `.planning/paper-certification/review/audit_v0.9.3.md`.

### 7.7 A blind review of the corrected paper

A second fresh agent read only the clean copy and its figures
(`.planning/paper-certification/review/blind_review_v0.9.3.md`). Verdict: **weak accept,
borderline, mean 3.83 of 5** (evidence relevance 4, falsifiability 4, scope calibration 4,
argument coherence 3, exploration integrity 5, methodological rigor 3). The same procedure
gave 3.67 for v0.9 and 3.17 for v0.3; the outside review by `gpt-6-astra` of v0.9.2 was a
reject. It recomputed the Wilson numbers, the zero-count sample sizes, Tables 6 to 10 and the
row sums of Table 4, and found them right.

Its major points, none of which a text edit closes:

1. The top of the headline gain (5.1 to 5.3) is on the 65% label, which is where `b1w` is over
   its level for a large pool. The abstract and section 8.1 now say so and give the gain under
   the two limits that kept their level there (2.6 and 3.6 for the Wald-t limit, 1.5 and 5.5
   for the bootstrap-t StratPPI limit, at n_s 100 and 200).
2. Every positive recommendation (the strata rule, the allocation, which bound at which rate)
   was chosen on the cells that validate it. A pool that played no part in the choice needs
   new generations, so a GPU run and a go-ahead.
3. The three-class rule under-reports uncertainty; interval estimates of each miss rate would
   say more. The recount's JSON already holds a 95% interval for every cell.
4. Single pools, single seeds, one author-annotator, reimplemented baselines.
5. The robot benchmark (section 10.1) applies a binomial bound to 120 trials without saying
   what is sampled.
6. Section 10.2 does not engage the literature on two-way clustering (multiway cluster
   variance, the pigeonhole bootstrap).
7. No trained policy is certified in human terms.

Of the eleven inconsistencies it listed, the plain errors are corrected (a figure subtitle
that still said no approximate bound holds at rare rates; "9 of 26" against Table 4's 28; a
bold and a plain 0.056 in Table A1; the source of the 2.6-2.7 medians; the abstract's "no
exact bound" against the per-pair row). Three are left in the draft's open checks.

