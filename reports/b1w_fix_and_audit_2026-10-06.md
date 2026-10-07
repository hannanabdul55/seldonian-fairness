# `b1w` zero-count bug, its effect on the certification paper, and a code audit (2026-10-06)

Scope: the stratified Wilson-type bound `b1w` of spike 013, every result that calls it, and a
read-only audit of the bound implementations, the `seldonian/llm` safety-test path and the
validation harness behind Table 4 of `reports/paper_certification.md`.

Status: the bound is fixed and checked, and every caller that could be rerun on a CPU has been
(section 4). **Still open:** `scripts/stratppi_validate.py` and `scripts/validity_recount.py`
need the PC (section 4.2), and the paper text is unchanged (section 3). None of it needs a GPU.

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
