# Seldonian core vs. Thomas et al. (2019): math audit (2026-09-27)

Scope: the classic (non-LLM) package, `seldonian/{bounds,objectives,seldonian,algorithm}.py`,
against *Preventing undesirable behavior of intelligent machines* (Science 366:999), its
supplement, and the quasi-Seldonian algorithm it describes: split D into a candidate set
D_c and a safety set D_s; choose theta_c on D_c against a *predicted* safety test; run
the safety test on D_s; return theta_c or No Solution Found.
The additional bounds (Bentkus, betting, ...) and `seldonian/llm/` are out of scope.

## Verified correct

| Paper element | Code | Check |
|---|---|---|
| Student-t upper bound: `mean + s / sqrt(n) * t_{1-delta, n-1}`, `s` with ddof 1 | `ttest_bounds` (numpy: `std(ddof=1)`; torch: `torch.std`, unbiased) | read |
| Candidate prediction: the same bound computed as if with `|D_s|` samples, width doubled | `ttest_bounds(..., n=|D_s|, predict=True)`; `_pack` doubles for the other bounds; `_safetyTest(predict=True)` passes `self.X_s.shape[0]` | read |
| Safety test on D_s only, candidate rejected if any `g_i` upper bound > 0 | every `fit` calls `_safetyTest(ub=True) > 0 -> None`; `safetyTest()` is `<= 0` | read |
| Per-constraint `delta_i` | each `g_hat` carries its own `delta` | read |
| Interval propagation through `g` expressions (sum, product, quotient, abs) | `RandomVariable` operators | read; `abs` gives lower 0 even when the interval excludes 0 (valid, slightly loose) |
| Confidence budget across base quantities | `|TPR_a - TPR_b|` uses 4 one-sided endpoints at `delta / 4` each (union bound) | Monte Carlo: miss 0.000-0.004 against delta 0.1 for t-test, Hoeffding, Clopper-Pearson |
| Subgroup counts in the predicted test | `_subgroup_n`: `|D_s| * |A_c| / |D_c|` when predicting, actual count on D_s | read |
| PDIS (RL example): `sum_t gamma^t (prod_{j<=t} pi_e/pi_b) r_t` | `estimate_vec`: `cumprod(pi_e * gamma / pi_b) / gamma` = `gamma^t prod rho` | algebra |
| RL constraint as a performance floor | lower bound on the PDIS mean at `delta`, `n = |D_s|`, predict-inflated on D_c | read |

## Findings

**1. Hoeffding ignores the range of the samples (bug; affects the RL classes).**
The paper's bound is `mean + (b - a) sqrt(ln(1/delta) / (2n))`. `hoeffdings_bounds`
has no `a, b` and always assumes width 1, although `DISTRIBUTION_FREE_BOUNDS` lists it.
`_resolve_bound` exempts `'hoeffdings'` from needing `bound_range`, so
`PDISSeldonianPolicyCMAES(bound='hoeffdings')` runs silently on PDIS returns (unbounded
above, and not in [0, 1]) with a far too narrow interval: on returns with sd 3 it gave
+-0.055. Passing a range instead raises `TypeError`. The fairness constraints are
unaffected (0/1 indicators). Fix: give `hoeffdings_bounds` the `a, b` parameters and the
`_unit_interval` / `_pack` path the other bounds use, and require `bound_range` for it in
`_resolve_bound`. (Importance-weighted returns have no finite upper bound without weight
clipping, so Hoeffding on PDIS needs a clipped estimator anyway, as the paper notes.)

**2. `stratify=True` chooses the split by looking at the safety set (guarantee broken).**
`SeldonianAlgorithmLogRegCMAES` and `LogisticRegressionSeldonianModel` with
`stratify=True` try 30 or 50 random splits and keep the one whose `g` on D_s is closest to
`g` on D_c (for a fixed random theta). D_s is then chosen by a rule that depends on D_s,
so it is no longer an independent sample and the safety test's `1 - delta` does not hold.
The paper requires D_s to be untouched until the safety test. Default is `False`. Fix:
remove the option, or select the split by a statistic computed without D_s's labels, or
report it as a heuristic without the guarantee.

**3. The candidate-selection barrier is soft (differs from the paper; quality, not validity).**
The paper's candidate objective puts every predicted-infeasible theta below every
predicted-feasible one (a large constant plus the predicted bound, which also guides the
search back to feasibility). Here the loss is `log_loss + 10000 * max(0, ub_pred)`: a
theta predicted to fail by less than about `(L_feasible - L_infeasible) / 10000` can win,
so the returned candidate may be one the predicted test already expects to fail.
`hard_barrier=True` (and the PDIS classes' flat `+10000`) is a true barrier, but gives no
gradient toward feasibility. The safety test still protects the guarantee; this only
costs solutions. Fix: `loss + (1e4 + ub_pred if ub_pred > 0 else 0)`.

**4. Known limit of the t-test, faithful to the paper.** On unanimous 0/1 subgroups the
sample sd is 0 and the interval has zero width, so the bound equals the point estimate
(the docstring already says so). The t-test is the paper's *quasi*-Seldonian choice
(CLT-based); the Monte Carlo above shows a 0.4% miss at p = 0.97, n = 40, within delta
but not guaranteed. Use `clopper_pearson` for rates when an exact guarantee is needed.

**Not in the paper, checked for consistency only.** The gradient classes
(`LogisticRegressionSeldonianGD`, `NeuralNetSeldonianGD`) replace black-box candidate
selection with a Lagrangian on a soft surrogate (`lambda^2 * g`, dual step `2 lambda g`,
the correct gradient). They use only a validation slice of D_c, never D_s, and keep the
hard safety test, so the guarantee is untouched.
