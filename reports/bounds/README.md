# Confidence bounds for the safety test: study and validation

The Seldonian safety test certifies a constraint such as `|TPR_a - TPR_b| <= 0.2` by
upper-bounding the true (population) value of the constraint from a held-out sample. The
whole safety guarantee rests on that bound: if the bound fails with probability larger than
`delta`, the algorithm is not Seldonian. This directory contains

* a study of alternative bounds on the mean of a bounded random variable
  (implemented in `seldonian/bounds.py`),
* a new two-sample bound for the rate *difference* that the fairness constraint actually
  needs (`seldonian.bounds.bentkus_diff_bounds`),
* an exact (enumeration-based) validation framework (`seldonian/bounds_eval.py`,
  `scripts/validate_bounds.py`) and its results (`*.md`, `*.json`, `*.png` in this folder).

Every bound uses the same contract: each endpoint is a *one-sided* bound at level `delta`,
`P(mean < lower) <= delta` and `P(mean > upper) <= delta`, matching the existing
`ttest_bounds`. The constraint objectives budget `delta` across endpoints accordingly.

## 1. What is wrong with the current bounds

**Student t (`ttest`)** is asymptotic. On indicator data (the per-sample true-positive
indicators that define a TPR) two things go wrong:

* when the rate is near 0 or 1 the sample is unanimous with substantial probability, the
  sample standard deviation is 0 and the "interval" is a point. With `n = 20` and
  `p = 0.02` the upper endpoint misses the mean 67% of the time at `delta = 0.05`;
* for skewed data the t quantile is simply the wrong quantile: the exact worst-case
  coverage over two-point laws (section 3) stays below `1 - delta` for every `n` tested.

**Hoeffding (`hoeffdings`)** is valid but ignores the variance and the location of the
mean: its width is `sqrt(ln(1/delta) / 2n)` whatever the data. For a rate of 5% at
`n = 100` it is about 3.5x wider than an exact binomial bound. Wide bounds mean
"No Solution Found" far more often than necessary.

## 2. Candidate bounds

| name | guarantee | idea |
|---|---|---|
| `clopper_pearson` | exact, Bernoulli only | invert the binomial CDF |
| `bentkus` | distribution-free on [a, b] | Bentkus (2004) Thm 1.2: `P(S >= s) <= e * P°(Bin(n, mean) >= s)` |
| `empirical_bentkus` | distribution-free | Bentkus Thm 1.1 (constant e²/2) with a high-probability variance bound |
| `chernoff_kl` | distribution-free | Hoeffding (1963) Thm 1 in KL form |
| `empirical_bernstein` | distribution-free | Maurer & Pontil (2009) Thm 4 |
| `anderson` | distribution-free | one-sided DKW envelope of the CDF (Anderson 1969) |
| `learned_miller_thomas` | **conjectured** | arXiv:1905.06208, order-statistic quantile bound |
| `betting` | distribution-free | Waudby-Smith & Ramdas (2023) hedged capital, predictable plug-in bets |
| `betting_mixture` | distribution-free | this work: fixed-`n`, order-free mixture-of-bets e-value |
| `bentkus_diff` | distribution-free, two-sample | this work: Bentkus Thm 1.1 on the pooled centred summands of two groups |
| `convex_order` | distribution-free | this work: the optimal convex-order binomial bound (section 2.4) |
| `convex_order_diff` | distribution-free, two-sample | this work: same argument with a two-binomial comparison variable |

`P°` is the log-linear interpolation of the binomial survival function between integers
(Bentkus' log-concave hull), which is what makes the inequality valid at non-integer
thresholds.

### 2.1 Why Bentkus is the right one-sample bound for indicator data

Hoeffding's KL bound is the Chernoff bound of a binomial; the exact binomial tail is
smaller than its Chernoff bound by a factor of order `sqrt(n)`. Bentkus' inequality
compares directly with the binomial tail and pays only a constant `e`. For Bernoulli data
the resulting bound is *exactly* Clopper-Pearson at level `delta / e`; for every other law
on `[0, 1]` it remains valid, because a two-point law is the worst case for the tail of a
sum of bounded variables. At `delta = 0.05` this costs one binomial quantile step
(`0.05 -> 0.0184`), roughly a z-score of 2.09 instead of 1.64, versus 2.45 for any
e-value / Chernoff type bound.

### 2.2 The mixture betting bound (`betting_mixture`) and where it comes from

For a hypothesised mean `m` and a bet fraction `u in (0, 1]`,
`K_u(m) = prod_i (1 + u (X_i - m) / m)` is non-negative and has expectation
`(1 + u (mu - m) / m)^n <= 1` whenever `mu <= m`. Any mixture `K(m) = sum_j w_j K_{u_j}(m)`
inherits `E K(m) <= 1`, so `P(K(m) >= 1/delta) <= delta` (Markov) and the smallest `m` with
`K(m) < 1/delta` is a valid lower confidence bound. `K` is non-increasing in `m`, so a root
finder computes it. This is a discrete universal-portfolio (Cover) capital evaluated at a
fixed time, rather than a regret bound on it, so nothing is lost to analysis.

*Derivation of the form.* Draw `B_i ~ Bernoulli(X_i)` independently: `B_i` is exactly
`Bernoulli(mu)`, so any Bernoulli e-value applies. The Beta-mixture likelihood ratio
`E_q[prod (q/m)^{B_i} ((1-q)/(1-m))^{1-B_i}]` is one; averaging it over the auxiliary
coins (Rao-Blackwellisation, which can only tighten Markov's inequality by Jensen) gives
`E_q prod_i (1 + (q - m)(X_i - m) / (m(1-m)))`, i.e. precisely the mixture over
`lambda = (q - m)/(m(1-m))` of the product bets above. The mixture bound is therefore the
derandomised binarised likelihood-ratio test, which explains both its validity and its
limitation: like every e-value it inherits Markov's `sqrt(2 ln(1/delta))` instead of the
exact `z_delta`, so it cannot beat Bentkus on Bernoulli data. Its advantages are

* **order invariance**: the sequential betting bound of Waudby-Smith & Ramdas changes
  with the order of the samples (section 6 measures how much an adversary gains by
  re-shuffling the safety set); the mixture bound is a function of the multiset;
* **exact prediction**: the bound depends on the data only through the empirical
  distribution, so "what would this bound be with `n_safety` samples" is a rescaling of the
  average log growth (used by the candidate selection with `predict=True`);
* **variance adaptivity**: bets larger than the worst-case Kelly bet
  `lambda_ref = sqrt(2 ln(1/delta) / (n m (1 - m)))` are in the mixture, so low-variance
  data get the empirical-Bernstein-like width that Bentkus' Theorem 1.2 cannot give.

The prior puts half its mass on `lambda_ref` (the worst-case-variance bet, so that the loss
relative to the plug-in strategy is `ln 2` in the common indicator case) and spreads the
rest geometrically up to the cap `lambda = 1/m`.

### 2.3 The two-sample bound (`bentkus_diff`)

The constraint needs an upper bound on `|mu_a - mu_b|`. The library used to combine two
one-sample intervals at `delta/4` per tail; the widths add, whereas the standard error of
the difference is `sqrt(var_a/n_a + var_b/n_b)`.

Write `M = (mean_a_hat - mu_a) - (mean_b_hat - mu_b)` as the sum of `N = n_a + n_b`
independent centred terms `(X_a,i - mu_a)/n_a` and `(mu_b - X_b,j)/n_b`. Each is bounded
above by `bb = max((1 - mu_a)/n_a, mu_b/n_b)` and their average variance is at most
`var = (mu_a(1-mu_a)/n_a + mu_b(1-mu_b)/n_b)/N` (the variance of a [0,1] variable with mean
`mu` is at most `mu(1-mu)`). Bentkus' Theorem 1.1 then gives
`P(M >= x) <= (e²/2) P°(S_N >= x)` where `S_N` sums `N` copies of the two-point variable
with mean 0, variance `var` and maximum `bb`; that is `P(Bin(N, q) >= (x bb + N var)/(bb² + var))`
with `q = var / (bb² + var)`. The unknown means are nuisance parameters: for each
hypothesised difference `d0` the tail is maximised over the feasible `mu_a` (dense grid
plus local refinement) and the bound is the smallest `d0` whose worst-case tail exceeds
the budget. Maximising over *all* feasible means throws away the observed rates (a 5% rate
has a twentieth of the worst-case variance), so the means are first localised with
one-sample Bentkus boxes that use 20% of `delta`, and the tail bound runs on the remaining
80% (Berger & Boos 1994: the failure probability is at most the sum). As with the
one-sample bound, Hoeffding's inequality for the same weighted sum is taken as a free
pointwise minimum. The result has the two-sample z-test's `sqrt(var_a/n_a + var_b/n_b)`
scaling with a distribution-free guarantee, at about 13 ms per evaluation.

### 2.4 A new formula: the optimal convex-order binomial bound

Hoeffding's KL bound and Bentkus' inequality are both instances of one argument.
(1) *Convex ordering.* For any convex `f`, replacing a `[0, 1]`-valued summand by a
Bernoulli variable with the same mean cannot decrease `E f` (because
`f(x) <= (1 - x) f(0) + x f(1)` on `[0, 1]`); applied summand by summand,
`E f(stat) <= E f(T)` where `T` is the statistic built from independent Bernoulli variables
with the same means. (2) *Markov on a convex majorant.* For any convex `f >= 1[s >= x]`,
`P(stat >= x) <= E f(stat) <= E f(T)`. Chernoff picks `f = exp(lambda (s - x))`; Bentkus picks
hinge functions and then pays a factor `e` to state the result in closed form.

Every convex majorant of the indicator lies above some hinge `(s - t)_+ / (x - t)` with
`t < x` (take the supporting line at `x`), so the tightest bound the argument can give is

    P(stat >= x) <= inf_{t < x} E[(T - t)_+] / (x - t).

`E[(T - t)_+]` is a finite sum over the support of `T`, so the bound is evaluated exactly
and any grid of `t` values is valid (each `t` alone is a bound). Consequences:

* **one sample** (`convex_order`): `T = Bin(n, m)/n`. Bentkus' Theorem 1.2 states that
  this infimum is at most `e * P°(Bin(n, m) >= n x)`, so the new bound is never looser
  than `bentkus`, and because `exp(lambda(s - x))` is itself a convex majorant it is never
  looser than `chernoff_kl` either. No interpolation is needed at non-integer `n x`.
* **two samples** (`convex_order_diff`): `T = Bin(n_a, mu_a)/n_a - Bin(n_b, mu_b)/n_b`, the
  exact law of the statistic under Bernoulli data. Unlike `bentkus_diff`, whose single
  two-point comparison variable averages the two groups' variances, a unanimous group keeps
  its `mu^n` tail, which is what the earlier bound lost at extreme rates. With
  `h(u) = E[(u - B)_+] = u F_B(u) - E[B; B <= u]` from prefix sums over the support of `B`,
  `E[(A - B - t)_+] = sum_k p_A(k) h(k/n_a - t)` costs `O(n_a log n_b)` per hinge. The
  nuisance means are localised with one-sample convex-order boxes (20% of the budget,
  Berger-Boos) and the tail is maximised over the box.
* The bound depends on the data only through the observed means and sample sizes, so the
  candidate-selection prediction (`n = n_safety`) is exact, the bound is order-free, and
  it is a smooth function of the means (implicit differentiation gives gradients for the
  gradient-based models if ever needed).

The same argument covers any linear combination of independent group rates (e.g.
`TPR_a - TPR_overall` or weighted parity gaps): only the comparison variable `T` changes.

## 3. Validation methodology

Monte Carlo cannot certify a `delta = 0.05` guarantee: distinguishing 5.0% from 5.5%
miscoverage needs ~10⁵ replications per cell, and the worst case has to be found first.
`seldonian/bounds_eval.py` therefore uses

1. **Exact enumeration on two-point laws.** A sample from `P(X = v1) = p, P(X = v0) = 1-p`
   is a multiset determined by the binomial count `k`, so coverage is
   `sum_k Bin(n, p; k) * 1[bound_k covers]` — computed to machine precision for every `p`
   on a 441-point grid and six atom pairs, for `n` from 5 to 1000. Two-point laws are the
   extreme points of the set of laws with a given mean and range and are where mean bounds
   break first. Order-dependent bounds are averaged over random orderings of each multiset
   (the conditional law of the order under i.i.d. sampling).
2. **Exact enumeration on three-point laws** (`n <= 40`, 66 probability vectors per atom
   set) to check that nothing special about two atoms is being exploited.
3. **Monte Carlo on continuous laws** (uniform, several Betas including U-shaped and
   heavily skewed, a spike-and-slab with 95% zeros, a clipped log-normal that mimics
   importance weights), 2000 replications, with a 99% Clopper-Pearson upper limit on every
   miscoverage estimate so that "no failure seen" carries an error bar.
4. **Exact enumeration of the two-sample constraint bound** over both binomial counts,
   minimised over a 25x25 grid of `(p_a, p_b)`.
5. **Shuffle-hacking**: miscoverage of the best of 20 orderings of the same sample.
6. **End-to-end**: solution rate, true violation rate (200k fresh samples) and accuracy of
   `LogisticRegressionSeldonianGD` per bound and sample size, 40 seeds each.

Tightness is reported as the expected one-sided slack `E[upper - mean]`, which is the
quantity that determines how often a satisfiable constraint is certified.

## 4. Results

### 4.1 Exact worst-case coverage (two-point laws, one-sided, delta = 0.05)

Minimum over six atom pairs and 441 mixing probabilities (`exact_two_point.md`):

| method | n=5 | n=20 | n=100 | n=1000 | verdict |
|---|---|---|---|---|---|
| ttest | 0.005 | 0.020 | 0.095 | 0.632 | invalid at every n |
| hoeffdings / anderson | 0.981 | 0.988 | 0.991 | 0.993 | valid, very loose |
| clopper_pearson (Bernoulli only) | 0.950 | 0.951 | 0.950 | 0.950 | exact |
| bentkus | 0.951 | 0.951 | 0.954 | 0.951 | valid, tight |
| chernoff_kl | 0.951 | 0.951 | 0.954 | 0.951 | valid |
| empirical_bentkus | 0.990 | 0.977 | 0.990 | 0.990 | valid |
| empirical_bernstein | 1.000 | 1.000 | 1.000 | 0.999 | valid, loose |
| learned_miller_thomas | 0.958 | 0.957 | 0.959 | 0.956 | conjecture holds here |
| betting (WSR) | 0.980 | 0.964 | 0.946* | 0.953 | valid (*32-ordering MC; 0.960 with 256) |
| betting_mixture | 0.966 | 0.969 | 0.977 | 0.975 | valid |
| convex_order | 0.951 | 0.951 | 0.954 | 0.951 | valid, tightest distribution-free |

The t-test's worst case is a rare event (`p = 0.001`): unanimous samples give a zero-width
interval. Even at `p = 0.05, n = 100` its one-sided miscoverage is 12%. Three-point laws
(`exact_three_point.md`, n <= 40) and seven continuous laws (`mc_continuous.md`, 2000
replications each) agree: every distribution-free bound holds, the t-test does not
(miscoverage 0.60 at n = 10 on the spike-and-slab law, still 0.068 at n = 1000).

### 4.2 Tightness (expected slack `E[upper - mean]`, one-sided, delta = 0.05)

Bernoulli data, exact:

| method | n=20, p=0.05 | n=100, p=0.05 | n=100, p=0.5 | n=1000, p=0.05 |
|---|---|---|---|---|
| ttest (invalid) | 0.066 | 0.035 | 0.083 | 0.011 |
| clopper_pearson | 0.162 | 0.051 | 0.086 | 0.013 |
| convex_order | 0.168 | 0.058 | 0.101 | 0.016 |
| bentkus | 0.190 | 0.066 | 0.107 | 0.016 |
| betting | 0.246 | 0.067 | 0.104 | 0.016 |
| chernoff_kl | 0.191 | 0.070 | 0.120 | 0.019 |
| betting_mixture | 0.207 | 0.076 | 0.129 | 0.020 |
| hoeffdings | 0.274 | 0.122 | 0.122 | 0.039 |

Continuous data (Monte Carlo, n = 100 / n = 1000):

| method | beta(2,8) | spike-slab | clipped log-normal | uniform |
|---|---|---|---|---|
| ttest (invalid) | 0.021 / 0.006 | 0.030 / 0.010 | 0.033 / 0.010 | 0.048 / 0.015 |
| learned_miller_thomas (conjectural) | 0.033 / 0.007 | 0.047 / 0.011 | 0.047 / 0.011 | 0.054 / 0.015 |
| betting | 0.050 / 0.008 | 0.063 / 0.014 | 0.060 / 0.014 | 0.061 / 0.019 |
| betting_mixture | 0.043 / 0.012 | 0.065 / 0.017 | 0.068 / 0.019 | 0.081 / 0.026 |
| bentkus | 0.098 / 0.028 | 0.063 / 0.015 | 0.087 / 0.024 | 0.108 / 0.033 |
| empirical_bentkus | 0.099 / 0.016 | 0.071 / 0.017 | 0.097 / 0.022 | 0.119 / 0.028 |
| hoeffdings | 0.123 / 0.039 | 0.123 / 0.039 | 0.122 / 0.039 | 0.123 / 0.039 |

Two regimes: on indicator data the exact-binomial family (Clopper-Pearson, convex order,
Bentkus) wins and no e-value method can catch it (Markov's `sqrt(2 ln 1/delta)` versus the
exact quantile); on continuous, lower-variance data the betting bounds are 2-3x tighter
than the binomial family because they adapt to the variance.

### 4.3 The fairness constraint `|TPR_a - TPR_b|` (exact, delta = 0.05)

Expected slack of the certified upper bound (`diff_exact.md`); all rows have coverage >= 0.993:

| method | (0.5,0.5) n=50 | (0.4,0.6) n=200 | (0.05,0.05) n=50 | (0.98,0.98) n=30 | (0.02,0.5) n=(30,150) |
|---|---|---|---|---|---|
| ttest rectangle (invalid) | 0.407 | 0.157 | 0.167 | - | 0.133 |
| clopper_pearson rectangle | 0.397 | 0.155 | 0.187 | 0.191 | 0.112 |
| bentkus rectangle | 0.441 | 0.178 | 0.210 | - | 0.127 |
| bentkus_diff | 0.335 | 0.169 | 0.227 | 0.268 | 0.146 |
| convex_order_diff | 0.313 | 0.162 (n=100) | 0.194 | 0.224 | 0.127 |

`bentkus_diff` is 16-20% tighter than the (invalid) t-test rectangle at moderate rates but
loses at extreme rates; `convex_order_diff` keeps the gain (21% tighter than the
Clopper-Pearson rectangle at (0.5, 0.5), 6-8% tighter than `bentkus_diff` everywhere) and
closes the gap at the extremes to within a few percent of the exact rectangle (section 2.4
explains why). Its minimum exact coverage over the (p_a, p_b) grid is 0.9876; its
(200, 200) table was skipped for cost (about 50 ms per evaluation). Column (0.4, 0.6) for
this row is at n = 100.

### 4.4 Does the safety test still fail?

`false_pass.md` computes exactly the probability that the safety test certifies a classifier
whose true gap is 0.25-0.3 (threshold 0.2). With 8 minority positives and
`(TPR_a, TPR_b) = (0.75, 1.0)` the t-test passes it **10.0%** of the time (delta = 5%);
every distribution-free bound passes it 0.0% of the time. This is the concrete failure the
new bounds remove: a classifier that is perfect on the majority group and mediocre on a
small minority group is certified as fair by the t-test more often than the guarantee
allows.

### 4.5 Order dependence (`shuffle_hack.md`)

The sequential betting bound changes with the order of the samples: re-shuffling the safety
set 20 times and keeping the best ordering raises its one-sided miscoverage from 3% to 19%
(uniform, n = 200). Averaging over 16 permutations or using the mixture bound removes the
effect (spread 0.001 and 0, respectively).

### 4.6 End to end (`seldonian_end_to_end.md`)

With `LogisticRegressionSeldonianGD` on the synthetic task, no method returned an unsafe
solution at any `n` (the constrained models collapse to majority-class predictors when the
budget is tight), so the end-to-end table only measures how soon each bound lets a solution
through: the exact-binomial family certifies solutions from n = 300, Hoeffding only from
n = 2500. Note that the convex-order difference bound costs ~0.1 s per call, which the
gradient model evaluates every epoch; it is meant for the safety test and CMA-ES
candidate selection.


## 5. Recommendations

* **Indicator-based constraints (TPR / recall gaps)**: use `method='convex_order_diff'`
  when the rates are moderate and `n` is not in the thousands, or `'clopper_pearson'`
  (one-sample rectangle) when a group rate is near 0 or 1 or evaluation speed matters. Both
  are valid; the t-test is not, and Hoeffding forfeits a factor 2-4 in width.
* **Bounded continuous quantities** (soft scores, normalised returns): `'betting'` when the
  data order is fixed and trusted, `'betting_mixture'` when the safety test must be a
  function of the sample alone; `'convex_order'` / `'bentkus'` when the data is nearly
  binary.
* **Never** rely on `learned_miller_thomas` for a guarantee: it is the tightest bound in
  every table, but its coverage is a conjecture, and even its Monte Carlo quantile has to
  be made conservative (done here) to avoid a 1% coverage deficit.
* Keep `delta` budgeting as in `objectives._rate_diff_bound`: two tails at `delta/2` for
  the difference bounds, four tails at `delta/4` for rectangles.
* Re-run `uv run python scripts/validate_bounds.py --stage exact --methods <name>` after
  changing any bound; the exact stage is the certificate, Monte Carlo is the sanity check.


## 6. References

* Anderson, T. W. (1969). Confidence limits for the expected value of an arbitrary bounded
  random variable with a continuous distribution function.
* Bentkus, V. (2004). On Hoeffding's inequalities. *Annals of Probability* 32(2).
* Hoeffding, W. (1963). Probability inequalities for sums of bounded random variables.
* Kuchibhotla, A. K. & Zheng, Q. (2021). Near-optimal confidence sequences for bounded
  random variables. UAI.
* Learned-Miller, E. & Thomas, P. S. (2019). A new confidence interval for the mean of a
  bounded random variable. arXiv:1905.06208.
* Maurer, A. & Pontil, M. (2009). Empirical Bernstein bounds and sample variance
  penalization. COLT.
* Phan, M., Thomas, P. S. & Learned-Miller, E. (2021). Towards practical mean bounds for
  small samples. ICML.
* Thomas, P. S. et al. (2019). Preventing undesirable behavior of intelligent machines.
  *Science* 366.
* Waudby-Smith, I. & Ramdas, A. (2023). Estimating means of bounded random variables by
  betting. *JRSS-B*.
