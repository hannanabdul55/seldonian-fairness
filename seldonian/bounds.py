import numpy as np
import torch
import numbers
from scipy.optimize import brentq
from scipy.special import logsumexp, betaincinv
from scipy.stats import t
from scipy.stats import binom as _binom


class RandomVariable:
    """
    Class that works just like any `number` in python, but also calculates the bounds that can be
    accessed with the ``upper`` and ``lower`` member variables. You can use this class just as any
    other number in python and then while these operations are done, bounds are also calculated.

    You can access the value using the ``value`` member variable.
    """
    def __init__(self, value, lower=None, upper=None):
        if not (isinstance(value, numbers.Number) or torch.is_tensor(value)):
            raise ValueError(
                "`value` parameter must be a non-null number")
        self.value = value
        self.upper = upper if upper is not None else value
        self.lower = lower if lower is not None else value
        pass

    def __str__(self):
        return f"( value={self.value}, upper_bound={self.upper}, lower_bound={self.lower} )"

    def __add__(self, other):
        if isinstance(other, numbers.Number) or torch.is_tensor(other):
            other = RandomVariable(other, lower=other, upper=other)

        if self.lower is None or self.upper is None or other.lower is None or other.upper is None:
            return RandomVariable(self.value + other.value)
        return RandomVariable(self.value + other.value, lower=self.lower + other.lower,
                              upper=self.upper + other.upper)

    def __neg__(self):
        if self.lower is None or self.upper is None:
            return RandomVariable(-self.value)
        return RandomVariable(-self.value, upper=-self.lower, lower=-self.upper)

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        if isinstance(other, numbers.Number) or torch.is_tensor(other):
            other = RandomVariable(other, lower=other, upper=other)

        if self.lower is None or self.upper is None or other.lower is None or other.upper is None:
            return RandomVariable(self.value * other.value)

        uu = self.upper * other.upper
        ul = self.upper * other.lower
        lu = self.lower * other.upper
        ll = self.lower * other.lower

        low = min(uu, ul, lu, ll)
        upper = max(uu, lu, ul, ll)
        return RandomVariable(self.value * other.value, lower=low, upper=upper)

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return (-self) + other

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        if isinstance(other, numbers.Number) or torch.is_tensor(other):
            other = RandomVariable(other, lower=other, upper=other)
        return other / self

    def __eq__(self, other):
        if not isinstance(other, RandomVariable):
            return NotImplemented
        return (self.value == other.value and self.lower == other.lower
                and self.upper == other.upper)

    def __truediv__(self, other):
        if isinstance(other, numbers.Number) or torch.is_tensor(other):
            other = RandomVariable(other, lower=other, upper=other)

        if self.lower is None or self.upper is None or other.lower is None or other.upper is None:
            return RandomVariable(self.value / other.value)

        # degenerate divisor [0, 0]: the quotient is unbounded
        if other.lower == 0 and other.upper == 0:
            return RandomVariable(float(np.sign(self.value)) * np.inf if self.value != 0
                                  else np.nan, lower=-np.inf, upper=np.inf)
        # if 0 not in [other.lower , other.upper]
        if other.lower * other.upper > 0:
            return self * RandomVariable(1 / other.value, lower=1 / other.upper,
                                         upper=1 / other.lower)
        # if other.lower is 0
        elif other.lower == 0:
            return self * RandomVariable(1 / other.value, lower=1 / other.upper, upper=np.inf)
        # if other.upper is 0
        elif other.upper == 0:
            return self * RandomVariable(1 / other.value, upper=1 / other.lower, lower=-np.inf)
        # if 0 in [other.lower , other.upper]
        else:
            return RandomVariable(self.value / other.value, lower=-np.inf, upper=np.inf)

    def __abs__(self):
        if self.lower is None or self.upper is None:
            return RandomVariable(abs(self.value))
        # |[lower, upper]| = [0, max(|lower|, |upper|)]
        return RandomVariable(abs(self.value), lower=0,
                              upper=max(abs(self.lower), abs(self.upper)))

    def __hash__(self):
        hash_val = self.value.__hash__()
        if self.lower is not None:
            hash_val += self.lower.__hash__()
        if self.upper is not None:
            hash_val += self.upper.__hash__()
        return hash_val


def min_bounds(*args):
    # min of intervals: every component is the minimum over all arguments
    min_rv = RandomVariable(np.inf, lower=np.inf, upper=np.inf)
    for arg in args:
        if not isinstance(arg, RandomVariable):
            arg = RandomVariable(value=arg)
        min_rv.value = min(min_rv.value, arg.value)
        min_rv.lower = min(min_rv.lower, arg.lower)
        min_rv.upper = min(min_rv.upper, arg.upper)
    return min_rv


def max_bounds(*args):
    # max of intervals: every component is the maximum over all arguments
    max_rv = RandomVariable(-np.inf, lower=-np.inf, upper=-np.inf)
    for arg in args:
        if not isinstance(arg, RandomVariable):
            arg = RandomVariable(value=arg)
        max_rv.value = max(max_rv.value, arg.value)
        max_rv.lower = max(max_rv.lower, arg.lower)
        max_rv.upper = max(max_rv.upper, arg.upper)
    return max_rv


def ttest_bounds(samples, delta, n=None, predict=False):
    """
    Student's t confidence interval on the sample mean (one-sided quantile at
    ``1 - delta``, doubled in width when ``predict=True`` for candidate-selection
    inflation, following Thomas et al. 2019).

    Caveat: on unanimous binary samples (all 0 or all 1) the sample standard
    deviation is 0 and the interval has zero width, so the bound degenerates to the
    point estimate. This is faithful to the published formulation but understates
    uncertainty for Bernoulli rates near 0 or 1.
    """
    if not (isinstance(samples, numbers.Number) or isinstance(samples,
                                                              np.ndarray) or torch.is_tensor(
            samples)):
        raise ValueError("`samples` argument should be a numpy array")
    is_tensor = torch.is_tensor(samples)
    if not is_tensor:
        samples = np.array(samples)

    if samples.ndim > 1:
        raise ValueError(f"`samples` should be a vector (1-D array). Got shape: {samples.shape}")
    if n is None:
        n = samples.numel() if is_tensor else samples.size
    if not is_tensor:
        dev = ((samples.std(ddof=1) / np.sqrt(n)) * t.ppf(1 - delta, n - 1)) * (1 + (1 * predict))
    else:
        dev = ((torch.std(samples.double()) / np.sqrt(n)) * t.ppf(1 - delta, n - 1)) * (
                    1 + (1 * predict))

    if torch.is_tensor(samples):
        samples = samples.double()
    sample_mean = samples.mean()
    return RandomVariable(sample_mean, lower=sample_mean - dev, upper=sample_mean + dev)


def hoeffdings_bounds(samples, delta, n=None, predict=False):
    """

    :param samples:
    :param delta:
    :param n:
    :param predict:
    :return:
    """
    if not (isinstance(samples, numbers.Number) or isinstance(samples,
                                                              np.ndarray) or torch.is_tensor(
            samples)):
        raise ValueError("`samples` argument should be a numpy array")
    is_tensor = torch.is_tensor(samples)
    if not is_tensor:
        samples = np.array(samples)
    if samples.ndim > 1:
        raise ValueError("`samples` should be a vector (1-D array)")
    if n is None:
        n = samples.numel() if is_tensor else samples.size
    dev = np.sqrt(np.log(1 / delta) / (2 * n)) * (1 + (1 * predict))
    if not is_tensor:
        sample_mean = samples.mean()
    else:
        sample_mean = torch.mean(samples.double())
    return RandomVariable(sample_mean, lower=sample_mean - dev, upper=sample_mean + dev)


# ======================================================================================
# Non-asymptotic confidence bounds for the mean of a bounded random variable
# ======================================================================================
#
# Every function below follows the same contract as ``ttest_bounds`` / ``hoeffdings_bounds``:
#
#     bound(samples, delta, n=None, predict=False, a=0.0, b=1.0) -> RandomVariable
#
# * ``samples`` are i.i.d. draws known to lie in ``[a, b]``.
# * Each endpoint is a *one-sided* bound at level ``delta``:
#   ``P(mean < lower) <= delta`` and ``P(mean > upper) <= delta`` separately (the two-sided
#   interval therefore holds with probability ``1 - 2 delta``). This matches the convention
#   of ``ttest_bounds`` and is what ``seldonian.objectives`` budgets for.
# * ``n`` lets the caller evaluate the bound *as if* the empirical distribution of
#   ``samples`` had been observed with ``n`` draws (used for predicting the outcome of the
#   safety test from the candidate set). ``predict=True`` additionally doubles the
#   deviation from the sample mean, following Thomas et al. (2019).
# * Endpoints are clipped to ``[a, b]`` (a valid bound intersected with the support is
#   still valid).
#
# The bounds and where they come from:
#
# ``clopper_pearson_bounds``  exact binomial interval; valid for Bernoulli (0/1) samples only.
# ``bentkus_bounds``          Bentkus (2004, Ann. Probab.) Theorem 1.2: the tail of a sum of
#                             [0,1] variables is at most ``e`` times the (log-linearly
#                             interpolated) binomial tail with the same mean. Equivalent to
#                             Clopper-Pearson at level ``delta / e`` but valid for *any*
#                             distribution on [0, 1]. With ``variance='empirical'`` uses
#                             Theorem 1.1 (constant e^2/2) with an empirical variance bound.
# ``chernoff_kl_bounds``      Hoeffding (1963) Theorem 1 in KL form (Chernoff bound).
# ``empirical_bernstein_bounds``  Maurer & Pontil (2009) Theorem 4.
# ``anderson_bounds``         Anderson (1969) via the one-sided DKW inequality.
# ``learned_miller_thomas_bounds``  Learned-Miller & Thomas (2019, arXiv:1905.06208); its
#                             coverage is a *conjecture* (proved only for Bernoulli and
#                             half-Bernoulli laws) - do not rely on it for safety guarantees.
# ``betting_bounds``          Waudby-Smith & Ramdas (2023) hedged-capital CI (sequential,
#                             predictable plug-in bets, running intersection).
# ``betting_mixture_bounds``  fixed-``n`` betting bound where the bet is integrated over a
#                             prior (a discrete universal-portfolio mixture). Unlike
#                             ``betting_bounds`` the result does not depend on the order of
#                             the samples, so the safety test cannot be "shuffle-hacked",
#                             and the bound depends on the data only through the empirical
#                             distribution, which makes the ``n``-prediction exact.
#                             Section "Derivation" in reports/bounds/README.md explains why
#                             this is the Rao-Blackwellised Bernoulli-likelihood-ratio test.
#
# Two-sample bound (difference of two group rates, the actual fairness constraint):
#
# ``bentkus_diff_bounds``     Bentkus Theorem 1.1 applied to the pooled centred sequence of
#                             two independent samples, giving a bound on ``mean_a - mean_b``
#                             that uses the combined variance instead of the sum of two
#                             one-sample widths.

_E = float(np.e)
_E2_OVER_2 = float(np.e ** 2 / 2)


def _as_float_array(samples):
    if torch.is_tensor(samples):
        samples = samples.detach().cpu().double().numpy()
    elif isinstance(samples, numbers.Number):
        samples = np.array([samples], dtype=float)
    elif isinstance(samples, np.ndarray):
        samples = samples.astype(float)
    else:
        raise ValueError("`samples` argument should be a numpy array")
    if samples.ndim > 1:
        raise ValueError(f"`samples` should be a vector (1-D array). Got shape: {samples.shape}")
    if samples.size == 0:
        raise ValueError("`samples` must contain at least one value")
    return samples


def _unit_interval(samples, a, b):
    """Map samples in [a, b] to [0, 1]; raise if any sample lies outside the range."""
    x = _as_float_array(samples)
    if not b > a:
        raise ValueError(f"range must satisfy a < b, got a={a}, b={b}")
    tol = 1e-9 * (b - a)
    if x.min() < a - tol or x.max() > b + tol:
        raise ValueError(
            f"samples must lie in [{a}, {b}] (observed [{x.min()}, {x.max()}]); pass the "
            "true range with the `a`, `b` arguments")
    return np.clip((x - a) / (b - a), 0.0, 1.0)


def _effective_n(x, n):
    if n is None:
        return int(x.size)
    if n < 1:
        raise ValueError("`n` must be a positive integer")
    return int(n)


def _pack(mean01, lower01, upper01, a, b, predict):
    """Apply prediction inflation, clip to the support and map back to [a, b]."""
    lower01 = min(lower01, mean01)
    upper01 = max(upper01, mean01)
    if predict:
        lower01 = mean01 - 2.0 * (mean01 - lower01)
        upper01 = mean01 + 2.0 * (upper01 - mean01)
    lower01 = float(np.clip(lower01, 0.0, 1.0))
    upper01 = float(np.clip(upper01, 0.0, 1.0))
    scale = b - a
    return RandomVariable(a + scale * float(mean01), lower=a + scale * lower01,
                          upper=a + scale * upper01)


def _binom_logsf_interp(t, n, p):
    """
    ``log P°(Bin(n, p) >= t)`` for real ``t``: the log-concave hull of the binomial survival
    function (Bentkus 2004, eq. 1.8), which is the log-linear interpolation between the
    integer jump points. For integer ``t`` it is exactly ``log P(Bin(n, p) >= t)``.
    """
    if t <= 0:
        return 0.0
    if t > n:
        return -np.inf
    if p <= 0.0:
        return 0.0 if t <= 0 else -np.inf
    if p >= 1.0:
        return 0.0
    k0 = int(np.floor(t))
    frac = t - k0
    # binom.logsf(k - 1) = log P(X > k - 1) = log P(X >= k)
    lo = _binom.logsf(k0 - 1, n, p)
    if frac == 0.0:
        return float(lo)
    hi = _binom.logsf(k0, n, p)
    return float((1.0 - frac) * lo + frac * hi)


def _first_crossing(g, lo, hi, grid=48):
    """
    Smallest root of ``g`` in ``[lo, hi]`` where ``g(lo) < 0 <= g(hi)`` is expected. A coarse
    scan locates the first sign change (so a non-monotone ``g`` still yields the *smallest*
    crossing, which is the conservative choice for a lower bound), then brentq refines it.
    """
    xs = np.linspace(lo, hi, grid)
    prev_x, prev_g = xs[0], g(xs[0])
    if prev_g >= 0:
        return lo
    for x in xs[1:]:
        cur = g(x)
        if cur >= 0:
            if not np.isfinite(prev_g):
                # brentq needs finite endpoints; a -inf tail means the crossing is above
                prev_g = -1e300
            try:
                return float(brentq(g, prev_x, x, xtol=1e-12, rtol=1e-10, maxiter=200))
            except ValueError:
                return float(prev_x)
        prev_x, prev_g = x, cur
    return hi


def clopper_pearson_bounds(samples, delta, n=None, predict=False, a=0.0, b=1.0):
    """
    Exact Clopper-Pearson (1934) binomial bounds. Valid *only* for Bernoulli samples (each
    sample equal to ``a`` or ``b``); anything else raises. For indicator data (e.g. the
    per-sample true-positive indicators in :mod:`seldonian.objectives`) this is the exact
    reference bound: coverage of each endpoint is at least ``1 - delta`` for every ``n``.
    """
    x = _unit_interval(samples, a, b)
    if not np.all((x < 1e-9) | (x > 1 - 1e-9)):
        raise ValueError("clopper_pearson_bounds requires binary samples (values a or b)")
    m = _effective_n(x, n)
    mean = float(x.mean())
    k = mean * m
    lower = 0.0 if k <= 0 else float(betaincinv(k, m - k + 1, delta))
    upper = 1.0 if k >= m else float(betaincinv(k + 1, m - k, 1 - delta))
    return _pack(mean, lower, upper, a, b, predict)


def _variance_upper_bound(x, n, delta):
    """
    High-probability upper bound on the standard deviation of a [0, 1] random variable from
    the unbiased sample variance (Maurer & Pontil 2009, Theorem 10):
    ``P( sd > sqrt(V_n) + sqrt(2 ln(1/delta) / (n - 1)) ) <= delta``.
    """
    if x.size < 2:
        return 0.5
    v = float(x.var(ddof=1))
    sd = np.sqrt(v) + np.sqrt(2.0 * np.log(1.0 / delta) / (n - 1))
    return float(min(sd, 0.5))


def _kl_bernoulli(p, q):
    p = min(max(p, 0.0), 1.0)
    out = 0.0
    if p > 0:
        out += p * np.log(p / q)
    if p < 1:
        out += (1 - p) * np.log((1 - p) / (1 - q))
    return out


def _bentkus_lower01(x, n, delta, variance, delta_var_frac):
    mean = float(x.mean())
    s = mean * n
    if s <= 0:
        return 0.0
    if variance is None:
        # Theorem 1.2: P(sum >= s) <= e * P°(Bin(n, m) >= s). Hoeffding's Theorem 1,
        # P(sum >= s) <= exp(-n KL(s/n || m)), bounds the *same* tail probability, so the
        # pointwise minimum of the two is still a valid tail bound (no union-bound cost);
        # the KL form is exact for unanimous samples where the constant e hurts most.
        def g(m):
            log_tail = min(1.0 + _binom_logsf_interp(s, n, m), -n * _kl_bernoulli(mean, m))
            return log_tail - np.log(delta)
        return min(_first_crossing(g, 1e-12, mean), mean)
    if variance != 'empirical':
        raise ValueError("variance must be None or 'empirical'")
    d1 = delta * delta_var_frac
    d2 = delta - d1
    sd_u = _variance_upper_bound(x, n, d1)
    var_u = sd_u ** 2

    # Theorem 1.1 with the two-point comparison variable eps(sigma^2, b)
    def g(m):
        bb = 1.0 - m
        var = min(var_u, m * (1.0 - m))
        if var <= 0 or bb <= 0:
            return -np.inf if s - n * m > 0 else np.inf
        q = var / (bb ** 2 + var)
        t = ((s - n * m) * bb + n * var) / (bb ** 2 + var)
        return np.log(_E2_OVER_2) + _binom_logsf_interp(t, n, q) - np.log(d2)
    return min(_first_crossing(g, 1e-12, mean), mean)


def bentkus_bounds(samples, delta, n=None, predict=False, a=0.0, b=1.0, variance=None,
                   delta_var_frac=0.25):
    """
    Bentkus (2004) bounds for the mean of a random variable supported on ``[a, b]``.

    ``variance=None`` uses Theorem 1.2: ``P(sum >= s) <= e * P°(Bin(n, mean) >= s)``, so the
    lower bound is the smallest mean ``m`` whose interpolated binomial tail at the observed
    sum exceeds ``delta / e``. This is Clopper-Pearson at level ``delta / e`` but holds for
    *every* distribution on the interval, not just Bernoulli. The tail bound is taken as
    the minimum of Bentkus' and Hoeffding's KL (Chernoff) bound, which is free (both bound
    the same probability) and removes the factor ``e`` for near-unanimous samples.

    ``variance='empirical'`` uses Theorem 1.1 (constant ``e^2 / 2``) with the comparison
    two-point variable's variance set to a high-probability upper bound on the true variance
    (Maurer-Pontil Theorem 10 at level ``delta * delta_var_frac``); the remaining budget
    goes to the tail bound. This adapts to low-variance data at the price of the split.
    """
    x = _unit_interval(samples, a, b)
    m = _effective_n(x, n)
    mean = float(x.mean())
    lower = _bentkus_lower01(x, m, delta, variance, delta_var_frac)
    upper = 1.0 - _bentkus_lower01(1.0 - x, m, delta, variance, delta_var_frac)
    return _pack(mean, lower, upper, a, b, predict)


def _chernoff_lower01(mean, n, delta):
    if mean <= 0:
        return 0.0
    target = np.log(1.0 / delta) / n

    def g(m):
        return target - _kl_bernoulli(mean, m)   # negative for m far below mean
    return min(_first_crossing(g, 1e-12, mean), mean)


def chernoff_kl_bounds(samples, delta, n=None, predict=False, a=0.0, b=1.0):
    """
    Hoeffding (1963) Theorem 1 in its KL (Chernoff) form: the lower bound is the smallest
    ``m`` with ``n * KL(mean || m) <= ln(1 / delta)``. Tighter than the quadratic
    ``hoeffdings_bounds`` near the ends of the support, valid for any distribution on
    ``[a, b]``, but still a Chernoff bound (loses a ``sqrt(n)`` factor versus Bentkus).
    """
    x = _unit_interval(samples, a, b)
    m = _effective_n(x, n)
    mean = float(x.mean())
    lower = _chernoff_lower01(mean, m, delta)
    upper = 1.0 - _chernoff_lower01(1.0 - mean, m, delta)
    return _pack(mean, lower, upper, a, b, predict)


def empirical_bernstein_bounds(samples, delta, n=None, predict=False, a=0.0, b=1.0):
    """
    Maurer & Pontil (2009) Theorem 4 (one-sided): with probability at least ``1 - delta``
    ``mean - sample_mean <= sqrt(2 V_n ln(2/delta) / n) + 7 ln(2/delta) / (3 (n - 1))``,
    where ``V_n`` is the unbiased sample variance. Symmetric, variance adaptive.
    """
    x = _unit_interval(samples, a, b)
    m = _effective_n(x, n)
    if m < 2:
        raise ValueError("empirical_bernstein_bounds needs n >= 2")
    mean = float(x.mean())
    v = float(x.var(ddof=1)) if x.size > 1 else 0.25
    ln = np.log(2.0 / delta)
    dev = np.sqrt(2.0 * v * ln / m) + 7.0 * ln / (3.0 * (m - 1))
    return _pack(mean, mean - dev, mean + dev, a, b, predict)


def _anderson_lower01(x, n, delta):
    z = np.sort(x)
    m = z.size
    eps = np.sqrt(np.log(1.0 / delta) / (2.0 * n))
    gaps = np.diff(np.concatenate(([0.0], z)))          # z_i - z_{i-1}, z_0 = 0
    survival = np.maximum(0.0, 1.0 - np.arange(m) / m - eps)   # 1 - F_hat - eps on [z_{i-1}, z_i)
    return float(np.sum(gaps * survival))


def anderson_bounds(samples, delta, n=None, predict=False, a=0.0, b=1.0):
    """
    Anderson (1969) bound: with probability ``1 - delta`` the CDF satisfies
    ``F <= F_hat + sqrt(ln(1/delta) / (2n))`` everywhere (one-sided DKW with Massart's
    constant), and the mean is the integral of ``1 - F``. Order-statistic based; tight for
    skewed distributions with mass near the ends of the support.
    """
    x = _unit_interval(samples, a, b)
    m = _effective_n(x, n)
    mean = float(x.mean())
    lower = _anderson_lower01(x, m, delta)
    upper = 1.0 - _anderson_lower01(1.0 - x, m, delta)
    return _pack(mean, lower, upper, a, b, predict)


def _lmt_upper01(x, n, delta, draws, seed):
    z = np.sort(x)
    if z.size != n:
        # resample the empirical distribution at n evenly spaced quantiles
        z = np.quantile(z, (np.arange(n) + 0.5) / n)
    z_ext = np.concatenate((z, [1.0]))
    gaps = np.diff(z_ext)                                   # z_{i+1} - z_i, i = 1..n
    rng = np.random.default_rng(seed)
    U = np.sort(rng.random((draws, n)), axis=1)
    induced = 1.0 - U @ gaps
    # Monte Carlo quantile: the k-th order statistic exceeds the true (1 - delta)-quantile
    # with probability >= 0.999 when k - 1 is the 0.999 quantile of Bin(draws, 1 - delta)
    k = int(min(draws, _binom.ppf(0.999, draws, 1.0 - delta) + 1))
    return float(np.partition(induced, k - 1)[k - 1])


def learned_miller_thomas_bounds(samples, delta, n=None, predict=False, a=0.0, b=1.0,
                                 draws=4000, seed=0):
    """
    Learned-Miller & Thomas (2019, arXiv:1905.06208): the upper bound is the ``1 - delta``
    quantile (over sorted uniform order statistics ``U``) of the induced mean
    ``1 - sum_i U_i (z_{i+1} - z_i)`` of the conservative completion of the ordered CDF pairs.
    Coverage is **conjectured**, proved only for Bernoulli / half-Bernoulli laws (for
    Bernoulli data it coincides with Clopper-Pearson). The quantile is estimated by Monte
    Carlo with a fixed seed; the order statistic is chosen so that it exceeds the true
    quantile with probability 0.999 (about ``delta * 0.8`` effective level at 4000 draws),
    otherwise the sampling error alone makes the bound undercover by ~1%.
    """
    x = _unit_interval(samples, a, b)
    m = _effective_n(x, n)
    mean = float(x.mean())
    upper = _lmt_upper01(x, m, delta, draws, seed)
    lower = 1.0 - _lmt_upper01(1.0 - x, m, delta, draws, seed + 1)
    return _pack(mean, lower, upper, a, b, predict)


def _refine_root_decreasing(f, lo, hi, target):
    """Root of ``f(m) = target`` for a non-increasing ``f`` with ``f(lo) >= target > f(hi)``."""
    try:
        return float(brentq(lambda m: f(m) - target, lo, hi, xtol=1e-10, maxiter=100))
    except ValueError:
        return float(lo)


def _wsr_lambdas(x, n, delta, c):
    """Predictable plug-in bets of Waudby-Smith & Ramdas (2023), Remark 3 (fixed n)."""
    t = np.arange(1, x.size + 1)
    csum = np.cumsum(x)
    mu_hat = (0.5 + csum) / (t + 1)
    resid = (x - mu_hat) ** 2
    sig2 = (0.25 + np.cumsum(resid)) / (t + 1)
    sig2_prev = np.concatenate(([0.25], sig2[:-1]))
    lam = np.sqrt(2.0 * np.log(1.0 / delta) / (n * sig2_prev))
    return lam


def _betting_lower01(x, n, delta, grid, c, permutations, seed):
    """Lower bound from the K^+ capital process (bets that the mean is above m)."""
    if x.size != n:
        rng = np.random.default_rng(seed)
        reps = int(np.ceil(n / x.size))
        x = rng.permutation(np.tile(x, reps))[:n]
    if not np.any(x > 0):
        return 0.0
    ms = np.linspace(1e-6, 1.0, grid)
    log_target = np.log(1.0 / delta)

    def log_capital_path(xs):
        lam = _wsr_lambdas(xs, n, delta, c)
        lam_m = np.minimum(lam[:, None], c / ms[None, :])            # (n, grid)
        return np.cumsum(np.log1p(lam_m * (xs[:, None] - ms[None, :])), axis=0)

    if permutations is None:
        # running intersection over t: reject m if the capital ever reached 1/delta
        score = log_capital_path(x).max(axis=0)
    else:
        rng = np.random.default_rng(seed)
        paths = [log_capital_path(rng.permutation(x))[-1] for _ in range(permutations)]
        score = logsumexp(np.stack(paths), axis=0) - np.log(permutations)
    rejected = score >= log_target
    if not rejected[0]:
        return 0.0
    j = int(np.max(np.nonzero(rejected)[0]))
    if j == grid - 1:
        return float(ms[-1])
    # refine between the last rejected grid point and the next one (score is non-increasing)
    def score_at(m):
        if permutations is None:
            lam = _wsr_lambdas(x, n, delta, c)
            path = np.cumsum(np.log1p(np.minimum(lam, c / m) * (x - m)))
            return float(path.max())
        rng2 = np.random.default_rng(seed)
        vals = []
        for _ in range(permutations):
            xp = rng2.permutation(x)
            lam = _wsr_lambdas(xp, n, delta, c)
            vals.append(float(np.sum(np.log1p(np.minimum(lam, c / m) * (xp - m)))))
        return float(logsumexp(vals) - np.log(permutations))
    return _refine_root_decreasing(score_at, ms[j], ms[j + 1], log_target)


def betting_bounds(samples, delta, n=None, predict=False, a=0.0, b=1.0, grid=512, c=0.5,
                   permutations=None, seed=0):
    """
    Hedged-capital confidence bounds of Waudby-Smith & Ramdas (2023) for a fixed sample
    size (their Remark 3), one side at a time. The bettor wagers a predictable fraction
    ``lambda_t = min(sqrt(2 ln(1/delta) / (n sigma_hat_{t-1}^2)), c / m)`` of its capital on
    ``X_t > m``; by Ville's inequality the capital exceeds ``1 / delta`` at any time with
    probability at most ``delta`` when ``m`` is at least the true mean, so every ``m`` at
    which it does is rejected (running intersection over ``t``).

    The bound depends on the *order* of the samples. ``permutations=R`` averages the final
    capital over ``R`` random permutations (a valid e-value by linearity of expectation;
    the running intersection is then dropped), which removes most of the order dependence
    at the cost of ``R`` passes. See :func:`betting_mixture_bounds` for an order-free bound.
    """
    x = _unit_interval(samples, a, b)
    m = _effective_n(x, n)
    mean = float(x.mean())
    lower = _betting_lower01(x, m, delta, grid, c, permutations, seed)
    upper = 1.0 - _betting_lower01(1.0 - x, m, delta, grid, c, permutations, seed + 1)
    return _pack(mean, lower, upper, a, b, predict)


def _mixture_grid(n, m, delta, size, ratio):
    """
    Bet fractions ``u = lambda * m`` for hypothesis ``m`` and their log prior weights.

    The Kelly-optimal bet at the boundary of a level-``delta`` bound is about
    ``lambda* = sqrt(2 ln(1/delta) / (n var))``; the variance of a [0, 1] variable with mean
    ``m`` is at most ``m (1 - m)``, so ``lambda_ref = sqrt(2 ln(1/delta) / (n m (1 - m)))`` is
    the smallest bet worth making and larger bets pay off for lower-variance data. The grid
    runs geometrically from ``lambda_ref / ratio`` up to the cap ``1 / m`` (``u = 1``). Half
    of the prior mass sits on the reference bet (the worst-case-variance regime, which is
    the common case for indicator data), the rest is spread evenly over the other points.
    The grid may depend on ``m``, ``n`` and ``delta`` because validity is pointwise in ``m``.
    """
    lam_ref = np.sqrt(2.0 * np.log(1.0 / delta) / (n * m * (1.0 - m)))
    u_ref = min(m * lam_ref, 1.0 - 1e-9)
    u_lo = max(u_ref / ratio, 1e-12)
    k = int(np.ceil(np.log((1.0 - 1e-9) / u_lo) / np.log(ratio))) + 1
    k = int(min(max(k, 2), size))
    u = np.geomspace(u_lo, 1.0 - 1e-9, k)
    # snap the point closest to the reference bet onto it and give it half the mass
    j = int(np.argmin(np.abs(np.log(u) - np.log(u_ref))))
    u[j] = u_ref
    w = np.full(k, 0.5 / max(k - 1, 1))
    w[j] = 0.5 if k > 1 else 1.0
    return u, np.log(w)


def _mixture_log_capital(vals, counts, n, m, u, log_w):
    """
    ``log sum_j w_j prod_i (1 + u_j (X_i - m) / m)`` with the product written as
    ``n * mean_i log(...)`` so that the empirical distribution can be evaluated at any ``n``.
    ``vals``/``counts`` describe the empirical distribution (unique values and multiplicities).
    """
    # (size_u, n_unique)
    with np.errstate(divide='ignore', invalid='ignore'):
        factor = 1.0 + u[:, None] * (vals[None, :] - m) / m
        logf = np.where(factor > 0, np.log(np.maximum(factor, 1e-300)), -np.inf)
    mean_log = (logf * counts[None, :]).sum(axis=1) / counts.sum()
    return float(logsumexp(log_w + n * mean_log))


def _mixture_lower01(x, n, delta, size, ratio):
    vals, counts = np.unique(x, return_counts=True)
    if not np.any(vals > 0):
        return 0.0
    mean = float(x.mean())
    log_target = np.log(1.0 / delta)

    def score(m):
        m = min(max(m, 1e-12), 1.0 - 1e-12)
        u, log_w = _mixture_grid(n, m, delta, size, ratio)
        return _mixture_log_capital(vals, counts, n, m, u, log_w)
    lo = 1e-9
    if score(lo) < log_target:
        return 0.0
    if score(mean) >= log_target:       # cannot happen (Jensen), kept for safety
        return mean
    return _refine_root_decreasing(score, lo, mean, log_target)


def betting_mixture_bounds(samples, delta, n=None, predict=False, a=0.0, b=1.0, size=12,
                           ratio=1.6):
    """
    Order-free fixed-``n`` betting bound (a discrete universal-portfolio mixture).

    For a candidate mean ``m`` and bet fraction ``u in (0, 1]`` the capital
    ``K_u(m) = prod_i (1 + u (X_i - m) / m)`` has expectation at most one whenever the true
    mean is at most ``m`` (a product of independent non-negative factors with mean
    ``1 + u (mu - m) / m``). Any weighted average over ``u`` therefore also has expectation
    at most one, and Markov's inequality gives ``P(K(m) >= 1/delta) <= delta``. The lower
    bound is the smallest ``m`` for which the mixed capital is below ``1 / delta``; ``K`` is
    non-increasing in ``m`` so a root finder suffices.

    The bet grid is geometric with ``ratio`` between neighbours, from just below the
    worst-case-variance Kelly bet up to the cap ``u = 1`` (at most ``size`` points, see
    :func:`_mixture_grid`). Because the capital is a product over samples the bound is
    invariant to permutations of the data and depends on the sample only through its
    empirical distribution, so evaluating it at another ``n`` simply rescales the average
    log growth. Mixing costs about ``log(2)`` in log-capital relative to the reference bet
    and ``log(2 (k - 1))`` relative to the best of the other bets.
    """
    x = _unit_interval(samples, a, b)
    m = _effective_n(x, n)
    mean = float(x.mean())
    lower = _mixture_lower01(x, m, delta, size, ratio)
    upper = 1.0 - _mixture_lower01(1.0 - x, m, delta, size, ratio)
    return _pack(mean, lower, upper, a, b, predict)


# --------------------------------------------------------------------------------------
# Two-sample bound on a difference of means
# --------------------------------------------------------------------------------------

def _binom_logsf_interp_vec(t, n, p):
    """Vectorised :func:`_binom_logsf_interp` over arrays ``t`` and ``p`` (same shape)."""
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    out = np.full(t.shape, -np.inf)
    k0 = np.floor(t)
    frac = t - k0
    with np.errstate(divide='ignore', invalid='ignore'):
        lo = _binom.logsf(k0 - 1, n, p)
        hi = _binom.logsf(k0, n, p)
    lo = np.where(np.isfinite(lo), lo, -np.inf)
    hi = np.where(np.isfinite(hi), hi, -np.inf)
    with np.errstate(invalid='ignore'):
        val = np.where(frac > 0, (1.0 - frac) * lo + frac * hi, lo)
    val = np.where(np.isnan(val), -np.inf, val)
    inside = (t > 0) & (t <= n) & (p > 0) & (p < 1)
    out[inside] = val[inside]
    out[t <= 0] = 0.0
    out[(p >= 1) & (t <= n)] = 0.0
    return out


def _diff_log_tail(x_dev, mu_a, mu_b, n_a, n_b):
    """
    ``log`` of Bentkus' Theorem 1.1 bound on
    ``P(mean_a_hat - mean_b_hat - (mu_a - mu_b) >= x_dev)`` for independent samples on
    [0, 1] (vectorised over ``mu_a``, ``mu_b``). The centred summands ``(X_a - mu_a)/n_a``
    and ``(mu_b - X_b)/n_b`` are bounded above by ``bb = max((1 - mu_a)/n_a, mu_b/n_b)``
    with average variance ``var = (mu_a(1-mu_a)/n_a + mu_b(1-mu_b)/n_b) / N``; the tail is
    at most ``(e^2/2) P°(S_N >= x_dev)`` where ``S_N`` sums ``N`` copies of the two-point
    variable ``eps(var, bb)``, i.e. ``P(Bin(N, q) >= (x_dev bb + N var) / (bb^2 + var))``
    with ``q = var / (bb^2 + var)``.
    """
    mu_a = np.asarray(mu_a, dtype=float)
    mu_b = np.asarray(mu_b, dtype=float)
    N = n_a + n_b
    bb = np.maximum((1.0 - mu_a) / n_a, mu_b / n_b)
    var = (mu_a * (1.0 - mu_a) / n_a + mu_b * (1.0 - mu_b) / n_b) / N
    with np.errstate(divide='ignore', invalid='ignore'):
        q = var / (bb ** 2 + var)
        t = (x_dev * bb + N * var) / (bb ** 2 + var)
    out = np.log(_E2_OVER_2) + _binom_logsf_interp_vec(t, N, q)
    degenerate = (bb <= 0) | (var <= 0)
    out = np.where(degenerate, 0.0 if x_dev <= 0 else -np.inf, out)
    # Hoeffding's inequality for the same weighted sum (ranges 1/n_a and 1/n_b) bounds the
    # same tail probability, so the pointwise minimum is still a valid tail bound. (A
    # split-union of the two one-sample KL tails was also tried; it never improved on the
    # two bounds above, because a bound built on the single statistic A - B cannot reproduce
    # the one-sample rectangle's rejection of "both groups unanimous".)
    if x_dev > 0:
        out = np.minimum(out, -2.0 * x_dev ** 2 / (1.0 / n_a + 1.0 / n_b))
    return out


def _diff_lower(mean_a, mean_b, n_a, n_b, delta, nuisance_grid=129, nuisance_delta_frac=0.2,
                x_a=None, x_b=None):
    """
    Lower confidence bound on ``mu_a - mu_b`` at one-sided level ``delta``.

    The comparison distribution depends on the unknown group means. Taking the supremum
    over all feasible means wastes the information in the observed rates (a 5% rate has a
    twentieth of the worst-case variance), so the nuisance parameters are first localised
    with one-sample Bentkus bounds at level ``delta * nuisance_delta_frac`` (split over the
    four tails) and the tail bound is run at the remaining ``delta * (1 - frac)``; the
    failure probability is at most the sum (Berger & Boos 1994).
    """
    d_hat = mean_a - mean_b
    d1 = delta * nuisance_delta_frac
    d2 = delta - d1
    log_delta = np.log(d2)
    box_a, box_b = (0.0, 1.0), (0.0, 1.0)
    if d1 > 0 and x_a is not None and x_b is not None:
        box_a = (_bentkus_lower01(x_a, n_a, d1 / 4, None, 0.25),
                 1.0 - _bentkus_lower01(1.0 - x_a, n_a, d1 / 4, None, 0.25))
        box_b = (_bentkus_lower01(x_b, n_b, d1 / 4, None, 0.25),
                 1.0 - _bentkus_lower01(1.0 - x_b, n_b, d1 / 4, None, 0.25))

    def worst_log_tail(d0):
        x_dev = d_hat - d0
        lo = max(0.0, d0, box_a[0], box_b[0] + d0)
        hi = min(1.0, 1.0 + d0, box_a[1], box_b[1] + d0)
        if hi < lo:
            # no nuisance value inside the confidence boxes is consistent with d0
            return -np.inf
        mus = np.linspace(lo, hi, nuisance_grid)
        vals = _diff_log_tail(x_dev, mus, mus - d0, n_a, n_b)
        j = int(np.argmax(vals))
        best = float(vals[j])
        # refine the supremum over the nuisance mean around the best grid point
        left, right = mus[max(j - 1, 0)], mus[min(j + 1, nuisance_grid - 1)]
        if right > left and np.isfinite(best):
            fine = np.linspace(left, right, 33)
            best = max(best, float(np.max(_diff_log_tail(x_dev, fine, fine - d0, n_a, n_b))))
        return best

    def g(d0):
        return worst_log_tail(d0) - log_delta
    if g(-1.0) >= 0:
        return -1.0
    return min(_first_crossing(g, -1.0, d_hat, grid=24), d_hat)


def bentkus_diff_bounds(samples_a, samples_b, delta, n_a=None, n_b=None, predict=False,
                        a=0.0, b=1.0, nuisance_delta_frac=0.2):
    """
    Bounds on ``mean(samples_a) - mean(samples_b)`` for two independent samples supported
    on ``[a, b]``, each endpoint one-sided at level ``delta``.

    Instead of intersecting two one-sample intervals (whose widths add), Bentkus'
    Theorem 1.1 is applied once to the pooled sequence of centred, scaled summands, so the
    width scales with ``sqrt(var_a / n_a + var_b / n_b)`` like a two-sample z-test while
    remaining a non-asymptotic, distribution-free guarantee. The unknown group means enter
    the comparison distribution; the bound takes the supremum over them (the nuisance
    parameters) for every hypothesised difference, after localising them with one-sample
    Bentkus confidence boxes that use ``nuisance_delta_frac`` of the budget (Berger-Boos).
    """
    xa = _unit_interval(samples_a, a, b)
    xb = _unit_interval(samples_b, a, b)
    na = _effective_n(xa, n_a)
    nb = _effective_n(xb, n_b)
    ma, mb = float(xa.mean()), float(xb.mean())
    lower = _diff_lower(ma, mb, na, nb, delta, x_a=xa, x_b=xb,
                        nuisance_delta_frac=nuisance_delta_frac)
    upper = -_diff_lower(mb, ma, nb, na, delta, x_a=xb, x_b=xa,
                         nuisance_delta_frac=nuisance_delta_frac)
    d_hat = ma - mb
    lower, upper = min(lower, d_hat), max(upper, d_hat)
    if predict:
        lower = d_hat - 2.0 * (d_hat - lower)
        upper = d_hat + 2.0 * (upper - d_hat)
    scale = b - a
    return RandomVariable(scale * d_hat, lower=scale * max(lower, -1.0),
                          upper=scale * min(upper, 1.0))


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------

BOUNDS = {
    'ttest': ttest_bounds,
    'hoeffdings': hoeffdings_bounds,
    'clopper_pearson': clopper_pearson_bounds,
    'bentkus': bentkus_bounds,
    'empirical_bentkus': lambda *args, **kw: bentkus_bounds(*args, variance='empirical', **kw),
    'chernoff_kl': chernoff_kl_bounds,
    'empirical_bernstein': empirical_bernstein_bounds,
    'anderson': anderson_bounds,
    'learned_miller_thomas': learned_miller_thomas_bounds,
    'betting': betting_bounds,
    'betting_mixture': betting_mixture_bounds,
}

#: bounds whose coverage guarantee holds for every distribution on the stated range
DISTRIBUTION_FREE_BOUNDS = ('hoeffdings', 'bentkus', 'empirical_bentkus', 'chernoff_kl',
                            'empirical_bernstein', 'anderson', 'betting', 'betting_mixture')


def get_bound(method):
    """Look up a one-sample bound by name (see :data:`BOUNDS`)."""
    if callable(method):
        return method
    try:
        return BOUNDS[method]
    except KeyError:
        raise ValueError(f"unknown bound '{method}'; choose one of {sorted(BOUNDS)}") from None


# --------------------------------------------------------------------------------------
# Convex-order binomial bounds (new)
# --------------------------------------------------------------------------------------
#
# Hoeffding (1963, Thm 1) and Bentkus (2004, Thm 1.2) both rest on the same two steps:
# (1) for every convex f, replacing each [0, 1]-valued summand by a Bernoulli variable with
# the same mean cannot decrease E f (convex ordering; Hoeffding 1956), and (2) Markov's
# inequality applied to a convex majorant f of the indicator 1[s >= x]. Chernoff takes
# f = exp(lambda (s - x)); Bentkus takes hinge functions and then bounds the result in
# closed form at the price of the constant e. Every convex majorant of the indicator lies
# above some hinge (s - t)_+ / (x - t) with t < x, so the *tightest* bound the argument can
# give is
#
#     P(stat >= x) <= inf_{t < x} E[(T - t)_+] / (x - t),
#
# where T is the statistic computed on independent Bernoulli variables with the same means
# (a binomial proportion for a rate, the difference of two independent binomial
# proportions for a rate difference). The expectation is a finite sum, so the bound is
# evaluated exactly; any grid of t values is valid, since each t alone gives a bound.


def _hinge_tail_log(values, probs, x, t_grid):
    """``log min_t sum_k probs_k (values_k - t)_+ / (x - t)`` over ``t_grid`` (all ``< x``)."""
    pos = np.maximum(values[:, None] - t_grid[None, :], 0.0)              # (K, M)
    num = probs @ pos                                                     # (M,)
    with np.errstate(divide='ignore'):
        ratios = np.log(num) - np.log(x - t_grid)
    return float(np.min(ratios))


def _t_grid(x, m, size=40):
    """Hinge knots between the hypothesised mean ``m`` and the observed value ``x``."""
    gamma = np.geomspace(0.02, 4.0, size)
    return x - gamma * max(x - m, 1e-9)


def _convex_order_lower01(mean, n, delta):
    if mean <= 0:
        return 0.0
    ks = np.arange(n + 1) / n
    log_delta = np.log(delta)

    def g(m):
        probs = _binom.pmf(np.arange(n + 1), n, m)
        return _hinge_tail_log(ks, probs, mean, _t_grid(mean, m)) - log_delta
    return min(_first_crossing(g, 1e-12, mean), mean)


def convex_order_bounds(samples, delta, n=None, predict=False, a=0.0, b=1.0):
    """
    Optimal convex-order binomial bound for the mean of a random variable on ``[a, b]``.

    The lower bound is the smallest ``m`` with
    ``inf_t E[(Bin(n, m)/n - t)_+] / (mean - t) >= delta``. This is exactly the quantity
    Bentkus' Theorem 1.2 bounds by ``e * P°(Bin(n, m) >= n * mean)``, so it is never looser
    than :func:`bentkus_bounds` and never looser than the Chernoff (KL) bound, and it holds
    for every distribution on the interval. For non-integer ``n * mean`` (the prediction
    path) no interpolation is needed.
    """
    x = _unit_interval(samples, a, b)
    m = _effective_n(x, n)
    mean = float(x.mean())
    lower = _convex_order_lower01(mean, m, delta)
    upper = 1.0 - _convex_order_lower01(1.0 - mean, m, delta)
    return _pack(mean, lower, upper, a, b, predict)


def _diff_hinge_log(x_obs, mu_a, mu_b, n_a, n_b, t_grid):
    """
    ``log inf_t E[(A - B - t)_+] / (x_obs - t)`` for independent ``A = Bin(n_a, mu_a)/n_a``
    and ``B = Bin(n_b, mu_b)/n_b``, vectorised over arrays ``mu_a``, ``mu_b`` (returns one
    value per nuisance point). With ``h(u) = E[(u - B)_+] = u F_B(u) - E[B; B <= u]``
    (prefix sums over the support of ``B``), ``E[(A - B - t)_+] = sum_k p_A(k) h(k/n_a - t)``.
    """
    mu_a = np.atleast_1d(np.asarray(mu_a, dtype=float))
    mu_b = np.atleast_1d(np.asarray(mu_b, dtype=float))
    ka = np.arange(n_a + 1) / n_a
    kb = np.arange(n_b + 1) / n_b
    pa = _binom.pmf(np.arange(n_a + 1)[None, :], n_a, mu_a[:, None])         # (G, n_a+1)
    pb = _binom.pmf(np.arange(n_b + 1)[None, :], n_b, mu_b[:, None])         # (G, n_b+1)
    zeros = np.zeros((pb.shape[0], 1))
    cdf = np.concatenate((zeros, np.cumsum(pb, axis=1)), axis=1)              # (G, n_b+2)
    partial = np.concatenate((zeros, np.cumsum(pb * kb[None, :], axis=1)), axis=1)
    u = ka[:, None] - t_grid[None, :]                                        # (n_a+1, M)
    idx = np.searchsorted(kb, u, side='right')                               # count of kb <= u
    h = u[None] * cdf[:, idx] - partial[:, idx]                              # (G, n_a+1, M)
    num = np.einsum('gk,gkm->gm', pa, h)                                     # (G, M)
    with np.errstate(divide='ignore'):
        ratios = np.log(np.maximum(num, 0.0)) - np.log(x_obs - t_grid)[None, :]
    return ratios.min(axis=1)


def _convex_order_diff_lower(mean_a, mean_b, n_a, n_b, delta, nuisance_grid=33,
                             nuisance_delta_frac=0.2):
    """Lower bound on ``mu_a - mu_b``; nuisance means localised as in :func:`_diff_lower`."""
    d_hat = mean_a - mean_b
    d1 = delta * nuisance_delta_frac
    d2 = delta - d1
    log_delta = np.log(d2)
    box_a = (_convex_order_lower01(mean_a, n_a, d1 / 4),
             1.0 - _convex_order_lower01(1.0 - mean_a, n_a, d1 / 4))
    box_b = (_convex_order_lower01(mean_b, n_b, d1 / 4),
             1.0 - _convex_order_lower01(1.0 - mean_b, n_b, d1 / 4))

    def worst_log_tail(d0):
        if d0 >= d_hat:
            return 0.0
        lo = max(0.0, d0, box_a[0], box_b[0] + d0)
        hi = min(1.0, 1.0 + d0, box_a[1], box_b[1] + d0)
        if hi < lo:
            return -np.inf
        # hinge knots between the hypothesised mean d0 and the observed statistic d_hat
        t_grid = _t_grid(d_hat, d0)
        mus = np.linspace(lo, hi, nuisance_grid)
        vals = _diff_hinge_log(d_hat, mus, mus - d0, n_a, n_b, t_grid)
        j = int(np.argmax(vals))
        best = float(vals[j])
        left, right = mus[max(j - 1, 0)], mus[min(j + 1, nuisance_grid - 1)]
        if right > left and np.isfinite(best):
            fine = np.linspace(left, right, 9)
            best = max(best, float(np.max(_diff_hinge_log(d_hat, fine, fine - d0, n_a, n_b,
                                                          t_grid))))
        return best

    # the worst-case tail bound is non-decreasing in d0: bisect for the smallest d0 that
    # is not rejected
    lo, hi = -1.0, d_hat
    if worst_log_tail(lo) >= log_delta:
        return lo
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        if worst_log_tail(mid) >= log_delta:
            hi = mid
        else:
            lo = mid
    return hi


def convex_order_diff_bounds(samples_a, samples_b, delta, n_a=None, n_b=None, predict=False,
                             a=0.0, b=1.0, nuisance_delta_frac=0.2):
    """
    Optimal convex-order bound on ``mean(samples_a) - mean(samples_b)`` for two independent
    samples on ``[a, b]``; each endpoint one-sided at level ``delta``.

    The comparison variable is the difference of two independent binomial proportions with
    the hypothesised means, so (unlike :func:`bentkus_diff_bounds`, whose single two-point
    comparison averages the two groups) a unanimous group keeps its exact ``mu^n`` tail.
    The unknown means are localised with one-sample convex-order boxes (Berger-Boos, using
    ``nuisance_delta_frac`` of the budget) and the tail is maximised over the box.
    """
    xa = _unit_interval(samples_a, a, b)
    xb = _unit_interval(samples_b, a, b)
    na = _effective_n(xa, n_a)
    nb = _effective_n(xb, n_b)
    ma, mb = float(xa.mean()), float(xb.mean())
    lower = _convex_order_diff_lower(ma, mb, na, nb, delta,
                                     nuisance_delta_frac=nuisance_delta_frac)
    upper = -_convex_order_diff_lower(mb, ma, nb, na, delta,
                                      nuisance_delta_frac=nuisance_delta_frac)
    d_hat = ma - mb
    lower, upper = min(lower, d_hat), max(upper, d_hat)
    if predict:
        lower = d_hat - 2.0 * (d_hat - lower)
        upper = d_hat + 2.0 * (upper - d_hat)
    scale = b - a
    return RandomVariable(scale * d_hat, lower=scale * max(lower, -1.0),
                          upper=scale * min(upper, 1.0))


BOUNDS['convex_order'] = convex_order_bounds
DISTRIBUTION_FREE_BOUNDS = DISTRIBUTION_FREE_BOUNDS + ('convex_order',)
