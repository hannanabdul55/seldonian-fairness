"""
Validation tools for the confidence bounds in :mod:`seldonian.bounds`.

A bound's only job is to hold: ``P(mean < lower) <= delta`` for *every* distribution on the
range and *every* sample size. Monte Carlo alone cannot certify that (the miscoverage of a
bad bound on an adversarial distribution can be 40%, or 5.3% - the second needs ~10^5
replications to detect). This module therefore offers three complementary checks:

1. **Exact enumeration** (:func:`exact_two_point`, :func:`exact_three_point`). For a
   distribution with ``k`` atoms the sample is a multiset, so the coverage of a
   permutation-invariant bound is a finite sum over multinomial counts. No Monte Carlo
   error at all; the coverage of every method at every mixing probability is computed to
   machine precision. Two-point laws are the extreme points of the set of distributions on
   an interval with a given mean and range, and are the classical worst cases for mean
   bounds (Bentkus 2004), so this is also where invalid bounds break first.
2. **Adversarial search** (:func:`worst_case_two_point`): sweep the atom locations and
   mixing probability and report the *minimum* coverage. A valid bound never dips below
   ``1 - delta``; the t-test does, spectacularly.
3. **Monte Carlo** for continuous laws (:func:`mc_coverage`), reported with a
   Clopper-Pearson interval on the miscoverage so that "looks fine" has an error bar.

Tightness is measured as the expected one-sided slack ``E[upper - mean]`` (and
``E[mean - lower]``), i.e. how much of the confidence budget is wasted, rather than raw
width, so that asymmetric bounds are compared fairly.

Order-dependent bounds (``'betting'`` without permutation averaging) are not functions of
the multiset, so their enumerated coverage averages the bound over ``orderings`` uniformly
random permutations of each multiset (the conditional law of the order given the
multiset under i.i.d. sampling). The multinomial weights stay exact; only the inner
average carries Monte Carlo error.
"""

from itertools import combinations_with_replacement

import numpy as np
from scipy.special import gammaln, betaincinv
from scipy.stats import binom

from seldonian.bounds import get_bound, bentkus_diff_bounds
from seldonian.objectives import _rate_diff_bound

ORDER_DEPENDENT = ('betting',)


def endpoints(method, samples, delta, n=None, **kwargs):
    """One-sided ``(lower, upper)`` endpoints of ``method`` on ``samples`` (as given)."""
    rv = get_bound(method)(np.asarray(samples, dtype=float), delta, n=n, **kwargs)
    return float(rv.lower), float(rv.upper)


def is_order_dependent(method, **kwargs):
    return method in ORDER_DEPENDENT and kwargs.get('permutations') is None


def multiset_endpoints(method, samples, delta, orderings=1, seed=0, **kwargs):
    """
    Endpoints for every one of ``orderings`` random permutations of ``samples`` (shape
    ``(orderings, 2)``); permutation-invariant bounds are evaluated once and repeated.
    """
    x = np.asarray(samples, dtype=float)
    if not is_order_dependent(method, **kwargs):
        return np.tile(np.array(endpoints(method, x, delta, **kwargs)), (orderings, 1))
    rng = np.random.default_rng(seed)
    return np.array([endpoints(method, rng.permutation(x), delta, **kwargs)
                     for _ in range(orderings)])


def _multiset(values, counts):
    return np.repeat(np.asarray(values, dtype=float), counts)


# --------------------------------------------------------------------------------------
# exact enumeration
# --------------------------------------------------------------------------------------

def two_point_table(method, n, delta, v0, v1, orderings=32, **kwargs):
    """
    Endpoints for the sample with ``k`` copies of ``v1`` and ``n - k`` copies of ``v0``:
    arrays ``lower[k, r]``, ``upper[k, r]`` over ``r < orderings`` random orderings (a
    single column for permutation-invariant bounds).
    """
    reps = orderings if is_order_dependent(method, **kwargs) else 1
    lower = np.empty((n + 1, reps))
    upper = np.empty((n + 1, reps))
    for k in range(n + 1):
        ends = multiset_endpoints(method, _multiset([v0, v1], [n - k, k]), delta,
                                  orderings=reps, seed=1000 * n + k, **kwargs)
        lower[k], upper[k] = ends[:, 0], ends[:, 1]
    return lower, upper


def exact_two_point(method, n, delta, v0, v1, p, table=None, **kwargs):
    """
    Exact coverage and slack of ``method`` on the law ``P(X = v1) = p``, ``P(X = v0) = 1-p``.

    Returns a dict with ``cov_lower = P(lower <= mean)``, ``cov_upper = P(upper >= mean)``,
    ``slack_lower = E[mean - lower]`` and ``slack_upper = E[upper - mean]``. ``p`` may be an
    array. ``table`` (from :func:`two_point_table`) avoids recomputing the bounds.
    """
    if table is None:
        table = two_point_table(method, n, delta, v0, v1, **kwargs)
    lower, upper = table
    p = np.atleast_1d(np.asarray(p, dtype=float))
    mean = v0 + (v1 - v0) * p
    k = np.arange(n + 1)
    w = binom.pmf(k[None, :], n, p[:, None])                      # (len(p), n + 1)
    tol = 1e-12
    # average the indicator / slack over the orderings, then weight by the exact pmf
    cov_lower = (w * (lower[None] <= mean[:, None, None] + tol).mean(axis=2)).sum(axis=1)
    cov_upper = (w * (upper[None] >= mean[:, None, None] - tol).mean(axis=2)).sum(axis=1)
    slack_lower = (w * (mean[:, None] - lower.mean(axis=1)[None, :])).sum(axis=1)
    slack_upper = (w * (upper.mean(axis=1)[None, :] - mean[:, None])).sum(axis=1)
    return {'p': p, 'mean': mean, 'cov_lower': cov_lower, 'cov_upper': cov_upper,
            'slack_lower': slack_lower, 'slack_upper': slack_upper}


def worst_case_two_point(method, n, delta, value_pairs, p_grid, **kwargs):
    """
    Minimum one-sided coverage of ``method`` over two-point laws with atoms in
    ``value_pairs`` and mixing probabilities in ``p_grid``. Returns
    ``(min_cov_lower, min_cov_upper, argmin_lower, argmin_upper)`` where the argmins are
    ``(v0, v1, p)`` triples.
    """
    best = [np.inf, np.inf, None, None]
    for v0, v1 in value_pairs:
        res = exact_two_point(method, n, delta, v0, v1, p_grid, **kwargs)
        for side, key in ((0, 'cov_lower'), (1, 'cov_upper')):
            j = int(np.argmin(res[key]))
            if res[key][j] < best[side]:
                best[side] = float(res[key][j])
                best[side + 2] = (v0, v1, float(p_grid[j]))
    return tuple(best)


def _log_multinomial(counts, probs):
    counts = np.asarray(counts)
    probs = np.asarray(probs, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        lp = np.where(counts > 0, counts * np.log(probs), 0.0)
    return gammaln(counts.sum() + 1) - gammaln(counts + 1).sum() + lp.sum()


def exact_three_point(method, n, delta, values, probs, orderings=16, **kwargs):
    """
    Exact coverage / slack on the three-atom law ``P(X = values[j]) = probs[j]`` by
    enumerating the ``(n+1)(n+2)/2`` count vectors. ``probs`` may be a list of probability
    vectors (the bound evaluations are shared).
    """
    values = np.asarray(values, dtype=float)
    prob_list = [np.asarray(pr, dtype=float)
                 for pr in (probs if isinstance(probs, list) else [probs])]
    combos = list(combinations_with_replacement(range(3), n))
    counts = np.array([[c.count(j) for j in range(3)] for c in combos])
    reps = orderings if is_order_dependent(method, **kwargs) else 1
    lower = np.empty((len(combos), reps))
    upper = np.empty((len(combos), reps))
    for i, cnt in enumerate(counts):
        ends = multiset_endpoints(method, _multiset(values, cnt), delta, orderings=reps,
                                  seed=i, **kwargs)
        lower[i], upper[i] = ends[:, 0], ends[:, 1]
    out = []
    for pr in prob_list:
        mean = float(values @ pr)
        logw = np.array([_log_multinomial(cnt, pr) for cnt in counts])
        w = np.exp(logw)
        out.append({'probs': pr, 'mean': mean,
                    'cov_lower': float((w * (lower <= mean + 1e-12).mean(axis=1)).sum()),
                    'cov_upper': float((w * (upper >= mean - 1e-12).mean(axis=1)).sum()),
                    'slack_lower': float((w * (mean - lower.mean(axis=1))).sum()),
                    'slack_upper': float((w * (upper.mean(axis=1) - mean)).sum())})
    return out if isinstance(probs, list) else out[0]


# --------------------------------------------------------------------------------------
# Monte Carlo for continuous laws
# --------------------------------------------------------------------------------------

DISTRIBUTIONS = {
    # name: (sampler(rng, size), true mean)
    'uniform': (lambda rng, m: rng.random(m), 0.5),
    'beta(0.5,0.5)': (lambda rng, m: rng.beta(0.5, 0.5, m), 0.5),
    'beta(2,8)': (lambda rng, m: rng.beta(2, 8, m), 0.2),
    'beta(8,2)': (lambda rng, m: rng.beta(8, 2, m), 0.8),
    'beta(0.3,3)': (lambda rng, m: rng.beta(0.3, 3, m), 0.3 / 3.3),
    # rare large values: 95% zeros, otherwise uniform on [0.7, 1]
    'spike-slab': (lambda rng, m: np.where(rng.random(m) < 0.95, 0.0, rng.uniform(0.7, 1.0, m)),
                   0.05 * 0.85),
    # importance-weight-like: clipped log-normal, mostly tiny with a heavy right tail
    'clipped-lognormal': (lambda rng, m: np.minimum(1.0, np.exp(rng.normal(-3.0, 1.5, m))), None),
}


def true_mean(name, draws=4_000_000, seed=123):
    sampler, mean = DISTRIBUTIONS[name]
    if mean is None:
        # numerically integrate via a huge sample once (error ~1e-4)
        rng = np.random.default_rng(seed)
        mean = float(sampler(rng, draws).mean())
    return mean


def mc_coverage(method, dist, n, delta, reps=2000, seed=0, mean=None, **kwargs):
    """
    Monte Carlo miscoverage of ``method`` on ``dist`` (a key of :data:`DISTRIBUTIONS` or a
    ``(sampler, mean)`` pair). Returns miscoverage rates per side with a 99% Clopper-Pearson
    upper confidence limit, and the mean one-sided slack.
    """
    if isinstance(dist, str):
        sampler = DISTRIBUTIONS[dist][0]
        mean = true_mean(dist) if mean is None else mean
    else:
        sampler, mean = dist
    rng = np.random.default_rng(seed)
    miss_lower = miss_upper = 0
    slack_lower = slack_upper = 0.0
    for _ in range(reps):
        x = sampler(rng, n)
        lower, upper = endpoints(method, x, delta, **kwargs)
        miss_lower += lower > mean + 1e-12
        miss_upper += upper < mean - 1e-12
        slack_lower += mean - lower
        slack_upper += upper - mean
    return {'miss_lower': miss_lower / reps, 'miss_upper': miss_upper / reps,
            'miss_lower_ucl': cp_upper(miss_lower, reps, 0.01),
            'miss_upper_ucl': cp_upper(miss_upper, reps, 0.01),
            'slack_lower': slack_lower / reps, 'slack_upper': slack_upper / reps,
            'reps': reps}


def cp_upper(k, n, alpha):
    """Clopper-Pearson upper confidence limit for a binomial proportion."""
    return 1.0 if k >= n else float(betaincinv(k + 1, n - k, 1 - alpha))


# --------------------------------------------------------------------------------------
# two-sample difference bound
# --------------------------------------------------------------------------------------

def diff_upper_table(method, n_a, n_b, delta):
    """
    ``upper[k_a, k_b]``: the certified upper bound on ``|rate_a - rate_b|`` produced by
    :func:`seldonian.objectives._rate_diff_bound` (which splits ``delta`` as the
    constraint does) when the two Bernoulli samples have ``k_a`` and ``k_b`` successes.
    """
    upper = np.full((n_a + 1, n_b + 1), np.nan)
    for ka in range(n_a + 1):
        xa = _multiset([0.0, 1.0], [n_a - ka, ka])
        for kb in range(n_b + 1):
            xb = _multiset([0.0, 1.0], [n_b - kb, kb])
            rv = _rate_diff_bound(xa, xb, delta, None, n_a + n_b, method, False)
            upper[ka, kb] = rv.upper
    return upper


def exact_diff(method, n_a, n_b, delta, p_a, p_b, table=None):
    """Exact coverage of the constraint bound on ``|p_a - p_b|`` and its expected slack."""
    if table is None:
        table = diff_upper_table(method, n_a, n_b, delta)
    d = abs(p_a - p_b)
    wa = binom.pmf(np.arange(n_a + 1), n_a, p_a)
    wb = binom.pmf(np.arange(n_b + 1), n_b, p_b)
    w = wa[:, None] * wb[None, :]
    return {'cov': float((w * (table >= d - 1e-12)).sum()),
            'slack': float((w * (table - d)).sum())}


def diff_endpoints(samples_a, samples_b, delta):
    rv = bentkus_diff_bounds(samples_a, samples_b, delta)
    return float(rv.lower), float(rv.upper)


# --------------------------------------------------------------------------------------
# order dependence ("shuffle hacking")
# --------------------------------------------------------------------------------------

def shuffle_hack(method, dist, n, delta, tries=20, reps=300, seed=0, **kwargs):
    """
    For an order-dependent bound: probability that *some* of ``tries`` random orderings of
    the same sample yields a lower bound above the true mean (an adversary who re-shuffles
    until the safety test passes), versus the honest single-ordering miscoverage.
    """
    sampler, mean = DISTRIBUTIONS[dist][0], true_mean(dist)
    rng = np.random.default_rng(seed)
    honest = hacked = 0
    spread = 0.0
    fn = get_bound(method)
    for _ in range(reps):
        x = sampler(rng, n)
        lows = np.array([float(fn(rng.permutation(x), delta, **kwargs).lower)
                         for _ in range(tries)])
        honest += lows[0] > mean
        hacked += lows.max() > mean
        spread += lows.max() - lows.min()
    return {'miss_single': honest / reps, 'miss_best_of_tries': hacked / reps,
            'mean_spread': spread / reps, 'tries': tries, 'reps': reps}
