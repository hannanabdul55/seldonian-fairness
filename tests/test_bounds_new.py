"""Tests for the non-asymptotic mean bounds added to seldonian.bounds and their
validation helpers. Coverage checks use exact enumeration (no Monte Carlo)."""

import numpy as np
import pytest
import torch
from scipy.stats import beta as beta_dist, binom

from seldonian.bounds import (
    BOUNDS,
    DISTRIBUTION_FREE_BOUNDS,
    RandomVariable,
    anderson_bounds,
    bentkus_bounds,
    bentkus_diff_bounds,
    betting_bounds,
    betting_mixture_bounds,
    chernoff_kl_bounds,
    clopper_pearson_bounds,
    convex_order_bounds,
    convex_order_diff_bounds,
    empirical_bernstein_bounds,
    get_bound,
    learned_miller_thomas_bounds,
    ttest_bounds,
)
from seldonian import bounds_eval
from seldonian.objectives import ghat_tpr_diff, _rate_diff_bound

NEW_METHODS = [m for m in BOUNDS if m not in ('ttest', 'hoeffdings')]
GENERAL_METHODS = [m for m in NEW_METHODS if m != 'clopper_pearson']


@pytest.fixture(scope='module')
def rng():
    return np.random.default_rng(0)


class TestContract:
    @pytest.mark.parametrize('method', NEW_METHODS)
    def test_binary_sample_returns_ordered_interval_containing_mean(self, method, rng):
        x = rng.binomial(1, 0.3, 80).astype(float)
        rv = get_bound(method)(x, 0.05)
        assert isinstance(rv, RandomVariable)
        assert rv.lower <= rv.value <= rv.upper
        assert 0.0 <= rv.lower and rv.upper <= 1.0
        assert np.isclose(rv.value, x.mean())

    @pytest.mark.parametrize('method', GENERAL_METHODS)
    def test_continuous_sample(self, method, rng):
        x = rng.beta(2, 5, 60)
        rv = get_bound(method)(x, 0.05)
        assert rv.lower < x.mean() < rv.upper

    @pytest.mark.parametrize('method', GENERAL_METHODS)
    def test_smaller_delta_widens(self, method, rng):
        x = rng.beta(2, 5, 60)
        wide = get_bound(method)(x, 0.01)
        narrow = get_bound(method)(x, 0.2)
        assert wide.lower <= narrow.lower + 1e-12
        assert wide.upper >= narrow.upper - 1e-12

    @pytest.mark.parametrize('method', GENERAL_METHODS)
    def test_predict_doubles_deviation_within_support(self, method, rng):
        x = rng.beta(2, 5, 60)
        plain = get_bound(method)(x, 0.05)
        pred = get_bound(method)(x, 0.05, predict=True)
        assert np.isclose(pred.upper - pred.value,
                          min(2 * (plain.upper - plain.value), 1 - plain.value))
        assert np.isclose(pred.value - pred.lower,
                          min(2 * (plain.value - plain.lower), plain.value))

    @pytest.mark.parametrize('method', GENERAL_METHODS)
    def test_range_rescaling(self, method, rng):
        x = rng.beta(2, 5, 60)
        unit = get_bound(method)(x, 0.05)
        scaled = get_bound(method)(3 * x - 1, 0.05, a=-1.0, b=2.0)
        assert np.isclose(scaled.lower, 3 * unit.lower - 1)
        assert np.isclose(scaled.upper, 3 * unit.upper - 1)

    @pytest.mark.parametrize('method', GENERAL_METHODS)
    def test_rejects_out_of_range(self, method):
        with pytest.raises(ValueError):
            get_bound(method)(np.array([0.2, 1.5]), 0.05)

    @pytest.mark.parametrize('method', GENERAL_METHODS)
    def test_accepts_tensor(self, method, rng):
        x = torch.tensor(rng.beta(2, 5, 40))
        rv = get_bound(method)(x, 0.05)
        assert float(rv.lower) < float(x.mean()) < float(rv.upper)

    @pytest.mark.parametrize('method', GENERAL_METHODS)
    def test_larger_n_shrinks_slack(self, method, rng):
        x = rng.beta(2, 5, 50)
        small = get_bound(method)(x, 0.05, n=50)
        large = get_bound(method)(x, 0.05, n=5000)
        assert large.upper - large.lower < small.upper - small.lower

    def test_unknown_method(self):
        with pytest.raises(ValueError):
            get_bound('nope')

    def test_clopper_pearson_rejects_non_binary(self):
        with pytest.raises(ValueError):
            clopper_pearson_bounds(np.array([0.2, 0.5]), 0.05)


class TestClosedForms:
    def test_clopper_pearson_matches_beta_quantiles(self):
        x = np.array([1.0] * 7 + [0.0] * 13)
        rv = clopper_pearson_bounds(x, 0.05)
        assert np.isclose(rv.lower, beta_dist.ppf(0.05, 7, 14))
        assert np.isclose(rv.upper, beta_dist.ppf(0.95, 8, 13))

    def test_clopper_pearson_unanimous_sample_has_width(self):
        # the t-test collapses to a point here; the exact bound does not
        x = np.zeros(30)
        assert ttest_bounds(x, 0.05).upper == 0.0
        assert clopper_pearson_bounds(x, 0.05).upper == pytest.approx(1 - 0.05 ** (1 / 30))

    def test_bentkus_is_clopper_pearson_at_delta_over_e(self):
        x = np.array([1.0] * 12 + [0.0] * 28)
        b = bentkus_bounds(x, 0.05)
        cp = clopper_pearson_bounds(x, 0.05 / np.e)
        assert np.isclose(b.lower, cp.lower, atol=1e-6)
        assert np.isclose(b.upper, cp.upper, atol=1e-6)

    def test_bentkus_tail_identity_at_solution(self):
        x = np.array([1.0] * 12 + [0.0] * 28)
        lower = bentkus_bounds(x, 0.05).lower
        assert np.isclose(np.e * binom.sf(11, 40, lower), 0.05, rtol=1e-6)

    def test_chernoff_kl_solution(self):
        x = np.array([1.0] * 12 + [0.0] * 28)
        m = chernoff_kl_bounds(x, 0.05).lower
        p = 0.3
        kl = p * np.log(p / m) + (1 - p) * np.log((1 - p) / (1 - m))
        assert np.isclose(40 * kl, np.log(1 / 0.05), rtol=1e-6)

    def test_empirical_bernstein_formula(self):
        x = np.random.default_rng(1).beta(2, 5, 100)
        rv = empirical_bernstein_bounds(x, 0.05)
        ln = np.log(2 / 0.05)
        dev = np.sqrt(2 * x.var(ddof=1) * ln / 100) + 7 * ln / (3 * 99)
        assert np.isclose(rv.upper - rv.value, dev)

    def test_anderson_reduces_to_hoeffding_on_binary_data(self):
        # with atoms only at 0 and 1 the DKW envelope shifts the mean by exactly eps
        x = np.array([1.0] * 12 + [0.0] * 28)
        rv = anderson_bounds(x, 0.05)
        eps = np.sqrt(np.log(1 / 0.05) / 80)
        assert np.isclose(rv.value - rv.lower, eps)

    def test_learned_miller_thomas_equals_clopper_pearson_on_binary(self):
        x = np.array([1.0] * 5 + [0.0] * 15)
        lmt = learned_miller_thomas_bounds(x, 0.05, draws=200000)
        cp = clopper_pearson_bounds(x, 0.05)
        assert np.isclose(lmt.upper, cp.upper, atol=3e-3)
        assert np.isclose(lmt.lower, cp.lower, atol=3e-3)

    def test_mixture_is_permutation_invariant(self, rng):
        x = rng.beta(2, 5, 50)
        a = betting_mixture_bounds(x, 0.05)
        b = betting_mixture_bounds(rng.permutation(x), 0.05)
        assert a == b

    def test_sequential_betting_depends_on_order_but_averaging_reduces_it(self, rng):
        x = rng.beta(2, 5, 50)
        a = betting_bounds(x, 0.05)
        b = betting_bounds(x[::-1].copy(), 0.05)
        assert a != b
        pa = betting_bounds(x, 0.05, permutations=64, seed=1)
        pb = betting_bounds(x[::-1].copy(), 0.05, permutations=64, seed=2)
        assert abs(pa.lower - pb.lower) < abs(a.lower - b.lower)

    def test_mixture_prediction_is_exact_rescaling(self, rng):
        # the bound only sees the empirical distribution, so n=len(x)*3 with the sample
        # tripled equals n=len(x)*3 with the original sample
        x = rng.beta(2, 5, 30)
        tiled = betting_mixture_bounds(np.tile(x, 3), 0.05)
        scaled = betting_mixture_bounds(x, 0.05, n=90)
        assert np.isclose(tiled.lower, scaled.lower, atol=1e-9)
        assert np.isclose(tiled.upper, scaled.upper, atol=1e-9)


class TestExactCoverage:
    """Exact (enumerated) coverage on two-point laws must be >= 1 - delta for every valid
    bound, and the t-test must fail on skewed Bernoulli data."""

    P_GRID = np.linspace(0.005, 0.995, 199)
    PAIRS = [(0.0, 1.0), (0.0, 0.5), (0.5, 1.0), (0.0, 0.1), (0.9, 1.0)]

    @pytest.mark.parametrize('method', list(DISTRIBUTION_FREE_BOUNDS))
    @pytest.mark.parametrize('n', [5, 20])
    def test_distribution_free_bounds_cover(self, method, n):
        lo, hi, _, _ = bounds_eval.worst_case_two_point(method, n, 0.1, self.PAIRS,
                                                        self.P_GRID)
        # the sequential betting bound is averaged over random orderings (Monte Carlo)
        tol = 0.02 if bounds_eval.is_order_dependent(method) else 1e-9
        assert lo >= 0.9 - tol
        assert hi >= 0.9 - tol

    def test_clopper_pearson_covers_bernoulli(self):
        lo, hi, _, _ = bounds_eval.worst_case_two_point('clopper_pearson', 25, 0.1,
                                                        [(0.0, 1.0)], self.P_GRID)
        assert lo >= 0.9 - 1e-9 and hi >= 0.9 - 1e-9

    def test_ttest_fails_on_rare_events(self):
        res = bounds_eval.exact_two_point('ttest', 20, 0.05, 0.0, 1.0, np.array([0.02]))
        # with 20 samples and p = 0.02 the sample is unanimous 67% of the time -> point CI
        assert res['cov_upper'][0] < 0.5

    def test_hoeffding_is_much_looser_than_bentkus(self):
        h = bounds_eval.exact_two_point('hoeffdings', 50, 0.05, 0.0, 1.0, np.array([0.1]))
        b = bounds_eval.exact_two_point('bentkus', 50, 0.05, 0.0, 1.0, np.array([0.1]))
        assert b['slack_upper'][0] < 0.8 * h['slack_upper'][0]

    def test_three_point_enumeration(self):
        out = bounds_eval.exact_three_point('bentkus', 6, 0.1, [0.0, 0.5, 1.0],
                                            [[0.2, 0.5, 0.3], [0.6, 0.1, 0.3]])
        for r in out:
            assert r['cov_lower'] >= 0.9 and r['cov_upper'] >= 0.9

    def test_mc_coverage_reports_interval(self):
        r = bounds_eval.mc_coverage('bentkus', 'beta(2,8)', 20, 0.1, reps=50, seed=0)
        assert 0 <= r['miss_upper'] <= r['miss_upper_ucl'] <= 1


class TestDifferenceBound:
    def test_contains_point_estimate_and_is_tighter_than_rectangle(self, rng):
        xa = rng.binomial(1, 0.4, 150).astype(float)
        xb = rng.binomial(1, 0.6, 150).astype(float)
        rv = bentkus_diff_bounds(xa, xb, 0.025)
        assert rv.lower <= rv.value <= rv.upper
        assert np.isclose(rv.value, xa.mean() - xb.mean())
        rect = bentkus_bounds(xa, 0.025) - bentkus_bounds(xb, 0.025)
        assert (rv.upper - rv.lower) < (rect.upper - rect.lower)

    def test_symmetry(self, rng):
        xa = rng.binomial(1, 0.4, 60).astype(float)
        xb = rng.binomial(1, 0.7, 90).astype(float)
        ab = bentkus_diff_bounds(xa, xb, 0.05)
        ba = bentkus_diff_bounds(xb, xa, 0.05)
        assert np.isclose(ab.lower, -ba.upper) and np.isclose(ab.upper, -ba.lower)

    def test_exact_coverage_of_constraint_bound(self):
        table = bounds_eval.diff_upper_table('bentkus_diff', 12, 9, 0.1)
        for pa, pb in [(0.1, 0.5), (0.5, 0.5), (0.9, 0.2), (0.02, 0.98)]:
            r = bounds_eval.exact_diff('bentkus_diff', 12, 9, 0.1, pa, pb, table=table)
            assert r['cov'] >= 0.9 - 1e-9

    def test_objective_accepts_new_methods(self):
        rng = np.random.default_rng(3)
        X = rng.random((400, 3))
        X[:, 1] = rng.binomial(1, 0.5, 400)
        y = rng.binomial(1, 0.6, 400)
        y_pred = rng.binomial(1, 0.6, 400)
        vals = {}
        for method in ['ttest', 'clopper_pearson', 'bentkus', 'bentkus_diff', 'betting_mixture']:
            g = ghat_tpr_diff(1, method=method, threshold=0.2)
            vals[method] = g(X, y, y_pred, delta=0.05)
            assert np.isfinite(vals[method])
        # the joint bound beats the one-sample rectangle built from the same inequality
        assert vals['bentkus_diff'] < vals['bentkus']

    def test_rate_diff_predict_uses_scaled_subgroup_sizes(self):
        a = np.array([1.0] * 30 + [0.0] * 20)
        b = np.array([1.0] * 20 + [0.0] * 30)
        small = _rate_diff_bound(a, b, 0.05, 100, 100, 'bentkus_diff', True)
        large = _rate_diff_bound(a, b, 0.05, 10000, 100, 'bentkus_diff', True)
        assert large.upper < small.upper


class TestConvexOrder:
    def test_one_sample_never_looser_than_bentkus_or_chernoff(self):
        for k, n in [(0, 20), (1, 20), (6, 20), (12, 40), (95, 100)]:
            x = np.array([1.0] * k + [0.0] * (n - k))
            co = convex_order_bounds(x, 0.05)
            for other in (bentkus_bounds(x, 0.05), chernoff_kl_bounds(x, 0.05)):
                assert co.lower >= other.lower - 1e-9
                assert co.upper <= other.upper + 1e-9

    def test_one_sample_exact_coverage(self):
        lo, hi, _, _ = bounds_eval.worst_case_two_point(
            'convex_order', 15, 0.1, [(0.0, 1.0), (0.0, 0.5), (0.2, 1.0)],
            np.linspace(0.01, 0.99, 99))
        assert lo >= 0.9 - 1e-9 and hi >= 0.9 - 1e-9

    def test_diff_tighter_than_bentkus_diff_and_symmetric(self, rng):
        xa = rng.binomial(1, 0.4, 60).astype(float)
        xb = rng.binomial(1, 0.65, 90).astype(float)
        co = convex_order_diff_bounds(xa, xb, 0.05)
        bd = bentkus_diff_bounds(xa, xb, 0.05)
        assert co.lower <= co.value <= co.upper
        assert (co.upper - co.lower) < (bd.upper - bd.lower)
        back = convex_order_diff_bounds(xb, xa, 0.05)
        assert np.isclose(co.lower, -back.upper, atol=1e-6)
        assert np.isclose(co.upper, -back.lower, atol=1e-6)

    def test_diff_exact_coverage(self):
        table = bounds_eval.diff_upper_table('convex_order_diff', 10, 8, 0.1)
        for pa, pb in [(0.1, 0.5), (0.5, 0.5), (0.9, 0.2), (0.02, 0.98), (1.0, 0.7)]:
            r = bounds_eval.exact_diff('convex_order_diff', 10, 8, 0.1, pa, pb, table=table)
            assert r['cov'] >= 0.9 - 1e-9

    def test_objective_accepts_convex_order_diff(self):
        rng = np.random.default_rng(5)
        X = rng.random((300, 3))
        X[:, 1] = rng.binomial(1, 0.5, 300)
        y = rng.binomial(1, 0.6, 300)
        y_pred = rng.binomial(1, 0.6, 300)
        g_new = ghat_tpr_diff(1, method='convex_order_diff', threshold=0.2)(X, y, y_pred, delta=0.05)
        g_old = ghat_tpr_diff(1, method='bentkus_diff', threshold=0.2)(X, y, y_pred, delta=0.05)
        assert np.isfinite(g_new) and g_new <= g_old + 1e-9
