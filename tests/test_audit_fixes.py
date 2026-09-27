"""Regression tests for the 2026-09-27 audit against Thomas et al. (2019)
(reports/seldonian_math_audit.md)."""
import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from seldonian.bounds import hoeffdings_bounds
from seldonian.objectives import ghat_tpr_diff
from seldonian.seldonian import (BARRIER, LogisticRegressionSeldonianModel, _barrier,
                                 _resolve_bound)
from seldonian.synthetic import make_synthetic


class TestHoeffdingRange:
    def test_width_scales_with_range(self):
        x = np.random.default_rng(0).uniform(-3, 7, 200)
        rv = hoeffdings_bounds(x, 0.05, a=-3, b=7)
        assert np.isclose(rv.upper - rv.value, 10 * np.sqrt(np.log(1 / 0.05) / 400))

    def test_samples_outside_range_raise(self):
        with pytest.raises(ValueError, match="must lie in"):
            hoeffdings_bounds(np.array([0.5, 3.0]), 0.05)

    def test_rl_bound_requires_range(self):
        with pytest.raises(ValueError, match="bound_range"):
            _resolve_bound('hoeffdings', None)
        fn, kw = _resolve_bound('hoeffdings', (0, 20))
        assert kw == {'a': 0.0, 'b': 20.0}
        fn(np.array([1.0, 5.0, 19.0]), 0.05, **kw)       # accepted, no TypeError

    def test_coverage_on_wide_range(self):
        rng = np.random.default_rng(1)
        miss = sum(hoeffdings_bounds(rng.uniform(0, 10, 50), 0.1, a=0, b=10).upper < 5
                   for _ in range(2000))
        assert miss / 2000 <= 0.1


class TestStratifyDoesNotLookAtSafetySet:
    def test_split_is_label_stratified(self):
        X, y, A_idx = make_synthetic(400, 4, seed=0)
        ghats = [{"fn": ghat_tpr_diff(A_idx, threshold=0.2), "delta": 0.05}]
        m = LogisticRegressionSeldonianModel(X, y, g_hats=ghats, stratify=True, random_seed=3,
                                             verbose=False)
        _, X_s, _, y_s = train_test_split(X, y, test_size=0.5, random_state=3, stratify=y)
        assert np.array_equal(m.X_s, X_s) and np.array_equal(m.y_s, y_s)


class TestBarrier:
    def test_every_failing_theta_ranks_below_every_passing_one(self):
        assert _barrier(0.0) == 0.0 and _barrier(-0.3) == 0.0
        assert _barrier(1e-9) > BARRIER - 1        # a tiny violation still pays the barrier
        assert _barrier(0.5) > _barrier(0.1)       # and a larger one pays more
