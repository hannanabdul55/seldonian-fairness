"""Regression tests for bugs found in the deep code review."""

import numpy as np
import pytest

from seldonian.bounds import RandomVariable
from seldonian.objectives import ghat_tpr_diff, ghat_tpr_diff_t
from seldonian.seldonian import (
    LogisticRegressionSeldonianGD,
    LogisticRegressionSeldonianModel,
    PDISSeldonianPolicyCMAES,
    SeldonianAlgorithmLogRegCMAES,
    SeldonianCEMPDISPolicy,
)
from seldonian.synthetic import make_synthetic


def episodes(n, reward=1.0, pi_b=0.5):
    return [[[0, 0, reward, pi_b]] for _ in range(n)]


class TestPdisEstimateListBug:
    def test_per_episode_estimates_survive_default_minimize(self):
        m = SeldonianCEMPDISPolicy(episodes(10), 2, 2, gamma=0.95)
        out = m.pdis_estimate(np.zeros((2, 2)), m.D_c, sum_red=False)  # minimize=True
        assert len(out) == len(m.D_c)
        assert all(np.isfinite(v) for v in out)


class TestSeeding:
    def test_scipy_model_theta_is_seeded(self):
        X = np.random.default_rng(0).random((50, 3))
        y = (X[:, 0] > 0.5).astype(int)
        a = LogisticRegressionSeldonianModel(X, y, g_hats=[], verbose=False,
                                             random_seed=5)
        b = LogisticRegressionSeldonianModel(X, y, g_hats=[], verbose=False,
                                             random_seed=5)
        assert np.array_equal(a.theta, b.theta)

    @pytest.mark.parametrize("cls,kwargs", [
        (PDISSeldonianPolicyCMAES, {"multiprocessing": False}),
        (SeldonianCEMPDISPolicy, {}),
    ])
    def test_rl_policies_are_seeded(self, cls, kwargs):
        D = episodes(20)
        a = cls(D, 2, 2, gamma=0.9, random_seed=4, **kwargs)
        b = cls(D, 2, 2, gamma=0.9, random_seed=4, **kwargs)
        assert np.array_equal(a.theta, b.theta)
        assert a.D_c == b.D_c


class TestMultiConstraintMax:
    def test_safety_test_returns_worst_violation(self):
        # group column 1: only group A=1 has a TPR deficit
        X = np.zeros((400, 2))
        X[:200, 1] = 1
        y = np.ones(400)
        loose = {"fn": ghat_tpr_diff_t(1, threshold=0.9), "delta": 0.05}
        tight = {"fn": ghat_tpr_diff_t(1, threshold=0.0), "delta": 0.05}
        m1 = LogisticRegressionSeldonianGD(X, y, g_hats=[loose, tight], epochs=1)
        m2 = LogisticRegressionSeldonianGD(X, y, g_hats=[tight, loose], epochs=1)
        # same data, same predictions (untrained same-seed models): the reported
        # violation must not depend on constraint order
        assert m1._safetyTest() == m2._safetyTest()


class TestHardBarrierSemantics:
    def test_cmaes_safety_test_reports_real_value(self):
        np.random.seed(0)
        X, y, A_idx = make_synthetic(800, 4)
        ghats = [{"fn": ghat_tpr_diff(A_idx, threshold=0.0), "delta": 0.05}]
        model = SeldonianAlgorithmLogRegCMAES(X, y, g_hats=ghats, hard_barrier=True,
                                              maxiter=50)
        val = model._safetyTest(predict=False)
        # the safety test must return the actual ghat value, never the barrier 1
        assert val != 1 or val == 0

    def test_cmaes_fit_returns_none_on_unsatisfiable_constraint(self):
        np.random.seed(0)
        X, y, A_idx = make_synthetic(600, 4)
        # negative threshold can never be satisfied: |diff| bound - (-1) > 0 always
        ghats = [{"fn": ghat_tpr_diff(A_idx, threshold=-1.0), "delta": 0.05}]
        model = SeldonianAlgorithmLogRegCMAES(X, y, g_hats=ghats, maxiter=100)
        assert model.fit() is None


class TestRandomVariableProtocol:
    def test_reflected_ops(self):
        rv = RandomVariable(2.0, lower=1.0, upper=3.0)
        assert (5 + rv).value == 7.0
        s = 5 - rv
        assert (s.value, s.lower, s.upper) == (3.0, 2.0, 4.0)
        assert (2 * rv).value == 4.0
        q = 6 / rv
        assert q.value == 3.0
        assert q.lower == 2.0
        assert q.upper == 6.0

    def test_equality(self):
        assert RandomVariable(1.0, 0.0, 2.0) == RandomVariable(1.0, 0.0, 2.0)
        assert RandomVariable(1.0, 0.0, 2.0) != RandomVariable(1.0, 0.0, 3.0)

    def test_degenerate_zero_divisor(self):
        q = RandomVariable(1.0, 1.0, 1.0) / RandomVariable(0.0, 0.0, 0.0)
        assert q.lower == -np.inf
        assert q.upper == np.inf


class TestMakeSyntheticGuard:
    def test_rejects_sensitive_attribute_at_column_zero(self):
        with pytest.raises(ValueError):
            make_synthetic(100, 4, A_idx=0)


class TestDeltaAllocation:
    def test_rate_diff_bound_splits_delta_across_tails(self):
        # per Seldonian Engine "equal" allocation: 2 base nodes x 2 tails -> delta/4
        # per tail, so the propagated bound must match one built from delta/4 intervals
        from seldonian.bounds import ttest_bounds
        from seldonian.objectives import _rate_diff_bound

        rng = np.random.default_rng(0)
        a = rng.random(50) > 0.4
        b = rng.random(60) > 0.6
        got = _rate_diff_bound(a, b, delta=0.05, n=None, total_size=110,
                               method='ttest', predict=False)
        expected = abs(ttest_bounds(b, 0.0125) - ttest_bounds(a, 0.0125))
        assert got == expected
        # and it must be strictly wider than the old full-delta version
        old = abs(ttest_bounds(b, 0.05) - ttest_bounds(a, 0.05))
        assert got.upper > old.upper
