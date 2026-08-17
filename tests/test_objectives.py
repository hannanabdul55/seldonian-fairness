"""Black-box tests for seldonian.objectives: subgroup rate helpers and g-hat
constraint constructors."""

import numpy as np
import pytest

from seldonian.objectives import (
    Constraint,
    ghat_recall_rate,
    ghat_tpr_diff,
    ghat_tpr_diff_t,
    recall_rate,
    tpr_rate,
)


def fair_data(n_per_group=200):
    """Two groups; classifier predicts every positive correctly in both."""
    X = np.vstack([np.ones((n_per_group, 2)), np.zeros((n_per_group, 2))])
    X[:, 0] = np.arange(2 * n_per_group)  # non-sensitive feature
    y = np.tile([0, 1], n_per_group)
    return X, y, y.copy()  # y_pred == y_true


def unfair_data(n_per_group=200):
    """Group A=1 gets all positives wrong; group A=0 gets all right."""
    X = np.vstack([np.ones((n_per_group, 2)), np.zeros((n_per_group, 2))])
    y = np.tile([0, 1], n_per_group)
    y_pred = y.copy()
    y_pred[(X[:, 1] == 1) & (y == 1)] = 0
    return X, y, y_pred


class TestTprRate:
    def test_overall_tpr_without_subgroup(self):
        y_true = np.array([1, 1, 1, 0])
        y_pred = np.array([1, 0, 1, 1])
        assert tpr_rate()(None, y_true, y_pred).mean() == pytest.approx(2 / 3)

    def test_subgroup_tpr(self):
        X = np.array([[1.0], [1.0], [0.0], [0.0]])
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([1, 0, 1, 1])
        assert tpr_rate(0, 1)(X, y_true, y_pred).mean() == 0.5
        assert tpr_rate(0, 0)(X, y_true, y_pred).mean() == 1.0

    def test_negatives_do_not_count(self):
        X = np.ones((4, 1))
        y_true = np.array([1, 0, 0, 0])
        y_pred = np.ones(4)
        assert tpr_rate(0, 1)(X, y_true, y_pred).mean() == 1.0

    def test_recall_equals_tpr(self):
        X, y_true, y_pred = unfair_data()
        a = tpr_rate(1, 1)(X, y_true, y_pred)
        b = recall_rate(1, 1)(X, y_true, y_pred)
        assert np.array_equal(a, b)


class TestGhatTprDiff:
    @pytest.mark.parametrize("ghat_factory", [ghat_tpr_diff, ghat_recall_rate])
    def test_fair_predictions_satisfy_constraint(self, ghat_factory):
        X, y_true, y_pred = fair_data()
        g = ghat_factory(1, threshold=0.2)(X, y_true, y_pred, delta=0.05, ub=False)
        assert g == pytest.approx(-0.2)

    @pytest.mark.parametrize("ghat_factory", [ghat_tpr_diff, ghat_recall_rate])
    def test_unfair_predictions_violate_constraint(self, ghat_factory):
        X, y_true, y_pred = unfair_data()
        g = ghat_factory(1, threshold=0.2)(X, y_true, y_pred, delta=0.05, ub=False)
        assert g == pytest.approx(0.8)  # |1 - 0| - 0.2

    def test_upper_bound_at_least_point_estimate(self):
        X, y_true, y_pred = unfair_data()
        ghat = ghat_tpr_diff(1, threshold=0.2)
        ub = ghat(X, y_true, y_pred, delta=0.05, ub=True)
        point = ghat(X, y_true, y_pred, delta=0.05, ub=False)
        assert ub >= point

    def test_threshold_shifts_result_linearly(self):
        X, y_true, y_pred = unfair_data()
        g_small = ghat_tpr_diff(1, threshold=0.1)(X, y_true, y_pred, delta=0.05, ub=False)
        g_large = ghat_tpr_diff(1, threshold=0.5)(X, y_true, y_pred, delta=0.05, ub=False)
        assert g_small - g_large == pytest.approx(0.4)

    def test_missing_subgroup_is_a_violation(self):
        X = np.ones((10, 2))
        y = np.ones(10)
        assert ghat_tpr_diff(1, threshold=0.2)(X, y, y, delta=0.05) == np.inf

    def test_hoeffdings_method(self):
        X, y_true, y_pred = fair_data()
        g = ghat_tpr_diff(1, method='hoeffdings', threshold=0.2)(
            X, y_true, y_pred, delta=0.05, ub=True)
        assert np.isfinite(g)

    def test_predict_mode_widens_bound(self):
        X, y_true, y_pred = unfair_data()
        ghat = ghat_tpr_diff(1, threshold=0.2)
        plain = ghat(X, y_true, y_pred, delta=0.05, ub=True)
        predicted = ghat(X, y_true, y_pred, delta=0.05, n=len(X), predict=True, ub=True)
        assert predicted >= plain

    def test_torch_variant_matches_numpy_on_arrays(self):
        X, y_true, y_pred = unfair_data()
        g_np = ghat_tpr_diff(1, threshold=0.2)(X, y_true, y_pred, delta=0.05, ub=False)
        g_t = ghat_tpr_diff_t(1, threshold=0.2)(X, y_true, y_pred, delta=0.05, ub=False)
        assert g_t == pytest.approx(g_np)


class TestConstraintABC:
    def test_call_is_abstract(self):
        class Dummy(Constraint):
            pass

        with pytest.raises(NotImplementedError):
            Dummy()()
