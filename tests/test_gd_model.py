"""Black-box tests for the gradient-based Seldonian logistic regression."""

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from seldonian.objectives import ghat_tpr_diff_t, tpr_rate
from seldonian.seldonian import LogisticRegressionSeldonianGD
from seldonian.synthetic import make_synthetic


def separable_data(n=400, d=4, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    w = rng.standard_normal(d)
    y = (X @ w > 0).astype(int)
    return X, y


def constrained_setup(n=8000):
    np.random.seed(1)
    X, y, A_idx = make_synthetic(n, 5, tp_a=0.4, tp_b=0.8, seed=1)
    ghats = [{"fn": ghat_tpr_diff_t(A_idx, threshold=0.2), "delta": 0.05}]
    return X, y, A_idx, ghats


class TestLogisticRegressionSeldonianGD:
    def test_unconstrained_learns_separable_data(self):
        X, y = separable_data()
        model = LogisticRegressionSeldonianGD(X, y, g_hats=[], epochs=300, random_seed=0)
        result = model.fit()
        assert result is model  # no constraints -> safety trivially passes
        preds = model.predict(X)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (len(X),)
        assert set(np.unique(preds)) <= {0, 1}
        assert accuracy_score(y, preds) > 0.8

    def test_constrained_fit_satisfies_hard_safety_test(self):
        X, y, A_idx, ghats = constrained_setup()
        model = LogisticRegressionSeldonianGD(X, y, g_hats=ghats, random_seed=0)
        result = model.fit()
        assert result is model
        assert float(model._safetyTest()) == 0
        assert model.safetyTest() is True
        preds = model.predict(X)
        assert balanced_accuracy_score(y, preds) > 0.5
        # constrained model keeps the empirical TPR gap under the 0.2 threshold
        tpr_a = tpr_rate(A_idx, 1)(X, y, preds).mean()
        tpr_b = tpr_rate(A_idx, 0)(X, y, preds).mean()
        assert abs(tpr_a - tpr_b) < 0.2

    def test_deterministic_for_seed(self):
        X, y, _, ghats = constrained_setup(2000)
        preds = []
        for _ in range(2):
            model = LogisticRegressionSeldonianGD(X, y, g_hats=ghats, epochs=100,
                                                  random_seed=3)
            model.fit()
            preds.append(model.predict(X))
        assert np.array_equal(preds[0], preds[1])

    def test_explicit_safety_data(self):
        X, y = separable_data(300)
        X_s, y_s = separable_data(100, seed=8)
        model = LogisticRegressionSeldonianGD(X, y, g_hats=[], safety_data=(X_s, y_s),
                                              epochs=50)
        assert model.X_s is X_s
        assert model.X.shape[0] == 300  # candidate data not split

    def test_rejects_unboundable_constraint(self):
        # every sample in one group -> the other subgroup can never be bounded
        X = np.ones((200, 3))
        y = np.tile([0, 1], 100)
        ghats = [{"fn": ghat_tpr_diff_t(1, threshold=0.2), "delta": 0.05}]
        model = LogisticRegressionSeldonianGD(X, y, g_hats=ghats, epochs=10)
        with pytest.raises(RuntimeError):
            model.fit()
