"""Black-box tests for the Seldonian classification models."""

import importlib

import numpy as np
import pytest
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from seldonian.objectives import ghat_tpr_diff, ghat_tpr_diff_t
from seldonian.seldonian import (
    LogisticRegressionSeldonianModel,
    SeldonianAlgorithmLogRegCMAES,
    VanillaNN,
)
from seldonian.synthetic import make_synthetic

CORE_MODULES = [
    "seldonian.algorithm",
    "seldonian.bounds",
    "seldonian.cmaes",
    "seldonian.constraint",
    "seldonian.examples",
    "seldonian.objectives",
    "seldonian.parser",
    "seldonian.policy",
    "seldonian.seldonian",
    "seldonian.synthetic",
    "seldonian.utils",
]


@pytest.mark.parametrize("module", CORE_MODULES)
def test_module_imports(module):
    importlib.import_module(module)


def separable_data(n=400, d=4, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    w = rng.standard_normal(d)
    y = (X @ w > 0).astype(int)
    return X, y


class TestLogisticRegressionSeldonianModel:
    def test_unconstrained_learns_separable_data(self):
        X, y = separable_data()
        model = LogisticRegressionSeldonianModel(X, y, g_hats=[], verbose=False)
        result = model.fit()
        assert result is model  # no constraints -> safety trivially passes
        preds = model.predict(X)
        assert preds.shape == (len(X),)
        assert set(np.unique(preds)) <= {0, 1}
        assert accuracy_score(y, preds) > 0.8

    def test_safety_test_contract(self):
        X, y = separable_data()
        model = LogisticRegressionSeldonianModel(X, y, g_hats=[], verbose=False)
        model.fit()
        assert float(model._safetyTest()) >= 0
        assert model.safetyTest() in (True, False)
        assert model.safetyTest() is True  # unconstrained model must pass

    def test_constrained_fit_returns_model_or_none(self):
        np.random.seed(1)
        X, y, A_idx = make_synthetic(5000, 5)
        ghats = [{"fn": ghat_tpr_diff(A_idx, threshold=0.2), "delta": 0.05}]
        model = LogisticRegressionSeldonianModel(X, y, g_hats=ghats, verbose=False)
        result = model.fit(opt="Powell")
        assert result is model or result is None
        if result is model:
            assert model.safetyTest() is True
            assert balanced_accuracy_score(y, model.predict(X)) > 0.5

    def test_explicit_safety_data_used(self):
        X, y = separable_data(300)
        X_s, y_s = separable_data(100, seed=8)
        model = LogisticRegressionSeldonianModel(
            X, y, g_hats=[], safety_data=(X_s, y_s), verbose=False)
        assert model.X_s is X_s
        assert model.y_s is y_s
        # candidate data is not split when safety data is supplied
        assert model.X.shape[0] == 300

    def test_data_returns_candidate_set(self):
        X, y = separable_data(200)
        model = LogisticRegressionSeldonianModel(X, y, g_hats=[], test_size=0.5,
                                                 verbose=False)
        X_c, y_c = model.data()
        assert X_c.shape[0] == y_c.shape[0]
        assert X_c.shape[0] < 200  # part went to the safety set


class TestSeldonianAlgorithmLogRegCMAES:
    def test_constrained_fit_and_predict(self):
        np.random.seed(1)
        X, y, A_idx = make_synthetic(1500, 5)
        ghats = [{"fn": ghat_tpr_diff(A_idx, threshold=0.2), "delta": 0.05}]
        model = SeldonianAlgorithmLogRegCMAES(X, y, g_hats=ghats, maxiter=800,
                                              random_seed=1)
        model.fit()
        preds = model.predict(X)
        assert preds.shape == (len(X),)
        assert set(np.unique(preds)) <= {0, 1}
        assert float(model._safetyTest()) >= 0

    def test_parameters_returns_learnt_theta(self):
        X, y = separable_data(200)
        model = SeldonianAlgorithmLogRegCMAES(X, y, g_hats=[], maxiter=400)
        model.fit()
        theta, C = model.parameters()
        assert np.asarray(theta).size == X.shape[1] + 1
        assert np.all(np.isfinite(np.asarray(theta, dtype=float)))


class TestVanillaNN:
    @pytest.fixture(autouse=True)
    def _seed_torch(self):
        # network weight init is drawn from torch's global RNG
        torch.manual_seed(0)

    def test_unconstrained_training(self):
        # batch size is 300, so with ~180 candidate samples each epoch is a single
        # gradient step - give it enough epochs to actually learn
        X, y = separable_data(300, seed=11)
        model = VanillaNN(X, y, g_hats=[], epochs=200)
        model.fit()
        preds = model.predict(X)
        assert torch.is_tensor(preds)
        assert preds.shape[0] == len(X)
        assert accuracy_score(y, preds.cpu().numpy()) > 0.6

    def test_pmf_predictions_are_probabilities(self):
        X, y = separable_data(200, seed=12)
        model = VanillaNN(X, y, g_hats=[], epochs=2)
        model.fit()
        pmf = model.predict(X, pmf=True).detach().cpu().numpy()
        assert np.all((pmf >= 0) & (pmf <= 1))

    def test_constrained_training_reports_safety(self):
        np.random.seed(2)
        X, y, A_idx = make_synthetic(600, 5)
        ghats = [{"fn": ghat_tpr_diff_t(A_idx, threshold=0.2), "delta": 0.05}]
        model = VanillaNN(X, y, g_hats=ghats, epochs=2)
        model.fit()
        safety = float(model._safetyTest())
        assert np.isfinite(safety)
        assert safety >= 0
