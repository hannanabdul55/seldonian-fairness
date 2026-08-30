"""Black-box tests for the gradient-based Seldonian neural network."""

import numpy as np
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from seldonian.objectives import ghat_tpr_diff_t, tpr_rate
from seldonian.seldonian import LogisticRegressionSeldonianGD, NeuralNetSeldonianGD
from seldonian.synthetic import make_synthetic


def constrained_setup(n=8000):
    np.random.seed(1)
    X, y, A_idx = make_synthetic(n, 5, tp_a=0.4, tp_b=0.8, seed=1)
    ghats = [{"fn": ghat_tpr_diff_t(A_idx, threshold=0.2), "delta": 0.05}]
    return X, y, A_idx, ghats


class TestNeuralNetSeldonianGD:
    def test_learns_nonlinear_boundary_linear_cannot(self):
        X, y = make_moons(2000, noise=0.15, random_state=0)
        net = NeuralNetSeldonianGD(X, y, g_hats=[], epochs=400, random_seed=0)
        net.fit()
        linear = LogisticRegressionSeldonianGD(X, y, g_hats=[], epochs=400,
                                               random_seed=0)
        linear.fit()
        nn_acc = accuracy_score(y, net.predict(X))
        lin_acc = accuracy_score(y, linear.predict(X))
        assert nn_acc > 0.95
        assert nn_acc > lin_acc + 0.05

    def test_constrained_fit_satisfies_hard_safety_test(self):
        X, y, A_idx, ghats = constrained_setup()
        model = NeuralNetSeldonianGD(X, y, g_hats=ghats, random_seed=0)
        result = model.fit()
        assert result is model
        assert float(model._safetyTest()) == 0
        preds = model.predict(X)
        assert balanced_accuracy_score(y, preds) > 0.5
        tpr_a = tpr_rate(A_idx, 1)(X, y, preds).mean()
        tpr_b = tpr_rate(A_idx, 0)(X, y, preds).mean()
        assert abs(tpr_a - tpr_b) < 0.2

    def test_custom_model_is_used(self):
        X, y, _, ghats = constrained_setup(2000)
        custom = nn.Sequential(nn.Linear(X.shape[1], 8), nn.Tanh(), nn.Linear(8, 2))
        model = NeuralNetSeldonianGD(X, y, g_hats=ghats, model=custom, epochs=50)
        assert model.mod is custom
        model.fit()
        assert model.predict(X).shape == (len(X),)

    def test_hidden_layers_configure_architecture(self):
        X, y, _, _ = constrained_setup(1000)
        model = NeuralNetSeldonianGD(X, y, g_hats=[], hidden_layers=(4,), epochs=10)
        linear_layers = [m for m in model.mod if isinstance(m, nn.Linear)]
        assert len(linear_layers) == 2
        assert linear_layers[0].out_features == 4

    def test_deterministic_for_seed(self):
        X, y, _, ghats = constrained_setup(2000)
        preds = []
        for _ in range(2):
            model = NeuralNetSeldonianGD(X, y, g_hats=ghats, epochs=80, random_seed=3)
            model.fit()
            preds.append(model.predict(X))
        assert np.array_equal(preds[0], preds[1])
