import numpy as np
import pytest
from sklearn.metrics import accuracy_score

from seldonian.examples import LogisticRegressionCMAES


def make_separable(n=400, d=4, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    w = rng.standard_normal(d)
    y = (X @ w > 0).astype(int)
    return X, y


@pytest.mark.parametrize("optimizer", ["native", "pycma"])
def test_cmaes_fits_separable_data(optimizer):
    X, y = make_separable()
    model = LogisticRegressionCMAES(X, y, optimizer=optimizer, maxiter=2000, random_seed=0)
    model.fit()
    assert np.all(np.isfinite(model.theta))
    assert accuracy_score(y, model.predict(X)) > 0.8


@pytest.mark.parametrize("optimizer", ["native", "pycma"])
def test_cmaes_deterministic_with_seed(optimizer):
    X, y = make_separable()
    thetas = []
    for _ in range(2):
        model = LogisticRegressionCMAES(X, y, optimizer=optimizer, maxiter=800, random_seed=3)
        model.fit()
        thetas.append(np.asarray(model.theta, dtype=float).flatten())
    np.testing.assert_allclose(thetas[0], thetas[1])


def test_cmaes_rejects_unknown_optimizer():
    X, y = make_separable(n=50)
    with pytest.raises(ValueError):
        LogisticRegressionCMAES(X, y, optimizer="bogus")
