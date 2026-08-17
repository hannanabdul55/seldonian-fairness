"""Black-box tests for seldonian.synthetic.make_synthetic."""

import numpy as np

from seldonian.synthetic import make_synthetic


class TestMakeSynthetic:
    def test_shapes_and_ranges(self):
        X, y, A_idx = make_synthetic(500, 6)
        assert X.shape == (500, 6)
        assert y.shape == (500,)
        assert 1 <= A_idx <= 4
        assert set(np.unique(y)) <= {0.0, 1.0}
        assert set(np.unique(X[:, A_idx])) <= {0.0, 1.0}

    def test_accepts_float_n(self):
        X, y, _ = make_synthetic(1e2, 4)
        assert X.shape[0] == 100

    def test_deterministic_for_same_seed(self):
        X1, y1, a1 = make_synthetic(200, 5, seed=42)
        X2, y2, a2 = make_synthetic(200, 5, seed=42)
        assert np.array_equal(X1, X2)
        assert np.array_equal(y1, y2)
        assert a1 == a2

    def test_different_seeds_differ(self):
        X1, _, _ = make_synthetic(200, 5, seed=1)
        X2, _, _ = make_synthetic(200, 5, seed=2)
        assert not np.array_equal(X1, X2)

    def test_group_positive_rates_match_parameters(self):
        X, y, A_idx = make_synthetic(20000, 5, tp_a=0.3, tp_b=0.7, seed=0)
        rate_a = y[X[:, A_idx] == 1].mean()
        rate_b = y[X[:, A_idx] == 0].mean()
        assert abs(rate_a - 0.3) < 0.03
        assert abs(rate_b - 0.7) < 0.03

    def test_first_feature_is_informative_but_not_perfect(self):
        X, y, _ = make_synthetic(5000, 5, seed=0)
        # positives have shifted mean on feature 0
        assert X[y == 1, 0].mean() > X[y == 0, 0].mean() + 0.5
        # but the label is not perfectly recoverable from feature 0 alone
        best_threshold_acc = max(
            ((X[:, 0] > thr) == y).mean() for thr in np.linspace(-2, 3, 51))
        assert best_threshold_acc < 0.95
