"""Black-box tests for seldonian.utils."""

import numpy as np

from seldonian.utils import cross_entropy_loss, sigmoid


class TestSigmoid:
    def test_midpoint(self):
        assert sigmoid(0) == 0.5

    def test_saturation(self):
        assert sigmoid(50) > 0.999
        assert sigmoid(-50) < 0.001

    def test_symmetry(self):
        assert np.isclose(sigmoid(2.0) + sigmoid(-2.0), 1.0)

    def test_vectorized_and_monotonic(self):
        x = np.linspace(-5, 5, 11)
        out = sigmoid(x)
        assert out.shape == x.shape
        assert np.all(np.diff(out) > 0)
        assert np.all((out > 0) & (out < 1))


class TestCrossEntropyLoss:
    def test_known_values(self):
        losses = cross_entropy_loss(np.array([0.9, 0.1]), np.array([1, 0]))
        assert np.allclose(losses, [-np.log(0.9), -np.log(0.9)])

    def test_confident_correct_prediction_near_zero(self):
        assert cross_entropy_loss(np.array([0.9999]), np.array([1]))[0] < 1e-3

    def test_confident_wrong_prediction_is_large(self):
        assert cross_entropy_loss(np.array([0.0001]), np.array([1]))[0] > 5

    def test_uncertain_prediction_is_log2(self):
        assert np.isclose(cross_entropy_loss(np.array([0.5]), np.array([1]))[0],
                          np.log(2))
