"""Black-box tests for seldonian.bounds: RandomVariable interval arithmetic and
concentration bounds (ttest / Hoeffding's)."""

import numpy as np
import pytest
import torch
from scipy.stats import t as t_dist

from seldonian.bounds import (
    RandomVariable,
    hoeffdings_bounds,
    max_bounds,
    min_bounds,
    ttest_bounds,
)


class TestRandomVariableConstruction:
    def test_plain_value_collapses_interval(self):
        rv = RandomVariable(3.0)
        assert rv.value == 3.0
        assert rv.lower == 3.0
        assert rv.upper == 3.0

    def test_explicit_bounds_kept(self):
        rv = RandomVariable(1.0, lower=0.0, upper=2.0)
        assert (rv.value, rv.lower, rv.upper) == (1.0, 0.0, 2.0)

    def test_rejects_non_numeric(self):
        with pytest.raises(ValueError):
            RandomVariable("not a number")

    def test_str_mentions_value_and_bounds(self):
        s = str(RandomVariable(1.0, lower=0.0, upper=2.0))
        assert "1.0" in s and "0.0" in s and "2.0" in s


class TestRandomVariableArithmetic:
    def test_addition(self):
        c = RandomVariable(1.0, 0.0, 2.0) + RandomVariable(10.0, 5.0, 15.0)
        assert (c.value, c.lower, c.upper) == (11.0, 5.0, 17.0)

    def test_addition_with_scalar(self):
        c = RandomVariable(1.0, 0.0, 2.0) + 5
        assert (c.value, c.lower, c.upper) == (6.0, 5.0, 7.0)

    def test_negation_swaps_bounds(self):
        c = -RandomVariable(1.0, 0.0, 2.0)
        assert (c.value, c.lower, c.upper) == (-1.0, -2.0, 0.0)

    def test_subtraction(self):
        c = RandomVariable(5.0, 4.0, 6.0) - RandomVariable(1.0, 0.0, 2.0)
        assert (c.value, c.lower, c.upper) == (4.0, 2.0, 6.0)

    def test_multiplication_positive_intervals(self):
        c = RandomVariable(2.0, 1.0, 3.0) * RandomVariable(4.0, 2.0, 5.0)
        assert (c.value, c.lower, c.upper) == (8.0, 2.0, 15.0)

    def test_multiplication_mixed_sign_intervals(self):
        c = RandomVariable(-1.0, -2.0, 1.0) * RandomVariable(3.0, 2.0, 4.0)
        # products of endpoints: {-8, -4, 2, 4} -> [-8, 4]
        assert (c.value, c.lower, c.upper) == (-3.0, -8.0, 4.0)

    def test_division_by_positive_interval(self):
        c = RandomVariable(4.0, 2.0, 8.0) / RandomVariable(2.0, 1.0, 4.0)
        assert c.value == 2.0
        assert c.lower == 0.5
        assert c.upper == 8.0

    def test_division_by_interval_containing_zero(self):
        c = RandomVariable(1.0, 1.0, 1.0) / RandomVariable(0.5, -1.0, 1.0)
        assert c.lower == -np.inf
        assert c.upper == np.inf

    def test_abs_straddling_zero(self):
        c = abs(RandomVariable(-1.0, -3.0, 2.0))
        assert (c.value, c.lower, c.upper) == (1.0, 0.0, 3.0)

    def test_interval_contains_value_through_chain(self):
        a = RandomVariable(1.0, 0.5, 1.5)
        b = RandomVariable(2.0, 1.0, 3.0)
        for rv in [a + b, a - b, a * b, a / b, abs(a - b)]:
            assert rv.lower <= rv.value <= rv.upper


class TestMinMaxBounds:
    def test_componentwise_min(self):
        lo = min_bounds(RandomVariable(1.0, 0.0, 2.0), RandomVariable(2.0, -5.0, 3.0))
        assert (lo.value, lo.lower, lo.upper) == (1.0, -5.0, 2.0)

    def test_componentwise_max(self):
        hi = max_bounds(RandomVariable(1.0, 0.0, 2.0), RandomVariable(2.0, -5.0, 3.0))
        assert (hi.value, hi.lower, hi.upper) == (2.0, 0.0, 3.0)

    def test_accepts_plain_numbers(self):
        lo = min_bounds(RandomVariable(1.0, 0.0, 2.0), 5.0)
        assert (lo.value, lo.lower, lo.upper) == (1.0, 0.0, 2.0)


class TestTtestBounds:
    def test_interval_is_symmetric_around_mean(self):
        samples = np.array([0.0, 1.0, 1.0, 0.0, 1.0])
        rv = ttest_bounds(samples, delta=0.05)
        assert np.isclose(rv.value, samples.mean())
        assert np.isclose(rv.value - rv.lower, rv.upper - rv.value)

    def test_matches_manual_formula(self):
        samples = np.array([0.0, 1.0, 1.0, 0.0, 1.0])
        rv = ttest_bounds(samples, delta=0.05)
        dev = samples.std(ddof=1) / np.sqrt(5) * t_dist.ppf(0.95, 4)
        assert np.isclose(rv.upper - rv.value, dev)

    def test_width_shrinks_with_more_samples(self):
        rng = np.random.default_rng(0)
        small = ttest_bounds(rng.random(20), delta=0.05)
        large = ttest_bounds(rng.random(2000), delta=0.05)
        assert (large.upper - large.lower) < (small.upper - small.lower)

    def test_predict_doubles_width(self):
        samples = np.random.default_rng(1).random(50)
        plain = ttest_bounds(samples, delta=0.05)
        predicted = ttest_bounds(samples, delta=0.05, predict=True)
        assert np.isclose(predicted.upper - predicted.value,
                          2 * (plain.upper - plain.value))

    def test_smaller_delta_widens_interval(self):
        samples = np.random.default_rng(2).random(50)
        wide = ttest_bounds(samples, delta=0.01)
        narrow = ttest_bounds(samples, delta=0.2)
        assert (wide.upper - wide.lower) > (narrow.upper - narrow.lower)

    def test_rejects_2d_input(self):
        with pytest.raises(ValueError):
            ttest_bounds(np.ones((3, 3)), delta=0.05)

    def test_rejects_non_array(self):
        with pytest.raises(ValueError):
            ttest_bounds("bogus", delta=0.05)

    def test_accepts_tensor(self):
        rv = ttest_bounds(torch.tensor([0.0, 1.0, 1.0, 0.0]), delta=0.05)
        assert np.isclose(float(rv.value), 0.5)
        assert float(rv.lower) < 0.5 < float(rv.upper)


class TestHoeffdingsBounds:
    def test_matches_closed_form(self):
        samples = np.random.default_rng(3).random(100)
        rv = hoeffdings_bounds(samples, delta=0.05)
        dev = np.sqrt(np.log(1 / 0.05) / (2 * 100))
        assert np.isclose(rv.value, samples.mean())
        assert np.isclose(rv.upper - rv.value, dev)

    def test_predict_doubles_width(self):
        samples = np.random.default_rng(4).random(100)
        plain = hoeffdings_bounds(samples, delta=0.05)
        predicted = hoeffdings_bounds(samples, delta=0.05, predict=True)
        assert np.isclose(predicted.upper - predicted.value,
                          2 * (plain.upper - plain.value))

    def test_rejects_2d_input(self):
        with pytest.raises(ValueError):
            hoeffdings_bounds(np.ones((2, 2)), delta=0.05)

    def test_accepts_tensor(self):
        rv = hoeffdings_bounds(torch.tensor([0.2, 0.4, 0.6, 0.8]), delta=0.05)
        assert np.isclose(float(rv.value), 0.5)
        assert float(rv.lower) < 0.5 < float(rv.upper)
