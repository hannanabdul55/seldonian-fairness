"""
Tests for seldonian.llm.smooth_bounds (phase 2 of the safe-RL bounds workstream, property P12):
the smooth predicted bounds equal or upper-bound the exact library bounds, and their autograd
derivatives agree with finite differences.
"""
import numpy as np
import pytest
import torch

from seldonian.bounds import clopper_pearson_bounds, empirical_bernstein_bounds
from seldonian.llm.smooth_bounds import Interval, cp_upper, eb_upper, smooth_max


def test_cp_upper_equals_library_at_real_k():
    rng = np.random.default_rng(0)
    for _ in range(200):
        m0 = int(rng.integers(10, 500))
        k1 = int(rng.integers(0, m0))
        n = int(rng.integers(20, 800))
        x = np.r_[np.ones(k1), np.zeros(m0 - k1)]
        exact = float(clopper_pearson_bounds(x, 0.05, n=n).upper)
        assert float(cp_upper(k1 / m0, n, 0.05)) == pytest.approx(exact, abs=1e-12)


@pytest.mark.parametrize("n,delta", [(50, 0.1), (205, 0.1), (1000, 0.05)])
def test_cp_upper_gradient_matches_finite_difference(n, delta):
    for mu0 in np.linspace(0.005, 0.8, 25):
        mu = torch.tensor(mu0, dtype=torch.float64, requires_grad=True)
        cp_upper(mu, n, delta).backward()
        h = 1e-5
        fd = (float(cp_upper(mu0 + h, n, delta)) - float(cp_upper(mu0 - h, n, delta))) / (2 * h)
        assert mu.grad.item() == pytest.approx(fd, rel=1e-4)
        assert mu.grad.item() > 0          # monotone in the rate


def test_eb_upper_matches_library_and_is_differentiable():
    rng = np.random.default_rng(1)
    for _ in range(100):
        x = rng.beta(2, 5, size=int(rng.integers(10, 300)))
        exact = float(empirical_bernstein_bounds(x, 0.1).upper)
        mu = torch.tensor(x.mean(), dtype=torch.float64, requires_grad=True)
        var = torch.tensor(x.var(ddof=1), dtype=torch.float64, requires_grad=True)
        u = eb_upper(mu, var, len(x), 0.1)
        # the library clips to 1; the smooth one does not, so it is equal or above
        assert float(u) >= exact - 1e-12
        if exact < 1.0:
            assert float(u) == pytest.approx(exact, abs=1e-6)
        u.backward()
        assert mu.grad.item() == pytest.approx(1.0)
        assert var.grad.item() > 0


def test_smooth_max_upper_bounds_max():
    rng = np.random.default_rng(2)
    for _ in range(200):
        xs = list(rng.normal(size=int(rng.integers(2, 6))) * 0.1)
        tau = 10 ** rng.uniform(-4, -1)
        s = float(smooth_max(xs, tau))
        assert max(xs) - 1e-12 <= s <= max(xs) + tau * np.log(len(xs)) + 1e-12


def test_interval_abs_difference_upper_bounds_exact():
    # |mu_a - mu_b| with both endpoints from Clopper-Pearson: the smooth upper endpoint is
    # at least the exact interval-arithmetic one
    a = Interval(0.05, cp_upper(0.10, 300, 0.025))
    b = Interval(0.02, cp_upper(0.04, 300, 0.025))
    d = abs(a - b)
    exact = max(float(a.upper) - 0.02, 0.05 - float(b.upper))
    assert float(d.upper) >= exact - 1e-12
    assert float(d.upper) <= exact + 1e-3 * np.log(2) + 1e-12


def test_gradient_flows_through_interval_composition():
    mu_a = torch.tensor(0.10, dtype=torch.float64, requires_grad=True)
    mu_b = torch.tensor(0.04, dtype=torch.float64, requires_grad=True)
    a = Interval(mu_a - 0.05, cp_upper(mu_a, 300, 0.025))
    b = Interval(mu_b - 0.02, cp_upper(mu_b, 300, 0.025))
    g = Interval.max(abs(a - b), a.scale(0.5)).upper - 0.1
    g.backward()
    assert mu_a.grad.item() > 0           # raising the larger rate raises the bound
    assert mu_b.grad.item() < 0           # raising the smaller one narrows the gap
