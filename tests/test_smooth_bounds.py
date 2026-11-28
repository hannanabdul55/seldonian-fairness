"""
Tests for seldonian.llm.smooth_bounds (phase 2 of the safe-RL bounds workstream, property P12):
the smooth predicted bounds equal or upper-bound the exact library bounds, and their autograd
derivatives agree with finite differences.
"""
import json
import os
import sys

import numpy as np
import pytest
import torch

from seldonian.bounds import (bentkus_bounds, betting_mixture_bounds, clopper_pearson_bounds,
                              empirical_bernstein_bounds)
from seldonian.llm.smooth_bounds import (Interval, bentkus_lower, bentkus_upper,
                                         betting_influence, betting_lower, betting_upper,
                                         cp_lower, cp_upper, eb_upper, repeated_measures,
                                         smooth_max, spec_delta_side, spec_interval,
                                         spec_is_monotone)


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


# ------------------------------------------------------------------ Clopper-Pearson lower

def test_cp_lower_equals_library():
    rng = np.random.default_rng(3)
    for _ in range(100):
        m0 = int(rng.integers(10, 400))
        k1 = int(rng.integers(0, m0 + 1))
        n = int(rng.integers(20, 600))
        x = np.r_[np.ones(k1), np.zeros(m0 - k1)]
        exact = float(clopper_pearson_bounds(x, 0.05, n=n).lower)
        assert float(cp_lower(k1 / m0, n, 0.05)) == pytest.approx(exact, abs=1e-9)


# ------------------------------------------------------------------ Bentkus

def test_bentkus_equals_library_and_is_monotone():
    rng = np.random.default_rng(4)
    for _ in range(60):
        x = rng.beta(1, 6, size=int(rng.integers(20, 300)))
        n = int(rng.integers(50, 500))
        rv = bentkus_bounds(x, 0.05, n=n)
        mu = torch.tensor(x.mean(), dtype=torch.float64, requires_grad=True)
        u = bentkus_upper(mu, n, 0.05)
        assert float(u) == pytest.approx(float(rv.upper), abs=1e-9)
        assert float(bentkus_lower(x.mean(), n, 0.05)) == pytest.approx(float(rv.lower),
                                                                       abs=1e-9)
        u.backward()
        assert mu.grad.item() > 0


def test_bentkus_gradient_matches_finite_difference_off_kinks():
    n, delta = 300, 0.05
    for mu0 in np.linspace(0.02, 0.6, 15) + 0.5 / n:     # half-way between integer k
        mu = torch.tensor(mu0, dtype=torch.float64, requires_grad=True)
        bentkus_upper(mu, n, delta).backward()
        h = 1e-4
        fd = (float(bentkus_upper(mu0 + h, n, delta))
              - float(bentkus_upper(mu0 - h, n, delta))) / (2 * h)
        assert mu.grad.item() == pytest.approx(fd, rel=1e-3)


# ------------------------------------------------------------------ betting mixture

@pytest.mark.parametrize("n", [None, 400])
def test_betting_equals_library(n):
    rng = np.random.default_rng(5)
    for _ in range(30):
        x = rng.beta(1.5, 6, size=int(rng.integers(30, 300)))
        rv = betting_mixture_bounds(x, 0.05, n=n)
        nn = n or x.size
        assert float(betting_upper(x, nn, 0.05)) == pytest.approx(float(rv.upper), abs=1e-7)
        assert float(betting_lower(x, nn, 0.05)) == pytest.approx(float(rv.lower), abs=1e-7)


def test_betting_on_a_wider_range_maps_back():
    rng = np.random.default_rng(6)
    x = rng.uniform(-1, 1, size=200) * 0.3
    rv = betting_mixture_bounds(x, 0.05, a=-1.0, b=1.0)
    assert float(betting_upper(x, 200, 0.05, a=-1.0, b=1.0)) == pytest.approx(float(rv.upper),
                                                                            abs=1e-7)


def test_betting_influence_matches_reweighting():
    # psi_i: moving weight eps onto sample i changes the bound by eps * psi_i
    rng = np.random.default_rng(7)
    x = rng.beta(2, 8, size=120)
    n, delta = 300, 0.05
    u0, psi = betting_influence(x, n, delta)
    assert float(np.mean(psi)) == pytest.approx(0.0, abs=1e-9)
    eps = 1e-5
    for i in rng.choice(x.size, size=10, replace=False):
        p = np.full(x.size, (1 - eps) / x.size)
        p[i] += eps
        u1, _ = betting_influence(x, n, delta, weights=p)
        assert (u1 - u0) / eps == pytest.approx(psi[i], rel=2e-3, abs=1e-5)


def test_betting_influence_is_increasing_in_harm():
    # monotone in each sample (safe to steer on): influence rises with the sample's value
    rng = np.random.default_rng(8)
    x = np.clip(rng.normal(0.3, 0.01, size=200), 0, 1)
    x[0] = 0.0
    _, psi = betting_influence(x, 200, 0.05)
    order = np.argsort(x)
    assert np.all(np.diff(psi[order]) >= -1e-9)


def test_betting_weight_gradient_through_autograd():
    rng = np.random.default_rng(9)
    x = rng.beta(2, 8, size=80)
    w = torch.ones(x.size, dtype=torch.float64, requires_grad=True)
    u = betting_upper(x, 200, 0.05, weights=w)
    u.backward()
    _, psi = betting_influence(x, 200, 0.05)
    assert np.allclose(w.grad.numpy(), psi / x.size, atol=1e-12)


# ------------------------------------------------------------------ specs of spike 016

SPIKE016 = os.path.join(os.path.dirname(__file__), "..", ".planning", "spikes",
                        "016-prompt-to-spec-compile")


def _spike016_specs():
    sys.path.insert(0, SPIKE016)
    import speclab
    specs = {}
    for name in ("results.json", "results_v2.json"):
        for r in json.load(open(os.path.join(SPIKE016, name))):
            t = r.get("dsl")
            if t and t != "None":
                sp = speclab.parse_dsl(t)
                specs[json.dumps(sp, sort_keys=True)] = sp
    return speclab, list(specs.values())


def test_smooth_spec_evaluator_upper_bounds_exact_for_every_016_spec():
    speclab, specs = _spike016_specs()
    assert len(specs) > 100
    from seldonian.bounds import RandomVariable
    rng = np.random.default_rng(10)
    checked = 0
    for sp in specs:
        for _ in range(5):
            pts = {k: rng.uniform(0.1, 0.6) for k in sp["measures"]}
            wid = {k: (rng.uniform(0.01, 0.08), rng.uniform(0.01, 0.08)) for k in pts}
            exact_in = {k: RandomVariable(p, lower=p - wid[k][0], upper=p + wid[k][1])
                        for k, p in pts.items()}
            mus = {k: torch.tensor(p, dtype=torch.float64, requires_grad=True)
                   for k, p in pts.items()}
            smooth_in = {k: Interval(mus[k] - wid[k][0], mus[k] + wid[k][1]) for k in pts}
            ex = speclab._evaluator(sp["expr"])(exact_in)
            ex_u = float(ex.upper) if isinstance(ex, RandomVariable) else float(ex)
            sm = spec_interval(sp, smooth_in).upper
            assert float(sm) >= ex_u - 1e-12
            assert float(sm) <= ex_u + 0.01          # tau = 1e-3, a few smooth maxes
            sm.backward()
            assert all(np.isfinite(m.grad.item()) for m in mus.values())
            checked += 1
    assert checked == 5 * len(specs)


def test_spec_delta_split_and_repeated_measure_lint():
    speclab, specs = _spike016_specs()
    for sp in specs:
        assert spec_is_monotone(sp) == speclab.is_monotone(sp)
    sp = {"measures": {"a": {}, "b": {}}, "expr": ["-", ["m", "a"], ["*", ["m", "b"],
                                                                     ["m", "a"]]]}
    assert repeated_measures(sp) == ["a"]
    assert spec_delta_side(sp, 0.1) == pytest.approx(0.025)
