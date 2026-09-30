"""Checks for stratbounds.py (spike 013, stage 0).

1. Pooled sanity: at one stratum, b1 ~ Wald and b2 sits beside the project's betting and
   Clopper-Pearson bounds.
2. Coverage: stratified Bernoulli data with known stratum means (including near-0 strata),
   P(true mean > UB) <= delta, for pool-free (fixed W) and two-phase (W from a pool of N).
3. Gain: width of the stratified bound vs the pooled bound on the same kind of data.

    ../../../.venv/bin/python check_bounds.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from seldonian.bounds import betting_mixture_bounds, clopper_pearson_bounds  # noqa: E402

import stratbounds as SB  # noqa: E402

rng = np.random.default_rng(0)
delta = 0.1

print("## 1. pooled (one stratum), n = 200")
for p in (0.05, 0.3, 0.5):
    x = (rng.random(200) < p).astype(float)
    s = x.sum()
    t0 = time.time()
    u2 = SB.b2(s, 200, 1.0, delta)
    dt = time.time() - t0
    print(f"p {p}: mean {x.mean():.3f}  b1 {SB.b1(s, 200, 1.0, delta):.4f}  b2 {u2:.4f} "
          f"({dt * 1000:.0f} ms)  project betting {betting_mixture_bounds(x, delta).upper:.4f}  "
          f"CP {clopper_pearson_bounds(x, delta).upper:.4f}")

print("\n## 2-3. coverage and width, proportional allocation, 2000 reps each")
configs = {
    "H4 spread": np.array([0.02, 0.15, 0.4, 0.8]),
    "H4 flat": np.array([0.3, 0.3, 0.3, 0.3]),
    "H8 spread": np.linspace(0.0, 0.9, 8),
    "H2 low": np.array([0.0, 0.1]),
}
for name, mu_h in configs.items():
    H = len(mu_h)
    for n_s in (100, 400):
        n = np.full(H, n_s // H)
        W = np.full(H, 1.0 / H)
        truth = float(np.sum(W * mu_h))
        miss = {"b1": 0, "b2": 0, "b1_pool": 0, "b2_pool": 0}
        width = {k: [] for k in miss}
        reps = 2000 if n_s == 100 else 1000
        for r in range(reps):
            s = rng.binomial(n, mu_h)
            est = SB.estimate(s, n, W)
            for k, fn in (("b1", SB.b1), ("b2", SB.b2)):
                ub = fn(s, n, W, delta)
                miss[k] += truth > ub
                width[k].append(ub - est)
            # the pooled bound on the same stratified sample (arm S1): valid by AM-GM
            for k, fn in (("b1_pool", SB.b1), ("b2_pool", SB.b2)):
                ub = fn(s.sum(), n.sum(), 1.0, delta)
                miss[k] += truth > ub
                width[k].append(ub - s.sum() / n.sum())
        print(f"{name} n_s {n_s}: " + "  ".join(
            f"{k} miss {miss[k] / reps:.3f} w {np.mean(width[k]):.4f}" for k in miss)
              + f"  | ESS b1 {(np.mean(width['b1_pool']) / np.mean(width['b1'])) ** 2:.2f}"
              f" b2 {(np.mean(width['b2_pool']) / np.mean(width['b2'])) ** 2:.2f}")

print("\n## two-phase: pool of N from a population, target = population mean")
mu_h = np.array([0.02, 0.15, 0.4, 0.8])
for N, n_s in ((1000, 400), (5000, 200)):
    miss = {"b1": 0, "b2": 0}
    reps = 1000
    for r in range(reps):
        Nh = rng.multinomial(N, np.full(4, 0.25))
        W = Nh / N
        n = np.maximum(np.round(n_s * W).astype(int), 2)
        s = rng.binomial(n, mu_h)
        truth = 0.25 * mu_h.sum()
        for k, fn in (("b1", SB.b1), ("b2", SB.b2)):
            miss[k] += truth > fn(s, n, W, delta, N=N)
    print(f"N {N} n_s {n_s}: " + "  ".join(f"{k} miss {miss[k] / reps:.3f}" for k in miss))
