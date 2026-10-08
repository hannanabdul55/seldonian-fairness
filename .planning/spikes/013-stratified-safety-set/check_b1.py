"""Coverage and width of the approximate bounds (b1 Wald-t, b1w Wilson-type), stratified vs
pooled on the same stratified sample, over rate profiles including low-rate strata.

    ../../../.venv/bin/python check_b1.py
"""
import numpy as np

import stratbounds as SB

# b1w at one stratum is the Wilson upper limit at every count, zero positives included
# (closed form z^2 / (n + z^2) there); the grid in b1w has step <= 1 / 4000
from scipy.stats import norm
for delta in (0.05, 0.1):
    z = norm.ppf(1 - delta)
    for n_ in (25, 100, 200, 400):
        k_ = np.arange(n_ + 1)
        p_ = k_ / n_
        wil = (p_ + z * z / (2 * n_) + z * np.sqrt(p_ * (1 - p_) / n_ + z * z / (4 * n_ * n_))) / (1 + z * z / n_)
        got = np.array([SB.b1w(k, n_, 1.0, delta) for k in k_])
        assert np.all(got >= wil - 1e-9) and np.all(got - wil <= 2.6e-4), (delta, n_)
    assert SB.b1w([0, 0, 0, 0], [25] * 4, [0.25] * 4, delta) > 0.0

rng = np.random.default_rng(1)
configs = {
    "H4 spread": [0.02, 0.15, 0.4, 0.8], "H4 flat": [0.3] * 4, "H8 spread": list(np.linspace(0, 0.9, 8)),
    "H2 low": [0.0, 0.1], "H4 rare": [0.0, 0.0, 0.01, 0.07], "H4 mid": [0.2, 0.3, 0.4, 0.5],
}
print("| config | n_s | delta | b1 miss | b1w miss | b1_pool miss | b1w_pool miss | ESS b1 | ESS b1w |")
print("|---|---|---|---|---|---|---|---|---|")
for name, mu_h in configs.items():
    mu_h = np.array(mu_h)
    H = len(mu_h)
    W = np.full(H, 1 / H)
    truth = float(W @ mu_h)
    for n_s in (100, 200, 400):
        for delta in (0.05, 0.1):
            n = np.full(H, n_s // H)
            reps = 4000
            miss = dict(b1=0, b1w=0, b1_pool=0, b1w_pool=0)
            wid = {k: 0.0 for k in miss}
            for _ in range(reps):
                sh = rng.binomial(n, mu_h)
                est = SB.estimate(sh, n, W)
                for k, fn in (("b1", SB.b1), ("b1w", SB.b1w)):
                    u = fn(sh, n, W, delta); miss[k] += truth > u; wid[k] += u - est
                    u = fn(sh.sum(), n.sum(), 1.0, delta); miss[k + "_pool"] += truth > u
                    wid[k + "_pool"] += u - sh.sum() / n.sum()
            f = {k: v / reps for k, v in miss.items()}
            print(f"| {name} | {n_s} | {delta} | {f['b1']:.4f} | {f['b1w']:.4f} | {f['b1_pool']:.4f} "
                  f"| {f['b1w_pool']:.4f} | {(wid['b1_pool'] / wid['b1']) ** 2:.2f} "
                  f"| {(wid['b1w_pool'] / wid['b1w']) ** 2:.2f} |")
