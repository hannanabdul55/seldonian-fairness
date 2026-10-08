"""Exact miss probability of the one-sided Wilson upper limit, by enumeration over binomial counts.

The limit at count k of n is the larger root U(k) of (m - k / n)^2 = z^2 m (1 - m) / n, z the
1 - delta normal quantile. It misses a true rate p when U(X) < p, X ~ Binomial(n, p). U is
increasing in k, so the miss probability at p is P(X <= k(p)), with k(p) the largest count whose
limit is under p. Between two consecutive limits it falls as p grows, so its supremum over an
interval is taken just above a limit U(k), where it equals P(X <= k | p = U(k)). Nothing is
simulated.

At zero positives U(0) = z^2 / (n + z^2), and the miss just above it is (n / (n + z^2))^n, which
is close to exp(-z^2) at any n and tends to it as n grows: 0.067 at delta 0.05 and 0.194 at delta
0.10.

The same enumeration is done for Clopper-Pearson, whose miss probability cannot exceed delta, as
a check of the method. It also asserts that ``b1w`` of spike 013 at one stratum is this limit.

    .venv/bin/python scripts/wilson_exact.py
    -> results/paper/wilson_exact.json, results/paper/wilson_exact.md
"""
import json
import os
import sys

import numpy as np
from scipy.stats import beta, binom, norm

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "results", "paper")
sys.path.insert(0, os.path.join(ROOT, ".planning", "spikes", "013-stratified-safety-set"))
import stratbounds as SB       # noqa: E402

NS = (59, 100, 149, 200, 299, 400, 1000)
DELTAS = (0.05, 0.1)
BANDS = ((0.005, 0.01), (0.01, 0.02), (0.02, 0.05), (0.05, 0.20), (0.20, 0.50), (0.50, 0.80), (0.80, 0.95), (0.95, 0.99))
GRID = 4001          # rates per band for the mean and the share over delta


def wilson_upper(k, n, delta):
    z2 = norm.ppf(1 - delta) ** 2
    ph = np.asarray(k, dtype=float) / n
    return (ph + z2 / (2 * n) + np.sqrt(z2 * (ph * (1 - ph) / n + z2 / (4 * n * n)))) / (1 + z2 / n)


def cp_upper(k, n, delta):
    k = np.asarray(k, dtype=float)
    return np.where(k >= n, 1.0, beta.ppf(1 - delta, k + 1, np.maximum(n - k, 1e-12)))


def miss_at(p, n, delta, upper=None):
    """P(U(X) < p) for X ~ Binomial(n, p): the miss probability of an i.i.d. sample at true rate p."""
    U = (upper or wilson_upper)(np.arange(n + 1), n, delta)
    return float(binom.pmf(np.arange(n + 1), n, p)[U < p].sum())


def peaks(upper, n, delta):
    """For every count k < n: the limit U(k) and the supremum of the miss probability just above it."""
    k = np.arange(n)
    U = upper(k, n, delta)
    return U, binom.cdf(k, n, U)


def band(n, delta, U, P, lo, hi):
    """Over true rates in [lo, hi): the supremum of the miss (taken just above the limits that fall in the band, or at
    the band's lower end), its mean over a uniform grid of rates, and the share of the grid with a miss over delta."""
    k = np.arange(n + 1)
    Uall = wilson_upper(k, n, delta)
    ps = np.linspace(lo, hi, GRID, endpoint=False)
    kmax = np.searchsorted(Uall, ps, side="left") - 1                    # largest count whose limit is under p
    m = np.where(kmax >= 0, binom.cdf(np.maximum(kmax, 0), n, ps), 0.0)
    inside = (U >= lo) & (U < hi)
    return dict(lo=lo, hi=hi, sup=float(max(m[0], P[inside].max() if inside.any() else 0.0)), mean=float(m.mean()),
                share_over=float((m > delta).mean()))


def main():
    os.makedirs(OUT, exist_ok=True)
    # b1w at one stratum is the Wilson limit (the grid of b1w is 1 / 4000 of the distance to 1)
    gap = max(abs(SB.b1w(k, n, 1.0, d) - float(wilson_upper(k, n, d))) for n in (25, 100, 200, 400) for d in DELTAS for k in range(n))
    assert gap < 2.6e-4, gap
    rows, L = [], []
    for d in DELTAS:
        z2 = norm.ppf(1 - d) ** 2
        for n in NS:
            U, P = peaks(wilson_upper, n, d)
            Uc, Pc = peaks(cp_upper, n, d)
            assert Pc.max() <= d + 1e-9, (n, d, Pc.max())                # Clopper-Pearson never exceeds delta
            assert abs(U[0] - z2 / (n + z2)) < 1e-12 and abs(P[0] - (n / (n + z2)) ** n) < 1e-12
            over = P > d
            last = int(np.flatnonzero(over).max()) if over.any() else -1
            row = dict(delta=d, n=n, u0=float(U[0]), miss_at_u0=float(P[0]), worst=float(P.max()), worst_rate=float(U[int(P.argmax())]),
                       worst_count=int(P.argmax()), counts_over=int(over.sum()), highest_rate_over=float(U[last]) if last >= 0 else None,
                       cp_worst=float(Pc.max()),
                       bands=[band(n, d, U, P, lo, hi) for lo, hi in BANDS])
            rows.append(row)
    lim = {d: float(np.exp(-norm.ppf(1 - d) ** 2)) for d in DELTAS}
    json.dump(dict(b1w_gap=gap, limit_at_zero=lim, rows=rows), open(os.path.join(OUT, "wilson_exact.json"), "w"), indent=1)

    L = ["# The one-sided Wilson upper limit: exact miss probability", "",
         "`scripts/wilson_exact.py`. Enumeration over binomial counts; nothing is simulated. The miss probability at a true rate p "
         "is P(U(X) < p). It is largest just above a limit U(k), where it equals P(X <= k) at p = U(k); the table gives those "
         "suprema. `b1w` of spike 013 at one stratum equals this limit at every count for n in 25, 100, 200, 400 "
         f"(largest gap {gap:.5f}, the step of its grid).", "",
         f"At zero positives the limit is z^2 / (n + z^2) and the miss just above it is (n / (n + z^2))^n, which is close to "
         f"exp(-z^2) at any n and tends to it as n grows: {lim[0.05]:.4f} at delta 0.05 and {lim[0.1]:.4f} at delta 0.10. At the "
         "other end, just above the limit at n - 1 positives, the miss is 1 - U(n - 1)^n, the largest in the table: about 0.20 at "
         "delta 0.05 and 0.26 at delta 0.10.", "",
         "| delta | n | limit at 0 positives | miss just above it | largest miss, any rate | at rate | counts k with a miss over delta above U(k) | "
         "highest such rate | Clopper-Pearson, largest miss |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        hi = "-" if r["highest_rate_over"] is None else f"{r['highest_rate_over']:.3f}"
        L.append(f"| {r['delta']} | {r['n']} | {r['u0']:.4f} | {r['miss_at_u0']:.4f} | {r['worst']:.4f} | {r['worst_rate']:.4f} | "
                 f"{r['counts_over']} of {r['n']} | {hi} | {r['cp_worst']:.4f} |")
    L += ["", "## By band of the true rate", "",
          "For true rates in each band: the mean miss probability over a uniform grid of rates / its supremum / the share of "
          "the band where it is over delta. The mean shows the systematic part: under delta at rates below one half and over "
          "it above, where the limit is too short.", "",
          "| delta | n | " + " | ".join(f"{lo:g}-{hi:g}" for lo, hi in BANDS) + " |", "|---|---|" + "---|" * len(BANDS)]
    for r in rows:
        if r["n"] in (100, 200, 400):
            L.append(f"| {r['delta']} | {r['n']} | " + " | ".join(f"{b['mean']:.3f} / {b['sup']:.3f} / {100 * b['share_over']:.0f}%"
                                                                 for b in r["bands"]) + " |")
    open(os.path.join(OUT, "wilson_exact.md"), "w").write("\n".join(L) + "\n")
    print("-> results/paper/wilson_exact.md")


if __name__ == "__main__":
    main()
