"""Spike 017: derive, then test. Identities and coverage of cert017's routes on known truth.

1. ``cp_upper``/``cp_lower`` against the project's ``clopper_pearson_bounds``.
2. Miss rate of every route on a parametric judge (``Y ~ Bern(r)``, ``f | Y`` Bernoulli with
   the given recall and false-alarm rate), where the truth is ``r`` exactly.
3. PPI++'s gain formula ``1 / (1 - rho^2 Nu / (Nu + n))`` against the Monte-Carlo variance
   ratio, and plain PPI's ``var(Y) / (var(Y - f) + var(f) n / Nu)``.
4. ``strat_exact`` when labels are drawn by judge stratum instead of at random.

    ../../../.venv/bin/python check_cert.py          # about 1 min, writes check_cert.md
"""
import os

import numpy as np

import cert017 as c
from seldonian.bounds import clopper_pearson_bounds

HERE = os.path.dirname(os.path.abspath(__file__))
DELTA, N, REPS = 0.05, 2000, 20000


def draw(rng, r, sens, fa, reps, n_all):
    y = (rng.random((reps, n_all)) < r).astype(float)
    f = (rng.random((reps, n_all)) < np.where(y == 1, sens, fa)).astype(float)
    return y, f


def main():
    rng = np.random.default_rng(17)
    L = ["# Spike 017: checks on known truth", ""]

    # 1
    worst = 0.0
    for _ in range(300):
        n = int(rng.integers(1, 400))
        k = int(rng.integers(0, n + 1))
        d = float(rng.choice([0.05, 0.0125, 0.1]))
        x = np.r_[np.ones(k), np.zeros(n - k)]
        rv = clopper_pearson_bounds(x, d)
        worst = max(worst, abs(rv.upper - c.cp_upper(k, n, d)), abs(rv.lower - c.cp_lower(k, n, d)))
    L += [f"1. Clopper-Pearson vs the project function, 300 random (k, n, delta): "
          f"largest difference {worst:.1e}.", ""]

    # 2 and 3
    L += ["2. Miss rate (target <= 0.05) and mean bound, N = 2000 judged, n labelled at random, "
          f"{REPS} draws. `gain` = var(labels-alone estimate) / var(estimate).", "",
          "`block` = finite-sample block PPI with lam from an independent labelled sample of the "
          "same size (2000 draws).", "",
          "| rate | sens | FA | n | rho^2 | classical | naive | youden | ppi | ppi++ | ppi++ wilson "
          "| exact3 | strat | block | ppi gain (formula / MC) | ppi++ gain (formula / MC) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    fmt = lambda u, truth: f"{(u < truth).mean():.3f} ({u.mean():.3f})"  # noqa: E731
    for r in (0.2, 0.05, 0.013):
        for sens, fa in ((0.9, 0.05), (0.67, 0.216), (0.5, 0.5)):
            for n in (100, 225):
                y, f = draw(rng, r, sens, fa, REPS, N)
                yl, fl, fu = y[:, :n], f[:, :n], f[:, n:]
                yp, fp_ = draw(rng, r, sens, fa, 2000, n)              # pilot for lam
                lam = np.clip(c.lam_hat(yp, fp_, n / (N - n)), 0, 1)
                blk = c.ppi_block(yl[:2000], fl[:2000], fu[:2000], DELTA, lam)
                e1, _, _ = c.ppi_point(yl, fl, fu, 1.0)
                e2, _, _ = c.ppi_point(yl, fl, fu, None)
                v0 = yl.mean(1).var()
                r2 = c.rho2_binary(r, sens, fa)
                q = sens * r + fa * (1 - r)
                var_d = r * (1 - sens) + (1 - r) * fa - (r - q) ** 2      # var(Y - f)
                g1 = r * (1 - r) / (var_d + q * (1 - q) * n / (N - n))
                L.append(
                    f"| {r} | {sens} | {fa} | {n} | {r2:.3f} | {fmt(c.classical(yl, DELTA), r)} "
                    f"| {fmt(c.naive(f, DELTA), r)} | {fmt(c.youden(yl, fl, f, DELTA), r)} "
                    f"| {fmt(c.ppi_clt(yl, fl, fu, DELTA), r)} | {fmt(c.ppipp_clt(yl, fl, fu, DELTA), r)} "
                    f"| {fmt(c.ppipp_wilson(yl, fl, fu, DELTA), r)} "
                    f"| {fmt(c.ppi_exact3(yl, fl, f, DELTA), r)} | {fmt(c.strat_exact(yl, fl, f, DELTA), r)} "
                    f"| {fmt(blk, r)} "
                    f"| {g1:.2f} / {v0 / e1.var():.2f} "
                    f"| {c.gain_ppipp(r2, n, N - n):.2f} / {v0 / e2.var():.2f} |")
    L.append("")

    # 4: labels by judge stratum (half to flagged), truth r
    L += ["4. `strat_exact` with labels drawn by judge stratum (n = 225: up to 112 flagged, the "
          "rest cleared), 20000 draws.", "", "| rate | sens | FA | miss | mean bound | "
          "classical at random, mean bound |", "|---|---|---|---|---|---|"]
    for r, sens, fa in ((0.2, 0.9, 0.05), (0.013, 0.67, 0.216), (0.013, 0.8, 0.015), (0.05, 0.8, 0.015)):
        y, f = draw(rng, r, sens, fa, REPS, N)
        u = np.empty(REPS)
        n = 225
        for i in range(REPS):
            fl1 = np.flatnonzero(f[i] == 1)[:n // 2]
            fl0 = np.flatnonzero(f[i] == 0)[:n - len(fl1)]
            idx = np.r_[fl1, fl0]
            u[i] = c.strat_exact(y[i, idx], f[i, idx], f[i], DELTA)[0]
        L.append(f"| {r} | {sens} | {fa} | {(u < r).mean():.3f} | {u.mean():.4f} "
                 f"| {c.classical(y[:, :n], DELTA).mean():.4f} |")
    open(os.path.join(HERE, "check_cert.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
