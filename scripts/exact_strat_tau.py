"""B5 sensitivity, NOT registered: where the threshold sits within Clopper-Pearson's step.

``exact_strat.py`` (registered) sets tau_q to the smallest threshold at which CP on a random draw of n_s labels
certifies with probability at least q. That threshold is always one of CP's own limits, U(k*), so CP is scored at
the edge of its step and any arm whose limits fall on another lattice (cp_strat, shifted by its allocation slack;
the pooled Wilson bound; b1 and b1w) is scored at an arbitrary point of its own. Here the threshold is spread
evenly over CP's step [U(k*), U(k*+1)), where CP's pass probability does not change, and each arm's power ESS is
averaged over J thresholds. Same cells, draws (seeds from the cell's name), arms and power ESS as the registered
run, except b2 (its limit is far above in every cell). Written after the registered run's results were seen.

    OMP_NUM_THREADS=1 .venv/bin/python scripts/exact_strat_tau.py --workers 14
    -> results/paper/exact_strat_tau.json, results/paper/exact_strat_tau.md
"""
import argparse
import json
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import binom

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exact_strat as EX     # noqa: E402

J = 8
D = 0.05
ARMS = ("pooled Wilson", "cp_strat", "b1", "b1w")


def job(cell):
    t0 = time.time()
    P = EX.params(cell)
    theta, W, n_h, n_s, truth = P["theta"], P["W"], P["n_h"], P["n_s"], P["truth"]
    n = int(n_h.sum())
    slack = float(np.maximum(W - n_h / n, 0.0).sum())
    pmf_s = EX.pb_pmf(n_h, theta)
    pmf_r = binom.pmf(np.arange(n_s + 1), n_s, truth)
    rng = np.random.default_rng(zlib.crc32(EX.key(cell).encode()))
    S = rng.binomial(n_h[None, :], theta[None, :], size=(EX.REPS, len(n_h)))
    U = EX.cp_upper(np.arange(n_s + 1), n_s, D)
    ub = {"pooled Wilson": (pmf_r, EX.wilson_counts(n_s, D)),
          "cp_strat": (pmf_s, np.minimum(EX.cp_upper(np.arange(n + 1), n, D) + slack, 1.0))}
    for arm, fn in (("b1", EX.SB.b1), ("b1w", EX.SB.b1w)):
        memo, v = {}, np.empty(EX.REPS)
        for r in range(EX.REPS):
            k = tuple(S[r])
            if k not in memo:
                memo[k] = fn(S[r], n_h, W, D)
            v[r] = memo[k]
        ub[arm] = (None, v)
    out = dict(group=cell[0], cell=EX.key(cell), truth=truth, n_s=n_s, n=n, slack=slack, ess={})
    for q in EX.QS:
        t0q = EX.tau_q(n_s, truth, D, q)
        k = int(np.searchsorted(U, t0q - 1e-12))
        t1 = float(U[k + 1]) if k + 1 <= n_s else 1.0
        taus = [t0q + (j + 0.5) / J * (t1 - t0q) for j in range(J)]
        cp_pp = float(binom.cdf(k, n_s, truth))
        acc = {a: [] for a in ARMS}
        for t in taus:
            ref = EX.power_n(truth, D, t, cp_pp, n_s) / n_s
            for a in ARMS:
                pmf, v = ub[a]
                pp = float(pmf[v <= t + 1e-12].sum()) if pmf is not None else float(np.mean(v <= t + 1e-12))
                labels = n if a in EX.STRAT_ARMS else n_s
                acc[a].append(EX.power_n(truth, D, t, pp, n_s) / labels / ref)
        out["ess"][str(q)] = {a: dict(mean=float(np.nanmean(x)), at_tau=None) for a, x in acc.items()}
    print(f"{EX.key(cell)} {time.time() - t0:.0f}s", flush=True)
    return out


def report(rows, reg, path):
    regd = {(r["cell"], r["arm"]): r for r in reg["rows"] if r["delta"] == D}
    L = ["# B5 sensitivity (not registered): the threshold spread over Clopper-Pearson's step",
         "",
         "`scripts/exact_strat_tau.py`. Delta 0.05. Each arm's power ESS is averaged over 8 thresholds spread evenly over "
         "CP's step [U(k*), U(k*+1)) above the registered tau_q, where CP's own pass probability is constant. The registered "
         "value (threshold at the step's lower edge) is beside it. Written after the registered results were seen.",
         "",
         "| group | rates | q | arm | cells | registered: median (min-max) | spread over the step: median (min-max) |",
         "|---|---|---|---|---|---|---|"]
    for g in ("real", "costly", "grid"):
        for cls, f in (("mid (5-95%)", lambda t: EX.MIDRANGE[0] <= t <= EX.MIDRANGE[1]), ("rare (<5%)", lambda t: t < EX.RARE)):
            sel = [r for r in rows if r["group"] == g and f(r["truth"])]
            if not sel:
                continue
            for q in ("0.5", "0.8"):
                for a in ARMS:
                    new = np.array([r["ess"][q][a]["mean"] for r in sel])
                    old = np.array([regd[(r["cell"], a)]["power_ess"][q] for r in sel])
                    L.append(f"| {g} | {cls} | {q} | {a} | {len(sel)} | {np.nanmedian(old):.2f} ({np.nanmin(old):.2f}-{np.nanmax(old):.2f}) | "
                             f"{np.nanmedian(new):.2f} ({np.nanmin(new):.2f}-{np.nanmax(new):.2f}) |")
    sel = [r for r in rows if r["group"] in ("real", "costly") and EX.MIDRANGE[0] <= r["truth"] <= EX.MIDRANGE[1]]
    b1_over = sum(r["ess"]["0.8"]["b1"]["mean"] > r["ess"]["0.8"]["cp_strat"]["mean"] for r in sel if r["group"] == "real")
    nreal = sum(r["group"] == "real" for r in sel)
    cs8 = np.array([r["ess"]["0.8"]["cp_strat"]["mean"] for r in sel])
    cs5 = np.array([r["ess"]["0.5"]["cp_strat"]["mean"] for r in sel])
    L += ["", "## The registered statements re-read on the spread thresholds (not a test)", "",
          f"- cp_strat, real and costly cells in range ({len(sel)}), q 0.8: min {np.nanmin(cs8):.2f}, median {np.nanmedian(cs8):.2f}"
          f" (X4 asked for >= 0.95 and >= 1.10).",
          f"- the same at q 0.5: {np.nanmin(cs5):.2f}-{np.nanmax(cs5):.2f} (X5 asked for [0.85, 1.15]).",
          f"- b1 above cp_strat at q 0.8 in {b1_over} of {nreal} real cells in range (X8 asked for all)."]
    open(path, "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()
    path = os.path.join(EX.OUT, "exact_strat_tau.json")
    if not os.path.exists(path):
        t0 = time.time()
        rows = []
        with ProcessPoolExecutor(a.workers) as ex:
            for i, r in enumerate(ex.map(job, EX.cells())):
                rows.append(r)
                print(f"{i + 1} {time.time() - t0:.0f}s", flush=True)
        json.dump(dict(J=J, delta=D, rows=rows), open(path, "w"), default=float)
    report(json.load(open(path))["rows"], json.load(open(os.path.join(EX.OUT, "exact_strat.json"))),
           os.path.join(EX.OUT, "exact_strat_tau.md"))
    print("-> results/paper/exact_strat_tau.md")


if __name__ == "__main__":
    main()
