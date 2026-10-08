"""The labels-alone rows of the paper's Table 4, by enumeration (plan step R4, item 3).

The training paper's section 6.2 resampled 0/1 harm labels from a pool of 3,600 responses at a
rate of 0.034 and printed each bound's miss rate at n = 200 to 2,400 (5,000 draws). The pool's
labels were lost on 2026-09-19. They are not needed: n labels drawn with replacement from a 0/1
pool are a Binomial(n, pool rate) count, and a bound on 0/1 labels depends on the sample only
through that count. So the miss probability of every bound is a finite sum,

    miss(p) = sum_k Binom(k; n, p) [ U(k, n) < p ],

which this script computes with the bounds of ``seldonian.llm.policy.BOUNDS``, the functions
``scripts/resample_calibration.py`` called.

1. The printed table against the enumeration. The pool's count is known only through its
   printed rate, so each count from 121 to 124 of 3,600 is tried; the printed rate is compared
   with the exact one in units of its Monte Carlo standard error at 5,000 draws.
2. The same bounds over a grid of rates from 0.5% to 20%, at delta 0.05 and 0.10: the largest
   miss probability of each bound at each n, and where it occurs.

CPU only, about a minute.

    .venv/bin/python scripts/binomial_rows.py
    -> results/paper/binomial_rows.md, results/paper/binomial_rows.json
"""
import json
import os
import sys

sys.dont_write_bytecode = True

import numpy as np
from scipy.stats import binom

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
from seldonian.llm.policy import BOUNDS  # noqa: E402

OUT = os.path.join(ROOT, "results", "paper", "binomial_rows")
NS = (200, 400, 800, 1200, 2400)
DELTAS = (0.05, 0.10)
NAMES = (("ttest", "Student-t"), ("clopper_pearson", "Clopper-Pearson"), ("bentkus", "Bentkus"),
         ("betting_mixture", "betting mixture"), ("hoeffding", "Hoeffding"), ("anderson", "Anderson"))
POOL_N, DRAWS = 3600, 5000
POOL_COUNTS = (121, 122, 123, 124)           # the counts that print as a rate of 0.034
GRID = np.round(np.arange(0.005, 0.2001, 0.0025), 4)
# reports/paper_seldonian_llm.md, section 6.2, GRPO pool, delta 0.1 (miss rate; n in the order of NS)
PRINTED = {"ttest": (0.184, 0.115, 0.121, 0.101, 0.101), "clopper_pearson": (0.084, 0.067, 0.083, 0.074, 0.084),
           "bentkus": (0.029, 0.035, 0.034, 0.023, 0.032), "betting_mixture": (0.007, 0.016, 0.011, 0.009, 0.010),
           "hoeffding": (0.000,) * 5, "anderson": (0.000,) * 5}


def uppers(bound, n, delta, kmax):
    """U(k, n) for k = 0..kmax."""
    fn = BOUNDS[bound]
    out = np.empty(kmax + 1)
    for k in range(kmax + 1):
        x = np.concatenate([np.ones(k), np.zeros(n - k)])
        out[k] = float(fn(x, delta).upper)
    return out


def miss(U, n, p):
    k = np.arange(len(U))
    tail = binom.sf(len(U) - 1, n, p)          # mass beyond the counts evaluated: counted as no miss
    assert tail < 1e-9, (n, p, tail)
    return float(binom.pmf(k, n, p)[U < p].sum())


def main():
    res = dict(ns=NS, deltas=DELTAS, pool_counts=POOL_COUNTS, grid=[float(GRID[0]), float(GRID[-1]), 0.0025],
               reproduce={}, grid_max={})
    table = {}
    for bound, _ in NAMES:
        for n in NS:
            kmax = int(binom.isf(1e-12, n, GRID[-1])) + 1
            for d in DELTAS:
                table[(bound, n, d)] = uppers(bound, n, d, min(kmax, n))
        print(bound, "done", flush=True)
    # 1. the printed table
    for c in POOL_COUNTS:
        p = c / POOL_N
        rows, zs = {}, []            # the printed table has one row for Hoeffding and Anderson: 25 values
        for bound, _ in NAMES:
            ex = [miss(table[(bound, n, 0.10)], n, p) for n in NS]
            z = [(pr - e) / np.sqrt(max(e * (1 - e), 1e-12) / DRAWS) if e > 1e-9 else (0.0 if pr == 0 else float("inf"))
                 for pr, e in zip(PRINTED[bound], ex)]
            rows[bound] = dict(exact=ex, printed=list(PRINTED[bound]), z=z)
            if bound != "anderson":
                zs += [abs(v) for v in z]
        res["reproduce"][str(c)] = dict(rate=p, rows=rows, max_abs_z=float(max(zs)), within_2=int(sum(v <= 2 for v in zs)),
                                        cells=len(zs), sum_z2=float(sum(v * v for v in zs)))
    # 2. the grid of rates
    for bound, _ in NAMES:
        for d in DELTAS:
            for n in NS:
                ms = np.array([miss(table[(bound, n, d)], n, p) for p in GRID])
                i = int(ms.argmax())
                res["grid_max"][f"{bound}|{d}|{n}"] = dict(max=float(ms[i]), at=float(GRID[i]), share_over=float((ms > d + 1e-12).mean()),
                                                         mean=float(ms.mean()))
    json.dump(res, open(OUT + ".json", "w"), indent=1)
    report(res)


def report(res):
    best = min(res["reproduce"], key=lambda c: res["reproduce"][c]["sum_z2"])
    b = res["reproduce"][best]
    L = ["# The labels-alone rows of Table 4, by enumeration", "",
         "`scripts/binomial_rows.py`. Labels drawn with replacement from a 0/1 pool are a binomial count, so each bound's miss "
         "probability is a finite sum over counts. Nothing is simulated and the lost pool is not needed, only its rate.", "",
         "## 1. The training paper's printed table (delta 0.10, 5,000 draws) against the exact values", "",
         f"The pool held {POOL_N:,} labels at a printed rate of 0.034, which is a count of 121 to 124. The count is not "
         "recorded, so it is chosen here by its fit to the 25 printed values. Agreement by count "
         "(z = printed minus exact, in Monte Carlo standard errors of a 5,000-draw estimate):", "",
         "| pool count | rate | cells within 2 se | largest abs z | sum of z squared |", "|---|---|---|---|---|"]
    for c, r in res["reproduce"].items():
        L.append(f"| {c} | {r['rate']:.5f} | {r['within_2']} of {r['cells']} | {r['max_abs_z']:.2f} | {r['sum_z2']:.1f} |")
    L += ["", f"At the best-fitting count, {best} of {POOL_N:,}:", "",
          "| bound | " + " | ".join(f"n {n}: printed / exact" for n in NS) + " |", "|---|" + "---|" * len(NS)]
    for bound, name in NAMES:
        r = b["rows"][bound]
        L.append(f"| {name} | " + " | ".join(f"{p:.3f} / {e:.4f}" for p, e in zip(r["printed"], r["exact"])) + " |")
    L += ["", "## 2. Largest miss probability over rates of 0.5% to 20% (steps of 0.25 points)", "",
          "Each cell: the largest exact miss, the rate at which it occurs, and the share of the grid's rates at which the miss "
          "exceeds delta.", ""]
    for d in DELTAS:
        L += [f"**delta {d}**", "", "| bound | " + " | ".join(f"n {n}" for n in NS) + " |", "|---|" + "---|" * len(NS)]
        for bound, name in NAMES:
            cells = []
            for n in NS:
                g = res["grid_max"][f"{bound}|{d}|{n}"]
                cells.append(f"{g['max']:.3f} at {100 * g['at']:.2f}% ({100 * g['share_over']:.0f}%)")
            L.append(f"| {name} | " + " | ".join(cells) + " |")
        L.append("")
    open(OUT + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    if "--render" in sys.argv:
        report(json.load(open(OUT + ".json")))
    else:
        main()
