"""Spike 017: the routing rule between the exact bound and the judge-assisted one.

The bootstrap-t PPI++ bound holds its level but returns 1 when the labels hold too few
positives, and Clopper-Pearson on the labels is exact but ignores the judge. A certificate
must pick one *by a rule fixed in advance*. This script scores three rules on the
prevalence-shifted plasmode (015's refusal judge, rate dialled from 0.2 to 0.013):

- ``k >= k0``   PPI++ (logit feature, bootstrap-t) when the labels hold at least ``k0``
                positives and ``k0`` negatives, else Clopper-Pearson;
- ``split``     the smaller of the two, each at ``delta / 2`` (valid by the union bound);
- ``min``       the smaller of the two at full ``delta`` (not valid on paper; the ceiling).

    OMP_NUM_THREADS=1 ../../../.venv/bin/python route017.py      # ~4 min; writes route.md
"""
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import cert017 as c
import plasmode017 as pm

DELTA, REPS, BIG_N = 0.05, 3000, 4000
K0 = (5, 10, 20, 30)


def cell(job):
    variant, rate, n, seed = job
    y_pool, p_pool = pm.load_pools()[("refusal", variant, 0)]
    rng = np.random.default_rng(seed)
    y, p = pm.sample(rng, y_pool, p_pool, REPS, BIG_N, rate)
    yl = y[:, :n]
    fl, fu = pm.logit01(p[:, :n]), pm.logit01(p[:, n:])
    k = yl.sum(1)
    out = dict(variant=variant, rate=rate, n=n, k_mean=float(k.mean()))
    cl = c.classical(yl, DELTA)
    bt = c.ppipp_boot(yl, fl, fu, DELTA, seed=seed)
    cl2 = c.classical(yl, DELTA / 2)
    bt2 = c.ppipp_boot(yl, fl, fu, DELTA / 2, seed=seed + 1)
    rules = {"classical": cl, "boot": bt, "split": np.minimum(cl2, bt2), "min": np.minimum(cl, bt)}
    for k0 in K0:
        rules[f"k>={k0}"] = np.where((k >= k0) & (n - k >= k0), bt, cl)
    for name, u in rules.items():
        out[name] = (float((u < rate).mean()), float(u.mean()))
    return out


def main():
    jobs = [(v, r, n, 100 + i) for i, (v, r, n) in enumerate(
        (v, r, n) for v in ("rubric", "raw") for r in (0.013, 0.02, 0.05, 0.1, 0.2) for n in (225, 1000))]
    with ProcessPoolExecutor(6) as ex:
        rows = list(ex.map(cell, jobs))
    names = ["classical", "boot"] + [f"k>={k}" for k in K0] + ["split", "min"]
    L = ["# Spike 017: routing between Clopper-Pearson and bootstrap-t PPI++", "",
         f"015's refusal judge, logit feature, prevalence-shifted; {BIG_N} judged, n labelled, "
         f"{REPS} draws, delta {DELTA}. Miss (mean bound).", "",
         "| judge | rate | n | mean positives | " + " | ".join(names) + " |",
         "|---|---|---|---|" + "---|" * len(names)]
    for r in rows:
        L.append(f"| {r['variant']} | {r['rate']} | {r['n']} | {r['k_mean']:.1f} | "
                 + " | ".join(f"{r[k][0]:.3f} ({r[k][1]:.4f})" for k in names) + " |")
    L += ["", "| rule | largest miss over the 20 cells | cells above 0.05 |", "|---|---|---|"]
    for k in names:
        m = [r[k][0] for r in rows]
        L.append(f"| {k} | {max(m):.3f} | {sum(x > 0.05 for x in m)} |")
    open(os.path.join(c.HERE, "route.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
