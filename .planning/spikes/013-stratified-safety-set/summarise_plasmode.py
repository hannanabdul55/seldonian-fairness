"""Tables for the plasmode (bandit_plasmode.json or real_plasmode.json).

Groups rows by (env/pool, candidate); within a group the widths are averaged over seeds.
- validity: coverage failure per arm and bound (should be <= delta)
- gain: ESS = (width_R / width_arm)^2 for the same bound family at the same n_s, delta
- H3: realised ESS of S2 against the pre-flight 1 / (1 - G), over all S2 cells
- pass rate at tau = truth + Delta for R vs S2

    ../../../.venv/bin/python summarise_plasmode.py bandit_plasmode.json [--key env]
"""
import argparse
import json
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("--key", default="env")
ap.add_argument("--bound", default="b1w")
ap.add_argument("--delta", type=float, default=0.1)
a = ap.parse_args()
rows = [r for r in json.load(open(a.path))]
K = a.key


def agg(R):
    return dict(miss=np.mean([r["miss"] for r in R]), width=np.mean([r["width"] for r in R]),
                reps=sum(r["reps"] for r in R), G=np.mean([r["pf_G"] for r in R]),
                pass_=np.mean([r["pass_+0.02"] for r in R]), pass0=np.mean([r["pass_+0.00"] for r in R]))


groups = defaultdict(list)
for r in rows:
    groups[(r[K], r["cand"], r["arm"], r["k"], r["H"], r["n_s"], r["delta"], r["bound"])].append(r)
A = {g: agg(v) for g, v in groups.items()}
envs = sorted({g[0] for g in A})
cands = sorted({g[1] for g in A})

print(f"### Validity and gain at k = 8, H = 4, bound {a.bound}, delta {a.delta}\n")
print(f"| {K} | cand | n_s | arm | miss | width | ESS vs R | predicted ESS | pass at +0.02 |")
print("|---|---|---|---|---|---|---|---|---|")
for e in envs:
    for c in cands:
        for n_s in (100, 200, 400):
            base = A.get((e, c, "R", 8, 4, n_s, a.delta, a.bound))
            if base is None:
                continue
            for arm in ("R", "S1", "S2", "M", "P", "O"):
                x = A.get((e, c, arm, 8, 4, n_s, a.delta, a.bound))
                if x is None:
                    continue
                ess = (base["width"] / x["width"]) ** 2
                pred = 1 / (1 - x["G"]) if arm == "S2" else float("nan")
                print(f"| {e} | {c} | {n_s} | {arm} | {x['miss']:.3f} | {x['width']:.4f} | {ess:.2f} "
                      f"| {pred:.2f} | {x['pass_']:.3f} |")

print(f"\n### H3/H4: S2 realised vs predicted ESS over k x H x n_s (bound {a.bound}, delta {a.delta})\n")
real, pred, lab = [], [], []
tab = defaultdict(list)
for g, x in A.items():
    e, c, arm, k, H, n_s, d, b = g
    if arm != "S2" or d != a.delta or b != a.bound:
        continue
    base = A.get((e, c, "R", 8, 4, n_s, d, b))
    ess = (base["width"] / x["width"]) ** 2
    real.append(ess)
    pred.append(1 / (1 - x["G"]))
    tab[(e, c, k)].append((H, n_s, ess, 1 / (1 - x["G"])))
real, pred = np.array(real), np.array(pred)
err = np.abs(real - pred)
print(f"cells {len(real)}; median |realised - predicted| {np.median(err):.3f}; "
      f"90th pct {np.percentile(err, 90):.3f}; Spearman {spearmanr(real, pred)[0]:.3f}\n")
print(f"| {K} | cand | k | realised ESS by H (n_s 100 / 200 / 400) | predicted |")
print("|---|---|---|---|---|")
for (e, c, k), v in sorted(tab.items()):
    byH = defaultdict(dict)
    pr = {}
    for H, n_s, ess, p in v:
        byH[H][n_s] = ess
        pr[H] = p
    cells = "; ".join(f"H{H}: " + "/".join(f"{byH[H].get(n, float('nan')):.2f}" for n in (100, 200, 400))
                      for H in sorted(byH))
    preds = "; ".join(f"H{H}: {pr[H]:.2f}" for H in sorted(pr))
    print(f"| {e} | {c} | {k} | {cells} | {preds} |")
