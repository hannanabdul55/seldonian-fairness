"""Tables for stage 0a (inloop.json).

Per environment and rule: coverage failure of each bound against the pool truth and the
population truth, its mean width (UB - estimate), the solution rate it would give
(predicted feasible and UB <= threshold), and the ESS gain of the stratified b1w over the
random rule's pooled b1w. Wilson 95% upper limits in brackets.

    ../../../.venv/bin/python summarise_inloop.py inloop.json
"""
import json
import sys

import numpy as np


def hi(k, n, z=1.96):
    p = k / n
    return (p + z * z / (2 * n) + z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)


rows = json.load(open(sys.argv[1]))
print("| env | rule | ICC_cand | rho | bound | target | miss (95% hi) | width | solution | ESS vs random b1w |")
print("|---|---|---|---|---|---|---|---|---|---|")
for env in dict.fromkeys(r["env"] for r in rows):
    base = [r for r in rows if r["env"] == env and r["rule"] == "random"]
    w0 = np.mean([r["b1w_pooled"] - r["est_pool"] for r in base])
    for rule in dict.fromkeys(r["rule"] for r in rows):
        R = [r for r in rows if r["env"] == env and r["rule"] == rule]
        n = len(R)
        specs = [("ttest", "est_pool", "pop"), ("b1w_pooled", "est_pool", "pop")]
        if "b1w_strat_pop" in R[0]:
            specs += [("b1w_strat_pop", "est_strat", "pop"), ("b1w_strat_pool", "est_strat", "pool"),
                      ("b1_strat_pop", "est_strat", "pop")]
        for b, est, tgt in specs:
            truth = "truth_pop" if tgt == "pop" else "truth_pool"
            k = sum(r[truth] > r[b] for r in R)
            w = np.mean([r[b] - r[est] for r in R])
            sol = np.mean([r["predicted_feasible"] and r[b] <= r["threshold"] for r in R])
            ess = (w0 / w) ** 2 if b.startswith("b1w") else float("nan")
            print(f"| {env} | {rule} | {np.mean([r['icc_cand'] for r in R]):.2f} "
                  f"| {np.mean([r['rho'] for r in R]):.2f} | {b} | {tgt} | {k / n:.3f} ({hi(k, n):.3f}) "
                  f"| {w:.4f} | {sol:.3f} | {ess:.2f} |")
