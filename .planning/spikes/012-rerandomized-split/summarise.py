"""Tables for spike 012 from the sweep files.

    ../../../.venv/bin/python summarise.py wald_n5000_inf1.json ...

Columns: solution rate; miss = passed and truly violating; uncovered = true gap above the
safety-set upper bound (should be <= delta for a valid bound); optimism = mean of
(d_c - d_true) / se_s, how much better the candidate looks on D_c than in truth; leak =
mean (d_s - d_true) / se_s, the same on D_s (0 on fresh data); corr = correlation of the
two errors across seeds (0 for a random split of i.i.d. data); agree = predicted pass ==
safety-test pass; acc = true accuracy when a solution is returned; project CP = the
project's Clopper-Pearson safety test on the same candidate.
"""
import json
import sys

import numpy as np


def wilson_hi(k, n, z=1.96):
    p = k / n
    return (p + z * z / (2 * n) + z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)


def table(rows):
    splits = list(dict.fromkeys(r["split"] for r in rows))
    out = ["| split | solution | miss (95% hi) | uncovered (95% hi) | optimism | leak | corr "
           "| agree | acc if pass | CP solution | CP miss | tries |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for sp in splits:
        R = [r for r in rows if r["split"] == sp]
        n = len(R)
        miss = sum(r["miss"] for r in R)
        unc = sum(r["uncovered"] for r in R)
        ec = np.array([(r["d_c"] - r["d_true"]) / r["se_s"] for r in R if r["se_s"] > 0])
        es = np.array([(r["d_s"] - r["d_true"]) / r["se_s"] for r in R if r["se_s"] > 0])
        corr = np.corrcoef(ec, es)[0, 1]
        agree = np.mean([r["pred_pass"] == r["passed"] for r in R])
        acc = [r["acc_true"] for r in R if r["passed"]]
        out.append(
            f"| {sp} | {np.mean([r['passed'] for r in R]):.3f} "
            f"| {miss / n:.4f} ({wilson_hi(miss, n):.4f}) | {unc / n:.4f} ({wilson_hi(unc, n):.4f}) "
            f"| {ec.mean():+.3f} | {es.mean():+.3f} (se {es.std() / np.sqrt(len(es)):.3f}) "
            f"| {corr:+.3f} | {agree:.3f} | {np.mean(acc) if acc else float('nan'):.4f} "
            f"| {np.mean([r['passed_proj'] for r in R]):.3f} | {np.mean([r['miss_proj'] for r in R]):.4f} "
            f"| {np.mean([r.get('split_tries', 1) for r in R]):.1f} |")
    return "\n".join(out)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        rows = json.load(open(p))
        r0 = rows[0]
        print(f"\n### {p}: n = {r0['n']}, tau = {r0['tau']}, bound {r0['bound']}, "
              f"predicted-width inflation {r0['inflate']}, {len(rows) // len({r['split'] for r in rows})} seeds per arm\n")
        print(table(rows))
