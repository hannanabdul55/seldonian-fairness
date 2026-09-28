"""Pre-flight check: is a reference-rate stratified safety set worth it for this pool?

Input: a JSON-lines file with one row per prompt, ``{"labels": [0, 1, ...]}``: k judged
reference samples of the constraint label (k >= 2, ideally 8), taken before any split.

Output: the label's rate, ICC_ref, the covariate's reliability, the predicted ESS gain of
8 equal strata (random tie-breaking) for rho = 1 and for rho = 0.8, with the two-phase cap
when the certificate targets the population the pool was drawn from, and a verdict against
spike 013's use-case map. The formula over-predicts by about 0-20% at H = 8 on real data
(spike 013 stage 3). Treat the result as an upper estimate and a ranking.

    ../../../.venv/bin/python preflight.py ref_labels.jsonl --n-s 200
"""
import argparse
import json

import numpy as np

from plasmode import C_H, icc_from_samples, reliability


def assess(Y, n_s, H=8, rho=1.0):
    Y = np.asarray(Y, dtype=float)
    N, k = Y.shape
    rate = float(Y.mean())
    icc = icc_from_samples(Y)
    rel = reliability(icc, k)
    G = icc * rho ** 2 * rel * C_H[H]
    return dict(rate=rate, icc_ref=icc, rel_k=rel, G=G,
                ess_pool=1 / (1 - G), ess_population=1 / (1 - G + G * n_s / N))


def verdict(r):
    if not 0.05 <= r["rate"] <= 0.95:
        return ("NO: rate outside 5-95%; approximate bounds miss far more than delta here "
                "(013: up to 0.45 at 1% rates), with or without strata")
    if r["ess_pool"] < 1.2:
        return "NO: predicted gain under 1.2x; not worth the reference samples"
    return "YES: stratify on the reference rate (8 equal strata, random ties), use b1w"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--n-s", type=int, default=200, help="planned safety-set size")
    a = ap.parse_args()
    Y = [json.loads(l)["labels"] for l in open(a.path)]
    for rho in (1.0, 0.8):
        r = assess(Y, a.n_s, rho=rho)
        print(f"rho {rho}: " + ", ".join(f"{k} {v:.3f}" for k, v in r.items()))
    print(verdict(assess(Y, a.n_s)))


if __name__ == "__main__":
    main()
