"""Pre-flight check: is a reference-rate stratified safety set worth it for this pool?

Input: a JSON-lines file with one row per prompt, ``{"labels": [0, 1, ...]}``: k judged
reference samples of the constraint label (k >= 2, ideally 8), taken before any split.

Output: the label's rate, ICC_ref, the covariate's reliability, the predicted ESS gain of
8 equal strata (random tie-breaking) for rho = 1 and for rho = 0.8, with the two-phase cap
when the certificate targets the population the pool was drawn from, and a verdict against
spike 013's use-case map. The formula over-predicts by about 0-20% at H = 8 on real data
(spike 013 stage 3). Treat the result as an upper estimate and a ranking.

    ../../../.venv/bin/python preflight.py ref_labels.jsonl --n-s 200 [--pushed] [--source]

The verdict follows the rule of the certification paper (section 8.1, after the confirmation
pools R5 and R5b): no strata and an exact bound on a random draw for a rare or near-one label;
no strata above about 80% (the gain under the Wald-t limit was 0.75 to 1.13 at 82% and 0.4 to
0.65 at 93%); otherwise strata with the stratified Wald-t limit (``stratbounds.b1``), with
the sampled-pool term (``N=``) when ``--source`` says the certificate is about the population
the pool was drawn from. ``b1w`` is narrower, but once the pool is much larger than the safety
set it kept its level only at rates under one half. The Wald-t limit has no finite-sample
guarantee: on synthetic i.i.d. data it was over its level in 10 of 36 cells.

``--pushed``: the constraint label is one the training targets directly (a Lagrangian
multiplier against a reward that pulls the other way). Spike 014 found no compression of
the per-prompt rates under that push (ICC_cand / ICC_ref 1.0-1.5 in the bandit, 1.01 on
Granite over-refusal) and a persistence rho set by ICC_ref rather than by how hard the label
is pushed: 0.69 at ICC_ref 0.26, 0.87 at 0.51, 0.98 at 0.75 in the bandit, 0.92 on Granite
at 0.72. The pushed verdict uses rho interpolated from that table at ICC_ref, keeps
ICC_cand = ICC_ref, and is an upper estimate as the rest is: on the one real point it
predicted 2.55 against a realised 2.21 (the formula at the measured rho 0.92 gave 2.26).
"""
import argparse
import json

import numpy as np

from plasmode import C_H, icc_from_samples, reliability

PUSHED_RHO = ((0.26, 0.69), (0.51, 0.87), (0.75, 0.98))   # spike 014 bandit: (ICC_ref, rho)


def pushed_rho(icc):
    """rho under direct pressure on the label, by ICC_ref (spike 014, bandit; clipped to the table)."""
    xs, ys = zip(*PUSHED_RHO)
    return float(np.interp(icc, xs, ys))


def assess(Y, n_s, H=8, rho=1.0):
    Y = np.asarray(Y, dtype=float)
    N, k = Y.shape
    rate = float(Y.mean())
    icc = icc_from_samples(Y)
    rel = reliability(icc, k)
    G = icc * rho ** 2 * rel * C_H[H]
    return dict(rate=rate, icc_ref=icc, rel_k=rel, rho=rho, G=G,
                ess_pool=1 / (1 - G), ess_population=1 / (1 - G + G * n_s / N))


def verdict(r, pushed=False, source=False):
    if not 0.05 <= r["rate"] <= 0.95:
        return ("NO: rate outside 5-95%; no strata, Clopper-Pearson or a betting bound on a random "
                "draw (approximate bounds miss far more than delta here, 013: up to 0.45 at 1% rates)")
    if r["rate"] > 0.80:
        return ("NO: rate above 80%; under the Wald-t limit the strata gained nothing or lost "
                "(0.75-1.13 at 82%, 0.4-0.65 at 93%), so draw the safety set at random")
    if r["ess_pool"] < 1.2:
        return "NO: predicted gain under 1.2x; not worth the reference samples"
    bound = ("the stratified Wald-t limit (b1) with the sampled-pool term (N = pool size)" if source
             else "the stratified Wald-t limit (b1)")
    out = (f"YES: stratify on the reference rate (8 equal strata, random ties), use {bound}; "
           "it has no finite-sample guarantee (over in 10 of 36 synthetic cells)")
    if r["rate"] < 0.5 and not source:
        out += "; b1w is narrower and, with a pool much larger than the safety set, kept its level in our tests only at rates under one half"
    if pushed:
        out += f"; the label is pushed, so rho {r['rho']:.2f} from spike 014's table (one real point ran 13% under it)"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--n-s", type=int, default=200, help="planned safety-set size")
    ap.add_argument("--pushed", action="store_true",
                    help="the training targets this label directly (spike 014's rho table)")
    ap.add_argument("--source", action="store_true",
                    help="the certificate is about the prompt source the pool was drawn from")
    a = ap.parse_args()
    Y = [json.loads(l)["labels"] for l in open(a.path)]
    for rho in (1.0, 0.8):
        r = assess(Y, a.n_s, rho=rho)
        print(f"rho {rho}: " + ", ".join(f"{k} {v:.3f}" for k, v in r.items()))
    if a.pushed:
        r = assess(Y, a.n_s, rho=pushed_rho(assess(Y, a.n_s)["icc_ref"]))
        print("pushed: " + ", ".join(f"{k} {v:.3f}" for k, v in r.items()))
        print(verdict(r, pushed=True, source=a.source))
    else:
        print(verdict(assess(Y, a.n_s), source=a.source))


if __name__ == "__main__":
    main()
