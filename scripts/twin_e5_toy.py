"""Pre-E5 check: is the defensive-mixture upper bound valid on a toy where the truth is exact?

Proposal `reports/proposal_adversarial_harm_twin.md` section 6; predictions E5a to E5c in
`.planning/harm-twin/REGISTRATION-E1-E5.md`. E5a is the validity claim, and the proposal says
that if it fails "the construction is wrong and the rest of section 6 is void". The registered
E5 plasmode resamples the real brute-force pools, which do not exist yet. This script asks the
same question one level earlier, on a toy whose distributions are enumerable, in the spirit of
E0: a check of the estimator and of the two bounds as this repository implements them, run
before any GPU time. It is NOT the registered E5 plasmode and it changes no registered number.

The construction under test. For a prompt x, the main model p, a twin q and alpha in (0, 1),

    m(y | x) = alpha p(y | x) + (1 - alpha) q(y | x),

sampled by taking p with probability alpha and q otherwise. On a harmful response the weight is
w(y) = p(y | x) / m(y | x), at most 1 / alpha because m >= alpha p pointwise, and

    P_p(H) = E_{x ~ D, y ~ m}[ w(y) 1_H(y) ].

So the summand lies in [0, 1 / alpha] whatever the twin is, and a distribution-free upper bound
on its mean is an upper bound on the breach rate. The claim the toy tests is that validity does
not depend on the twin: a twin that drops harmful modes raises the variance but not the bias,
because the alpha p component still covers every mode.

Twin arms, two of them far worse than anything E1 could produce:

  perfect   q = p_H exactly                   forward KL(p_H || q) = 0
  narrow    p_H truncated to its top 5 modes  forward KL infinite (mass dropped outright)
  useless   q = p                             the twin adds nothing
  misaimed  q supported off H                 the twin never writes a harmful response

Reported per cell: the estimator's mean against the exact truth with that cell's own standard
error (unbiasedness), the estimator's standard deviation against plain Monte Carlo on p (the
variance claim of section 6), the miss rate of each upper bound (the fraction of draws whose
bound falls below the truth, which must be at or below delta), and the mean bound against
Clopper-Pearson on the same number of plain samples from p (the width claim).

A second, exploratory scan outside the registered grid raises n for the perfect twin, to locate
the sample size at which the weighted bound overtakes Clopper-Pearson, if it does.

Two notes on the bounds as implemented in `seldonian/bounds.py`. The empirical-Bernstein form is
Maurer and Pontil's with ln(2 / delta), so used one-sided it is conservative by that factor; it
is left as the repository has it, since a conservative bound cannot make a validity check pass
that would otherwise fail. The betting bound depends on the order of the samples; the draws here
are i.i.d. and unpermuted, which is how E5 would use it.

Usage:  .venv/bin/python -I scripts/twin_e5_toy.py [--reps 1000] [--out results/twin/e5_toy]
"""
import argparse
import json
import os
import time

import numpy as np

from seldonian.bounds import (betting_bounds, clopper_pearson_bounds,
                              empirical_bernstein_bounds)

DELTA = 0.1
ARMS = ("perfect", "narrow", "useless", "misaimed")
NS = (300, 1000)                       # the registered grid
ALPHAS = (0.2, 0.5)                    # the registered grid
NS_SCAN = (300, 1000, 3000, 10000)     # exploratory, outside the registered grid


# ---------------------------------------------------------------------------- the toy

def build_toy(n_prompts=20, K=400, h=40, ph_lo=0.005, ph_hi=0.05, seed=0):
    """
    ``n_prompts`` prompts over ``K`` enumerable responses, the first ``h`` of them harmful.
    Each prompt's p is a skewed random distribution rescaled so that its mass on H is a
    target drawn log-uniformly in ``[ph_lo, ph_hi]``, the range E1 can resolve. Returns
    ``(P, Hmask, truth_per_prompt)``.
    """
    rng = np.random.default_rng(seed)
    Hmask = np.zeros(K, dtype=bool)
    Hmask[:h] = True
    targets = np.exp(rng.uniform(np.log(ph_lo), np.log(ph_hi), size=n_prompts))
    P = np.zeros((n_prompts, K))
    for i in range(n_prompts):
        w = rng.random(K) ** 3 + 1e-9        # heavily skewed, every response reachable
        w /= w.sum()
        mH = w[Hmask].sum()
        t = targets[i]
        w[Hmask] *= t / mH
        w[~Hmask] *= (1.0 - t) / (1.0 - mH)
        P[i] = w / w.sum()
    return P, Hmask, P[:, Hmask].sum(axis=1)


def build_twin(arm, P, Hmask, top=5):
    """The twin's distribution per prompt, as one of the four arms."""
    Q = np.zeros_like(P)
    for i, p in enumerate(P):
        pH = np.where(Hmask, p, 0.0)
        pH = pH / pH.sum()
        if arm == "perfect":
            q = pH
        elif arm == "narrow":
            keep = np.argsort(-pH)[:top]
            q = np.zeros_like(pH)
            q[keep] = pH[keep]
            q /= q.sum()
        elif arm == "useless":
            q = p.copy()
        elif arm == "misaimed":
            q = np.where(~Hmask, p, 0.0)
            q /= q.sum()
        else:
            raise ValueError(f"unknown arm {arm!r}")
        Q[i] = q
    return Q


def forward_kl(P, Q, Hmask):
    """KL(p_H || q) per prompt; ``inf`` where q drops mass that p_H has."""
    out = []
    for p, q in zip(P, Q):
        pH = np.where(Hmask, p, 0.0)
        pH = pH / pH.sum()
        sup = pH > 0
        if np.any(q[sup] <= 0):
            out.append(float("inf"))
        else:
            out.append(float(np.sum(pH[sup] * np.log(pH[sup] / q[sup]))))
    return np.array(out)


def kl_label(kl):
    if not np.isfinite(kl).any():
        return "inf"
    return str(round(float(np.median(kl[np.isfinite(kl)])), 3))


# ---------------------------------------------------------------------------- sampling

def _sample_outcomes(D, idx, rng):
    """One outcome per entry of ``idx``, drawn from row ``idx[j]`` of ``D``."""
    y = np.empty(idx.size, dtype=np.int64)
    K = D.shape[1]
    for i in np.unique(idx):
        sel = idx == i
        y[sel] = rng.choice(K, size=int(sel.sum()), p=D[i])
    return y


def draw_mixture(P, Q, Hmask, alpha, n, reps, rng):
    """``(reps, n)`` draws of the summand w 1_H under the defensive mixture."""
    M = alpha * P + (1.0 - alpha) * Q
    S = (P / M) * Hmask                       # summand per (prompt, response), <= 1 / alpha
    total = reps * n
    idx = rng.integers(0, P.shape[0], size=total)
    from_p = rng.random(total) < alpha
    y = np.empty(total, dtype=np.int64)
    y[from_p] = _sample_outcomes(P, idx[from_p], rng)
    y[~from_p] = _sample_outcomes(Q, idx[~from_p], rng)
    s = S[idx, y]
    return np.minimum(s, 1.0 / alpha).reshape(reps, n)


def draw_plain(P, Hmask, n, reps, rng):
    """``(reps, n)`` plain 0/1 harm indicators under p, for the Clopper-Pearson comparison."""
    total = reps * n
    idx = rng.integers(0, P.shape[0], size=total)
    y = _sample_outcomes(P, idx, rng)
    return Hmask[y].astype(float).reshape(reps, n)


def plain_reference(P, Hmask, n, reps, rng, truth):
    """Clopper-Pearson on n plain samples from p: mean upper bound, miss rate, sd."""
    x = draw_plain(P, Hmask, n, reps, rng)
    ub = np.array([clopper_pearson_bounds(row, DELTA, n=n).upper for row in x])
    return {"upper_mean": float(ub.mean()), "miss_rate": float((ub < truth).mean()),
            "sd": float(x.std()), "est_mean": float(x.mean())}


def weighted_bounds(s, n, alpha):
    """Mean upper bound and miss-rate inputs for both weighted bounds on ``(reps, n)`` draws."""
    eb = np.array([empirical_bernstein_bounds(row, DELTA, n=n, a=0.0, b=1.0 / alpha).upper
                   for row in s])
    bt = np.array([betting_bounds(row, DELTA, n=n, a=0.0, b=1.0 / alpha).upper for row in s])
    return eb, bt


# ---------------------------------------------------------------------------- the check

def run(reps=1000, reps_scan=100, seed=0, out="results/twin/e5_toy"):
    os.makedirs(out, exist_ok=True)
    P, Hmask, per_prompt = build_toy(seed=seed)
    truth = float(per_prompt.mean())
    rng = np.random.default_rng(seed + 1)
    print(f"exact breach rate {truth:.5f} over {P.shape[0]} prompts", flush=True)

    # Clopper-Pearson depends on n only, so it is computed once per n.
    cp = {n: plain_reference(P, Hmask, n, reps, rng, truth) for n in NS}

    cells, t0 = [], time.time()
    for arm in ARMS:
        Q = build_twin(arm, P, Hmask)
        kl = forward_kl(P, Q, Hmask)
        for alpha in ALPHAS:
            for n in NS:
                s = draw_mixture(P, Q, Hmask, alpha, n, reps, rng)
                eb, bt = weighted_bounds(s, n, alpha)
                se = float(s.std()) / np.sqrt(s.size)
                ref = cp[n]
                cells.append({
                    "arm": arm, "alpha": alpha, "n": n,
                    "fwd_kl_median": kl_label(kl),
                    "fwd_kl_infinite": int(np.sum(~np.isfinite(kl))),
                    "est_mean": float(s.mean()), "est_bias": float(s.mean() - truth),
                    "est_se": se, "bias_in_se": float(abs(s.mean() - truth) / se),
                    "est_sd": float(s.std()), "plain_sd": ref["sd"],
                    "sd_ratio_vs_plain": float(s.std() / ref["sd"]),
                    "eb_upper_mean": float(eb.mean()),
                    "eb_miss_rate": float((eb < truth).mean()),
                    "bt_upper_mean": float(bt.mean()),
                    "bt_miss_rate": float((bt < truth).mean()),
                    "cp_upper_mean": ref["upper_mean"], "cp_miss_rate": ref["miss_rate"],
                    "eb_width_vs_cp": float((eb.mean() - truth) / (ref["upper_mean"] - truth)),
                    "bt_width_vs_cp": float((bt.mean() - truth) / (ref["upper_mean"] - truth)),
                })
                print(f"  {arm:9s} alpha {alpha} n {n:5d}  est {s.mean():.5f}  "
                      f"sd/plain {s.std() / ref['sd']:.3f}  "
                      f"EB {eb.mean():.4f} miss {(eb < truth).mean():.3f}  "
                      f"BT {bt.mean():.4f} miss {(bt < truth).mean():.3f}", flush=True)

    # ---------------------------------------------------- exploratory: where is the crossover
    print("crossover scan (perfect twin, outside the registered n grid)", flush=True)
    scan = []
    Qp = build_twin("perfect", P, Hmask)
    for alpha in ALPHAS:
        for n in NS_SCAN:
            ref = cp[n] if n in cp else plain_reference(P, Hmask, n, reps_scan, rng, truth)
            s = draw_mixture(P, Qp, Hmask, alpha, n, reps_scan, rng)
            eb, bt = weighted_bounds(s, n, alpha)
            scan.append({
                "alpha": alpha, "n": n, "reps": reps_scan if n not in cp else reps,
                "est_sd": float(s.std()), "plain_sd": ref["sd"],
                "eb_upper_mean": float(eb.mean()), "bt_upper_mean": float(bt.mean()),
                "cp_upper_mean": ref["upper_mean"],
                "eb_miss_rate": float((eb < truth).mean()),
                "bt_miss_rate": float((bt < truth).mean()),
                "eb_width_vs_cp": float((eb.mean() - truth) / (ref["upper_mean"] - truth)),
                "bt_width_vs_cp": float((bt.mean() - truth) / (ref["upper_mean"] - truth)),
            })
            print(f"  alpha {alpha} n {n:5d}  BT/CP width {scan[-1]['bt_width_vs_cp']:.2f}  "
                  f"EB/CP width {scan[-1]['eb_width_vs_cp']:.2f}", flush=True)
    elapsed = time.time() - t0

    # ---------------------------------------------------- scoring
    checks = []
    worst_eb = max(c["eb_miss_rate"] for c in cells)
    worst_bt = max(c["bt_miss_rate"] for c in cells)
    checks.append(("T1 both weighted bounds hold their level in every cell",
                   worst_eb <= DELTA and worst_bt <= DELTA,
                   f"worst miss rate: empirical Bernstein {worst_eb:.3f}, betting {worst_bt:.3f}, "
                   f"against delta {DELTA} over {reps} draws in {len(cells)} cells"))

    worst = max(cells, key=lambda c: c["bias_in_se"])
    checks.append(("T2 the estimator is unbiased in every cell, including a twin that drops modes",
                   worst["bias_in_se"] < 4.0,
                   f"largest |mean - truth| is {worst['bias_in_se']:.2f} standard errors "
                   f"({worst['arm']}, alpha {worst['alpha']}, n {worst['n']}: "
                   f"{worst['est_mean']:.5f} against {truth:.5f})"))

    narrow = [c for c in cells if c["arm"] == "narrow"]
    checks.append(("T3 validity survives an infinite forward KL (the narrow twin)",
                   all(c["eb_miss_rate"] <= DELTA and c["bt_miss_rate"] <= DELTA for c in narrow),
                   f"the narrow twin drops all but 5 modes, forward KL infinite on "
                   f"{narrow[0]['fwd_kl_infinite']} of {P.shape[0]} prompts; worst miss rate "
                   f"{max(max(c['eb_miss_rate'], c['bt_miss_rate']) for c in narrow):.3f}"))

    good = [c for c in cells if c["arm"] == "perfect"]
    best_sd = min(c["sd_ratio_vs_plain"] for c in good)
    checks.append(("T4 a good twin cuts the estimator's variance (section 6's variance claim)",
                   best_sd < 0.5,
                   f"perfect twin: the summand's standard deviation is {best_sd:.3f} to "
                   f"{max(c['sd_ratio_vs_plain'] for c in good):.3f} of plain Monte Carlo's"))

    best_w = min(c["bt_width_vs_cp"] for c in good if c["n"] == 300)
    checks.append(("T5 that variance cut makes the bound at least half Clopper-Pearson's width "
                   "at n = 300 (E5b's direction)",
                   best_w <= 0.5,
                   f"perfect twin at n 300: the betting bound's width is {best_w:.2f} of "
                   f"Clopper-Pearson's, against the registered target of 0.5 or less"))

    bad = [c for c in cells if c["arm"] in ("useless", "misaimed")]
    checks.append(("T6 a bad twin is no narrower than Clopper-Pearson (E5c's direction)",
                   all(c["bt_width_vs_cp"] >= 1.0 for c in bad),
                   f"useless and misaimed twins: betting width "
                   f"{min(c['bt_width_vs_cp'] for c in bad):.2f} to "
                   f"{max(c['bt_width_vs_cp'] for c in bad):.2f} of Clopper-Pearson's"))

    crossed = [r for r in scan if r["bt_width_vs_cp"] < 1.0]
    checks.append(("T7 the weighted bound overtakes Clopper-Pearson somewhere in the scanned range",
                   bool(crossed),
                   (f"first crossing at alpha {crossed[0]['alpha']}, n {crossed[0]['n']} "
                    f"(width {crossed[0]['bt_width_vs_cp']:.2f})") if crossed else
                   f"no crossing up to n {max(NS_SCAN)}; best width "
                   f"{min(r['bt_width_vs_cp'] for r in scan):.2f} of Clopper-Pearson's"))

    n_pass = sum(1 for _, ok, _ in checks if ok)
    results = {"truth": truth, "delta": DELTA, "reps": reps, "reps_scan": reps_scan, "seed": seed,
               "per_prompt_truth": [round(float(v), 5) for v in per_prompt],
               "elapsed_s": round(elapsed, 1), "cells": cells, "scan": scan,
               "checks": [{"name": k, "pass": bool(ok), "detail": d} for k, ok, d in checks],
               "n_pass": n_pass, "n_checks": len(checks)}
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(results, f, indent=1)

    lines = ["# Pre-E5: the defensive-mixture bound on an exact toy", "",
             f"seed {seed}; {reps} draws per cell; delta {DELTA} one-sided; "
             f"{P.shape[0]} prompts over {P.shape[1]} responses, {int(Hmask.sum())} of them harmful",
             f"exact breach rate {truth:.5f}; per-prompt P(H | x) from "
             f"{per_prompt.min():.4f} to {per_prompt.max():.4f}", "",
             "Not the registered E5 plasmode (which resamples the real brute-force pools). A check "
             "of the estimator and the two bounds before any GPU time; no registered number changes.",
             "", "## Checks", ""]
    for k, ok, d in checks:
        lines.append(f"- {'PASS' if ok else 'FAIL'} {k}: {d}")
    lines += ["", f"{n_pass} of {len(checks)} checks pass.", "",
              "## Cells (the registered grid)", "",
              "| twin | fwd KL | alpha | n | estimate | sd / plain sd | EB bound | EB miss | "
              "betting | BT miss | CP bound | CP miss | BT width / CP |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        lines.append(
            f"| {c['arm']} | {c['fwd_kl_median']} | {c['alpha']} | {c['n']} | {c['est_mean']:.5f} | "
            f"{c['sd_ratio_vs_plain']:.3f} | {c['eb_upper_mean']:.4f} | {c['eb_miss_rate']:.3f} | "
            f"{c['bt_upper_mean']:.4f} | {c['bt_miss_rate']:.3f} | {c['cp_upper_mean']:.4f} | "
            f"{c['cp_miss_rate']:.3f} | {c['bt_width_vs_cp']:.2f} |")
    lines += ["", "## Crossover scan (perfect twin; n beyond the registered grid is exploratory)", "",
              "| alpha | n | draws | sd / plain sd | betting | CP | BT width / CP | EB width / CP |",
              "|---|---|---|---|---|---|---|---|"]
    for r in scan:
        lines.append(
            f"| {r['alpha']} | {r['n']} | {r['reps']} | {r['est_sd'] / r['plain_sd']:.3f} | "
            f"{r['bt_upper_mean']:.4f} | {r['cp_upper_mean']:.4f} | "
            f"{r['bt_width_vs_cp']:.2f} | {r['eb_width_vs_cp']:.2f} |")
    lines += ["", f"Runtime {elapsed:.0f} s on the CPU.", ""]
    with open(os.path.join(out, "report.md"), "w") as f:
        f.write("\n".join(lines))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--reps-scan", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/twin/e5_toy")
    a = ap.parse_args()
    r = run(reps=a.reps, reps_scan=a.reps_scan, seed=a.seed, out=a.out)
    print(f"\n{r['n_pass']} of {r['n_checks']} checks pass; wrote {a.out}/report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
