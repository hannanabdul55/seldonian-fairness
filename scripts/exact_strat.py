"""B2 and B5 (blind review of v0.9.10, M3 and M8): an exact stratified bound on the paper's cells, and the
probability of certifying.

Registered in `.planning/paper-certification/B25_registration.md` before any outcome was computed.

Cells. Every cell is drawn with replacement within strata, the limit of a pool far larger than the safety
set, which is where the approximate limits were found over their level:

- ``real``    every large-draw cell of ``replacement_check.py`` (spike 013's C1-C3 at steps 0, 100, 200;
              spike 014's step 100 and 200), of step R5 (K1-K3), of step R5b (K4) and of O6 (K5): 58 cells.
- ``costly``  O3's ten XSTest human labels, two stratifiers, n_s 50 and 100: 40 cells.
- ``grid``    spike 013's synthetic profiles (``check_b1.py``), n_s 100, 200, 400: 18 cells.

In each cell the stratified draw's count in stratum h is Bin(n_h, theta_h), with theta_h the stratum's rate
and n_h the allocation of the cell's own script; the random draw's count is Bin(n_s, truth).

Arms. Exact: Clopper-Pearson on a random draw (``CP``, the baseline); ``cp_strat``, Clopper-Pearson on the
pooled count of the stratified draw plus the allocation slack sum_h max(0, W_h - n_h / n), which covers the
gap between the weighted mean and the pooled count's mean (a sum of independent Bernoulli variables is less
spread than the binomial with its mean, Hoeffding 1956; its exact miss is computed and reported in every
cell); ``b2``, spike 013's union-intersection betting bound, unchanged. Approximate: the Wald-t ``b1`` and
``b1w`` on the stratified draw, the pooled Wilson bound on a random draw. ``CP``, pooled Wilson and
``cp_strat`` are computed exactly from their count distributions; ``b1``, ``b1w`` and ``b2`` from R draws.

Measures per cell, delta and arm: the miss; the mean excess of the limit over the truth; the width gain
against CP and against the pooled Wilson bound, (baseline excess / arm excess)^2; and the probability of
certifying. For q in (0.5, 0.8), tau_q is the smallest threshold at which CP on a random draw of n_s labels
certifies with probability at least q; an arm's pass probability is P(limit <= tau_q), and its power ESS is
the number of labels a random draw needs for CP to reach that pass probability at tau_q (interpolated on
the running maximum of CP's pass probability over n), over the arm's own number of labels, divided by the
same ratio for CP itself (which removes the interpolation's discreteness: CP's own power ESS is 1).

    OMP_NUM_THREADS=1 .venv/bin/python scripts/exact_strat.py run       # about an hour on 14 cores
    .venv/bin/python scripts/exact_strat.py report
    -> results/paper/exact_strat.json, results/paper/exact_strat.md
"""
import argparse
import collections
import functools
import json
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import binom

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "results", "paper")
sys.path.insert(0, HERE)
import confirm_pool as CPL        # noqa: E402
import costly_label as CL         # noqa: E402
import replacement_check as RC    # noqa: E402

SB, PM = RC.SB, RC.PM
cp_upper = RC.c.cp_upper
DELTAS = (0.05, 0.1)
QS = (0.5, 0.8)
H = 8
REPS = 4000
RARE = 0.05
MIDRANGE = (0.05, 0.95)
EXACT = ("CP", "cp_strat", "b2")
ARMS = ("CP", "pooled Wilson", "cp_strat", "b2", "b1", "b1w")
STRAT_ARMS = ("cp_strat", "b2", "b1", "b1w")
GRID = {   # check_b1.py's profiles, equal weights and equal allocation
    "H4 spread": [0.02, 0.15, 0.4, 0.8], "H4 flat": [0.3] * 4, "H8 spread": list(np.linspace(0, 0.9, 8)),
    "H2 low": [0.0, 0.1], "H4 rare": [0.0, 0.0, 0.01, 0.07], "H4 mid": [0.2, 0.3, 0.4, 0.5],
}
_LOAD, _LABELS = RC.RP.load, RC.RP.LABELS
PATCHES = {"R5": "results/paper/confirm/judged.jsonl", "R5b": "results/paper/confirm/mid/judged.jsonl",
           "O6": "results/paper/o6/judged.jsonl"}
PATCH_LABELS = {"R5": {"K1": ["refusal"], "K2": ["refusal", "unsafe"], "K3": ["refusal", "unsafe"]},
                "R5b": {"K4": ["refusal", "unsafe"]}, "O6": {"K5": ["refusal"]}}


# ---------------------------------------------------------------- cells

def cells():
    out = []
    for p, labs in _LABELS.items():
        for lab in labs:
            for s in (0, 100, 200):
                out += [("real", "013", None, p, lab, s, n) for n in (100, 200)]
    out += [("real", "014", None, "C1", "refusal", s, n) for s in (100, 200) for n in (100, 200)]
    for tag, step in (("R5", 175), ("R5b", 175)):
        out += [("real", "013", tag, p, lab, step, n) for p, labs in PATCH_LABELS[tag].items() for lab in labs for n in (100, 200)]
    out += [("real", "013", "O6", "K5", "refusal", s, n) for s in (100, 200) for n in (100, 200)]
    out += [("costly", t, e, st, n) for t in CL.MODELS for e in CL.EVENTS for st in CL.STRATIFIERS for n in CL.NS]
    out += [("grid", name, n) for name in GRID for n in (100, 200, 400)]
    return out


def key(cell):
    return ":".join(str(x) for x in cell)


def params(cell):
    """theta_h, W_h, n_h, n_s and the truth of one cell."""
    if cell[0] == "real":
        _, src, tag, pool, lab, step, n_s = cell
        RC.RP.load, RC.RP.LABELS = _LOAD, _LABELS
        if tag is not None:
            CPL.LABELS = PATCH_LABELS[tag]
            CPL.patch(os.path.join(ROOT, PATCHES[tag]))
        P, f, strata = RC.setup(src, pool, lab, step)
        E = P["draw"].__defaults__[0]               # the responses the draw picks from, per prompt
        th_i = E.mean(axis=1)
        N = len(th_i)
        cnt = np.bincount(strata, minlength=H)
        W = cnt / N
        theta = np.array([th_i[strata == h].mean() for h in range(H)])
        n_h = np.array([min(max(2, int(round(n_s * c / N))), c) for c in cnt])     # RC.draw_split, replace=True
        truth = float(P["truth"])
        label = f"{tag or src}:{pool}:{lab}:{step}"
    elif cell[0] == "costly":
        _, target, event, strat, n_s = cell
        pids, human, prob, ref = CL.load()
        y = np.array([l in CL.EVENTS[event] for l in human[target]], dtype=float)
        f = CL.stratifier(strat, target, prob, ref)
        N = len(y)
        srng = np.random.default_rng(zlib.crc32(f"O3:{strat}:{target}".encode()))
        strata = PM.quantile_strata(f, H, srng, ties="random")
        cnt = np.bincount(strata, minlength=H)
        W = cnt / N
        theta = np.array([y[strata == h].mean() for h in range(H)])
        n_h = np.array([max(2, int(round(n_s * c / N))) for c in cnt])             # CL.draw_split, replace=True
        truth = float(y.mean())
        label = f"O3:{target}:{event}:{strat}"
    else:
        _, name, n_s = cell
        theta = np.array(GRID[name], dtype=float)
        W = np.full(len(theta), 1.0 / len(theta))
        n_h = np.full(len(theta), n_s // len(theta))
        truth = float(W @ theta)
        label = f"grid:{name}"
    assert abs(float(W @ theta) - truth) < 1e-9, (cell, float(W @ theta), truth)
    return dict(theta=theta, W=W, n_h=n_h, n_s=int(n_s), truth=truth, label=label)


# ---------------------------------------------------------------- exact count distributions

def pb_pmf(n_h, theta):
    """pmf of a sum of independent Bin(n_h, theta_h)."""
    pmf = np.array([1.0])
    for n, t in zip(n_h, theta):
        pmf = np.convolve(pmf, binom.pmf(np.arange(n + 1), n, t))
    return pmf


def exact_arm(pmf, ub, truth, taus):
    ub = np.asarray(ub)
    return dict(miss=float(pmf[ub < truth].sum()), excess=float(pmf @ ub - truth),
                passp={q: float(pmf[ub <= t + 1e-12].sum()) for q, t in taus.items()})


def wilson_counts(n, d):
    return np.array([SB.b1w(int(k), n, 1.0, d) for k in range(n + 1)])


def tau_q(n, p, d, q):
    """Smallest threshold at which CP on Bin(n, p) certifies with probability >= q."""
    U = cp_upper(np.arange(n + 1), n, d)
    cdf = binom.cdf(np.arange(n + 1), n, p)
    return float(U[int(np.argmax(cdf >= q - 1e-12))])


def cp_pass(n, p, d, tau):
    U = cp_upper(np.arange(n + 1), n, d)
    ok = np.flatnonzero(U <= tau + 1e-12)
    return float(binom.cdf(ok[-1], n, p)) if len(ok) else 0.0


@functools.lru_cache(maxsize=None)
def _cp_curve(p, d, tau, n_ref):
    ns = np.unique(np.round(np.geomspace(2, 64 * n_ref, 400)).astype(int))
    return ns, np.maximum.accumulate([cp_pass(int(n), p, d, tau) for n in ns])


def power_n(p, d, tau, target, n_ref):
    """Labels a random draw needs for CP to certify at tau with probability ``target`` (interpolated in log n on
    the running maximum of CP's pass probability); nan when it is never reached below 64 n_ref."""
    if target <= 0.0:
        return 0.0
    ns, pp = _cp_curve(p, d, tau, n_ref)
    if pp[-1] < target:
        return float("nan")
    i = int(np.argmax(pp >= target))
    if i == 0:
        return float(ns[0])
    x0, x1, y0, y1 = np.log(ns[i - 1]), np.log(ns[i]), pp[i - 1], pp[i]
    return float(np.exp(x0 + (x1 - x0) * (target - y0) / max(y1 - y0, 1e-15)))


# ---------------------------------------------------------------- one cell

def job(cell):
    t0 = time.time()
    P = params(cell)
    theta, W, n_h, n_s, truth = P["theta"], P["W"], P["n_h"], P["n_s"], P["truth"]
    n = int(n_h.sum())
    slack = float(np.maximum(W - n_h / n, 0.0).sum())
    pmf_s = pb_pmf(n_h, theta)
    pmf_r = binom.pmf(np.arange(n_s + 1), n_s, truth)
    rng = np.random.default_rng(zlib.crc32(key(cell).encode()))
    S = rng.binomial(n_h[None, :], theta[None, :], size=(REPS, len(n_h)))
    rows = []
    for d in DELTAS:
        taus = {q: tau_q(n_s, truth, d, q) for q in QS}
        res = {"CP": exact_arm(pmf_r, cp_upper(np.arange(n_s + 1), n_s, d), truth, taus),
               "pooled Wilson": exact_arm(pmf_r, wilson_counts(n_s, d), truth, taus),
               "cp_strat": exact_arm(pmf_s, np.minimum(cp_upper(np.arange(n + 1), n, d) + slack, 1.0), truth, taus)}
        memo = {}
        for arm, fn in (("b2", SB.b2), ("b1", SB.b1), ("b1w", SB.b1w)):
            ub = np.empty(REPS)
            for r in range(REPS):
                k = (arm, tuple(S[r]))
                if k not in memo:
                    memo[k] = fn(S[r], n_h, W, d)
                ub[r] = memo[k]
            res[arm] = dict(miss=float(np.mean(ub < truth)), excess=float(ub.mean() - truth),
                            passp={q: float(np.mean(ub <= t + 1e-12)) for q, t in taus.items()}, R=REPS)
        ref = {q: power_n(truth, d, taus[q], res["CP"]["passp"][q], n_s) / n_s for q in QS}
        for arm, v in res.items():
            labels = n if arm in STRAT_ARMS else n_s
            v["labels"] = labels
            v["power_ess"] = {q: power_n(truth, d, taus[q], v["passp"][q], n_s) / labels / ref[q] for q in QS}
            v["power_ref"] = {q: ref[q] for q in QS}
            v["gain_cp"] = (res["CP"]["excess"] / v["excess"]) ** 2 if v["excess"] > 0 else float("inf")
            v["gain_wilson"] = (res["pooled Wilson"]["excess"] / v["excess"]) ** 2 if v["excess"] > 0 else float("inf")
            rows.append(dict(group=cell[0], cell=key(cell), label=P["label"], n_s=n_s, n=n, slack=slack, truth=truth,
                             delta=d, tau={str(q): t for q, t in taus.items()}, arm=arm,
                             **{k: ({str(q): x for q, x in val.items()} if isinstance(val, dict) else val) for k, val in v.items()}))
    print(f"{key(cell)} {time.time() - t0:.0f}s", flush=True)
    return rows


# ---------------------------------------------------------------- report

def classify(r):
    """Exact arms computed from their distribution are over when the miss exceeds delta at all; drawn ones by
    the paper's rule (2 Monte Carlo SE)."""
    if "R" not in r:
        return "over" if r["miss"] > r["delta"] + 1e-12 else "at or under"
    return RC.classify(r["miss"], r["delta"], r["R"])


def mid(r):
    return MIDRANGE[0] <= r["truth"] <= MIDRANGE[1]


def score(rows):
    by = collections.defaultdict(list)
    for r in rows:
        by[r["arm"]].append(r)
    realc = lambda r: r["group"] in ("real", "costly")
    out = {}
    v = [r for a in ("b2", "cp_strat") for r in by[a] if classify(r) == "over"]
    out["X1"] = (not v, f"{len(v)} of {len(by['b2']) + len(by['cp_strat'])} exact-arm cells over")
    v = [r for r in by["b2"] if not r["gain_cp"] < 1]
    out["X2"] = (not v, f"b2 width gain against CP: max {max(r['gain_cp'] for r in by['b2']):.2f}; {len(v)} cells at or above 1")
    sel = [r for r in by["b2"] if realc(r) and r["truth"] >= RARE and r["delta"] == 0.05]
    v = [r for r in sel if not r["power_ess"]["0.8"] < 1]
    out["X3"] = (not v, f"b2 power ESS at 0.8: max {max(r['power_ess']['0.8'] for r in sel):.2f} over {len(sel)} cells; {len(v)} at or above 1")
    sel = [r for r in by["cp_strat"] if realc(r) and mid(r) and r["delta"] == 0.05]
    pe = np.array([r["power_ess"]["0.8"] for r in sel])
    out["X4"] = (bool(pe.min() >= 0.95 and np.median(pe) >= 1.10),
                 f"cp_strat power ESS at 0.8 over {len(sel)} cells: min {pe.min():.2f}, median {np.median(pe):.2f}")
    pe = np.array([r["power_ess"]["0.5"] for r in sel])
    out["X5"] = (bool(pe.min() >= 0.85 and pe.max() <= 1.15), f"cp_strat power ESS at 0.5: {pe.min():.2f}-{pe.max():.2f}")
    sel = [r for r in by["cp_strat"] if realc(r) and r["truth"] < RARE and r["delta"] == 0.05]
    pe = np.array([r["power_ess"]["0.8"] for r in sel])
    out["X6"] = (bool(pe.min() >= 0.8 and pe.max() <= 1.2), f"rare cells ({len(sel)}): cp_strat power ESS at 0.8 {pe.min():.2f}-{pe.max():.2f}")
    sel = [r for r in by["b1"] if r["group"] == "real" and mid(r) and r["delta"] == 0.05]
    v = [r for r in sel if not r["gain_cp"] >= r["gain_wilson"]]
    out["X7"] = (not v, f"b1 gain against CP >= against pooled Wilson in {len(sel) - len(v)} of {len(sel)} cells")
    cs = {(r["cell"], r["delta"]): r for r in by["cp_strat"]}
    v = [r for r in sel if not r["power_ess"]["0.8"] > cs[(r["cell"], r["delta"])]["power_ess"]["0.8"]]
    out["X8"] = (not v, f"b1 power ESS at 0.8 above cp_strat's in {len(sel) - len(v)} of {len(sel)} cells")
    return out


def report(res, path):
    rows = res["rows"]
    L = ["# B2 and B5: an exact stratified bound and the probability of certifying", "",
         f"`scripts/exact_strat.py`, registered in `.planning/paper-certification/B25_registration.md`. "
         f"{len(res['cells'])} cells, drawn with replacement within strata; b1, b1w and b2 from {REPS:,} draws a cell, "
         "CP, pooled Wilson and cp_strat exactly.", "", "## Registered predictions", "", "| # | held | detail |", "|---|---|---|"]
    for k, (ok, det) in score(rows).items():
        L.append(f"| {k} | {'kept' if ok else '**refuted**'} | {det} |")
    L += ["", "## Summary by group, delta and arm", "",
          "| group | rates | delta | arm | cells over / unresolved / at or under | miss range | width gain vs CP (median, range) | "
          "power ESS at 0.5 (median) | power ESS at 0.8 (median, range) |", "|---|---|---|---|---|---|---|---|---|"]
    for g in ("real", "costly", "grid"):
        for cls, f in (("mid (5-95%)", mid), ("rare (<5%)", lambda r: r["truth"] < RARE), ("high (>95%)", lambda r: r["truth"] > MIDRANGE[1])):
            for d in DELTAS:
                for arm in ARMS:
                    sel = [r for r in rows if r["group"] == g and f(r) and r["delta"] == d and r["arm"] == arm]
                    if not sel:
                        continue
                    k = collections.Counter(classify(r) for r in sel)
                    m = [r["miss"] for r in sel]
                    gc = np.array([r["gain_cp"] for r in sel])
                    p5 = np.array([r["power_ess"]["0.5"] for r in sel])
                    p8 = np.array([r["power_ess"]["0.8"] for r in sel])
                    L.append(f"| {g} | {cls} | {d} | {arm} | {k['over']} / {k['unresolved']} / {k['at or under']} | "
                             f"{min(m):.3f}-{max(m):.3f} | {np.median(gc):.2f} ({gc.min():.2f}-{gc.max():.2f}) | "
                             f"{np.nanmedian(p5):.2f} | {np.nanmedian(p8):.2f} ({np.nanmin(p8):.2f}-{np.nanmax(p8):.2f}) |")
    L += ["", "## Every cell at delta 0.05", "",
          "| cell | truth | n_s / strat labels | slack | tau_0.8 | CP pass | cp_strat miss, pass, power ESS | b2 miss, gain, power ESS | "
          "b1 miss, gain vs CP / vs Wilson, power ESS | b1w miss, power ESS |", "|---|---|---|---|---|---|---|---|---|---|"]
    by = {(r["cell"], r["delta"], r["arm"]): r for r in rows}
    for c in res["cells"]:
        k = key(c)
        if (k, 0.05, "CP") not in by:
            continue
        cp, cs, b2, b1, bw = (by[(k, 0.05, a)] for a in ("CP", "cp_strat", "b2", "b1", "b1w"))
        L.append(f"| {cs['label']} n_s {cs['n_s']} | {cs['truth']:.3f} | {cs['n_s']} / {cs['n']} | {cs['slack']:.4f} | "
                 f"{cs['tau']['0.8']:.3f} | {cp['passp']['0.8']:.2f} | {cs['miss']:.3f}, {cs['passp']['0.8']:.2f}, {cs['power_ess']['0.8']:.2f} | "
                 f"{b2['miss']:.3f}, {b2['gain_cp']:.2f}, {b2['power_ess']['0.8']:.2f} | "
                 f"{b1['miss']:.3f}, {b1['gain_cp']:.2f} / {b1['gain_wilson']:.2f}, {b1['power_ess']['0.8']:.2f} | "
                 f"{bw['miss']:.3f}, {bw['power_ess']['0.8']:.2f} |")
    open(path, "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "report", "check"])
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()
    path = os.path.join(OUT, "exact_strat.json")
    cs = cells()
    if a.stage == "check":                    # shapes only: every cell's allocation, no label drawn, no bound computed
        for c in cs:
            P = params(c)
            print(key(c), P["label"], P["n_s"], int(P["n_h"].sum()), len(P["n_h"]))
        print(len(cs), "cells")
        return
    if a.stage == "run":
        t0 = time.time()
        rows = []
        with ProcessPoolExecutor(a.workers) as ex:
            for i, r in enumerate(ex.map(job, cs)):
                rows += r
                print(f"{i + 1}/{len(cs)} {time.time() - t0:.0f}s", flush=True)
        json.dump(dict(reps=REPS, cells=cs, rows=rows), open(path, "w"), default=float)
    report(json.load(open(path)), os.path.join(OUT, "exact_strat.md"))
    print("-> results/paper/exact_strat.md")


if __name__ == "__main__":
    main()
