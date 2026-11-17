"""W2 / R2: a skew-corrected stratified Wald-t limit (`b1s`) and its check.

The stratified Wald-t limit `b1` is a one-sided upper limit centred on the estimate. For the studentised
estimate T = (mu_hat - mu) / sqrt(V_hat) the one-term Edgeworth expansion is
P(T <= x) = Phi(x) + g (2 x^2 + 1) / 6 phi(x), g = k3 / V^1.5 the estimate's skewness (Hall 1992, section 2.6;
Johnson 1978), so the limit misses with probability delta + g (2 z^2 + 1) / 6 phi(z): over its level whenever
the estimate is right-skewed, that is at rates under one half, more so at low rates and small strata. `b1s`
adds the term that cancels it and never subtracts:

    b1s = b1 + max(0, k3_hat / V_hat) (2 t^2 + 1) / 6,
    k3_hat = sum_h W_h^3 p~_h (1 - p~_h) (1 - 2 p~_h) / n_h^2,

with b1's own smoothed rates p~_h = (s_h + 0.5) / (n_h + 1), variance V_hat and t quantile.

    .venv/bin/python scripts/waldt_skew.py dev       # the 116 cells of exact_strat.py (seen), 40,000 draws
    .venv/bin/python scripts/waldt_skew.py check     # the confirmation grid: cells and allocations only
    .venv/bin/python scripts/waldt_skew.py run       # the confirmation grid (registered)
    .venv/bin/python scripts/waldt_skew.py report
"""
import collections
import json
import os
import sys
import zlib

import numpy as np
from scipy.stats import binom, t as tdist

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "results", "paper", "waldt_fix")
DELTAS = (0.05, 0.1)
REPS = 40000
RARE = 0.05


def limits(S, n, W, delta):
    """Vectorised b1 and b1s for an (R, H) array of counts. Returns (est, b1, b1s)."""
    S = np.asarray(S, dtype=float)
    n = np.asarray(n, dtype=float)[None, :]
    W = np.asarray(W, dtype=float)[None, :]
    p = S / n
    pt = (S + 0.5) / (n + 1.0)
    m = np.maximum(n - 1, 1)
    v_h = W ** 2 * pt * (1 - pt) / m
    var = v_h.sum(axis=1)
    df = np.maximum(var ** 2 / np.sum(v_h ** 2 / m, axis=1), 1.0)
    mu = np.sum(W * p, axis=1)
    tq = tdist.ppf(1 - delta, df)
    b1 = mu + tq * np.sqrt(var)
    k3 = np.sum(W ** 3 * pt * (1 - pt) * (1 - 2 * pt) / n ** 2, axis=1)
    b1s = b1 + np.maximum(k3 / var, 0.0) * (2 * tq ** 2 + 1) / 6
    return mu, b1, b1s


def b1s(s, n, W, delta):
    return float(limits(np.atleast_2d(s), n, W, delta)[2][0])


def cp_upper(k, n, d):
    from scipy.special import betaincinv
    k = np.asarray(k, dtype=float)
    return np.where(k >= n, 1.0, betaincinv(np.minimum(k + 1, n), np.maximum(n - k, 1e-9), 1 - d))


def classify(m, delta, R):
    se = np.sqrt(delta * (1 - delta) / R)
    return "over" if m > delta + 2 * se else "unresolved" if m > delta else "at or under"


def one(theta, W, n_h, truth, seed, tag):
    """Miss and mean excess of b1 and b1s, and of Clopper-Pearson on a random draw of sum(n_h) (exact)."""
    rng = np.random.default_rng(seed)
    S = rng.binomial(np.asarray(n_h)[None, :], np.asarray(theta)[None, :], size=(REPS, len(n_h)))
    n = int(np.sum(n_h))
    pmf = binom.pmf(np.arange(n + 1), n, truth)
    rows = []
    for d in DELTAS:
        _, u1, us = limits(S, n_h, W, d)
        cp = cp_upper(np.arange(n + 1), n, d)
        ex_cp = float(pmf @ cp - truth)
        for arm, u in (("b1", u1), ("b1s", us)):
            ex = float(u.mean() - truth)
            rows.append(dict(cell=tag, truth=float(truth), n=n, H=len(n_h), delta=d, arm=arm, R=REPS,
                             miss=float(np.mean(u < truth)), excess=ex, gain_cp=(ex_cp / ex) ** 2))
    return rows


# ------------------------------------------------------------------ the seen cells (development)

def dev():
    sys.path.insert(0, HERE)
    import exact_strat as ES
    rows = []
    for c in ES.cells():
        P = ES.params(c)
        for r in one(P["theta"], P["W"], P["n_h"], P["truth"], zlib.crc32(("dev:" + ES.key(c)).encode()), ES.key(c)):
            rows.append(dict(group=c[0], **r))
    json.dump(rows, open(os.path.join(OUT, "dev.json"), "w"))
    summarise(rows, os.path.join(OUT, "dev.md"), "Development: the 116 cells of `exact_strat.py` (seen before the limit was written)",
              groups=("real", "costly", "grid"))


# ------------------------------------------------------------------ the confirmation grid (registered)

SHAPES = {   # stratum rates as multiples of the overall rate before rescaling; the mean is set to the cell's rate
    "flat": lambda H: np.ones(H),
    "linear": lambda H: np.linspace(0.2, 1.8, H),
    "one hot": lambda H: np.r_[np.full(H - 1, 0.3), 0.3 + 0.7 * H],      # most of the rate in the last stratum
    "half zero": lambda H: np.r_[np.zeros(H // 2), np.full(H - H // 2, 1.0)],
}
RATES = (0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90)
HS = (2, 4, 8)
NS = (100, 200, 400)
WEIGHTS = ("equal", "skewed")      # skewed: W_h proportional to 1, 2, ..., H; allocation proportional, floor 2


def confirm_cells():
    out = []
    for shape in SHAPES:
        for H in HS:
            for rate in RATES:
                for wname in WEIGHTS:
                    W = np.full(H, 1.0 / H) if wname == "equal" else np.arange(1, H + 1) / np.arange(1, H + 1).sum()
                    raw = SHAPES[shape](H)
                    theta = raw * rate / float(W @ raw)
                    if theta.max() > 1.0:          # the shape cannot carry this rate: clip and spread the rest evenly
                        for _ in range(50):
                            theta = np.minimum(theta, 1.0)
                            gap = rate - float(W @ theta)
                            free = theta < 1.0
                            if abs(gap) < 1e-12 or not free.any():
                                break
                            theta[free] += gap / float(W[free].sum())
                    theta = np.clip(theta, 0.0, 1.0)
                    for n_s in NS:
                        n_h = np.maximum(2, np.round(n_s * W).astype(int))
                        out.append(dict(tag=f"{shape}:H{H}:r{rate}:{wname}:n{n_s}", shape=shape, H=H, rate=rate, weights=wname,
                                        n_s=n_s, theta=theta, W=W, n_h=n_h, truth=float(W @ theta)))
    return out


def run():
    rows = []
    for c in confirm_cells():
        for r in one(c["theta"], c["W"], c["n_h"], c["truth"], zlib.crc32(("confirm:" + c["tag"]).encode()), c["tag"]):
            rows.append(dict(group="confirm", shape=c["shape"], weights=c["weights"], n_s=c["n_s"], rate=c["rate"], **r))
    json.dump(rows, open(os.path.join(OUT, "confirm.json"), "w"))


def band(t):
    return "rare (<5%)" if t < RARE else "5% to 50%" if t <= 0.5 else "above 50%"


def summarise(rows, path, title, groups):
    L = [f"# {title}", "", f"`scripts/waldt_skew.py`; {REPS:,} draws a cell; a cell is over when its miss exceeds delta by more than "
         "2 Monte Carlo SE. `b1` is the stratified Wald-t limit, `b1s` its skew-corrected form. Gain is against "
         "Clopper-Pearson on a random draw of the same number of labels, (CP excess / arm excess)^2.", "",
         "| group | rates | delta | arm | cells | over / unresolved / at or under | largest miss | gain vs CP (median, range) |",
         "|---|---|---|---|---|---|---|---|"]
    for g in groups:
        for b in ("rare (<5%)", "5% to 50%", "above 50%"):
            for d in DELTAS:
                for arm in ("b1", "b1s"):
                    sel = [r for r in rows if r["group"] == g and band(r["truth"]) == b and r["delta"] == d and r["arm"] == arm]
                    if not sel:
                        continue
                    k = collections.Counter(classify(r["miss"], d, r["R"]) for r in sel)
                    gc = np.array([r["gain_cp"] for r in sel])
                    L.append(f"| {g} | {b} | {d} | {arm} | {len(sel)} | {k['over']} / {k['unresolved']} / {k['at or under']} | "
                             f"{max(r['miss'] for r in sel):.4f} | {np.median(gc):.2f} ({gc.min():.2f}-{gc.max():.2f}) |")
    by = {(r["cell"], r["delta"], r["arm"]): r for r in rows}
    ratio = collections.defaultdict(list)
    for (c, d, a), r in by.items():
        if a == "b1s":
            ratio[(r["group"], band(r["truth"]), d)].append(r["gain_cp"] / by[(c, d, "b1")]["gain_cp"])
    L += ["", "## Cost of the correction: gain of `b1s` over gain of `b1`, cell by cell", "",
          "| group | rates | delta | median | range |", "|---|---|---|---|---|"]
    for (g, b, d), v in sorted(ratio.items()):
        L.append(f"| {g} | {b} | {d} | {np.median(v):.3f} | {min(v):.3f}-{max(v):.3f} |")
    L += ["", "## Every cell where either limit is over", "", "| cell | truth | n | delta | b1 miss | b1s miss |", "|---|---|---|---|---|---|"]
    for (c, d, a), r in by.items():
        if a != "b1":
            continue
        s = by[(c, d, "b1s")]
        if "over" in (classify(r["miss"], d, r["R"]), classify(s["miss"], d, s["R"])):
            L.append(f"| {c} | {r['truth']:.3f} | {r['n']} | {d} | {r['miss']:.4f} | {s['miss']:.4f} |")
    open(path, "w").write("\n".join(L) + "\n")
    print("\n".join(L[:60]))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    cmd = sys.argv[1]
    if cmd == "dev":
        dev()
    elif cmd == "check":
        cs = confirm_cells()
        print(len(cs), "cells")
        for c in cs[::37]:
            print(c["tag"], np.round(c["theta"], 3), c["n_h"], round(c["truth"], 4))
        assert all(abs(c["truth"] - c["rate"]) < 1e-9 for c in cs), "a shape missed its rate"
    elif cmd == "run":
        run()
    elif cmd == "report":
        summarise(json.load(open(os.path.join(OUT, "confirm.json"))), os.path.join(OUT, "confirm.md"),
                  "Confirmation: a synthetic grid fixed before it was run", groups=("confirm",))
