"""The gains table of the workshop paper for the skew-corrected stratified Wald-t limit `b1s` (not registered).

Cells, labels, strata, draws and the pooled Wilson baseline are those of the runs behind the table:
`replacement_check.job` (the pool's own rate, drawn with replacement, 40,000 draws, the same seeds, so the uncorrected
limit's figures reproduce), `twophase_check.job` (the prompt source, 10,000 replications; the bootstrap arm is left
out, so the baseline's draws differ from the earlier run's) and `costly_label.job` (human labels, with replacement).
With the sampled-pool term, `b1s` adds the term to the variance as `b1` does and divides the second phase's third
moment by the total variance.

    OMP_NUM_THREADS=1 .venv/bin/python scripts/waldt_skew_gains.py      # -> results/paper/waldt_fix/gains.{json,md}
"""
import collections
import json
import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import t as tdist

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "results", "paper", "waldt_fix")
sys.path.insert(0, HERE)
import confirm_pool as CPL        # noqa: E402
import costly_label as CL         # noqa: E402
import replacement_check as RC    # noqa: E402
import twophase_check as TP       # noqa: E402

SB, PM, H, K_REF, DELTAS = RC.SB, RC.PM, RC.H, RC.K_REF, RC.DELTAS
_LOAD, _LABELS = RC.RP.load, RC.RP.LABELS
RUNS = {   # tag: (judged file, pool, labels, steps)
    "R5": ("results/paper/confirm/judged.jsonl", {"K1": ["refusal"], "K2": ["refusal", "unsafe"], "K3": ["refusal", "unsafe"]}, (175,)),
    "R5b": ("results/paper/confirm/mid/judged.jsonl", {"K4": ["refusal", "unsafe"]}, (175,)),
    "O6": ("results/paper/o6/judged.jsonl", {"K5": ["refusal"]}, (100, 200)),
    "B3": ("results/paper/o6b/judged.jsonl", {"K5": ["refusal"]}, (100, 200)),
}
R_LARGE, R_TWO = 40000, 10000


def lim(S, n, W, delta, N=None):
    """(b1, b1s) for (R, H) arrays of counts, sizes and weights."""
    S, n, W = (np.asarray(x, dtype=float) for x in (S, n, W))
    p = S / n
    pt = (S + 0.5) / (n + 1.0)
    m = np.maximum(n - 1, 1)
    v_h = W ** 2 * pt * (1 - pt) / m
    var = v_h.sum(axis=1)
    df = np.maximum(var ** 2 / np.sum(v_h ** 2 / m, axis=1), 1.0)
    mu = np.sum(W * p, axis=1)
    if N is not None:
        var = var + np.sum(W * (p - mu[:, None]) ** 2, axis=1) / N
    tq = tdist.ppf(1 - delta, df)
    b1 = mu + tq * np.sqrt(var)
    k3 = np.sum(W ** 3 * pt * (1 - pt) * (1 - 2 * pt) / n ** 2, axis=1)
    return b1, b1 + np.maximum(k3 / var, 0.0) * (2 * tq ** 2 + 1) / 6


def wilson(k, n_s, d):
    tab = {int(x): SB.b1w(int(x), n_s, 1.0, d) for x in np.unique(k)}
    return np.array([tab[int(x)] for x in k])


def rows_from(tag, env, step, n_s, claim, truth, S, n, W, kr, N=None):
    out = []
    for d in DELTAS:
        u1, us = lim(S, n, W, d, N)
        uw = wilson(kr, n_s, d)
        for i, r in enumerate(range(0, len(u1), 4000)):     # the uncorrected limit against the scalar code, a few draws
            assert abs(u1[r] - SB.b1(S[r], n[r], W[r], d, N=N)) < 1e-9
        ex = {a: float(u.mean() - truth) for a, u in (("b1", u1), ("b1s", us), ("pooled Wilson", uw))}
        for a, u in (("b1", u1), ("b1s", us), ("pooled Wilson", uw)):
            out.append(dict(run=tag, env=env, step=step, n=n_s, claim=claim, delta=d, arm=a, truth=truth, reps=len(u),
                            miss=float(np.mean(u < truth)), excess=ex[a], ess=(ex["pooled Wilson"] / ex[a]) ** 2))
    return out


def job(args):
    tag, pool, lab, step = args
    judged, labels, _ = RUNS[tag]
    RC.RP.load, RC.RP.LABELS = _LOAD, _LABELS
    CPL.LABELS = labels
    CPL.patch(os.path.join(ROOT, judged))
    P, f, strata = RC.setup("013", pool, lab, step)
    draw, truth, N = P["draw"], float(P["truth"]), len(P["p_truth"])
    W0 = np.bincount(strata) / N
    own = zlib.crc32(f"013:{pool}:{lab}".encode())
    rows = []
    for n_s in (100, 200):
        S, n, kr = [], [], []
        for r in range(R_LARGE):                                           # replacement_check.job, the large draw
            rr = np.random.default_rng([own, step, r, n_s])
            ids, st = RC.draw_split(strata, n_s, rr, True)
            y = draw(ids, rr)
            S.append([y[st == h].sum() for h in range(H)]); n.append([(st == h).sum() for h in range(H)])
            rr = np.random.default_rng([own, step, r, n_s])
            kr.append(draw(rr.choice(N, size=n_s, replace=True), rr).sum())
        rows += rows_from(tag, f"{pool}:{lab}", step, n_s, "pool", truth, np.array(S), np.array(n), np.tile(W0, (R_LARGE, 1)), np.array(kr))
        own2 = zlib.crc32(f"twophase:013:{pool}:{lab}".encode())
        f_all = P["cov"][:, :K_REF].mean(axis=1)
        S, n, Ws, kr = [], [], [], []
        for r in range(R_TWO):                                             # twophase_check.job without the bootstrap arm
            rr = np.random.default_rng([own2, step, r, n_s])
            pid = rr.integers(0, N, N)
            st_all = PM.quantile_strata(f_all[pid], H, rr, ties="random")
            ids, st = PM._draw_split(st_all, n_s, rr)
            y = draw(pid[ids], rr)
            S.append([y[st == h].sum() for h in range(H)]); n.append([(st == h).sum() for h in range(H)])
            Ws.append(np.bincount(st_all, minlength=H) / N)
            kr.append(draw(pid[rr.choice(N, size=n_s, replace=False)], rr).sum())
        rows += rows_from(tag, f"{pool}:{lab}", step, n_s, "source", truth, np.array(S), np.array(n), np.array(Ws), np.array(kr), N=N)
    print(tag, pool, lab, step, flush=True)
    return rows


def costly_job(args):
    target, event, strat = args
    pids, human, prob, ref = CL.load()
    y_all = np.array([l in CL.EVENTS[event] for l in human[target]], dtype=float)
    f = CL.stratifier(strat, target, prob, ref)
    N, truth = len(y_all), float(y_all.mean())
    srng = np.random.default_rng(zlib.crc32(f"O3:{strat}:{target}".encode()))
    strata = PM.quantile_strata(f, CL.H, srng, ties="random")
    W = np.bincount(strata, minlength=CL.H) / N
    rows = []
    for n_s in CL.NS:
        S, n, kr = [], [], []
        for r in range(CL.REPS["large"]):
            rr = np.random.default_rng([zlib.crc32(f"{target}:{event}:{strat}:large".encode()), n_s, r])
            ids, st = CL.draw_split(strata, n_s, rr, True)
            y = y_all[ids]
            S.append([y[st == h].sum() for h in range(CL.H)]); n.append([(st == h).sum() for h in range(CL.H)])
            kr.append(y_all[rr.choice(N, size=n_s, replace=True)].sum())
        rows += rows_from("O3", f"{target}:{event}:{strat}", 0, n_s, "pool", truth, np.array(S), np.array(n),
                          np.tile(W, (len(S), 1)), np.array(kr))
    return rows


def report(rows, path):
    by = {(r["run"], r["env"], r["step"], r["n"], r["claim"], r["delta"], r["arm"]): r for r in rows}
    cls = lambda r: RC.classify(r["miss"], r["delta"], r["reps"])
    L = ["# Gains of the skew-corrected stratified Wald-t limit on the cells of the paper's gains table", "",
         "`scripts/waldt_skew_gains.py` (not registered). ESS against the pooled Wilson bound on a random draw, delta 0.05; "
         "miss at delta 0.05 / 0.10, **bold** when over by the paper's rule. `b1` is the uncorrected limit on the same draws.", "",
         "| run | label | rate | step | claim | n_s | ESS b1 | ESS b1s | miss b1 | miss b1s |", "|---|---|---|---|---|---|---|---|---|---|"]
    def m(r):
        return f"**{r['miss']:.3f}**" if cls(r) == RC.OVER else f"{r['miss']:.3f}"
    for k in sorted({k[:5] for k in by if k[0] != "O3"}):
        g = lambda d, a: by[k + (d, a)]
        L.append(f"| {k[0]} | {k[1]} | {g(0.05, 'b1')['truth']:.3f} | {k[2]} | {k[4]} | {k[3]} | {g(0.05, 'b1')['ess']:.2f} | {g(0.05, 'b1s')['ess']:.2f} | "
                 f"{m(g(0.05, 'b1'))} / {m(g(0.1, 'b1'))} | {m(g(0.05, 'b1s'))} / {m(g(0.1, 'b1s'))} |")
    L += ["", "## Cells over, by run group and claim (both levels)", "", "| group | claim | pairs | b1 over | b1s over |", "|---|---|---|---|---|"]
    grp = collections.defaultdict(list)
    for r in rows:
        if r["arm"] != "pooled Wilson":
            mid = 0.05 <= r["truth"] <= 0.95
            grp[("human labels" if r["run"] == "O3" else "language-model pools", "mid-rate" if mid else "rare or near one", r["claim"], r["arm"])].append(r)
    for (g, band, c) in sorted({k[:3] for k in grp}):
        a, b = grp[(g, band, c, "b1")], grp[(g, band, c, "b1s")]
        L.append(f"| {g}, {band} | {c} | {len(a)} | {sum(cls(r) == RC.OVER for r in a)} | {sum(cls(r) == RC.OVER for r in b)} |")
    L += ["", "## Human labels (XSTest), mid-rate labels, with replacement, delta 0.05", "",
          "| stratifier | n_s | labels | median ESS b1 | median ESS b1s | range b1s |", "|---|---|---|---|---|---|"]
    for strat in CL.STRATIFIERS:
        for n_s in CL.NS:
            v = [(by[k[:5] + (0.05, "b1")]["ess"], by[k[:5] + (0.05, "b1s")]["ess"]) for k in {k[:5] for k in by}
                 if k[0] == "O3" and k[1].endswith(":" + strat) and k[3] == n_s and 0.05 <= by[k[:5] + (0.05, "b1")]["truth"] <= 0.95]
            a, b = np.array(v).T
            L.append(f"| {strat} | {n_s} | {len(v)} | {np.median(a):.2f} | {np.median(b):.2f} | {b.min():.2f}-{b.max():.2f} |")
    open(path, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        report(json.load(open(os.path.join(OUT, "gains.json"))), os.path.join(OUT, "gains.md"))
        sys.exit()
    jobs = [(tag, p, lab, s) for tag, (_, labels, steps) in RUNS.items() for p, labs in labels.items() for lab in labs for s in steps]
    cj = [(t, e, s) for t in CL.MODELS for e in CL.EVENTS for s in CL.STRATIFIERS]
    with ProcessPoolExecutor(12) as ex:
        rows = [r for rs in ex.map(job, jobs) for r in rs] + [r for rs in ex.map(costly_job, cj) for r in rs]
    json.dump(rows, open(os.path.join(OUT, "gains.json"), "w"))
    report(rows, os.path.join(OUT, "gains.md"))
