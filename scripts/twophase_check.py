"""Reference-rate strata when the claim is about the prompt source, not the pool (plan step R4, item 1).

The stored cells of section 8.1 take the pool of about 500 prompts as the population. A user
who certifies a policy wants the rate over the source the prompts came from. Then the pool is
itself a sample, the strata are rebuilt on every pool, and the gain is capped: section 6.4 gives
``1 / (1 - G + G n_s / N)``. This script measures that case on the same real responses.

One replication, two phases:

1. a pool of N prompts drawn with replacement from the stored pool, which stands for the source
   (a prompt keeps its stored reference responses; the source's rate is the stored pool's rate);
2. 8 equal rank strata of the 8-sample reference rate rebuilt on that pool (random ties), a
   proportional stratified safety set of n_s prompts drawn from it without replacement, and one
   response of the checkpoint for each.

Arms, fixed before the run:

- ``b1w, sampled-pool term``        ``b1w(..., N=N)``: adds ``sum_h W_h (p_h - mu)^2 / N``
- ``Wald-t b1, sampled-pool term``  ``b1(..., N=N)``
- ``b1w, no term``                  the bound of the stored cells, to show what leaving the term out costs
- ``StratPPI estimator, bootstrap-t``  as in the stored cells, with each stratum's N_h (its own term for the pool)
- ``pooled Wilson`` and ``Clopper-Pearson`` on a simple random sample of n_s prompts of the same pool,
  which is an i.i.d. sample of the source: the baseline, and the exact control.

ESS as in ``replacement_check.py``: (the pooled Wilson bound's mean excess over the truth / the
arm's)^2. The cap is computed from the with-replacement ESS of ``b1w`` in
``results/paper/replacement_check.json`` (G = 1 - 1 / ESS).

    OMP_NUM_THREADS=1 .venv/bin/python scripts/twophase_check.py          # about 10 minutes on 12 cores
    .venv/bin/python scripts/twophase_check.py --report-only
    -> results/paper/twophase_check.json, results/paper/twophase_check.md
"""
import argparse
import collections
import json
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

sys.dont_write_bytecode = True

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "results", "paper")
sys.path.insert(0, HERE)
import replacement_check as RC  # noqa: E402

BL, PM, P14, RP, SB, c = RC.BL, RC.PM, RC.P14, RC.RP, RC.SB, RC.c
DELTAS, K_REF, H = RC.DELTAS, RC.K_REF, RC.H
MID, RARE = RC.MID, RC.RARE
STRAT = ("b1w, sampled-pool term", "Wald-t b1, sampled-pool term", "b1w, no term", "StratPPI estimator, bootstrap-t")
RANDOM = ("pooled Wilson", "Clopper-Pearson")
ARMS = STRAT + RANDOM


def job(args):
    src, pool, lab, step, reps = args
    PM.TIES = "random"
    by, meta = RP.load() if src == "013" else P14.load("s0")
    P, _ = RP.build_pool(by, meta, pool, lab, step, False, "eval")
    f_all = P["cov"][:, :K_REF].mean(axis=1)
    draw, truth, N = P["draw"], float(P["truth"]), len(P["p_truth"])
    own = zlib.crc32(f"twophase:{src}:{pool}:{lab}".encode())
    rows = []
    for n_s in (100, 200):
        acc = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in range(reps):
            rr = np.random.default_rng([own, step, r, n_s])
            pid = rr.integers(0, N, N)                                    # phase 1: a pool from the source
            f = f_all[pid]
            strata = PM.quantile_strata(f, H, rr, ties="random")
            W = np.bincount(strata, minlength=H) / N
            stats = [(int((strata == h).sum()), float(f[strata == h].mean()), float(f[strata == h].var(ddof=1))) for h in range(H)]
            ids, st = PM._draw_split(strata, n_s, rr)                     # phase 2: the safety set
            y = draw(pid[ids], rr)
            s = np.array([y[st == h].sum() for h in range(H)]); n = np.array([(st == h).sum() for h in range(H)])
            est = SB.estimate(s, n, W)
            spb, sp_est = BL.stratppi_boot(y, f[ids], st, stats, W, DELTAS, rr)
            ids_r = rr.choice(N, size=n_s, replace=False)                 # the baseline: a random sample of the same pool
            y_r = draw(pid[ids_r], rr)
            for d, ub_ in zip(DELTAS, spb):
                for arm, ub, e in (("b1w, sampled-pool term", SB.b1w(s, n, W, d, N=N), est),
                                   ("Wald-t b1, sampled-pool term", SB.b1(s, n, W, d, N=N), est),
                                   ("b1w, no term", SB.b1w(s, n, W, d), est),
                                   ("StratPPI estimator, bootstrap-t", ub_, sp_est),
                                   ("pooled Wilson", SB.b1w(y_r.sum(), n_s, 1.0, d), y_r.mean()),
                                   ("Clopper-Pearson", float(c.cp_upper(y_r.sum(), n_s, d)), y_r.mean())):
                    acc[(arm, d)]["ub"].append(ub); acc[(arm, d)]["est"].append(e)
        for (arm, d), v in acc.items():
            rows.append(dict(src=src, env=f"{pool}:{lab}", step=step, n=n_s, N=N, delta=d, arm=arm, truth=truth, reps=reps,
                             sd_est=float(np.std(v["est"])), **BL.summarise(v["ub"], v["est"], truth)))
    return rows


def report(res, path):
    rows, R = res["rows"], res["reps"]
    rep = json.load(open(os.path.join(OUT, "replacement_check.json")))["rows"]
    key = lambda r: (r["src"], r["env"], r["step"], r["n"], r["delta"])      # noqa: E731
    large = {(key(r), r["arm"]): r for r in rep if r["draw"] == "large"}
    stored = {(key(r), r["arm"]): r for r in rep if r["draw"] == "stored"}
    by = collections.defaultdict(dict)
    for r in rows:
        by[key(r)][r["arm"]] = r

    def sel(arm, d, envs):
        return [r for r in rows if r["arm"] == arm and r["delta"] == d and r["env"] in envs]

    L = ["# Reference-rate strata for a claim about the prompt source", "",
         f"`scripts/twophase_check.py`. Every replication draws a pool of N prompts from the stored pool (the source), rebuilds "
         f"the 8 strata on it, and draws the safety set from that pool; the truth is the source's rate. {R:,} replications a "
         "cell; cells and labels as in `replacement_check.md`. Classes by the paper's rule.", "",
         "## 1. Cells over / unresolved / at or under", "",
         "| arm | mid-rate labels, delta 0.05 | delta 0.10 | largest miss, 0.05; 0.10 | rare labels, delta 0.05 | delta 0.10 |",
         "|---|---|---|---|---|---|"]
    dig = {}
    for arm in ARMS:
        cells = {(d, g): sel(arm, d, envs) for d in DELTAS for g, envs in (("mid", MID), ("rare", RARE))}
        dig[arm] = {f"{g}|{d}": dict(counts=RC.counts(v), largest=max(r["miss"] for r in v), cells=len(v)) for (d, g), v in cells.items()}
        L.append(f"| {arm} | {RC.counts(cells[(0.05, 'mid')])} | {RC.counts(cells[(0.1, 'mid')])} | "
                 f"{max(r['miss'] for r in cells[(0.05, 'mid')]):.3f}; {max(r['miss'] for r in cells[(0.1, 'mid')]):.3f} | "
                 f"{RC.counts(cells[(0.05, 'rare')])} | {RC.counts(cells[(0.1, 'rare')])} |")
    res["digest"] = dig
    L += ["", "## 2. Effective sample size for the source's rate, by label", "",
          "ESS against the pooled Wilson bound on a random sample of the same size (each from the truth to the limit), at delta "
          "0.05, median over a label's checkpoints. *Pool claim* is `b1w`'s ESS for the pool's own rate, drawn without "
          "replacement (the stored design) and with replacement (a large pool). *Cap* is section 6.4's "
          "`1 / (1 - G + G n_s / N)` with `G = 1 - 1 / ESS` from the with-replacement value.", "",
          "| label (rate) | n_s | pool claim, stored; large pool | cap | " + " | ".join(a for a in STRAT) + " |",
          "|---|---|---|---|" + "---|" * len(STRAT)]
    ess = {}
    groups = collections.defaultdict(list)
    for k in by:
        if k[4] == 0.05:
            groups[(k[0], k[1], k[3])].append(k)
    for (src, env, n_s), ks in sorted(groups.items(), key=lambda x: (x[0][1] not in MID, x[0][1], x[0][0], x[0][2])):
        vals = collections.defaultdict(list)
        for k in ks:
            base = by[k]["pooled Wilson"]["excess"]
            for arm in STRAT:
                vals[arm].append((base / by[k][arm]["excess"]) ** 2)
            e_st = (stored[(k, "pooled Wilson")]["excess"] / stored[(k, "b1w")]["excess"]) ** 2
            e_lg = (large[(k, "pooled Wilson")]["excess"] / large[(k, "b1w")]["excess"]) ** 2
            G = 1 - 1 / e_lg
            vals["stored"].append(e_st); vals["large"].append(e_lg)
            vals["cap"].append(1 / (1 - G + G * n_s / by[k]["pooled Wilson"]["N"]))
        m = {a: float(np.median(v)) for a, v in vals.items()}
        rate = float(np.mean([by[k]["pooled Wilson"]["truth"] for k in ks]))
        name = f"{env}{' (014)' if src == '014' else ''}"
        ess[f"{name}|{n_s}"] = dict(rate=rate, **m)
        L.append(f"| {name} ({rate:.2f}) | {n_s} | {m['stored']:.2f}; {m['large']:.2f} | {m['cap']:.2f} | "
                 + " | ".join(f"{m[a]:.2f}" for a in STRAT) + " |")
    res["ess"] = ess
    L += ["", "## 3. Every cell at delta 0.05", "",
          "| label | step | n_s | truth | " + " | ".join(f"{a}: miss" for a in ARMS) + " |", "|---|---|---|---|" + "---|" * len(ARMS)]
    for k in sorted(by, key=lambda k: (k[1] not in MID, k[1], k[0], k[2], k[3])):
        if k[4] != 0.05:
            continue
        g = by[k]
        cells = []
        for a in ARMS:
            m = g[a]["miss"]
            cl = RC.classify(m, 0.05, g[a]["reps"])
            cells.append(f"**{m:.3f}**" if cl == RC.OVER else (f"{m:.3f}?" if cl == RC.UNRES else f"{m:.3f}"))
        L.append(f"| {k[1]}{' (014)' if k[0] == '014' else ''} | {k[2]} | {k[3]} | {g[ARMS[0]]['truth']:.3f} | " + " | ".join(cells) + " |")
    json.dump(res, open(os.path.join(OUT, "twophase_check.json"), "w"))
    open(path, "w").write("\n".join(L) + "\n")
    print("\n".join(L[:40]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    path = os.path.join(OUT, "twophase_check.json")
    if not a.report_only:
        jobs = [("013", p, lab, s, a.reps) for p, labs in RP.LABELS.items() for lab in labs for s in (0, 100, 200)]
        jobs += [("014", "C1", "refusal", s, a.reps) for s in (100, 200)]
        t0 = time.time(); rows = []
        with ProcessPoolExecutor(a.workers) as ex:
            for i, r in enumerate(ex.map(job, jobs)):
                rows += r; print(f"{i + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
        json.dump(dict(reps=a.reps, rows=rows), open(path, "w"))
    report(json.load(open(path)), os.path.join(OUT, "twophase_check.md"))


if __name__ == "__main__":
    main()
