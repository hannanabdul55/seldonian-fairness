"""StratPPI (Fisch et al., NeurIPS 2024, arXiv:2406.04291) beside this project's bounds, on the
same resamples (paper plan P14).

StratPPI for a mean, as published: within stratum k, ``est_k = mean(y) + lam_k (mean_pool(f) -
mean_lab(f))`` with ``lam_k = cov(y, f) / ((1 + n_k / N_k) var(f))``; the estimate is the weighted
sum, its variance ``sum w_k^2 (var(y - lam_k f) / n_k + lam_k^2 var(f) / N_k)``, and the interval
uses normal quantiles (their Eq. 14). One-sided here: estimate + z(1 - delta) * se.

Part A, the reference model's rate as the predictor (spikes 013 and 014). Every prompt has the
reference's 8-sample rate; the safety set is drawn within 8 equal rank strata of it (random ties,
proportional allocation) and one trained-policy label is drawn per prompt, exactly as 013's
plasmode does, with the same seeds. On those draws: the stratified Wilson-type bound ``b1w``
(this project's), the stratified Wald-t ``b1``, and StratPPI. On a simple random sample of the
same size: pooled Wilson (013's baseline R) and unstratified PPI++ with a normal and a
bootstrap-t limit.

Part B, a judge as the predictor (spike 017's refusal pools). The population is the pool (or the
pool reweighted to a rarer rate); strata are K equal-mass bins of the judge's logit on it, with
known weights, which is the setting the StratPPI paper assumes. n labelled and N - n unlabelled
responses are drawn within strata in proportion. ``b1w`` on the same stratified labels ignores the
judge within strata. Labels alone (Clopper-Pearson) and PPI++ (normal, bootstrap-t) draw N at
random and label the first n.

Both parts also run StratPPI's estimator with a bootstrap-t limit, which is not in the StratPPI
paper: the question is whether the limit, not the estimator, is what fails at these sizes.

    OMP_NUM_THREADS=1 .venv/bin/python scripts/stratppi_baseline.py --reps 5000
    -> results/paper/stratppi.json, results/paper/stratppi.md
"""
import argparse
import collections
import json
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import norm

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SP = os.path.join(ROOT, ".planning", "spikes")
for d in ("013-stratified-safety-set", "014-pushed-label-stratification", "017-calibration-carrying-certificate"):
    sys.path.insert(0, os.path.join(SP, d))
import cert017 as c          # noqa: E402
import plasmode as PM        # noqa: E402
import plasmode014 as P14    # noqa: E402
import plasmode017 as PL     # noqa: E402
import real_plasmode as RP   # noqa: E402
import stratbounds as SB     # noqa: E402

OUT = os.path.join(ROOT, "results", "paper")
DELTAS = (0.05, 0.1)
K_REF, H = 8, 8


def stratppi_point(y, f, st, pool, W):
    """StratPPI's estimate and its variance. ``pool[h]`` = (N_h, mean f, var f) of the predictor over
    the stratum's unlabelled draws; ``y``, ``f``, ``st`` are the labelled draws."""
    est = var = 0.0
    for h, (Nh, fbar, fvar) in enumerate(pool):
        m = st == h
        yh, fh = y[m], f[m]
        n = len(yh)
        lam = 0.0
        if n >= 3 and fvar > 0:
            lam = float(np.cov(yh, fh, ddof=1)[0, 1]) / ((1 + n / Nh) * fvar)
        est += W[h] * (yh.mean() + lam * (fbar - fh.mean()))
        var += W[h] ** 2 * (np.var(yh - lam * fh, ddof=1) / n + lam ** 2 * fvar / Nh)
    return est, var


def stratppi(y, f, st, pool, W, deltas):
    """Upper bounds as published: estimate + z(1 - delta) * se."""
    est, var = stratppi_point(y, f, st, pool, W)
    return [est + norm.ppf(1 - d) * np.sqrt(var) for d in deltas], est


def stratppi_boot(y, f, st, pool, W, deltas, rng, boots=300):
    """The same estimator with a bootstrap-t limit (not in the StratPPI paper): labels resampled
    within strata, the stratum's unlabelled mean perturbed by its standard error, as 017's
    ``ppipp_boot`` does for PPI++."""
    est, var = stratppi_point(y, f, st, pool, W)
    eb, vb = np.zeros(boots), np.zeros(boots)
    for h, (Nh, fbar, fvar) in enumerate(pool):
        m = st == h
        yh, fh = y[m], f[m]
        n = len(yh)
        idx = rng.integers(0, n, size=(boots, n))
        ys, fs = yh[idx], fh[idx]
        ym, fm = ys.mean(1), fs.mean(1)
        lam = np.zeros(boots)
        if n >= 3 and fvar > 0:
            lam = ((ys - ym[:, None]) * (fs - fm[:, None])).sum(1) / (n - 1) / ((1 + n / Nh) * fvar)
        fub = fbar + np.sqrt(fvar / Nh) * rng.standard_normal(boots)
        eb += W[h] * (ym + lam * (fub - fm))
        vb += W[h] ** 2 * ((ys - lam[:, None] * fs).var(axis=1, ddof=1) / n + lam ** 2 * fvar / Nh)
    with np.errstate(divide="ignore", invalid="ignore"):
        tt = np.where(vb > 0, (eb - est) / np.sqrt(vb), np.where(eb < est, -np.inf, np.inf))
    out = []
    for d in deltas:
        q = np.quantile(tt, d, method="lower")
        out.append(float(np.clip(est - q * np.sqrt(var), 0.0, 1.0)) if np.isfinite(q) else 1.0)
    return out, est


def summarise(ub, est, truth):
    ub = np.asarray(ub)
    # a NaN bound counts as a miss (ub < truth is False for NaN)
    return dict(miss=float((~(ub >= truth)).mean()), excess=float(ub.mean() - truth),
                width=float((ub - np.asarray(est)).mean()), pass02=float((ub <= truth + 0.02).mean()))


def part_a(job):
    src, pool, lab, step, reps = job
    PM.TIES = "random"
    by, meta = RP.load() if src == "013" else P14.load("s0")
    P, _ = RP.build_pool(by, meta, pool, lab, step, False, "eval")
    cov, draw, truth = P["cov"], P["draw"], float(P["truth"])
    N = len(P["p_truth"])
    f = cov[:, :K_REF].mean(axis=1)
    rng = np.random.default_rng([step, zlib.crc32(repr(("S2", K_REF, H)).encode())])   # 013's S2 strata
    strata = PM.quantile_strata(f, H, rng, ties="random")
    W = np.bincount(strata) / N
    pool_stats = [(int((strata == h).sum()), float(f[strata == h].mean()), float(f[strata == h].var(ddof=1)))
                  for h in range(H)]
    rows = []
    for n_s in (100, 200):
        acc = collections.defaultdict(lambda: collections.defaultdict(list))
        yr, flr, fur = [], [], []
        for r in range(reps):
            rr = np.random.default_rng([step, r, n_s])
            ids, st = PM._draw_split(strata, n_s, rr)
            y = draw(ids, rr)
            s = np.array([y[st == h].sum() for h in range(H)]); n = np.array([(st == h).sum() for h in range(H)])
            est = SB.estimate(s, n, W)
            sp, sp_est = stratppi(y, f[ids], st, pool_stats, W, DELTAS)
            spb, _ = stratppi_boot(y, f[ids], st, pool_stats, W, DELTAS, rr)
            for d, u, ub_ in zip(DELTAS, sp, spb):
                acc[("S2 + StratPPI, bootstrap-t", d)]["ub"].append(ub_); acc[("S2 + StratPPI, bootstrap-t", d)]["est"].append(sp_est)
                acc[("S2 + b1w (this paper)", d)]["ub"].append(SB.b1w(s, n, W, d)); acc[("S2 + b1w (this paper)", d)]["est"].append(est)
                acc[("S2 + Wald-t b1", d)]["ub"].append(SB.b1(s, n, W, d)); acc[("S2 + Wald-t b1", d)]["est"].append(est)
                acc[("S2 + StratPPI", d)]["ub"].append(u); acc[("S2 + StratPPI", d)]["est"].append(sp_est)
            rr = np.random.default_rng([step, r, n_s])               # 013's arm R on its own draw
            ids = rr.choice(N, size=n_s, replace=False)
            y = draw(ids, rr)
            for d in DELTAS:
                acc[("random + pooled Wilson (R)", d)]["ub"].append(SB.b1w(y.sum(), n_s, 1.0, d))
                acc[("random + pooled Wilson (R)", d)]["est"].append(y.mean())
            rest = np.ones(N, bool); rest[ids] = False
            yr.append(y); flr.append(f[ids]); fur.append(f[rest])
        yr, flr, fur = np.array(yr), np.array(flr), np.array(fur)
        e2 = c.ppi_point(yr, flr, fur, None)[0]
        for d in DELTAS:
            acc[("random + PPI++ normal", d)] = dict(ub=c.ppipp_clt(yr, flr, fur, d), est=e2)
            acc[("random + PPI++ bootstrap-t", d)] = dict(ub=c.ppipp_boot(yr, flr, fur, d, seed=step + n_s), est=e2)
        for (arm, d), v in acc.items():
            rows.append(dict(part="A", src=src, env=f"{pool}:{lab}", step=step, n=n_s, delta=d, arm=arm, truth=truth,
                             reps=reps, **summarise(v["ub"], v["est"], truth)))
    return rows


def part_b(job):
    key, y_pool, p_pool, n, big_n, reps, rate, seed = job
    rng = np.random.default_rng(seed)
    truth = float(y_pool.mean()) if rate is None else rate
    d = PL.DELTA
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    # The population: pool items with mass q (uniform, or reweighted so the label's rate is ``rate``,
    # which is what ``PL.sample`` draws from). Strata are fixed on it: K equal-mass rank bins of the
    # judge's logit, weights known. This is the setting the StratPPI paper assumes.
    x_pool = PL.logit01(p_pool)
    q = np.full(len(y_pool), 1.0 / len(y_pool)) if rate is None else np.where(
        y_pool == 1, rate / (y_pool == 1).sum(), (1 - rate) / (y_pool == 0).sum())
    order = np.lexsort((rng.random(len(y_pool)), x_pool))
    cum = np.cumsum(q[order]) - q[order] / 2
    strata_b = {}
    for K in (5, 10):
        st_pool = np.empty(len(y_pool), dtype=int); st_pool[order] = np.minimum((cum * K).astype(int), K - 1)
        strata_b[K] = (st_pool, np.array([q[st_pool == h].sum() for h in range(K)]), q)
    done = 0
    while done < reps:
        b = min(500, reps - done)
        y, p = PL.sample(rng, y_pool, p_pool, b, big_n, rate)
        x = PL.logit01(p)
        yl, xl, xu = y[:, :n], x[:, :n], x[:, n:]
        e2 = c.ppi_point(yl, xl, xu, None)[0]
        for arm, ub, est in (("labels alone, Clopper-Pearson", c.classical(yl, d), yl.mean(1)),
                             ("PPI++ normal", c.ppipp_clt(yl, xl, xu, d), e2),
                             ("PPI++ bootstrap-t (this paper)", c.ppipp_boot(yl, xl, xu, d, seed=seed + done), e2)):
            acc[arm]["ub"].append(ub); acc[arm]["est"].append(est)
        for K in (5, 10):
            st_pool, W, q = strata_b[K]
            share = np.maximum(2, np.rint(n * W).astype(int)); nu = np.maximum(2, np.rint((big_n - n) * W).astype(int))
            members = [np.flatnonzero(st_pool == h) for h in range(K)]
            probs = [q[mh] / q[mh].sum() for mh in members]
            for r in range(b):
                ids = np.concatenate([rng.choice(members[h], size=share[h], p=probs[h]) for h in range(K)])
                st = st_pool[ids]
                pool_stats = []
                for h in range(K):
                    fu = x_pool[rng.choice(members[h], size=nu[h], p=probs[h])]
                    pool_stats.append((int(nu[h]), float(fu.mean()), float(fu.var(ddof=1))))
                yy, ff = y_pool[ids], x_pool[ids]
                (u,), est = stratppi(yy, ff, st, pool_stats, W, (d,))
                (ub_,), _ = stratppi_boot(yy, ff, st, pool_stats, W, (d,), rng)
                acc[f"StratPPI, K={K}"]["ub"].append([u]); acc[f"StratPPI, K={K}"]["est"].append([est])
                acc[f"StratPPI, bootstrap-t, K={K}"]["ub"].append([ub_]); acc[f"StratPPI, bootstrap-t, K={K}"]["est"].append([est])
                s = np.array([yy[st == h].sum() for h in range(K)])
                acc[f"judge strata + b1w, K={K}"]["ub"].append([SB.b1w(s, share, W, d)])
                acc[f"judge strata + b1w, K={K}"]["est"].append([SB.estimate(s, share, W)])
        done += b
    return [dict(part="B", env="|".join(map(str, key)), n=n, N=big_n, delta=d, arm=arm, truth=truth, reps=reps,
                 rate="pool" if rate is None else rate,
                 **summarise(np.concatenate(v["ub"]), np.concatenate(v["est"]), truth)) for arm, v in acc.items()]


def report(rows, path, reps_a, reps_b):
    """Tables: a cell is ``miss; ESS; pass``. ESS is given only where the arm holds its level in that
    cell (largest miss over checkpoints within two Monte Carlo standard errors of delta)."""
    L = ["# StratPPI beside this project's bounds, same resamples", "",
         "- miss: share of draws with the bound under the truth; it should be at most delta.",
         "- ESS: (mean bound minus truth, for the block's first arm) / (the same for this arm), squared: how many",
         "  times more safety samples the first arm would need for the same bound. Given only where the arm is valid.",
         "- pass: share of draws certifying tau = truth + 0.02.", ""]

    def cellfmt(miss, base, x, d, reps, with_pass=True):
        ok = miss <= d + 2 * np.sqrt(d * (1 - d) / reps)
        ess = f"ESS {(base['excess'] / x['excess']) ** 2:.2f}" if ok and x["excess"] > 0 else "**invalid**"
        return f"{miss:.3f}; {ess}" + (f"; pass {x['pass02']:.2f}" if with_pass and ok else "")

    A = [r for r in rows if r["part"] == "A"]
    arms = ["random + pooled Wilson (R)", "S2 + b1w (this paper)", "S2 + Wald-t b1", "S2 + StratPPI", "S2 + StratPPI, bootstrap-t",
            "random + PPI++ normal", "random + PPI++ bootstrap-t"]
    envs = sorted({(r["src"], r["env"]) for r in A})
    summary = collections.defaultdict(lambda: dict(miss=[], ess=[], cells=0, valid=0))
    for d in DELTAS:
        L += [f"## A. The reference rate as stratifier and predictor, delta {d}", "",
              "Largest miss over the label's checkpoints; ESS and pass at its last checkpoint.", "",
              "| label (rate) | n_s | " + " | ".join(arms) + " |", "|---|---|" + "---|" * len(arms)]
        for src, env in envs:
            for n in (100, 200):
                cell = [r for r in A if (r["src"], r["env"], r["n"], r["delta"]) == (src, env, n, d)]
                last = max(r["step"] for r in cell)
                base = next(r for r in cell if r["arm"] == arms[0] and r["step"] == last)
                out = []
                for arm in arms:
                    miss = max(r["miss"] for r in cell if r["arm"] == arm)
                    x = next(r for r in cell if r["arm"] == arm and r["step"] == last)
                    out.append(cellfmt(miss, base, x, d, reps_a))
                    if 0.05 <= base["truth"] <= 0.95:
                        s = summary[(arm, d)]; s["miss"].append(miss); s["cells"] += 1
                        if miss <= d + 2 * np.sqrt(d * (1 - d) / reps_a):
                            s["valid"] += 1; s["ess"].append((base["excess"] / x["excess"]) ** 2)
                L.append(f"| {env}{' pushed (014)' if src == '014' else ''} ({base['truth']:.2f}) | {n} | " + " | ".join(out) + " |")
        L.append("")
    L += ["## A, summary over the mid-rate labels (5-95%)", "", "| arm | delta | largest miss | cells valid | median ESS where valid |", "|---|---|---|---|---|"]
    for arm in arms:
        for d in DELTAS:
            s = summary[(arm, d)]
            L.append(f"| {arm} | {d} | {max(s['miss']):.3f} | {s['valid']} of {s['cells']} | "
                     f"{(np.median(s['ess']) if s['ess'] else float('nan')):.2f} |")
    B = [r for r in rows if r["part"] == "B"]
    arms = ["labels alone, Clopper-Pearson", "PPI++ normal", "PPI++ bootstrap-t (this paper)", "StratPPI, K=5", "StratPPI, K=10",
            "StratPPI, bootstrap-t, K=5", "StratPPI, bootstrap-t, K=10", "judge strata + b1w, K=5", "judge strata + b1w, K=10"]
    d = PL.DELTA
    L += ["", f"## B. A judge's logit as the predictor, delta {d}", "",
          "| pool | rate | n of N | " + " | ".join(arms) + " |", "|---|---|---|" + "---|" * len(arms)]
    worst = collections.defaultdict(float)
    for key in sorted({(r["env"], str(r["rate"]), r["n"], r["N"]) for r in B}):
        cell = {r["arm"]: r for r in B if (r["env"], str(r["rate"]), r["n"], r["N"]) == key}
        base = cell[arms[0]]
        for a_ in arms:
            worst[a_] = max(worst[a_], cell[a_]["miss"])
        L.append(f"| {key[0]} | {base['truth']:.3f} | {key[2]} of {key[3]} | " + " | ".join(
            cellfmt(cell[a_]["miss"], base, cell[a_], d, reps_b, with_pass=False) for a_ in arms) + " |")
    L += ["", "Largest miss over the cells of B: " + "; ".join(f"{a_} {worst[a_]:.3f}" for a_ in arms) + "."]
    open(path, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5000)
    ap.add_argument("--reps-b", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "stratppi.json")
    if not a.report_only:
        jobs_a = [("013", p, lab, s, a.reps) for p, labs in RP.LABELS.items() for lab in labs for s in (0, 100, 200)]
        jobs_a += [("014", "C1", "refusal", s, a.reps) for s in (100, 200)]
        pools = PL.load_pools()
        jobs_b, seed = [], 7000
        for key in (("refusal", "rubric", 0), ("refusal", "raw", 0)):
            for n, big_n, rate in ((100, 2000, None), (225, 2000, None), (500, 2000, None), (225, 4000, 0.05), (1000, 4000, 0.05),
                                   (225, 4000, 0.013), (1000, 4000, 0.013)):
                seed += 1
                jobs_b.append((key, *pools[key], n, big_n, a.reps_b, rate, seed))
        t0 = time.time(); rows = []
        with ProcessPoolExecutor(a.workers) as ex:
            for i, r in enumerate(ex.map(part_a, jobs_a)):
                rows += r; print(f"A {i + 1}/{len(jobs_a)} {time.time() - t0:.0f}s", flush=True)
            for i, r in enumerate(ex.map(part_b, jobs_b)):
                rows += r; print(f"B {i + 1}/{len(jobs_b)} {time.time() - t0:.0f}s", flush=True)
        json.dump(dict(reps=a.reps, reps_b=a.reps_b, rows=rows), open(path, "w"))
    res = json.load(open(path))
    report(res["rows"], os.path.join(OUT, "stratppi.md"), res["reps"], res["reps_b"])


if __name__ == "__main__":
    main()
