"""Does "StratPPI's published interval runs over its level at safety-test sizes" survive? (P14 check)

``scripts/stratppi_baseline.py`` measured StratPPI (Fisch et al., NeurIPS 2024, arXiv:2406.04291)
with our own implementation, proportional allocation and a predictor that is also the stratifier.
This script tests that finding five ways, on the baseline's pools and draw schemes (its random
streams are replayed, so its arms come out to the last digit and everything new sits beside them).

1. Is our implementation StratPPI? The authors released no code (their checklist: "Code may be made
   available at a future date"), so the comparison is with (a) Algorithm 1 written a second time in
   its general M-estimator form, (b) ``ppi_py``, the PPI++ authors' library, applied within strata
   and composed, (c) GLIDE's ``StratifiedPPIMeanEstimator``, a third party's StratPPI, and (d) the
   paper's own simulation (Figure 2), on identical inputs. The libraries run in a separate
   environment (``P14_PYTHON``); their plug-in conventions are then replayed here, checked to
   machine precision against them, and used as arms in every validity cell.
2. The paper's allocation of labels across strata: the oracle rule (Proposition 3) and the
   heuristic it runs on real data (Appendix B), against proportional.
3. PPBoot (Zrnic 2024, arXiv:2405.18379), basic and power-tuned, on the judge-logit cells.
4. A predictor that is not the stratifier: strata on reference samples 1-4, predictor 5-8.
5. Labels per stratum from 5 to 200, by the number of strata and the safety-set size.

Every cell is classified by one rule: with se = sqrt(delta (1 - delta) / R), "over" if the miss
rate exceeds delta + 2 se, "unresolved, above delta" if it lies in (delta, delta + 2 se], "at or
under" otherwise. Delta is 0.05, one-sided: the upper end of a published two-sided 90% interval.

The reference environment (kept off the project's own): ``ppi_py`` at commit 3d1f0c6 and GLIDE at a558db7
(0.11.0) cloned under /mnt/d/seldonian-runs/p14-src and installed with ``uv pip install`` into
/mnt/d/seldonian-runs/p14-venv. Only task 1 needs it.

    OMP_NUM_THREADS=1 .venv/bin/python scripts/stratppi_validate.py            # about 20 minutes, 5 processes
    .venv/bin/python scripts/stratppi_validate.py --task1-only            # the comparison of implementations alone
    .venv/bin/python scripts/stratppi_validate.py --report-only
    -> results/paper/stratppi_validate.json, results/paper/stratppi_validate.md
"""
import argparse
import collections
import json
import os
import subprocess
import sys
import tempfile
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "results", "paper")
REF_PY = os.environ.get("P14_PYTHON", "/mnt/d/seldonian-runs/p14-venv/bin/python")
DELTA = 0.05
Z = 1.6448536269514722               # norm.ppf(0.95)
NS, HS = (100, 200), (4, 8, 16)
KS_B = (5, 10, 20)
WORKER = "--reference-worker" in sys.argv
if not WORKER:
    sys.path.insert(0, HERE)
    import stratppi_baseline as BL    # noqa: E402
    c, PM, P14, PL, RP, SB = BL.c, BL.PM, BL.P14, BL.PL, BL.RP, BL.SB


# ------------------------------------------------------------------ StratPPI from sufficient statistics

CONV = {  # plug-in conventions the paper leaves open: cov ddof, var(f) sample, lam clipped, variance ddof
    "ours": dict(cd=1, pooled=False, clip=False, vd=1, nmin=3),      # stratppi_baseline.stratppi_point
    "ppi_py": dict(cd=0, pooled=True, clip=True, vd=0, nmin=0),      # ppi_py.ppi_mean_ci within a stratum
    "glide": dict(cd=1, pooled=True, clip=False, vd=1, nmin=0),      # glide.engines.ppi_core
}


def suff(y, f, st, H):
    """Per stratum: n, mean y, mean f and the centred sums of squares and products of the labels."""
    n = np.bincount(st, minlength=H).astype(float)
    yb, fb = np.bincount(st, y, H) / n, np.bincount(st, f, H) / n
    return (n, yb, fb, np.maximum(np.bincount(st, y * y, H) - n * yb ** 2, 0.0),
            np.maximum(np.bincount(st, f * f, H) - n * fb ** 2, 0.0), np.bincount(st, y * f, H) - n * yb * fb)


def suff_rows(y, f):
    """The same for one stratum per row of (reps, n) arrays."""
    n = np.full(y.shape[0], float(y.shape[1]))
    yb, fb = y.mean(1), f.mean(1)
    yc, fc = y - yb[:, None], f - fb[:, None]
    return n, yb, fb, (yc * yc).sum(1), (fc * fc).sum(1), (yc * fc).sum(1)


def spp(S, U, W, conv="ours"):
    """StratPPI's estimate and plug-in variance (Algorithm 1 for a mean). ``S`` from ``suff``; ``U`` =
    (N, mean f, centred sum of squares of f) of each stratum's unlabelled draws; arrays (..., H)."""
    n, yb, fb, syy, sff, syf = S
    N, fu, ssu = U
    k = CONV[conv]
    vf = (sff + ssu + n * N / (n + N) * (fb - fu) ** 2) / (n + N - 1) if k["pooled"] else ssu / (N - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = np.where((vf > 1e-12) & (n >= k["nmin"]), syf / (n - k["cd"]) / ((1 + n / N) * vf), 0.0)
    if k["clip"]:
        lam = np.clip(lam, 0.0, 1.0)
    est = (W * (yb + lam * (fu - fb))).sum(-1)
    var = (W ** 2 * (np.maximum(syy - 2 * lam * syf + lam ** 2 * sff, 0.0) / (n - k["vd"]) / n
                     + lam ** 2 * ssu / (N - k["vd"]) / N)).sum(-1)
    return est, var


def algorithm1(strata, W):
    """Algorithm 1 of the paper a second time, in its M-estimator form for l(y; theta) = (y - theta)^2 / 2:
    the minimiser of the stratified loss from its gradient, the Hessian, the two covariances per
    stratum (the theorem's sign: grad l(Y) - lam grad l(f)), lam from Example 1 with var(f) on the
    unlabelled draws. Shares no code with ``stratppi_point``."""
    lam = []
    for y, f, fu in strata:
        vf = np.sum((fu - fu.mean()) ** 2) / (len(fu) - 1)
        cyf = np.sum((y - y.mean()) * (f - f.mean())) / (len(y) - 1)
        lam.append(cyf / ((1 + len(y) / len(fu)) * vf) if vf > 0 and len(y) >= 3 else 0.0)

    def grad(theta):          # d/d theta of sum_k w_k L_k^PP(theta)
        return sum(w * (l * np.mean(theta - fu) + np.mean((theta - y) - l * (theta - f)))
                   for w, l, (y, f, fu) in zip(W, lam, strata))
    g0, g1 = grad(0.0), grad(1.0)
    theta = -g0 / (g1 - g0)                                    # the loss is quadratic: one secant step
    A = sum(w * 1.0 for w in W)                                # Hessian of the mean loss
    sig = 0.0
    for w, l, (y, f, fu) in zip(W, lam, strata):
        gf = l * (theta - fu)
        gd = (theta - y) - l * (theta - f)
        vf = np.sum((gf - gf.mean()) ** 2) / (len(fu) - 1)
        vd = np.sum((gd - gd.mean()) ** 2) / (len(y) - 1)
        sig += w ** 2 * (vf / len(fu) + vd / len(y)) / A ** 2
    return theta, sig


def allocate(rho, n, cap=None, floor=2):
    """Integer labels per stratum in shares ``rho``, summing to n (largest remainders), each at least
    ``floor`` (a variance needs two; the paper gives no floor) and at most ``cap``."""
    rho = np.asarray(rho, dtype=float)
    K = len(rho)
    rho = rho / rho.sum() if rho.sum() > 0 else np.full(K, 1.0 / K)
    cap = np.full(K, n) if cap is None else np.asarray(cap, dtype=int)
    nk = np.clip(np.floor(n * rho).astype(int), floor, cap)
    while nk.sum() != n:
        gap = n * rho - nk
        if nk.sum() < n:
            ok = np.flatnonzero(nk < cap)
            nk[ok[np.argmax(gap[ok])]] += 1
        else:
            ok = np.flatnonzero(nk > floor)
            nk[ok[np.argmin(gap[ok])]] -= 1
    return nk


def ppboot(y, f, fu, rng, B=1000, B_lam=50):
    """PPBoot (Zrnic 2024, Algorithm 1) and its power-tuned form for a mean, percentile limits; B and
    the 50 resamples for lam are ``ppi_py.ppboot``'s defaults. Returns {name: (upper, lower, estimate)}
    at one-sided ``DELTA``. The unlabelled resample's mean is drawn through the counts of its distinct
    values, which is the same distribution as resampling the values."""
    n, nu = len(y), len(fu)
    v, cnt = np.unique(fu, return_counts=True)

    def means(b):
        idx = rng.integers(0, n, size=(b, n))
        return y[idx].mean(1), f[idx].mean(1), rng.multinomial(nu, cnt / nu, size=b) @ v / nu
    ym, fm, um = means(B_lam)
    den = fm.var() + um.var()
    lam = float(np.cov(ym, fm)[0, 1] / den) if den > 0 else 0.0
    ym, fm, um = means(B)
    out = {}
    for name, l in (("PPBoot (lam = 1)", 1.0), ("PPBoot (power-tuned)", lam)):
        th = l * um + ym - l * fm
        out[name] = (float(np.quantile(th, 1 - DELTA)), float(np.quantile(th, DELTA)),
                     float(l * fu.mean() + y.mean() - l * f.mean()))
    return out


def summ(ub, est, truth, lo=None):
    ub, est = np.asarray(ub, dtype=float), np.asarray(est, dtype=float)
    # a NaN bound counts as a miss (ub < truth is False for NaN)
    out = dict(miss=float((~(ub >= truth)).mean()), excess=float(ub.mean() - truth), width=float((ub - est).mean()))
    if lo is not None:
        out["miss_lo"] = float((np.asarray(lo) > truth).mean())
    return out


def verdict(miss, reps, d=DELTA):
    se = np.sqrt(d * (1 - d) / reps)
    return "over" if miss > d + 2 * se else ("unresolved" if miss > d else "under")


# ------------------------------------------------------------------ part A: reference-rate strata

def job_a(job):
    src, pool, lab, step, reps = job
    PM.TIES = "random"
    by, meta = RP.load() if src == "013" else P14.load("s0")
    P, _ = RP.build_pool(by, meta, pool, lab, step, False, "eval")
    cov, draw, truth = P["cov"], P["draw"], float(P["truth"])
    p_i = draw.__defaults__[0].mean(axis=1)        # each prompt's rate in the half the labels are drawn from
    N = len(p_i)
    preds = {"ref 1-8": cov[:, :8].mean(axis=1), "ref 1-4": cov[:, :4].mean(axis=1), "ref 5-8": cov[:, 4:8].mean(axis=1)}
    designs = [("ref 1-8", H, n_s, "prop", ("ref 1-8",)) for H in HS for n_s in NS]
    designs += [("ref 1-8", 8, n_s, a, ("ref 1-8",)) for n_s in NS for a in ("heur", "heur10", "opt")]
    designs += [("ref 1-4", 8, n_s, "prop", ("ref 1-4", "ref 5-8")) for n_s in NS]
    rows, check = [], 0.0
    base = dict(part="A", src=src, env=f"{pool}:{lab}", step=step, truth=truth, reps=reps)
    for n_s in NS:                                 # the baseline's arm R on its own draw: the ESS denominator
        ub, est = [], []
        for r in range(reps):
            rr = np.random.default_rng([step, r, n_s])
            y = draw(rr.choice(N, size=n_s, replace=False), rr)
            ub.append(SB.b1w(y.sum(), n_s, 1.0, DELTA)); est.append(y.mean())
        rows.append(dict(base, n=n_s, H=1, strat="-", pred="-", alloc="-", arm="random + pooled Wilson (R)", **summ(ub, est, truth)))
    for strat, H, n_s, alloc, pred_list in designs:
        fs = preds[strat]
        rng = np.random.default_rng([step, zlib.crc32(repr(("S2", 8 if strat == "ref 1-8" else 4, H)).encode())])
        strata = PM.quantile_strata(fs, H, rng, ties="random")       # at 8 samples and H = 8: 013's S2 strata
        mem = [np.flatnonzero(strata == h) for h in range(H)]
        Nh = np.array([len(m) for m in mem], dtype=float)
        W = Nh / N
        nk, tag = None, {"heur": 1, "opt": 2, "heur10": 3}.get(alloc)
        if alloc in ("heur", "heur10"):       # Appendix B: sigma_k^2 = mean c (1 - c) + var c, c the autorater's confidence
            nk = allocate(W * np.sqrt([np.mean(fs[m] * (1 - fs[m])) + np.var(fs[m]) for m in mem]), n_s, cap=Nh,
                          floor=2 if alloc == "heur" else 10)
        if alloc == "opt":        # Proposition 3 with the pool's own moments (the oracle)
            sig = []
            for m in mem:
                vf = np.var(fs[m])
                lam = np.cov(p_i[m], fs[m], ddof=0)[0, 1] / vf / (1 + n_s / N) if vf > 0 else 0.0
                sig.append(np.sqrt(np.mean(p_i[m] * (1 - p_i[m])) + np.var(p_i[m] - lam * fs[m])))
            nk = allocate(W * np.array(sig), n_s, cap=Nh)
        whole = {pr: (Nh, np.array([preds[pr][m].mean() for m in mem]),
                      np.array([np.sum((preds[pr][m] - preds[pr][m].mean()) ** 2) for m in mem])) for pr in pred_list}
        stats = {pr: [(int(Nh[h]), float(whole[pr][1][h]), float(whole[pr][2][h] / (Nh[h] - 1))) for h in range(H)]
                 for pr in pred_list}
        acc = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in range(reps):
            if alloc == "prop":
                rr = np.random.default_rng([step, r, n_s])             # the baseline's draw
                ids, st = PM._draw_split(strata, n_s, rr)
            else:
                rr = np.random.default_rng([step, r, n_s, tag])
                ids = np.concatenate([rr.choice(mem[h], size=nk[h], replace=False) for h in range(H)])
                st = np.repeat(np.arange(H), nk)
            y = draw(ids, rr)
            s, n = np.bincount(st, y, H), np.bincount(st, minlength=H)
            e0 = SB.estimate(s, n, W)
            for arm, ub in (("b1w", SB.b1w(s, n, W, DELTA)), ("Wald-t b1", SB.b1(s, n, W, DELTA))):
                acc[("-", arm)]["ub"].append(ub); acc[("-", arm)]["est"].append(e0)
            for pr in pred_list:
                fl = preds[pr][ids]
                (ub,), est = BL.stratppi_boot(y, fl, st, stats[pr], W, (DELTA,), rr)
                acc[(pr, "StratPPI estimator, bootstrap-t")]["ub"].append(ub)
                acc[(pr, "StratPPI estimator, bootstrap-t")]["est"].append(est)
                S = suff(y, fl, st, H)
                acc[(pr, "S")]["S"].append(S)
                if r < 100:                # the vectorised arm is the baseline's function
                    e1, v1 = BL.stratppi_point(y, fl, st, stats[pr], W)
                    e2, v2 = spp(S, whole[pr], W)
                    check = max(check, abs(e1 - e2), abs(v1 - v2) / max(v1, 1e-12))
        tagr = dict(base, n=n_s, H=H, strat=strat, alloc=alloc, nk=[] if nk is None else [int(a) for a in nk])
        for (pr, arm), v in acc.items():
            if arm != "S":
                rows.append(dict(tagr, pred=pr, arm=arm, **summ(v["ub"], v["est"], truth)))
                continue
            S = tuple(np.array(a) for a in zip(*v["S"]))
            U = whole[pr]
            variants = [("StratPPI as published", U, "ours")]
            if alloc == "prop":            # the paper's protocol: the unlabelled set is the rest of the stratum
                n, fb, sff = S[0], S[2], S[4]
                Nr = U[0] - n
                fr = (U[0] * U[1] - n * fb) / Nr
                ssr = np.maximum(U[2] - sff - n * (fb - U[1]) ** 2 - Nr * (fr - U[1]) ** 2, 0.0)
                variants += [(f"StratPPI as published, unlabelled = rest of stratum ({cv})", (Nr, fr, ssr), cv)
                             for cv in ("ours", "ppi_py", "glide")]
            for arm, Uv, cv in variants:
                est, var = spp(S, Uv, W, cv)
                se = np.sqrt(var)
                rows.append(dict(tagr, pred=pr, arm=arm, **summ(est + Z * se, est, truth, est - Z * se)))
    return rows, check


# ------------------------------------------------------------------ part B: judge-logit strata

def job_b(job):
    key, y_pool, p_pool, n, big_n, reps, rate, seed = job
    rng = np.random.default_rng(seed)              # the baseline's stream, consumed in the baseline's order
    rng2 = np.random.default_rng([seed, 2])        # everything this script adds
    truth = float(y_pool.mean()) if rate is None else rate
    d = DELTA
    x_pool = PL.logit01(p_pool)
    M = len(y_pool)
    q = np.full(M, 1.0 / M) if rate is None else np.where(
        y_pool == 1, rate / (y_pool == 1).sum(), (1 - rate) / (y_pool == 0).sum())
    order = np.lexsort((rng.random(M), x_pool))
    cum = np.cumsum(q[order]) - q[order] / 2
    G = {}
    for K in KS_B:
        st_pool = np.empty(M, dtype=int); st_pool[order] = np.minimum((cum * K).astype(int), K - 1)
        W = np.array([q[st_pool == h].sum() for h in range(K)])
        members = [np.flatnonzero(st_pool == h) for h in range(K)]
        probs = [q[m] / q[m].sum() for m in members]
        alloc = {"prop": np.maximum(2, np.rint(n * W).astype(int))}                  # the baseline's rounding
        if K != 20:
            sh, so = [], []
            for m, w in zip(members, probs):
                pp, xx, yy = p_pool[m], x_pool[m], y_pool[m]
                sh.append(np.sqrt(w @ (pp * (1 - pp)) + w @ pp ** 2 - (w @ pp) ** 2))     # Appendix B, c = P(Yes)
                vx = w @ xx ** 2 - (w @ xx) ** 2
                lam = (w @ (xx * yy) - (w @ xx) * (w @ yy)) / vx / (1 + n / (big_n - n)) if vx > 1e-12 else 0.0
                res = yy - lam * xx
                so.append(np.sqrt(max(w @ res ** 2 - (w @ res) ** 2, 0.0)))               # Proposition 3, oracle
            alloc["heur"], alloc["opt"] = allocate(W * np.array(sh), n), allocate(W * np.array(so), n)
            alloc["heur10"] = allocate(W * np.array(sh), n, floor=min(10, n // K))
        G[K] = dict(st=st_pool, W=W, members=members, probs=probs, alloc=alloc,
                    nu=np.maximum(2, np.rint((big_n - n) * W).astype(int)))
    acc = collections.defaultdict(lambda: collections.defaultdict(list))

    def add(k, ub, est, lo=None):
        acc[k]["ub"].append(np.atleast_1d(ub)); acc[k]["est"].append(np.atleast_1d(est))
        if lo is not None:
            acc[k]["lo"].append(np.atleast_1d(lo))

    def strat_arms(K, al, ids, nk, stats, U, r_):
        g = G[K]
        st = np.repeat(np.arange(K), nk)
        yy, xx = y_pool[ids], x_pool[ids]
        (ub,), est = BL.stratppi_boot(yy, xx, st, stats, g["W"], (d,), r_)
        add((K, al, "logit", "StratPPI estimator, bootstrap-t"), ub, est)
        s = np.bincount(st, yy, K)
        add((K, al, "-", "b1w"), SB.b1w(s, nk, g["W"], d), SB.estimate(s, nk, g["W"]))
        acc[(K, al, "logit", "S")]["S"].append(suff(yy, xx, st, K) + U["logit"])
        if al == "prop":
            acc[(K, al, "p", "S")]["S"].append(suff(yy, p_pool[ids], st, K) + U["p"])

    done = 0
    while done < reps:
        b = min(500, reps - done)
        y, p = PL.sample(rng, y_pool, p_pool, b, big_n, rate)
        x = PL.logit01(p)
        yl, xl, xu, pl, pu = y[:, :n], x[:, :n], x[:, n:], p[:, :n], p[:, n:]
        e2, v2, _ = c.ppi_point(yl, xl, xu, None)
        add((0, "-", "-", "labels alone, Clopper-Pearson"), c.classical(yl, d), yl.mean(1))
        add((0, "-", "logit", "PPI++ normal"), c.ppipp_clt(yl, xl, xu, d), e2, e2 - Z * np.sqrt(v2))
        add((0, "-", "logit", "PPI++ bootstrap-t"), c.ppipp_boot(yl, xl, xu, d, seed=seed + done), e2)
        nu_ = np.full(b, float(xu.shape[1]))
        for pr, fl, fu in (("logit", xl, xu), ("p", pl, pu)):
            S = tuple(a[:, None] for a in suff_rows(yl, fl))
            U = tuple(a[:, None] for a in (nu_, fu.mean(1), fu.var(axis=1) * nu_))
            for cv in ("ours", "ppi_py") if pr == "logit" else ("ours",):
                e, v = spp(S, U, np.ones(1), cv)
                add((0, "-", pr, f"PPI++ normal ({cv})"), e + Z * np.sqrt(v), e, e - Z * np.sqrt(v))
        for r in range(b):
            for name, (ub, lo, est) in ppboot(yl[r], xl[r], xu[r], rng2).items():
                add((0, "-", "logit", name), ub, est, lo)
        for K in KS_B:
            g = G[K]
            r_ = rng if K != 20 else rng2          # K = 5 and 10 replay the baseline; K = 20 is new
            for r in range(b):
                ids = np.concatenate([r_.choice(g["members"][h], size=g["alloc"]["prop"][h], p=g["probs"][h]) for h in range(K)])
                uid = [r_.choice(g["members"][h], size=g["nu"][h], p=g["probs"][h]) for h in range(K)]
                stats = [(int(g["nu"][h]), float(x_pool[u].mean()), float(x_pool[u].var(ddof=1))) for h, u in enumerate(uid)]
                U = {pr: (g["nu"].astype(float), np.array([v[u].mean() for u in uid]),
                          np.array([np.sum((v[u] - v[u].mean()) ** 2) for u in uid]))
                     for pr, v in (("logit", x_pool), ("p", p_pool))}
                strat_arms(K, "prop", ids, g["alloc"]["prop"], stats, U, r_)
                for al in ("heur", "heur10", "opt") if K != 20 else ():
                    nk = g["alloc"][al]
                    ids = np.concatenate([rng2.choice(g["members"][h], size=nk[h], p=g["probs"][h]) for h in range(K)])
                    strat_arms(K, al, ids, nk, stats, U, rng2)
        done += b
    rows = []
    base = dict(part="B", env="|".join(map(str, key)), n=n, N=big_n, truth=truth, reps=reps,
                rate="pool" if rate is None else rate)
    for (K, al, pr, arm), v in acc.items():
        tagr = dict(base, K=K, alloc=al, pred=pr, nk=[int(a) for a in G[K]["alloc"][al]] if K else [])
        if arm != "S":
            lo = np.concatenate(v["lo"]) if v["lo"] else None
            rows.append(dict(tagr, arm=arm, **summ(np.concatenate(v["ub"]), np.concatenate(v["est"]), truth, lo)))
            continue
        A = tuple(np.array(a) for a in zip(*v["S"]))
        for cv in ("ours", "ppi_py", "glide") if (pr, al) == ("logit", "prop") else ("ours",):
            est, var = spp(A[:6], A[6:], G[K]["W"], cv)
            se = np.sqrt(var)
            rows.append(dict(tagr, arm="StratPPI as published" + ("" if cv == "ours" else f" ({cv})"),
                             **summ(est + Z * se, est, truth, est - Z * se)))
    return rows


# ------------------------------------------------------------------ task 1: identical inputs, other people's code

def t1_cases():
    """Labelled and unlabelled draws per stratum for the comparison: the paper's two-stratum Gaussian
    design, a binary five-stratum design, and draws from one plasmode cell of each part."""
    cases = []
    for seed in range(5):
        rng = np.random.default_rng([14, seed])
        st = []
        for sg in (0.25, 4.0):
            yy, yu = rng.standard_normal(50), rng.standard_normal(5000)
            st.append((yy, yy + sg * rng.standard_normal(50), yu + sg * rng.standard_normal(5000)))
        cases.append(dict(name=f"Gaussian, K = 2 (the paper's section 5.1), seed {seed}", W=[0.5, 0.5], strata=st, kind="synthetic"))
    for seed in range(5):
        rng = np.random.default_rng([15, seed])
        st = []
        for rate in (0.02, 0.05, 0.1, 0.3, 0.7):
            pp = np.clip(rate + 0.15 * rng.standard_normal(420), 0.001, 0.999)
            yy = (rng.random(420) < np.clip(pp + 0.1, 0, 1)).astype(float)
            st.append((yy[:20], pp[:20], pp[20:]))
        cases.append(dict(name=f"binary, K = 5, 20 labels per stratum, seed {seed}", W=[0.2] * 5, strata=st, kind="synthetic"))
    PM.TIES = "random"
    by, meta = RP.load()
    P, _ = RP.build_pool(by, meta, "C1", "refusal", 200, False, "eval")
    f = P["cov"][:, :8].mean(axis=1)
    strata = PM.quantile_strata(f, 8, np.random.default_rng([200, zlib.crc32(repr(("S2", 8, 8)).encode())]), ties="random")
    for r in range(5):
        rr = np.random.default_rng([200, r, 200])
        ids, st_ = PM._draw_split(strata, 200, rr)
        y = P["draw"](ids, rr)
        cases.append(dict(name=f"plasmode A: C1:refusal, step 200, n_s 200, draw {r}", W=(np.bincount(strata) / len(f)).tolist(),
                          strata=[(y[st_ == h], f[ids][st_ == h], f[strata == h]) for h in range(8)], kind="plasmode"))
    y_pool, p_pool = PL.load_pools()[("refusal", "rubric", 0)]
    x_pool = PL.logit01(p_pool)
    rng = np.random.default_rng(7002)
    order = np.lexsort((rng.random(len(y_pool)), x_pool))
    st_pool = np.empty(len(y_pool), dtype=int); st_pool[order] = np.arange(len(y_pool)) * 10 // len(y_pool)
    for r in range(5):
        st = []
        for h in range(10):
            m = np.flatnonzero(st_pool == h)
            i, u = rng.choice(m, size=22), rng.choice(m, size=178)
            st.append((y_pool[i], x_pool[i], x_pool[u]))
        cases.append(dict(name=f"plasmode B: refusal|rubric|0, n 220 of 2,000, K 10, draw {r}", W=[0.1] * 10, strata=st, kind="plasmode"))
    return cases


def reference_worker(path_in, path_out):
    """Runs under ``P14_PYTHON``: ppi_py and GLIDE on the cases, nothing of this project imported."""
    import warnings
    warnings.simplefilter("ignore")
    from glide.engines.ppi_core import _compute_mean_estimate, _compute_std_estimate, _compute_tuning_parameter
    from glide.estimators import StratifiedPPIMeanEstimator
    from ppi_py import ppboot as pp_boot
    from ppi_py import ppi_mean_ci, ppi_mean_pointestimate
    job = json.load(open(path_in))
    out = []
    for case in job["cases"]:
        W = np.array(case["W"])
        res = dict(name=case["name"], undefined=0)
        acc = collections.defaultdict(lambda: [0.0, 0.0])
        for w, (y, f, fu), lam_ours in zip(W, case["strata"], case["lam_ours"]):
            y, f, fu = np.array(y), np.array(f), np.array(fu)
            flat = np.var(np.concatenate([f, fu])) == 0           # ppi_py divides by zero, GLIDE raises
            res["undefined"] += int(flat)
            for name, lam in (("ppi_py", 0.0 if flat else None), ("ppi_py, our lam", lam_ours)):
                e = float(np.ravel(ppi_mean_pointestimate(y, f, fu, lam=lam))[0])
                lo, hi = ppi_mean_ci(y, f, fu, alpha=0.1, lam=lam)
                acc[name][0] += w * e
                acc[name][1] += w ** 2 * (float(np.ravel(hi)[0] - np.ravel(lo)[0]) / (2 * Z)) ** 2
            lam = 0.0 if flat else _compute_tuning_parameter(y, f, fu, power_tuning=True)
            acc["glide"][0] += w * _compute_mean_estimate(y, f, fu, lam)
            acc["glide"][1] += w ** 2 * _compute_std_estimate(y, f, fu, lam) ** 2
        res.update({k: [float(v[0]), float(v[1])] for k, v in acc.items()})
        try:                         # the public estimator: weights are its own (sample shares)
            yt = np.concatenate([np.r_[y, np.full(len(fu), np.nan)] for y, f, fu in case["strata"]])
            yp = np.concatenate([np.r_[f, fu] for y, f, fu in case["strata"]])
            gr = np.concatenate([np.full(len(y) + len(fu), h) for h, (y, f, fu) in enumerate(case["strata"])])
            r = StratifiedPPIMeanEstimator().estimate(yt, yp, gr, confidence_level=0.90)
            res["glide_full"] = [float(r.mean), float(r.std) ** 2]
        except ValueError as e:
            res["glide_full"] = str(e)
        y, f, fu = (np.concatenate([np.array(s[i]) for s in case["strata"]]) for i in range(3))     # unstratified
        lo, hi = ppi_mean_ci(y, f, fu, alpha=0.1)
        res["pp_ppi_py"] = [float(np.ravel(ppi_mean_pointestimate(y, f, fu))[0]), (float(np.ravel(hi)[0] - np.ravel(lo)[0]) / (2 * Z)) ** 2]
        lam = _compute_tuning_parameter(y, f, fu, power_tuning=True)
        res["pp_glide"] = [float(_compute_mean_estimate(y, f, fu, lam)), float(_compute_std_estimate(y, f, fu, lam)) ** 2]
        out.append(res)
    np.random.seed(0)
    boots = []
    for case in job["boot"]:
        y, f, fu = (np.array(case[k]) for k in ("y", "f", "fu"))
        one = pp_boot(np.mean, y, f, fu, lam=1, n_resamples=job["B_big"], alpha=0.1)
        tuned = [float(pp_boot(np.mean, y, f, fu, n_resamples=1000, alpha=0.1)[1]) for _ in range(job["tuned_runs"])]
        boots.append(dict(one=float(one[1]), tuned_mean=float(np.mean(tuned)), tuned_sd=float(np.std(tuned))))
    json.dump(dict(cases=out, boot=boots), open(path_out, "w"))


def paper_sim(trials=2000):
    """The paper's simulation (section 5.1, Figure 2): Y ~ N(0, 1), f = Y + mu_k + sigma_k eps, two equal
    strata, N = 10,000, a two-sided 90% interval. ``fig`` is the mean width read off the figure."""
    fig = {("homogeneous", "prop"): (0.235, 0.167, 0.077), ("different variance", "prop"): (0.235, 0.167, 0.077),
           ("different variance", "opt"): (0.222, 0.152, 0.068)}
    out = []
    W = np.array([0.5, 0.5])
    for (scn, al), widths in fig.items():
        sg = np.array([1.0, 1.0] if scn == "homogeneous" else [0.25, 4.0])
        for n, w_fig in zip((100, 200, 1000), widths):
            rng = np.random.default_rng([16, n, zlib.crc32((scn + al).encode())])
            nk = np.array([n // 2, n // 2]) if al == "prop" else allocate(W * sg / np.sqrt(1 + sg ** 2), n)
            S, U = [], []
            for k in range(2):
                y, yu = rng.standard_normal((trials, nk[k])), rng.standard_normal((trials, 5000))
                fu = yu + sg[k] * rng.standard_normal((trials, 5000))
                S.append(suff_rows(y, y + sg[k] * rng.standard_normal((trials, nk[k]))))
                U.append((np.full(trials, 5000.0), fu.mean(1), fu.var(axis=1) * 5000))
            S, U = tuple(np.stack(a, 1) for a in zip(*S)), tuple(np.stack(a, 1) for a in zip(*U))
            est, var = spp(S, U, W)
            out.append(dict(scenario=scn, alloc=al, n=n, nk=nk.tolist(), width=float((2 * Z * np.sqrt(var)).mean()), width_fig=w_fig,
                            width_ppi_py=float((2 * Z * np.sqrt(spp(S, U, W, "ppi_py")[1])).mean()),
                            coverage=float((np.abs(est) <= Z * np.sqrt(var)).mean()), trials=trials))
    return out


def task1():
    cases = t1_cases()
    ours, rep = [], collections.defaultdict(lambda: dict(est=0.0, var=0.0, ub=0.0, est_p=0.0, var_p=0.0, ub_p=0.0))
    for case in cases:
        W = np.array(case["W"])
        y, f, st = (np.concatenate(a) for a in ([s[0] for s in case["strata"]], [s[1] for s in case["strata"]],
                                                [np.full(len(s[0]), h) for h, s in enumerate(case["strata"])]))
        stats = [(len(s[2]), float(s[2].mean()), float(s[2].var(ddof=1))) for s in case["strata"]]
        est, var = BL.stratppi_point(y, f, st, stats, W)
        (ub,), _ = BL.stratppi(y, f, st, stats, W, (DELTA,))
        case["lam_ours"] = [float(np.cov(s[0], s[1], ddof=1)[0, 1]) / ((1 + len(s[0]) / len(s[2])) * s[2].var(ddof=1))
                            if len(s[0]) >= 3 and s[2].var(ddof=1) > 0 else 0.0 for s in case["strata"]]
        S = suff(y, f, st, len(W))
        U = (np.array([float(len(s[2])) for s in case["strata"]]), np.array([s[2].mean() for s in case["strata"]]),
             np.array([np.sum((s[2] - s[2].mean()) ** 2) for s in case["strata"]]))
        yy, ff, fu = (np.concatenate([s[i] for s in case["strata"]]) for i in range(3))
        e_pp, v_pp, _ = c.ppi_point(yy, ff, fu, None)
        Sp = tuple(np.atleast_1d(a) for a in suff(yy, ff, np.zeros(len(yy), dtype=int), 1))
        Up = (np.array([float(len(fu))]), np.array([fu.mean()]), np.array([np.sum((fu - fu.mean()) ** 2)]))
        ours.append(dict(est=est, var=var, ub=ub, alg1=algorithm1(case["strata"], W),
                         conv={cv: spp(S, U, W, cv) for cv in CONV}, pp=(float(e_pp[0]), float(v_pp[0])),
                         pp_conv={cv: spp(Sp, Up, np.ones(1), cv) for cv in ("ppi_py", "glide")}))
    y_pool, p_pool = PL.load_pools()[("refusal", "rubric", 0)]
    rng = np.random.default_rng(7102)
    boot = []
    for n in (100, 225):
        i = rng.integers(0, len(y_pool), size=2000)
        boot.append(dict(y=y_pool[i[:n]], f=PL.logit01(p_pool[i[:n]]), fu=PL.logit01(p_pool[i[n:]])))
    job = dict(cases=[dict(name=k["name"], W=k["W"], lam_ours=k["lam_ours"], strata=[[a.tolist() for a in s] for s in k["strata"]])
                      for k in cases], boot=[{k: v.tolist() for k, v in b.items()} for b in boot], B_big=20000, tuned_runs=100)
    with tempfile.TemporaryDirectory() as tmp:
        json.dump(job, open(os.path.join(tmp, "in.json"), "w"))
        subprocess.run([REF_PY, os.path.abspath(__file__), "--reference-worker", os.path.join(tmp, "in.json"),
                        os.path.join(tmp, "out.json")], check=True)
        ref = json.load(open(os.path.join(tmp, "out.json")))

    def cmp(name, kind, a, b):               # a, b = (est, var); upper limits at z(0.95)
        r = rep[name]
        ua, ub_ = a[0] + Z * np.sqrt(a[1]), b[0] + Z * np.sqrt(b[1])
        for sfx in ("", "_p") if kind == "plasmode" else ("",):
            r["est" + sfx] = max(r["est" + sfx], abs(a[0] - b[0]))
            r["var" + sfx] = max(r["var" + sfx], abs(a[1] - b[1]) / b[1])
            r["ub" + sfx] = max(r["ub" + sfx], abs(ua - ub_))
    undefined, full_err = collections.Counter(), 0
    for case, o, r in zip(cases, ours, ref["cases"]):
        me, k = (o["est"], o["var"]), case["kind"]
        undefined[case["name"][9] if k == "plasmode" else "synthetic"] += r["undefined"]
        cmp("ours vs Algorithm 1 rewritten (M-estimator form, same plug-ins)", k, me, o["alg1"])
        cmp("ours vs vectorised arm used in the cells (`spp`, ours)", k, me, o["conv"]["ours"])
        cmp("ours vs ppi_py within strata, composed", k, me, r["ppi_py"])
        cmp("ours vs ppi_py within strata, our lam passed in", k, me, r["ppi_py, our lam"])
        cmp("ours vs GLIDE core functions, known weights", k, me, r["glide"])
        if isinstance(r["glide_full"], list):
            cmp("ours vs GLIDE `StratifiedPPIMeanEstimator` (its own weights)", k, me, r["glide_full"])
        else:
            full_err += 1
        cmp("replayed ppi_py conventions (`spp`, ppi_py) vs ppi_py", k, o["conv"]["ppi_py"], r["ppi_py"])
        cmp("replayed GLIDE conventions (`spp`, glide) vs GLIDE core", k, o["conv"]["glide"], r["glide"])
        cmp("unstratified PPI++: cert017.ppi_point vs ppi_py.ppi_mean_ci", k, o["pp"], r["pp_ppi_py"])
        cmp("unstratified PPI++: cert017.ppi_point vs GLIDE core", k, o["pp"], r["pp_glide"])
        cmp("unstratified PPI++: replayed ppi_py conventions vs ppi_py", k, o["pp_conv"]["ppi_py"], r["pp_ppi_py"])
    pb = []
    for b, r in zip(boot, ref["boot"]):
        rng = np.random.default_rng(len(b["y"]))
        one = ppboot(b["y"], b["f"], b["fu"], rng, B=20000)["PPBoot (lam = 1)"]
        tuned = [ppboot(b["y"], b["f"], b["fu"], rng)["PPBoot (power-tuned)"][0] for _ in range(100)]
        est, var, _ = c.ppi_point(b["y"], b["f"], b["fu"], 1.0)
        pb.append(dict(n=len(b["y"]), se=float(np.sqrt(var[0])), one=one[0], one_ref=r["one"], tuned_mean=float(np.mean(tuned)),
                       tuned_sd=float(np.std(tuned)), tuned_mean_ref=r["tuned_mean"], tuned_sd_ref=r["tuned_sd"]))
    ties = dict(A={}, B={})                    # strata in which the predictor takes one value
    by, meta = RP.load()
    for pool, lab in (("C1", "refusal"), ("C2", "unsafe"), ("C2", "refusal"), ("C3", "refusal")):
        f = RP.build_pool(by, meta, pool, lab, 200, False, "eval")[0]["cov"][:, :8].mean(axis=1)
        st = PM.quantile_strata(f, 8, np.random.default_rng([200, zlib.crc32(repr(("S2", 8, 8)).encode())]), ties="random")
        ties["A"][f"{pool}:{lab}"] = sum(len(np.unique(f[st == h])) == 1 for h in range(8))
    for (key, seed) in ((("refusal", "rubric", 0), 7001), (("refusal", "raw", 0), 7008)):       # the pool-rate cells
        y_pool, p_pool = PL.load_pools()[key]
        x = PL.logit01(p_pool)
        order = np.lexsort((np.random.default_rng(seed).random(len(x)), x))
        st = np.empty(len(x), dtype=int); st[order] = np.arange(len(x)) * 10 // len(x)
        ties["B"][key[1]] = dict(constant=sum(len(np.unique(x[st == h])) == 1 for h in range(10)), zero=float((p_pool == 0).mean()))
    return dict(compare=rep, undefined={k: undefined[k] for k in ("A", "B", "synthetic")}, glide_full_refused=full_err, n_cases=len(cases), ties=ties,
                names=[k["name"] for k in cases], ppboot=pb, sim=paper_sim())


# ------------------------------------------------------------------ report

def counts(cells):
    k = collections.Counter(verdict(r["miss"], r["reps"]) for r in cells)
    return k["over"], k["unresolved"], k["under"]


def fmt_counts(cells):
    return "%d / %d / %d" % counts(cells)


def med(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return f"{np.median(xs):.2f}" if xs else "-"


def report(res, path):
    rows, t1, reps_a, reps_b = res["rows"], res["task1"], res["reps_a"], res["reps_b"]
    A = [r for r in rows if r["part"] == "A"]
    B = [r for r in rows if r["part"] == "B"]
    mid = lambda r: 0.05 <= r["truth"] <= 0.95                                    # noqa: E731
    se_a, se_b = (np.sqrt(DELTA * (1 - DELTA) / n) for n in (reps_a, reps_b))
    base_a = {(r["src"], r["env"], r["step"], r["n"]): r for r in A if r["arm"].startswith("random")}
    base_b = {(r["env"], str(r["rate"]), r["n"]): r for r in B if r["arm"].startswith("labels alone")}
    PUB, BOOT = "StratPPI as published", "StratPPI estimator, bootstrap-t"

    def ess(r):
        b = base_a[(r["src"], r["env"], r["step"], r["n"])] if r["part"] == "A" else base_b[(r["env"], str(r["rate"]), r["n"])]
        return (b["excess"] / r["excess"]) ** 2 if r["excess"] > 0 else None

    def mess(cells):
        return med([ess(r) for r in cells if verdict(r["miss"], r["reps"]) != "over"])

    def mx(cells):
        return max(r["miss"] for r in cells)

    def line(label, cells, extra=""):
        return f"| {label} | {len(cells)} | {fmt_counts(cells)} | {mx(cells):.3f} | {mess(cells)} |{extra}"
    HEAD = ["| arm | cells | over / unresolved / at or under | largest miss | median ESS where not over |", "|---|---|---|---|---|"]

    def sel(part, **kw):
        return [r for r in part if all(r.get(k) == v for k, v in kw.items())]

    def narrowest(cs, arms, key):
        """In how many cells each arm has the smallest mean bound among the arms that are not over there."""
        best = collections.Counter()
        for k in {key(r) for r in cs}:
            ok = [r for r in cs if key(r) == k and r["arm"] in arms and verdict(r["miss"], r["reps"]) != "over"]
            if ok:
                best[min(ok, key=lambda r: r["excess"])["arm"]] += 1
        return best

    # the cell sets every section draws on
    a8 = [r for r in sel(A, strat="ref 1-8", H=8) if mid(r)]                     # part A, the paper's Table 3 design
    a8p = sel(a8, alloc="prop")
    b510 = [r for r in B if r["K"] in (5, 10)]
    b510p = sel(b510, alloc="prop")
    pa, pb_ = sel(a8p, arm=PUB), sel(b510p, arm=PUB, pred="logit")
    ba, bb = sel(a8p, arm=BOOT), sel(b510p, arm=BOOT)
    wa, wb = sel(a8p, arm="b1w"), sel(b510p, arm="b1w")
    rem = {cv: sel(a8p, arm=f"{PUB}, unlabelled = rest of stratum ({cv})") for cv in CONV}
    tab3 = collections.defaultdict(float)
    for r in pa:
        tab3[(r["src"], r["env"], r["n"])] = max(tab3[(r["src"], r["env"], r["n"])], r["miss"])
    tab3_over = sum(verdict(m, reps_a) == "over" for m in tab3.values())
    over_env = collections.Counter(r["env"] for r in pa if verdict(r["miss"], r["reps"]) == "over")
    worst_env = max(over_env, key=over_env.get)
    two = lambda cells: sum(verdict(r["miss"] + r["miss_lo"], r["reps"], 0.1) == "over" for r in cells)      # noqa: E731
    rare = [r for r in sel(A, strat="ref 1-8", H=8, alloc="prop") if r["truth"] < 0.05]
    al = {(p, a, arm): [r for r in cells if r["alloc"] == a and r["arm"] == arm and r["pred"] in ("logit", "ref 1-8", "-")]
          for p, cells in (("A", a8), ("B", b510)) for a in ("prop", "heur", "heur10", "opt") for arm in (PUB, BOOT, "b1w")}
    big_opt = [r for r in b510 if r["alloc"] == "opt" and min(r["nk"]) >= 10]
    arms3 = ("labels alone, Clopper-Pearson", "PPI++ normal", "PPI++ bootstrap-t", "PPBoot (lam = 1)", "PPBoot (power-tuned)")
    t4 = {(s, p, arm): [r for r in sel(A, strat=s, H=8, alloc="prop", pred=p, arm=arm) if mid(r)]
          for s, p in (("ref 1-8", "ref 1-8"), ("ref 1-4", "ref 1-4"), ("ref 1-4", "ref 5-8"), ("ref 1-8", "-"), ("ref 1-4", "-"))
          for arm in (PUB, BOOT, "b1w")}
    labels = sorted({(r["src"], r["env"]) for r in A if mid(r)})
    last = {k: max(r["step"] for r in A if (r["src"], r["env"]) == k) for k in labels}

    def last_med(cells):          # the paper's Table 3 counting: ESS at the label's last checkpoint, 10 cells
        return med([ess(r) for r in cells if r["step"] == last[(r["src"], r["env"])]])

    def last_cell(src, env, n_s, H, strat, pred, arm):
        return next(r for r in A if (r["src"], r["env"], r["step"], r["n"], r["H"], r["strat"], r["alloc"], r["pred"], r["arm"])
                    == (src, env, last[(src, env)], n_s, H, strat, "prop", pred, arm))
    sweep = {(n_s, H, arm): [r for r in sel(A, strat="ref 1-8", H=H, n=n_s, alloc="prop", arm=arm) if mid(r)]
             for n_s in NS for H in HS for arm in ("b1w", "Wald-t b1", PUB, BOOT)}
    bins = ((0, 12, "5-11"), (12, 30, "20-25"), (30, 60, "45-50"), (60, 1000, "100-200"))
    bsweep = {lab: [r for r in B if r["K"] and r["alloc"] == "prop" and lo <= r["n"] / r["K"] < hi and r["pred"] in ("logit", "-")]
              for lo, hi, lab in bins}
    by_rate = lambda arm, K, rate: med([ess(r) for r in B if (r["arm"], r["K"], str(r["rate"])) == (arm, K, rate)      # noqa: E731
                                        and r["alloc"] in ("prop", "-") and r["pred"] in ("logit", "-")])
    c3 = {(n_s, H, arm): ess(last_cell("013", "C3:refusal", n_s, H, "ref 1-8", "ref 1-8" if arm == BOOT else "-", arm))
          for n_s in NS for H in HS for arm in (BOOT, "b1w")}
    cmp_ = t1["compare"]
    worst = max(cmp_, key=lambda k: cmp_[k]["ub"] if k.startswith("ours vs") else 0)
    sim = {(s["scenario"], s["alloc"], s["n"]): s for s in t1["sim"]}

    L = ["# StratPPI's interval at safety-test sizes: five checks of the P14 finding", "",
         f"`scripts/stratppi_validate.py`; {reps_a:,} draws per cell in part A (reference-rate strata), {reps_b:,} in part B",
         "(judge-logit strata); delta 0.05, one-sided (the upper end of a two-sided 90% interval).", "",
         f"- **over**: miss > delta + 2 se ({DELTA + 2 * se_a:.4f} in A, {DELTA + 2 * se_b:.4f} in B); **unresolved**: delta < miss <= delta + 2 se;",
         "  **at or under**: miss <= delta. Counts are written over / unresolved / at or under.",
         "- A cell of part A is one label at one checkpoint and one safety-set size: 13 label-checkpoints with a rate of",
         "  5-95% (C2:refusal at step 0 is at 95.4% and is left out) by two sizes, 26 cells. The paper's Table 3 takes the",
         "  largest miss over a label's checkpoints, a maximum of two or three cells judged by a one-cell threshold; both",
         "  counts are given. A cell of part B is one judge wording, rate, n and number of strata K.",
         "- ESS: (mean bound minus truth of the reference arm / the same for this arm) squared. The reference arm is a random",
         "  split with a pooled Wilson bound in A and the labels alone with Clopper-Pearson in B. Medians are over the cells",
         "  where the arm is not over.",
         "- The baseline's arms are replayed on its own random streams: the largest difference from `stratppi.json` in any",
         f"  shared cell's miss rate is {res['replay_diff']:.4f}, and the vectorised arm used here differs from the baseline's",
         f"  `stratppi_point` by at most {res['check_a']:.0e} (relative) on the draws checked.", ""]

    # ---- task 1
    L += ["## 1. Is our implementation StratPPI?", "",
          "**What was compared.** No authors' code is public: the NeurIPS checklist in the arXiv source says \"Code may be made",
          "available at a future date\", and a search found none. So this is not a comparison with the authors' code. It is a",
          f"comparison on identical inputs ({t1['n_cases']} inputs: 5 seeds each of the paper's two-stratum Gaussian design and a binary",
          "five-stratum design, 5 draws of the plasmode cell C1:refusal step 200 n_s 200, 5 draws of refusal|rubric|0 at n 220 of",
          "2,000 with K = 10) with: (a) Algorithm 1 written a second time in its general M-estimator form; (b) `ppi_py` (the",
          "PPI++ authors' library, commit 3d1f0c6), `ppi_mean_ci` within each stratum, composed with the known weights; (c) GLIDE",
          "0.11.0 (EmertonData, a third party that cites Fisch et al.), its core functions with the known weights and its public",
          "`StratifiedPPIMeanEstimator`; and (d) the paper's own simulation. Relative difference for the variance, absolute for",
          "the estimate and the upper limit.", "",
          "| comparison | max abs diff, estimate | max rel diff, variance | max abs diff, upper limit | the same on the 10 plasmode inputs |",
          "|---|---|---|---|---|"]
    for name, r in cmp_.items():
        L.append(f"| {name} | {r['est']:.2e} | {r['var']:.2e} | {r['ub']:.2e} | {r['est_p']:.1e}; {r['var_p']:.1e}; {r['ub_p']:.1e} |")
    und = t1["undefined"]
    L += ["", "The differences from the libraries are three plug-in choices the paper leaves open: `ppi_py` clips lam to [0, 1] and",
          "divides variances by n; `ppi_py` and GLIDE take var(f) over the labelled and unlabelled draws together, ours over the",
          "unlabelled. Replayed here, each library's choices reproduce it to rounding error (rows 7, 8, 11), and those replays are",
          "the arms in the table of cells below.",
          f"The predictor is constant in {und['A']} of the 40 strata of the plasmode-A inputs (the 8-sample reference rate is 0 for 71% of",
          f"the prompts) and in {und['B']} of the 50 strata of the plasmode-B inputs (the rubric judge's stored P(Yes) is 0 for {100 * t1['ties']['B']['rubric']['zero']:.0f}% of the responses). `ppi_py` divides by zero there and",
          f"GLIDE raises; both were given lam = 0, which is what ours does. GLIDE's public estimator refused {t1['glide_full_refused']} of the {t1['n_cases']} inputs,",
          "all 10 plasmode inputs, for that reason.", "",
          "The paper's simulation (section 5.1: two-sided 90%, N = 10,000, K = 2), against the mean widths read off its Figure 2",
          "(to about 0.004):", "",
          "| scenario | allocation | n | labels per stratum | mean width, ours | with ppi_py's conventions | Figure 2 | coverage, ours |",
          "|---|---|---|---|---|---|---|---|"]
    for s in t1["sim"]:
        L.append(f"| {s['scenario']} | {s['alloc']} | {s['n']} | {s['nk'][0]}, {s['nk'][1]} | {s['width']:.3f} | {s['width_ppi_py']:.3f} | "
                 f"{s['width_fig']:.3f} | {s['coverage']:.3f} |")
    L += ["", "PPBoot, ours against `ppi_py.ppboot` on two draws of refusal|rubric|0 (N = 2,000): the upper limit with lam = 1 and",
          "20,000 resamples, and the mean and sd over 100 runs of the power-tuned limit with the default 1,000 resamples.", "",
          "| n | se of the estimate | lam = 1: ours | ppi_py | tuned, mean (sd): ours | ppi_py |", "|---|---|---|---|---|---|"]
    for b in t1["ppboot"]:
        L.append(f"| {b['n']} | {b['se']:.4f} | {b['one']:.4f} | {b['one_ref']:.4f} | {b['tuned_mean']:.4f} ({b['tuned_sd']:.4f}) | "
                 f"{b['tuned_mean_ref']:.4f} ({b['tuned_sd_ref']:.4f}) |")
    L += ["", "The published interval in the validity cells, by whose plug-in choices are used:", ""] + HEAD
    L.append(line("A, H = 8: StratPPI as published (the baseline's arm: unlabelled = the whole stratum, ours)", pa))
    for cv in CONV:
        L.append(line(f"A, H = 8: the same with the unlabelled set = the rest of the stratum, as in the paper's experiments ({cv})", rem[cv]))
    for arm, pr in ((PUB, "logit"), (PUB + " (ppi_py)", "logit"), (PUB + " (glide)", "logit"), (PUB, "p")):
        L.append(line(f"B, K = 5 and 10: {arm}, predictor {'the logit' if pr == 'logit' else 'P(Yes), as in the paper'}", sel(b510p, arm=arm, pred=pr)))
    for j in ("raw", "rubric"):
        L.append(line(f"B, K = 5 and 10, the `{j}` judge alone (logit constant in {t1['ties']['B'][j]['constant']} of 10 strata): StratPPI as published",
                      [r for r in pb_ if f"|{j}|" in r["env"]]))
    for arm in ("PPI++ normal", "PPI++ normal (ppi_py)"):
        L.append(line(f"B, unstratified: {arm}", sel(B, arm=arm, pred="logit")))
    L.append(line("B, unstratified: PPI++ normal, predictor P(Yes)", sel(B, arm="PPI++ normal (ours)", pred="p")))
    L += ["", f"Two-sided, as the interval is published (90%, nominal miss 0.10, over if above {0.1 + 2 * np.sqrt(0.09 / reps_a):.4f} in A): "
          f"A {two(pa)} of {len(pa)} over (largest {max(r['miss'] + r['miss_lo'] for r in pa):.3f}); B {two(pb_)} of {len(pb_)} "
          f"(largest {max(r['miss'] + r['miss_lo'] for r in pb_):.3f}).",
          f"The lower limit alone misses up to {max(r['miss_lo'] for r in pa):.3f} in A (on the 93% label, the mirror image) and {max(r['miss_lo'] for r in pb_):.3f} in B.",
          f"By the paper's Table 3 counting (largest miss over checkpoints, 10 label-by-size cells) the baseline's arm is over in {tab3_over} of {len(tab3)}.",
          f"Per cell it is over in {counts(pa)[0]} of {len(pa)}: {over_env[worst_env]} of the {len(sel(pa, env=worst_env))} cells of {worst_env} (the 9% label) and "
          f"{counts(pa)[0] - over_env[worst_env]} of the other {len(pa) - len(sel(pa, env=worst_env))}, all at n_s 100. "
          f"Rare labels (under 5%): over in {counts(sel(rare, arm=PUB))[0]} of {len(sel(rare, arm=PUB))} cells (misses {min(r['miss'] for r in sel(rare, arm=PUB)):.3f}-{mx(sel(rare, arm=PUB)):.3f}); "
          f"`b1w` in {counts(sel(rare, arm='b1w'))[0]}, the bootstrap-t limit in {counts(sel(rare, arm=BOOT))[0]}.", "",
          f"**Verdict.** Our implementation is Algorithm 1 of the paper to rounding error ({cmp_['ours vs Algorithm 1 rewritten (M-estimator form, same plug-ins)']['ub']:.0e} on the upper limit). "
          f"The largest discrepancy from someone else's code is against `ppi_py` composed within strata: {cmp_[worst]['est']:.3f} on the estimate, "
          f"{100 * cmp_[worst]['var']:.0f}% on the variance and {cmp_[worst]['ub']:.3f} on the upper limit, all of it `ppi_py`'s clipping of lam to [0, 1] and its "
          f"division by n; against GLIDE it is {cmp_['ours vs GLIDE core functions, known weights']['ub']:.4f} on the upper limit. Our unstratified PPI++ equals GLIDE's to rounding error and differs "
          f"from `ppi_py` by the same two choices. On the paper's own simulation ours gives its Figure 2 widths at proportional allocation to the "
          f"third decimal and its coverage (0.885-0.901 against a nominal 0.90); at the oracle allocation it matches at n 200 and 1,000 and is "
          f"{sim[('different variance', 'opt', 100)]['width']:.3f} against {sim[('different variance', 'opt', 100)]['width_fig']:.3f} at n 100, where `ppi_py`'s conventions give {sim[('different variance', 'opt', 100)]['width_ppi_py']:.3f}. "
          f"None of this lowers a count: in part B the published interval is over in {counts(pb_)[0]} of {len(pb_)} cells under our choices, "
          f"{counts(sel(b510p, arm=PUB + ' (ppi_py)'))[0]} under `ppi_py`'s, {counts(sel(b510p, arm=PUB + ' (glide)'))[0]} under GLIDE's and {counts(sel(b510p, arm=PUB, pred='p'))[0]} with the judge's "
          f"probability as the predictor (the paper's set-up); in part A in {counts(pa)[0]} of {len(pa)} (baseline), {counts(rem['ours'])[0]}, {counts(rem['glide'])[0]} and {counts(rem['ppi_py'])[0]} "
          "with the unlabelled set drawn as the paper draws it. What the comparison cannot rule out is a choice in the authors' unreleased code that "
          "neither library makes.", ""]

    # ---- task 2
    L += ["## 2. The paper's allocation of labels", "",
          "`opt`: Proposition 3, labels in proportion to w_k sd(Y - lam_k f | stratum), from the pool's own moments (the oracle;",
          "the paper runs it only in simulation). `heur`: the rule the paper runs on real data (Appendix B), w_k sqrt(mean c(1 - c)",
          "+ var c) with c the autorater's confidence (the reference rate in A, the judge's P(Yes) in B). The unlabelled draws",
          "stay proportional, as in the paper. The paper gives no floor and its rule can return no label for a stratum; `heur`",
          "and `opt` are run with at least 2 labels per stratum and `heur10` with at least 10. All use exactly n labels;",
          "`prop` is the baseline's rounding (220 of 225 at K = 10).", ""] + HEAD
    for p, lab in (("A", "A, H = 8"), ("B", "B, K = 5 and 10")):
        for arm in (PUB, BOOT, "b1w"):
            for a in ("prop", "heur", "heur10", "opt"):
                L.append(line(f"{lab}: {arm}, {a}", al[(p, a, arm)]))
    smallest = lambda p, a: min(min(r["nk"]) for r in al[(p, a, PUB)])      # noqa: E731
    L += ["", "Smallest stratum under each rule (labels): " + "; ".join(
        f"{p}: " + ", ".join(f"{a} {smallest(p, a)}" for a in ("heur", "heur10", "opt")) for p in ("A", "B")) + ".",
        f"In the {len(sel(big_opt, arm=PUB))} cells of B where the oracle rule leaves every stratum 10 labels or more: published "
        f"{fmt_counts(sel(big_opt, arm=PUB))}, bootstrap-t {fmt_counts(sel(big_opt, arm=BOOT))}, `b1w` {fmt_counts(sel(big_opt, arm='b1w'))}.", "",
        f"**Verdict.** No. With the oracle allocation the published interval is over in {counts(al[('A', 'opt', PUB)])[0]} of {len(al[('A', 'opt', PUB)])} cells of A "
        f"(largest miss {mx(al[('A', 'opt', PUB)]):.3f}; {counts(al[('A', 'prop', PUB)])[0]} and {mx(al[('A', 'prop', PUB)]):.3f} with proportional) and in "
        f"{counts(al[('B', 'opt', PUB)])[0]} of {len(al[('B', 'opt', PUB)])} of B ({mx(al[('B', 'opt', PUB)]):.3f}; {counts(al[('B', 'prop', PUB)])[0]} and {mx(al[('B', 'prop', PUB)]):.3f}). "
        f"With the heuristic it is over in {counts(al[('A', 'heur', PUB)])[0]} and {counts(al[('B', 'heur', PUB)])[0]}, with misses up to {mx(al[('A', 'heur', PUB)]):.3f} and "
        f"{mx(al[('B', 'heur', PUB)]):.3f}: the rubric judge's P(Yes) is near 0 or 1 in most strata, so the rule sends nearly every label to one stratum "
        f"(2 or 10 labels are left in each of the others), and the paper itself reports the heuristic as too aggressive with uncalibrated confidences. "
        f"The allocation also takes away the repair: the bootstrap-t limit on the same estimator, not over in any cell under proportional "
        f"allocation, is over in {counts(al[('A', 'opt', BOOT)])[0]} of {len(al[('A', 'opt', BOOT)])} (A) and {counts(al[('B', 'opt', BOOT)])[0]} of {len(al[('B', 'opt', BOOT)])} (B) with the oracle rule "
        f"(largest {mx(al[('A', 'opt', BOOT)]):.3f} and {mx(al[('B', 'opt', BOOT)]):.3f}) and in {counts(al[('A', 'heur', BOOT)])[0]} and {counts(al[('B', 'heur', BOOT)])[0]} with the heuristic. "
        f"`b1w` is not over under any rule except one cell of `heur10` in B ({mx(al[('B', 'heur10', 'b1w')]):.3f}), and is widest under the paper's rules "
        f"(median ESS {mess(al[('A', 'opt', 'b1w')])} in A and {mess(al[('B', 'opt', 'b1w')])} in B with the oracle rule against {mess(al[('A', 'prop', 'b1w')])} and {mess(al[('B', 'prop', 'b1w')])} "
        f"proportional). In the cells of A where the published interval is not over, the oracle allocation does narrow it (median ESS {mess(al[('A', 'opt', PUB)])} against "
        f"{mess(al[('A', 'prop', PUB)])}). The floor of 2 is ours; in the {len(sel(big_opt, arm=PUB))} cells of B where the oracle rule leaves every stratum 10 labels or more the published "
        f"interval is over in {counts(sel(big_opt, arm=PUB))[0]}.", ""]

    # ---- task 3
    L += ["## 3. PPBoot on the judge-logit cells", "",
          "Unstratified, the judge's logit as the prediction, 1,000 resamples (50 for lam), the percentile limit of the paper's",
          "Algorithm 1; `width` is the mean of upper limit minus estimate.", "",
          HEAD[0] + " median width |", HEAD[1] + "---|"]
    for arm in arms3:
        cs = sel(B, arm=arm)
        L.append(line(arm, cs, f" {np.median([r['width'] for r in cs]):.4f} |"))
    L += ["", "| judge | rate | n of N | " + " | ".join(arms3) + " |", "|---|---|---|" + "---|" * len(arms3)]
    for k in sorted(base_b):
        cs = {r["arm"]: r for r in B if (r["env"], str(r["rate"]), r["n"]) == k and r["K"] == 0}
        bold = lambda a: "**" if verdict(cs[a]["miss"], reps_b) == "over" else ""      # noqa: E731
        L.append(f"| {k[0]} | {cs[arms3[0]]['truth']:.3f} | {k[2]} of {cs[arms3[0]]['N']} | " + " | ".join(
            f"{bold(a)}{cs[a]['miss']:.3f}{bold(a)}; {cs[a]['width']:.4f}" for a in arms3) + " |")
    p1, pt = sel(B, arm=arms3[3]), sel(B, arm=arms3[4])
    L += ["", "Cell: miss; width. Bold: over.", "",
          f"**Verdict.** PPBoot does not hold its level here. The basic form is {fmt_counts(p1)} over its 14 cells (largest miss {mx(p1):.3f}) and "
          f"the power-tuned form {fmt_counts(pt)} ({mx(pt):.3f}); PPI++ with a normal limit is {fmt_counts(sel(B, arm=arms3[1]))} ({mx(sel(B, arm=arms3[1])):.3f}) and with a "
          f"bootstrap-t limit {fmt_counts(sel(B, arm=arms3[2]))} ({mx(sel(B, arm=arms3[2])):.3f}). PPBoot's limit is a percentile limit, first-order like the normal one, "
          f"and its misses sit between the two. Its widths are close to the normal limit's (median {np.median([r['width'] for r in pt]):.4f} tuned against "
          f"{np.median([r['width'] for r in sel(B, arm=arms3[1])]):.4f}), narrower than the bootstrap-t limit's ({np.median([r['width'] for r in sel(B, arm=arms3[2])]):.4f}) because it is over. "
          f"The basic form is over in {counts([r for r in p1 if '|raw|' in r['env']])[0]} of the 7 cells of the `raw` judge and in "
          f"{counts([r for r in p1 if '|rubric|' in r['env']])[0]} of the 7 of the `rubric` judge.", ""]

    # ---- task 4
    L += ["## 4. A predictor that is not the stratifier (part A, H = 8)", "",
          "The 8 reference samples split: strata on the mean of samples 1-4, the regression on the mean of samples 5-8, against",
          "the baseline (both on all 8) and a control (both on samples 1-4).", "",
          HEAD[0] + " median ESS, last checkpoints (Table 3's 10 cells) |", HEAD[1] + "---|"]
    for s, p, lab in (("ref 1-8", "ref 1-8", "strata 1-8, predictor 1-8 (Table 3)"), ("ref 1-4", "ref 1-4", "strata 1-4, predictor 1-4 (control)"),
                      ("ref 1-4", "ref 5-8", "strata 1-4, predictor 5-8")):
        for arm in (PUB, BOOT):
            L.append(line(f"{lab}: {arm}", t4[(s, p, arm)], f" {last_med(t4[(s, p, arm)])} |"))
        if p != "ref 5-8":
            L.append(line(f"strata {s[4:]}: b1w (no predictor)", t4[(s, "-", "b1w")], f" {last_med(t4[(s, '-', 'b1w')])} |"))
    L += ["", "| label, last checkpoint (rate) | n_s | b1w, strata 1-8 | bootstrap-t StratPPI, 1-8 / 1-8 | b1w, strata 1-4 | bootstrap-t StratPPI, 1-4 / 1-4 | bootstrap-t StratPPI, 1-4 / 5-8 |",
          "|---|---|---|---|---|---|---|"]
    for src, env in labels:
        for n_s in NS:
            cells = [last_cell(src, env, n_s, 8, s, p, arm) for s, p, arm in (
                ("ref 1-8", "-", "b1w"), ("ref 1-8", "ref 1-8", BOOT), ("ref 1-4", "-", "b1w"), ("ref 1-4", "ref 1-4", BOOT), ("ref 1-4", "ref 5-8", BOOT))]
            L.append(f"| {env}{' pushed' if src == '014' else ''} ({cells[0]['truth']:.2f}) | {n_s} | " + " | ".join(f"{r['miss']:.3f}; {ess(r):.2f}" for r in cells) + " |")
    s48 = t4[("ref 1-4", "ref 5-8", BOOT)]
    e4 = lambda src, env, s, p: ess(last_cell(src, env, 100, 8, s, p, BOOT))      # noqa: E731
    L += ["", "Cell: miss; ESS.", "",
          f"**Verdict.** Giving the regression a predictor independent of the stratifier does not buy anything. With a bootstrap-t limit the split "
          f"design is {fmt_counts(s48)} (largest miss {mx(s48):.3f}), with a median ESS of {mess(s48)} over all cells and {last_med(s48)} over Table 3's ten, "
          f"against {mess(t4[('ref 1-8', 'ref 1-8', BOOT)])} and {last_med(t4[('ref 1-8', 'ref 1-8', BOOT)])} with all 8 samples in both roles and {mess(t4[('ref 1-4', 'ref 1-4', BOOT)])} and "
          f"{last_med(t4[('ref 1-4', 'ref 1-4', BOOT)])} for the control. It helps one label (over-refusal at n_s 100: {e4('013', 'C1:refusal', 'ref 1-4', 'ref 5-8'):.2f} and "
          f"{e4('014', 'C1:refusal', 'ref 1-4', 'ref 5-8'):.2f} pushed, against {e4('013', 'C1:refusal', 'ref 1-8', 'ref 1-8'):.2f} and {e4('014', 'C1:refusal', 'ref 1-8', 'ref 1-8'):.2f}) and hurts the two "
          f"high-rate ones at that size ({e4('013', 'C2:refusal', 'ref 1-4', 'ref 5-8'):.2f} against {e4('013', 'C2:refusal', 'ref 1-8', 'ref 1-8'):.2f}; "
          f"{e4('013', 'C3:refusal', 'ref 1-4', 'ref 5-8'):.2f} against {e4('013', 'C3:refusal', 'ref 1-8', 'ref 1-8'):.2f}). `b1w` on all 8 samples is at {mess(t4[('ref 1-8', '-', 'b1w')])} and {last_med(t4[('ref 1-8', '-', 'b1w')])}. The published interval is over "
          f"in {counts(t4[('ref 1-4', 'ref 5-8', PUB)])[0]} of {len(t4[('ref 1-4', 'ref 5-8', PUB)])} cells with the independent predictor (largest {mx(t4[('ref 1-4', 'ref 5-8', PUB)]):.3f}), more than with the shared one "
          f"({counts(pa)[0]}). The explanation in the paper, that sharing the 8 samples leaves the regression little to add, is not supported: given "
          "an independent predictor the regression adds no more. Why was not isolated; the split also halves the samples behind the strata "
          f"(`b1w` on strata of 4 samples: {last_med(t4[('ref 1-4', '-', 'b1w')])} over Table 3's ten cells).", ""]

    # ---- task 5
    L += ["## 5. Labels per stratum", "",
          "Part A: proportional allocation, the reference rate (8 samples) as stratifier and predictor, H strata, 13 cells a row.",
          "`narrowest`: cells where the arm has the smallest mean bound among the three arms other than the published interval",
          "that are not over in the cell (the published interval is over somewhere in every block, so it is not a candidate).", "",
          "| n_s | H | labels per stratum | arm | over / unresolved / at or under | largest miss | median ESS where not over | narrowest |",
          "|---|---|---|---|---|---|---|---|"]
    cand = ("b1w", "Wald-t b1", BOOT)
    best_a = {(n_s, H): narrowest([r for arm in cand for r in sweep[(n_s, H, arm)]], cand, lambda r: (r["src"], r["env"], r["step"]))
              for n_s in NS for H in HS}
    best_b = {lab: narrowest(bsweep[lab], ("b1w", BOOT), lambda r: (r["env"], str(r["rate"]), r["n"], r["K"])) for lab in bsweep}
    for H in HS:
        for n_s in NS:
            best = best_a[(n_s, H)]
            for arm in ("b1w", "Wald-t b1", PUB, BOOT):
                ca = sweep[(n_s, H, arm)]
                L.append(f"| {n_s} | {H} | {n_s / H:.1f} | {arm} | {fmt_counts(ca)} | {mx(ca):.3f} | {mess(ca)} | "
                         f"{'-' if arm == PUB else f'{best[arm]} of {len(ca)}'} |")
    L += ["", "ESS at each label's last checkpoint, bootstrap-t StratPPI / `b1w`:", "",
          "| label (rate) | " + " | ".join(f"n_s {n_s}, H {H} ({n_s / H:.1f})" for n_s in NS for H in HS) + " |", "|---|" + "---|" * 6]
    for src, env in labels:
        L.append(f"| {env}{' pushed' if src == '014' else ''} ({last_cell(src, env, 100, 8, 'ref 1-8', '-', 'b1w')['truth']:.2f}) | " + " | ".join(
            f"{ess(last_cell(src, env, n_s, H, 'ref 1-8', 'ref 1-8', BOOT)):.2f} / {ess(last_cell(src, env, n_s, H, 'ref 1-8', '-', 'b1w')):.2f}"
            for n_s in NS for H in HS) + " |")
    L += ["", "Part B: proportional allocation, K = 5, 10 or 20 strata of the judge's logit; cells grouped by labels per stratum (n / K).", "",
          "| labels per stratum | (n, K) | arm | over / unresolved / at or under | largest miss | median ESS where not over | narrowest |",
          "|---|---|---|---|---|---|---|"]
    for _, _, lab in bins:
        cs, best = bsweep[lab], best_b[lab]
        nk = ", ".join(sorted({f"({r['n']}, {r['K']})" for r in cs}))
        for arm in ("b1w", PUB, BOOT):
            ca = sel(cs, arm=arm)
            L.append(f"| {lab} | {nk} | {arm} | {fmt_counts(ca)} | {mx(ca):.3f} | {mess(ca)} | {'-' if arm == PUB else f'{best[arm]} of {len(ca)}'} |")
    L += ["", "Median ESS in part B by rate (20%, 5%, 1.3%): " + "; ".join(
        f"{name} K = {K}: " + ", ".join(by_rate(arm, K, rt) for rt in ("pool", "0.05", "0.013"))
        for arm, name in ((BOOT, "bootstrap-t StratPPI"), ("b1w", "`b1w`")) for K in KS_B) + ".", ""]
    allb = [r for lab in bsweep for r in bsweep[lab]]
    alla = lambda arm: [r for n_s in NS for H in HS for r in sweep[(n_s, H, arm)]]      # noqa: E731
    L += [f"**Verdict.** *Where each arm holds.* The published interval is over in every block: {counts(alla(PUB))[0]} of {len(alla(PUB))} cells of A, from "
          f"{counts(sweep[(200, 4, PUB)])[0]} of 13 at 50 labels per stratum to {counts(sweep[(100, 16, PUB)])[0]} of 13 at 6, and {counts(sel(allb, arm=PUB))[0]} of {len(sel(allb, arm=PUB))} of B, "
          f"still {counts(sel(bsweep['100-200'], arm=PUB))[0]} of {len(sel(bsweep['100-200'], arm=PUB))} at 100-200 labels per stratum (largest {mx(sel(bsweep['100-200'], arm=PUB)):.3f}, at the 1.3% rate). "
          f"More labels per stratum shrink the excess and do not remove it at these rates. The bootstrap-t limit on the same estimator is not over "
          f"in any cell at any size ({fmt_counts(alla(BOOT))} in A, {fmt_counts(sel(allb, arm=BOOT))} in B, largest {max(mx(alla(BOOT)), mx(sel(allb, arm=BOOT))):.3f}). "
          f"`b1w` is {fmt_counts(alla('b1w'))} in A, its one cell over at 6 labels per stratum ({mx(sweep[(100, 16, 'b1w')]):.3f}), and {fmt_counts(sel(allb, arm='b1w'))} in B. "
          f"*Which is narrowest.* In A the bootstrap-t StratPPI has the larger median ESS in four of the six blocks: with 4 strata at either size "
          f"({mess(sweep[(100, 4, BOOT)])} against {mess(sweep[(100, 4, 'b1w')])} for `b1w` at 25 labels per stratum, {mess(sweep[(200, 4, BOOT)])} against {mess(sweep[(200, 4, 'b1w')])} at 50) and at "
          f"n_s 200 with 8 or 16 strata ({mess(sweep[(200, 8, BOOT)])} against {mess(sweep[(200, 8, 'b1w')])} at 25, {mess(sweep[(200, 16, BOOT)])} against {mess(sweep[(200, 16, 'b1w')])} at 12.5). `b1w` is "
          f"ahead at n_s 100 with 8 or 16 strata ({mess(sweep[(100, 8, 'b1w')])} against {mess(sweep[(100, 8, BOOT)])} at 12.5, {mess(sweep[(100, 16, 'b1w')])} against {mess(sweep[(100, 16, BOOT)])} at 6). "
          f"So 25 labels per stratum is not the dividing line: 12.5 is enough at n_s 200 and not at n_s 100. The cell the paper singles out "
          f"(refusal on harmful prompts, n_s 100, 8 strata: {c3[(100, 8, BOOT)]:.2f} against {c3[(100, 8, 'b1w')]:.2f}) is {c3[(200, 16, BOOT)]:.2f} against {c3[(200, 16, 'b1w')]:.2f} at the same "
          f"12.5 labels per stratum with n_s 200, and {c3[(100, 4, BOOT)]:.2f} against {c3[(100, 4, 'b1w')]:.2f} at n_s 100 with 4 strata. Cell by cell the "
          f"bootstrap-t StratPPI is the narrowest of the three in " + ", ".join(f"{best_a[(100, H)][BOOT]} and {best_a[(200, H)][BOOT]} of 13 cells with {H} strata" for H in HS) +
          f" (n_s 100 and 200), so at n_s 100 with 8 strata the two are level by cells ({best_a[(100, 8)][BOOT]} and {best_a[(100, 8)]['b1w']}) and `b1w` leads on the median because the "
          f"bootstrap-t limit loses badly on the two high-rate labels. In B `b1w` is the narrowest in {best_b['5-11']['b1w']} of {len(sel(bsweep['5-11'], arm='b1w'))} cells at 5-11 "
          f"labels per stratum and {best_b['20-25']['b1w']} of {len(sel(bsweep['20-25'], arm='b1w'))} at 20-25; the bootstrap-t StratPPI in {best_b['45-50'][BOOT]} of {len(sel(bsweep['45-50'], arm=BOOT))} at 45-50 and "
          f"{best_b['100-200'][BOOT]} of {len(sel(bsweep['100-200'], arm=BOOT))} at 100-200. At the 1.3% "
          f"rate it is useless at any size (ESS {by_rate(BOOT, 10, '0.013')} at K = 10) and `b1w` is not ({by_rate('b1w', 10, '0.013')}).", ""]

    # ---- the claim, and the sentences
    L += ["## Does the claim survive?", "",
          f"\"StratPPI's published interval runs over its nominal level in these settings\" **survives**: {counts(pb_)[0]} of {len(pb_)} judge-logit cells and "
          f"{tab3_over} of 10 reference-rate cells by the paper's counting ({counts(pa)[0]} of {len(pa)} per checkpoint, {counts(pa)[1]} more unresolved), unchanged by "
          "the public implementations' conventions, by the paper's own allocation rules, by an independent predictor, or by reading the interval "
          "two-sided. Its qualifiers: (1) it is a statement about Algorithm 1 of the paper, which our code reproduces to rounding error; there is "
          "no authors' code to run. (2) It is a finite-sample, one-sided result at rates of 1-20% and strata of 6-200 labels. The paper claims "
          "asymptotic coverage and shows two-sided coverage for two Gaussian strata of 50 labels or more, which we reproduce; nothing here "
          "contradicts a statement the paper makes. (3) In part A the excess is small outside the 9% label (at most "
          f"{max(r['miss'] for r in pa if r['env'] != worst_env):.3f}) and part A sits outside the paper's regime: 62 unlabelled items a stratum, and a predictor that is constant in "
          f"{min(t1['ties']['A'].values())} to {max(t1['ties']['A'].values())} of the 8 strata, so StratPPI there is mostly a stratified Wald interval and the finding is the Wald interval's. "
          f"The same holds for the `rubric` judge in part B (logit constant in {t1['ties']['B']['rubric']['constant']} of 10 strata). The `raw` judge in part B is the paper's regime, "
          f"a continuous score with no constant stratum, and there the published interval is over in {counts([r for r in pb_ if '|raw|' in r['env']])[0]} of 14 cells, "
          f"with misses of {min(r['miss'] for r in pb_ if '|raw|' in r['env']):.3f}-{max(r['miss'] for r in pb_ if '|raw|' in r['env']):.3f}: over at every size at the 1.3% and 5% rates, and within about a point of delta at the 20% rate "
          "with 500 labels. (4) \"A bootstrap-t limit on the same estimator holds\" survives only with proportional allocation.", "",
          "## Sentences in sections 5 and 6 of `reports/paper_certification.md` that need to change", "",
          "Quoted as they stand, with what the evidence above supports instead.", "",
          "**Section 5**", "",
          "1. \"We ran it on the same strata and the same 5,000 draws per cell, with the reference rate as the predictor [P14]\"",
          "   - Say what \"it\" is: Algorithm 1 of the paper in our code, which agrees with a second writing of the algorithm to rounding error and",
          f"     is over in as many cells or more under the plug-in choices of `ppi_py` and GLIDE ({counts(rem['ppi_py'])[0]} and {counts(rem['glide'])[0]} of 26 against {counts(pa)[0]}); the authors have released no code.",
          f"     Say also that the reference rate is constant within {min(t1['ties']['A'].values())} to {max(t1['ties']['A'].values())} of the 8 strata, so in those strata StratPPI is the stratum's plain mean.",
          "2. \"It exceeds delta in 4 of 10 cells at delta 0.05 (one of them marginally, at 0.056; 3 of 10 at delta 0.1), most at the 9% label, and in",
          "   every rare-label cell\"",
          f"   - The 4 of 10 is the largest miss over two or three checkpoints against a one-cell threshold. Add the per-cell count: over in {counts(pa)[0]} of {len(pa)}, unresolved",
          f"     above delta in {counts(pa)[1]}, at or under in {counts(pa)[2]}; all {over_env[worst_env]} cells of the 9% label and {counts(pa)[0] - over_env[worst_env]} of the other {len(pa) - len(sel(pa, env=worst_env))}, all at n_s 100. Add that the paper's oracle",
          f"     allocation leaves it over in {counts(al[('A', 'opt', PUB)])[0]} of 26 (largest {mx(al[('A', 'opt', PUB)]):.3f}) and its heuristic in {counts(al[('A', 'heur', PUB)])[0]} ({mx(al[('A', 'heur', PUB)]):.3f}).",
          "3. \"The same estimator with a bootstrap-t limit holds in all ten cells (largest miss 0.045) and its median ESS is 2.10 against 2.16 for",
          "   `b1w`.\"",
          f"   - Add \"with proportional allocation\". Under the paper's oracle allocation the same limit is over in {counts(al[('A', 'opt', BOOT)])[0]} of 26 cells (largest {mx(al[('A', 'opt', BOOT)]):.3f}),",
          f"     under its heuristic in {counts(al[('A', 'heur', BOOT)])[0]} ({mx(al[('A', 'heur', BOOT)]):.3f}). Per checkpoint, proportional: {fmt_counts(ba)} of 26.",
          "4. \"We attribute that cell, without a test, to strata of about twelve labels on a heavily tied predictor.\"",
          f"   - Now tested, and the attribution does not hold as written: at the same 12.5 labels per stratum reached with n_s 200 and 16 strata the ESS is {c3[(200, 16, BOOT)]:.2f}",
          f"     against {c3[(200, 16, 'b1w')]:.2f} for `b1w`; the drop is at n_s 100 with 8 or 16 strata ({c3[(100, 8, BOOT)]:.2f}, {c3[(100, 16, BOOT)]:.2f}) and absent with 4 ({c3[(100, 4, BOOT)]:.2f}).",
          "5. \"The within-stratum regression on the reference rate adds little once the limit is valid. We keep `b1w` as the bound for",
          "   reference-rate strata at these sizes and note the bootstrap-t StratPPI as the candidate for larger strata: it was ahead where strata",
          "   held 25 labels, the larger of the two sizes we ran.\"",
          f"   - Replace the last clause with the sweep: the bootstrap-t StratPPI has the larger median ESS with 4 strata at either size ({mess(sweep[(100, 4, BOOT)])} against {mess(sweep[(100, 4, 'b1w')])},",
          f"     {mess(sweep[(200, 4, BOOT)])} against {mess(sweep[(200, 4, 'b1w')])}) and at n_s 200 with 8 or 16 strata ({mess(sweep[(200, 8, BOOT)])} against {mess(sweep[(200, 8, 'b1w')])}; {mess(sweep[(200, 16, BOOT)])} against {mess(sweep[(200, 16, 'b1w')])}), and the smaller one at n_s 100 with 8 or 16",
          f"     ({mess(sweep[(100, 8, BOOT)])} against {mess(sweep[(100, 8, 'b1w')])}; {mess(sweep[(100, 16, BOOT)])} against {mess(sweep[(100, 16, 'b1w')])}). The line is the safety-set size as much as labels per stratum; \"larger strata\" should read",
          f"     \"n_s of 200, or 4 strata\". \"Adds little\" stands ({mess(ba)} against {mess(wa)} for `b1w` on the same strata, median ESS over 26 cells).",
          "6. \"In our predictor comparison the predictor is the same reference rate that defines the strata, which leaves the regression little to",
          "   add.\"",
          f"   - Tested and not the reason. Strata on samples 1-4 with samples 5-8 as the predictor: bootstrap-t limit {fmt_counts(s48)} of 26, median ESS over Table 3's ten cells",
          f"     {last_med(s48)}, against {last_med(t4[('ref 1-8', 'ref 1-8', BOOT)])} with all 8 samples in both roles and {last_med(t4[('ref 1-8', '-', 'b1w')])} for `b1w`. Replace with: an independent predictor from a split of the 8 samples does",
          "     not improve on using all 8 for both.",
          "7. \"We did not run StratPPI's optimal allocation, and the implementation is ours, written from the paper's equations.\"",
          "   - Both halves are out of date. Replace with what was run: the oracle and heuristic allocations (item 2 and 3 above), and the comparison of",
          "     the implementation with Algorithm 1 rewritten, `ppi_py`, GLIDE and the paper's Figure 2; keep \"the authors have released no code\".", "",
          "**Section 6**", "",
          "8. \"StratPPI as published, with 5 or 10 strata, is over its level in 26 of 28 cells (misses 0.059-0.239 in those 26), as PPI++ with a normal",
          "   limit is in all 14 (0.061-0.232).\"",
          f"   - Stands. Add: {counts(sel(b510p, arm=PUB + ' (ppi_py)'))[0]} of 28 with `ppi_py`'s plug-in choices, {counts(sel(b510p, arm=PUB + ' (glide)'))[0]} with GLIDE's, {counts(sel(b510p, arm=PUB, pred='p'))[0]} with the judge's probability as the predictor; {counts(al[('B', 'opt', PUB)])[0]} of 28 with",
          f"     the paper's oracle allocation and {counts(al[('B', 'heur', PUB)])[0]} with its heuristic, where misses reach {mx(al[('B', 'heur', PUB)]):.2f} on the overconfident judge; and with 20 strata,",
          f"     {counts(sel([r for r in B if r['K'] == 20], arm=PUB, pred='logit'))[0]} of 14.",
          "9. \"The StratPPI estimator with a bootstrap-t limit holds in all 28 (largest miss 0.056) and is the most efficient valid route: with 10 strata",
          "   its median ESS against the labels alone is 2.54 at a 20% rate and 1.87 at 5% (2.47 and 1.76 with 5 strata), where unstratified PPI++ with",
          "   the same limit gives 1.67 and 1.39.\"",
          f"   - \"Holds in all 28\" should read: not over in any of 28, unresolved above delta in {counts(bb)[1]} (largest 0.056), and add \"with proportional allocation\":",
          f"     with the paper's oracle allocation it is over in {counts(al[('B', 'opt', BOOT)])[0]} of 28 (largest {mx(al[('B', 'opt', BOOT)]):.3f}). \"The most efficient valid route\" is true among 5 and 10 strata;",
          f"     with 20 strata `b1w` gives {by_rate('b1w', 20, 'pool')} at the 20% rate against {by_rate(BOOT, 20, 'pool')} for the bootstrap-t StratPPI (and {by_rate('b1w', 20, '0.05')} against {by_rate(BOOT, 20, '0.05')} at 5%).",
          f"     PPBoot belongs in this list and is not a valid route: over in {counts(p1)[0]} of 14 cells in its basic form and {counts(pt)[0]} power-tuned (largest {mx(p1):.3f} and {mx(pt):.3f}).",
          "10. \"Stratifying on the judge and ignoring it within strata (`b1w`) also holds (largest miss 0.053), at 2.41 and 1.42 with 10 strata (2.24 and",
          "    1.26 with 5).\"",
          f"    - By the one rule: {fmt_counts(wb)} of 28. Add 20 strata ({by_rate('b1w', 20, 'pool')} and {by_rate('b1w', 20, '0.05')}) and that at 5-11 labels per stratum `b1w` is",
          f"      narrower than the bootstrap-t StratPPI in {best_b['5-11']['b1w']} of {len(sel(bsweep['5-11'], arm='b1w'))} cells.",
          "11. \"So the third row of Table 4 has a better form when labelling can follow scoring: stratify on the judge's logit, StratPPI's estimator, a",
          "    bootstrap-t limit.\"",
          "    - Add the conditions the evidence carries: proportional allocation (not the paper's optimal or heuristic rule), and about 45 labels per",
          f"      stratum or more for it to be the narrower choice ({best_b['45-50'][BOOT]} of {len(sel(bsweep['45-50'], arm=BOOT))} and {best_b['100-200'][BOOT]} of {len(sel(bsweep['100-200'], arm=BOOT))} cells); at 20-25 the two split the cells ({best_b['20-25'][BOOT]} and {best_b['20-25']['b1w']} of {len(sel(bsweep['20-25'], arm=BOOT))})",
          f"      and at 5-11 `b1w` is narrower in {best_b['5-11']['b1w']} of {len(sel(bsweep['5-11'], arm='b1w'))}. At the 1.3% rate the count rule's fallback stands.",
          "12. \"a bootstrap for PPI is PPBoot (Zrnic, 2024). Our contribution in this section is narrower: evidence that the published normal-quantile",
          "    intervals, stratified or not, do not hold their level at the sample sizes and rates of a safety test; a studentised bootstrap that does\"",
          f"    - PPBoot is now run: its percentile limit is over in {counts(p1)[0]} of 14 cells (basic) and {counts(pt)[0]} of 14 (power-tuned). The sentence should set the",
          "      studentised bootstrap against PPBoot's percentile bootstrap as well as against the normal limits.", "",
          "**Outside sections 5 and 6, depending on the same evidence** (not asked for, listed so they are not missed)", "",
          "- Section 9 (limits): \"The StratPPI and PPI++ intervals are ours, written from the papers' equations, with proportional allocation, and not checked",
          "  against the authors' code. PPBoot was not run.\" Every clause is out of date except that no authors' StratPPI code exists to check against.",
          "- Table 2, row \"StratPPI as published (normal limit)\": the counts stand; the row \"StratPPI estimator with a bootstrap-t limit\" needs",
          "  \"proportional allocation\" in its setting. A PPBoot row can be added among the bounds that fail (up to 0.148).",
          "- Abstract and contributions: \"in our implementation\" can become \"in our implementation, which is the published algorithm to rounding error and gives",
          "  the same counts under two public libraries' conventions\"; \"a bootstrap-t limit restores the level\" needs \"with proportional allocation\" for StratPPI.",
          "- Section 10 (related work): \"(the normal-quantile intervals of PPI++ and StratPPI do not; their estimators with a bootstrap-t limit do)\" can add that PPBoot's",
          "  percentile interval does not either."]
    open(path, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5000, help="draws per cell, part A (the baseline's 5,000)")
    ap.add_argument("--reps-b", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--task1-only", action="store_true", help="redo the comparison of implementations, keep the cells")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "stratppi_validate.json")
    if a.task1_only:
        res = json.load(open(path))
        res["task1"] = task1()
        json.dump(res, open(path, "w"))
    elif not a.report_only:
        t0 = time.time()
        t1 = task1()
        print(f"task 1 inputs compared, {time.time() - t0:.0f}s", flush=True)
        jobs_a = [("013", p, lab, s, a.reps) for p, labs in RP.LABELS.items() for lab in labs for s in (0, 100, 200)]
        jobs_a += [("014", "C1", "refusal", s, a.reps) for s in (100, 200)]
        pools = PL.load_pools()
        jobs_b, seed = [], 7000                    # the baseline's cells and seeds
        for key in (("refusal", "rubric", 0), ("refusal", "raw", 0)):
            for n, big_n, rate in ((100, 2000, None), (225, 2000, None), (500, 2000, None), (225, 4000, 0.05), (1000, 4000, 0.05),
                                   (225, 4000, 0.013), (1000, 4000, 0.013)):
                seed += 1
                jobs_b.append((key, *pools[key], n, big_n, a.reps_b, rate, seed))
        rows, check = [], 0.0
        with ProcessPoolExecutor(a.workers) as ex:
            fa = [ex.submit(job_a, j) for j in jobs_a]
            fb = [ex.submit(job_b, j) for j in sorted(jobs_b, key=lambda j: -j[3])]
            for i, f in enumerate(fa):
                r, ch = f.result()
                rows += r; check = max(check, ch); print(f"A {i + 1}/{len(fa)} {time.time() - t0:.0f}s", flush=True)
            for i, f in enumerate(fb):
                rows += f.result(); print(f"B {i + 1}/{len(fb)} {time.time() - t0:.0f}s", flush=True)
        json.dump(dict(reps_a=a.reps, reps_b=a.reps_b, check_a=check, task1=t1, rows=rows), open(path, "w"))
    res = json.load(open(path))
    old = json.load(open(os.path.join(OUT, "stratppi.json")))       # the baseline's rows, for the replay check
    diff = 0.0
    if (old["reps"], old["reps_b"]) == (res["reps_a"], res["reps_b"]):
        mine = {}
        for r in res["rows"]:
            if r["part"] == "A" and (r["strat"], r["H"], r["alloc"]) == ("ref 1-8", 8, "prop"):
                name = {"b1w": "S2 + b1w (this paper)", "Wald-t b1": "S2 + Wald-t b1", "StratPPI as published": "S2 + StratPPI",
                        "StratPPI estimator, bootstrap-t": "S2 + StratPPI, bootstrap-t"}.get(r["arm"])
                mine[("A", r["src"], r["env"], r["step"], r["n"], name)] = r["miss"]
            if r["part"] == "A" and r["arm"].startswith("random"):
                mine[("A", r["src"], r["env"], r["step"], r["n"], "random + pooled Wilson (R)")] = r["miss"]
            if r["part"] == "B" and r["alloc"] in ("prop", "-") and r["pred"] in ("logit", "-"):
                name = {"b1w": f"judge strata + b1w, K={r['K']}", "StratPPI as published": f"StratPPI, K={r['K']}",
                        "StratPPI estimator, bootstrap-t": f"StratPPI, bootstrap-t, K={r['K']}", "PPI++ normal": "PPI++ normal",
                        "PPI++ bootstrap-t": "PPI++ bootstrap-t (this paper)",
                        "labels alone, Clopper-Pearson": "labels alone, Clopper-Pearson"}.get(r["arm"])
                mine[("B", r["env"], str(r["rate"]), r["n"], name)] = r["miss"]
        hits = 0
        for r in old["rows"]:
            if r["delta"] != DELTA:
                continue
            k = ("A", r["src"], r["env"], r["step"], r["n"], r["arm"]) if r["part"] == "A" else ("B", r["env"], str(r["rate"]), r["n"], r["arm"])
            if k in mine:
                diff = max(diff, abs(mine[k] - r["miss"])); hits += 1
        print(f"replay check: {hits} shared cells, largest difference in miss {diff:.4f}")
    else:
        diff = float("nan")
    res["replay_diff"] = diff
    report(res, os.path.join(OUT, "stratppi_validate.md"))


if __name__ == "__main__":
    if WORKER:
        reference_worker(sys.argv[2], sys.argv[3])
    else:
        main()
