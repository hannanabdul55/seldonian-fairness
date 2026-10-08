"""Two standard two-way bounds on AgentDojo's crossed design (paper section 10.2; plan step R2).

The registration is ``.planning/paper-certification/R2_registration.md``, committed before this
script was run on the benchmark's tables. It fixes the three bounds, the schemes, the sizes, the
seed and the reading. Nothing below was tuned on the tables: ``--selftest`` checks the arithmetic
on random tables only.

Bounds, all upper limits at delta 0.05 on a pipeline's attack success rate:

- ``cgm_t``    multiway cluster-robust variance (Cameron, Gelbach and Miller, 2011),
               ``V2 = V_user + V_inj - V_pair`` (the larger one-way variance if that is not
               positive), with a Student-t quantile on ``min(m_user, m_inj) - 1`` degrees of
               freedom; 1 when the table holds no success.
- ``pig_t``    the pigeonhole bootstrap (Owen, 2007: user tasks and injection tasks resampled
               independently), studentised by ``sqrt(V2)`` in every resample; the bootstrap-t
               limit of section 6.3.
- ``quad``     the two one-way cluster bootstrap-t margins added in quadrature. First computed
               in ``agentdojo_recheck.py`` on these tables with another seed; here it is a
               replication on fresh draws.

Resampling schemes as in ``agentdojo_recheck.py``: (a) user tasks redrawn, (b) injection tasks,
(c) both; the table's own rate is the truth.

    OMP_NUM_THREADS=1 .venv/bin/python scripts/agentdojo_twoway.py --selftest
    OMP_NUM_THREADS=1 .venv/bin/python scripts/agentdojo_twoway.py --workers 12
    -> results/paper/agentdojo_twoway.md, results/paper/agentdojo_twoway.json
    .venv/bin/python scripts/agentdojo_twoway.py --render
"""
import argparse
import collections
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.dont_write_bytecode = True

import numpy as np
from scipy.stats import beta, t as student_t

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import agentdojo_recheck as AR  # noqa: E402

OUT = os.path.join(ROOT, "results", "paper", "agentdojo_twoway")
DELTA, TAU = 0.05, 0.05
SEED = 20261008          # agentdojo_recheck.py used 20261004
BOUNDS = (("naive", "per-pair Clopper-Pearson"), ("t_user", "cluster bootstrap-t by user task"),
          ("t_inj", "cluster bootstrap-t by injection task"), ("cgm_t", "multiway variance, t quantile"),
          ("pig_t", "pigeonhole bootstrap-t"), ("quad", "one-way margins in quadrature"))
REGISTERED = ("cgm_t", "pig_t", "quad")
BIG = 200_000
CHUNK = 4000


# ---------------------------------------------------------------- the multiway variance

def v2_table(Y0, V):
    """est, V2 and its three parts for one table (0/1 outcomes ``Y0`` on the valid cells ``V``)."""
    n, k = V.sum(), Y0.sum()
    est = k / n
    su, nu_ = Y0.sum(1), V.sum(1)
    si, ni_ = Y0.sum(0), V.sum(0)
    mu, mi = int((nu_ > 0).sum()), int((ni_ > 0).sum())
    vu = mu / (mu - 1) * ((su - est * nu_) ** 2).sum() / n ** 2
    vi = mi / (mi - 1) * ((si - est * ni_) ** 2).sum() / n ** 2
    vp = est * (1 - est) / (n - 1)
    v2 = vu + vi - vp
    if v2 <= 0:
        v2 = max(vu, vi)
    return float(est), float(v2), float(vu), float(vi), float(vp), mu, mi


def cgm_t(Y0, V, delta=DELTA):
    est, v2, _, _, _, mu, mi = v2_table(Y0, V)
    if v2 <= 0:
        return 1.0
    return float(np.clip(est + student_t.ppf(1 - delta, min(mu, mi) - 1) * np.sqrt(v2), 0, 1))


def v2_resamples(Y0, V, Wu, Wi):
    """est and V2 of the tables in which user task j appears ``Wu[b, j]`` times and injection task l
    ``Wi[b, l]`` times; a task drawn twice is two clusters."""
    Yf, Vf = Y0.astype(np.float32), V.astype(np.float32)
    A = (Wi @ Yf.T).astype(np.float64)       # successes of each user task under the column weights
    Bz = (Wi @ Vf.T).astype(np.float64)      # its number of pairs
    Cc = (Wu @ Yf).astype(np.float64)
    Dz = (Wu @ Vf).astype(np.float64)
    Wu64, Wi64 = Wu.astype(np.float64), Wi.astype(np.float64)
    nb = (Wu64 * Bz).sum(1)
    kb = (Wu64 * A).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        eb = kb / nb
        mu = (Wu64 * (Bz > 0)).sum(1)
        mi = (Wi64 * (Dz > 0)).sum(1)
        vu = mu / (mu - 1) * (Wu64 * (A - eb[:, None] * Bz) ** 2).sum(1) / nb ** 2
        vi = mi / (mi - 1) * (Wi64 * (Cc - eb[:, None] * Dz) ** 2).sum(1) / nb ** 2
        vp = eb * (1 - eb) / (nb - 1)
        v2 = vu + vi - vp
        v2 = np.where(v2 > 0, v2, np.maximum(vu, vi))
    return eb, v2


def pig_t(Y0, V, delta, rng, boots):
    """Pigeonhole bootstrap-t: ``est - q * sqrt(V2)`` with ``q`` the lower delta-quantile of the
    studentised resampled errors; 1 when that quantile is not finite or the table has no variance."""
    est, v2 = v2_table(Y0, V)[:2]
    if v2 <= 0:
        return 1.0
    nu, ni = Y0.shape
    ts = []
    for b0 in range(0, boots, CHUNK):
        b = min(CHUNK, boots - b0)
        Wu = AR.counts(rng.integers(0, nu, (b, nu)), nu)
        Wi = AR.counts(rng.integers(0, ni, (b, ni)), ni)
        eb, vb = v2_resamples(Y0, V, Wu, Wi)
        with np.errstate(divide="ignore", invalid="ignore"):
            ts.append(np.where(vb > 0, (eb - est) / np.sqrt(vb), np.where(eb < est, -np.inf, np.inf)))
    tt = ts[0] if len(ts) == 1 else np.concatenate(ts)
    tt = np.where(np.isnan(tt), -np.inf, tt)       # an empty resample counts against the bound
    q = np.quantile(tt, delta, method="lower")
    return float(np.clip(est - q * np.sqrt(v2), 0, 1)) if np.isfinite(q) else 1.0


def all_bounds(M, rng, boots, delta=DELTA):
    """Generator calls in a fixed order: t(user), t(inj), pigeonhole."""
    Y0, V = AR.prep(M)
    n, k = int(V.sum()), int(Y0.sum())
    est = k / n
    naive = 1.0 if k >= n else float(beta.ppf(1 - delta, k + 1, n - k))
    tu = AR.cluster_t_arr(V.sum(1), Y0.sum(1), delta, rng, boots)
    ti = AR.cluster_t_arr(V.sum(0), Y0.sum(0), delta, rng, boots)
    pg = pig_t(Y0, V, delta, rng, boots)
    quad = float(np.clip(est + np.sqrt((tu - est) ** 2 + (ti - est) ** 2), 0, 1))
    return naive, tu, ti, cgm_t(Y0, V, delta), pg, quad, k


# ---------------------------------------------------------------- self-test on random tables

def selftest():
    rng = np.random.default_rng(1)
    worst = 0.0
    for case in range(40):
        nu, ni = int(rng.integers(5, 30)), int(rng.integers(4, 12))
        V = rng.random((nu, ni)) < rng.uniform(0.4, 1.0)
        V[np.arange(nu), rng.integers(0, ni, nu)] = True
        V[rng.integers(0, nu, ni), np.arange(ni)] = True
        ru, ci = rng.beta(0.5, 3, nu), rng.beta(0.5, 3, ni)
        Y0 = ((rng.random((nu, ni)) < np.clip(ru[:, None] + ci[None, :], 0, 1)) & V).astype(float)
        b = 60
        iu, ij = rng.integers(0, nu, (b, nu)), rng.integers(0, ni, (b, ni))
        eb, vb = v2_resamples(Y0, V, AR.counts(iu, nu), AR.counts(ij, ni))
        for r in range(b):          # the same tables built row by row and column by column
            Yr, Vr = Y0[iu[r]][:, ij[r]], V[iu[r]][:, ij[r]]
            if Vr.sum() < 2 or (Vr.sum(1) > 0).sum() < 2 or (Vr.sum(0) > 0).sum() < 2:
                continue
            e, v2 = v2_table(Yr, Vr)[:2]
            worst = max(worst, abs(e - eb[r]), abs(v2 - vb[r]) / max(v2, 1e-12) if v2 > 0 else abs(vb[r]))
    assert worst < 1e-6, worst
    Z, Vz = np.zeros((6, 5)), np.ones((6, 5), bool)
    assert cgm_t(Z, Vz) == 1.0 and pig_t(Z, Vz, DELTA, rng, 200) == 1.0, "no success must give no limit"
    one = Z.copy()
    one[0, 0] = 1
    assert 1 / 30 < cgm_t(one, Vz) <= 1.0 and 1 / 30 < pig_t(one, Vz, DELTA, rng, 2000) <= 1.0
    # a table with independent cells: all three bounds should sit near their level
    miss = collections.Counter()
    reps = 400
    for _ in range(reps):
        Y = (rng.random((40, 20)) < 0.2).astype(float)
        out = all_bounds(Y, rng, 500)
        for (key, _), v in zip(BOUNDS, out):
            miss[key] += v < 0.2
    print("vectorised V2 against the direct form, largest difference:", worst)
    print("independent cells at a 20% rate, 40 x 20, miss over", reps, "tables:",
          {k: round(v / reps, 3) for k, v in miss.items()})


# ---------------------------------------------------------------- the registered run

def run_cell(args):
    p, pi, scheme, si, M, reps, boots, seed = args
    rng = np.random.default_rng(np.random.SeedSequence([seed, pi, si]))
    nu, ni = M.shape
    out = np.empty((reps, 7))
    for r in range(reps):
        iu = rng.integers(0, nu, nu) if scheme in "ac" else np.arange(nu)
        ij = rng.integers(0, ni, ni) if scheme in "bc" else np.arange(ni)
        out[r] = all_bounds(M[iu][:, ij], rng, boots)
    return p, scheme, out


def real_cell(args):
    p, M, seeds, boots = args
    Y0, V = AR.prep(M)
    n, k = int(V.sum()), int(Y0.sum())
    est = k / n
    big = np.random.default_rng([SEED, 99])
    tu = AR.cluster_t_arr(V.sum(1), Y0.sum(1), DELTA, big, BIG)
    ti = AR.cluster_t_arr(V.sum(0), Y0.sum(0), DELTA, big, BIG)
    pg = pig_t(Y0, V, DELTA, big, BIG)
    e, v2, vu, vi, vp, mu, mi = v2_table(Y0, V)
    conv = dict(naive=1.0 if k >= n else float(beta.ppf(1 - DELTA, k + 1, n - k)), t_user=tu, t_inj=ti,
                cgm_t=cgm_t(Y0, V), pig_t=pg, quad=float(np.clip(est + np.sqrt((tu - est) ** 2 + (ti - est) ** 2), 0, 1)))
    spread = collections.defaultdict(list)
    for s in seeds:
        rng = np.random.default_rng([SEED, 1000 + s])
        vals = all_bounds(M, rng, boots)
        for (key, _), v in zip(BOUNDS, vals):
            spread[key].append(float(v))
    return p, dict(n=n, k=k, rate=est, m_user=mu, m_inj=mi, v_user=vu, v_inj=vi, v_pair=vp, v2=v2, converged=conv,
                   spread={key: dict(min=min(v), max=max(v)) for key, v in spread.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=4000)
    ap.add_argument("--boots", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--render", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.render:
        return report(json.load(open(OUT + ".json")))
    t0 = time.time()
    R_by, _ = AR.load_rows()
    tables = {p: AR.table(R)[0] for p, R in R_by.items()}
    pipes = list(tables)
    truth = {p: float(np.nanmean(tables[p])) for p in pipes}
    jobs = [(p, pi, sc, si, tables[p], a.reps, a.boots, SEED)
            for pi, p in enumerate(pipes) for si, (sc, _) in enumerate(AR.SCHEMES)]
    sim, real = {p: {} for p in pipes}, {}
    with ProcessPoolExecutor(a.workers) as ex:
        for p, d in ex.map(real_cell, [(p, tables[p], list(range(a.seeds)), a.boots) for p in pipes]):
            real[p] = d
        print(f"real-data limits done {time.time() - t0:.0f}s", flush=True)
        for i, (p, sc, out) in enumerate(ex.map(run_cell, jobs)):
            s = {}
            for j, (key, _) in enumerate(BOUNDS):
                v = out[:, j]
                s[key] = dict(miss=float(np.mean(v < truth[p])), median=float(np.median(v)),
                              no_limit=float(np.mean(v >= 1.0)), pass_at_tau=float(np.mean(v <= TAU)))
            s["zero_success_resamples"] = float(np.mean(out[:, 6] == 0))
            sim[p][sc] = s
            if (i + 1) % 12 == 0:
                print(f"{i + 1}/{len(jobs)} cells {time.time() - t0:.0f}s", flush=True)
    res = dict(args=vars(a), seed=SEED, delta=DELTA, mc_se=float(np.sqrt(DELTA * (1 - DELTA) / a.reps)),
               truth=truth, resampling=sim, real=real, seconds=time.time() - t0)
    json.dump(res, open(OUT + ".json", "w"), indent=1)
    report(res)


def report(res):
    sim, real, reps = res["resampling"], res["real"], res["args"]["reps"]
    pipes = list(sim)
    thr = DELTA + 2 * res["mc_se"]
    L = ["# Two-way bounds on AgentDojo's crossed design (paper section 10.2)", "",
         f"`scripts/agentdojo_twoway.py`, registered in `.planning/paper-certification/R2_registration.md` before the "
         f"run. {len(pipes)} pipelines, {reps:,} resampled tables per pipeline and scheme, {res['args']['boots']:,} inner "
         f"bootstrap draws, delta {DELTA}, seed {res['seed']}. A cell is over its level above a miss of {thr:.4f}.", "",
         "## 1. Cells over / unresolved / at or under, by scheme", "",
         "| bound | (a) user tasks resampled | (b) injection tasks resampled | (c) both | largest miss (a); (b); (c) |",
         "|---|---|---|---|---|"]
    dig = {}
    for key, name in BOUNDS:
        row, worst = [], []
        dig[key] = {}
        for sc, _ in AR.SCHEMES:
            ms = {p: sim[p][sc][key]["miss"] for p in pipes}
            cl = collections.Counter(AR.classify(m, reps) for m in ms.values())
            row.append(f"{cl['over']} / {cl['unresolved']} / {cl['under']}")
            worst.append(f"{max(ms.values()):.3f}")
            dig[key][sc] = dict(over=cl["over"], unresolved=cl["unresolved"], under=cl["under"], worst=max(ms.values()),
                                least=min(ms.values()), over_pipelines=[p for p in pipes if AR.classify(ms[p], reps) == "over"])
        L.append(f"| {name}{' (registered)' if key in REGISTERED else ''} | " + " | ".join(row) + " | " + "; ".join(worst) + " |")
    res["digest"] = dig
    L += ["", "## 2. Width on the published tables", "",
          f"Limits from {BIG:,} inner draws. The margin is the limit minus the rate; the ratio is to the margin of the "
          "bound clustered by user task.", "",
          "| bound | median margin ratio | range | pipelines certifying 5% | no limit returned |", "|---|---|---|---|---|"]
    width = {}
    for key, name in BOUNDS:
        ratio = [(real[p]["converged"][key] - real[p]["rate"]) / (real[p]["converged"]["t_user"] - real[p]["rate"]) for p in pipes]
        passed = [p for p in pipes if real[p]["converged"][key] <= TAU]
        nolim = sum(real[p]["converged"][key] >= 1.0 for p in pipes)
        width[key] = dict(median=float(np.median(ratio)), min=float(min(ratio)), max=float(max(ratio)), passes=passed, no_limit=int(nolim))
        L.append(f"| {name} | {np.median(ratio):.2f} | {min(ratio):.2f} to {max(ratio):.2f} | {len(passed)}"
                 f"{' (' + ', '.join(passed) + ')' if passed else ''} | {nolim} |")
    res["width"] = width
    L += ["", "## 3. Per pipeline", "",
          "Miss under scheme (c), and the limit on the published table. **Bold** is over the level.", "",
          "| pipeline | pairs | rate | " + " | ".join(f"{k} miss (c)" for k in REGISTERED) + " | "
          + " | ".join(f"{k} limit" for k in ("t_user",) + REGISTERED) + " |", "|---|---|---|" + "---|" * 7]
    for p in pipes:
        d = real[p]
        cells = []
        for k in REGISTERED:
            m = sim[p]["c"][k]["miss"]
            c = AR.classify(m, reps)
            cells.append(f"**{m:.3f}**" if c == "over" else (f"{m:.3f}?" if c == "unresolved" else f"{m:.3f}"))
        L.append(f"| {p} | {d['n']} | {d['rate']:.3f} | " + " | ".join(cells) + " | "
                 + " | ".join(f"{d['converged'][k]:.3f}" for k in ("t_user",) + REGISTERED) + " |")
    L += ["", "## 4. Spread over bootstrap seeds on the published tables", "",
          f"Largest range of a limit over {res['args']['seeds']} seeds at {res['args']['boots']:,} inner draws: "
          + "; ".join(f"{name} {max(real[p]['spread'][key]['max'] - real[p]['spread'][key]['min'] for p in pipes):.3f}"
                      for key, name in BOUNDS if key not in ("naive", "cgm_t")) + "."]
    json.dump(res, open(OUT + ".json", "w"), indent=1)
    open(OUT + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
