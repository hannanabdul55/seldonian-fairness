"""M1 (blind review of v0.9.11): how often the bounds of Table 4 return no limit, and the Monte Carlo
error of the labelling-sheet rows with plantings as clusters. NOT registered: written after every result
it re-reads was seen; it changes no draw and no bound, it only counts.

Part 1, abstention. A bound with no limit returns 1 (no certificate), which cannot fall below the
truth, so the harnesses score it as a non-miss. Here each such row is read two ways: as published
(no limit = no certificate, NSF) and conservatively (no limit = a miss). A draw "returns no limit" when
its upper limit is 1 or more. Where a harness stored that share (``agentdojo_twoway.py``,
``agentdojo_betting.py``) it is read; elsewhere the harness's own draws are replayed with its seeds and
the bound recomputed with a copy that also reports the share, and the replayed miss rate must equal the
stored one to the last digit (asserted), so the counts are of the published draws:

- ``recheck``      ``agentdojo_recheck.py``'s resampling cells (Table 4 rows 10, 25, 26, 27)
- ``parametric``   ``agentdojo_parametric.py``'s cells (rows 49 to 52), with its stored fits
- ``stratppi_a``   ``stratppi_baseline.py`` part A, the bootstrap-t StratPPI limit (row 11, first part)
- ``stratppi_b``   the same, part B, judge-logit strata (row 11, second part; row 22's cells are not replayed)
- ``replacement``  ``replacement_check.py``'s 40,000-draw cells of the same limit (row 12)
- ``ppipp017``     spike 017's PPI++ with a bootstrap-t limit (row 9)

Part 2, the sheet rows (rows 8 and 28): ``harm017.py``'s 20 plantings x 200 redrawn sheets, replayed
with its generator; the miss rate's standard error with the 20 plantings as clusters,
sd(per-planting miss) / sqrt(20), against the binomial one over 4,000 draws.

    OMP_NUM_THREADS=1 .venv/bin/python scripts/m1_abstention_mc.py <stage> [--workers N]
    stages: stored recheck parametric stratppi_a stratppi_b replacement ppipp017 sheet report
    -> results/paper/m1_abstention_mc/<stage>.json; report -> results/paper/m1_abstention_mc.{json,md}
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

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "results", "paper")
PARTS = os.path.join(OUT, "m1_abstention_mc")
D17 = os.path.join(ROOT, ".planning", "spikes", "017-calibration-carrying-certificate")
sys.path.insert(0, HERE)
sys.path.insert(0, D17)


def classify(m, delta, R):
    se = np.sqrt(delta * (1 - delta) / R)
    return "over" if m > delta + 2 * se else ("unresolved" if m > delta else "at or under")


def cellrec(label, delta, R, miss, nolimit, **kw):
    return dict(label=label, delta=delta, reps=R, miss=float(miss), no_limit=float(nolimit),
                miss_cons=float(miss + nolimit), cls=classify(miss, delta, R), cls_cons=classify(miss + nolimit, delta, R), **kw)


def save(stage, groups):
    os.makedirs(PARTS, exist_ok=True)
    json.dump(groups, open(os.path.join(PARTS, stage + ".json"), "w"), indent=1)
    print(f"-> {PARTS}/{stage}.json", flush=True)


# ------------------------------------------------------------------ stored shares (AgentDojo)
def stage_stored(a):
    g = {}
    d = json.load(open(os.path.join(OUT, "agentdojo_twoway.json")))
    R = d["args"]["reps"]
    for rule, row in (("pig_t", 14), ("cgm_t", 30), ("quad", 31), ("t_user", 0)):
        for sch in "abc":
            g[f"tw_{rule}_{sch}"] = dict(row=row, source="agentdojo_twoway.json", cells=[
                cellrec(p, 0.05, R, v[sch][rule]["miss"], v[sch][rule]["no_limit"]) for p, v in d["resampling"].items()])
    d = json.load(open(os.path.join(OUT, "agentdojo_betting.json")))
    R = d["args"]["reps"]
    rows = {("bet_a1@uavg", "a"): 40, ("bet_a1@uavg", "a_s"): 40, ("bet_a2", "a"): 40, ("bet_a2", "a_s"): 40,
            ("t_user", "a_s"): 41, ("t_user", "b_s"): 42, ("t_user", "c_s"): 42,
            ("quad", "a_s"): 43, ("quad", "b_s"): 43, ("quad", "c_s"): 43}
    for (arm, sch), row in rows.items():
        g[f"o4_{arm}_{sch}"] = dict(row=row, source="agentdojo_betting.json", cells=[
            cellrec(p, 0.05, R, v[sch][arm]["miss"], v[sch][arm]["no_limit"]) for p, v in d["resampling"].items()])
    save("stored", g)


# ------------------------------------------------------------------ AgentDojo recheck, replayed
def _recheck_job(args):
    import agentdojo_recheck as AR
    p, pi, sc, si, M, reps, boots, seed, truth = args
    _, _, out = AR.run_cell((p, pi, sc, si, M, reps, boots, seed))
    b = AR.rules(*out.T)
    return p, sc, {k: (float(np.mean(b[k] < truth)), float(np.mean(b[k] >= 1.0))) for k, _ in AR.BOUNDS}


def stage_recheck(a):
    import agentdojo_recheck as AR
    d = json.load(open(os.path.join(OUT, "agentdojo_recheck.json")))
    R_by, _ = AR.load_rows()
    pipes = list(R_by)
    jobs = [(p, pi, sc, si, AR.table(R_by[p])[0], d["args"]["reps"], d["args"]["boots"], d["args"]["seed"],
             sum(r["y"] for r in R_by[p]) / len(R_by[p]))
            for pi, p in enumerate(pipes) for si, (sc, _) in enumerate(AR.SCHEMES)]
    res, t0 = collections.defaultdict(dict), time.time()
    with ProcessPoolExecutor(min(a.workers, 4)) as ex:
        for i, (p, sc, s) in enumerate(ex.map(_recheck_job, jobs), 1):
            res[p][sc] = s
            print(f"recheck {i}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    R = d["args"]["reps"]
    rows = {("t_user", "a"): 10, ("naive", "a"): 24, ("naive", "b"): 24, ("naive", "c"): 24, ("t_user", "b"): 25,
            ("t_user", "c"): 25, ("max", "a"): 26, ("max", "b"): 26, ("max", "c"): 26, ("twoway", "a"): 27,
            ("twoway", "b"): 27, ("twoway", "c"): 27}
    g = {}
    for (rule, sch), row in rows.items():
        cells = []
        for p in pipes:
            m, nl = res[p][sch][rule]
            assert m == d["resampling"][p][sch][rule]["miss"], (p, sch, rule, m)      # the published draws
            cells.append(cellrec(p, 0.05, R, m, nl))
        g[f"ad_{rule}_{sch}"] = dict(row=row, source="agentdojo_recheck.py replayed", cells=cells)
    save("recheck", g)


# ------------------------------------------------------------------ AgentDojo fitted model, replayed
def _param_job(args):
    import agentdojo_parametric as APm
    p, sc, M, U, f, reps, boots = args
    _, _, out, tr = APm.run_cell((p, sc, M, U, f, reps, boots))
    s = APm.summarise(out, tr)
    c = {k: out[:, j] for j, k in enumerate(APm.AB.COLS)}
    nl = {k: float(np.mean(c["bet_a1" if k == "bet_a1@uavg" else k] >= 1.0)) for k in APm.ARMS}
    return p, sc, {k: (s[k]["miss"], nl[k]) for k in APm.ARMS}


def stage_parametric(a):
    import agentdojo_parametric as APm
    d = json.load(open(os.path.join(OUT, "agentdojo_parametric.json")))
    tables = APm.AB.load()
    jobs = [(p, sc, tables[p][0], tables[p][1], d["fits"][p], d["reps"], d["boots"]) for p in tables for sc, _ in APm.SCHEMES]
    res, t0 = collections.defaultdict(dict), time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (p, sc, s) in enumerate(ex.map(_param_job, jobs), 1):
            res[p][sc] = s
            print(f"parametric {i}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    rows = {("t_user", "P_a"): 49, ("t_user", "P_c"): 50, ("bet_a1@uavg", "P_a"): 51, ("bet_a2", "P_a"): 51,
            ("pig_t", "P_a"): 52, ("pig_t", "P_c"): 52, ("quad", "P_a"): 52, ("quad", "P_c"): 52}
    g = {}
    for (arm, sch), row in rows.items():
        cells = []
        for p in tables:
            m, nl = res[p][sch][arm]
            assert m == d["sim"][p][sch][arm]["miss"], (p, sch, arm, m)
            cells.append(cellrec(p, 0.05, d["reps"], m, nl))
        g[f"b6_{arm}_{sch}"] = dict(row=row, source="agentdojo_parametric.py replayed", cells=cells)
    save("parametric", g)


# ------------------------------------------------------------------ StratPPI estimator, bootstrap-t
def stratppi_boot_flag(y, f, st, pool, W, deltas, rng, boots=300):
    """``stratppi_baseline.stratppi_boot`` line for line, returning also whether each limit is 1 or more."""
    import stratppi_baseline as BL
    est, var = BL.stratppi_point(y, f, st, pool, W)
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
    return out


def _sppi_a_job(args):
    import stratppi_baseline as BL
    src, pool, lab, step, reps = args
    PM, RP, P14 = BL.PM, BL.RP, BL.P14
    PM.TIES = "random"
    by, meta = RP.load() if src == "013" else P14.load("s0")
    P, _ = RP.build_pool(by, meta, pool, lab, step, False, "eval")
    cov, draw, truth = P["cov"], P["draw"], float(P["truth"])
    N = len(P["p_truth"])
    f = cov[:, :BL.K_REF].mean(axis=1)
    rng = np.random.default_rng([step, zlib.crc32(repr(("S2", BL.K_REF, BL.H)).encode())])
    strata = PM.quantile_strata(f, BL.H, rng, ties="random")
    W = np.bincount(strata) / N
    stats = [(int((strata == h).sum()), float(f[strata == h].mean()), float(f[strata == h].var(ddof=1))) for h in range(BL.H)]
    res = {}
    for n_s in (100, 200):
        ub = []
        for r in range(reps):
            rr = np.random.default_rng([step, r, n_s])
            ids, st = PM._draw_split(strata, n_s, rr)
            y = draw(ids, rr)
            ub.append(stratppi_boot_flag(y, f[ids], st, stats, W, BL.DELTAS, rr))
        ub = np.array(ub)
        for j, d in enumerate(BL.DELTAS):
            res[(n_s, d)] = (float((~(ub[:, j] >= truth)).mean()), float((ub[:, j] >= 1.0).mean()))
    return src, f"{pool}:{lab}", step, truth, res


def stage_stratppi_a(a):
    import stratppi_baseline as BL
    st = json.load(open(os.path.join(OUT, "stratppi.json")))
    jobs = [("013", p, lab, s, st["reps"]) for p, labs in BL.RP.LABELS.items() for lab in labs for s in (0, 100, 200)]
    jobs += [("014", "C1", "refusal", s, st["reps"]) for s in (100, 200)]
    stored = {(r["src"], r["env"], r["step"], r["n"], r["delta"]): r for r in st["rows"]
              if r["part"] == "A" and r["arm"] == "S2 + StratPPI, bootstrap-t"}
    cells, t0 = collections.defaultdict(list), time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (src, env, step, truth, res) in enumerate(ex.map(_sppi_a_job, jobs), 1):
            for (n_s, d), (m, nl) in res.items():
                assert m == stored[(src, env, step, n_s, d)]["miss"], (src, env, step, n_s, d, m)
                cells[d].append(cellrec(f"{env}{' pushed (014)' if src == '014' else ''}, n_s {n_s}, step {step}", d,
                                        st["reps"], m, nl, env=env, truth=truth))
            print(f"stratppi_a {i}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    save("stratppi_a", {f"p14_a_sboot_{d}": dict(row=11 if d == 0.05 else 0, source="stratppi_baseline.py part A replayed",
                                                 cells=c) for d, c in cells.items()})


def _sppi_b_job(args):
    import stratppi_baseline as BL
    PL = BL.PL
    key, y_pool, p_pool, n, big_n, reps, rate, seed = args
    rng = np.random.default_rng(seed)
    truth = float(y_pool.mean()) if rate is None else rate
    d = PL.DELTA
    x_pool = PL.logit01(p_pool)
    q = np.full(len(y_pool), 1.0 / len(y_pool)) if rate is None else np.where(
        y_pool == 1, rate / (y_pool == 1).sum(), (1 - rate) / (y_pool == 0).sum())
    order = np.lexsort((rng.random(len(y_pool)), x_pool))
    cum = np.cumsum(q[order]) - q[order] / 2
    strata_b = {}
    for K in (5, 10):
        st_pool = np.empty(len(y_pool), dtype=int); st_pool[order] = np.minimum((cum * K).astype(int), K - 1)
        strata_b[K] = (st_pool, np.array([q[st_pool == h].sum() for h in range(K)]), q)
    ub = {5: [], 10: []}
    done = 0
    while done < reps:
        b = min(500, reps - done)
        PL.sample(rng, y_pool, p_pool, b, big_n, rate)          # consumed as in part_b; its arms use no other draw
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
                ub[K].append(stratppi_boot_flag(yy, ff, st, pool_stats, W, (d,), rng)[0])
        done += b
    env = "|".join(map(str, key))
    return env, n, big_n, "pool" if rate is None else rate, truth, {
        K: (float((~(np.array(v) >= truth)).mean()), float((np.array(v) >= 1.0).mean())) for K, v in ub.items()}


def stage_stratppi_b(a):
    import stratppi_baseline as BL
    st = json.load(open(os.path.join(OUT, "stratppi.json")))
    pools = BL.PL.load_pools()
    jobs, seed = [], 7000
    for key in (("refusal", "rubric", 0), ("refusal", "raw", 0)):
        for n, big_n, rate in ((100, 2000, None), (225, 2000, None), (500, 2000, None), (225, 4000, 0.05), (1000, 4000, 0.05),
                               (225, 4000, 0.013), (1000, 4000, 0.013)):
            seed += 1
            jobs.append((key, *pools[key], n, big_n, st["reps_b"], rate, seed))
    stored = {(r["env"], r["n"], r["N"], str(r["rate"]), r["arm"]): r for r in st["rows"] if r["part"] == "B"}
    cells, t0 = [], time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (env, n, big_n, rate, truth, res) in enumerate(ex.map(_sppi_b_job, jobs), 1):
            for K, (m, nl) in res.items():
                assert m == stored[(env, n, big_n, str(rate), f"StratPPI, bootstrap-t, K={K}")]["miss"], (env, n, K, m)
                cells.append(cellrec(f"{env.replace('|', ' ')}, rate {rate}, n {n} of {big_n}, K={K}", 0.05, st["reps_b"], m, nl,
                                     truth=truth))
            print(f"stratppi_b {i}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    save("stratppi_b", {"p14_b_sboot": dict(row=11, source="stratppi_baseline.py part B replayed", cells=cells)})


def _repl_job(args):
    import replacement_check as RC
    BL = RC.BL
    src, pool, lab, step, big = args
    P, f, strata = RC.setup(src, pool, lab, step)
    draw, truth, N = P["draw"], float(P["truth"]), len(P["p_truth"])
    W = np.bincount(strata) / N
    stats = [(RC.UNBOUNDED, float(f[strata == h].mean()), float(f[strata == h].var(ddof=1))) for h in range(RC.H)]
    own = zlib.crc32(f"{src}:{pool}:{lab}".encode())
    res = {}
    for n_s in (100, 200):
        ub = []
        for r in range(big):
            rr = np.random.default_rng([own, step, r, n_s])
            ids, st = RC.draw_split(strata, n_s, rr, True)
            y = draw(ids, rr)
            ub.append(stratppi_boot_flag(y, f[ids], st, stats, W, BL.DELTAS, rr))
        ub = np.array(ub)
        for j, d in enumerate(BL.DELTAS):
            res[(n_s, d)] = (float((~(ub[:, j] >= truth)).mean()), float((ub[:, j] >= 1.0).mean()))
    return src, f"{pool}:{lab}", step, truth, res


def stage_replacement(a):
    import replacement_check as RC
    rc = json.load(open(os.path.join(OUT, "replacement_check.json")))
    B = rc["reps_large"]
    stored = {(r["src"], r["env"], r["step"], r["n"], r["delta"]): r for r in rc["rows"]
              if r["arm"] == "StratPPI estimator, bootstrap-t" and r["draw"] == "large"}
    jobs = [("013", p, lab, s, B) for p, labs in RC.RP.LABELS.items() for lab in labs for s in (0, 100, 200)
            if f"{p}:{lab}" in RC.MID]
    jobs += [("014", "C1", "refusal", s, B) for s in (100, 200)]
    cells, t0 = collections.defaultdict(list), time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (src, env, step, truth, res) in enumerate(ex.map(_repl_job, jobs), 1):
            for (n_s, d), (m, nl) in res.items():
                assert m == stored[(src, env, step, n_s, d)]["miss"], (src, env, step, n_s, d, m)
                cells[d].append(cellrec(f"{env}{' pushed (014)' if src == '014' else ''}, n_s {n_s}, step {step}", d, B, m, nl,
                                        env=env, truth=truth))
            print(f"replacement {i}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    save("replacement", {f"rep_sboot_{d}": dict(row=12, source="replacement_check.py large draws replayed", cells=c)
                         for d, c in cells.items()})


# ------------------------------------------------------------------ spike 017 PPI++ bootstrap-t
def ppipp_boot_flag(y_lab, f_lab, f_unl, delta, boots=300, seed=0):
    """``cert017.ppipp_boot`` (lam=None, degenerate="sign") line for line."""
    import cert017 as c
    y, fl, fu = c._2d(y_lab, f_lab, f_unl)
    reps, n = y.shape
    nu = fu.shape[1]
    rng = np.random.default_rng(seed)
    est, var, _ = c.ppi_point(y, fl, fu, None)
    se = np.sqrt(var)
    fbar_u = fu.mean(1)
    var_fu = fu.var(axis=1, ddof=1)
    out = np.empty(reps)
    for r in range(reps):
        idx = rng.integers(0, n, size=(boots, n))
        ys, fs = y[r][idx], fl[r][idx]
        ym, fm = ys.mean(1), fs.mean(1)
        fub = fbar_u[r] + np.sqrt(var_fu[r] / nu) * rng.standard_normal(boots)
        cov = ((ys - ym[:, None]) * (fs - fm[:, None])).sum(1) / (n - 1)
        lb = cov / ((1 + n / nu) * var_fu[r]) if var_fu[r] > 0 else np.zeros(boots)
        eb = ym + lb * (fub - fm)
        vb = (ys - lb[:, None] * fs).var(axis=1, ddof=1) / n + lb ** 2 * var_fu[r] / nu
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(vb > 0, (eb - est[r]) / np.sqrt(vb), np.where(eb < est[r], -np.inf, np.inf))
        q = np.quantile(t, delta, method="lower")
        out[r] = est[r] - q * se[r] if np.isfinite(q) else 1.0
    return np.clip(out, 0.0, 1.0)


def _ppipp_job(job):
    import plasmode017 as pm
    key, y_pool, p_pool, n, big_n, reps, rate, seed, block_reps = job
    rng = np.random.default_rng(seed)
    truth = float(y_pool.mean()) if rate is None else rate
    batch = 250 if big_n > 5000 else 500
    ub = collections.defaultdict(list)
    done = 0
    while done < reps:
        b = min(batch, reps - done)
        y, p = pm.sample(rng, y_pool, p_pool, b, big_n, rate)
        yl, pl, pu = y[:, :n], p[:, :n], p[:, n:]
        feats = pm.features(pl, pu, yl)
        for fname, (fl, fu) in feats.items():
            ub[fname].append(ppipp_boot_flag(yl, fl, fu, pm.DELTA, seed=seed + done))
        if done < block_reps:
            pm.sample(rng, y_pool, p_pool, min(b, block_reps - done), n, rate)      # the block arm's pilot draw
        done += b
    return dict(task=key[0], variant=key[1], wording=key[2], n=n, N=big_n, rate=truth, feats={
        f: (float((np.concatenate(v) < truth).mean()), float((np.concatenate(v) >= 1.0).mean()), int(sum(len(x) for x in v)))
        for f, v in ub.items()})


def stage_ppipp017(a):
    import plasmode017 as pm
    import plasmode017_big as big
    pools = pm.load_pools()
    jobs, seed = [], 0
    for key in sorted(pools):                                  # plasmode017.main, base
        if key[0] == "harm":
            continue
        for n in ((50, 100, 225, 500) if key[2] == 0 else (225,)):
            seed += 1
            jobs.append((key, *pools[key], n, 2000, 4000, None, seed, 1000))
    jobs += big.jobs_at(4000, pools)                           # the four large-set cells, continued to 4,000
    seed = 0
    for key in (("refusal", "rubric", 0), ("refusal", "raw", 0)):  # plasmode017.main --shift
        for rate in (0.2, 0.05, 0.013):
            for n in (225, 1000):
                seed += 1
                jobs.append((key, *pools[key], n, 4000, 4000, rate, 1000 + seed, 1000))
    rows = json.load(open(os.path.join(D17, "plasmode.json")))["rows"] + json.load(open(os.path.join(D17, "plasmode_shift.json")))["rows"]
    rows += json.load(open(os.path.join(OUT, "plasmode017_big.json")))["rows"]
    stored = {}
    for r in rows:
        if r["method"] == "boot":
            stored[(r["task"], r["variant"], r["wording"], r["n"], r["N"], round(r["rate"], 6), r["feat"])] = r
    cells, t0 = [], time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for i, res in enumerate(ex.map(_ppipp_job, jobs), 1):
            for fname, (m, nl, R) in res["feats"].items():
                s = stored[(res["task"], res["variant"], res["wording"], res["n"], res["N"], round(res["rate"], 6), fname)]
                assert R == s["reps"] and m == s["miss"], (res, fname, m, s["miss"])
                cells.append(cellrec(f"{res['task']} {res['variant']} {res['wording']}, rate {res['rate']:.3f}, n {res['n']} of "
                                     f"{res['N']}, {fname}", 0.05, R, m, nl, rate=res["rate"], n=res["n"]))
            print(f"ppipp017 {i}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    save("ppipp017", {"017_boot": dict(row=9, source="spike 017 plasmodes replayed (plasmode017.py, plasmode017_big.py)",
                                       cells=cells)})


# ------------------------------------------------------------------ part 2: the sheet rows, clustered
def stage_sheet(a):
    import design017 as dz
    import harm017 as hm
    d = dz.load()
    scores = hm.pop_scores(d)
    rng = np.random.default_rng(17)
    hp, big_np, cells = hm.design_strata(d)
    stored = {(r["wording"], r["rate"], r["feat"], r["route"]): r["miss"] for r in json.load(open(os.path.join(D17, "harm.json")))["planted"]}
    out = collections.defaultdict(list)
    for w in sorted(scores):
        fw = hm.FEATS["f01"](scores[w])
        for rate in (0.013, 0.05, 0.2):
            for fname in ("f01", "logit"):
                f = hm.FEATS[fname](scores[w])
                per = collections.defaultdict(list)
                for _ in range(20):
                    yp = hm.plant(rng, d, fw, rate)
                    truth = yp.mean()
                    k = collections.Counter()
                    for _ in range(200):
                        idx = hm.redraw(rng, cells)
                        est, _ = hm.estimates(hp, big_np, idx, yp[idx], f)
                        for r in ("weighted labels, b1w", "PPI as i.i.d."):
                            k[r] += est[r][1] < truth
                    for r in ("weighted labels, b1w", "PPI as i.i.d."):
                        per[r].append(k[r] / 200)
                for r, v in per.items():
                    v = np.array(v)
                    m = float(v.mean())
                    assert abs(m - stored[(w, rate, fname, r)]) < 1e-12, (w, rate, fname, r, m)
                    se_b = float(np.sqrt(0.05 * 0.95 / 4000))
                    se_c = float(v.std(ddof=1) / np.sqrt(len(v)))
                    se_c0 = float(np.sqrt(max(v.var(ddof=1), 0.05 * 0.95 / 200) / len(v)))      # never below the binomial within a planting
                    out[r].append(dict(label=f"wording {w}, planted rate {rate}, {fname}", wording=w, rate=rate, feat=fname, miss=m,
                                       per_planting=v.tolist(), se_binom=se_b, se_cluster=se_c,
                                       deff=float((se_c / se_b) ** 2) if se_b > 0 else None,
                                       cls_binom=classify(m, 0.05, 4000),
                                       cls_cluster="over" if m > 0.05 + 2 * se_c0 else ("unresolved" if m > 0.05 else "at or under")))
                print(f"sheet {w} {rate} {fname}", flush=True)
    save("sheet", {"017_sheet_b1w": dict(row=8, cells=out["weighted labels, b1w"]),
                   "017_sheet_ppi_iid": dict(row=28, cells=out["PPI as i.i.d."])})


# ------------------------------------------------------------------ report
def stage_report(a):
    G, L = {}, []
    for st in ("stored", "recheck", "parametric", "stratppi_a", "stratppi_b", "replacement", "ppipp017"):
        p = os.path.join(PARTS, st + ".json")
        if os.path.exists(p):
            G.update(json.load(open(p)))
    for gid in [k for k in G if k.startswith("p14_a_sboot_")]:     # row 11 holds the mid-rate labels; the rare ones apart
        g, d = G.pop(gid), gid.rsplit("_", 1)[1]
        mid = ("C1:refusal", "C2:unsafe", "C2:refusal", "C3:refusal")
        G[f"p14_a_sboot_mid_{d}"] = dict(g, row=11 if d == "0.05" else 0, cells=[c for c in g["cells"] if c["env"] in mid])
        G[f"p14_a_sboot_rare_{d}"] = dict(g, row=0, cells=[c for c in g["cells"] if c["env"] not in mid])
    sheet = json.load(open(os.path.join(PARTS, "sheet.json"))) if os.path.exists(os.path.join(PARTS, "sheet.json")) else {}
    cnt = lambda cs, k: tuple(sum(c[k] == x for c in cs) for x in ("over", "unresolved", "at or under"))   # noqa: E731
    summary = {}
    L += ["# M1: abstention and clustered Monte Carlo error (not registered)", "",
          "`scripts/m1_abstention_mc.py`. Written after every result it reads was seen; no draw or bound is changed. A draw "
          "*returns no limit* when its upper limit is 1 or more (no certificate). As published, such a draw is not a miss; "
          "read conservatively it is. Counts are over / unresolved / at or under by the paper's rule (se from the draw count). "
          "Replayed rows reproduce the published miss rate of every cell exactly (asserted).", "",
          "## 1. Abstention, by Table 4 row", "",
          "| row | group | cells | largest share with no limit | cells with any | counts as published | counts, no limit = miss | largest miss, no limit = miss |",
          "|---|---|---|---|---|---|---|---|"]
    for gid, g in sorted(G.items(), key=lambda kv: (kv[1]["row"] or 99, kv[0])):
        cs = g["cells"]
        s = dict(row=g["row"], cells=len(cs), max_no_limit=max(c["no_limit"] for c in cs),
                 any_no_limit=sum(c["no_limit"] > 0 for c in cs), counts=cnt(cs, "cls"), counts_cons=cnt(cs, "cls_cons"),
                 max_miss_cons=max(c["miss_cons"] for c in cs),
                 worst=max(cs, key=lambda c: c["no_limit"])["label"])
        summary[gid] = s
        L.append(f"| {g['row'] or '-'} | {gid} | {s['cells']} | {s['max_no_limit']:.4f} ({s['worst']}) | {s['any_no_limit']} | "
                 f"{' / '.join(map(str, s['counts']))} | {' / '.join(map(str, s['counts_cons']))} | {s['max_miss_cons']:.4f} |")
    if sheet:
        L += ["", "## 2. The sheet rows with plantings as clusters", "",
              "20 plantings x 200 sheets a cell. se(binomial) treats the 4,000 sheets as independent; se(cluster) is the sd of the "
              "20 per-planting miss rates over sqrt(20). Classified with the clustered se (never below the binomial within a "
              "planting).", "",
              "| row | cell | miss | se binomial | se cluster | design effect | class, binomial | class, clustered |",
              "|---|---|---|---|---|---|---|---|"]
        for gid, g in sheet.items():
            for c in g["cells"]:
                L.append(f"| {g['row']} | {c['label']} | {c['miss']:.4f} | {c['se_binom']:.4f} | {c['se_cluster']:.4f} | "
                         f"{c['deff']:.1f} | {c['cls_binom']} | {c['cls_cluster']} |")
            ds = [c["deff"] for c in g["cells"] if c["deff"] is not None and c["miss"] > 0]
            summary[gid] = dict(row=g["row"], counts=cnt(g["cells"], "cls_binom"), counts_cluster=cnt(g["cells"], "cls_cluster"),
                                deff_median=float(np.median(ds)) if ds else None, deff_max=float(max(ds)) if ds else None)
            L.append(f"| {g['row']} | summary | | | | median {summary[gid]['deff_median']:.1f}, max {summary[gid]['deff_max']:.1f} | "
                     f"{' / '.join(map(str, summary[gid]['counts']))} | {' / '.join(map(str, summary[gid]['counts_cluster']))} |")
    json.dump(dict(summary=summary, groups=G, sheet=sheet), open(os.path.join(OUT, "m1_abstention_mc.json"), "w"), indent=1)
    open(os.path.join(OUT, "m1_abstention_mc.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


STAGES = dict(stored=stage_stored, recheck=stage_recheck, parametric=stage_parametric, stratppi_a=stage_stratppi_a,
              stratppi_b=stage_stratppi_b, replacement=stage_replacement, ppipp017=stage_ppipp017, sheet=stage_sheet,
              report=stage_report)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(STAGES))
    ap.add_argument("--workers", type=int, default=6)
    STAGES[ap.parse_args().stage](ap.parse_args())
