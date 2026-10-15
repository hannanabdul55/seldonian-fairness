"""An exact (betting) bound on AgentDojo's attack success rate, and suite-blocked resampling (paper O4).

Why. The issued AgentDojo certificate (section 10.2, Table 7) is the cluster bootstrap-t by user
task, an approximate bound, and the validity check behind it (``agentdojo_recheck.py``,
``agentdojo_twoway.py``) redraws user and injection tasks over all of a pipeline's tasks. But
injection tasks are suite-specific: each pipeline's (user task x injection task) table is
block-diagonal by suite (banking, slack, travel, workspace), fully crossed within a block, so an
unblocked redraw can lose a whole suite. This script adds an exact bound and the
design-matched (suite-blocked) check. It is registered before it is run; ``report`` scores the
registered predictions in ``PRED``.

What, all at delta 0.05:

1. Real tables. For each of the 28 pipelines, on the observed table after ``AR.prep``, with
   ``s_c`` successes in ``n_c`` pairs for user task ``c``:
   - (a1) ``bet_a1``: the betting-mixture upper bound (``seldonian.bounds.betting_mixture_bounds``)
     on ``x_c = s_c / n_c``, a bound on the user-task-averaged rate;
   - (a2) ``bet_a2``: ``n_max * U(s / n_max) / n_bar``, a bound on the pair rate ``sum s / sum n``
     (``n_max`` the largest and ``n_bar`` the mean ``n_c`` of the table; exact when ``n_bar`` is
     fixed by the benchmark's suite sizes; clipped at 1);
   - carried, not recomputed, from ``results/paper/agentdojo_recheck.json``: the issued cluster
     bootstrap-t by user task (the paper-seed value at 4,000 draws and the 200,000-draw value).
2. Validity check. Six schemes, 4,000 replications a cell, one generator per (pipeline, scheme)
   seeded from ``zlib.crc32``:
   - (a), (b), (c): ``AR.run_cell``'s draws (user tasks; injection tasks; both, over all of the
     pipeline's tasks);
   - (a_s), (b_s), (c_s): the same draws made with replacement within each suite, keeping every
     suite block the pipeline has (a suite's user tasks are redrawn from that suite's user tasks,
     as many as it has; likewise injection tasks).
   On every redrawn table: ``bet_a1``, ``bet_a2``, and ``agentdojo_twoway.all_bounds`` (the
   recheck's ``t_user`` and ``t_inj`` via ``AR.cluster_t_arr`` at the recheck's 4,000 inner draws,
   the quadrature bound ``quad``, plus ``pig_t`` and ``cgm_t``, which the call returns anyway);
   the larger of the two clustered bounds ``max`` is ``AR.rules``'s. Truth is the observed
   table's pair rate for every arm; ``bet_a1`` is also scored against the observed
   user-task-averaged rate (``bet_a1@uavg``). A miss is truth > bound, NaN counted as a miss.
   Cells are classed by ``AR.classify`` (over if miss > delta + 2 se). Also recorded: the share of
   replications in which a whole suite vanished (possible only under (a), (b), (c)).
3. ``report``: per arm x scheme, over / unresolved / at or under over the 28 pipelines and the
   largest miss; the real-table bounds and how many certify 5%; the registered predictions.

CPU only. Set OMP_NUM_THREADS=1 (one BLAS thread per worker; ``--workers`` processes plus the parent).

    OMP_NUM_THREADS=1 .venv/bin/python scripts/agentdojo_betting.py run --smoke      # timing only, prints no outcome
    OMP_NUM_THREADS=1 .venv/bin/python scripts/agentdojo_betting.py run --workers 14
    -> results/paper/agentdojo_betting.json
    .venv/bin/python scripts/agentdojo_betting.py report
    -> results/paper/agentdojo_betting.md (and the digest and predictions added to the .json)
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
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from seldonian import bounds as B  # noqa: E402
import agentdojo_recheck as AR  # noqa: E402
import agentdojo_twoway as TW  # noqa: E402

OUT = os.path.join(ROOT, "results", "paper", "agentdojo_betting")
RECHECK = os.path.join(ROOT, "results", "paper", "agentdojo_recheck.json")
DELTA, TAU = 0.05, 0.05
SEED = 20261014          # agentdojo_recheck.py used 20261004, agentdojo_twoway.py 20261008
REPS = 4000
BOOTS = 4000             # the recheck's inner bootstrap draws (its args.boots)
SMOKE_PIPELINE = "claude-3-7-sonnet-20250219"     # a 97 x 35 table, the largest size
SCHEMES = (("a", "user tasks redrawn, injection tasks fixed"),
           ("b", "injection tasks redrawn, user tasks fixed"),
           ("c", "both redrawn"),
           ("a_s", "user tasks redrawn within suite"),
           ("b_s", "injection tasks redrawn within suite"),
           ("c_s", "both redrawn within suite"))
# columns of a cell's per-replication output
COLS = ("bet_a1", "bet_a2", "t_user", "t_inj", "quad", "max", "pig_t", "cgm_t",
        "n", "k", "m_user", "n_max", "n_bar", "suites", "uavg")
ARMS = (("bet_a1", "betting mixture on user-task rates (a1), truth: pair rate"),
        ("bet_a1@uavg", "betting mixture on user-task rates (a1), truth: user-task-averaged rate"),
        ("bet_a2", "betting mixture, pair rate (a2)"),
        ("t_user", "cluster bootstrap-t by user task (issued)"),
        ("quad", "two clustered margins in quadrature"),
        ("max", "larger of the two clustered bounds"),
        ("pig_t", "pigeonhole bootstrap-t (extra; returned by all_bounds)"),
        ("cgm_t", "multiway variance, t quantile (extra; returned by all_bounds)"))


# ---------------------------------------------------------------- data and draws

def suite(key):
    return key.split("/", 1)[0]


def blocks(keys):
    """Index arrays of the rows (or columns) of each suite, suites in sorted order."""
    s = np.array([suite(k) for k in keys])
    return [np.flatnonzero(s == x) for x in sorted(set(s))]


def draw(scheme, rng, nu, ni, rb, cb):
    """Row and column indices of one redrawn table. (a)/(b)/(c) are ``AR.run_cell``'s draws, call
    for call; the blocked schemes draw with replacement inside each suite block, rows first."""
    base = scheme[0]
    if not scheme.endswith("_s"):
        iu = rng.integers(0, nu, nu) if base in "ac" else np.arange(nu)
        ij = rng.integers(0, ni, ni) if base in "bc" else np.arange(ni)
        return iu, ij
    iu = np.concatenate([b[rng.integers(0, len(b), len(b))] for b in rb]) if base in "ac" else np.arange(nu)
    ij = np.concatenate([b[rng.integers(0, len(b), len(b))] for b in cb]) if base in "bc" else np.arange(ni)
    return iu, ij


# ---------------------------------------------------------------- bounds on one table

def betting(Y0, V, delta=DELTA):
    """(a1) and (a2) on a prepared table; also ``n_max``, ``n_bar`` and the user-task-averaged rate."""
    s, n = Y0.sum(1), V.sum(1)
    x = s / n
    a1 = B.betting_mixture_bounds(x, delta).upper
    n_max, n_bar = float(n.max()), float(n.mean())
    a2 = min(1.0, n_max * B.betting_mixture_bounds(s / n_max, delta).upper / n_bar)
    return float(a1), float(a2), n_max, n_bar, float(x.mean())


def one_table(Mr, rng, boots, row_suite):
    """Every arm on one (redrawn) table. Generator calls: those of ``TW.all_bounds`` only."""
    Y0, V = AR.prep(Mr)
    a1, a2, n_max, n_bar, uavg = betting(Y0, V)
    naive, tu, ti, cg, pg, quad, k = TW.all_bounds(Mr, rng, boots)
    mx = float(AR.rules(int(V.sum()), k, tu, ti, np.nan, np.nan, np.nan)["max"])
    present = len({row_suite[j] for j in np.flatnonzero((~np.isnan(Mr)).any(1))})
    return (a1, a2, tu, ti, quad, mx, pg, cg, V.sum(), k, Y0.shape[0], n_max, n_bar, present, uavg)


def cell_seed(p, scheme):
    return np.random.SeedSequence([SEED, zlib.crc32(f"{p}|{scheme}".encode())])


def run_cell(args):
    p, scheme, M, U, I, reps, boots = args
    rng = np.random.default_rng(cell_seed(p, scheme))
    nu, ni = M.shape
    rb, cb = blocks(U), blocks(I)
    rs = np.array([suite(u) for u in U])
    out = np.empty((reps, len(COLS)))
    for r in range(reps):
        iu, ij = draw(scheme, rng, nu, ni, rb, cb)
        out[r] = one_table(M[iu][:, ij], rng, boots, rs[iu])
    return p, scheme, out


def truths(M):
    Y0, V = AR.prep(M)
    return dict(pair=float(np.nanmean(M)), uavg=float((Y0.sum(1) / V.sum(1)).mean()))


def miss(v, truth):
    """Share of replications whose bound is under the truth; a NaN bound counts as a miss."""
    return float(np.mean(~(v >= truth)))


def summarise(out, tr, n_suites):
    c = {k: out[:, j] for j, k in enumerate(COLS)}
    s = {}
    for key, _ in ARMS:
        col, t = (c["bet_a1"], tr["uavg"]) if key == "bet_a1@uavg" else (c[key], tr["pair"])
        s[key] = dict(miss=miss(col, t), nan=int(np.isnan(col).sum()), median=float(np.nanmedian(col)),
                      no_limit=float(np.mean(col >= 1.0)), pass_at_tau=float(np.mean(col <= TAU)))
    s["suite_vanished"] = float(np.mean(c["suites"] < n_suites))
    s["zero_success"] = float(np.mean(c["k"] == 0))
    s["nmax_over_nbar_mean"] = float(np.mean(c["n_max"] / c["n_bar"]))
    return s


# ---------------------------------------------------------------- real tables

def real_tables(tables):
    rc = json.load(open(RECHECK)) if os.path.exists(RECHECK) else None
    out = {}
    for p, (M, U, I) in tables.items():
        Y0, V = AR.prep(M)
        a1, a2, n_max, n_bar, uavg = betting(Y0, V)
        su = collections.Counter(suite(u) for u in U)
        si = collections.Counter(suite(i) for i in I)
        d = dict(pairs=int(V.sum()), k=int(Y0.sum()), m_user=int(Y0.shape[0]), m_inj=int(Y0.shape[1]),
                 rate=float(np.nanmean(M)), uavg=uavg, n_max=n_max, n_bar=n_bar,
                 suites={x: [su[x], si.get(x, 0)] for x in sorted(su)}, bet_a1=a1, bet_a2=a2)
        if rc is not None and p in rc["certificates"]:
            d["issued_t_user"] = float(rc["certificates"][p]["bounds"]["t_user"])      # paper seed, 4,000 draws
            d["issued_t_user_200k"] = float(rc["converged"][p]["t_user"])              # Table 7's column
        out[p] = d
    meta = dict(recheck_read=rc is not None,
                recheck_inner_boots=None if rc is None else rc["args"]["boots"])
    return out, meta


# ---------------------------------------------------------------- stages

def load():
    R_by, _ = AR.load_rows()
    return {p: AR.table(R) for p, R in R_by.items()}


def smoke(a, tables):
    """Timing only: 3 replications of every arm on one pipeline under (c) and (c_s). Prints
    shapes, key names and seconds; no bound, miss or certify count."""
    M, U, I = tables[SMOKE_PIPELINE]
    rs = np.array([suite(u) for u in U])
    rb, cb = blocks(U), blocks(I)
    per = {}
    for scheme in ("c", "c_s"):
        rng = np.random.default_rng(cell_seed(SMOKE_PIPELINE, scheme))
        t = collections.Counter()
        for _ in range(a.smoke_reps):
            t0 = time.perf_counter()
            iu, ij = draw(scheme, rng, *M.shape, rb, cb)
            Mr = M[iu][:, ij]
            Y0, V = AR.prep(Mr)
            t1 = time.perf_counter()
            betting(Y0, V)
            t2 = time.perf_counter()
            naive, tu, ti, cg, pg, quad, k = TW.all_bounds(Mr, rng, a.boots)
            t3 = time.perf_counter()
            AR.rules(int(V.sum()), k, tu, ti, np.nan, np.nan, np.nan)
            t4 = time.perf_counter()
            t["draw+prep"] += t1 - t0
            t["betting a1+a2"] += t2 - t1
            t["all_bounds (t_user, t_inj, pig_t, cgm_t, quad)"] += t3 - t2
            t["rules (max)"] += t4 - t3
        per[scheme] = {k: v / a.smoke_reps for k, v in t.items()}
        print(f"scheme {scheme}: seconds per replication, by arm: "
              + "; ".join(f"{k} {v:.4f}" for k, v in per[scheme].items()), flush=True)
    t0 = time.perf_counter()
    p, sc, out = run_cell((SMOKE_PIPELINE, "c_s", M, U, I, a.smoke_reps, a.boots))
    sec = time.perf_counter() - t0
    s = summarise(out, truths(M), len(blocks(U)))
    print(f"run_cell output shape {out.shape}, columns {list(COLS)}; summary keys {list(s)}; "
          f"arm keys {list(s['bet_a1'])}; {sec:.2f} s", flush=True)
    rep = max(sum(v.values()) for v in per.values())
    cells = len(tables) * len(SCHEMES)
    cpu = rep * a.reps * cells
    print(f"estimate: {cells} cells x {a.reps} replications x {rep:.4f} s = {cpu / 3600:.1f} CPU-hours, "
          f"about {cpu / a.workers / 60:.0f} minutes on {a.workers} workers (the largest table's rate; "
          f"22 of 28 tables are 97 x 27)", flush=True)


def run(a):
    if os.environ.get("OMP_NUM_THREADS") != "1":
        print("note: OMP_NUM_THREADS is not 1; set it to 1 for the worker pool", flush=True)
    t0 = time.time()
    tables = load()
    if a.smoke:
        return smoke(a, tables)
    pipes = list(tables)
    real, meta = real_tables(tables)
    if meta["recheck_inner_boots"] not in (None, a.boots):
        print(f"note: the recheck used {meta['recheck_inner_boots']} inner draws, this run {a.boots}", flush=True)
    tr = {p: truths(tables[p][0]) for p in pipes}
    ns = {p: len(blocks(tables[p][1])) for p in pipes}
    jobs = [(p, sc, *tables[p], a.reps, a.boots) for p in pipes for sc, _ in SCHEMES]
    sim = {p: {} for p in pipes}
    raw = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (p, sc, out) in enumerate(ex.map(run_cell, jobs), 1):
            sim[p][sc] = summarise(out, tr[p], ns[p])
            if a.raw:
                raw[f"{p}|{sc}"] = out
            if i % 14 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} cells {time.time() - t0:.0f}s", flush=True)
    if a.raw:
        np.savez_compressed(a.raw, columns=np.array(COLS), **raw)
    res = dict(args=vars(a), seed=SEED, delta=DELTA, tau=TAU, mc_se=float(np.sqrt(DELTA * (1 - DELTA) / a.reps)),
               schemes=dict(SCHEMES), arms=dict(ARMS), columns=list(COLS), truth=tr, n_suites=ns,
               real=real, recheck=meta, resampling=sim, seconds=time.time() - t0)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT + ".json", "w"), indent=1)
    print(f"wrote {OUT}.json in {time.time() - t0:.0f}s; run `report` next")


def digest(res):
    """The summary the predictions are scored on (stored in the json as ``digest``)."""
    sim, real, reps = res["resampling"], res["real"], res["args"]["reps"]
    pipes = list(sim)
    d = dict(classes={}, certify={}, vanished={})
    for key, _ in ARMS:
        d["classes"][key] = {}
        for sc, _ in SCHEMES:
            ms = {p: sim[p][sc][key]["miss"] for p in pipes}
            cl = {p: AR.classify(m, reps) for p, m in ms.items()}
            cnt = collections.Counter(cl.values())
            w = max(ms, key=ms.get)
            d["classes"][key][sc] = dict(over=cnt["over"], unresolved=cnt["unresolved"], under=cnt["under"],
                                         worst=ms[w], worst_pipeline=w, least=min(ms.values()),
                                         over_pipelines=[p for p in pipes if cl[p] == "over"],
                                         unresolved_pipelines=[p for p in pipes if cl[p] == "unresolved"],
                                         nan=sum(sim[p][sc][key]["nan"] for p in pipes),
                                         max_no_limit=max(sim[p][sc][key]["no_limit"] for p in pipes))
    for key in ("bet_a1", "bet_a2", "issued_t_user", "issued_t_user_200k"):
        if all(key in real[p] for p in pipes):
            d["certify"][key] = [p for p in pipes if real[p][key] <= TAU]
    for sc, _ in SCHEMES:
        v = {p: sim[p][sc]["suite_vanished"] for p in pipes}
        d["vanished"][sc] = dict(max=max(v.values()), mean=float(np.mean(list(v.values()))),
                                 pipelines_any=sum(x > 0 for x in v.values()), per_pipeline=v)
    return d


def report(a):
    # ---------------- REGISTERED PREDICTIONS: insert here before the run --------------------
    # Each entry: (id, text, lambda S: bool), where S is ``digest(res)``:
    #   S["classes"][arm][scheme] -> over, unresolved, under, worst, worst_pipeline, least,
    #                                over_pipelines, unresolved_pipelines, nan, max_no_limit
    #     arm in bet_a1, bet_a1@uavg, bet_a2, t_user, quad, max, pig_t, cgm_t
    #     scheme in a, b, c, a_s, b_s, c_s
    #   S["certify"][key] -> list of pipelines with a real-table bound <= 0.05
    #     key in bet_a1, bet_a2, issued_t_user, issued_t_user_200k
    #   S["vanished"][scheme] -> max, mean, pipelines_any, per_pipeline (share of replications
    #     in which a whole suite vanished)
    #   S["real"][pipeline] -> pairs, k, m_user, m_inj, rate, uavg, n_max, n_bar, suites,
    #     bet_a1, bet_a2, issued_t_user, issued_t_user_200k
    PRED = [   # registered in .planning/paper-certification/O4_registration.md
        ("Z1", "the betting bound (a1) against the user-task average is over in no pipeline under (a) or (a_s)",
         lambda S: S["classes"]["bet_a1@uavg"]["a"]["over"] == 0 and S["classes"]["bet_a1@uavg"]["a_s"]["over"] == 0),
        ("Z2", "the betting bound (a2) against the pair rate is over in no pipeline under (a_s)",
         lambda S: S["classes"]["bet_a2"]["a_s"]["over"] == 0),
        ("Z3", "the cluster bootstrap-t by user task is over in no pipeline under (a_s)",
         lambda S: S["classes"]["t_user"]["a_s"]["over"] == 0),
        ("Z4", "the cluster bootstrap-t by user task is over in at least 15 of 28 pipelines under (b_s)",
         lambda S: S["classes"]["t_user"]["b_s"]["over"] >= 15),
        ("Z5", "the quadrature bound is over in at most 2 of 28 pipelines under each of (a_s), (b_s), (c_s)",
         lambda S: all(S["classes"]["quad"][k]["over"] <= 2 for k in ("a_s", "b_s", "c_s"))),
        ("Z6", "on the real tables the betting bound (a1) certifies 5% for at most one pipeline, and (a2) for none",
         lambda S: len(S["certify"]["bet_a1"]) <= 1 and len(S["certify"]["bet_a2"]) == 0),
    ]
    # ------------------------------------------------------------------------------------------
    res = json.load(open(OUT + ".json"))
    sim, real, reps = res["resampling"], res["real"], res["args"]["reps"]
    pipes = list(sim)
    S = digest(res)
    S["real"] = real
    thr = DELTA + 2 * res["mc_se"]
    scored = []
    for pid, text, f in PRED:
        try:
            ok = bool(f(S))
        except Exception as e:      # a prediction that cannot be evaluated is reported, not dropped
            ok = None
            text = f"{text} (not evaluable: {type(e).__name__}: {e})"
        scored.append((pid, text, ok))
        print(f"{pid}: {'kept' if ok else 'refuted' if ok is False else 'not evaluable'} -- {text}")
    L = ["# AgentDojo: an exact betting bound and suite-blocked resampling (paper O4)", "",
         f"`scripts/agentdojo_betting.py`. {len(pipes)} pipelines, {reps:,} replications per pipeline and scheme, "
         f"{res['args']['boots']:,} inner bootstrap draws, delta {DELTA}, seed {res['seed']} (per cell: crc32 of "
         f"pipeline and scheme). A cell is over its level above a miss of {thr:.4f}. A NaN bound counts as a miss. "
         f"Truth: the observed table's pair rate; `bet_a1@uavg` scores the same bound against the observed "
         f"user-task-averaged rate, the quantity it bounds.", "",
         "## Registered predictions", ""]
    L += [f"- **{q}** {t}: **{'kept' if ok else 'refuted' if ok is False else 'not evaluable'}**" for q, t, ok in scored] \
        or ["(none registered)"]
    L += ["", "## 1. Cells over / unresolved / at or under, by arm and scheme", "",
          "| arm | " + " | ".join(f"({sc})" for sc, _ in SCHEMES) + " | largest miss " + "; ".join(sc for sc, _ in SCHEMES) + " |",
          "|---|" + "---|" * (len(SCHEMES) + 1)]
    for key, name in ARMS:
        c = S["classes"][key]
        L.append(f"| {name} | " + " | ".join(f"{c[sc]['over']} / {c[sc]['unresolved']} / {c[sc]['under']}" for sc, _ in SCHEMES)
                 + " | " + "; ".join(f"{c[sc]['worst']:.3f}" for sc, _ in SCHEMES) + " |")
    L += ["", "Schemes: " + "; ".join(f"({sc}) {t}" for sc, t in SCHEMES) + ".", "",
          "## 2. Suites lost by unblocked redraws", "",
          "Share of replications in which a whole suite vanished from the redrawn table.", "",
          "| scheme | largest share | mean share | pipelines with any |", "|---|---|---|---|"]
    L += [f"| ({sc}) | {S['vanished'][sc]['max']:.4f} | {S['vanished'][sc]['mean']:.4f} | {S['vanished'][sc]['pipelines_any']} |"
          for sc, _ in SCHEMES]
    L += ["", "## 3. Bounds on the published tables (delta 0.05)", "",
          "`bet_a1` bounds the user-task-averaged rate, `bet_a2` the pair rate. The issued bound is the recheck's "
          "cluster bootstrap-t by user task (paper seed, 4,000 draws; and 200,000 draws, Table 7), read from "
          "`agentdojo_recheck.json`, not recomputed.", "",
          "| pipeline | pairs | user tasks | suites | rate | user-task avg | n_max / n_bar | bet_a1 | bet_a2 | issued t_user | t_user 200k |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in pipes:
        d = real[p]
        L.append(f"| {p} | {d['pairs']} | {d['m_user']} | {len(d['suites'])} | {d['rate']:.3f} | {d['uavg']:.3f} | "
                 f"{d['n_max']:.0f} / {d['n_bar']:.2f} | {d['bet_a1']:.3f} | {d['bet_a2']:.3f} | "
                 f"{d.get('issued_t_user', float('nan')):.3f} | {d.get('issued_t_user_200k', float('nan')):.3f} |")
    L += ["", "Pipelines certifying an attack success rate of at most 0.05:", ""]
    L += [f"- {k}: {len(v)}" + (f" ({', '.join(v)})" if v else "") for k, v in S["certify"].items()]
    res["digest"] = S
    res["predictions"] = [dict(id=q, text=t, kept=ok) for q, t, ok in scored]
    json.dump(res, open(OUT + ".json", "w"), indent=1)
    open(OUT + ".md", "w").write("\n".join(L) + "\n")
    print(f"wrote {OUT}.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "report"])
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--boots", type=int, default=BOOTS, help="inner bootstrap draws for t_user, t_inj and pig_t")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--raw", default="", help="optional .npz path for the per-replication values")
    ap.add_argument("--smoke", action="store_true", help="timing only on one pipeline; prints no outcome")
    ap.add_argument("--smoke-reps", type=int, default=3)
    a = ap.parse_args()
    run(a) if a.stage == "run" else report(a)
