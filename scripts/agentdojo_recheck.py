"""Recheck of the AgentDojo certificate's validity (paper section 8.2, Table 6, Table 2 rows [020 P]).

Spike 020's check resampled user tasks only, for 6 of 28 pipelines, 400 times, and never
resampled the rule Table 6 uses. This redoes it for all 28 pipelines at delta 0.05. Each
pipeline's observed (user task x injection task) table is the population and its full-table
rate the truth. Three resampling schemes: (a) user tasks drawn with replacement, injection
tasks fixed; (b) injection tasks drawn, user tasks fixed; (c) both drawn. Draws are over all of
a pipeline's user (injection) tasks, not within suite, as in ``plasmode020.py``; a re-drawn task
is a new cluster. Six bounds on every resample: per-pair Clopper-Pearson; cluster bootstrap-t by
user task; by injection task; the larger of the two (Table 6's rule); the dominant-cluster rule
(``cert020.py``: the unit with the larger one-way ICC, its ``icc_by`` replayed call for call);
the spike's two-way basic bootstrap. A seventh column, ``quad``, is this recheck's own and is in
neither the spike nor the paper: the two one-way margins added in quadrature.

The bound arithmetic is ``cert020.py``'s. ``cluster_t`` and ``icc_by`` are re-expressed on
arrays and checked bit for bit against the originals with a shared generator; ``twoway_t``'s
loop over inner resamples is replaced by count matrices (the same statistic, exact integer
sums) and checked against the loop. The real-data certificates call the original functions in
``cert020.main``'s order with its seed and are compared with ``cert.json``.

Also: the certificate for all 28 pipelines under each rule, with its spread over bootstrap
seeds; the two Meta-SecAlign-70B pipelines (where the successes sit, each bound over seeds and
two inner sizes); and the exact any-injection bound per user task, which bounds a stricter
quantity. CPU only; ``--workers`` processes plus the parent.

    OMP_NUM_THREADS=1 .venv/bin/python scripts/agentdojo_recheck.py --reps 4000 --boots 4000 --workers 4
    -> results/paper/agentdojo_recheck.md, results/paper/agentdojo_recheck.json   (13 minutes)
    .venv/bin/python scripts/agentdojo_recheck.py --render     # the .md again from the .json
"""
import argparse
import collections
import json
import lzma
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.dont_write_bytecode = True      # importing the spike's modules must not write under .planning/

import numpy as np
from scipy.stats import beta, binom, norm, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SPIKE = os.path.join(ROOT, ".planning", "spikes", "020-agentdojo-injection-certificate")
sys.path.insert(0, SPIKE)
import cert020 as C  # noqa: E402

EPISODES = os.path.join(ROOT, "results", "spikes", "020", "episodes.jsonl.xz")
OUT_DIR = os.path.join(ROOT, "results", "paper")
DELTA = 0.05
TAU = 0.05
SCHEMES = (("a", "user tasks resampled, injection tasks fixed"),
           ("b", "injection tasks resampled, user tasks fixed"),
           ("c", "both resampled"))
BOUNDS = (("naive", "per-pair CP"), ("t_user", "t(user)"), ("t_inj", "t(inj)"), ("max", "larger of two"),
          ("dom", "dominant cluster"), ("twoway", "two-way"), ("quad", "quadrature (not in the paper)"))
SECALIGN = ("Meta-SecAlign-70B", "Meta-SecAlign-70B-repeat_user_prompt")
CHUNK = 20000        # inner draws per block; above every size the spike uses
BIG = 200000         # inner bootstrap for the "converged" real-data limits
CLIFF = 0.08         # SecAlign: a user-task limit above this is on the far side of the cliff (section 3)


# ---------------------------------------------------------------- data

def load_rows():
    rows = [json.loads(l) for l in lzma.open(EPISODES, "rt")]
    by = collections.defaultdict(list)
    for r in rows:
        if r["y"] is not None:
            by[r["pipeline"]].append(r)
    return dict(sorted(by.items())), len(rows)


def table(R):
    """(user task x injection task) matrix in first-appearance order; NaN where no pair was run."""
    U = list(dict.fromkeys(r["user_task_key"] for r in R))
    I = list(dict.fromkeys(r["injection_task_key"] for r in R))
    ui = {u: i for i, u in enumerate(U)}
    ii = {v: i for i, v in enumerate(I)}
    M = np.full((len(U), len(I)), np.nan)
    for r in R:
        M[ui[r["user_task_key"]], ii[r["injection_task_key"]]] = r["y"]
    return M, U, I


def rows_of(M):
    """The row list ``cert020`` would see for a table: row-major, one fresh key per row and column."""
    return [dict(y=int(M[j, l]), user_task_key=f"u#{j}", injection_task_key=f"i#{l}")
            for j in range(M.shape[0]) for l in range(M.shape[1]) if not np.isnan(M[j, l])]


# ---------------------------------------------------------------- the spike's bounds on arrays

def prep(M):
    """Drop empty rows and columns and put columns in first-appearance order of a row-major scan
    (the cluster order ``cert020``'s dict grouping produces)."""
    V = ~np.isnan(M)
    r = V.any(1)
    M, V = M[r], V[r]
    c = V.any(0)
    M, V = M[:, c], V[:, c]
    order = np.lexsort((np.arange(M.shape[1]), V.argmax(0)))
    M, V = M[:, order], V[:, order]
    return np.where(V, M, 0.0), V


def cluster_t_arr(sizes, sums, delta, rng, boots, detail=False):
    """``cert020.cluster_t`` from the cluster sizes and sums (same arithmetic, same generator calls;
    an inner bootstrap above CHUNK is drawn in pieces, which the spike's sizes never reach)."""
    sizes = np.asarray(sizes)
    sums = np.asarray(sums, dtype=float)
    n, m = sizes.sum(), len(sizes)
    est = sums.sum() / n
    resid = sums - est * sizes
    se = float(np.sqrt(m / (m - 1) * (resid ** 2).sum()) / n)
    ts = []
    for b0 in range(0, boots, CHUNK):
        idx = rng.integers(0, m, size=(min(CHUNK, boots - b0), m))
        sz, sm = sizes[idx], sums[idx]
        nb = sz.sum(1)
        eb = sm.sum(1) / nb
        rb = sm - eb[:, None] * sz
        seb = np.sqrt(m / (m - 1) * (rb ** 2).sum(1)) / nb
        with np.errstate(divide="ignore", invalid="ignore"):
            ts.append(np.where(seb > 0, (eb - est) / seb, np.where(eb < est, -np.inf, np.inf)))
    t = ts[0] if len(ts) == 1 else np.concatenate(ts)
    q = np.quantile(t, delta, method="lower")
    ub = est - q * se if np.isfinite(q) and se > 0 else 1.0
    ub = float(np.clip(ub, 0, 1))
    if detail == "t":
        return ub, float(est), se, float(q), t
    return (ub, float(est), se, float(q)) if detail else ub


def counts(idx, m):
    b = idx.shape[0]
    return np.bincount((idx + np.arange(b)[:, None] * m).ravel(), minlength=b * m).reshape(b, m).astype(np.float32)


def twoway_arr(Y0, V, delta, rng, boots, idx=None):
    """``cert020.twoway_t`` with the loop over inner resamples replaced by count matrices: the mean
    of ``M[a][:, c]`` over its valid cells is ``wu' Y wi / wu' V wi`` with ``wu``, ``wi`` the draw
    counts. Sums are small integers, exact in float32."""
    nu, ni = Y0.shape
    Yf, Vf = Y0.astype(np.float32), V.astype(np.float32)
    ests = []
    for b0 in range(0, boots, CHUNK):
        b = min(CHUNK, boots - b0)
        a, c = idx if idx is not None else (rng.integers(0, nu, (b, nu)), rng.integers(0, ni, (b, ni)))
        Wu, Wi = counts(a, nu), counts(c, ni)
        num = ((Wu @ Yf) * Wi).sum(1, dtype=np.float64)
        den = ((Wu @ Vf) * Wi).sum(1, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            ests.append(num / den)
    ests = ests[0] if len(ests) == 1 else np.concatenate(ests)
    est = Y0.sum() / V.sum()
    return float(np.clip(2 * est - np.quantile(ests, delta), 0, 1))


def icc_groups(groups):
    """``cert020.icc_by`` after its grouping: the same seed-0 within-cluster draws, in cluster order."""
    k = int(round(np.mean([len(v) for v in groups])))
    rng = np.random.default_rng(0)
    Y = np.array([rng.choice(v, size=k, replace=len(v) < k) if len(v) != k else v for v in groups], dtype=float)
    return float(C.icc_from_samples(Y))


def bounds_on(M, rng, boots, delta=DELTA):
    """Every resampled quantity for one table. Generator calls: t(user), t(inj), two-way."""
    Y0, V = prep(M)
    n, k = int(V.sum()), int(Y0.sum())
    tu = cluster_t_arr(V.sum(1), Y0.sum(1), delta, rng, boots)
    ti = cluster_t_arr(V.sum(0), Y0.sum(0), delta, rng, boots)
    tw = twoway_arr(Y0, V, delta, rng, boots)
    icc_u = icc_groups([Y0[j, V[j]] for j in range(Y0.shape[0])])
    icc_i = icc_groups([Y0[V[:, l], l] for l in range(Y0.shape[1])])
    return n, k, tu, ti, tw, icc_u, icc_i


def rules(n, k, tu, ti, tw, icc_u, icc_i, delta=DELTA):
    """The seven bounds from the resampled quantities (arrays or scalars)."""
    n, k, tu, ti, tw = (np.asarray(x, dtype=float) for x in (n, k, tu, ti, tw))
    est = k / n
    naive = np.where(k >= n, 1.0, beta.ppf(1 - delta, k + 1, np.maximum(n - k, 1e-12)))
    quad = np.clip(est + np.sqrt((tu - est) ** 2 + (ti - est) ** 2), 0, 1)
    return dict(naive=naive, t_user=tu, t_inj=ti, max=np.maximum(tu, ti),
                dom=np.where(np.asarray(icc_u) >= np.asarray(icc_i), tu, ti), twoway=tw, quad=quad)


# ---------------------------------------------------------------- checks of the re-expression

def selfcheck(tables, boots=500):
    """Array versions against ``cert020``'s functions on rows, same generator state; the two-way
    count-matrix form against the spike's loop on the same index draws."""
    worst = dict(t_user=0.0, t_inj=0.0, icc_user=0.0, icc_inj=0.0, twoway=0.0)
    old = C.BOOTS
    C.BOOTS = boots
    rng0 = np.random.default_rng(7)
    cases = 0
    for p in ("Meta-SecAlign-70B", "gpt-4o-2024-05-13", "meta-llama_Llama-3.3-70B-Instruct-repeat_user_prompt"):
        M = tables[p][0]
        nu, ni = M.shape
        for scheme in ("real", "a", "b", "c"):
            iu = rng0.integers(0, nu, nu) if scheme in "ac" else np.arange(nu)
            ij = rng0.integers(0, ni, ni) if scheme in "bc" else np.arange(ni)
            Mr = M[iu][:, ij]
            S = rows_of(Mr)
            Y0, V = prep(Mr)
            for key, ax, name in (("user_task_key", 1, "t_user"), ("injection_task_key", 0, "t_inj")):
                ref = C.cluster_t(S, key, DELTA, np.random.default_rng(11))[0]
                got = cluster_t_arr(V.sum(ax), Y0.sum(ax), DELTA, np.random.default_rng(11), boots)
                worst[name] = max(worst[name], abs(ref - got))
            ref_u = C.icc_by(S, "user_task_key")[0]
            ref_i = C.icc_by(S, "injection_task_key")[0]
            worst["icc_user"] = max(worst["icc_user"], abs(ref_u - icc_groups([Y0[j, V[j]] for j in range(Y0.shape[0])])))
            worst["icc_inj"] = max(worst["icc_inj"], abs(ref_i - icc_groups([Y0[V[:, l], l] for l in range(Y0.shape[1])])))
            a = rng0.integers(0, Y0.shape[0], (boots, Y0.shape[0]))
            c = rng0.integers(0, Y0.shape[1], (boots, Y0.shape[1]))
            Mn = np.where(V, Y0, np.nan)
            with np.errstate(invalid="ignore"):
                loop = np.array([np.nanmean(Mn[a[b]][:, c[b]]) for b in range(boots)])
            ref = float(np.clip(2 * np.nanmean(Mn) - np.quantile(loop, DELTA), 0, 1))
            worst["twoway"] = max(worst["twoway"], abs(ref - twoway_arr(Y0, V, DELTA, None, boots, idx=(a, c))))
            cases += 1
    C.BOOTS = old
    return dict(cases=cases, boots=boots, max_abs_diff=worst)


# ---------------------------------------------------------------- the resampling cells

def run_cell(args):
    p, pi, scheme, si, M, reps, boots, seed = args
    rng = np.random.default_rng(np.random.SeedSequence([seed, pi, si]))
    nu, ni = M.shape
    out = np.empty((reps, 7))
    for r in range(reps):
        iu = rng.integers(0, nu, nu) if scheme in "ac" else np.arange(nu)
        ij = rng.integers(0, ni, ni) if scheme in "bc" else np.arange(ni)
        out[r] = bounds_on(M[iu][:, ij], rng, boots)
    return p, scheme, out


def summarise(out, truth, reps):
    b = rules(*out.T)
    s = {}
    for key, _ in BOUNDS:
        v = b[key]
        s[key] = dict(miss=float(np.mean(v < truth)), median=float(np.nanmedian(v)),
                      pass_at_tau=float(np.mean(v <= TAU)), nan=int(np.isnan(v).sum()))
    s["dominant_is_user"] = float(np.mean(out[:, 5] >= out[:, 6]))
    s["zero_success_resamples"] = float(np.mean(out[:, 1] == 0))
    return s


def classify(miss, reps):
    se = np.sqrt(DELTA * (1 - DELTA) / reps)
    return "over" if miss > DELTA + 2 * se else ("unresolved" if miss > DELTA else "under")


# ---------------------------------------------------------------- real data

def certificates(R_by, cert_json):
    """``cert020.main``'s calls, in its order, with its seed and BOOTS: the paper's numbers."""
    C.BOOTS = 4000
    rng = np.random.default_rng(20)
    res, diff = {}, 0.0
    for p, R in R_by.items():
        n, k = len(R), sum(r["y"] for r in R)
        icc_u, icc_i = C.icc_by(R, "user_task_key")[0], C.icc_by(R, "injection_task_key")[0]
        tu = C.cluster_t(R, "user_task_key", 0.05, rng)[0]
        ti = C.cluster_t(R, "injection_task_key", 0.05, rng)[0]
        tw = C.twoway_t(R, 0.05, rng)
        per_u = collections.defaultdict(int)
        for r in R:
            per_u[r["user_task_key"]] = max(per_u[r["user_task_key"]], r["y"])
        k_any, n_any = sum(per_u.values()), len(per_u)
        C.cluster_t(R, "user_task_key", 0.1, rng)       # main() draws these two next (its delta 0.1 table);
        C.cluster_t(R, "injection_task_key", 0.1, rng)  # replayed so the generator stays in step
        b = {key: float(v) for key, v in rules(n, k, tu, ti, tw, icc_u, icc_i).items()}
        res[p] = dict(n=n, k=k, rate=k / n, icc_u=icc_u, icc_i=icc_i, dominant="user" if icc_u >= icc_i else "inj",
                      n_user=n_any, n_inj=len({r["injection_task_key"] for r in R}), bounds=b,
                      any_inj=dict(k=k_any, n=n_any, rate=k_any / n_any, bound=C.cp_upper(k_any, n_any, DELTA)))
        if p in cert_json:
            j = cert_json[p]
            diff = max(diff, abs(j["t_user"] - tu), abs(j["t_inj"] - ti), abs(j["twoway"] - tw), abs(j["naive"] - b["naive"]),
                       abs(j["icc_u"] - icc_u), abs(j["icc_i"] - icc_i), abs(j["any_inj"] - res[p]["any_inj"]["bound"]))
    return res, diff


def seed_spread(M, seeds, boots, delta=DELTA):
    """Each bound on the real table over bootstrap seeds (array versions)."""
    Y0, V = prep(M)
    n, k = int(V.sum()), int(Y0.sum())
    icc_u = icc_groups([Y0[j, V[j]] for j in range(Y0.shape[0])])
    icc_i = icc_groups([Y0[V[:, l], l] for l in range(Y0.shape[1])])
    vals = collections.defaultdict(list)
    for s in seeds:
        rng = np.random.default_rng([1000 + s, boots])
        tu = cluster_t_arr(V.sum(1), Y0.sum(1), delta, rng, boots)
        ti = cluster_t_arr(V.sum(0), Y0.sum(0), delta, rng, boots)
        tw = twoway_arr(Y0, V, delta, rng, boots)
        for key, v in rules(n, k, tu, ti, tw, icc_u, icc_i, delta).items():
            vals[key].append(float(v))
    return {key: dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                      min=float(np.min(v)), max=float(np.max(v)), values=v) for key, v in vals.items()}


def spread_cell(args):
    p, M, seeds, boots = args
    return p, boots, seed_spread(M, seeds, boots)


def secalign(tables):
    """Where the two pipelines' successes sit, and why the user-task limit moves with the seed."""
    out = {}
    for p in SECALIGN:
        M, U, I = tables[p]
        Y0, V = prep(M)
        su, si_ = Y0.sum(1).astype(int), Y0.sum(0).astype(int)
        szu, szi = V.sum(1), V.sum(0)
        d = dict(per_user={U[j]: [int(su[j]), int(szu[j])] for j in np.argsort(-su, kind="stable") if su[j] > 0},
                 per_inj={I[l]: [int(si_[l]), int(szi[l])] for l in np.argsort(-si_, kind="stable") if si_[l] > 0},
                 n_user=len(U), n_inj=len(I), k=int(su.sum()), n=int(szu.sum()))
        ub, est, se, q, t = cluster_t_arr(szu, su, DELTA, np.random.default_rng(5), BIG, detail="t")
        more = [cluster_t_arr(szu, su, DELTA, np.random.default_rng(s), BIG) for s in range(6, 10)]
        d.update(est=est, se_user=se, t_user_big=ub, t_user_big_seeds=[ub] + more, q_big=q,
                 wald_user=est + norm.ppf(1 - DELTA) * se)
        d["by_delta"] = {f"{dl:g}": float(np.clip(est - np.quantile(t, dl, method="lower") * se, 0, 1))
                         for dl in (0.03, 0.04, 0.045, 0.05, 0.055, 0.06, 0.08, 0.10)}
        # the cliff: the share of inner draws whose own value would put the limit above CLIFF
        p_hi = float(np.mean(est - t * se > CLIFF))
        d["share_of_draws_above_cliff"] = p_hi
        # which inner draws those are: by the number of the 97 drawn slots that hold a compromised task
        rng = np.random.default_rng(6)
        m = len(su)
        hits, hi = [], []
        for _ in range(5):
            idx = rng.integers(0, m, (CHUNK, m))
            sz, sm = szu[idx], su[idx].astype(float)
            nb = sz.sum(1)
            eb = sm.sum(1) / nb
            seb = np.sqrt(m / (m - 1) * ((sm - eb[:, None] * sz) ** 2).sum(1)) / nb
            with np.errstate(divide="ignore", invalid="ignore"):
                tt = np.where(seb > 0, (eb - est) / seb, np.where(eb < est, -np.inf, np.inf))
            hits.append((su[idx] > 0).sum(1))
            hi.append(est - tt * se > CLIFF)
        hits, hi = np.concatenate(hits), np.concatenate(hi)
        d["draws_by_hits"] = {str(c): dict(share=float(np.mean(hits == c)),
                                           above_cliff=float(np.mean(hi[hits == c])) if np.any(hits == c) else 0.0)
                              for c in range(0, 5)}
        # one episode changed: each compromised task loses one success
        d["drop_one_success"] = {U[j]: cluster_t_arr(szu, su - (np.arange(m) == j), DELTA, np.random.default_rng(5), BIG)
                                 for j in np.flatnonzero(su)}
        out[p] = d
    a, b = (out[p]["per_user"] for p in SECALIGN)
    out["shared_compromised_user_tasks"] = sorted(set(a) & set(b))
    return out


def converged_cell(args):
    p, M = args
    Y0, V = prep(M)
    rng = np.random.default_rng(99)
    n, k = int(V.sum()), int(Y0.sum())
    tu = cluster_t_arr(V.sum(1), Y0.sum(1), DELTA, rng, BIG)
    ti = cluster_t_arr(V.sum(0), Y0.sum(0), DELTA, rng, BIG)
    tw = twoway_arr(Y0, V, DELTA, rng, BIG)
    icc_u = icc_groups([Y0[j, V[j]] for j in range(Y0.shape[0])])
    icc_i = icc_groups([Y0[V[:, l], l] for l in range(Y0.shape[1])])
    return p, {key: float(v) for key, v in rules(n, k, tu, ti, tw, icc_u, icc_i).items()}


# ---------------------------------------------------------------- report

def mark(miss, reps):
    c = classify(miss, reps)
    return f"**{miss:.3f}**" if c == "over" else (f"{miss:.3f}?" if c == "unresolved" else f"{miss:.3f}")


def digest(res):
    """Class counts, worst cells and pass counts, used by the report and stored in the json."""
    sim, cert, reps = res["resampling"], res["certificates"], res["args"]["reps"]
    pipes = list(sim)
    d = dict(classes={}, passes={})
    for key, _ in BOUNDS:
        d["classes"][key] = {}
        for sc, _ in SCHEMES:
            ms = {p: sim[p][sc][key]["miss"] for p in pipes}
            cl = collections.Counter(classify(m, reps) for m in ms.values())
            w = max(ms, key=ms.get)
            d["classes"][key][sc] = dict(over=cl["over"], unresolved=cl["unresolved"], under=cl["under"], worst=ms[w],
                                         worst_pipeline=w, least=min(ms.values()),
                                         over_pipelines=[p for p in pipes if classify(ms[p], reps) == "over"],
                                         unresolved_pipelines=[p for p in pipes if classify(ms[p], reps) == "unresolved"])
        b4 = str(res["boots_real"])
        per_seed = [sum(res["seed_spread"][p][b4][key]["values"][s] <= TAU for p in pipes) for s in range(res["args"]["seeds"])]
        d["passes"][key] = dict(paper_seed=[p for p in pipes if cert[p]["bounds"][key] <= TAU],
                                per_seed_min=int(min(per_seed)), per_seed_max=int(max(per_seed)),
                                converged=[p for p in pipes if res["converged"][p][key] <= TAU])
    d["passes"]["any_inj"] = [p for p in pipes if cert[p]["any_inj"]["bound"] <= TAU]
    # how close the two one-way margins are on the real table, against the larger-of-two rule's miss under (c)
    ratio = {p: min(cert[p]["bounds"][k] - cert[p]["rate"] for k in ("t_user", "t_inj"))
             / max(cert[p]["bounds"][k] - cert[p]["rate"] for k in ("t_user", "t_inj")) for p in pipes}
    miss_c = {p: sim[p]["c"]["max"]["miss"] for p in pipes}
    top = sorted(pipes, key=miss_c.get, reverse=True)[:5]
    d["margin_ratio"] = dict(ratio=ratio, spearman=float(spearmanr([ratio[p] for p in pipes], [miss_c[p] for p in pipes])[0]),
                             worst5=top, worst5_ratio=[min(ratio[p] for p in top), max(ratio[p] for p in top)])
    return d


def report(res):
    sim, cert, a, dg = res["resampling"], res["certificates"], res["args"], res["digest"]
    reps, boots, se = a["reps"], a["boots"], res["mc_se"]
    pipes = list(sim)
    npip = len(pipes)
    b4 = str(res["boots_real"])
    cls, pas = dg["classes"], dg["passes"]

    def cc(key, sc):
        c = cls[key][sc]
        return f"{c['over']} / {c['unresolved']} / {c['under']}"

    def rng_(key, sc):
        return f"{cls[key][sc]['least']:.3f}-{cls[key][sc]['worst']:.3f}"

    L = ["# AgentDojo certificate: validity recheck", "",
         "Recheck of `reports/paper_certification.md` section 8.2, Table 6 and the `[020 P]` rows of Table 2. "
         "Written by `scripts/agentdojo_recheck.py`; numbers in `agentdojo_recheck.json`.", "",
         f"**What was run.** All {npip} pipelines with runs under `important_instructions` ({res['n_pairs']:,} labelled pairs; "
         f"{res['n_rows']:,} rows in `episodes.jsonl.xz`, one without a label). Level delta {DELTA}. {reps:,} resamples per "
         f"(pipeline, scheme) cell, {npip * 3 * reps:,} in all, each with an inner bootstrap of {boots:,} draws per bound "
         + ("(the size `cert020.py` uses for the paper's numbers). " if boots == res["boots_real"] else ". ") + "A pipeline's observed table is the population and its "
         "full-table rate the truth; a miss is a bound below the truth. Tasks are drawn with replacement over all of a "
         "pipeline's user (injection) tasks, not within suite, as `plasmode020.py` does; a re-drawn task is a new cluster.", "",
         f"**Class rule, the same for every bound.** se = sqrt(delta (1 - delta) / R) = {se:.4f}. *Over*: miss > "
         f"{DELTA + 2 * se:.4f} (bold). *Unresolved, above delta*: {DELTA} < miss <= {DELTA + 2 * se:.4f} (marked `?`). "
         "*At or under*: otherwise.", "",
         "**The bounds are the spike's.** `cluster_t` and `icc_by` re-expressed on arrays agree with `cert020.py`'s "
         f"functions to the last bit on {res['selfcheck']['cases']} tables (real and resampled under each scheme; largest "
         f"difference {max(res['selfcheck']['max_abs_diff'].values()):.1e}); the two-way bootstrap's count-matrix form agrees "
         "with the spike's loop on the same index draws to the same precision. The real-data certificates below call the "
         f"original functions with `cert020.main`'s seed and call order and differ from `cert.json` by at most "
         f"{res['cert_json_max_abs_diff']:.1e}.", "",
         "**One column is not the spike's or the paper's.** `quad` = rate + sqrt((t(user) - rate)^2 + (t(inj) - rate)^2), "
         "the two one-way margins added in quadrature. It was added to this script before the resampling was run, as the "
         "obvious candidate if the larger-of-two rule failed. It has had this check and no other.", ""]

    # ---- main findings
    mx, tu, dm, nv, tw, qd = (cls[k] for k in ("max", "t_user", "dom", "naive", "twoway", "quad"))
    one = pas["max"]["paper_seed"]
    sec = res["secalign"]
    L += ["## Main findings", ""]
    if mx["b"]["over"] + mx["c"]["over"] > 0:
        L.append(f"1. **Table 6's rule (the larger of the two clustered bounds) is over its level once injection tasks are "
                 f"treated as sampled.** Over in {mx['b']['over']} of {npip} pipelines when injection tasks are resampled "
                 f"(worst miss {mx['b']['worst']:.3f}, {mx['b']['worst_pipeline']}) and in {mx['c']['over']} of {npip}, with "
                 f"{mx['c']['unresolved']} more unresolved above delta, when both are resampled (worst {mx['c']['worst']:.3f}, "
                 f"{mx['c']['worst_pipeline']}). It is at or under in {mx['a']['under']} of {npip} when only user tasks are resampled "
                 f"(worst {mx['a']['worst']:.3f}). This contradicts section 8.2's first sentence as a general statement.")
    else:
        L.append(f"1. **Table 6's rule (the larger of the two clustered bounds) holds its level under all three schemes** "
                 f"(worst miss {max(mx[s]['worst'] for s in 'abc'):.3f}).")
    L.append(f"2. **The cluster bootstrap-t by user task is valid for the scheme it was checked under and no other.** User "
             f"tasks resampled: {tu['a']['under']} at or under, {tu['a']['unresolved']} unresolved, {tu['a']['over']} over "
             f"(miss {rng_('t_user', 'a')}). Injection tasks resampled: over in {tu['b']['over']} (worst {tu['b']['worst']:.3f}). "
             f"Both: over in {tu['c']['over']} (worst {tu['c']['worst']:.3f}). The dominant-cluster rule of `cert020.py` is "
             f"over in {dm['b']['over']} and {dm['c']['over']} (worst {dm['b']['worst']:.3f}, {dm['c']['worst']:.3f}).")
    L.append(f"3. **\"One of {npip}\" survives on the real data, under every clustered rule.** "
             + (f"{one[0]} is the only pipeline at or under 5% " if len(one) == 1 else f"{len(one)} pipelines pass ")
             + f"under the larger-of-two rule, the dominant-cluster rule, t(user) alone and `quad`, at the paper's seed, at "
             f"each of {a['seeds']} other bootstrap seeds and with {BIG:,} inner draws (pass count over seeds: larger-of-two "
             f"{span(pas['max'])}, `quad` {span(pas['quad'])}). The {npip - len(one)} refusals cannot be overturned by finding 1: a bound that is "
             "too small and still above 5% stays above 5% when made valid."
             + (f" For {one[0]} itself the larger-of-two rule missed in at most "
                f"{max(sim[one[0]][s]['max']['miss'] for s in 'abc'):.3f} of resamples under any scheme, and `quad` gives "
                f"{cert[one[0]]['bounds']['quad']:.3f}." if len(one) == 1 else ""))
    if sec:
        sp0, sp1 = res["seed_spread"][SECALIGN[0]][b4]["t_user"], res["seed_spread"][SECALIGN[1]][b4]["t_user"]
        L.append(f"4. **Meta-SecAlign-70B's 0.104 is bootstrap noise.** With {BIG:,} inner draws its user-task bound is "
                 f"{big_range(res, SECALIGN[0])} and the variant's {big_range(res, SECALIGN[1])} (6 seeds each). At 4,000 draws, the paper's size, the same "
                 f"table gives {sp0['min']:.3f}-{sp0['max']:.3f} over {a['seeds']} seeds (variant {sp1['min']:.3f}-{sp1['max']:.3f}). "
                 "The two pipelines' successes sit in almost the same places (section 3). Both stay above 5% at every seed, "
                 "so the refusal stands; the number and the sentence built on it do not.")
    L.append(f"5. **Per-pair Clopper-Pearson** is over in {nv['a']['over']}, {nv['b']['over']} and {nv['c']['over']} of {npip} "
             f"(miss {rng_('naive', 'a')}, {rng_('naive', 'b')}, {rng_('naive', 'c')}); **the two-way basic bootstrap** in "
             f"{tw['a']['over']}, {tw['b']['over']} and {tw['c']['over']} (worst {tw['a']['worst']:.3f}, {tw['b']['worst']:.3f}, "
             f"{tw['c']['worst']:.3f}).")
    L.append(f"6. **`quad`** is at or under in {qd['a']['under']}, {qd['b']['under']} and {qd['c']['under']} of {npip}, with "
             f"{qd['a']['unresolved'] + qd['b']['unresolved'] + qd['c']['unresolved']} unresolved and "
             f"{qd['a']['over'] + qd['b']['over'] + qd['c']['over']} over (worst {max(qd[s]['worst'] for s in 'abc'):.3f}, "
             f"{qd[max('abc', key=lambda s: qd[s]['worst'])]['worst_pipeline']}, scheme {max('abc', key=lambda s: qd[s]['worst'])}). It is the only one of the "
             "seven that is not over anywhere. That is one check, run after it was proposed for the purpose.")
    L.append("")

    # ---- section 1
    L += ["## 1. Miss rates by scheme and bound", "",
          f"### 1.1 Class counts over the {npip} pipelines", "",
          "Cells: over / unresolved above delta / at or under; then the range of miss rates and the pipeline with the worst.", "",
          "| bound | (a) user tasks resampled | (b) injection tasks resampled | (c) both resampled |", "|---|---|---|---|"]
    for key, nm in BOUNDS:
        L.append(f"| {nm} | " + " | ".join(f"{cc(key, sc)}; {rng_(key, sc)}; {cls[key][sc]['worst_pipeline']}" for sc, _ in SCHEMES) + " |")
    L.append("")
    for sc, desc in SCHEMES:
        L += [f"### 1.{'abc'.index(sc) + 2} Scheme ({sc}): {desc}", "",
              "| pipeline | truth | " + " | ".join(nm for _, nm in BOUNDS) + " |", "|---|---|" + "---|" * len(BOUNDS)]
        for p in pipes:
            L.append(f"| {p} | {cert[p]['rate']:.3f} | " + " | ".join(mark(sim[p][sc][k]["miss"], reps) for k, _ in BOUNDS) + " |")
        L.append("| **over / unresolved / at or under** | | " + " | ".join(cc(k, sc) for k, _ in BOUNDS) + " |")
        L.append("")
    over_b = cls["t_inj"]["b"]["over_pipelines"]
    L += ["Three things the tables show beyond the counts.", "",
          f"- t(inj) is over its level even for the scheme that matches it: {len(over_b)} pipelines over when only injection "
          f"tasks are resampled ({', '.join(f'{p} {sim[p]['b']['t_inj']['miss']:.3f}' for p in over_b)}). There are "
          f"{min(c['n_inj'] for c in cert.values())}-{max(c['n_inj'] for c in cert.values())} injection tasks; the "
          "studentised bootstrap is short of clusters there. The larger-of-two rule inherits this, which is why it fails under "
          "(b) and not only under (c).",
          "- Under (c) the two sources of variation add and the larger of two one-way margins covers only the larger one. The "
          "rule's miss rate rises with how close the two margins are on the real table (rank correlation "
          f"{dg['margin_ratio']['spearman']:.2f} between the smaller-to-larger margin ratio and the miss rate, {npip} pipelines): "
          f"the five worst cells have ratios {dg['margin_ratio']['worst5_ratio'][0]:.2f}-{dg['margin_ratio']['worst5_ratio'][1]:.2f}. "
          "Where one unit carries nearly all the dependence (both SecAlign pipelines) it is far under.",
          f"- The dominant-cluster rule picks its unit from the data, and on resamples the pick moves: for "
          f"{sum(0.1 < sim[p]['c']['dominant_is_user'] < 0.9 for p in pipes)} of {npip} pipelines it chose the user task in "
          "between 10% and 90% of scheme (c) resamples. Table 6's caption says the two rules give the same verdicts; on the "
          "real data they do, but the dominant rule is the less valid of the two.", ""]

    # ---- section 2
    L += ["## 2. Certificates on the real data", "",
          f"Upper bounds at delta {DELTA}, from `cert020.py`'s own functions, seed and call order (the paper's numbers). "
          "`dominant` names the unit with the larger ICC. The last column is the exact alternative of section 4 and bounds "
          "a different, stricter quantity.", "",
          "| pipeline | pairs | successes | rate | per-pair CP | t(user) | t(inj) | larger of two | dominant | two-way | quad | "
          "any-injection: user tasks compromised, CP upper bound |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in pipes:
        c, b = cert[p], cert[p]["bounds"]
        L.append(f"| {p} | {c['n']} | {c['k']} | {c['rate']:.3f} | {b['naive']:.3f} | {b['t_user']:.3f} | {b['t_inj']:.3f} | "
                 f"{b['max']:.3f} | {b['dom']:.3f} ({c['dominant']}) | {b['twoway']:.3f} | {b['quad']:.3f} | "
                 f"{c['any_inj']['k']} of {c['any_inj']['n']}, {c['any_inj']['bound']:.3f} |")
    L += ["", f"**Pipelines passing \"attack success at most {TAU:.0%}\" under each rule.**", "",
          f"| rule | passes at the paper's seed | which | pass count over {a['seeds']} seeds at {int(b4):,} inner draws | passes at {BIG:,} inner draws | "
          "worst resampled miss (a / b / c) |", "|---|---|---|---|---|---|"]
    for key, nm in BOUNDS:
        ps = pas[key]
        L.append(f"| {nm} | {len(ps['paper_seed'])} | {', '.join(ps['paper_seed']) or 'none'} | "
                 + span(ps) + f" | {len(ps['converged'])} | " + " / ".join(mark(cls[key][sc]["worst"], reps) for sc, _ in SCHEMES) + " |")
    L.append(f"| any-injection (stricter quantity) | {len(pas['any_inj'])} | {', '.join(pas['any_inj']) or 'none'} | exact | exact | not resampled |")
    L += ["", f"**Spread of the clustered bounds over bootstrap seeds** ({a['seeds']} seeds, {int(b4):,} inner draws; then {BIG:,} draws, one seed). "
          "Shown for the pipelines whose rate is under 0.07; for the other "
          f"{sum(cert[p]['rate'] >= 0.07 for p in pipes)} the larger-of-two bound moves by at most "
          f"{max(res['seed_spread'][p][b4]['max']['max'] - res['seed_spread'][p][b4]['max']['min'] for p in pipes if cert[p]['rate'] >= 0.07):.3f} "
          f"across seeds and its smallest value is {min(res['seed_spread'][p][b4]['max']['min'] for p in pipes if cert[p]['rate'] >= 0.07):.3f}.", "",
          f"| pipeline | rate | t(user): paper, min-max, {BIG // 1000}k | t(inj): paper, min-max, {BIG // 1000}k | larger of two: min-max, {BIG // 1000}k | "
          f"quad: min-max, {BIG // 1000}k |", "|---|---|---|---|---|---|"]
    for p in pipes:
        if cert[p]["rate"] < 0.07:
            s, cv, b = res["seed_spread"][p][b4], res["converged"][p], cert[p]["bounds"]
            L.append(f"| {p} | {cert[p]['rate']:.3f} | {b['t_user']:.3f}, {s['t_user']['min']:.3f}-{s['t_user']['max']:.3f}, {cv['t_user']:.3f} | "
                     f"{b['t_inj']:.3f}, {s['t_inj']['min']:.3f}-{s['t_inj']['max']:.3f}, {cv['t_inj']:.3f} | "
                     f"{s['max']['min']:.3f}-{s['max']['max']:.3f}, {cv['max']:.3f} | {s['quad']['min']:.3f}-{s['quad']['max']:.3f}, {cv['quad']:.3f} |")
    L.append("")

    # ---- section 3
    if sec:
        L += ["## 3. The two Meta-SecAlign-70B pipelines", "",
              "**Where the successes sit** (successes / pairs in the cluster; clusters with none omitted).", "",
              "| | " + " | ".join(SECALIGN) + " |", "|---|---|---|",
              "| successes / pairs | " + " | ".join(f"{sec[p]['k']} / {sec[p]['n']}" for p in SECALIGN) + " |",
              "| by user task | " + " | ".join("; ".join(f"{k} {v[0]}/{v[1]}" for k, v in sec[p]["per_user"].items())
                                                + f" ({len(sec[p]['per_user'])} of {sec[p]['n_user']} tasks)" for p in SECALIGN) + " |",
              "| by injection task | " + " | ".join("; ".join(f"{k} {v[0]}/{v[1]}" for k, v in sec[p]["per_inj"].items())
                                                     + f" ({len(sec[p]['per_inj'])} of {sec[p]['n_inj']} tasks)" for p in SECALIGN) + " |",
              "", f"The three large user tasks are the same in both pipelines ({', '.join(sec['shared_compromised_user_tasks'])}) with "
              "counts " + " against ".join(", ".join(str(v[0]) for v in list(sec[p]["per_user"].values())[:3]) for p in SECALIGN)
              + "; each has two further user tasks with one success. By injection task the successes are spread thin in both "
              f"(no injection task above {max(v[0] for p in SECALIGN for v in sec[p]['per_inj'].values())}). The tables differ by one success in one user task and by which two slack tasks "
              "carry a single success. Nothing in where the successes sit separates 0.104 from 0.057.", "",
              f"**Each bound over {a['seeds']} bootstrap seeds at {' and '.join(f'{int(z):,}' for z in sizes_of(res))} inner draws, and one seed at {BIG:,}.** Mean (min-max).", "",
              "| bound | " + " | ".join(f"{p}: {sz} draws" for p in SECALIGN for sz in sizes_of(res)) + " | "
              + " | ".join(f"{p}: {BIG // 1000}k" for p in SECALIGN) + " |", "|---|" + "---|" * (2 * len(sizes_of(res)) + 2)]
        for key, nm in BOUNDS:
            if key == "naive":
                continue
            cells = []
            for p in SECALIGN:
                for sz in sizes_of(res):
                    s = res["seed_spread"][p][sz][key]
                    cells.append(f"{s['mean']:.3f} ({s['min']:.3f}-{s['max']:.3f})")
            L.append(f"| {nm} | " + " | ".join(cells) + " | " + " | ".join(f"{res['converged'][p][key]:.3f}" for p in SECALIGN) + " |")
        L += ["", "The paper's values (seed 20, 4,000 draws, drawn one pipeline after the other from one generator) are "
              + " and ".join(f"{cert[p]['bounds']['t_user']:.3f}" for p in SECALIGN) + ". Only the user-task bound, and the rules "
              "built on it, move with the seed; t(inj) and the two-way bound do not.", "",
              "**Why the user-task bound moves.** The limit is rate - q se, with q the lower 5% point of the studentised "
              "statistic over inner draws. With 5 of 97 user tasks compromised, an inner draw that holds few of them, or only "
              "the single-success ones, has a small estimate and a far smaller standard error, so its statistic is very "
              f"negative. Read as a function of the level, the limit has a cliff just below 0.05 ({BIG:,} draws):", "",
              "| level | " + " | ".join(sec[SECALIGN[0]]["by_delta"]) + " |", "|---|" + "---|" * len(sec[SECALIGN[0]]["by_delta"])]
        for p in SECALIGN:
            L.append(f"| t(user), {p} | " + " | ".join(f"{v:.3f}" for v in sec[p]["by_delta"].values()) + " |")
        L += ["", f"Share of inner draws whose own value would put the limit above {CLIFF}: "
              + "; ".join(f"{p} {sec[p]['share_of_draws_above_cliff']:.4f}" for p in SECALIGN)
              + f". The limit reads off the draw at rank 5%, so a finite bootstrap lands above {CLIFF} whenever more than 5% of "
              "its own draws fall in that share:", "",
              f"| inner draws | " + " | ".join(f"{p}: predicted P(limit > {CLIFF})" for p in SECALIGN) + " | "
              + " | ".join(f"{p}: seeds above {CLIFF}" for p in SECALIGN) + " |", "|---|---|---|---|---|"]
        for sz in sizes_of(res):
            L.append(f"| {sz} | " + " | ".join(f"{p_above(sec[p]['share_of_draws_above_cliff'], int(sz)):.3f}" for p in SECALIGN) + " | "
                     + " | ".join(f"{sum(v > CLIFF for v in res['seed_spread'][p][sz]['t_user']['values'])} of {a['seeds']}" for p in SECALIGN) + " |")
        L.append(f"| {BIG} | " + " | ".join(f"{p_above(sec[p]['share_of_draws_above_cliff'], BIG):.3f}" for p in SECALIGN) + " | "
                 + " | ".join("0 of 1" if res["converged"][p]["t_user"] <= CLIFF else "1 of 1" for p in SECALIGN) + " |")
        h = sec[SECALIGN[0]]["draws_by_hits"]
        L += ["", f"Which draws those are, for {SECALIGN[0]}: by the number of the 97 drawn slots that hold a compromised task "
              f"(share of all draws, share of those above {CLIFF}): "
              + "; ".join(f"{c} ({h[c]['share']:.3f}, {h[c]['above_cliff']:.2f})" for c in h) + ".", "",
              "**Verdict on the pair.** 0.104 against 0.057 is the inner bootstrap landing on either side of that cliff, not a "
              f"difference between the pipelines. At {BIG:,} draws the bounds are "
              + " and ".join(big_range(res, p) for p in SECALIGN) + " (6 seeds each), in the order of the raw rates ("
              + " and ".join(f"{sec[p]['est']:.4f}" for p in SECALIGN) + "). Taking one success away from any one compromised "
              f"user task moves the {BIG // 1000}k-draw bound of {SECALIGN[0]} to "
              f"{min(sec[SECALIGN[0]]['drop_one_success'].values()):.3f}-{max(sec[SECALIGN[0]]['drop_one_success'].values()):.3f}: "
              f"one episode moves it by under 0.01, one seed at 4,000 draws by up to "
              f"{res['seed_spread'][SECALIGN[0]][b4]['t_user']['max'] - res['seed_spread'][SECALIGN[0]][b4]['t_user']['min']:.3f}. Both "
              f"pipelines are refused at 5% at every seed and size tried (smallest value "
              f"{min(min(res['seed_spread'][p][sz]['t_user']['values']) for p in SECALIGN for sz in sizes_of(res)):.3f}). "
              "The resampling check of section 1 used 4,000 inner draws, so it includes this noise; t(user) held its level for "
              f"both pipelines under (a) ({sim[SECALIGN[0]]['a']['t_user']['miss']:.3f}, {sim[SECALIGN[1]]['a']['t_user']['miss']:.3f}).", ""]

    # ---- section 4
    anyb = [cert[p]["any_inj"]["bound"] for p in pipes]
    best = min(pipes, key=lambda p: cert[p]["any_inj"]["bound"])
    L += ["## 4. An exact alternative, on a stricter quantity", "",
          "Call a user task failed if any of its injections succeeded; the units are then the user tasks, one binary label "
          "each, and Clopper-Pearson is exact if user tasks are an independent sample with the injection set fixed. **This "
          "bounds the share of user tasks compromised by at least one injection, not the attack success rate over pairs**; "
          "it is at least as large as the pair rate and usually far larger, and it says nothing about new injection tasks. "
          f"Values are in the last column of the table in section 2: {min(anyb):.3f} ({best}, "
          f"{cert[best]['any_inj']['k']} of {cert[best]['any_inj']['n']}) to {max(anyb):.3f}. "
          f"{len(pas['any_inj'])} of {npip} pipelines are at or under 5% on it; with 97 user tasks and none compromised the "
          f"bound would be {C.cp_upper(0, 97, DELTA):.3f}, so the claim is reachable only at zero or one compromised task "
          f"(one of 97: {C.cp_upper(1, 97, DELTA):.3f}).", ""]

    # ---- section 5
    L += paper_section(res)
    L += ["## Reproduce", "",
          f"    OMP_NUM_THREADS=1 .venv/bin/python scripts/agentdojo_recheck.py --reps {reps} --boots {boots} --workers {a['workers']}",
          "", f"{a['workers']} worker processes and the parent, one thread each, CPU only, {res['seconds'] / 60:.0f} minutes. "
          f"Inner bootstrap {boots:,} draws for each of t(user), t(inj) and the two-way bound on every resample; {BIG:,} for the "
          f"converged real-data limits. Seed {a['seed']}.", ""]
    return "\n".join(L)


def sizes_of(res):
    return sorted(res["seed_spread"][SECALIGN[0]], key=int)


def span(ps):
    return f"{ps['per_seed_min']}" if ps["per_seed_min"] == ps["per_seed_max"] else f"{ps['per_seed_min']}-{ps['per_seed_max']}"


def big_range(res, p):
    """The user-task limit at BIG inner draws: the SecAlign analysis's seeds and the converged table's."""
    v = res["secalign"][p]["t_user_big_seeds"] + [res["converged"][p]["t_user"]]
    return f"{min(v):.3f}" if f"{min(v):.3f}" == f"{max(v):.3f}" else f"{min(v):.3f}-{max(v):.3f}"


def p_above(share, boots):
    """P(a ``boots``-draw limit lands above CLIFF): the draw at rank delta is one of the share beyond it."""
    return float(binom.sf(int(np.floor(DELTA * (boots - 1))), boots, share))


def paper_section(res):
    sim, cert, a, dg = res["resampling"], res["certificates"], res["args"], res["digest"]
    cls, pas, npip, reps = dg["classes"], dg["passes"], len(sim), a["reps"]
    sec = res["secalign"]
    b4 = str(res["boots_real"])

    def r(key, sc):
        return f"{cls[key][sc]['least']:.3f}-{cls[key][sc]['worst']:.3f}"

    def cnt(key, sc):
        c = cls[key][sc]
        return f"{c['over']} over, {c['unresolved']} unresolved, {c['under']} at or under"

    sp0 = res["seed_spread"][SECALIGN[0]][b4] if sec else None
    one = pas["max"]["paper_seed"]
    L = ["## 5. What the paper can and cannot say", "",
         "**Can say.**", "",
         f"- {len(one)} of {npip} pipelines ({', '.join(one)}) certifies attack success at most 5% at delta 0.05. The count is the "
         "same under t(user), the larger-of-two rule, the dominant-cluster rule and `quad`, and at every bootstrap seed tried.",
         "- Meta-SecAlign-70B, raw rate 2.2%, does not certify: its user-task bound is "
         + (f"{big_range(res, SECALIGN[0])} at {BIG:,} inner draws ({sp0['t_user']['min']:.3f}-{sp0['t_user']['max']:.3f} over seeds at 4,000), above 5% every time." if sec else "above 5%."),
         f"- With user tasks treated as the sampled unit and the benchmark's injection tasks as fixed, the cluster bootstrap-t by "
         f"user task was at or under its level in {cls['t_user']['a']['under']} of {npip} pipelines, unresolved above delta in "
         f"{cls['t_user']['a']['unresolved']} and over in {cls['t_user']['a']['over']} (miss {r('t_user', 'a')}; "
         f"{reps:,} resamples, se {res['mc_se']:.4f}). This is the reading the paper's limitations paragraph already takes "
         "(\"The clustered bound treats user tasks as sampled from a population of tasks like them\").",
         f"- Per-pair Clopper-Pearson fails under every scheme ({cnt('naive', 'a')} with user tasks resampled; all {npip} over "
         "under the other two).", "",
         "**Cannot say.**", "",
         "- That the larger of the two clustered bounds is a certificate that respects the crossed design. It is over its level "
         f"when injection tasks are sampled ({cnt('max', 'b')}) and when both are ({cnt('max', 'c')}; worst {cls['max']['c']['worst']:.3f}).",
         "- That the dominant-cluster rule is equivalent to it. The verdicts agree on the real data; the dominant rule's miss "
         f"rate reaches {cls['dom']['b']['worst']:.3f} and {cls['dom']['c']['worst']:.3f}.",
         "- That the clustered bound for Meta-SecAlign-70B is 0.104, or that 0.104 against the variant's 0.057 reflects anything "
         "about the two pipelines.",
         "- Anything about attack success on injection tasks outside the benchmark's, unless a bound valid under scheme (b) or "
         "(c) is adopted. Of the seven here only `quad` was not over, and it has had one check, made after it was proposed "
         "for this purpose.",
         "- That any of these bounds is valid, as opposed to not shown invalid. Every cell resamples a pipeline's own table, "
         "which cannot contain failure clusters the benchmark did not happen to hit. That matters most for the one pipeline "
         + (f"that passes ({cert[one[0]]['k']} successes in {cert[one[0]]['any_inj']['k']} user tasks)." if len(one) == 1 else "with few successes."), "",
         "**Sentences to change.** Quoted from `reports/paper_certification.md`.", "",
         "Section 8.2:", ""]

    def item(n, label, quote, comment):
        quote = f"`{quote}`" if quote.startswith("|") else quote
        pad = " " * (len(str(n)) + 2)
        return [f"{n}. {label}", "", f"{pad}> {quote}", "", f"{pad}{comment}", ""]

    tu_a, mx_a, mx_b, mx_c = cls["t_user"]["a"], cls["max"]["a"], cls["max"]["b"], cls["max"]["c"]
    L += item(1, "First sentence.",
              "On AgentDojo's published runs (19,380 episodes, the harness's `security` label), a certificate that respects the "
              "design is the studentised cluster bootstrap by user task, or the larger of the two clustered bounds where injection "
              "tasks carry more dependence [020] (Figure 6).",
              "Not supported as written. State which factor is sampled. With injection tasks fixed, the certificate is t(user): "
              f"at or under its level in {tu_a['under']} of {npip}, unresolved above delta in {tu_a['unresolved']}, over in "
              f"{tu_a['over']} (worst {tu_a['worst']:.3f}). Under that reading the larger-of-two rule adds nothing (it is only more "
              f"conservative, worst miss {mx_a['worst']:.3f}); under the other two it is not valid ({mx_b['over']} and {mx_c['over']} "
              f"of {npip} over).")
    L += item(2, "Second sentence.",
              "The check behind this is narrow: 6 of the 28 pipelines, 400 resamples each (standard error 0.011), and user tasks "
              "resampled with injection tasks held fixed, although injection tasks carry the larger design effects.",
              f"Replace: all {npip} pipelines, {reps:,} resamples per cell (standard error {res['mc_se']:.4f}), three schemes "
              "(user tasks, injection tasks, both).")
    L += item(3, "Third sentence.", "The larger-of-two rule itself was not resampled.",
              f"Replace with the result: at or under its level in {mx_a['under']} of {npip} with user tasks resampled; over in "
              f"{mx_b['over']} with injection tasks resampled (worst {mx_b['worst']:.3f}); over in {mx_c['over']} with both, and "
              f"unresolved in {mx_c['unresolved']} more (worst {mx_c['worst']:.3f}).")
    L += item(4, "Table 6's caption.",
              "The certificate is the larger of the two clustered bounds; taking the dominant cluster's bound, as the source "
              "analysis does, gives the same verdicts.",
              "The verdicts do agree (and agree with t(user) alone and with `quad`). The caption should not call the "
              "larger-of-two rule \"the certificate\" without the scope: at or under its level with injection tasks fixed, over "
              f"it otherwise. The dominant-cluster rule failed its check ({cls['dom']['b']['over']} and {cls['dom']['c']['over']} of "
              f"{npip} over under (b) and (c)).")
    n = 5
    if sec:
        six = ("claude-3-5-sonnet-20241022", "command-r", "claude-3-7-sonnet-20250219", "gpt-4o-2024-05-13-tool_filter", "gpt-4o-2024-05-13")
        move = max(res["seed_spread"][p][b4][k]["max"] - res["seed_spread"][p][b4][k]["min"] for p in six for k in ("t_user", "t_inj"))
        L += item(5, "Table 6, second row.", "| Meta-SecAlign-70B | 949 | 0.022 | 0.032 | 0.104 | 0.035 | NSF |",
                  f"0.104 is one seed's value. Replace with {big_range(res, SECALIGN[0])} ({BIG:,} inner draws), or give the range "
                  f"{sp0['t_user']['min']:.3f}-{sp0['t_user']['max']:.3f} over seeds at 4,000. NSF stands. The other five rows "
                  f"move by at most {move:.3f} across seeds and keep their verdicts.")
        L += item(6, "After the table.",
                  "A defended model whose raw rate is 2.2% does not: its 21 successes sit in 5 of 97 user tasks (ICC 0.72), so the "
                  "clustered bound is 0.104.",
                  f"The counts are right (21 successes, 5 of 97 user tasks). The bound is {big_range(res, SECALIGN[0])}, not 0.104. That it does not "
                  "certify stands.")
        L += item(7, "Last sentence.", "The failures that remain are concentrated, and only a bound that respects the design shows it.",
                  "The first half stands. The second needs the scope of sentence 1: the bound that shows it is t(user), for user "
                  "tasks as the sampled unit.")
        n = 8
    L += ["\"One pipeline of 28 certifies.\" and the section title stand" + (" as they are." if len(one) == 1 else f": no, {len(one)} certify."), "",
          "Table 2, rows tagged `[020 P]`:", ""]
    L += item(n, "First row.",
              "| cluster bootstrap-t, by user task | approximate | AgentDojo, 6 pipelines, user tasks resampled | 0.05 | 0.020-0.060 | [020 P] |",
              f"Replace setting and range: AgentDojo, {npip} pipelines, user tasks resampled, miss {r('t_user', 'a')} ({cnt('t_user', 'a')}). "
              f"For the other two schemes it belongs in the failing block: {r('t_user', 'b')} with injection tasks resampled "
              f"({cls['t_user']['b']['over']} over) and {r('t_user', 'c')} with both ({cls['t_user']['c']['over']} over).")
    L += item(n + 1, "Second row.",
              "| **Clopper-Pearson over pairs** | fails on a crossed design | AgentDojo, user tasks resampled | 0.05 | **0.048-0.275** | [020 P] |",
              f"Replace range: {r('naive', 'a')} over {npip} pipelines with user tasks resampled ({cls['naive']['a']['over']} over); "
              f"{r('naive', 'b')} with injection tasks resampled; {r('naive', 'c')} with both.")
    L += item(n + 2, "Third row.",
              "| **two-way bootstrap** | fails where positives sit in few clusters | AgentDojo | 0.05 | **up to 0.170** | [020 P] |",
              f"Replace: up to {cls['twoway']['a']['worst']:.3f} with user tasks resampled ({cls['twoway']['a']['over']} of {npip} over), "
              f"{cls['twoway']['b']['worst']:.3f} with injection tasks ({cls['twoway']['b']['over']} over), {cls['twoway']['c']['worst']:.3f} "
              f"with both ({cls['twoway']['c']['over']} over). \"Where positives sit in few clusters\" describes scheme (a) only.")
    L += [f"{n + 3}. A row is missing: the larger of the two clustered bounds, Table 6's rule. Miss {r('max', 'a')}, {r('max', 'b')} and "
          f"{r('max', 'c')} under the three schemes; by the paper's own classification it goes in the failing block for (b) and (c).", ""]
    L += item(n + 4, "Table 2's caption.", "Monte Carlo standard errors are 0.003-0.004 for 5,000 resamples and 0.011 for 400.",
              f"These rows now have {reps:,} resamples (se {res['mc_se']:.4f}); no AgentDojo row has 400.")
    L += ["The same numbers are repeated outside the passages asked about and would become inconsistent: the abstract "
          "(\"the usual per-pair bound misses in 5-28% of resamples\"), the contributions list (\"under a bound that respects "
          "the design\"), section 7.3 (\"missed in 5-28% of resamples at delta 0.05\") and the limitations (\"The AgentDojo "
          "check covers 6 of 28 pipelines and resamples one of the two crossed factors\"). The per-pair miss is now "
          f"{cls['naive']['a']['least']:.0%}-{cls['naive']['a']['worst']:.0%} with user tasks resampled and up to "
          f"{max(cls['naive'][s]['worst'] for s in 'bc'):.0%} under the other two schemes.", ""]
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=4000)
    ap.add_argument("--boots", type=int, default=4000, help="inner bootstrap size on every resample")
    ap.add_argument("--workers", type=int, default=4, help="worker processes (the parent makes one more)")
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--seeds", type=int, default=30, help="bootstrap seeds for the real-data spread")
    ap.add_argument("--raw", default="", help="optional .npz path for the per-resample quantities")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "agentdojo_recheck"))
    ap.add_argument("--render", action="store_true", help="rewrite the .md from the existing .json, no computation")
    a = ap.parse_args()
    assert a.workers <= 4, "at most 5 processes including the parent"
    if a.render:
        open(a.out + ".md", "w").write(report(json.load(open(a.out + ".json"))))
        return
    t0 = time.time()
    R_by, n_rows = load_rows()
    tables = {p: table(R) for p, R in R_by.items()}
    pipes = list(R_by)
    check = selfcheck(tables)
    print("selfcheck", check, flush=True)
    cert, cert_diff = certificates(R_by, json.load(open(os.path.join(SPIKE, "cert.json"))))
    print(f"real-data certificates, max |difference| from cert.json: {cert_diff:.2e}", flush=True)
    sec = secalign(tables)
    print(f"SecAlign analysis done {time.time() - t0:.0f}s", flush=True)

    sizes = sorted({1000, 4000, a.boots})
    seeds = list(range(a.seeds))
    jobs = [(p, pi, sc, si, tables[p][0], a.reps, a.boots, a.seed)
            for pi, p in enumerate(pipes) for si, (sc, _) in enumerate(SCHEMES)]
    raw, sim, spread, conv = {}, {p: {} for p in pipes}, {p: {} for p in pipes}, {}
    with ProcessPoolExecutor(a.workers) as ex:
        for p, boots, s in ex.map(spread_cell, [(p, tables[p][0], seeds, b) for p in pipes for b in sizes]):
            spread[p][str(boots)] = s
        for p, b in ex.map(converged_cell, [(p, tables[p][0]) for p in pipes]):
            conv[p] = b
        print(f"seed spread and {BIG}-draw limits done {time.time() - t0:.0f}s", flush=True)
        for done, (p, sc, out) in enumerate(ex.map(run_cell, jobs), 1):
            sim[p][sc] = summarise(out, cert[p]["rate"], a.reps)
            raw[f"{p}|{sc}"] = out
            print(f"[{done}/{len(jobs)}] {p} {sc} " + " ".join(f"{k}={sim[p][sc][k]['miss']:.3f}" for k, _ in BOUNDS)
                  + f" {time.time() - t0:.0f}s", flush=True)
    if a.raw:
        np.savez_compressed(a.raw, **raw)
    res = dict(args=vars(a), n_rows=n_rows, n_pairs=sum(len(R) for R in R_by.values()), delta=DELTA, tau=TAU,
               mc_se=float(np.sqrt(DELTA * (1 - DELTA) / a.reps)), inner_boots=a.boots, boots_real=4000, big=BIG,
               selfcheck=check, cert_json_max_abs_diff=cert_diff, certificates=cert, seed_spread=spread, converged=conv,
               resampling=sim, secalign=sec)
    res["digest"] = digest(res)
    res["seconds"] = time.time() - t0
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out + ".json", "w"), indent=1, default=float)
    open(a.out + ".md", "w").write(report(json.load(open(a.out + ".json"))))
    print(f"wrote {a.out}.md and .json in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
