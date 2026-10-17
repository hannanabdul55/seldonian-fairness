"""B6 (blind review of v0.9.10, M9): AgentDojo's bounds against a truth outside the observed table.

Registered in `.planning/paper-certification/B6_registration.md` before any outcome was computed.

Why. The validity checks of section 10.2 (``agentdojo_recheck.py``, ``agentdojo_twoway.py``,
``agentdojo_betting.py``) redraw rows and columns of each pipeline's observed table and score every bound
against that table's own rate. That is the dependence structure the pigeonhole bootstrap assumes, so the
check is partly circular. Here each table is replaced by a fitted generative model and the truth is the
model's rate, which no table shows.

Model, per pipeline. A crossed logit-normal model inside each suite block:
``P(success of user task i under injection task j) = expit(mu_s + u_i + v_j)``, with ``u_i ~ N(0, su^2)``
and ``v_j ~ N(0, sv^2)`` independent, ``s`` the suite, the table's block pattern (which pairs exist) kept.
Fit by simulated moments: on a grid of (su, sv) in 0, 0.25, ..., 6, each ``mu_s`` is set by quadrature so
that the model's suite rate equals the observed one, Jeffreys-smoothed ((k_s + 0.5) / (n_s + 1)); the grid
point kept is the one whose simulated within-suite variances of user-task rates and of injection-task rates
(64 tables, common random numbers) are closest on the log scale to the observed ones.

Two sampling schemes, 4,000 tables each:

- ``P_a``  injection tasks fixed (their ``v_j`` drawn once per pipeline), user tasks drawn afresh: the
           design the issued certificate assumes. Truth: the model's rate given those injection tasks.
- ``P_c``  both kinds of task drawn afresh. Truth: the model's population rate.

Truth is the pair rate (pairs weighted as in the table) for every arm and also, for the betting bound on
user-task rates, the average over user tasks (``bet_a1@uavg``). Arms: ``agentdojo_betting.one_table``
unchanged (``bet_a1``, ``bet_a2``, ``t_user``, ``t_inj``, ``quad``, ``max``, ``pig_t``, ``cgm_t``), at the
recheck's 4,000 inner bootstrap draws. A NaN bound counts as a miss. Cells are classed by the paper's rule.
Menzel's (2021) bootstrap is not run.

    OMP_NUM_THREADS=1 .venv/bin/python scripts/agentdojo_parametric.py fit        # the fitted models only
    OMP_NUM_THREADS=1 .venv/bin/python scripts/agentdojo_parametric.py run --workers 14
    .venv/bin/python scripts/agentdojo_parametric.py report
    -> results/paper/agentdojo_parametric.json, .md
"""
import argparse
import json
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import agentdojo_betting as AB   # noqa: E402

ROOT = AB.ROOT
OUT = os.path.join(ROOT, "results", "paper", "agentdojo_parametric")
SEED = 20261016
REPS = 4000
BOOTS = AB.BOOTS
SIGMAS = np.arange(0.0, 6.0001, 0.25)
FIT_SIMS = 64
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(81)
GH_W = GH_W / GH_W.sum()
SCHEMES = (("P_a", "injection tasks fixed, user tasks drawn from the model"),
           ("P_c", "both kinds of task drawn from the model"))
ARMS = [k for k, _ in AB.ARMS]


def gmean(mu, sd):
    """E expit(mu + sd Z)."""
    return float(GH_W @ expit(mu + sd * GH_X))


def layout(M, U):
    """Suite of every row, the observed-pair mask, and per suite: rows, columns, observed rate."""
    V = ~np.isnan(M)
    rs = np.array([AB.suite(u) for u in U])
    suites = sorted(set(rs))
    blk = {}
    for s in suites:
        r = np.flatnonzero(rs == s)
        c = np.flatnonzero(V[r].any(0))
        k, n = float(np.nansum(M[np.ix_(r, c)])), float(V[np.ix_(r, c)].sum())
        blk[s] = dict(rows=r, cols=c, k=k, n=n, rate=(k + 0.5) / (n + 1.0))
    return V, rs, blk


def moments(M, V, blk):
    """Pooled within-suite variance of user-task rates and of injection-task rates."""
    vu, vi = [], []
    for b in blk.values():
        sub, m = M[np.ix_(b["rows"], b["cols"])], V[np.ix_(b["rows"], b["cols"])]
        ru = np.nansum(sub, 1) / m.sum(1)
        ri = np.nansum(sub, 0) / m.sum(0)
        vu.append((ru - ru.mean()) ** 2)
        vi.append((ri - ri.mean()) ** 2)
    return float(np.concatenate(vu).mean()), float(np.concatenate(vi).mean())


def mus(blk, su, sv):
    sd = np.hypot(su, sv)
    return {s: brentq(lambda m: gmean(m, sd) - b["rate"], -40, 40) for s, b in blk.items()}


def simulate(shape, V, blk, mu, su, sv, rng, v_fixed=None):
    """One table from the model; ``v_fixed`` holds the injection effects under P_a."""
    nu, ni = shape
    u = rng.standard_normal(nu) * su
    v = v_fixed if v_fixed is not None else rng.standard_normal(ni) * sv
    eta = np.full(shape, -np.inf)
    for s, b in blk.items():
        eta[np.ix_(b["rows"], b["cols"])] = mu[s] + u[b["rows"]][:, None] + v[b["cols"]][None, :]
    Y = (rng.random(shape) < expit(eta)).astype(float)
    return np.where(V, Y, np.nan)


def fit(p, M, U):
    V, rs, blk = layout(M, U)
    obs = np.array(moments(M, V, blk))
    best = None
    for su in SIGMAS:
        for sv in SIGMAS:
            mu = mus(blk, su, sv)
            rng = np.random.default_rng([SEED, zlib.crc32(p.encode()), 0])      # common random numbers
            sim = np.mean([moments(simulate(M.shape, V, blk, mu, su, sv, rng), V, blk) for _ in range(FIT_SIMS)], axis=0)
            err = float(np.sum((np.log(sim + 1e-4) - np.log(obs + 1e-4)) ** 2))
            if best is None or err < best[0]:
                best = (err, float(su), float(sv), mu, sim)
    err, su, sv, mu, sim = best
    return dict(su=su, sv=sv, mu=mu, err=err, obs=obs.tolist(), sim=sim.tolist(),
                blk={s: dict(rows=len(b["rows"]), cols=len(b["cols"]), k=b["k"], n=b["n"]) for s, b in blk.items()})


def truths(M, U, f, v_fixed=None):
    """Pair rate and user-task average of the model: over v (P_c) or given ``v_fixed`` (P_a)."""
    V, rs, blk = layout(M, U)
    pair_num, uavg_num = 0.0, 0.0
    nu = sum(len(b["rows"]) for b in blk.values())
    npairs = V.sum()
    for s, b in blk.items():
        cnt = V[np.ix_(b["rows"], b["cols"])]
        if v_fixed is None:
            g = gmean(f["mu"][s], np.hypot(f["su"], f["sv"]))
            pair_num += g * cnt.sum()
            uavg_num += g * len(b["rows"])
        else:
            gj = np.array([gmean(f["mu"][s] + v_fixed[j], f["su"]) for j in b["cols"]])
            pair_num += float((cnt * gj[None, :]).sum())
            uavg_num += float(((cnt * gj[None, :]).sum(1) / cnt.sum(1)).sum())
    return dict(pair=float(pair_num / npairs), uavg=float(uavg_num / nu))


def v_draw(p, f, ni):
    return np.random.default_rng([SEED, zlib.crc32(p.encode()), 1]).standard_normal(ni) * f["sv"]


def run_cell(args):
    p, scheme, M, U, f, reps, boots = args
    V, rs, blk = layout(M, U)
    vf = v_draw(p, f, M.shape[1]) if scheme == "P_a" else None
    rng = np.random.default_rng(np.random.SeedSequence([SEED, zlib.crc32(f"{p}|{scheme}".encode())]))
    out = np.empty((reps, len(AB.COLS)))
    for r in range(reps):
        Mr = simulate(M.shape, V, blk, f["mu"], f["su"], f["sv"], rng, vf)
        out[r] = AB.one_table(Mr, rng, boots, rs)
    return p, scheme, out, truths(M, U, f, vf)


def fit_job(args):
    p, M, U = args
    return p, fit(p, M, U)


def classify(m, reps):
    se = np.sqrt(AB.DELTA * (1 - AB.DELTA) / reps)
    return "over" if m > AB.DELTA + 2 * se else ("unresolved" if m > AB.DELTA else "at or under")


def summarise(out, tr):
    c = {k: out[:, j] for j, k in enumerate(AB.COLS)}
    s = {}
    for key in ARMS:
        col, t = (c["bet_a1"], tr["uavg"]) if key == "bet_a1@uavg" else (c[key], tr["pair"])
        s[key] = dict(miss=AB.miss(col, t), nan=int(np.isnan(col).sum()), median=float(np.nanmedian(col)),
                      pass_at_tau=float(np.mean(col <= AB.TAU)))
    s["zero_success"] = float(np.mean(c["k"] == 0))
    return s


PRED = {
    "W1": "under P_a, the betting bound on user-task rates (truth: the user-task average) and the pair-rate betting bound are over for no pipeline",
    "W2": "under P_a, the bound clustered by user task is over for at most 2 of 28 pipelines",
    "W3": "under P_c, the bound clustered by user task is over for at least 8 of 28 pipelines",
    "W4": "under P_c, the quadrature bound is over for at most 2 of 28 pipelines",
    "W5": "under P_c, the betting bound on user-task rates (truth: the user-task average) is over for at least 2 of 28 pipelines",
}


def score(res):
    R = res["reps"]
    over = lambda sc, arm: sum(classify(res["sim"][p][sc][arm]["miss"], R) == "over" for p in res["sim"])
    return {"W1": (over("P_a", "bet_a1@uavg") == 0 and over("P_a", "bet_a2") == 0,
                   f"{over('P_a', 'bet_a1@uavg')} and {over('P_a', 'bet_a2')}"),
            "W2": (over("P_a", "t_user") <= 2, f"{over('P_a', 't_user')}"),
            "W3": (over("P_c", "t_user") >= 8, f"{over('P_c', 't_user')}"),
            "W4": (over("P_c", "quad") <= 2, f"{over('P_c', 'quad')}"),
            "W5": (over("P_c", "bet_a1@uavg") >= 2, f"{over('P_c', 'bet_a1@uavg')}")}


def report(res, path):
    R, sim, fits = res["reps"], res["sim"], res["fits"]
    L = ["# B6: AgentDojo's bounds against a fitted crossed model", "",
         f"`scripts/agentdojo_parametric.py`, registered in `.planning/paper-certification/B6_registration.md`. "
         f"{R:,} tables a pipeline and scheme, delta {AB.DELTA}; inner bootstrap {res['boots']:,} draws.", "",
         "## Registered predictions", "", "| # | prediction | held | count |", "|---|---|---|---|"]
    for k, (ok, det) in score(res).items():
        L.append(f"| {k} | {PRED[k]} | {'kept' if ok else '**refuted**'} | {det} |")
    L += ["", "## Over / unresolved / at or under, of 28 pipelines", "", "| arm | " + " | ".join(s for s, _ in SCHEMES) + " |",
          "|---|" + "---|" * len(SCHEMES)]
    for arm in ARMS:
        cells = []
        for sc, _ in SCHEMES:
            k = [classify(sim[p][sc][arm]["miss"], R) for p in sim]
            mx = max(sim[p][sc][arm]["miss"] for p in sim)
            cells.append(f"{k.count('over')} / {k.count('unresolved')} / {k.count('at or under')} (max {mx:.3f})")
        L.append(f"| {arm} | " + " | ".join(cells) + " |")
    L += ["", "## Per pipeline", "", "| pipeline | su | sv | table rate | truth P_a / P_c | t_user miss P_a / P_c | "
          "quad miss P_c | bet_a1@uavg miss P_a / P_c | P(t_user <= 5%) P_a / P_c | P(bet_a1 <= 5%) P_a |", "|---|---|---|---|---|---|---|---|---|---|"]
    for p in sim:
        a, c = sim[p]["P_a"], sim[p]["P_c"]
        L.append(f"| {p} | {fits[p]['su']:.2f} | {fits[p]['sv']:.2f} | {res['table_rate'][p]:.3f} | "
                 f"{res['truth'][p]['P_a']['pair']:.3f} / {res['truth'][p]['P_c']['pair']:.3f} | "
                 f"{a['t_user']['miss']:.3f} / {c['t_user']['miss']:.3f} | {c['quad']['miss']:.3f} | "
                 f"{a['bet_a1@uavg']['miss']:.3f} / {c['bet_a1@uavg']['miss']:.3f} | "
                 f"{a['t_user']['pass_at_tau']:.3f} / {c['t_user']['pass_at_tau']:.3f} | {a['bet_a1']['pass_at_tau']:.3f} |")
    L += ["", "## Fit (observed and simulated within-suite variances of user-task and injection-task rates)", "",
          "| pipeline | su | sv | observed | simulated | log error |", "|---|---|---|---|---|---|"]
    for p, f in fits.items():
        L.append(f"| {p} | {f['su']:.2f} | {f['sv']:.2f} | {f['obs'][0]:.4f}, {f['obs'][1]:.4f} | "
                 f"{f['sim'][0]:.4f}, {f['sim'][1]:.4f} | {f['err']:.3f} |")
    open(path, "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fit", "run", "report"])
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--boots", type=int, default=BOOTS)
    a = ap.parse_args()
    if a.stage == "report":
        res = json.load(open(OUT + ".json"))
        report(res, OUT + ".md")
        print(f"-> {OUT}.md")
        return
    tables = AB.load()
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        fits = dict(ex.map(fit_job, [(p, M, U) for p, (M, U, I) in tables.items()]))
    print(f"fitted {len(fits)} in {time.time() - t0:.0f}s", flush=True)
    if a.stage == "fit":
        json.dump(fits, open(OUT + "_fit.json", "w"), indent=1)
        print(f"-> {OUT}_fit.json")
        return
    jobs = [(p, sc, tables[p][0], tables[p][1], fits[p], a.reps, a.boots) for p in tables for sc, _ in SCHEMES]
    sim, truth = {p: {} for p in tables}, {p: {} for p in tables}
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (p, sc, out, tr) in enumerate(ex.map(run_cell, jobs), 1):
            sim[p][sc] = summarise(out, tr)
            truth[p][sc] = tr
            print(f"{i}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    res = dict(seed=SEED, reps=a.reps, boots=a.boots, fits=fits, truth=truth, sim=sim,
               table_rate={p: float(np.nanmean(tables[p][0])) for p in tables}, seconds=time.time() - t0)
    json.dump(res, open(OUT + ".json", "w"), indent=1)
    report(res, OUT + ".md")
    print(f"-> {OUT}.md")


if __name__ == "__main__":
    main()
