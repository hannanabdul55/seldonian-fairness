"""Do the reference-rate-strata bounds keep their level when the safety set is a small share of
the prompt population? (audit of 2026-10-06, item 12)

Spikes 013 and 014 and part A of ``scripts/stratppi_baseline.py`` draw 100 or 200 prompts without
replacement from a pool of about 500 and compare every bound with the pool's own rate. No bound
there has a finite-population correction, so at a sampling fraction of 20-40% each one is wider
than the spread of its estimate and the miss rates come out low. This script redraws the same
cells with replacement within the same strata, which is the limit of a pool much larger than the
safety set with the same strata and the same rate. Everything else is the baseline's: pools,
strata, sizes, estimators.

Three draws of every cell:

- ``stored``  without replacement, the baseline's seeds. The arms shared with
              ``results/paper/stratppi.json`` must come back to the last digit (asserted).
- ``paired``  with replacement, the same seeds and the same number of draws: only the draw changes.
- ``large``   with replacement, 40,000 draws from seeds of its own (a cell's label is in the seed,
              so labels no longer share a stream). The counts the paper quotes are these: at
              5,000 draws the largest miss over 28 cells is read too high and cells a point over
              their level are left unresolved.

With replacement the pool is taken as unbounded for StratPPI as well: the predictor's mean in
each stratum is known, so the estimator's N_h is infinite (its coefficient is not shrunk and its
variance carries no term for the unlabelled sample).

Arms on the stratified draw: ``b1w``, the Wald-t ``b1``, StratPPI's normal limit, StratPPI's
estimator with a bootstrap-t limit. Arms on a simple random draw of the same size: the pooled
Wilson bound (``b1w`` at one stratum) and Clopper-Pearson. With replacement the simple random draw
is i.i.d. Bernoulli at the pool's rate, so Clopper-Pearson is exact there and is the control for
the harness. The control is checked against the binomial: for both arms on the random draw the
script computes the exact miss probability at the pool's rate (with ``b1w``'s own limit at every
count, since its grid can sit a quarter of a thousandth above the closed-form Wilson limit) and
reports the standardised differences. The cell with the largest one is redrawn twice more.

Cells are classified by the paper's rule: with se = sqrt(delta (1 - delta) / R), over if the miss
exceeds delta + 2 se, unresolved if it lies in (delta, delta + 2 se], at or under otherwise.

    OMP_NUM_THREADS=1 .venv/bin/python scripts/replacement_check.py            # about 20 minutes on 12 cores
    .venv/bin/python scripts/replacement_check.py --control-only           # the binomial control on the stored cells
    .venv/bin/python scripts/replacement_check.py --report-only
    -> results/paper/replacement_check.json, results/paper/replacement_check.md
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
from scipy.stats import binom

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "results", "paper")
sys.path.insert(0, HERE)
import stratppi_baseline as BL    # noqa: E402

c, PM, P14, RP, SB = BL.c, BL.PM, BL.P14, BL.RP, BL.SB
DELTAS, K_REF, H = BL.DELTAS, BL.K_REF, BL.H
MID = ("C1:refusal", "C2:unsafe", "C2:refusal", "C3:refusal")
RARE = ("C2:gated", "C3:unsafe")
OVER, UNRES, UNDER = "over", "unresolved", "at or under"
SHARED = {"b1w": "S2 + b1w (this paper)", "Wald-t b1": "S2 + Wald-t b1", "StratPPI, normal limit": "S2 + StratPPI",
          "StratPPI estimator, bootstrap-t": "S2 + StratPPI, bootstrap-t", "pooled Wilson": "random + pooled Wilson (R)"}
ARMS = ("b1w", "Wald-t b1", "StratPPI, normal limit", "StratPPI estimator, bootstrap-t", "pooled Wilson", "Clopper-Pearson")
STRAT_ESS = ("b1w", "Wald-t b1", "StratPPI estimator, bootstrap-t")
RANDOM_ARMS = ("pooled Wilson", "Clopper-Pearson")
DRAWS = ("stored", "paired", "large")
UNBOUNDED = 1e12                   # N_h of a pool far larger than the safety set


def draw_split(strata, n_s, rng, replace):
    """013's proportional stratified draw; ``replace`` draws each stratum's prompts with replacement."""
    if not replace:
        return PM._draw_split(strata, n_s, rng)
    N = len(strata)
    idx, st = [], []
    for h in np.unique(strata):
        mem = np.flatnonzero(strata == h)
        k = max(2, int(round(n_s * len(mem) / N)))
        idx.append(rng.choice(mem, size=min(k, len(mem)), replace=True))
        st.append(np.full(len(idx[-1]), h))
    return np.concatenate(idx), np.concatenate(st)


def setup(src, pool, lab, step):
    PM.TIES = "random"
    by, meta = RP.load() if src == "013" else P14.load("s0")
    P, _ = RP.build_pool(by, meta, pool, lab, step, False, "eval")
    f = P["cov"][:, :K_REF].mean(axis=1)
    rng = np.random.default_rng([step, zlib.crc32(repr(("S2", K_REF, H)).encode())])   # 013's S2 strata
    strata = PM.quantile_strata(f, H, rng, ties="random")
    return P, f, strata


def job(args):
    src, pool, lab, step, reps, big = args
    P, f, strata = setup(src, pool, lab, step)
    draw, truth, N = P["draw"], float(P["truth"]), len(P["p_truth"])
    W = np.bincount(strata) / N
    finite = [(int((strata == h).sum()), float(f[strata == h].mean()), float(f[strata == h].var(ddof=1))) for h in range(H)]
    unbounded = [(UNBOUNDED, m, v) for _, m, v in finite]
    own = zlib.crc32(f"{src}:{pool}:{lab}".encode())
    rows = []
    for n_s in (100, 200):
        for name, replace, R, stats, seed in (("stored", False, reps, finite, lambda r: [step, r, n_s]),
                                              ("paired", True, reps, unbounded, lambda r: [step, r, n_s]),
                                              ("large", True, big, unbounded, lambda r: [own, step, r, n_s])):
            acc = collections.defaultdict(lambda: collections.defaultdict(list))
            for r in range(R):
                rr = np.random.default_rng(seed(r))
                ids, st = draw_split(strata, n_s, rr, replace)
                y = draw(ids, rr)
                s = np.array([y[st == h].sum() for h in range(H)]); n = np.array([(st == h).sum() for h in range(H)])
                est = SB.estimate(s, n, W)
                sp, sp_est = BL.stratppi(y, f[ids], st, stats, W, DELTAS)
                spb, _ = BL.stratppi_boot(y, f[ids], st, stats, W, DELTAS, rr)
                for d, u, ub_ in zip(DELTAS, sp, spb):
                    for arm, ub, e in (("StratPPI estimator, bootstrap-t", ub_, sp_est), ("b1w", SB.b1w(s, n, W, d), est),
                                       ("Wald-t b1", SB.b1(s, n, W, d), est), ("StratPPI, normal limit", u, sp_est)):
                        acc[(arm, d)]["ub"].append(ub); acc[(arm, d)]["est"].append(e)
                rr = np.random.default_rng(seed(r))                      # 013's arm R on its own draw
                ids = rr.choice(N, size=n_s, replace=replace)
                y = draw(ids, rr)
                for d in DELTAS:
                    for arm, ub in (("pooled Wilson", SB.b1w(y.sum(), n_s, 1.0, d)),
                                    ("Clopper-Pearson", float(c.cp_upper(y.sum(), n_s, d)))):
                        acc[(arm, d)]["ub"].append(ub); acc[(arm, d)]["est"].append(y.mean())
            for (arm, d), v in acc.items():
                rows.append(dict(src=src, env=f"{pool}:{lab}", step=step, n=n_s, N=N, delta=d, arm=arm, draw=name, replace=replace,
                                 truth=truth, reps=R, **BL.summarise(v["ub"], v["est"], truth)))
    return rows


def control(rows, redo_reps=40000):
    """The random-draw arms of the large pass against the exact binomial miss probability."""
    lim = {}

    def limits(arm, n, d):
        if (arm, n, d) not in lim:
            k = np.arange(n + 1)
            lim[(arm, n, d)] = np.array([SB.b1w(int(x), n, 1.0, d) for x in k]) if arm == "pooled Wilson" else c.cp_upper(k, n, d)
        return lim[(arm, n, d)]

    out = []
    for r in rows:
        if r["draw"] == "large" and r["arm"] in RANDOM_ARMS:
            U = limits(r["arm"], r["n"], r["delta"])
            ex = float(binom.pmf(np.arange(r["n"] + 1), r["n"], r["truth"])[U < r["truth"]].sum())
            se = np.sqrt(max(ex * (1 - ex), 1e-12) / r["reps"])
            out.append(dict(src=r["src"], env=r["env"], step=r["step"], n=r["n"], delta=r["delta"], arm=r["arm"], truth=r["truth"],
                            miss=r["miss"], exact=ex, z=float((r["miss"] - ex) / se)))
    assert all(o["exact"] <= o["delta"] + 1e-9 for o in out if o["arm"] == "Clopper-Pearson")
    z = np.array([o["z"] for o in out])
    w = max(out, key=lambda o: abs(o["z"]))
    P, _, _ = setup(w["src"], *w["env"].split(":"), w["step"])
    N, U, redo = len(P["p_truth"]), limits(w["arm"], w["n"], w["delta"]), []
    for base in (7, 12345):                                    # seeds no cell uses
        miss = 0
        for r in range(redo_reps):
            rr = np.random.default_rng([base, r, w["n"]])
            ids = rr.choice(N, size=w["n"], replace=True)
            miss += U[int(P["draw"](ids, rr).sum())] < w["truth"]
        redo.append(miss / redo_reps)
    cp = [o for o in out if o["arm"] == "Clopper-Pearson" and o["env"] in MID]
    return dict(cells=len(out), z_mean=float(z.mean()), z_sd=float(z.std(ddof=1)), z_max=float(np.abs(z).max()),
                cp_exact_mid=[min(o["exact"] / o["delta"] for o in cp), max(o["exact"] / o["delta"] for o in cp)],
                worst=w, redo=redo, redo_reps=redo_reps, rows=out)


def classify(m, delta, R):
    se = np.sqrt(delta * (1 - delta) / R)
    return OVER if m > delta + 2 * se else UNRES if m > delta else UNDER


def counts(rows):
    k = collections.Counter(classify(r["miss"], r["delta"], r["reps"]) for r in rows)
    return f"{k[OVER]} / {k[UNRES]} / {k[UNDER]}"


def tag(r):
    return f"{r['env']}{' (014)' if r['src'] == '014' else ''}"


def report(res, path):
    rows, R, B, ctl = res["rows"], res["reps"], res["reps_large"], res["control"]

    def sel(draw, arm=None, d=None, envs=None):
        return [r for r in rows if r["draw"] == draw and (arm is None or r["arm"] == arm) and (d is None or r["delta"] == d)
                and (envs is None or r["env"] in envs)]

    L = ["# Reference-rate strata redrawn with replacement", "",
         "`scripts/replacement_check.py`. The cells are those of part A of `scripts/stratppi_baseline.py`: 8 equal rank strata of "
         "the reference's 8-sample rate, proportional allocation, a safety set of 100 or 200 prompts from a pool of about 500, by "
         "label and checkpoint. Three draws of every cell:", "",
         f"- *stored*: without replacement, {R:,} draws, the baseline's seeds. The safety set is 20-40% of the pool and the truth is "
         "the pool's rate.",
         f"- *paired*: each stratum's prompts drawn with replacement, the same seeds and {R:,} draws, so only the draw changes.",
         f"- *large*: with replacement, {B:,} draws from seeds of its own. These are the counts to quote.", "",
         "With replacement is the limit of a pool far larger than the safety set, and StratPPI's N_h is taken as unbounded there "
         "(the predictor's stratum mean is known). Counts are cells over / unresolved / at or under delta (over: miss above delta "
         f"+ 2 se, se = sqrt(delta (1 - delta) / R); at delta 0.05 the band ends at {0.05 + 2 * np.sqrt(0.0475 / R):.4f} for {R:,} "
         f"draws and at {0.05 + 2 * np.sqrt(0.0475 / B):.4f} for {B:,}).", "",
         f"Replay check: the arms shared with `results/paper/stratppi.json` differ from it by at most {res['replay_diff']:.1e} in "
         f"miss over {res['replay_cells']} stored cells.", "",
         "Control: on the random draw with replacement the labels are i.i.d. at the pool's rate, so the miss probability of the "
         f"pooled Wilson bound and of Clopper-Pearson is a binomial sum. Over the {ctl['cells']} such cells of the large pass the "
         f"simulated miss minus the exact one, in standard errors, has mean {ctl['z_mean']:+.2f}, standard deviation "
         f"{ctl['z_sd']:.2f} and largest size {ctl['z_max']:.2f} ({ctl['worst']['arm']}, {tag(ctl['worst'])}, checkpoint "
         f"{ctl['worst']['step']}, n_s {ctl['worst']['n']}, delta {ctl['worst']['delta']}: {ctl['worst']['miss']:.4f} against "
         f"{ctl['worst']['exact']:.4f}). That cell redrawn twice more at {ctl['redo_reps']:,} draws gave "
         + " and ".join(f"{x:.4f}" for x in ctl["redo"]) + ". Clopper-Pearson's exact miss probability is at or under delta in "
         f"every one of these cells, as it must be ({ctl['cp_exact_mid'][0]:.2f} to {ctl['cp_exact_mid'][1]:.2f} of delta on the "
         "mid-rate labels), so a cell of its above delta is Monte Carlo noise. The pooled Wilson arm is `b1w` at one stratum, whose "
         "grid can sit 0.00025 above the closed-form limit; the exact values use the grid's limits. Where the pool's rate falls "
         "inside that gap the closed form gives a different miss (C2:unsafe, checkpoint 100, n_s 200, delta 0.1: 0.119 against "
         "0.076).", "",
         "## Counts", "",
         f"| bound | labels | delta | cells | stored, {R:,} | paired, {R:,} | large, {B:,} | largest miss, large |",
         "|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        for kind, envs in (("mid-rate (9-95%)", MID), ("rare (1-2%)", RARE)):
            for d in DELTAS:
                a, b, g = (sel(x, arm, d, envs) for x in DRAWS)
                L.append(f"| {arm} | {kind} | {d} | {len(g)} | {counts(a)} | {counts(b)} | {counts(g)} | {max(r['miss'] for r in g):.4f} |")
    L += ["", "`b1w` in the large pass, by label (cells over / unresolved / at or under; range of the miss):", "",
          "| label | rates of its cells | cells | delta 0.05 | delta 0.10 |", "|---|---|---|---|---|"]
    for env in ("C1:refusal", "C2:unsafe", "C3:refusal", "C2:refusal", "C2:gated", "C3:unsafe"):
        g = sel("large", "b1w", None, (env,))
        line = f"| {env} | {min(r['truth'] for r in g):.3f}-{max(r['truth'] for r in g):.3f} | {len(g) // 2} "
        for d in DELTAS:
            x = [r for r in g if r["delta"] == d]
            line += f"| {counts(x)}; {min(r['miss'] for r in x):.4f}-{max(r['miss'] for r in x):.4f} "
        L.append(line + "|")
    L += ["", "## Cells that are not at or under delta in the large pass, delta 0.05", "",
          "| bound | label | checkpoint | n_s | truth | miss, stored | miss, paired | miss, large | class, large |",
          "|---|---|---|---|---|---|---|---|---|"]
    key = lambda r: (r["arm"], r["src"], r["env"], r["step"], r["n"], r["delta"])      # noqa: E731
    by = {d: {key(r): r for r in sel(d)} for d in DRAWS}
    for arm in ARMS:
        if arm == "StratPPI, normal limit":
            x = sel("large", arm, 0.05)
            L.append(f"| {arm} | all 40 cells | | | | | | {min(r['miss'] for r in x):.4f}-{max(r['miss'] for r in x):.4f} | {counts(x)} |")
            continue
        for r in sel("large", arm, 0.05):
            if classify(r["miss"], 0.05, B) != UNDER:
                L.append(f"| {arm} | {tag(r)} | {r['step']} | {r['n']} | {r['truth']:.4f} | {by['stored'][key(r)]['miss']:.4f} | "
                         f"{by['paired'][key(r)]['miss']:.4f} | {r['miss']:.4f} | {classify(r['miss'], 0.05, B)} |")
    cols = [(a_, d_) for a_ in STRAT_ESS for d_ in ("stored", "large")]
    L += ["", "## Width and effective sample size, delta 0.05, last checkpoint", "",
          "ESS = (excess of the pooled Wilson bound over the truth / the arm's excess)^2, each on its own kind of draw.", "",
          "| label | n_s | truth | `b1w`, stored | `b1w`, large | Wald-t, stored | Wald-t, large | StratPPI bootstrap-t, stored | "
          "StratPPI bootstrap-t, large |", "|---|---|---|---|---|---|---|---|---|"]
    last = collections.defaultdict(int)
    for r in rows:
        last[(r["src"], r["env"])] = max(last[(r["src"], r["env"])], r["step"])
    med = collections.defaultdict(list)
    for (src, env), step in last.items():
        if env not in MID:
            continue
        for n in (100, 200):
            g = {(r["arm"], r["draw"]): r for r in rows if (r["src"], r["env"], r["step"], r["n"], r["delta"]) == (src, env, step, n, 0.05)}
            vals = [(g[("pooled Wilson", dr)]["excess"] / g[(arm, dr)]["excess"]) ** 2 for arm, dr in cols]
            for col, v in zip(cols, vals):
                med[col].append(v)
            L.append(f"| {env}{' (014)' if src == '014' else ''} | {n} | {g[('b1w', 'stored')]['truth']:.3f} | "
                     + " | ".join(f"{v:.2f}" for v in vals) + " |")
    L.append("| median of the ten | | | " + " | ".join(f"{np.median(med[col]):.2f}" for col in cols) + " |")
    open(path, "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5000, help="draws of the stored and paired passes (the baseline's 5,000)")
    ap.add_argument("--reps-large", type=int, default=40000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--control-only", action="store_true", help="redo the binomial control on the stored cells")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "replacement_check.json")
    if not a.report_only and not a.control_only:
        jobs = [("013", p, lab, s, a.reps, a.reps_large) for p, labs in RP.LABELS.items() for lab in labs for s in (0, 100, 200)]
        jobs += [("014", "C1", "refusal", s, a.reps, a.reps_large) for s in (100, 200)]
        t0 = time.time(); rows = []
        with ProcessPoolExecutor(a.workers) as ex:
            for i, r in enumerate(ex.map(job, jobs)):
                rows += r; print(f"{i + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
        old = json.load(open(os.path.join(OUT, "stratppi.json")))
        diff, hits = float("nan"), 0
        if old["reps"] == a.reps:
            mine = {(r["src"], r["env"], r["step"], r["n"], r["delta"], SHARED[r["arm"]]): r["miss"]
                    for r in rows if r["draw"] == "stored" and r["arm"] in SHARED}
            ds = [abs(mine[k] - r["miss"]) for r in old["rows"] if r["part"] == "A"
                  for k in [(r["src"], r["env"], r["step"], r["n"], r["delta"], r["arm"])] if k in mine]
            diff, hits = max(ds), len(ds)
            assert diff < 1e-12 and hits == len(mine), (diff, hits, len(mine))
        json.dump(dict(reps=a.reps, reps_large=a.reps_large, replay_diff=diff, replay_cells=hits, control=control(rows), rows=rows),
                  open(path, "w"))
    elif a.control_only:
        res = json.load(open(path))
        res["control"] = control(res["rows"])
        json.dump(res, open(path, "w"))
    report(json.load(open(path)), os.path.join(OUT, "replacement_check.md"))
    print("-> results/paper/replacement_check.md")


if __name__ == "__main__":
    main()
