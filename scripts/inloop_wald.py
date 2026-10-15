"""O2: the recommended safety-set design inside the training loop (synthetic bandit).

Registered in `.planning/paper-certification/O2_registration.md` before any run. Spike 014's
`bandit014.run` (the real `SeldonianLLMPolicy`, Lagrangian GRPO on `HeteroEnv`) at spike 013's
in-loop cell (pressure 1, eta 100, margin +0.03, 200 steps), with 8 equal rank strata of an
8-sample reference rate (`strat_ref`) against a random split, fresh seeds, R runs a cell.
Every bound is scored at delta 0.05 and 0.10 on the same safety labels; the policy's own
test (and so the candidate) is at delta 0.10.

    .venv/bin/python scripts/inloop_wald.py run --reps 8000 --workers 14
    .venv/bin/python scripts/inloop_wald.py report
"""
import argparse
import json
import lzma
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, ".planning", "spikes", "014-pushed-label-stratification"))
OUT = os.path.join(ROOT, "results", "paper", "inloop_wald")
RUNS = os.path.join(OUT, "runs.jsonl.xz")

ENVS = ("icc05", "icc26", "icc55", "icc75")
RULES = ("strat_ref", "random")
DELTAS = (0.05, 0.1)
SEED0 = 100000                      # spike 013 used seeds 0-499, spike 014 0-299
Z = 2.0                             # the paper's cell rule (scripts/validity_recount.py)
KEEP = ("seed", "env", "rule", "truth_pool", "truth_pop", "rho", "icc_ref", "icc_cand",
        "solution", "n_s", "N", "seconds")


def _job(a):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import bandit014 as B
    r = B.run(seed=a["seed"], env=a["env"], rule=a["rule"], pressure=1.0, eta=100.0,
              margin=0.03, steps=200, method="seldonian_lag")
    return {k: v for k, v in r.items() if k in KEEP or "@" in k}


def run(a):
    os.makedirs(OUT, exist_ok=True)
    done = set()
    if os.path.exists(RUNS):
        with lzma.open(RUNS, "rt") as f:
            done = {(d["env"], d["rule"], d["seed"]) for d in map(json.loads, f)}
    jobs = [dict(env=e, rule=r, seed=SEED0 + i) for e in ENVS for r in RULES for i in range(a.reps)
            if (e, r, SEED0 + i) not in done]
    print(f"{len(jobs)} runs to do ({len(done)} done)", flush=True)
    with ProcessPoolExecutor(a.workers) as ex, lzma.open(RUNS, "at") as f:
        for k, row in enumerate(ex.map(_job, jobs, chunksize=16)):
            f.write(json.dumps(row) + "\n")
            if k % 2000 == 0:
                print(k, flush=True)


def classify(m, delta, R):
    se = np.sqrt(delta * (1 - delta) / R)
    return "over" if m > delta + Z * se + 1e-12 else ("unresolved" if m > delta + 1e-12 else "under")


def report(a):
    with lzma.open(RUNS, "rt") as f:
        rows = [json.loads(l) for l in f]
    cells, ess = [], {}
    for e in ENVS:
        S = [r for r in rows if r["env"] == e and r["rule"] == "strat_ref"]
        Rn = [r for r in rows if r["env"] == e and r["rule"] == "random"]
        for d in DELTAS:
            for name, src, key, truth in (
                    ("Wald-t, 8 strata, pool rate", S, "b1_strat_pool", "truth_pool"),
                    ("Wald-t, 8 strata, sampled-pool term, source rate", S, "b1_strat_pop", "truth_pop"),
                    ("b1w, 8 strata, pool rate", S, "b1w_strat_pool", "truth_pool"),
                    ("b1w, 8 strata, sampled-pool term, source rate", S, "b1w_strat_pop", "truth_pop"),
                    ("pooled Wilson, random split, source rate", Rn, "b1w_pooled", "truth_pop"),
                    ("pooled Wilson, random split, pool rate", Rn, "b1w_pooled", "truth_pool"),
                    ("t-test, random split, source rate", Rn, "ttest", "truth_pop"),
                    ("t-test, 8 strata, source rate", S, "ttest", "truth_pop")):
                k = sum(r[truth] > r[f"{key}@{d}"] for r in src)
                R = len(src)
                cells.append(dict(env=e, delta=d, bound=name, R=R, k=k, miss=k / R,
                                  cls=classify(k / R, d, R),
                                  width=float(np.mean([r[f"{key}@{d}"] - r["est_strat@%s" % d]
                                                       if "strat" in key else
                                                       r[f"{key}@{d}"] - r["est_pool@%s" % d]
                                                       for r in src]))))
            w = {c["bound"]: c["width"] for c in cells if c["env"] == e and c["delta"] == d}
            base = w["pooled Wilson, random split, source rate"]
            ess[(e, d)] = dict(pop=(base / w["Wald-t, 8 strata, sampled-pool term, source rate"]) ** 2,
                               pool=(base / w["Wald-t, 8 strata, pool rate"]) ** 2,
                               rho=float(np.mean([r["rho"] for r in S])))

    def n_over(bound):
        return sum(c["cls"] == "over" for c in cells if c["bound"] == bound)

    def cl(bound, e, d):
        return next(c for c in cells if c["bound"] == bound and c["env"] == e and c["delta"] == d)

    pred = [
        ("Q1", "Wald-t, 8 strata, pool rate: over in at most 1 of 8 cells",
         n_over("Wald-t, 8 strata, pool rate") <= 1),
        ("Q2", "Wald-t, 8 strata, sampled-pool term, source rate: over in at most 2 of 8 cells",
         n_over("Wald-t, 8 strata, sampled-pool term, source rate") <= 2),
        ("Q3", "the same at icc05, delta 0.10: over (the 4-strata in-loop cell was over)",
         cl("Wald-t, 8 strata, sampled-pool term, source rate", "icc05", 0.1)["cls"] == "over"),
        ("Q4", "pooled Wilson on a random split, source rate: over in at most 2 of 8 cells",
         n_over("pooled Wilson, random split, source rate") <= 2),
        ("Q5", "gain of the Wald-t source-rate limit against the pooled Wilson bound, delta 0.10: "
               "at least 1.6 at icc75",
         ess[("icc75", 0.1)]["pop"] >= 1.6),
        ("Q6", "the same gain at icc05 between 0.9 and 1.15",
         0.9 <= ess[("icc05", 0.1)]["pop"] <= 1.15),
    ]
    L = ["# O2: the recommended design inside the training loop", "",
         f"Registered in `.planning/paper-certification/O2_registration.md`. {len(rows)} runs; "
         f"cell rule: over when the miss exceeds delta by more than {Z:g} Monte Carlo SE.", "",
         "## Predictions", ""]
    L += [f"- **{q}** {t}: **{'kept' if ok else 'refuted'}**" for q, t, ok in pred]
    L += ["", "## Cells", "", "| env | delta | bound | R | misses | miss | cell | mean width |",
          "|---|---|---|---|---|---|---|---|"]
    L += [f"| {c['env']} | {c['delta']} | {c['bound']} | {c['R']} | {c['k']} | {c['miss']:.4f} | "
          f"{c['cls']} | {c['width']:.4f} |" for c in cells]
    L += ["", "## Gain (ESS against the pooled Wilson bound on a random split, by mean width)", "",
          "| env | delta | rho | source rate | pool rate |", "|---|---|---|---|---|"]
    L += [f"| {e} | {d} | {v['rho']:.2f} | {v['pop']:.2f} | {v['pool']:.2f} |" for (e, d), v in ess.items()]
    open(os.path.join(OUT, "report.md"), "w").write("\n".join(L) + "\n")
    json.dump(dict(cells=cells, ess=[dict(env=e, delta=d, **v) for (e, d), v in ess.items()],
                   predictions=[dict(id=q, text=t, kept=bool(ok)) for q, t, ok in pred]),
              open(os.path.join(OUT, "report.json"), "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "report"])
    ap.add_argument("--reps", type=int, default=8000)
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()
    {"run": run, "report": report}[a.stage](a)
