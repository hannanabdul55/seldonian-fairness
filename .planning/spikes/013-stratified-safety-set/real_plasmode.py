"""Spike 013 stage 3: plasmode resampling on real Granite-3.3-2B generations.

Reads ``results/spikes/013/judged_full.jsonl`` (from ``gen013.py``). For every pool, label
and model state it builds a plasmode pool:

- covariate: the reference's ``cov`` labels (step 0, separate from every other sample)
- candidate: the state's ``cand`` labels, split into a truth half (per-prompt truth, and
  the pool truth as their mean) and an evaluation half (one label drawn per selected prompt
  per simulated safety set); ``--swap`` exchanges the halves (sensitivity check)
- meta: source (C1) or encoding (C2); C3 has none, so arm M is skipped there

Labels: C1 refusal; C2 non-refusal as the judge scores it (``unsafe``), the gated harm
label (sim >= 0.8 AND unsafe) and refusal; C3 unsafe and refusal (the pilot found plain-prompt
refusal the most heterogeneous label, ICC 0.91).

    ../../../.venv/bin/python real_plasmode.py --reps 5000 --out real_plasmode.json
"""
import argparse
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import plasmode as PM

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "..", "results", "spikes", "013")

LABELS = {"C1": ["refusal"], "C2": ["unsafe", "gated", "refusal"], "C3": ["unsafe", "refusal"]}


def label_of(row, lab):
    if lab == "gated":
        return [int(u and s >= 0.8) for u, s in zip(row["unsafe"], row["sim"])]
    return row[lab]


def load(tag="full"):
    rows = [json.loads(l) for l in open(os.path.join(SRC, f"judged_{tag}.jsonl"))]
    by = defaultdict(dict)
    meta = {}
    for r in rows:
        by[(r["pool"], r["step"], r["role"])][r["i"]] = r
        meta[(r["pool"], r["i"])] = r["meta"]
    return by, meta


def build_pool(by, meta, pool, lab, step, swap=False):
    cov_rows = by[(pool, 0, "cov")]
    cand_rows = by[(pool, step, "cand")]
    ids = sorted(set(cov_rows) & set(cand_rows))
    cov = np.array([label_of(cov_rows[i], lab) for i in ids], dtype=float)
    cand = np.array([label_of(cand_rows[i], lab) for i in ids], dtype=float)
    half = cand.shape[1] // 2
    truth_s, eval_s = (cand[:, half:], cand[:, :half]) if swap else (cand[:, :half], cand[:, half:])
    p_truth = truth_s.mean(axis=1)

    def draw(sel, rng, E=eval_s):
        return E[sel, rng.integers(0, E.shape[1], len(sel))]

    m = np.array([meta[(pool, i)] for i in ids])
    return dict(cov=cov, draw=draw, p_truth=p_truth, meta=m), truth_s


def cells_for(pool, N):
    n_s_list = tuple(n for n in (100, 200, 400) if n <= 0.4 * N)
    cells = PM.standard_cells(n_s_list=n_s_list)
    if pool == "C3":
        cells = [c for c in cells if c["arm"] != "M"]
    return cells


def job(args):
    pool, lab, step, reps, swap = args
    t0 = time.time()
    by, meta = load()
    P, truth_s = build_pool(by, meta, pool, lab, step, swap)
    N = len(P["p_truth"])
    rows = PM.run_cells(P, cells_for(pool, N), reps, seed=step)
    out = []
    for r in rows:
        pf = PM.preflight(P["cov"], P["p_truth"], truth_s, k=r["k"], H=r["H"])
        out.append(dict(r, env=f"{pool}:{lab}", pool=pool, label=lab, cand=f"step{step}", swap=swap, N=N,
                        rate=float(P["p_truth"].mean()), **{f"pf_{k}": v for k, v in pf.items()}))
    return out, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5000)
    ap.add_argument("--steps", default="0,100,200")
    ap.add_argument("--swap", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    jobs = [(p, lab, int(s), a.reps, a.swap) for p, labs in LABELS.items() for lab in labs
            for s in a.steps.split(",")]
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for r, dt in ex.map(job, jobs):
            rows.extend(r)
    json.dump(rows, open(a.out, "w"))
    print(f"{len(jobs)} pool/label/state jobs, {len(rows)} rows in {time.time() - t0:.0f} s -> {a.out}")


if __name__ == "__main__":
    main()
