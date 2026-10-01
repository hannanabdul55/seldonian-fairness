"""Spike 014 stage 3: 013's real-data plasmode on the pushed-label checkpoints.

Covariate: 013's 8 reference samples per C1 prompt (step 0, ``judged_full.jsonl``).
Candidates: 014's checkpoints at steps 100 and 200 (``judged_s0.jsonl``), K samples per
prompt split into a truth half and an evaluation half, exactly as ``013/real_plasmode.py``.
Also 013's own side-effect checkpoints at the same steps for the like-for-like comparison.
Random tie-breaking, H 8, k 8: 013's final rule.

    ../../../.venv/bin/python plasmode014.py --reps 5000 --out plasmode.json
    ../../../.venv/bin/python ../013-stratified-safety-set/summarise_plasmode.py plasmode.json --key env
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "013-stratified-safety-set"))
import plasmode as PM  # noqa: E402
import real_plasmode as RP  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SRC013 = os.path.join(REPO, "results", "spikes", "013", "judged_full.jsonl")
SRC014 = os.path.join(REPO, "results", "spikes", "014")


def load(tag):
    by, meta = defaultdict(dict), {}
    for r in map(json.loads, open(SRC013)):
        if r["pool"] == "C1" and r["step"] == 0 and r["role"] == "cov":
            by[("C1", 0, "cov")][r["i"]] = r
            meta[("C1", r["i"])] = r["meta"]
        elif r["pool"] == "C1" and r["role"] == "cand":
            by[("C1", r["step"], "side")][r["i"]] = r          # 013's side-effect checkpoints
    for r in map(json.loads, open(os.path.join(SRC014, f"judged_{tag}.jsonl"))):
        by[("C1", r["step"], "cand")][r["i"]] = r
    return by, meta


def job(args):
    tag, role, step, reps, swap = args
    PM.TIES = "random"
    t0 = time.time()
    by, meta = load(tag)
    if role == "side":
        by[("C1", step, "cand")] = by[("C1", step, "side")]
    P, truth_s = RP.build_pool(by, meta, "C1", "refusal", step, swap, "eval")
    N = len(P["p_truth"])
    rows = PM.run_cells(P, RP.cells_for("C1", N), reps, seed=step)
    out = []
    for r in rows:
        pf = PM.preflight(P["cov"], P["p_truth"], truth_s, k=r["k"], H=r["H"])
        out.append(dict(r, env=f"C1:refusal:{role}", pool="C1", label="refusal", role=role,
                        cand=f"step{step}", swap=swap, N=N, rate=float(P["p_truth"].mean()),
                        **{f"pf_{k}": v for k, v in pf.items()}))
    return out, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5000)
    ap.add_argument("--tag", default="s0")
    ap.add_argument("--steps", default="100,200")
    ap.add_argument("--swap", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    jobs = [(a.tag, role, int(s), a.reps, a.swap) for role in ("cand", "side") for s in a.steps.split(",")]
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for r, dt in ex.map(job, jobs):
            rows.extend(r)
    json.dump(rows, open(os.path.join(HERE, a.out), "w"))
    print(f"{len(jobs)} jobs, {len(rows)} rows in {time.time() - t0:.0f} s -> {a.out}")


if __name__ == "__main__":
    main()
