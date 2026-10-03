"""Spike 020, validity check of the bounds under the real cluster structure (post hoc, exploratory).

For a pipeline, the 97 observed user-task clusters are taken as the population (truth = the
observed pair rate). Each rep draws 97 clusters with replacement, computes each upper bound at
delta 0.05, and records a miss (bound < truth). The naive bound ignores clustering; the
cluster-t bound respects it; the two-way basic bootstrap is the exploratory one.

    ../../../.venv/bin/python plasmode020.py --reps 400
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cert020 as C  # noqa: E402

PIPES = ("Meta-SecAlign-70B", "claude-3-5-sonnet-20241022", "claude-3-7-sonnet-20250219",
         "gpt-4o-2024-05-13-tool_filter", "gemini-2.0-flash-001", "gpt-4o-2024-05-13")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--boots", type=int, default=1000)
    a = ap.parse_args()
    C.BOOTS = a.boots
    rows = [json.loads(l) for l in open(os.path.join(C.OUT, "episodes.jsonl"))]
    rng = np.random.default_rng(2020)
    L = ["| pipeline | truth | miss naive | miss Wilson | miss t(user) | miss t(inj) | miss two-way | median t(user) | median naive |",
         "|---|---|---|---|---|---|---|---|---|"]
    out = {}
    for p in PIPES:
        R = [r for r in rows if r["pipeline"] == p and r["y"] is not None]
        by = collections.defaultdict(list)
        for r in R:
            by[r["user_task_key"]].append(r)
        clusters = list(by.values())
        truth = np.mean([r["y"] for r in R])
        miss = collections.Counter(); vals = collections.defaultdict(list)
        for rep in range(a.reps):
            idx = rng.integers(0, len(clusters), len(clusters))
            S = []
            for j, i in enumerate(idx):
                for r in clusters[i]:
                    S.append(dict(r, user_task_key=f"{r['user_task_key']}#{j}"))   # a re-drawn cluster is a new cluster
            n = len(S); k = sum(r["y"] for r in S)
            b = dict(naive=C.cp_upper(k, n, 0.05),
                     wilson=C.SB.b1w(np.array([k]), np.array([n]), np.array([1.0]), 0.05),
                     t_user=C.cluster_t(S, "user_task_key", 0.05, rng)[0],
                     t_inj=C.cluster_t(S, "injection_task_key", 0.05, rng)[0],
                     twoway=C.twoway_t(S, 0.05, rng))
            for kk, v in b.items():
                miss[kk] += v < truth; vals[kk].append(v)
        out[p] = dict(truth=truth, **{f"miss_{k}": miss[k] / a.reps for k in miss},
                      **{f"med_{k}": float(np.median(v)) for k, v in vals.items()})
        L.append(f"| {p} | {truth:.3f} | {miss['naive'] / a.reps:.3f} | {miss['wilson'] / a.reps:.3f} | {miss['t_user'] / a.reps:.3f} "
                 f"| {miss['t_inj'] / a.reps:.3f} | {miss['twoway'] / a.reps:.3f} | {np.median(vals['t_user']):.3f} | {np.median(vals['naive']):.3f} |")
        print(L[-1], flush=True)
    json.dump(out, open(os.path.join(HERE, "plasmode.json"), "w"), indent=1)
    open(os.path.join(HERE, "plasmode.md"), "w").write(
        f"# Validity under cluster resampling ({a.reps} reps, delta 0.05; miss should be <= 0.05)\n\n" + "\n".join(L) + "\n")


if __name__ == "__main__":
    main()
