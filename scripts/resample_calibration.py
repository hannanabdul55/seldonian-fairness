"""Bound calibration on real judge labels, by resampling stored episodes.

Pools the safety-set episodes of one method over seeds, rebuilds the harm labels
(adversarial prompts) and refusal labels (benign prompts) from the judge cache
without loading any model, treats the pooled label mean as the population truth,
and resamples safety sets of size n with replacement. For every bound and delta
it reports the empirical failure rate P(upper bound < truth), which the guarantee
says must be at most delta, the mean one-sided width, and the wall-clock per call.

Labels are 0/1, so a bound depends on the sample only through the number of ones;
each distinct count in the resamples is evaluated once and weighted by how often
it occurred. Resampling n labels with replacement from the pool is exactly a
Binomial(n, pool rate) draw, which is what this does.

Example
  uv run scripts/resample_calibration.py                       # reference + grpo pools
  uv run scripts/resample_calibration.py --methods reference --n 400 1200 --R 2000
"""
import argparse
import glob
import json
import os
import time

import numpy as np

from seldonian.llm.data import read_jsonl
from seldonian.llm.judges import build_judge
from seldonian.llm.policy import BOUNDS

CONSTRAINTS = {
    # name: (judge name, prompt group)
    "harm": ("qwen3guard", "adversarial"),
    "refusal": ("qwen3guard_refusal", "benign"),
}
DEFAULT_BOUNDS = ["ttest", "hoeffding", "clopper_pearson", "bentkus", "empirical_bernstein",
                  "chernoff_kl", "anderson", "betting_mixture"]


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pattern", default="results/llm/ab/{method}/seed*/episodes_s.jsonl",
                   help="glob for one method's episodes; {method} is substituted")
    p.add_argument("--methods", nargs="+", default=["reference", "grpo"])
    p.add_argument("--constraints", nargs="+", default=list(CONSTRAINTS),
                   choices=list(CONSTRAINTS))
    p.add_argument("--n", type=int, nargs="+", default=[200, 400, 800, 1200, 2400])
    p.add_argument("--deltas", type=float, nargs="+", default=[0.05, 0.1])
    p.add_argument("--bounds", nargs="+", default=DEFAULT_BOUNDS)
    p.add_argument("--R", type=int, default=5000, help="resamples per n")
    p.add_argument("--R-slow", type=int, default=None,
                   help="resamples for the slow bounds (betting_mixture); default = R")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache-dir", default=".cache/judges")
    p.add_argument("--out", default="results/calibration")
    return p.parse_args()


def cached_labels(judge, prompts, responses):
    """Labels from the judge cache only; ``None`` where the pair was never judged."""
    out = []
    for p, r in zip(prompts, responses):
        key = judge._key(p, r, None)
        out.append(judge._cache.get(key))
    return out


def load_pool(pattern, method, group, judge):
    files = sorted(glob.glob(pattern.format(method=method)))
    if not files:
        raise SystemExit(f"no episodes match {pattern.format(method=method)}")
    eps = [e for f in files for e in read_jsonl(f) if e.get("group") == group]
    labels = cached_labels(judge, [e["prompt"] for e in eps], [e["response"] for e in eps])
    kept = np.asarray([v for v in labels if v is not None], dtype=float)
    missing = len(labels) - kept.size
    return kept, missing, files


def calibrate(labels, n_list, deltas, bounds, R, R_slow, seed):
    truth = float(labels.mean())
    rng = np.random.default_rng(seed)
    rows = []
    for n in n_list:
        # resample with replacement; only the count of ones matters for 0/1 data
        counts_all = rng.choice(labels, size=(R, n), replace=True).sum(axis=1).astype(int)
        for bound in bounds:
            R_b = R_slow if (bound == "betting_mixture" and R_slow) else R
            counts = counts_all[:R_b]
            ks, freq = np.unique(counts, return_counts=True)
            fn = BOUNDS[bound]
            for delta in deltas:
                uppers = np.empty(ks.size)
                t0 = time.perf_counter()
                for j, k in enumerate(ks):
                    x = np.concatenate([np.ones(int(k)), np.zeros(n - int(k))])
                    uppers[j] = float(fn(x, delta).upper)
                per_call = (time.perf_counter() - t0) / ks.size
                fail = float(np.sum(freq * (uppers < truth)) / freq.sum())
                width = float(np.sum(freq * (uppers - ks / n)) / freq.sum())
                rows.append({"n": n, "bound": bound, "delta": delta, "fail_rate": fail,
                             "mean_width": width, "mean_upper": float(np.sum(freq * uppers) / freq.sum()),
                             "resamples": int(freq.sum()), "distinct_counts": int(ks.size),
                             "seconds_per_call": per_call})
                print(f"  n={n:5d} {bound:20s} delta={delta:.2f} fail={fail:.4f} "
                      f"width={width:.4f} ({per_call * 1e3:.1f} ms/call)", flush=True)
    return truth, rows


def markdown(rows, n_list, bounds, deltas, truth, size):
    out = []
    for delta in deltas:
        out.append(f"\ndelta = {delta}; population rate {truth:.4f} from {size} labels; "
                   "cells are failure rate / mean one-sided width\n")
        out.append("| bound | " + " | ".join(f"n={n}" for n in n_list) + " |")
        out.append("|---|" + "---|" * len(n_list))
        for b in bounds:
            cells = []
            for n in n_list:
                r = next(x for x in rows if x["n"] == n and x["bound"] == b and x["delta"] == delta)
                cells.append(f"{r['fail_rate']:.3f} / {r['mean_width']:.3f}")
            out.append(f"| {b} | " + " | ".join(cells) + " |")
    out.append("\nwall-clock per bound call (ms), largest n:")
    n_max = max(n_list)
    for b in bounds:
        r = next(x for x in rows if x["n"] == n_max and x["bound"] == b and x["delta"] == deltas[0])
        out.append(f"- {b}: {r['seconds_per_call'] * 1e3:.2f}")
    return "\n".join(out)


def main():
    args = parse()
    os.makedirs(args.out, exist_ok=True)
    for method in args.methods:
        for name in args.constraints:
            judge_name, group = CONSTRAINTS[name]
            # cache_only: the GPU belongs to the training queue; never load a judge model
            judge = build_judge(judge_name, cache_dir=args.cache_dir, cache_only=True)
            labels, missing, files = load_pool(args.pattern, method, group, judge)
            print(f"\n## {method} / {name} ({group} prompts, {judge.name})\n"
                  f"{labels.size} labels from {len(files)} files; {missing} episodes dropped "
                  "(label not in the cache)", flush=True)
            if labels.size < 2:
                print("  too few labels; skipped")
                continue
            t0 = time.time()
            truth, rows = calibrate(labels, args.n, args.deltas, args.bounds, args.R, args.R_slow,
                                    args.seed)
            result = {"method": method, "constraint": name, "group": group,
                      "judge": judge.name, "files": files, "labels": int(labels.size),
                      "dropped": int(missing), "population_rate": truth, "R": args.R,
                      "R_slow": args.R_slow, "seed": args.seed, "rows": rows,
                      "seconds": time.time() - t0}
            path = os.path.join(args.out, f"bounds_{method}_{name}.json")
            with open(path, "w") as f:
                json.dump(result, f, indent=2)
            print(markdown(rows, args.n, args.bounds, args.deltas, truth, labels.size))
            print(f"\nwrote {path} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
