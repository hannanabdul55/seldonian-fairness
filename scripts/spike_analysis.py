"""Training-signal spikes versus predicted-test outcomes, from stored runs (no GPU).

For every Seldonian Lagrangian run that has a predicted-test ``history`` (a
checkpoint every 30 steps with the predicted constraint value ``g``, feasibility
and the multipliers) and a trainer log (per-5-step reward, reward std, KL, grad
norm, length, clip ratio; from ``result.json["train_log"]`` when present, else
parsed from the run's console log), this script cuts the training trajectory into
the intervals between consecutive predicted tests, summarises the trainer
signals in each interval, and asks whether those summaries separate the intervals
that ended infeasible from the ones that stayed feasible, and whether they track
the size of the move in the constraint.

This is the cheap first step of the "aha / TD-error spike" idea in
``reports/ideas.md``: GRPO has no critic, so the spike candidates are the
between-step jumps in mean reward (surprise), the within-group reward spread, the
KL to the reference, and the gradient norm.

    uv run scripts/spike_analysis.py [--out results/spike]
"""
import argparse
import ast
import csv
import glob
import json
import os
import re

import numpy as np

SIGNALS = {  # column in the trainer log -> short name
    "reward": "reward", "reward_std": "reward_std", "kl": "kl", "grad_norm": "grad_norm",
    "loss": "loss", "completions/mean_length": "length", "completions/clipped_ratio": "clip",
}
DICT_LINE = re.compile(r"\{'loss'.*?\}")


def parse_console_log(path, n_steps):
    """Trainer dicts printed by TRL, one per logging step; step assigned by position."""
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, errors="replace") as f:
        for line in f:
            for m in DICT_LINE.finditer(line):
                try:
                    d = ast.literal_eval(m.group(0))
                except (ValueError, SyntaxError):
                    continue
                if "epoch" in d and "train_runtime" in d:
                    continue
                rows.append({k: float(v) for k, v in d.items()
                             if k in SIGNALS and _isnum(v)})
    if rows:
        every = n_steps / len(rows)
        for i, r in enumerate(rows):
            r["step"] = (i + 1) * every
    return rows


def _isnum(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def load_runs(patterns):
    runs = []
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            d = json.load(open(f))
            hist = d.get("history") or []
            if len(hist) < 2:
                continue
            out_dir = f.split("/" + d["task"] + "/")[0]
            log = os.path.join(out_dir, "logs", f"{d['task']}_{d['method']}_seed{d['seed']}.log")
            rows = parse_console_log(log, d["config"]["steps"])
            tl = d.get("train_log") or []
            if tl and not rows:
                rows = [{SIGNALS.get(k, k): v for k, v in r.items() if k in SIGNALS or k == "step"}
                        for r in tl if "step" in r]
            elif tl and rows and "grad_norm" not in rows[0]:
                pass
            if not rows:
                continue
            for r in rows:  # normalise column names
                for k, short in SIGNALS.items():
                    if k in r and short not in r:
                        r[short] = r.pop(k)
            runs.append({"file": f, "name": out_dir.replace("results/", "") + f"/s{d['seed']}",
                         "task": d["task"], "seed": d["seed"], "history": hist, "rows": rows,
                         "thresholds": d["thresholds"], "selected": d.get("selected") or {},
                         "safety_test": d.get("safety_test") or {}, "config": d["config"]})
    return runs


def interval_features(rows, lo, hi, run_scale):
    """Summaries of the trainer signals over steps in (lo, hi]."""
    seg = [r for r in rows if lo < r["step"] <= hi]
    if len(seg) < 2:
        return None
    feat = {"n_log": len(seg)}
    for s in ("reward", "reward_std", "kl", "grad_norm", "length", "clip"):
        vals = np.array([r[s] for r in seg if s in r], dtype=float)
        if vals.size < 2:
            continue
        diffs = np.diff(vals)
        feat[f"{s}_mean"] = vals.mean()
        feat[f"{s}_max"] = vals.max()
        feat[f"{s}_slope"] = vals[-1] - vals[0]
        feat[f"{s}_jump"] = np.abs(diffs).max()
        sc = run_scale.get(s)
        if sc:
            feat[f"{s}_jump_z"] = np.abs(diffs).max() / sc  # spike: largest jump in run-sd units
    return feat


def run_scales(rows):
    """Per-run sd of the between-log differences, the unit for the spike z-scores."""
    out = {}
    for s in ("reward", "reward_std", "kl", "grad_norm", "length", "clip"):
        vals = np.array([r[s] for r in rows if s in r], dtype=float)
        if vals.size > 3:
            sd = np.diff(vals).std()
            out[s] = sd if sd > 0 else None
    return out


def worst_g(h):
    return max(v for v in h["g"].values() if np.isfinite(v))


def binding(h, thresholds):
    """Constraint with the largest predicted g at this checkpoint."""
    return max(h["g"], key=lambda k: h["g"][k])


def auc(pos, neg):
    """P(feature of an infeasible interval > feature of a feasible one), ties half."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if pos.size == 0 or neg.size == 0:
        return np.nan
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return gt + 0.5 * eq


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        return np.nan
    rx = np.argsort(np.argsort(x[ok]))
    ry = np.argsort(np.argsort(y[ok]))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--patterns", nargs="+",
                   default=["results/llm_r4/*/ab/seldonian_lag/seed*/result.json",
                            "results/llm_r5/*/brevity/seldonian_lag/seed*/result.json",
                            "results/llm_r6/*/*/seldonian_lag/seed*/result.json"])
    p.add_argument("--out", default="results/spike")
    p.add_argument("--task", default=None, help="restrict to one task (ab, brevity)")
    p.add_argument("--tag", default="all", help="suffix for the output files")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    runs = [r for r in load_runs(args.patterns) if args.task is None or r["task"] == args.task]
    print(f"{len(runs)} runs with a history and a trainer log")

    intervals = []
    for run in runs:
        scale = run_scales(run["rows"])
        prev_step, prev_g, prev_rate = 0, None, None
        for k, h in enumerate(run["history"]):
            feat = interval_features(run["rows"], prev_step, h["step"], scale)
            name = binding(h, run["thresholds"])
            rate = h["rates"][name]
            row = {"run": run["name"], "task": run["task"], "seed": run["seed"], "k": k + 1,
                   "step": h["step"], "binding": name, "feasible": int(h["feasible"]),
                   "g": worst_g(h), "rate": rate,
                   "d_rate": (rate - prev_rate) if prev_rate is not None else np.nan,
                   "d_g": (worst_g(h) - prev_g) if prev_g is not None else np.nan,
                   "lam_before": (run["history"][k - 1]["lambdas"].get(name, np.nan) if k else
                                  run["config"].get("lam0", np.nan)),
                   "pred_reward": h["reward"]}
            if feat:
                row.update(feat)
            intervals.append(row)
            prev_step, prev_g, prev_rate = h["step"], worst_g(h), rate

    cols = sorted({k for r in intervals for k in r}, key=lambda c: (c not in (
        "run", "task", "seed", "k", "step", "binding", "feasible", "g", "rate", "d_rate", "d_g",
        "lam_before", "pred_reward"), c))
    with open(os.path.join(args.out, f"intervals_{args.tag}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in intervals:
            w.writerow({c: r.get(c, "") for c in cols})

    feats = [c for c in cols if c.endswith(("_mean", "_max", "_slope", "_jump", "_jump_z"))]
    infeasible = [r for r in intervals if not r["feasible"]]
    feasible = [r for r in intervals if r["feasible"]]
    lines = [f"# Training-signal spikes vs predicted-test outcomes ({args.tag})\n",
             f"{len(runs)} Seldonian Lagrangian runs, {len(intervals)} intervals between predicted "
             f"tests ({len(infeasible)} ended infeasible, {len(feasible)} feasible). Per interval the "
             "trainer signals (every 5 steps) are summarised; `_jump` is the largest absolute "
             "change between consecutive logs, `_jump_z` the same in units of the run's sd of "
             "changes (a spike score). AUC is P(infeasible interval > feasible interval); rho is "
             "Spearman with the predicted constraint value g at the end of the interval and with "
             "its change from the previous checkpoint.\n",
             "Runs: " + ", ".join(r["name"] for r in runs) + "\n",
             "| feature | mean, infeasible | mean, feasible | AUC | rho with g | rho with d_g | rho with d_rate |",
             "|---|---|---|---|---|---|---|"]
    stats = []
    for fname in feats:
        pos = [r[fname] for r in infeasible if fname in r]
        neg = [r[fname] for r in feasible if fname in r]
        allv = [r.get(fname, np.nan) for r in intervals]
        a = auc(pos, neg)
        stats.append((fname, np.mean(pos) if pos else np.nan, np.mean(neg) if neg else np.nan, a,
                      spearman(allv, [r["g"] for r in intervals]),
                      spearman(allv, [r["d_g"] for r in intervals]),
                      spearman(allv, [r["d_rate"] for r in intervals])))
    stats.sort(key=lambda s: -abs(s[3] - 0.5) if np.isfinite(s[3]) else 0)
    for s in stats:
        lines.append(f"| {s[0]} | {s[1]:.3g} | {s[2]:.3g} | {s[3]:.2f} | {s[4]:+.2f} | {s[5]:+.2f} | {s[6]:+.2f} |")

    # the multiplier before the interval, as a control (the obvious predictor)
    lam_pos = [r["lam_before"] for r in infeasible]
    lam_neg = [r["lam_before"] for r in feasible]
    lines.append(f"\nControl, multiplier at the start of the interval: mean {np.mean(lam_pos):.1f} "
                 f"(infeasible) vs {np.mean(lam_neg):.1f} (feasible), AUC {auc(lam_pos, lam_neg):.2f}.")
    # first interval only (the multiplier is the same for all runs there)
    first = [r for r in intervals if r["k"] == 1]
    lines.append(f"First intervals only (k = 1, {len(first)} runs, multiplier lam0 for all): "
                 f"infeasible {sum(1 for r in first if not r['feasible'])}.")
    for fname in ("reward_jump_z", "kl_max", "grad_norm_max", "reward_std_mean", "length_slope"):
        pos = [r[fname] for r in first if not r["feasible"] and fname in r]
        neg = [r[fname] for r in first if r["feasible"] and fname in r]
        if pos and neg:
            lines.append(f"- {fname}: {np.mean(pos):.3g} vs {np.mean(neg):.3g}, AUC {auc(pos, neg):.2f}")

    # within-run: does the interval with the biggest reward spike coincide with the biggest move?
    hits, total = 0, 0
    for run in runs:
        rs = [r for r in intervals if r["run"] == run["name"] and "reward_jump_z" in r and np.isfinite(r["d_rate"])]
        if len(rs) < 3:
            continue
        spike = max(rs, key=lambda r: r["reward_jump_z"])
        move = max(rs, key=lambda r: abs(r["d_rate"]))
        hits += spike["k"] == move["k"]
        total += 1
    lines.append(f"\nWithin-run: the interval with the largest reward-jump z is the interval with the "
                 f"largest |change in the binding rate| in {hits} of {total} runs (chance about "
                 f"{1 / 4:.2f}).")

    # selected checkpoint: predicted vs safety-set gap against run-level spikiness
    lines.append("\n| run | selected step | predicted rate | safety-set rate | gap | run max reward_jump_z | run max kl | run max grad_norm |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for run in runs:
        sel = run["selected"].get("step")
        st = run["safety_test"]
        if not st or sel is None:
            continue
        h = next((h for h in run["history"] if h["step"] == sel), run["history"][-1])
        name = binding(h, run["thresholds"])
        pred, meas = h["rates"][name], st["rates"].get(name, np.nan)
        rs = [r for r in intervals if r["run"] == run["name"]]
        mx = lambda key: max((r[key] for r in rs if key in r), default=np.nan)
        lines.append(f"| {run['name']} | {sel} | {pred:.3f} | {meas:.3f} | {meas - pred:+.3f} | "
                     f"{mx('reward_jump_z'):.2f} | {mx('kl_max'):.3f} | {mx('grad_norm_max'):.2f} |")

    with open(os.path.join(args.out, f"spike_analysis_{args.tag}.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
