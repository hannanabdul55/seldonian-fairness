"""Spike profile of a training run: how many jumps, and how late (no GPU).

Version 1 of the "cumulative TD error as potential for harm" idea in
``reports/ideas.md``. For every stored run with a trainer log (GRPO, composite and
Seldonian arms of Rounds 4-6; the log is 30 points at steps 5, 10, ..., 150), and
for each of the signals reward, reward spread, KL and gradient norm:

- the between-log jumps ``d_t = x_t - x_{t-1}`` are scaled by the run's own robust
  spread of jumps (median absolute deviation times 1.4826), so a *spike* is a jump
  more than 2 robust sd from the run's typical jump;
- spikes are counted per 30-step interval (five intervals), giving a profile
  ``c_1..c_5``; the cumulative absolute z-score per interval is the "cumulative TD
  error" analogue;
- the profile is summarised by its total, its least-squares slope over the five
  intervals, and its *late share* ``(c_4 + c_5) / total`` (0.4 under a flat profile).

Runs are grouped by task, method and the multiplier setting, and the summaries are
compared across groups and against each run's outcome (solution, feasible
checkpoints, breach). The hypothesis under test: runs whose spikes come late are
still moving, and their safety-test outcome is the least stable.

    uv run scripts/spike_profile.py [--out results/spike] [--z 2.0]
"""
import argparse
import csv
import glob
import json
import os

import numpy as np

from spike_analysis import SIGNALS, parse_console_log  # noqa: E402  (same directory)

PROFILE_SIGNALS = ("reward", "reward_std", "kl", "grad_norm")


def load_all(patterns):
    runs = []
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            d = json.load(open(f))
            if "method" not in d or "config" not in d or d["method"] == "reference":
                continue
            out_dir = f.split("/" + d["task"] + "/")[0]
            log = os.path.join(out_dir, "logs", f"{d['task']}_{d['method']}_seed{d['seed']}.log")
            rows = parse_console_log(log, d["config"]["steps"])
            if len(rows) < 10:
                continue
            for r in rows:
                for k, short in SIGNALS.items():
                    if k in r and short not in r:
                        r[short] = r.pop(k)
            cfg = d["config"]
            st = d.get("safety_test") or d.get("eval_s") or {}
            thr = d.get("thresholds", {})
            rates = st.get("rates", {})
            breach = any(rates.get(k, 0) > v for k, v in thr.items())
            hist = d.get("history") or []
            setting = ""
            if d["method"] == "seldonian_lag":
                setting = ("floor-always" if cfg.get("lam_floor_always") else
                           "floor-armed" if cfg.get("lam_floor", 0) > 0 else "no-floor")
            elif d["method"] == "composite":
                setting = f"lam{cfg.get('lam', [None])[0]}"
            pressure = cfg.get("long_bonus") or cfg.get("compliance_bonus") or cfg.get("bias_bonus") or 0
            runs.append({
                "name": out_dir.replace("results/", "") + f"/{d['method']}/s{d['seed']}",
                "task": d["task"], "method": d["method"], "setting": setting,
                "pressure": pressure, "seed": d["seed"], "rows": rows, "steps": cfg["steps"],
                "solution": bool(d.get("solution_found")), "breach": bool(breach),
                "feasible": sum(h["feasible"] for h in hist) if hist else None,
                "n_pred": len(hist), "reward": st.get("reward"),
                "selected": (d.get("selected") or {}).get("step"),
            })
    return runs


def profile(rows, steps, z_thr, n_int=5):
    """Spike counts and cumulative |z| per interval, per signal and pooled."""
    edges = np.linspace(0, steps, n_int + 1)
    out = {}
    pooled_counts = np.zeros(n_int)
    pooled_cum = np.zeros(n_int)
    for s in PROFILE_SIGNALS:
        pts = [(r["step"], r[s]) for r in rows if s in r]
        if len(pts) < 6:
            continue
        st = np.array([p[0] for p in pts][1:])
        d = np.diff([p[1] for p in pts])
        mad = np.median(np.abs(d - np.median(d))) * 1.4826
        if mad <= 0:
            mad = d.std() or 1.0
        z = np.abs(d) / mad
        counts = np.zeros(n_int)
        cum = np.zeros(n_int)
        for step, zz in zip(st, z):
            k = min(int(np.searchsorted(edges, step, side="left")) - 1, n_int - 1)
            k = max(k, 0)
            counts[k] += zz > z_thr
            cum[k] += zz
        out[s] = {"counts": counts, "cum": cum, "z": z, "steps": st}
        pooled_counts += counts
        pooled_cum += cum
    out["pooled"] = {"counts": pooled_counts, "cum": pooled_cum}
    return out


def summarise(counts):
    total = counts.sum()
    x = np.arange(len(counts))
    slope = float(np.polyfit(x, counts, 1)[0]) if total > 0 else 0.0
    late = float(counts[-2:].sum() / total) if total > 0 else np.nan
    return total, slope, late


def group_key(r):
    if r["method"] == "seldonian_lag":
        return f"{r['task']} seldonian {r['setting']}"
    return f"{r['task']} {r['method']}"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--patterns", nargs="+",
                   default=["results/llm_r4/*/ab/*/seed*/result.json",
                            "results/llm_r5/*/brevity/*/seed*/result.json",
                            "results/llm_r6/*/*/*/seed*/result.json"])
    p.add_argument("--out", default="results/spike")
    p.add_argument("--z", type=float, default=2.0, help="spike threshold in robust sd units")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    runs = load_all(args.patterns)
    rows_out = []
    for r in runs:
        pr = profile(r["rows"], r["steps"], args.z)
        total, slope, late = summarise(pr["pooled"]["counts"])
        cum_total, cum_slope, cum_late = summarise(pr["pooled"]["cum"])
        rec = {"run": r["name"], "group": group_key(r), "task": r["task"], "method": r["method"],
               "setting": r["setting"], "pressure": r["pressure"], "seed": r["seed"],
               "solution": int(r["solution"]), "breach": int(r["breach"]),
               "feasible": r["feasible"], "n_pred": r["n_pred"], "selected": r["selected"],
               "reward": r["reward"], "spikes": int(total), "spike_slope": slope,
               "late_share": late, "cum_z": cum_total, "cum_z_slope": cum_slope,
               "cum_z_late": cum_late,
               "profile": "/".join(str(int(c)) for c in pr["pooled"]["counts"])}
        for s in PROFILE_SIGNALS:
            if s in pr:
                t, sl, la = summarise(pr[s]["counts"])
                rec[f"{s}_spikes"] = int(t)
                rec[f"{s}_late"] = la
        rows_out.append(rec)

    cols = list(rows_out[0].keys())
    with open(os.path.join(args.out, "profile_runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows_out)

    lines = [f"# Spike profiles: how many jumps, and how late (z > {args.z} robust sd)\n",
             f"{len(rows_out)} runs with a trainer log (30 points each). Spikes are pooled over "
             "reward, reward spread, KL and gradient norm; `profile` is the count per 30-step "
             "interval; `late share` is the fraction of spikes in the last two intervals (0.4 if "
             "flat); `cum |z|` is the cumulative absolute jump z-score, the cumulative-TD-error "
             "analogue, with its own late share.\n",
             "## By group\n",
             "| group | runs | spikes / run | late share | spike slope | cum z late share | solutions | breaches |",
             "|---|---|---|---|---|---|---|---|"]
    groups = sorted({r["group"] for r in rows_out})
    for g in groups:
        rs = [r for r in rows_out if r["group"] == g]
        ls = [r["late_share"] for r in rs if np.isfinite(r["late_share"])]
        cl = [r["cum_z_late"] for r in rs if np.isfinite(r["cum_z_late"])]
        lines.append(f"| {g} | {len(rs)} | {np.mean([r['spikes'] for r in rs]):.1f} | "
                     f"{np.mean(ls) if ls else float('nan'):.2f} | "
                     f"{np.mean([r['spike_slope'] for r in rs]):+.2f} | "
                     f"{np.mean(cl) if cl else float('nan'):.2f} | "
                     f"{sum(r['solution'] for r in rs if r['method'] == 'seldonian_lag')}/"
                     f"{sum(1 for r in rs if r['method'] == 'seldonian_lag')} | "
                     f"{sum(r['breach'] for r in rs)}/{len(rs)} |")

    lines += ["\n## Per run, sorted by late share\n",
              "| run | pressure | outcome | feasible | spikes | profile | late share | cum z late |",
              "|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows_out, key=lambda r: -(r["late_share"] if np.isfinite(r["late_share"]) else -1)):
        outcome = ("solution" if r["solution"] else "NSF") if r["method"] == "seldonian_lag" else \
                  ("breach" if r["breach"] else "within")
        feas = f"{r['feasible']}/{r['n_pred']}" if r["n_pred"] else "-"
        lines.append(f"| {r['run']} | {r['pressure']} | {outcome} | {feas} | {r['spikes']} | "
                     f"{r['profile']} | {r['late_share']:.2f} | {r['cum_z_late']:.2f} |")

    # outcome contrasts among Seldonian runs
    sel = [r for r in rows_out if r["method"] == "seldonian_lag" and np.isfinite(r["late_share"])]
    nsf = [r["late_share"] for r in sel if not r["solution"]]
    sol = [r["late_share"] for r in sel if r["solution"]]
    lines.append(f"\nSeldonian runs: late share {np.mean(nsf):.2f} for NSF (n={len(nsf)}) vs "
                 f"{np.mean(sol):.2f} for solutions (n={len(sol)}).")
    fe = [(r["late_share"], r["feasible"] / r["n_pred"]) for r in sel if r["n_pred"]]
    if len(fe) > 3:
        a = np.array(fe)
        rx, ry = np.argsort(np.argsort(a[:, 0])), np.argsort(np.argsort(a[:, 1]))
        lines.append(f"Spearman(late share, feasible fraction) over {len(fe)} Seldonian runs: "
                     f"{np.corrcoef(rx, ry)[0, 1]:+.2f}.")
    base = [r for r in rows_out if r["method"] in ("grpo", "composite") and np.isfinite(r["late_share"])]
    br = [r["late_share"] for r in base if r["breach"]]
    wi = [r["late_share"] for r in base if not r["breach"]]
    if br and wi:
        lines.append(f"Baselines: late share {np.mean(br):.2f} for breaching runs (n={len(br)}) vs "
                     f"{np.mean(wi):.2f} for runs within the threshold (n={len(wi)}).")
    with open(os.path.join(args.out, "profile.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
