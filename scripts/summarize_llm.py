"""Summarize run_llm_rl.py results into one table per task.

    uv run scripts/summarize_llm.py [--out results/llm] [--task ab]
    uv run scripts/summarize_llm.py --out results/llm_r5/v8 results/llm_r6/b1_v8 --task brevity

With several ``--out`` directories every row is labelled by the directory's basename
(a leading ``dir`` column) and the summary is per (directory, method).

Columns: constraint rates on D_s with the threshold each run used, the reward-model
mean, whether the run returned a solution (Seldonian: passed the safety test;
baselines: always, with `viol` marking a threshold breach that carries no
guarantee), and wall-clock.
"""
import argparse
import glob
import json
import os


def load(out, task):
    rows = []
    for path in sorted(glob.glob(os.path.join(out, task, "*", "seed*", "result.json"))):
        with open(path) as f:
            r = json.load(f)
        if "safety_test" in r:
            rates, reward = r["safety_test"]["rates"], r["safety_test"]["reward"]
            upper = r["safety_test"]["upper"]
            length = r["safety_test"]["mean_length"]
        else:
            rates, reward = r["eval_s"]["rates"], r["eval_s"]["reward"]
            upper, length = {}, r["eval_s"]["mean_length"]
        rows.append({
            "method": r["method"], "seed": r["seed"], "solution": r["solution_found"],
            "rates": rates, "upper": upper, "thresholds": r["thresholds"], "reward": reward,
            "length": length, "minutes": r["total_seconds"] / 60,
            "selected": (r.get("selected") or {}).get("step"),
            "n_pred": len(r.get("history", [])),
            "n_feasible": sum(h["feasible"] for h in r.get("history", [])),
        })
    return rows


def fmt_rate(rows_rates, upper, thr, name):
    v = rows_rates.get(name)
    if v is None:
        return "-"
    s = f"{v:.3f}"
    if name in upper:
        s += f" (ub {upper[name]:.3f})"
    if thr.get(name) is not None and v > thr[name]:
        s += " viol"
    return s


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", nargs="+", default=["results/llm"],
                   help="one or more result directories (rows are labelled by basename)")
    p.add_argument("--task", default="ab")
    args = p.parse_args()
    multi = len(args.out) > 1
    rows = []
    for out in args.out:
        for r in load(out, args.task):
            r["dir"] = os.path.basename(os.path.normpath(out))
            rows.append(r)
    if not rows:
        print("no results yet")
        return
    names = sorted({k for r in rows for k in r["rates"]})
    head = ["method", "seed", "sol"]
    head += [f"{n} (tau)" for n in names] + ["reward", "len", "ckpt", "feasible", "min"]
    if multi:
        head = ["dir"] + head
    table = [head]
    for r in rows:
        table.append([r["method"], str(r["seed"]), "yes" if r["solution"] else "NSF"] +
                     [fmt_rate(r["rates"], r["upper"], r["thresholds"], n) +
                      f" ({r['thresholds'].get(n, float('nan')):.3f})" for n in names] +
                     [f"{r['reward']:.2f}", f"{r['length']:.0f}", str(r["selected"]),
                      f"{r['n_feasible']}/{r['n_pred']}" if r["n_pred"] else "-",
                      f"{r['minutes']:.0f}"])
        if multi:
            table[-1].insert(0, r["dir"])
    widths = [max(len(row[i]) for row in table) for i in range(len(head))]
    for i, row in enumerate(table):
        print("  ".join(c.ljust(w) for c, w in zip(row, widths)))
        if i == 0:
            print("  ".join("-" * w for w in widths))

    print()
    for d, method in sorted({(r["dir"], r["method"]) for r in rows}):
        sub = [r for r in rows if r["dir"] == d and r["method"] == method]
        sol = sum(r["solution"] for r in sub)
        viol = sum(any(r["rates"].get(n, 0) > r["thresholds"].get(n, 1) for n in names)
                   for r in sub)
        rew = sum(r["reward"] for r in sub) / len(sub)
        means = []
        for n in names:
            vals = [r["rates"][n] for r in sub if r["rates"].get(n) is not None]
            if vals:
                means.append(f"{n}={sum(vals) / len(vals):.3f}")
        label = f"{d} {method}" if multi else method
        print(f"{label:24s} runs={len(sub)} solution={sol}/{len(sub)} "
              f"threshold_breach_on_D_s={viol}/{len(sub)} mean_reward={rew:.2f}"
              + "".join(" " + m for m in means))


if __name__ == "__main__":
    main()
