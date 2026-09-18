"""Merge the PyRIT red-team batteries under results/redteam into one table.

    uv run scripts/summarize_redteam.py results/redteam/c0_harm results/redteam/ref_harm ...
    uv run scripts/summarize_redteam.py --all results/redteam

Rows are techniques, columns are batteries (policy x constraint); each cell is
``rate (lower-upper)`` with a marker: ``**`` certified breach (lower > tau), ``*``
point breach, ``ok`` the safety test would pass (upper <= tau), blank inconclusive.
A second table gives the analytic resampling reference ``1 - (1 - p)^k`` with ``p``
the baseline rate, for reading the resample rows against the i.i.d. case.
"""
import argparse
import glob
import json
import os
import sys


def load(path):
    with open(os.path.join(path, "summary.json")) as f:
        return json.load(f)


def marker(row):
    if row["certified_breach"]:
        return "**"
    if row["point_breach"]:
        return "*"
    if row["certificate_holds"]:
        return "ok"
    return ""


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dirs", nargs="*")
    p.add_argument("--all", default=None, help="every battery under this directory")
    p.add_argument("--out", default=None, help="write the markdown here too")
    args = p.parse_args()
    dirs = list(args.dirs)
    if args.all:
        dirs += sorted(d for d in glob.glob(os.path.join(args.all, "*"))
                       if os.path.exists(os.path.join(d, "summary.json")))
    if not dirs:
        sys.exit("no batteries")
    batteries = [(os.path.basename(d.rstrip("/")), load(d)) for d in dirs]

    techniques = []
    for _, s in batteries:
        for r in s["rows"]:
            if r["technique"] not in techniques:
                techniques.append(r["technique"])
    techniques.sort(key=lambda t: (t != "baseline", t))

    lines = ["| technique | " + " | ".join(f"{n} (tau {s['tau']:.3f}, n {s['n_prompts']})"
                                          for n, s in batteries) + " |",
             "|---|" + "---|" * len(batteries)]
    for t in techniques:
        cells = []
        for _, s in batteries:
            row = next((r for r in s["rows"] if r["technique"] == t), None)
            if row is None or row["n"] == 0:
                cells.append("")
                continue
            err = f" e{row['errors']}" if row["errors"] else ""
            cells.append(f"{row['rate']:.3f} ({row['lower']:.3f}-{row['upper']:.3f}){err} {marker(row)}".strip())
        lines.append(f"| {t} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)

    ref_lines = ["| battery | baseline p | 1-(1-p)^4 | resample_4 | 1-(1-p)^8 | resample_8 | 1-(1-p)^16 | resample_16 |",
                 "|---|---|---|---|---|---|---|---|"]
    for n, s in batteries:
        rows = {r["technique"]: r for r in s["rows"]}
        if "baseline" not in rows:
            continue
        pb = rows["baseline"]["rate"]
        cells = [n, f"{pb:.3f}"]
        for k in (4, 8, 16):
            cells.append(f"{1 - (1 - pb) ** k:.3f}")
            r = rows.get(f"resample_{k}")
            cells.append(f"{r['rate']:.3f}" if r and r["n"] else "")
        ref_lines.append("| " + " | ".join(cells) + " |")
    ref = "\n".join(ref_lines)

    meta = "\n".join(f"- {n}: policy `{s['policy']}`, {s['constraint']} on {s['n_prompts']} "
                     f"{s['group'] or 'all'} prompts, {s['bound']} at delta {s['delta']:.3f}; safety test "
                     f"rate {((s.get('safety_test') or {}).get('rates') or {}).get(s['constraint'], float('nan')):.3f}; "
                     f"{s['scenario_seconds'] / 60:.0f} min, {s['policy_requests']} policy requests"
                     for n, s in batteries)
    out = f"{meta}\n\n{table}\n\nResampling against the i.i.d. reference:\n\n{ref}\n"
    print(out)
    if args.out:
        with open(args.out, "w") as f:
            f.write(out)


if __name__ == "__main__":
    main()
