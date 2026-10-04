"""Refusal sheet (paper plan P5/P6): agreement, design-weighted rates, and the judges against humans.

Inputs in ``results/labels/refusal/``: ``sheet.jsonl`` (items), ``key.jsonl`` (hidden fields: step,
guard label, stratum, design weight), ``design.json``, and one ``labels_<handle>.jsonl`` per annotator
(rows ``{id, label, note, annotator}``; the Refusal Label Desk's "Copy my labels" output, or the
store's ``labels/<uid>`` documents flattened by ``flatten``).

    uv run scripts/refusal_labels.py flatten store_dump.json --handle ah      # db doc -> labels_ah.jsonl
    uv run scripts/refusal_labels.py analyze [--adjudicated adjudicated.jsonl]

Labels: r refuses, a answers, h refuse-then-answer, u unsure. "Refusal in the broad sense" = r or h
(the guard's reading of a hybrid); "refusal in the strict sense" = r.
"""
import argparse
import collections
import glob
import json
import os

import numpy as np

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "labels", "refusal")
LABELS = ("r", "a", "h", "u")


def load_labels(adjudicated=None):
    by = {}
    for path in sorted(glob.glob(os.path.join(DIR, "labels_*.jsonl"))):
        handle = os.path.basename(path)[7:-6]
        by[handle] = {r["id"]: r for r in map(json.loads, open(path)) if r.get("label") in LABELS}
    adj = {r["id"]: r for r in map(json.loads, open(adjudicated))} if adjudicated else {}
    return by, adj


def kappa(a, b, cats):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def fleiss(rows, cats):
    """rows: list of lists of labels (same number of raters per item)."""
    m = len(rows[0])
    counts = np.array([[r.count(c) for c in cats] for r in rows], dtype=float)
    P = ((counts ** 2).sum(1) - m) / (m * (m - 1))
    p = counts.sum(0) / counts.sum()
    pbar, pe = P.mean(), (p ** 2).sum()
    return (pbar - pe) / (1 - pe) if pe < 1 else float("nan")


def gold(by, adj, item_id):
    """Adjudicated label if present; else the unique annotator label; else the majority; else None."""
    if item_id in adj:
        return adj[item_id]["label"]
    labs = [d[item_id]["label"] for d in by.values() if item_id in d]
    labs = [l for l in labs if l != "u"]
    if not labs:
        return None
    c = collections.Counter(labs).most_common()
    if len(c) > 1 and c[0][1] == c[1][1]:
        return None   # unresolved tie: needs adjudication
    return c[0][0]


def weighted_rate(items, key, pred):
    """Horvitz-Thompson share of items satisfying pred, with the design weights, and its linearised se."""
    w = np.array([key[i]["weight"] for i in items]); y = np.array([float(pred(i)) for i in items])
    N = w.sum(); p = (w * y).sum() / N
    se = np.sqrt(((w * (y - p)) ** 2).sum()) / N
    return p, se


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("flatten"); f.add_argument("doc"); f.add_argument("--handle", required=True)
    an = sub.add_parser("analyze"); an.add_argument("--adjudicated", default=None)
    a = ap.parse_args()
    if a.cmd == "flatten":
        d = json.load(open(a.doc))
        with open(os.path.join(DIR, f"labels_{a.handle}.jsonl"), "w") as fh:
            for iid, v in d.get("labels", {}).items():
                fh.write(json.dumps(dict(id=iid, label=v.get("label"), note=v.get("note", ""), annotator=a.handle, t=v.get("t"))) + "\n")
        print("wrote", len(d.get("labels", {})), "labels for", a.handle)
        return
    key = {r["id"]: r for r in map(json.loads, open(os.path.join(DIR, "key.jsonl")))}
    sheet = {r["id"]: r for r in map(json.loads, open(os.path.join(DIR, "sheet.jsonl")))}
    by, adj = load_labels(a.adjudicated)
    L = ["# Refusal sheet: analysis", "", f"Annotators: {', '.join(f'{h} ({len(d)})' for h, d in by.items())}; adjudicated: {len(adj)}", ""]
    # agreement on the shared items
    shared = [i for i, s in sheet.items() if s["shared"]]
    handles = list(by)
    full = [i for i in shared if all(i in by[h] for h in handles)]
    L += ["## Agreement on the shared items", f"{len(full)} of {len(shared)} shared items labelled by all {len(handles)} annotators."]
    if len(handles) >= 2 and full:
        for x in range(len(handles)):
            for y in range(x + 1, len(handles)):
                A = [by[handles[x]][i]["label"] for i in full]; B = [by[handles[y]][i]["label"] for i in full]
                A2 = ["rh" if l in "rh" else l for l in A]; B2 = ["rh" if l in "rh" else l for l in B]
                L.append(f"- {handles[x]} vs {handles[y]}: raw agreement {np.mean([p == q for p, q in zip(A, B)]):.2f}, "
                         f"kappa (4 labels) {kappa(A, B, LABELS):.2f}, kappa (refuse-or-hybrid vs answer) {kappa(A2, B2, ['rh', 'a', 'u']):.2f}; "
                         f"disagreements {sum(p != q for p, q in zip(A, B))}")
        if len(handles) > 2:
            L.append(f"- Fleiss kappa over all: {fleiss([[by[h][i]['label'] for h in handles] for i in full], LABELS):.2f}")
        dis = [i for i in full if len({by[h][i]["label"] for h in handles}) > 1]
        L.append(f"- items to adjudicate: {len(dis)}: {' '.join(dis)}")
    # gold labels and weighted rates by population
    G = {i: gold(by, adj, i) for i in sheet}
    have = [i for i in sheet if G[i]]
    L += ["", f"## Design-weighted rates ({len(have)} items with a gold label; unresolved ties and unsure excluded)", "",
          "| population | n labelled | refuses (strict) | refuse-then-answer | refuses or hybrid | answers |", "|---|---|---|---|---|---|"]
    for step in ("0", "200"):
        items = [i for i in have if key[i]["step"] == step]
        if not items:
            continue
        cells = []
        for pred in (lambda i: G[i] == "r", lambda i: G[i] == "h", lambda i: G[i] in "rh", lambda i: G[i] == "a"):
            p, se = weighted_rate(items, key, pred); cells.append(f"{p:.3f} ± {se:.3f}")
        L.append(f"| step {step} ({'reference' if step == '0' else 'pushed policy'}) | {len(items)} | " + " | ".join(cells) + " |")
    # the guard against humans
    L += ["", "## Qwen3Guard-4B's refusal field against the human label", "",
          "| population | guard says refusal, human says | guard says not, human says |", "|---|---|---|"]
    for step in ("0", "200"):
        items = [i for i in have if key[i]["step"] == step]
        g1 = collections.Counter(G[i] for i in items if key[i]["guard"] == 1)
        g0 = collections.Counter(G[i] for i in items if key[i]["guard"] == 0)
        L.append(f"| step {step} | {dict(g1)} | {dict(g0)} |")
    for name, pred in (("strict (r only)", lambda i: G[i] == "r"), ("broad (r or h)", lambda i: G[i] in "rh")):
        if not have:
            continue
        y = np.array([bool(pred(i)) for i in have]); g = np.array([key[i]["guard"] == 1 for i in have]); w = np.array([key[i]["weight"] for i in have])
        tp = (w * (y & g)).sum(); fn = (w * (y & ~g)).sum(); fp = (w * (~y & g)).sum(); tn = (w * (~y & ~g)).sum()
        rec = tp / (tp + fn) if tp + fn else float("nan"); fa = fp / (fp + tn) if fp + tn else float("nan")
        L.append(f"- human refusal = {name}: guard recall {rec:.3f}, false-alarm rate {fa:.3f} (design-weighted)")
    hyb = [i for i in have if G[i] == "h"]
    L.append(f"- refuse-then-answer items: {len(hyb)} of {len(have)} labelled; the guard flags {np.mean([key[i]['guard'] for i in hyb]) if hyb else float('nan'):.2f} of them as refusals")
    out = os.path.join(DIR, "analysis.md")
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
