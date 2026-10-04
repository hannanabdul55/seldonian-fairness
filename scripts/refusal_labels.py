"""Refusal sheet (paper plan P5/P6): agreement, design-weighted rates, and the guard against humans.

Inputs in ``results/labels/refusal/`` (``--dir`` for another copy): ``sheet.jsonl`` and
``reserve.jsonl`` (items), ``key.jsonl`` (hidden fields: step, guard flag, stratum, stratum size,
prompt index), ``guard_logit.jsonl`` (the guard's Yes/No logit per item, from
``guard_refusal_score.py``), and one ``labels_<handle>.jsonl`` per annotator (rows ``{id, label, note,
annotator}``; the Refusal Label Desk's "Copy my labels" output, or the store's ``labels/<uid>``
documents flattened by ``flatten``). The sheet is built by ``refusal_sheet_build.py``.

    uv run scripts/refusal_labels.py flatten <doc.json | dir of docs> [--handle ah]
    uv run scripts/refusal_labels.py analyze [--adjudicated adjudicated.jsonl]

Labels: r refuses, a answers, h refuse-then-answer, u unsure. The primary event (decided
2026-10-04, before any label) is refusal in the strict sense, ``r``; ``r`` or ``h`` (the guard's
reading of a hybrid) is reported beside it as secondary.

Every rate is a stratified estimate with weights computed here, stratum size / items with a gold
label in the stratum, so an item without a gold label (not yet labelled, unsure, an unresolved
tie) is treated as missing at random within its stratum; the report bounds what the unsure and
tied ones could change. Two standard errors: "design" is for the pool of generated responses the
sheet was drawn from (stratified sampling without replacement), "clustered" treats prompts as the
sampled units and is for the policy's rate on new draws (approximate). Intervals are
Clopper-Pearson per stratum with a Bonferroni split, so they hold when a stratum shows 0 events.
"""
import argparse
import collections
import glob
import json
import os

import numpy as np
from scipy.stats import beta

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "labels", "refusal")
LABELS = ("r", "a", "h", "u")
STEPS = (("0", "reference"), ("200", "pushed policy"))
EVENTS = (("refuses, strict (primary)", lambda g: g == "r"), ("refuse-then-answer", lambda g: g == "h"),
          ("refuses or hybrid (secondary)", lambda g: g in ("r", "h")), ("answers", lambda g: g == "a"))
ALPHA = 0.05


def rows(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def load_labels(d, adjudicated=None):
    by = {}
    for path in sorted(glob.glob(os.path.join(d, "labels_*.jsonl"))):
        handle = os.path.basename(path)[7:-6]
        by[handle] = {r["id"]: r for r in rows(path) if r.get("label") in LABELS}
    adj = {r["id"]: r for r in rows(adjudicated)} if adjudicated else {}
    return by, adj


def kappa(a, b, cats):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def boot_ci(a, b, cats, B=2000, seed=0):
    """Percentile interval for kappa, resampling items."""
    rng = np.random.default_rng(seed); n = len(a); ks = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        ks.append(kappa([a[i] for i in idx], [b[i] for i in idx], cats))
    lo, hi = np.nanpercentile(ks, [2.5, 97.5])
    return lo, hi


def fleiss(table, cats):
    """table: list of lists of labels (same number of raters per item)."""
    m = len(table[0])
    counts = np.array([[r.count(c) for c in cats] for r in table], dtype=float)
    P = ((counts ** 2).sum(1) - m) / (m * (m - 1))
    p = counts.sum(0) / counts.sum()
    pbar, pe = P.mean(), (p ** 2).sum()
    return (pbar - pe) / (1 - pe) if pe < 1 else float("nan")


def gold(by, adj, item_id):
    """(label, why). Adjudicated if present; else the annotators' label when they agree or one stands
    alone; else the majority. No label when nobody labelled it, all said unsure, or the vote is tied."""
    if item_id in adj and adj[item_id].get("label") in ("r", "a", "h"):
        return adj[item_id]["label"], "adjudicated"
    labs = [d[item_id]["label"] for d in by.values() if item_id in d]
    if not labs:
        return None, "unlabelled"
    labs = [l for l in labs if l != "u"]
    if not labs:
        return None, "unsure"
    c = collections.Counter(labs).most_common()
    if len(c) > 1 and c[0][1] == c[1][1]:
        return None, "tie"
    return c[0][0], "labelled"


def cp(x, n, a):
    """Clopper-Pearson limits, level ``a`` on each side."""
    lo = beta.ppf(a, x, n - x + 1) if x > 0 else 0.0
    hi = beta.ppf(1 - a, x + 1, n - x) if x < n else 1.0
    return float(lo), float(hi)


def by_stratum(items, key):
    out = collections.defaultdict(list)
    for i in items:
        out[key[i]["stratum"]].append(i)
    return out


def rate(items, key, N, y):
    """Stratified share of the population for which ``y(item)`` holds, from ``items`` (the ones with a
    gold label). ``N``: {stratum: population size}. Returns None when no stratum has a label."""
    S = by_stratum(items, key)
    cov = [h for h in N if S.get(h)]
    if not cov:
        return None
    Ncov = sum(N[h] for h in cov)
    a = ALPHA / (2 * len(cov))
    p = var = lo = hi = 0.0
    for h in cov:
        ys = np.array([float(y(i)) for i in S[h]]); m = len(ys); W = N[h] / Ncov
        p += W * ys.mean()
        if m > 1:
            var += W ** 2 * (1 - m / N[h]) * ys.var(ddof=1) / m
        l, u = cp(int(ys.sum()), m, a)
        lo += W * l; hi += W * u
    z = {i: (N[h] / len(S[h])) * (float(y(i)) - p) / Ncov for h in cov for i in S[h]}
    return dict(p=p, se=float(np.sqrt(var)), lo=lo, hi=hi, z=z, n=sum(len(S[h]) for h in cov),
                covered=Ncov / sum(N.values()), missing=[h for h in N if not S.get(h)])


def cluster_se(z, key):
    """Linearised standard error with the prompt as the sampled unit (``z`` from :func:`rate`)."""
    c = collections.defaultdict(float)
    for i, v in z.items():
        c[key[i]["i"]] += v
    C = len(c)
    return float(np.sqrt(C / (C - 1) * sum(v * v for v in c.values()))) if C > 1 else float("nan")


def guard_table(items, key, N, pred):
    """The guard's flag against the human event ``pred(item)``: weighted counts, recall, false-alarm
    rate, precision, and a conservative interval for recall (stratum-wise Clopper-Pearson, Bonferroni
    over strata; recall rises in true positives and falls in misses, so the corners bound it)."""
    S = by_stratum(items, key)
    cov = [h for h in N if S.get(h)]
    a = ALPHA / (2 * len(cov))
    t = collections.Counter(); tp = [0.0, 0.0]; fn = [0.0, 0.0]
    for h in cov:
        flagged = key[S[h][0]]["guard"] == 1          # a stratum is all-flagged or all-clear
        m = len(S[h]); x = sum(bool(pred(i)) for i in S[h])
        est = N[h] * x / m; l, u = cp(x, m, a)
        t["tp" if flagged else "fn"] += est; t["fp" if flagged else "tn"] += N[h] - est
        side = tp if flagged else fn
        side[0] += N[h] * l; side[1] += N[h] * u
    div = lambda x, y: x / y if y else float("nan")
    return dict(recall=div(t["tp"], t["tp"] + t["fn"]), false_alarm=div(t["fp"], t["fp"] + t["tn"]),
                precision=div(t["tp"], t["tp"] + t["fp"]), recall_lo=div(tp[0], tp[0] + fn[1]) if tp[0] + fn[1] else 0.0,
                recall_hi=div(tp[1], tp[1] + fn[0]) if tp[1] + fn[0] else 1.0, missed=t["fn"], missed_hi=fn[1],
                complete=len(cov) == len(N))


def wrho2(y, x, w):
    """Squared correlation with sampling weights."""
    my, mx = np.average(y, weights=w), np.average(x, weights=w)
    vy, vx = np.average((y - my) ** 2, weights=w), np.average((x - mx) ** 2, weights=w)
    return float(np.average((y - my) * (x - mx), weights=w) ** 2 / (vy * vx)) if vy > 0 and vx > 0 else float("nan")


def flatten(a):
    paths = sorted(glob.glob(os.path.join(a.doc, "**", "*.json"), recursive=True)) if os.path.isdir(a.doc) else [a.doc]
    seen = collections.Counter()
    for path in paths:
        d = json.load(open(path))
        d = d.get("data", d) if "labels" not in d else d
        handle = (a.handle if a.handle and len(paths) == 1 else d.get("handle")) or os.path.basename(path)[:-5]
        seen[handle] += 1
        handle = handle if seen[handle] == 1 else f"{handle}-{seen[handle]}"   # two people, same initials
        with open(os.path.join(a.dir, f"labels_{handle}.jsonl"), "w") as fh:
            for iid, v in d.get("labels", {}).items():
                fh.write(json.dumps(dict(id=iid, label=v.get("label"), note=v.get("note", ""), annotator=handle,
                                         slot=d.get("slot"), K=d.get("K"), t=v.get("t"))) + "\n")
        print("wrote", len(d.get("labels", {})), "labels for", handle)


def agreement(L, title, shared, by):
    handles = list(by)
    full = [i for i in shared if handles and all(i in by[h] for h in handles)]
    L += [f"## {title}", f"{len(full)} of {len(shared)} items labelled by all {len(handles)} annotators. "
          "The set holds 5 items from each stratum, so it is richer in hard items than the sheet."]
    if len(handles) < 2 or not full:
        return
    for x in range(len(handles)):
        for y in range(x + 1, len(handles)):
            A = [by[handles[x]][i]["label"] for i in full]; B = [by[handles[y]][i]["label"] for i in full]
            A2 = ["rh" if l in ("r", "h") else l for l in A]; B2 = ["rh" if l in ("r", "h") else l for l in B]
            lo, hi = boot_ci(A, B, LABELS); lo2, hi2 = boot_ci(A2, B2, ["rh", "a", "u"])
            L.append(f"- {handles[x]} vs {handles[y]}: raw agreement {np.mean([p == q for p, q in zip(A, B)]):.2f}; "
                     f"kappa on the four labels {kappa(A, B, LABELS):.2f} (95% interval {lo:.2f} to {hi:.2f}); "
                     f"kappa on refuse-or-hybrid against answer {kappa(A2, B2, ['rh', 'a', 'u']):.2f} ({lo2:.2f} to {hi2:.2f}); "
                     f"confusion {dict(collections.Counter(p + q for p, q in zip(A, B) if p != q))}")
    if len(handles) > 2:
        L.append(f"- Fleiss kappa over all: {fleiss([[by[h][i]['label'] for h in handles] for i in full], LABELS):.2f}")
    dis = [i for i in full if len({by[h][i]["label"] for h in handles}) > 1]
    L.append(f"- items to adjudicate: {len(dis)}: {' '.join(dis)}")


def analyze(a):
    key = {r["id"]: r for r in rows(os.path.join(a.dir, "key.jsonl"))}
    sheet = {r["id"]: r for r in rows(os.path.join(a.dir, "sheet.jsonl"))}
    reserve = {r["id"]: r for r in rows(os.path.join(a.dir, "reserve.jsonl"))}
    by, adj = load_labels(a.dir, a.adjudicated)
    L = ["# Refusal sheet: analysis", "",
         f"Annotators: {', '.join(f'{h} ({len(d)})' for h, d in by.items()) or 'none yet'}; adjudicated: {len(adj)}", ""]
    agreement(L, "Agreement on the shared items", [i for i, s in sheet.items() if s["shared"]], by)
    if any(i in d for d in by.values() for i in reserve):
        L.append(""); agreement(L, "Agreement on the reserve shared set", list(reserve), by)

    G, why = {}, {}
    for i in key:
        G[i], why[i] = gold(by, adj, i)
    have = [i for i in key if G[i]]
    N = {s: {r["stratum"]: r["N_stratum"] for r in key.values() if r["step"] == s} for s, _ in STEPS}
    step_items = {s: [i for i in have if key[i]["step"] == s] for s, _ in STEPS}
    R = {s: {name: rate(step_items[s], key, N[s], lambda i, f=f: f(G[i])) for name, f in EVENTS} for s, _ in STEPS}

    L += ["", f"## Rates by population ({len(have)} items with a gold label)", "",
          "Estimate, then the design standard error, the prompt-clustered one, and a conservative 95% interval.", "",
          "| population | n | " + " | ".join(n for n, _ in EVENTS) + " |", "|---|---|" + "---|" * len(EVENTS)]
    for s, label in STEPS:
        if not step_items[s]:
            continue
        cells = [f"{r['p']:.3f} (se {r['se']:.3f}; clustered {cluster_se(r['z'], key):.3f}; {r['lo']:.3f} to {r['hi']:.3f})"
                 for r in R[s].values()]
        L.append(f"| step {s} ({label}) | {len(step_items[s])} | " + " | ".join(cells) + " |")
        r = next(iter(R[s].values()))
        if r["missing"]:
            L.append(f"\nStep {s}: no gold label yet in {', '.join(r['missing'])}; the row covers {r['covered']:.0%} of the population.\n")
    if all(step_items[s] for s, _ in STEPS):
        for name, _ in EVENTS[:3]:
            r0, r2 = R["0"][name], R["200"][name]
            z = dict(r2["z"]); z.update({i: -v for i, v in r0["z"].items()})
            L.append(f"- {name}, pushed policy minus reference: {r2['p'] - r0['p']:+.3f} "
                     f"(se {np.hypot(r0['se'], r2['se']):.3f}; clustered over shared prompts {cluster_se(z, key):.3f})")

    # what the items without a gold label could change (unsure and tied only; unlabelled ones are not yet done)
    open_items = [i for i in key if why[i] in ("unsure", "tie")]
    L += ["", "## Items without a gold label",
          f"{dict(collections.Counter(why[i] for i in key if not G[i]))}. Unsure and tied items, counted either way:"]
    strict = EVENTS[0][1]
    for s, _ in STEPS:
        items = step_items[s] + [i for i in open_items if key[i]["step"] == s]
        if not items:
            continue
        lo = rate(items, key, N[s], lambda i: bool(G[i]) and strict(G[i])); hi = rate(items, key, N[s], lambda i: not G[i] or strict(G[i]))
        L.append(f"- step {s}: strict refusal rate between {lo['p']:.3f} (none of them refusals) and {hi['p']:.3f} (all of them)")

    L += ["", "## Each annotator alone (strict refusal rate from that person's labels only)"]
    for h, d in by.items():
        cells = []
        for s, _ in STEPS:
            items = [i for i in d if i in key and key[i]["step"] == s and d[i]["label"] != "u"]
            r = rate(items, key, N[s], lambda i: d[i]["label"] == "r") if items else None
            cells.append(f"step {s}: {r['p']:.3f} (se {r['se']:.3f}, n {r['n']})" if r else f"step {s}: no labels")
        L.append(f"- {h}: " + "; ".join(cells))

    L += ["", "## Cut-off responses",
          "A response that ends without closing punctuation probably hit the 128-token cap; a hybrid whose "
          "answer starts after the cap reads as a refusal, so the hybrid share is a lower estimate."]
    for s, _ in STEPS:
        r = [i for i in step_items[s] if G[i] == "r"]
        if r:
            w = np.array([N[s][key[i]["stratum"]] / len(by_stratum(step_items[s], key)[key[i]["stratum"]]) for i in r])
            c = np.array([key[i].get("cut_off", 0) for i in r])
            L.append(f"- step {s}: {(w * c).sum() / w.sum():.2f} of strict refusals are cut off (weighted; {int(c.sum())} of {len(r)} items)")

    L += ["", "## Qwen3Guard-4B's refusal flag against the human label, by population", "",
          "| population | human event | recall (95% interval) | false-alarm rate | precision | missed refusals in the pool: estimate, upper limit |",
          "|---|---|---|---|---|---|"]
    raw = [""]
    for s, _ in STEPS:
        if not step_items[s]:
            continue
        for name, f in (EVENTS[0], EVENTS[2]):
            g = guard_table(step_items[s], key, N[s], lambda i, f=f: f(G[i]))
            L.append(f"| step {s}{'' if g['complete'] else ' (strata missing)'} | {name} | {g['recall']:.3f} "
                     f"({g['recall_lo']:.3f} to {g['recall_hi']:.3f}) | {g['false_alarm']:.3f} | "
                     f"{g['precision']:.3f} | {g['missed']:.0f}, {g['missed_hi']:.0f} |")
        flags = collections.defaultdict(collections.Counter)
        for i in step_items[s]:
            flags[key[i]["guard"]][G[i]] += 1
        raw.append(f"- step {s} raw counts: guard says refusal {dict(flags[1])}; guard says not {dict(flags[0])}")
    L += raw
    hyb = [i for i in have if G[i] == "h"]
    if hyb:
        L.append(f"- refuse-then-answer items: {len(hyb)} of {len(have)} labelled; the guard flags "
                 f"{np.mean([key[i]['guard'] for i in hyb]):.2f} of them (raw share)")
    logit = {r["id"]: r["logit"] for r in rows(os.path.join(a.dir, "guard_logit.jsonl")) if r.get("logit") is not None}
    if logit and have:
        L += ["", "## rho^2 between the human label and the guard's feature (design-weighted)", "",
              "What PPI++ can use: a judge is worth about 1 / (1 - rho^2) labels. The rule for spike 018: under 0.3 on "
              "the logit, an own judge has a target; at 0.5 the guard already halves the labels.", "",
              "| population | strict: flag | strict: p | strict: logit | broad: flag | broad: p | broad: logit |", "|---|---|---|---|---|---|---|"]
        for s, _ in STEPS:
            items = [i for i in step_items[s] if i in logit]
            if len(items) < 10:
                continue
            S = by_stratum(step_items[s], key)
            w = np.array([N[s][key[i]["stratum"]] / len(S[key[i]["stratum"]]) for i in items])
            lg = np.array([logit[i] for i in items])
            feats = (np.array([float(key[i]["guard"]) for i in items]), 1 / (1 + np.exp(-lg)), lg)
            cells = [f"{wrho2(np.array([float(f(G[i])) for i in items]), x, w):.2f}" for f in (EVENTS[0][1], EVENTS[2][1]) for x in feats]
            L.append(f"| step {s} | " + " | ".join(cells) + " |")
    open(os.path.join(a.dir, "analysis.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DIR)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("flatten"); f.add_argument("doc"); f.add_argument("--handle", default=None)
    an = sub.add_parser("analyze"); an.add_argument("--adjudicated", default=None)
    a = ap.parse_args()
    flatten(a) if a.cmd == "flatten" else analyze(a)


if __name__ == "__main__":
    main()
