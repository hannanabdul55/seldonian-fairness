"""Tables for spike 015 from results/spikes/015/scores.jsonl. Writes results.md.

Fidelity: accuracy, false-alarm rate (Clopper-Pearson 90%), recall, AUC against the
reference label, plus the trained guard's own numbers on the same items where available.
Calibration: 10-bin ECE and the Brier score of P(Yes).
Paraphrase: flip rate of the label against the canonical wording, and the spread of the
measured rate across wordings (what the constraint's *value* would be, wording to wording).
Rubric artifact (arXiv:2609.02942): ``blank`` AUC (a response-reading judge should be near
0.5) and its correlation with the canonical scores; ``swap`` AUC against the response
actually shown (should stay high) and the flip rate against canonical.

    ../../../.venv/bin/python analyze015.py
"""
import json
import os

import numpy as np
from scipy.stats import beta

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "..", "results", "spikes", "015")


def cp(k, n, d=0.05):
    lo = 0.0 if k == 0 else float(beta.ppf(d, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - d, k + 1, n - k))
    return lo, hi


def auc(y, p):
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    # average ranks over ties
    vals = np.concatenate([pos, neg])
    for v in np.unique(vals):
        m = vals == v
        ranks[m] = ranks[m].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def ece(y, p, bins=10):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= 1.0)
        if m.sum():
            out += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(out)


def load():
    rows = [json.loads(l) for l in open(os.path.join(OUT, "scores.jsonl"))]
    d = {}
    for r in rows:
        d.setdefault((r["task"], r["variant"], r["wording"]), []).append(r)
    return d


def block(rows, ref_key="ref"):
    y = np.array([r[ref_key] for r in rows])
    p = np.array([r["p"] for r in rows], dtype=float)
    lab = (p > 0.5).astype(int)
    neg, pos = y == 0, y == 1
    fa = int(lab[neg].sum())
    lo, hi = cp(fa, int(neg.sum()))
    return dict(n=len(rows), rate=float(lab.mean()), ref_rate=float(y.mean()),
                acc=float((lab == y).mean()), fa=fa, n_neg=int(neg.sum()),
                fa_rate=float(fa / max(neg.sum(), 1)), fa_lo=lo, fa_hi=hi,
                caught=int(lab[pos].sum()), n_pos=int(pos.sum()),
                auc=auc(y, p), ece=ece(y, p), brier=float(np.mean((p - y) ** 2)))


def main():
    d = load()
    tasks = sorted({k[0] for k in d})
    L = ["# Spike 015 results", "",
         "P(Yes) from one forward pass of Qwen3-8B (4-bit, thinking off). `rubric` = compiled "
         "instructions, `raw` = the developer's sentence as the instruction. Wording 0 is "
         "canonical; 1-5 are model-written paraphrases. FA = flagged when the reference says "
         "no (90% Clopper-Pearson).", ""]

    L += ["## Fidelity and calibration (canonical wording)", "",
          "| task | variant | n | measured rate | reference rate | accuracy | FA rate (90% CI) "
          "| caught | AUC | ECE | Brier |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in tasks:
        for v in ("rubric", "raw"):
            rows = d.get((t, v, 0))
            if not rows:
                continue
            b = block(rows)
            L.append(f"| {t} | {v} | {b['n']} | {b['rate']:.3f} | {b['ref_rate']:.3f} "
                     f"| {b['acc']:.3f} | {b['fa_rate']:.3f} ({b['fa_lo']:.3f}-{b['fa_hi']:.3f}) "
                     f"| {b['caught']}/{b['n_pos']} | {b['auc']:.3f} | {b['ece']:.3f} "
                     f"| {b['brier']:.3f} |")
    # the trained guard on the same harm items
    hrows = d.get(("harm", "rubric", 0))
    if hrows and hrows[0].get("guard") is not None:
        y = np.array([r["ref"] for r in hrows])
        g = np.array([r["guard"] for r in hrows])
        fa = int(g[y == 0].sum())
        lo, hi = cp(fa, int((y == 0).sum()))
        L.append(f"| harm | guard (Qwen3Guard-4B) | {len(hrows)} | {g.mean():.3f} "
                 f"| {y.mean():.3f} | {(g == y).mean():.3f} | {fa / max((y == 0).sum(), 1):.3f} "
                 f"({lo:.3f}-{hi:.3f}) | {int(g[y == 1].sum())}/{int((y == 1).sum())} | - | - | - |")

    L += ["", "## Paraphrase robustness (rubric variant, wordings 0-5)", "",
          "Flip rate = share of items whose label differs from the canonical wording's. "
          "The rate spread is what the constraint's measured value would be, wording to wording.",
          "", "| task | wordings | measured rate: min-max | mean flip rate vs canonical "
          "| max pairwise disagreement | AUC range |", "|---|---|---|---|---|---|"]
    for t in tasks:
        ws = sorted(w for (tt, v, w) in d if tt == t and v == "rubric")
        if len(ws) < 2:
            continue
        byid = {w: {r["id"]: r["p"] for r in d[(t, "rubric", w)]} for w in ws}
        ids = sorted(set.intersection(*[set(byid[w]) for w in ws]))
        labs = {w: np.array([byid[w][i] > 0.5 for i in ids], dtype=int) for w in ws}
        rates = [float(labs[w].mean()) for w in ws]
        flips = [float((labs[w] != labs[0]).mean()) for w in ws[1:]]
        pair = max(float((labs[a] != labs[b]).mean()) for a in ws for b in ws if a < b)
        aucs = [block(d[(t, "rubric", w)])["auc"] for w in ws]
        L.append(f"| {t} | {len(ws)} | {min(rates):.3f}-{max(rates):.3f} | {np.mean(flips):.3f} "
                 f"| {pair:.3f} | {min(aucs):.3f}-{max(aucs):.3f} |")

    L += ["", "## Rubric artifact (canonical rubric)", "",
          "`blank`: the response removed. A judge that reads the response should be near AUC "
          "0.5 and uncorrelated with its own real scores. `swap`: responses shuffled across "
          "prompts; AUC is against the response actually shown, so a response-reading judge "
          "keeps it high.", "",
          "| task | blank AUC | blank rate | corr(blank, canonical) | swap AUC (shown response) "
          "| swap flip rate vs canonical |", "|---|---|---|---|---|---|"]
    for t in tasks:
        base = d.get((t, "rubric", 0))
        if not base:
            continue
        bp = {r["id"]: r["p"] for r in base}
        bl = d.get((t, "blank", 0))
        sw = d.get((t, "swap", 0))
        cells = []
        if bl:
            b = block(bl)
            ids = [r["id"] for r in bl if r["id"] in bp]
            c = np.corrcoef([r["p"] for r in bl if r["id"] in bp], [bp[i] for i in ids])[0, 1] \
                if len(ids) > 2 else float("nan")
            cells += [f"{b['auc']:.3f}", f"{b['rate']:.3f}", f"{c:+.3f}"]
        else:
            cells += ["-", "-", "-"]
        if sw:
            b = block(sw, ref_key="ref_shown")
            lab_s = np.array([r["p"] > 0.5 for r in sw if r["id"] in bp], dtype=int)
            lab_c = np.array([bp[r["id"]] > 0.5 for r in sw if r["id"] in bp], dtype=int)
            cells += [f"{b['auc']:.3f}", f"{float((lab_s != lab_c).mean()):.3f}"]
        else:
            cells += ["-", "-"]
        L.append(f"| {t} | " + " | ".join(cells) + " |")

    path = os.path.join(HERE, "results.md")
    open(path, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
