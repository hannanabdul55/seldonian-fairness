"""How much of a judged rate's variance is prompt-to-prompt? (spike 012 follow-up)

Uses the spike-005 screen (``results/screen/<model>/judged.jsonl``: 2 judged responses per
prompt). With two responses per prompt the one-way ANOVA estimator gives the between-prompt
variance ``s2_b = (MSB - MSW) / 2`` and the intra-prompt correlation
``ICC = s2_b / (s2_b + MSW)``: the share of the per-response Bernoulli variance that a
perfectly balanced safety set could remove (spike 012's bandit: 5.5% bought nothing, 22%
bought about +3.5 points of solution rate, with the exact covariate). ``meta`` is the part
of the between-prompt variance explained by metadata cells (kind x encoding): what
stratifying on metadata alone captures. 95% intervals: 2000 bootstrap resamples of prompts.

    ../../../.venv/bin/python heterogeneity.py
"""
import glob
import json
import os

import numpy as np

SCREEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                      "results", "screen")


def icc(pairs):
    """(ICC, mean) for an (n, 2) array of 0/1 labels."""
    x = np.asarray(pairs, dtype=float)
    n = len(x)
    if n < 3:
        return np.nan, np.nan
    m = x.mean()
    msb = 2 * ((x.mean(axis=1) - m) ** 2).sum() / (n - 1)
    msw = ((x - x.mean(axis=1, keepdims=True)) ** 2).sum() / n
    s2b = max((msb - msw) / 2, 0.0)
    tot = s2b + msw
    return (s2b / tot if tot > 0 else np.nan), m


def meta_share(pairs, cells):
    """Share of the total variance explained by cell means (between-cell, bias-corrected)."""
    x = np.asarray(pairs, dtype=float)
    m = x.mean()
    tot = m * (1 - m)
    if tot == 0:
        return np.nan
    between = 0.0
    for c in np.unique(cells):
        xc = x[cells == c]
        pc = xc.mean()
        # the cell mean's own sampling noise (2 responses per prompt, clustered) inflates it
        noise = xc.mean(axis=1).var(ddof=1) / len(xc) if len(xc) > 1 else 0.0
        between += len(xc) / len(x) * ((pc - m) ** 2 - noise)
    return max(between, 0.0) / tot


def boot(fn, pairs, cells, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(pairs)
    vals = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        vals.append(fn(pairs[i], cells[i]))
    return np.nanpercentile(vals, [2.5, 97.5])


def main():
    out = ["| model | label | subset | prompts | rate | ICC (95% CI) | metadata share (95% CI) |",
           "|---|---|---|---|---|---|---|"]
    rows_json = []
    for path in sorted(glob.glob(os.path.join(SCREEN, "*", "judged.jsonl"))):
        model = os.path.basename(os.path.dirname(path))
        R = [json.loads(l) for l in open(path)]
        for label, subset, keep in (("unsafe", "plain", lambda r: r["kind"] == "f_plain"),
                                    ("refusal", "all", lambda r: True),
                                    ("unsafe (non-refusal on encoded)", "encoded",
                                     lambda r: r["kind"] == "f_enc")):
            key = label.split()[0]
            S = [r for r in R if keep(r) and len(r[key]) == 2]
            pairs = np.array([r[key] for r in S])
            cells = np.array([f"{r['kind']}:{r['enc']}" for r in S])
            if len(S) < 10:
                continue
            v, m = icc(pairs)
            lo, hi = boot(lambda p, c: icc(p)[0], pairs, cells)
            if len(np.unique(cells)) > 1:
                ms = meta_share(pairs, cells)
                mlo, mhi = boot(meta_share, pairs, cells)
                mtxt = f"{ms:.2f} ({mlo:.2f}-{mhi:.2f})"
            else:
                ms, mtxt = None, "-"
            out.append(f"| {model} | {label} | {subset} | {len(S)} | {m:.3f} "
                       f"| {v:.2f} ({lo:.2f}-{hi:.2f}) | {mtxt} |")
            rows_json.append(dict(model=model, label=label, subset=subset, n=len(S), rate=m,
                                  icc=v, icc_ci=[lo, hi], meta=ms))
    print("\n".join(out))
    json.dump(rows_json, open("heterogeneity.json", "w"), indent=1)


if __name__ == "__main__":
    main()
