"""Spike 017, stage A: is the human sheet an i.i.d. sample of the scored population?

It is not, and this script says by how much. ``scripts/screen_labels.py`` drew, per model x
encoding cell, 4 responses the 0.6B guard flagged and 2 it cleared (3 + 3 on plain prompts).
That is a stratified sample of the screen's 4,800 harmful-prompt responses with known
inclusion probabilities, so design weights exist. This module rebuilds the strata and the
weights and compares, for quantities known on the whole population, the exact value, the
unweighted sheet mean and the design-weighted (Horvitz-Thompson) mean.

A stratum with fewer than 2 definite labels is merged with the other flag stratum of its
cell (two strata: one whose single response drew a `?`, one with a single label).

    ../../../.venv/bin/python design017.py        # CPU, seconds; writes design.md
"""
import collections
import json
import os

import numpy as np

import score017 as s

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = os.path.join(s.SCREEN, "labels")
S015 = os.path.join(s.REPO, "results", "spikes", "015", "scores.jsonl")


def load():
    """
    Population and sheet. Returns a dict of arrays over the 4,800 responses (``ids``,
    ``model``, ``enc``, ``g`` = 0.6B flag, ``ans`` = not refused, ``h`` = stratum index,
    ``lab`` = index into the sheet or -1) and over the 225 definite labels (``pop_idx``,
    ``y`` = human h or c, ``label``, ``sid``), plus ``N_h``, ``n_h``.
    """
    pop = s.harm_pop()
    key = {r["id"]: r for r in s.r15.read_jsonl(os.path.join(LABELS, "key.jsonl"))}
    human = {r["id"]: r["label"] for r in s.r15.read_jsonl(os.path.join(LABELS, "labels_ah.jsonl"))}
    pos = {r["id"]: i for i, r in enumerate(pop)}
    sheet = sorted((sid, pos[key[sid]["conversation_id"]], lab) for sid, lab in human.items()
                   if lab in "hcng")
    cell = [(r["model"], r["enc"]) for r in pop]
    raw = [(m, e, r["unsafe06"]) for (m, e), r in zip(cell, pop)]
    n_raw = collections.Counter(raw[i] for _, i, _ in sheet)
    merged = {c for c in set(cell) if min(n_raw[(c[0], c[1], 0)], n_raw[(c[0], c[1], 1)]) < 2
              and (c[0], c[1], 0) in set(raw) and (c[0], c[1], 1) in set(raw)}
    strat = [(m, e, -1) if (m, e) in merged else (m, e, g) for (m, e, g) in raw]
    names = sorted(set(strat))
    idx = {k: i for i, k in enumerate(names)}
    h = np.array([idx[k] for k in strat])
    lab = -np.ones(len(pop), dtype=int)
    for j, (_, i, _) in enumerate(sheet):
        lab[i] = j
    pi = np.array([i for _, i, _ in sheet])
    return dict(ids=[r["id"] for r in pop], model=np.array([r["model"] for r in pop]),
                enc=np.array([r["enc"] for r in pop]),
                g=np.array([r["unsafe06"] for r in pop], dtype=float),
                ans=np.array([r["not_refused"] for r in pop], dtype=float),
                h=h, strata=names, merged=sorted(merged), lab=lab, pop_idx=pi,
                y=np.array([float(l in "hc") for _, _, l in sheet]),
                label=[l for _, _, l in sheet], sid=[sid for sid, _, _ in sheet],
                N_h=np.bincount(h, minlength=len(names)).astype(float),
                n_h=np.bincount(h[pi], minlength=len(names)).astype(float))


def ht(d, x_sheet):
    """Design-weighted mean of a sheet variable and its stratified standard error (with fpc)."""
    hs = d["h"][d["pop_idx"]]
    big_w = d["N_h"] / d["N_h"].sum()
    est, var = 0.0, 0.0
    for k in range(len(d["N_h"])):
        x = x_sheet[hs == k]
        est += big_w[k] * x.mean()
        if len(x) > 1:
            var += big_w[k] ** 2 * (1 - len(x) / d["N_h"][k]) * x.var(ddof=1) / len(x)
    return float(est), float(np.sqrt(var))


def weights(d):
    hs = d["h"][d["pop_idx"]]
    return d["N_h"][hs] / d["n_h"][hs]


def scores015(d, wording=0, variant="rubric"):
    """015's compiled-judge P(Yes) on the sheet items, aligned with ``d['sid']``."""
    p = {}
    for line in open(S015):
        r = json.loads(line)
        if r["task"] == "harm" and r["variant"] == variant and r["wording"] == wording:
            p[r["id"]] = r["p"]
    return np.array([p[i] for i in d["sid"]])


def main():
    d = load()
    w = weights(d)
    n, big_n = len(d["y"]), len(d["ids"])
    pi = d["pop_idx"]
    p0 = scores015(d)
    f0 = (p0 > 0.5).astype(float)
    L = ["# Spike 017, stage A: the sheet is a stratified sample, not an i.i.d. one", "",
         f"Population: {big_n} responses (6 models x up to 7 encodings). Sheet: {n} definite "
         f"labels in {len(d['strata'])} strata (model x encoding x 0.6B flag; "
         f"{len(d['merged'])} cells merged across the flag: {', '.join('/'.join(c) for c in d['merged'])}).",
         f"Labels per stratum: {dict(sorted(collections.Counter(d['n_h'].astype(int)).items()))}. "
         f"Weights N_h / n_h run {w.min():.1f} to {w.max():.1f} (median {np.median(w):.1f}); Kish "
         f"design effect {n * (w ** 2).sum() / w.sum() ** 2:.2f}, so the sheet is worth about "
         f"{w.sum() ** 2 / (w ** 2).sum():.0f} equal-weight labels.", "",
         "| quantity | population, exact | sheet, unweighted | sheet, design-weighted (se) |",
         "|---|---|---|---|"]
    for name, pop_v, sheet_v in (("0.6B guard flags it", d["g"], d["g"][pi]),
                                 ("not refused (guard)", d["ans"], d["ans"][pi]),
                                 ("plain prompt", (d["enc"] == "plain").astype(float),
                                  (d["enc"][pi] == "plain").astype(float)),
                                 ("model is 14B", (d["model"] == "14B").astype(float),
                                  (d["model"][pi] == "14B").astype(float))):
        e, se = ht(d, sheet_v)
        L.append(f"| {name} | {pop_v.mean():.3f} | {sheet_v.mean():.3f} | {e:.3f} ({se:.3f}) |")
    for name, v in (("human: harmful (h or c)", d["y"]),
                    ("compiled judge flags it (015 canonical, p > 0.5)", f0),
                    ("compiled judge, mean p", p0),
                    ("rectifier, mean of (human - judge label)", d["y"] - f0),
                    ("rectifier, mean of (human - p)", d["y"] - p0)):
        e, se = ht(d, v)
        L.append(f"| {name} | see stage D | {v.mean():.3f} | {e:.3f} ({se:.3f}) |")
    neg = d["y"] == 0
    a = neg & (d["ans"][pi] == 1)
    r = neg & (d["ans"][pi] == 0)
    items, _ = s.r15.harm_items()
    guard = {it["id"]: it["guard"] for it in items}
    g4 = np.array([float(guard[i]) for i in d["sid"]])
    L += ["", "False alarms on human-negative responses, by whether the guard says the response "
          "refused (spike 006's answer-rate-aware correction assumes they fall on answers only):", "",
          "| judge | responses | n | flagged | rate, unweighted | rate, design-weighted |",
          "|---|---|---|---|---|---|"]
    for jname, fj in (("compiled rubric (015 canonical)", f0), ("Qwen3Guard-4B (bake-off)", g4)):
        for name, m in (("answered", a), ("refused", r), ("all negatives", neg)):
            L.append(f"| {jname} | {name} | {int(m.sum())} | {int(fj[m].sum())} | {fj[m].mean():.3f} "
                     f"| {(w[m] * fj[m]).sum() / w[m].sum():.3f} |")
    pos = d["y"] == 1
    L += ["", f"Human positives: {int(pos.sum())}, weights {np.round(w[pos], 1).tolist()}, "
          f"compiled judge p {np.round(p0[pos], 3).tolist()}, answered {d['ans'][pi][pos].tolist()}, "
          f"0.6B flag {d['g'][pi][pos].tolist()}."]
    open(os.path.join(HERE, "design.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
