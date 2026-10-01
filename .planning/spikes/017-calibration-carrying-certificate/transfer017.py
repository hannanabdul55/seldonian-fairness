"""Spike 017, stage E: does a calibration measured on one population carry to another?

A constraint that "ships with its calibration" can carry three things from the labelled
population to the one being certified: the judge's recall and false-alarm rate (Youden /
Rogan-Gladen), the PPI rectifier ``E[Y - f]``, or a recalibration map for ``p`` (Platt).
Each is valid only if the carried quantity is the same on both sides. Real shifts, with the
gold label (Qwen3Guard-4B's refusal field) known on both sides:

- ``source``   xstest <-> orbench, inside 013's over-refusal pool at step 0 (015's scores);
- ``training`` the over-refusal pool at step 0 -> the same prompts after 200 GRPO steps;
- ``pool``     over-refusal prompts (rate 0.20) <-> plain harmful prompts (rate 0.65).

For every shift and refusal wording: recall and false-alarm rate on both sides with a Fisher
exact test of equality, the three carried point estimates against the target's gold rate,
and a plasmode of the carried Youden bound (225 source labels, 2,000 target responses, 4,000
draws): how often it falls below the target's gold rate.

    ../../../.venv/bin/python transfer017.py        # CPU, about 1 min; writes transfer.md
"""
import collections
import json
import os

import numpy as np
from scipy.stats import fisher_exact

import cert017 as c
import plasmode017 as pm

HERE = os.path.dirname(os.path.abspath(__file__))
S017 = os.path.join(c.REPO, "results", "spikes", "017", "scores_pop.jsonl")
DELTA, REPS = 0.05, 4000


def load():
    """{(set, wording): (y, p, meta)} for c1_0 (015), c1_200 and c3_0 (017)."""
    d = collections.defaultdict(list)
    for line in open(pm.S015):
        r = json.loads(line)
        if r["task"] == "refusal" and r["variant"] == "rubric":
            d[("c1_0", r["wording"])].append((r["id"], r["ref"], r["p"], r["meta"]))
    for line in open(S017):
        r = json.loads(line)
        if r["set"].startswith("ref_"):
            d[(r["set"][4:], r["wording"])].append((r["id"], r["refusal"], r["p"], r["meta"]))
    out = {}
    for k, rows in d.items():
        rows = sorted(rows)
        out[k] = (np.array([r[1] for r in rows], dtype=float), np.array([r[2] for r in rows]),
                  np.array([r[3] for r in rows]))
    return out


def counts(y, f):
    return int((y * f).sum()), int(y.sum()), int(((1 - y) * f).sum()), int((1 - y).sum())


def rate_ci(k, n):
    return f"{k / n:.3f} ({float(c.cp_lower(k, n, 0.05)):.3f}-{float(c.cp_upper(k, n, 0.05)):.3f})" \
        if n else "-"


def carried(src, tgt):
    """Point estimates of the target rate from the target's judge scores and source labels."""
    ys, ps = src
    yt, pt = tgt
    fs, ft = (ps > 0.5).astype(float), (pt > 0.5).astype(float)
    tp, pos, fp, neg = counts(ys, fs)
    sens, fa = tp / pos, fp / neg
    rg = (ft.mean() - fa) / (sens - fa) if sens > fa else float("nan")
    ab = pm.platt_fit(pm.logit01(ps)[None, :], ys[None, :])
    platt = float(pm.platt_apply(ab, pm.logit01(pt)[None, :]).mean())
    ppi = float(ft.mean() + (ys - fs).mean())
    return rg, platt, ppi


def youden_plasmode(src, tgt, rng, n_cal=225, big_n=2000):
    ys, ps = src
    yt, pt = tgt
    i = rng.integers(0, len(ys), size=(REPS, n_cal))
    j = rng.integers(0, len(yt), size=(REPS, big_n))
    cal = c.rates(ys[i], (ps[i] > 0.5).astype(float))
    ft = (pt[j] > 0.5).astype(float)
    u = c.youden(None, None, ft, DELTA, cal=cal)
    return float((u < yt.mean()).mean()), float(np.median(u))


def main():
    d = load()
    rng = np.random.default_rng(17)
    sub = lambda k, m: tuple(a[d[k][2] == m] for a in d[k][:2])      # noqa: E731
    full = lambda k: d[k][:2]                                          # noqa: E731
    shifts = [("source: xstest -> orbench", lambda w: (sub(("c1_0", w), "xstest"), sub(("c1_0", w), "orbench"))),
              ("source: orbench -> xstest", lambda w: (sub(("c1_0", w), "orbench"), sub(("c1_0", w), "xstest"))),
              ("training: step 0 -> step 200", lambda w: (full(("c1_0", w)), full(("c1_200", w)))),
              ("pool: over-refusal -> harmful", lambda w: (full(("c1_0", w)), full(("c3_0", w)))),
              ("pool: harmful -> over-refusal", lambda w: (full(("c3_0", w)), full(("c1_0", w))))]
    L = ["# Spike 017, stage E: carrying a calibration across populations", "",
         "Compiled refusal judge (015's rubrics, label = p > 0.5), gold = Qwen3Guard-4B's refusal "
         "field. Rates with 90% Clopper-Pearson intervals. `p(same)` = Fisher exact test that "
         "the rate is equal on both sides. Carried estimates of the target's rate: `RG` = "
         "Rogan-Gladen with the source's recall and false-alarm rate; `Platt` = mean of the "
         "target's p after the source's recalibration map; `rect` = target's judged rate plus "
         "the source's PPI rectifier. `Youden bound` = plasmode of the carried bound (225 source "
         "labels, 2,000 target responses): share of draws below the target's gold rate "
         "(target <= 0.05) and its median.", ""]
    summary = []
    for name, pick in shifts:
        L += [f"## {name}", "",
              "| wording | recall: source | recall: target | p(same) | false alarms: source "
              "| false alarms: target | p(same) | target gold rate | target judged rate | RG | Platt "
              "| rect | Youden bound: miss, median |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for w in range(6):
            src, tgt = pick(w)
            (ys, ps), (yt, pt) = src, tgt
            a = counts(ys, (ps > 0.5).astype(float))
            b = counts(yt, (pt > 0.5).astype(float))
            p_s = fisher_exact([[a[0], a[1] - a[0]], [b[0], b[1] - b[0]]])[1]
            p_f = fisher_exact([[a[2], a[3] - a[2]], [b[2], b[3] - b[2]]])[1]
            rg, platt, ppi = carried(src, tgt)
            miss, med = youden_plasmode(src, tgt, rng)
            truth = yt.mean()
            L.append(f"| {w} | {rate_ci(a[0], a[1])} | {rate_ci(b[0], b[1])} | {p_s:.3g} "
                     f"| {rate_ci(a[2], a[3])} | {rate_ci(b[2], b[3])} | {p_f:.3g} | {truth:.3f} "
                     f"| {(pt > 0.5).mean():.3f} | {rg:.3f} | {platt:.3f} | {ppi:.3f} "
                     f"| {miss:.3f}, {med:.3f} |")
            summary.append(dict(shift=name, wording=w, p_sens=p_s, p_fa=p_f, truth=float(truth),
                                rg=rg, platt=platt, rect=ppi, youden_miss=miss, youden_median=med,
                                sens_src=a[0] / a[1], sens_tgt=b[0] / b[1],
                                fa_src=a[2] / a[3], fa_tgt=b[2] / b[3]))
        L.append("")
    L += ["## Summary", "", "| shift | wordings with recall different (p < 0.05) | with false alarms "
          "different | mean abs error: RG | Platt | rect | wordings where the carried Youden bound "
          "misses > 0.05 |", "|---|---|---|---|---|---|---|"]
    for name, _ in shifts:
        s = [r for r in summary if r["shift"] == name]
        err = lambda k: np.nanmean([abs(r[k] - r["truth"]) for r in s])      # noqa: E731
        L.append(f"| {name} | {sum(r['p_sens'] < 0.05 for r in s)}/6 | {sum(r['p_fa'] < 0.05 for r in s)}/6 "
                 f"| {err('rg'):.3f} | {err('platt'):.3f} | {err('rect'):.3f} "
                 f"| {sum(r['youden_miss'] > 0.05 for r in s)}/6 |")
    json.dump(summary, open(os.path.join(HERE, "transfer.json"), "w"), indent=1)
    open(os.path.join(HERE, "transfer.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
