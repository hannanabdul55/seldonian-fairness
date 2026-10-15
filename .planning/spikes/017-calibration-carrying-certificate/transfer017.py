"""Spike 017, stage E: does a calibration measured on one population carry to another?

A constraint that "ships with its calibration" can carry three things from the labelled
population to the one being certified: the judge's recall and false-alarm rate (Youden /
Rogan-Gladen), the PPI rectifier ``E[Y - f]``, or a recalibration map for ``p`` (Platt).
Each is valid only if the carried quantity is the same on both sides. Real shifts, with the
gold label (Qwen3Guard-4B's refusal field) known on both sides:

- ``source``   xstest <-> orbench, inside 013's over-refusal pool at step 0 (015's scores);
- ``training`` the over-refusal pool at step 0 -> the same prompts after 200 GRPO steps;
- ``training, constrained`` (E8, 2026-10-01) step 0 -> spike 014's checkpoints at steps 100
  and 200, where ``LagrangianReward`` drove the refusal label;
- ``pool``     over-refusal prompts (rate 0.20) <-> plain harmful prompts (rate 0.65).
- ``training, constrained (014 s1/s2)`` (the paper's R6): the same as E8 for two more seeds of
  014's run. Appended after E8's shifts, so the shared random stream gives every earlier row
  unchanged; a seed enters only when all six wordings of both steps are scored on all 500
  items. With a new seed present, a section "Constrained runs, one row per seed and step"
  follows the summary (and ``transfer_seeds.json``): the quantities of the paper's section 9.2
  and Appendix B, per run.

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

R6_TAGS = ("s1", "s2")                          # spike 014 seeds added for R6; s0 is E8's ``c1_014_*``
Z_OVER = 2.0                                    # scripts/validity_recount.py: over if miss > delta + 2 se

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
              ("pool: harmful -> over-refusal", lambda w: (full(("c3_0", w)), full(("c1_0", w)))),
              # E8 (2026-10-01): spike 014's Lagrangian-constrained checkpoints, same prompts and ids
              ("training, constrained (014): step 0 -> step 100", lambda w: (full(("c1_0", w)), full(("c1_014_100", w)))),
              ("training, constrained (014): step 0 -> step 200", lambda w: (full(("c1_0", w)), full(("c1_014_200", w))))]
    shifts = [(n, f) for n, f in shifts if all(k in d for k in (("c1_0", 0),)) and
              (not n.startswith("training, constrained") or (f"c1_014_{n[-3:]}", 0) in d)]
    # R6: two more seeds of 014's constrained run, appended after every earlier shift (same rng stream)
    n_src = len(d[("c1_0", 0)][0])
    seeds = [t for t in R6_TAGS
             if all(len(d.get((f"c1_014{t}_{st}", w), ((),))[0]) == n_src for st in (100, 200) for w in range(6))]
    for t in R6_TAGS:
        if t not in seeds and any(k[0].startswith(f"c1_014{t}_") for k in d):
            print(f"R6 seed {t}: scores incomplete (not all 6 wordings x 2 steps x {n_src} items), left out")
    for t in seeds:
        for st in (100, 200):
            shifts.append((f"training, constrained (014 {t}): step 0 -> step {st}",
                           lambda w, key=f"c1_014{t}_{st}": (full(("c1_0", w)), full((key, w)))))
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
    if seeds:
        L += per_seed(summary, ["s0"] + seeds)
    json.dump(summary, open(os.path.join(HERE, "transfer.json"), "w"), indent=1)
    open(os.path.join(HERE, "transfer.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def guard_side(tag, step):
    """Pool rate (all K samples), the reference's, and Appendix B's hybrid-shape share of the guard's flags."""
    import re
    import sys
    sys.path.insert(0, os.path.join(c.REPO, "scripts"))
    import refusal_sheet_build as rb
    hyb = re.compile(rb.HYB, re.I)

    def side(g, j, st, role):
        G, J = rb.load(g, j, st, role)
        y = [v for i in G for v in J[i]]
        flags = [(t, v) for i in G if i not in rb.EXAMPLE_PROMPTS for t, v in zip(G[i], J[i]) if v]
        share = np.mean([bool(hyb.search(t.strip())) and len(t.split()) > rb.HYB_MIN_WORDS for t, _ in flags])
        return float(np.mean(y)), float(share), len(flags)
    ref = side("results/spikes/013/gen_full.jsonl", "results/spikes/013/judged_full.jsonl", 0, "cov")
    tgt = side(f"results/spikes/014/gen_{tag}.jsonl", f"results/spikes/014/judged_{tag}.jsonl", step, "cand")
    return ref, tgt


def per_seed(summary, tags):
    """One row per run and step: the numbers section 9.2 and Appendix B quote for the one run."""
    se = np.sqrt(DELTA * (1 - DELTA) / REPS)
    rows, out = [], []
    for t in tags:
        for st in (100, 200):
            name = f"training, constrained (014{'' if t == 's0' else ' ' + t}): step 0 -> step {st}"
            s = sorted((r for r in summary if r["shift"] == name), key=lambda r: r["wording"])
            m = [r["youden_miss"] for r in s]
            (ref_rate, ref_share, ref_flags), (rate, share, flags) = guard_side(t, st)
            w0 = s[0]
            rec = dict(seed=t, step=st, pool_rate=rate, pool_rate_ref=ref_rate, moved=rate - ref_rate,
                       gold_rate_500=w0["truth"], recall_w0_src=w0["sens_src"], recall_w0_tgt=w0["sens_tgt"],
                       p_recall_w0=w0["p_sens"], recall_differs=int(sum(r["p_sens"] < 0.05 for r in s)),
                       youden_misses=m, youden_over=int(sum(x > DELTA + Z_OVER * se + 1e-12 for x in m)),
                       youden_unresolved=int(sum(DELTA + 1e-12 < x <= DELTA + Z_OVER * se + 1e-12 for x in m)),
                       youden_above_005=int(sum(x > DELTA for x in m)),
                       rg_mae=float(np.nanmean([abs(r["rg"] - r["truth"]) for r in s])),
                       hybrid_share_of_flags=share, hybrid_share_ref=ref_share, guard_flags=flags)
            out.append(rec)
            rows.append(f"| {t} | {st} | {rate:.3f} ({rate - ref_rate:+.3f}) | {w0['truth']:.3f} "
                        f"| {w0['sens_src']:.2f} -> {w0['sens_tgt']:.2f} ({w0['p_sens']:.2g}) | {rec['recall_differs']}/6 "
                        f"| {rec['youden_over']}/6 over, {rec['youden_unresolved']}/6 unresolved "
                        f"({', '.join(f'{x:.3f}' for x in m)}) | {rec['rg_mae']:.3f} "
                        f"| {share:.0%} of {flags} (reference {ref_share:.0%}) |")
    json.dump(out, open(os.path.join(HERE, "transfer_seeds.json"), "w"), indent=1)
    return ["", "## Constrained runs, one row per seed and step (R6)", "",
            f"Pool rate = 014's guard refusal rate on all K samples of the 500 C1 prompts (reference: 013's "
            f"8 samples, {out[0]['pool_rate_ref']:.3f}). Gold rate and recall are on the 500 scored items. "
            f"`over` = carried Youden bound's miss > {DELTA} + {Z_OVER:g} se ({REPS} draws), the rule of "
            "`scripts/validity_recount.py`; `unresolved` = above 0.05 but within it. Hybrid share = "
            "the guard's flags with a refusal opener and more than 40 words (Appendix B; frame of "
            "`scripts/refusal_sheet_build.py`, its 10 example prompts left out).", "",
            "| seed | step | pool rate (moved) | gold rate, 500 items | recall, wording 0 (Fisher p) | recall "
            "differs | carried Youden bound (misses, wordings 0-5) | RG error | hybrid share of the guard's flags |",
            "|---|---|---|---|---|---|---|---|---|"] + rows


if __name__ == "__main__":
    main()
