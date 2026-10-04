"""Figures for the certification paper (plan step P12).

    .venv/bin/python scripts/paper_figures.py            # all figures -> reports/figs/
    .venv/bin/python scripts/paper_figures.py fig3 fig6  # some

Each figure is written as a PDF (for the paper), a PNG (to look at) and a CSV holding every number
drawn (the table view). Style follows the dataviz reference palette: one axis per plot, thin marks,
solid hairline grid, a legend for two or more series with a few direct labels, text in ink and never
in a series colour. Colours are the palette's first three categorical slots, which the validator
passes for every pair in light mode (`validate_palette.js "#2a78d6,#eb6834,#1baf7a" --mode light
--pairs all`); aqua is under 3:1 on the surface, so every chart that uses it carries direct labels and
its CSV.
"""
import collections
import csv
import json
import lzma
import os
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SP = os.path.join(ROOT, ".planning", "spikes")
OUT = os.path.join(ROOT, "reports", "figs")
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, CONTEXT = "#2a78d6", "#eb6834", "#1baf7a", "#c3c2b7"
DOT = dict(markersize=7.5, markeredgecolor=SURFACE, markeredgewidth=1.4, linestyle="none")

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8.5, "pdf.fonttype": 42,
})


def style(ax, grid="x"):
    ax.grid(axis=grid, color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    if grid == "x":
        ax.spines["left"].set_visible(False)


def heading(fig, title, subtitle, legend=None, ncol=3):
    """Left-aligned title and subtitle wrapped to the figure, then the legend; returns the top of the plot area."""
    W, H = fig.get_size_inches()
    tt, ss = textwrap.fill(title, int(W * 10.4)), textwrap.fill(subtitle, int(W * 14.0))
    y = 1 - 0.08 / H
    fig.text(0.012, y, tt, fontsize=11, fontweight="bold", color=INK, va="top", ha="left", linespacing=1.2)
    y -= (0.205 * (tt.count("\n") + 1) + 0.04) / H
    fig.text(0.012, y, ss, fontsize=8.8, color=INK2, va="top", ha="left", linespacing=1.3)
    y -= (0.17 * (ss.count("\n") + 1) + 0.05) / H
    if legend:
        fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(0.004, y), ncol=ncol, columnspacing=1.3, handletextpad=0.2, labelspacing=0.3)
        y -= (0.215 * int(np.ceil(len(legend) / ncol)) + 0.10) / H
    return y - 0.10 / H


def save(fig, name, header, rows):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=200)
    plt.close(fig)
    with open(os.path.join(OUT, f"{name}.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(header); w.writerows(rows)
    print(f"{name}: {len(rows)} rows")


def cp(x, n, a):
    return (beta.ppf(a, x, n - x + 1) if x > 0 else 0.0, beta.ppf(1 - a, x + 1, n - x) if x < n else 1.0)


# ---------------------------------------------------------------------------------------- fig 1
def fig1():
    """Miss rate over delta for every bound in the paper's Table 2."""
    S = json.load(open(os.path.join(ROOT, "results", "paper", "stratppi.json")))["rows"]
    mid = lambda r: r["part"] == "A" and 0.05 <= r["truth"] <= 0.95 and r["delta"] == 0.05
    rng = lambda arm, part: (min(r["miss"] for r in S if r["arm"] == arm and (mid(r) if part == "A" else r["part"] == "B")),
                             max(r["miss"] for r in S if r["arm"] == arm and (mid(r) if part == "A" else r["part"] == "B")))
    b_pub = [r["miss"] for r in S if r["part"] == "B" and r["arm"] in ("StratPPI, K=5", "StratPPI, K=10")]
    b_boot = [r["miss"] for r in S if r["part"] == "B" and r["arm"].startswith("StratPPI, bootstrap-t")]
    rows = [  # group, bound and setting, delta, lowest miss, highest miss, source
        ("exact", "Clopper-Pearson · harm labels at 3.4%", 0.10, 0.067, 0.084, "training paper 6.2"),
        ("exact", "Clopper-Pearson · refusal pools", 0.05, 0.000, 0.051, "017 B6, P14"),
        ("exact", "Bentkus · harm labels at 3.4%", 0.10, 0.023, 0.035, "training paper 6.2"),
        ("exact", "betting mixture · harm labels at 3.4%", 0.10, 0.007, 0.016, "training paper 6.2"),
        ("exact", "Hoeffding, Anderson · harm labels at 3.4%", 0.10, 0.000, 0.000, "training paper 6.2"),
        ("approximate", "stratified Wilson-type (b1w) · mid-rate labels", 0.05, 0.023, 0.054, "013 validity_H8"),
        ("approximate", "b1w · mid-rate labels", 0.10, 0.069, 0.097, "013 validity_H8"),
        ("approximate", "b1w · label pushed by training", 0.05, 0.011, 0.023, "014"),
        ("approximate", "b1w · design-weighted labelling sheet", 0.05, 0.001, 0.059, "017 Results 4"),
        ("approximate", "PPI++, bootstrap-t limit", 0.05, 0.000, 0.053, "017 B6, P14"),
        ("approximate", "StratPPI estimator, bootstrap-t · reference-rate strata", 0.05, *rng("S2 + StratPPI, bootstrap-t", "A"), "P14"),
        ("approximate", "StratPPI estimator, bootstrap-t · judge strata", 0.05, min(b_boot), max(b_boot), "P14"),
        ("approximate", "cluster bootstrap-t by user task · AgentDojo", 0.05, 0.020, 0.060, "020 plasmode"),
        ("fails", "Student-t · harm labels at 3.4%, n up to 800", 0.10, 0.115, 0.184, "training paper 6.2"),
        ("fails", "StratPPI as published · reference-rate strata", 0.05, *rng("S2 + StratPPI", "A"), "P14"),
        ("fails", "StratPPI as published · judge strata", 0.05, min(b_pub), max(b_pub), "P14"),
        ("fails", "PPI++, normal limit", 0.05, 0.061, 0.241, "017 B6, P14"),
        ("fails", "two-way bootstrap · AgentDojo", 0.05, 0.000, 0.170, "020 plasmode"),
        ("fails", "Clopper-Pearson over pairs · AgentDojo (crossed design)", 0.05, 0.048, 0.275, "020 plasmode"),
        ("fails", "b1w at 1-2% rates, 100 safety prompts", 0.05, 0.238, 0.443, "013 validity_H8"),
        ("fails", "stratified sheet read as a random sample", 0.05, 0.005, 0.980, "017 Results 4"),
        ("fails", "the judge's rate alone", 0.05, 1.000, 1.000, "017 B6"),
    ]
    col = {"exact": BLUE, "approximate": AQUA, "fails": ORANGE}
    names = {"exact": "exact at every sample size", "approximate": "approximate, holds where tested", "fails": "does not hold"}
    fig, ax = plt.subplots(figsize=(8.6, 8.2))
    LEFT = 0.04
    y, ticks, labels = 0, [], []
    for g in ("exact", "approximate", "fails"):
        ax.text(-0.02, y - 0.05, names[g].upper(), transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=7.6,
                color=MUTED, fontweight="semibold")
        y += 1
        for _, label, d, lo, hi, _ in [r for r in rows if r[0] == g]:
            a, b = max(lo / d, LEFT), max(hi / d, LEFT)
            ax.plot([a, b], [y, y], color=col[g], linewidth=3, solid_capstyle="round", alpha=0.45)
            ax.plot([b], [y], marker="o", color=col[g], clip_on=False, zorder=5, **DOT)
            if hi == 0:
                ax.text(LEFT * 1.18, y, "no miss", fontsize=8, color=INK2, va="center", ha="left")
            ticks.append(y); labels.append(f"{label}  (δ {d:.2f})")
            y += 1
        y += 0.5
    ax.axvline(1, color=INK2, linewidth=1.0)
    ax.text(1.07, -0.62, "the level the bound claims", ha="left", va="center", fontsize=8, color=INK2)
    ax.set_xscale("log"); ax.set_xlim(LEFT, 28); ax.set_ylim(y - 0.6, -1.0)
    ax.set_xticks([0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20]); ax.set_xticklabels(["0.05", "0.1", "0.2", "0.5", "1", "2", "5", "10", "20"])
    ax.set_yticks(ticks); ax.set_yticklabels(labels)
    ax.set_xlabel("miss rate ÷ δ  (1 = the level the bound claims)")
    style(ax, "x")
    ax.minorticks_off()
    h = [plt.Line2D([], [], marker="o", color=col[g], label=names[g], **DOT) for g in col]
    top = heading(fig, "Which bounds hold their level", "Each row is a bound in one setting. The bar runs from its best cell to its worst and the dot is the "
                  "worst; at or left of the line the bound holds. The approximate bounds' worst cells sit at 0.5 to 1.2 times the level, "
                  "within about two Monte Carlo standard errors of it.", h)
    fig.subplots_adjust(left=0.495, right=0.975, top=top, bottom=0.075)
    save(fig, "fig1_validity", ["group", "bound and setting", "delta", "lowest miss", "highest miss", "highest miss / delta", "source"],
         [(g, l, d, f"{lo:.3f}", f"{hi:.3f}", f"{hi / d:.2f}", s) for g, l, d, lo, hi, s in rows])


# ---------------------------------------------------------------------------------------- fig 2
def fig2():
    """Realised gain of reference-rate strata against the reference's ICC, with the pre-flight curve."""
    def ess_rows(rows, n_s=200, delta=0.1):
        out = {}
        for r in rows:
            if r["bound"] != "b1w" or r["n_s"] != n_s or r["delta"] != delta:
                continue
            k = (r["env"], r["cand"], r.get("seed", 0))
            if r["arm"] == "R":
                out.setdefault(k, {})["R"] = r["width"]
            elif r["arm"] == "S2" and r["k"] == 8 and r["H"] == 8:
                out.setdefault(k, {}).update(S=r["width"], icc=r["pf_icc_ref"], rate=r.get("rate", r["truth"]))
        return {k: dict(v, ess=(v["R"] / v["S"]) ** 2) for k, v in out.items() if "R" in v and "S" in v}
    bandit = ess_rows(json.load(lzma.open(os.path.join(ROOT, "results", "spikes", "013", "bandit_plasmode.json.xz"))))
    real = ess_rows(json.load(lzma.open(os.path.join(ROOT, "results", "spikes", "013", "real_plasmode_randomties.json.xz"))))
    p14 = json.load(open(os.path.join(SP, "014-pushed-label-stratification", "plasmode.json")))
    pushed = ess_rows(p14 if isinstance(p14, list) else p14["rows"])
    names = {"C1:refusal": "over-refusal (17%)", "C3:refusal": "refusal of harmful requests (66%)", "C2:unsafe": "non-refusal, encoded (9%)",
             "C2:refusal": "refusal, encoded (94%)"}
    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    x = np.linspace(0.0, 0.93, 200)
    rel = 8 * x / (1 + 7 * x)
    note = dict(fontsize=8, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.7, shrinkA=1, shrinkB=4))
    for rho in (1.0, 0.8):
        ax.plot(x, 1 / (1 - x * rho ** 2 * rel * 0.96), color=INK2 if rho == 1 else MUTED, linewidth=1.5)
    ax.annotate("pre-flight formula, reference and\ntrained rates fully correlated", xy=(0.775, 3.55), xytext=(0.30, 4.55), ha="left", va="center", **note)
    ax.annotate("pre-flight formula,\ncorrelation 0.8", xy=(0.90, 2.19), xytext=(0.80, 1.35), ha="left", va="center", **note)
    bx = [v["icc"] for v in bandit.values()]; by = [v["ess"] for v in bandit.values()]
    ax.plot(bx, by, marker="o", color=CONTEXT, markersize=4.5, markeredgecolor=SURFACE, markeredgewidth=0.6, linestyle="none", zorder=2)
    table = [("bandit", k[0], k[1], f"{v['icc']:.3f}", f"{v['ess']:.2f}") for k, v in sorted(bandit.items())]
    place = {"C1:refusal": (0.50, 3.45, "right"), "C3:refusal": (0.825, 5.33, "right"), "C2:unsafe": (0.33, 2.15, "right"), "C2:refusal": (0.50, 0.93, "left")}
    for (env, cand, _), v in sorted(real.items()):
        if cand != "step200" or env not in names:
            continue
        ax.plot([v["icc"]], [v["ess"]], marker="o", color=BLUE, zorder=4, **DOT)
        tx, ty, ha = place[env]
        ax.annotate(names[env], xy=(v["icc"], v["ess"]), xytext=(tx, ty), ha=ha, va="center", **note)
        table.append(("real, side effect", env, cand, f"{v['icc']:.3f}", f"{v['ess']:.2f}"))
    for (env, cand, _), v in sorted(pushed.items()):
        if cand == "step200" and env.endswith(":cand"):
            ax.plot([v["icc"]], [v["ess"]], marker="o", color=ORANGE, zorder=4, **DOT)
            ax.annotate("the same label when\ntraining pushes it", xy=(v["icc"], v["ess"]), xytext=(0.50, 2.85), ha="right", va="center", **note)
            table.append(("real, pushed", env, cand, f"{v['icc']:.3f}", f"{v['ess']:.2f}"))
    ax.set_xlim(0, 0.95); ax.set_ylim(0.8, 6.0)
    ax.set_xlabel("intraclass correlation of the reference model's label (share of variance between prompts)")
    ax.set_ylabel("effective safety samples vs a random split")
    style(ax, "y")
    h = [plt.Line2D([], [], marker="o", color=CONTEXT, label="synthetic bandit cells", markersize=4.5, markeredgecolor=SURFACE, markeredgewidth=0.6, linestyle="none"),
         plt.Line2D([], [], marker="o", color=BLUE, label="real label, side effect of training", **DOT),
         plt.Line2D([], [], marker="o", color=ORANGE, label="real label, targeted by training", **DOT)]
    top = heading(fig, "The gain from reference-rate strata can be predicted before labelling",
                  "8 strata of an 8-sample reference rate, 200 safety prompts, δ 0.1, 200 training steps. Labels with rates of 1-2% are left out: no approximate bound holds there.", h)
    fig.subplots_adjust(left=0.085, right=0.975, top=top, bottom=0.11)
    save(fig, "fig2_strata_ess", ["series", "label or environment", "checkpoint", "ICC_ref", "ESS"], table)


# ---------------------------------------------------------------------------------------- fig 3
def fig3():
    """AgentDojo: the per-pair bound against the clustered one, per pipeline."""
    c = json.load(open(os.path.join(SP, "020-agentdojo-injection-certificate", "cert.json")))
    P = sorted(c.items(), key=lambda kv: kv[1]["rate"])
    fig, ax = plt.subplots(figsize=(8.6, 8.2))
    table = []
    for y, (name, r) in enumerate(P):
        cl = max(r["t_user"], r["t_inj"])
        ax.plot([r["rate"], cl], [y, y], color=GRID, linewidth=1.6, zorder=1)
        ax.plot([r["rate"]], [y], marker="o", color=CONTEXT, zorder=2, **DOT)
        ax.plot([r["naive"]], [y], marker="o", color=BLUE, zorder=3, **DOT)
        ax.plot([cl], [y], marker="o", color=ORANGE, zorder=4, **DOT)
        table.append((name, r["n"], f"{r['rate']:.3f}", f"{r['naive']:.3f}", f"{r['t_user']:.3f}", f"{r['t_inj']:.3f}", f"{cl:.3f}",
                      "pass" if cl <= 0.05 else "no solution found"))
    ax.axvline(0.05, color=INK2, linewidth=1.0)
    ax.text(0.0535, len(P) - 0.15, "5% threshold", ha="left", va="center", fontsize=8, color=INK2)
    idx = {n: i for i, (n, _) in enumerate(P)}
    for n, txt in (("claude-3-5-sonnet-20241022", "the one pipeline that certifies"), ("Meta-SecAlign-70B", "2.2% raw, 10.4% once clustered:\n21 successes in 5 of 97 user tasks")):
        r = c[n]
        ax.annotate(txt, xy=(max(r["t_user"], r["t_inj"]), idx[n]), xytext=(0.135, idx[n] + (0.05 if "one" in txt else 0.85)), fontsize=8, color=INK2,
                    va="center", ha="left", arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.7, shrinkA=0, shrinkB=4))
    ax.set_xscale("log"); ax.set_xlim(0.008, 1.0); ax.set_ylim(-0.7, len(P) + 0.4)
    ax.set_xticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]); ax.set_xticklabels(["1%", "2%", "5%", "10%", "20%", "50%", "100%"]); ax.minorticks_off()
    ax.set_yticks(range(len(P))); ax.set_yticklabels([n for n, _ in P])
    ax.set_xlabel("targeted attack success rate (upper bounds at δ 0.05)")
    style(ax, "x")
    h = [plt.Line2D([], [], marker="o", color=k, label=l, **DOT) for k, l in ((CONTEXT, "observed rate"), (BLUE, "bound treating pairs as independent"),
                                                                                (ORANGE, "bound respecting the crossed design"))]
    top = heading(fig, "On AgentDojo's published runs, one pipeline in 28 certifies a 5% attack success rate",
                  "The usual bound treats every user-task and injection-task pair as independent. The clustered bound is the larger of two cluster "
                  "bootstraps, by user task and by injection task.", h)
    fig.subplots_adjust(left=0.43, right=0.97, top=top, bottom=0.07)
    save(fig, "fig6_agentdojo", ["pipeline", "pairs", "rate", "per-pair bound", "clustered by user task", "clustered by injection task", "certificate", "at 5%"], table)


# ---------------------------------------------------------------------------------------- fig 4
def fig4():
    """RoboDojo-RC: what 120 trials can certify."""
    b = json.load(open(os.path.join(SP, "019-external-trace-certificate", "bounds.json")))["per"]
    n = 120
    ks = np.arange(0, 13)
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    table = []
    for d, col, lab in ((0.05, BLUE, "δ 0.05"), (0.10, ORANGE, "δ 0.10")):
        up = [cp(int(k), n, d)[1] for k in ks]
        ax.plot(ks, up, color=col, linewidth=1.5, marker="o", markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=1.0, label=lab)
        ax.text(ks[-1] + 0.25, up[-1], lab, fontsize=8.5, color=INK2, va="center")
        table += [(lab, int(k), f"{u:.4f}") for k, u in zip(ks, up)]
    ax.axhline(0.05, color=INK2, linewidth=1.0)
    ax.text(12.9, 0.046, "5% threshold", fontsize=8, color=INK2, va="top", ha="right")
    names = {"claude-opus-5": "Claude Opus 5", "claude-opus-5-5": "Claude Opus 5.5", "gpt-6-astra": "GPT-6 Astra"}
    at = collections.defaultdict(list)
    for m, r in b.items():
        at[r["k"]].append(names.get(m, m))
        table.append((names.get(m, m), r["k"], f"{r['up05']:.4f}"))
    for k, ms in at.items():
        u = cp(k, n, 0.05)[1]
        ax.annotate(" and ".join(sorted(ms)) + f"\n{k} stops: bound {u:.3f}", xy=(k, u), xytext=(9.3, 0.082) if k > 5 else (0.2, 0.118), fontsize=8, color=INK2,
                    ha="left", va="center", arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.7, shrinkA=1, shrinkB=4))
    ax.set_xlim(-0.5, 13.0); ax.set_ylim(0, 0.17); ax.set_xticks(ks)
    ax.set_yticks([0, 0.05, 0.10, 0.15]); ax.set_yticklabels(["0%", "5%", "10%", "15%"])
    ax.set_xlabel("safety stops observed in 120 trials"); ax.set_ylabel("upper bound on the stop rate")
    style(ax, "y")
    h = [plt.Line2D([], [], marker="o", color=c_, label="upper bound at " + l_, markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=1.0, linewidth=1.5)
         for c_, l_ in ((BLUE, "δ 0.05"), (ORANGE, "δ 0.10"))]
    top = heading(fig, "What a 120-trial benchmark can certify",
                  "Clopper-Pearson upper bounds on the safety-stop rate. Nothing observed certifies 2.5%; two stops already miss a 5% claim at δ 0.05.", h)
    fig.subplots_adjust(left=0.10, right=0.86, top=top, bottom=0.13)
    save(fig, "fig5_robodojo", ["series or model", "stops", "upper bound"], table)


# ---------------------------------------------------------------------------------------- fig 5
def fig5():
    """A judge's calibration carried to another population: the carried bound's miss rate, and recall."""
    T = json.load(open(os.path.join(SP, "017-calibration-carrying-certificate", "transfer.json")))
    order = [("training: step 0 -> step 200", "200 training steps, label moved as a side effect"),
             ("training, constrained (014): step 0 -> step 100", "100 training steps that target the label"),
             ("training, constrained (014): step 0 -> step 200", "200 training steps that target the label"),
             ("source: orbench -> xstest", "prompt source: OR-Bench to XSTest"),
             ("source: xstest -> orbench", "prompt source: XSTest to OR-Bench"),
             ("pool: over-refusal -> harmful", "prompt pool: benign to harmful"),
             ("pool: harmful -> over-refusal", "prompt pool: harmful to benign")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.8, 4.4), sharey=True, gridspec_kw=dict(width_ratios=[1.3, 1]))
    table = []
    small = dict(DOT, markersize=6.5, markeredgewidth=1.1)
    for y, (key, lab) in enumerate(order):
        rows = [r for r in T if r["shift"] == key]
        xs = np.array([r["youden_miss"] for r in rows])
        jit = np.linspace(-0.3, 0.3, len(xs))[np.argsort(np.argsort(xs))]
        a1.plot(xs, y + jit, marker="o", color=BLUE, clip_on=False, zorder=4, **small)
        src, tgt = np.mean([r["sens_src"] for r in rows]), np.mean([r["sens_tgt"] for r in rows])
        a2.plot([src, tgt], [y, y], color=GRID, linewidth=1.6, zorder=1)
        a2.plot([src], [y], marker="o", color=CONTEXT, zorder=2, **DOT)
        a2.plot([tgt], [y], marker="o", color=BLUE, zorder=3, **DOT)
        table += [(lab, r["wording"], f"{r['youden_miss']:.3f}", f"{r['sens_src']:.3f}", f"{r['sens_tgt']:.3f}") for r in rows]
    a1.axvline(0.05, color=INK2, linewidth=1.0)
    a1.text(0.065, -0.72, "δ 0.05", fontsize=8, color=INK2, va="center", ha="left")
    a1.set_xlim(-0.05, 1.03); a1.set_ylim(len(order) - 0.5, -1.0)
    a1.set_yticks(range(len(order))); a1.set_yticklabels([l for _, l in order])
    a1.set_xlabel("miss rate of the carried bound")
    a2.set_xlim(-0.02, 0.5); a2.set_xlabel("judge's recall (mean of six wordings)")
    for a in (a1, a2):
        style(a, "x")
    h = [plt.Line2D([], [], marker="o", color=CONTEXT, label="recall where the calibration was measured", **DOT),
         plt.Line2D([], [], marker="o", color=BLUE, label="where it was applied (both panels)", **DOT)]
    top = heading(fig, "A judge's calibration does not carry, least of all across training that targets the label",
                  "A rubric judge's recall and false-alarm rate are measured on one population and used to bound the refusal rate of another. "
                  "Left: how often that bound falls below the truth, one dot per wording of the judge's rubric. Right: what happened to the judge's recall.", h)
    fig.subplots_adjust(left=0.375, right=0.98, top=top, bottom=0.13, wspace=0.12)
    save(fig, "fig4_carrying", ["shift", "judge wording", "carried bound miss rate", "recall where measured", "recall where applied"], table)


# ---------------------------------------------------------------------------------------- fig 6
def fig6():
    """StratPPI as published and with a bootstrap-t limit, beside this paper's bounds: miss against efficiency."""
    S = json.load(open(os.path.join(ROOT, "results", "paper", "stratppi.json")))["rows"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 4.9))
    table = []
    # panel A: reference-rate strata, mid-rate labels, delta 0.05
    A = [r for r in S if r["part"] == "A" and r["delta"] == 0.05 and 0.05 <= r["truth"] <= 0.95]
    armsA = (("S2 + b1w (this paper)", BLUE, "stratified Wilson-type bound"), ("S2 + StratPPI", ORANGE, "StratPPI as published"),
             ("S2 + StratPPI, bootstrap-t", AQUA, "StratPPI estimator, bootstrap-t limit"))
    for arm, col, lab in armsA:
        for src, env, n in sorted({(r["src"], r["env"], r["n"]) for r in A}):
            cell = [r for r in S if r["part"] == "A" and (r["src"], r["env"], r["n"], r["delta"]) == (src, env, n, 0.05)]
            last = max(r["step"] for r in cell)
            base = next(r for r in cell if r["arm"] == "random + pooled Wilson (R)" and r["step"] == last)
            x = max(r["miss"] for r in cell if r["arm"] == arm)
            me = next(r for r in cell if r["arm"] == arm and r["step"] == last)
            ess = (base["excess"] / me["excess"]) ** 2
            a1.plot([x], [ess], marker="o", color=col, **DOT)
            table.append(("reference-rate strata", lab, f"{env} {'pushed' if src == '014' else ''} n={n}", f"{x:.3f}", f"{ess:.2f}"))
    a1.set_title("Reference model's rate as stratifier and predictor", loc="left", fontsize=9, color=INK2, pad=6)
    # panel B: judge strata
    B = [r for r in S if r["part"] == "B"]
    base = {(r["env"], str(r["rate"]), r["n"]): r["excess"] for r in B if r["arm"].startswith("labels alone")}
    armsB = (("PPI++ bootstrap-t (this paper)", BLUE, "PPI++, bootstrap-t limit"), ("StratPPI, K=10", ORANGE, "StratPPI as published"),
             ("StratPPI, bootstrap-t, K=10", AQUA, "StratPPI estimator, bootstrap-t limit"))
    for arm, col, lab in armsB:
        for r in B:
            if r["arm"] == arm:
                ess = (base[(r["env"], str(r["rate"]), r["n"])] / r["excess"]) ** 2 if r["excess"] > 0 else 0.0
                assert ess < 7.2, "a point is off the scale: widen the axis"
                a2.plot([r["miss"]], [ess], marker="o", color=col, clip_on=False, zorder=4, **DOT)
                table.append(("judge-logit strata", lab, f"{r['env']} rate={r['truth']:.3f} n={r['n']}", f"{r['miss']:.3f}", f"{ess:.2f}"))
    a2.set_title("A judge's logit as stratifier and predictor", loc="left", fontsize=9, color=INK2, pad=6)
    for a, xmax in ((a1, 0.10), (a2, 0.26)):
        a.axvline(0.05, color=INK2, linewidth=1.0)
        a.set_xlim(0, xmax); a.set_ylim(0, 7.2)
        a.set_xlabel("largest miss rate (δ 0.05)")
        style(a, "y")
    a1.text(0.0515, 7.05, "δ", fontsize=8.5, color=INK2, va="top", ha="left")
    a1.set_ylabel("effective samples vs the baseline")
    a2.annotate("1.3% rate, 225 labels:\nthe bootstrap-t bounds\nreturn nothing usable", xy=(0.008, 0.06), xytext=(0.105, 0.75), fontsize=8, color=INK2, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.7, shrinkA=1, shrinkB=4))
    h = [plt.Line2D([], [], marker="o", color=BLUE, label="this paper's bound for the setting: stratified Wilson-type (left), PPI++ with a bootstrap-t limit (right)", **DOT),
         plt.Line2D([], [], marker="o", color=ORANGE, label="StratPPI as published, with its normal-quantile limit", **DOT),
         plt.Line2D([], [], marker="o", color=AQUA, label="StratPPI's estimator with a bootstrap-t limit", **DOT)]
    top = heading(fig, "StratPPI's interval is shorter partly because it runs over its level; with an honest limit its estimator holds",
                  "Each dot is one cell (a label and a sample size). Right of the line the bound misses more often than it claims. "
                  "Baseline: a random split with a pooled Wilson bound (left), the labels alone (right).", h, ncol=1)
    fig.subplots_adjust(left=0.07, right=0.985, top=top - 0.045, bottom=0.11, wspace=0.17)
    save(fig, "fig3_stratppi", ["panel", "method", "cell", "largest miss", "ESS vs baseline"], table)


# ---------------------------------------------------------------------------------------- fig 7
def fig7():
    """What the guard flags, by human class, on XSTest's published completions (safe prompts)."""
    R = [json.loads(l) for l in open(os.path.join(ROOT, "results", "labels", "xstest", "guard_scores.jsonl"))]
    R = [r for r in R if r["safe"] == 1]
    models = [("gpt4", "GPT-4"), ("mistralinstruct", "Mistral instruct"), ("mistralguard", "Mistral, guardrail prompt"),
              ("llama2new", "Llama 2, no system prompt"), ("llama2orig", "Llama 2, original system prompt"), (None, "all five (1,250)")]
    classes = (("a", BLUE, "full compliance"), ("p", AQUA, "partial refusal"), ("r", ORANGE, "full refusal"))
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    table = []
    for y, (m, name) in enumerate(models):
        rows = [r for r in R if m is None or r["model"] == m]
        for j, (c, col, lab) in enumerate(classes):
            n = sum(r["human"] == c for r in rows); x = sum(r["guard"] for r in rows if r["human"] == c)
            if n < 5:
                table.append((name, lab, x, n, "", "", "")); continue
            lo, hi = cp(x, n, 0.025)
            yy = y + (j - 1) * 0.22
            ax.plot([lo, hi], [yy, yy], color=col, linewidth=1.6, alpha=0.55, solid_capstyle="round")
            ax.plot([x / n], [yy], marker="o", color=col, **DOT)
            table.append((name, lab, x, n, f"{x / n:.3f}", f"{lo:.3f}", f"{hi:.3f}"))
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(len(models) - 0.45, -0.55)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_yticks(range(len(models))); ax.set_yticklabels([n for _, n in models])
    ax.set_xlabel("share of responses the guard flags as a refusal (95% interval)")
    style(ax, "x")
    h = [plt.Line2D([], [], marker="o", color=col, label="humans: " + lab, **DOT) for _, col, lab in classes]
    top = heading(fig, "The guard's refusal flag is broad: it marks most partial refusals too",
                  "Qwen3Guard-4B on the responses of five models to XSTest's 250 safe prompts, against the published human label. "
                  "A class with fewer than 5 responses is not drawn.", h)
    fig.subplots_adjust(left=0.335, right=0.97, top=top, bottom=0.125)
    save(fig, "fig7_guard_xstest", ["model", "human class", "flagged", "responses", "share flagged", "low", "high"], table)


# ---------------------------------------------------------------------------------------- fig 8
def fig8():
    """Our own sheet: the guard's flag rate and the human rates, reference against pushed policy."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import refusal_labels as RL
    D = RL.DIR
    key = {r["id"]: r for r in RL.rows(os.path.join(D, "key.jsonl"))}
    by, adj = RL.load_labels(D)
    G = {i: RL.gold(by, adj, i)[0] for i in key}
    design = json.load(open(os.path.join(D, "design.json")))["population"]
    quantities = (("the guard's refusal flag", None), ("human label: refuses", lambda g: g == "r"),
                  ("human label: refuse-then-answer", lambda g: g == "h"))
    steps = (("0", BLUE, "reference model"), ("200", ORANGE, "policy trained under the constraint"))
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    table = []
    for y, (name, ev) in enumerate(quantities):
        for j, (s, col, lab) in enumerate(steps):
            yy = y + (j - 0.5) * 0.3
            if ev is None:
                tot = sum(v for k, v in design.items() if k.startswith(s + "|")); flagged = sum(v for k, v in design.items() if k.startswith(s + "|g1"))
                p = flagged / tot
                ax.plot([p], [yy], marker="o", color=col, **DOT)
                table.append((name, lab, f"{p:.3f}", "", "", tot))
                continue
            items = [i for i in key if G[i] and key[i]["step"] == s]
            N = {r["stratum"]: r["N_stratum"] for r in key.values() if r["step"] == s}
            r = RL.rate(items, key, N, lambda i: ev(G[i]))
            ax.plot([r["lo"], r["hi"]], [yy, yy], color=col, linewidth=1.6, alpha=0.55, solid_capstyle="round")
            ax.plot([r["p"]], [yy], marker="o", color=col, **DOT)
            table.append((name, lab, f"{r['p']:.3f}", f"{r['lo']:.3f}", f"{r['hi']:.3f}", r["n"]))
    ax.set_xlim(0, 0.26); ax.set_ylim(len(quantities) - 0.45, -0.55)
    ax.set_xticks([0, 0.05, 0.10, 0.15, 0.20, 0.25]); ax.set_xticklabels(["0%", "5%", "10%", "15%", "20%", "25%"])
    ax.set_yticks(range(len(quantities))); ax.set_yticklabels([n for n, _ in quantities])
    ax.set_xlabel("share of responses to benign prompts")
    style(ax, "x")
    h = [plt.Line2D([], [], marker="o", color=col, label=lab, **DOT) for _, col, lab in steps]
    top = heading(fig, "In human terms the two policies refuse equally often, and less often than the guard says",
                  "220 responses labelled by one annotator, a stratified sample read with its design weights; bars are conservative 95% intervals. "
                  "The guard's rate is its share of the whole pool, so it has no interval.", h)
    fig.subplots_adjust(left=0.345, right=0.97, top=top, bottom=0.145)
    save(fig, "fig8_refusal_sheet", ["quantity", "policy", "rate", "low", "high", "n"], table)


# keyed by the number the paper gives each figure
FIGS = dict(fig1=fig1, fig2=fig2, fig3=fig6, fig4=fig5, fig5=fig4, fig6=fig3, fig7=fig7, fig8=fig8)

if __name__ == "__main__":
    for name in (sys.argv[1:] or list(FIGS)):
        FIGS[name]()
