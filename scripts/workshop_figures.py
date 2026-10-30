"""Column-width figures for the workshop version of the certification paper.

Reads the CSVs that ``scripts/paper_figures.py`` writes next to the long draft's figures (no number is recomputed)
and draws them for one column of the AAAI two-column format, without the long draft's titles and subtitles (the
caption carries the text).

    .venv/bin/python scripts/workshop_figures.py
    -> reports/workshop/fig_carry.pdf, reports/workshop/fig_agentdojo.pdf
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
FIGS = os.path.join(ROOT, "reports", "figs")
OUT = os.path.join(ROOT, "reports", "workshop")

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"], "font.size": 7.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.size": 0, "pdf.fonttype": 42,
})

SHORT = {
    "200 training steps, label moved as a side effect": "update, label moved as side effect",
    "100 steps that target the label, seed 0": "update targets label, 100 steps, seed 0",
    "200 steps that target the label, seed 0": "update targets label, 200 steps, seed 0",
    "100 steps that target the label, seed 1": "update targets label, 100 steps, seed 1",
    "200 steps that target the label, seed 1": "update targets label, 200 steps, seed 1",
    "100 steps that target the label, seed 2": "update targets label, 100 steps, seed 2",
    "200 steps that target the label, seed 2": "update targets label, 200 steps, seed 2",
    "prompt source: OR-Bench to XSTest": "prompt source: OR-Bench to XSTest",
    "prompt source: XSTest to OR-Bench": "prompt source: XSTest to OR-Bench",
    "prompt pool: benign to harmful": "prompt pool: benign to harmful",
    "prompt pool: harmful to benign": "prompt pool: harmful to benign",
}


def carry():
    rows = list(csv.DictReader(open(os.path.join(FIGS, "fig4_carrying.csv"))))
    order = list(dict.fromkeys(r["shift"] for r in rows))
    assert set(order) == set(SHORT), set(order) ^ set(SHORT)
    fig, ax = plt.subplots(figsize=(3.3, 2.45))
    for i, s in enumerate(order):
        y = len(order) - 1 - i
        xs = sorted(float(r["carried bound miss rate"]) for r in rows if r["shift"] == s)
        assert len(xs) == 6, (s, len(xs))
        ax.scatter(xs, [y + 0.12 * ((k % 3) - 1) for k in range(6)], s=9, c="#3a6ea5", linewidths=0, zorder=3)
    ax.axvline(0.05, color="0.25", lw=0.7, zorder=2)
    ax.text(0.06, len(order) - 0.45, r"$\delta = 0.05$", fontsize=6.5, color="0.25", va="bottom")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([SHORT[s] for s in reversed(order)], fontsize=6.5)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.6, len(order) - 0.2)
    ax.set_xlabel("miss rate of the carried bound")
    for y in (3.5, 1.5):
        ax.axhline(y, color="0.85", lw=0.5, zorder=1)
    ax.grid(axis="x", color="0.9", lw=0.5, zorder=0)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "fig_carry.pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(OUT, "fig_carry.png"), dpi=200, bbox_inches="tight", pad_inches=0.02)


def agentdojo(n_show=9):
    rows = list(csv.DictReader(open(os.path.join(FIGS, "fig6_agentdojo.csv"))))
    assert len(rows) == 28, len(rows)
    rows = sorted(rows, key=lambda r: float(r["rate"]))[:n_show]
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        rate, pp, cl = (float(r[k]) for k in ("rate", "per-pair bound", "clustered by user task"))
        ax.plot([rate, max(pp, cl)], [y, y], color="0.85", lw=1.2, zorder=1)
        ax.scatter([rate], [y], s=11, c="0.6", zorder=3, linewidths=0, label="observed rate" if i == 0 else None)
        ax.scatter([pp], [y], s=11, c="#3a6ea5", zorder=3, linewidths=0, label="pairs as independent" if i == 0 else None)
        ax.scatter([cl], [y], s=11, c="#c8553d", zorder=4, linewidths=0, label="clustered by user task" if i == 0 else None)
    ax.axvline(0.05, color="0.25", lw=0.7, zorder=2)
    ax.set_xscale("log")
    ax.set_xticks([0.01, 0.02, 0.05, 0.1])
    ax.set_xticklabels(["1%", "2%", "5%", "10%"])
    ax.minorticks_off()
    ax.set_yticks(range(len(rows)))
    name = lambda p: (p.replace("meta-llama_", "").replace("gpt-4o-2024-05-13-", "gpt-4o + ").replace("-repeat_user_prompt", " (repeat)")
                      .replace("_", " ").replace("pi detector", "PI detector"))
    ax.set_yticklabels([name(r["pipeline"]) for r in reversed(rows)], fontsize=6)
    ax.set_xlim(0.008, 0.2)
    ax.set_xlabel(r"rate and upper bounds at $\delta = 0.05$")
    ax.grid(axis="x", color="0.9", lw=0.5, zorder=0)
    fig.legend(loc="upper center", ncol=3, fontsize=6, frameon=False, handletextpad=0.1, columnspacing=0.8)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(OUT, "fig_agentdojo.pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(OUT, "fig_agentdojo.png"), dpi=200, bbox_inches="tight", pad_inches=0.02)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    carry()
    agentdojo()
