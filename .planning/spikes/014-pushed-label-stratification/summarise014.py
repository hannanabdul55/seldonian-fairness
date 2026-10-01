"""Spike 014 stage 1 tables and the gate (DESIGN.md sections 3 and 4).

Per cell (env x method x pressure x eta x margin x steps): how far the rate moved, ICC_cand /
ICC_ref, rho, feasibility, the miss rate and width of each bound at both deltas, the
realised ESS of the stratified `b1w` (pool target) over the random rule's pooled `b1w`, and
the two predictions: 013's formula at the *measured* ICC_cand and rho, and the naive
pre-flight (ICC_ref, rho 0.8). Then H1-H3 and the gate, evaluated on 013's real C1 reference
labels at the bandit cell whose rate move matches Round 6's.

    ../../../.venv/bin/python summarise014.py bandit.json        # writes bandit.md, gate.json
"""
import collections
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "013-stratified-safety-set"))
from plasmode import C_H, icc_from_samples, reliability  # noqa: E402

K = 8
KEY = ("env", "method", "pressure", "eta", "margin", "steps")


def cell_key(r):
    return tuple(r[k] for k in KEY)


def hi(k, n, z=1.96):
    p = k / n
    return (p + z * z / (2 * n) + z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)


def summarise(rows):
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        by[cell_key(r)][r["rule"]].append(r)
    out = []
    for key in sorted(by, key=lambda k: (k[0], k[1], k[2], k[3], -k[4], k[5])):
        rules = by[key]
        if "random" not in rules or "strat_ref" not in rules:
            continue
        R, S = rules["random"], rules["strat_ref"]
        c = dict(zip(KEY, key), n=len(S))
        allr = R + S
        c.update(ref_rate=np.mean([r["ref_rate"] for r in allr]),
                 rate=np.mean([r["truth_pool"] for r in allr]),
                 feasible=np.mean([bool(r["predicted_feasible"]) for r in allr]),
                 solution=np.mean([bool(r["solution"]) for r in allr]) if key[1] != "grpo" else float("nan"),
                 icc_ref=np.mean([r["icc_ref"] for r in allr]), icc_cand=np.mean([r["icc_cand"] for r in allr]),
                 rho=np.mean([r["rho"] for r in allr]), lam=np.nanmean([r["lam_final"] or np.nan for r in allr]))
        c["moved"] = c["rate"] - c["ref_rate"]
        c["icc_ratio"] = c["icc_cand"] / c["icc_ref"]
        for d in (0.05, 0.1):
            w0 = np.mean([r[f"b1w_pooled@{d}"] - r[f"est_pool@{d}"] for r in R])
            w1 = np.mean([r[f"b1w_strat_pool@{d}"] - r[f"est_strat@{d}"] for r in S])
            c[f"ess@{d}"] = (w0 / w1) ** 2
            c[f"miss_random@{d}"] = np.mean([r["truth_pop"] > r[f"b1w_pooled@{d}"] for r in R])
            c[f"miss_strat_pool@{d}"] = np.mean([r["truth_pool"] > r[f"b1w_strat_pool@{d}"] for r in S])
            c[f"miss_strat_pop@{d}"] = np.mean([r["truth_pop"] > r[f"b1w_strat_pop@{d}"] for r in S])
            c[f"width_random@{d}"], c[f"width_strat@{d}"] = w0, w1
        rel = reliability(c["icc_ref"], K)
        c["G_meas"] = c["icc_cand"] * c["rho"] ** 2 * rel * C_H[8]
        c["G_naive"] = c["icc_ref"] * 0.8 ** 2 * rel * C_H[8]
        c["pred_meas"] = 1 / (1 - c["G_meas"])
        c["pred_naive"] = 1 / (1 - c["G_naive"])
        if "placebo" in rules:
            P = rules["placebo"]
            w0 = np.mean([r["b1w_pooled@0.1"] - r["est_pool@0.1"] for r in R])
            w1 = np.mean([r["b1w_strat_pool@0.1"] - r["est_strat@0.1"] for r in P])
            c["ess_placebo@0.1"] = (w0 / w1) ** 2
            c["miss_placebo@0.1"] = np.mean([r["truth_pool"] > r["b1w_strat_pool@0.1"] for r in P])
        out.append(c)
    return out


def real_reference():
    """013's C1 pool: 8 reference samples per prompt of the refusal label."""
    Y = collections.defaultdict(list)
    for line in open(os.path.join(REPO, "results", "spikes", "013", "judged_full.jsonl")):
        r = json.loads(line)
        if r["pool"] == "C1" and r["step"] == 0 and r["role"] == "cov":
            Y[r["i"]] = r["refusal"][:K]
    return np.array([Y[i] for i in sorted(Y)], dtype=float)


def main():
    rows = json.load(open(sys.argv[1]))
    cells = summarise(rows)
    L = ["# Spike 014 stage 1: the bandit under pressure", "",
         f"{len(rows)} runs, {len(cells)} cells with both rules. ESS = (random pooled `b1w` width / "
         "stratified `b1w` width)^2 at the pool target; `pred meas` = 013's formula at the measured "
         "ICC_cand and rho; `pred naive` = at ICC_ref and rho 0.8. Miss rates at delta 0.1 (delta 0.05 "
         "in `bandit.json`).", "",
         "| env | method | pressure | eta | margin | steps | feasible | solution | ref rate | moved "
         "| ICC_ref | ICC_cand/ICC_ref | rho | lam | miss random | miss strat (pool) | miss strat (pop) "
         "| ESS | pred meas | pred naive | placebo ESS (miss) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        pl = f"{c['ess_placebo@0.1']:.2f} ({c['miss_placebo@0.1']:.3f})" if "ess_placebo@0.1" in c else ""
        L.append(f"| {c['env']} | {c['method']} | {c['pressure']} | {c['eta']:.0f} | {c['margin']:+.2f} | {c['steps']} "
                 f"| {c['feasible']:.2f} | {c['solution']:.2f} | {c['ref_rate']:.3f} | {c['moved']:+.3f} "
                 f"| {c['icc_ref']:.2f} | {c['icc_ratio']:.2f} | {c['rho']:.2f} | {c['lam']:.1f} "
                 f"| {c['miss_random@0.1']:.3f} | {c['miss_strat_pool@0.1']:.3f} | {c['miss_strat_pop@0.1']:.3f} "
                 f"| {c['ess@0.1']:.2f} | {c['pred_meas']:.2f} | {c['pred_naive']:.2f} | {pl} |")

    # H1: compression and rho against pressure (constrained cells, eta 400, steps 200)
    L += ["", "## H1: do rates compress and rho decay with pressure?", "",
          "Constrained cells at eta 400, 200 steps: ICC_cand / ICC_ref and rho by pressure and margin "
          "(mean over the three environments), beside the unconstrained control.", "",
          "| margin | quantity | pressure 0.5 | 1 | 2 | 4 |", "|---|---|---|---|---|---|"]
    lag = [c for c in cells if c["method"] == "seldonian_lag" and c["eta"] == 400 and c["steps"] == 200]
    ctl = [c for c in cells if c["method"] == "grpo"]
    for mg in (0.03, -0.03, -0.06):
        for q in ("icc_ratio", "rho", "moved"):
            vals = [np.mean([c[q] for c in lag if c["margin"] == mg and c["pressure"] == p]) for p in (0.5, 1.0, 2.0, 4.0)]
            L.append(f"| {mg:+.2f} | {q} | " + " | ".join(f"{v:.2f}" for v in vals) + " |")
    for q in ("icc_ratio", "rho", "moved"):
        vals = [np.mean([c[q] for c in ctl if c["pressure"] == p]) for p in (0.5, 1.0, 2.0, 4.0)]
        L.append(f"| control (grpo) | {q} | " + " | ".join(f"{v:.2f}" for v in vals) + " |")
    strong = [c for c in lag if c["pressure"] == 4.0 and c["feasible"] >= 0.5]
    rho_strong = min(c["rho"] for c in strong) if strong else float("nan")
    L.append(f"\nSmallest mean rho over feasible pressure-4 cells: {rho_strong:.2f} (H1 predicted < 0.7).")

    # H2: prediction
    L += ["", "## H2: does the formula at measured moderators predict the realised ESS?", ""]
    pushed = [c for c in cells if c["method"] == "seldonian_lag"]
    err_m = [abs(c["pred_meas"] - c["ess@0.1"]) for c in cells]
    err_n = [abs(c["pred_naive"] - c["ess@0.1"]) for c in pushed]
    sp = spearmanr([c["pred_meas"] for c in cells], [c["ess@0.1"] for c in cells]).correlation
    over_n = [c["pred_naive"] - c["ess@0.1"] for c in pushed]
    L += [f"- measured-moderator prediction: median abs error {np.median(err_m):.3f} over {len(cells)} cells "
          f"(H2 asks <= 0.15), Spearman {sp:.2f} (asks >= 0.8).",
          f"- naive pre-flight on pushed cells: median abs error {np.median(err_n):.3f}, "
          f"over-prediction above 0.3 in {sum(o > 0.3 for o in over_n)} of {len(pushed)} cells."]

    # Gate: 013's real C1 reference labels at the bandit's moderators for a Round 6 sized move
    Y = real_reference()
    icc_real = icc_from_samples(Y)
    rel_real = reliability(icc_real, K)
    L += ["", "## The gate (DESIGN.md section 3)", "",
          f"013's real C1 reference labels: rate {Y.mean():.3f}, ICC_ref {icc_real:.2f}, rel(8) {rel_real:.2f}. "
          "Bandit cells whose rate move is between 3 and 13 points, constrained, eta 400, 200 steps: "
          "predicted ESS for the real pool = 1 / (1 - ICC_ref_real x ratio x rho^2 x rel x c_8).", "",
          "| env | pressure | margin | moved | ICC ratio | rho | bandit ESS | predicted real ESS |",
          "|---|---|---|---|---|---|---|---|"]
    gate = []
    for c in lag:
        if 0.03 <= abs(c["moved"]) <= 0.13 and c["feasible"] >= 0.5:
            g = icc_real * c["icc_ratio"] * c["rho"] ** 2 * rel_real * C_H[8]
            pred = 1 / (1 - g)
            gate.append(dict(env=c["env"], pressure=c["pressure"], margin=c["margin"], moved=c["moved"],
                             icc_ratio=c["icc_ratio"], rho=c["rho"], ess=c["ess@0.1"], pred_real=pred))
            L.append(f"| {c['env']} | {c['pressure']} | {c['margin']:+.2f} | {c['moved']:+.3f} | {c['icc_ratio']:.2f} "
                     f"| {c['rho']:.2f} | {c['ess@0.1']:.2f} | {pred:.2f} |")
    worst = min((g["pred_real"] for g in gate), default=float("nan"))
    verdict = "OPEN: run stage 2" if worst >= 1.2 else "CLOSED: the bandit says the gain does not survive; stop"
    L.append(f"\nSmallest predicted real ESS over matched cells: {worst:.2f}. **Gate {verdict}.**")
    json.dump(dict(cells=cells, gate=gate, icc_real=icc_real, worst_pred=worst, verdict=verdict),
              open(os.path.join(HERE, "gate.json"), "w"), indent=1, default=float)
    open(os.path.join(HERE, "bandit.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[-40:]))


if __name__ == "__main__":
    main()
