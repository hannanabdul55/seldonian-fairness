"""What a human-terms certificate costs (paper plan P9, reframed 2026-10-04).

The constraint is: trained policy's strict refusal rate minus the reference's is at most a margin,
in human labels, at delta 0.05. From the first sheet's measured rates this gives the number of
labelled prompt pairs needed, with labels alone and with the guard's logit as a predictor, and the
margin a given number of pairs can be expected to certify. A design calculation, not a result:
normal-type limits, the pairing correlation taken from the guard's flags on the same pools, and
the guard's rho^2 for a difference assumed equal to its measured value for a level.

Two guard columns, because the guard only removes the part of the uncertainty it can see. "Guard's
mean known" is the claim about the pool's own prompts, with guard-only responses in number well
above the labelled pairs. "Other pool prompts only" is the claim about new prompts from the same
source, where the pool's remaining prompts are all the unlabelled data there is: the variance
factor is 1 - rho^2 (POOL - n) / POOL, and no number of labels inside the pool reaches some margins.

    .venv/bin/python scripts/p9_budget.py   ->  results/paper/p9_budget.md
"""
import json
import math
import os
import sys

import numpy as np
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
import refusal_labels as RL  # noqa: E402
from refusal_sheet_build import EXAMPLE_PROMPTS, load  # noqa: E402

DELTA, RHO2, POOL = 0.05, 0.60, 490          # P6: rho^2 of the human strict label with the guard's logit, 0.61 and 0.60


def main():
    key = {r["id"]: r for r in RL.rows(os.path.join(RL.DIR, "key.jsonl"))}
    by, adj = RL.load_labels(RL.DIR)
    G = {i: RL.gold(by, adj, i)[0] for i in key}
    p = {}
    for s in ("0", "200"):
        N = {r["stratum"]: r["N_stratum"] for r in key.values() if r["step"] == s}
        p[s] = RL.rate([i for i in key if G[i] and key[i]["step"] == s], key, N, lambda i: G[i] == "r")["p"]
    d = p["200"] - p["0"]
    # pairing: how correlated two single responses to the same prompt are across the two policies (guard's flag)
    _, j0 = load("results/spikes/013/gen_full.jsonl", "results/spikes/013/judged_full.jsonl", 0, "cov")
    _, j2 = load("results/spikes/014/gen_s0.jsonl", "results/spikes/014/judged_s0.jsonl", 200)
    ids = sorted(i for i in set(j0) & set(j2) if i not in EXAMPLE_PROMPTS)
    a, b = np.array([np.mean(j0[i]) for i in ids]), np.array([np.mean(j2[i]) for i in ids])
    corr = float(np.cov(a, b)[0, 1] / math.sqrt(a.mean() * (1 - a.mean()) * b.mean() * (1 - b.mean())))
    v0, v1 = p["0"] * (1 - p["0"]), p["200"] * (1 - p["200"])
    var = {"labels alone, unpaired": v0 + v1, "labels alone, paired by prompt": v0 + v1 - 2 * corr * math.sqrt(v0 * v1)}
    vp = var["labels alone, paired by prompt"]
    var["paired, with the guard, guard's mean known"] = vp * (1 - RHO2)
    z = norm.ppf(1 - DELTA)
    inpool = "paired, with the guard, other pool prompts only"

    def pool_pairs(m, zz):
        """Smallest n <= POOL with zz^2 vp (1 - RHO2 (POOL - n) / POOL) / n <= (m - d)^2."""
        den = (m - d) ** 2 - zz ** 2 * vp * RHO2 / POOL
        n = zz ** 2 * vp * (1 - RHO2) / den if den > 0 else math.inf
        return f"{math.ceil(n):,}" if n <= POOL else f"not within {POOL} prompts"
    L = ["# What a human-terms certificate costs", "",
         f"Measured on the first sheet (one annotator): strict refusal {p['0']:.3f} for the reference, {p['200']:.3f} for the trained policy "
         f"(difference {d:+.3f}). Correlation of two single responses to the same prompt across the two policies, by the guard's flag on "
         f"{len(ids)} prompts: {corr:.2f}. Guard rho^2 {RHO2}. One-sided test at delta {DELTA}, normal-type limit.", "",
         "## Prompt pairs needed to certify a margin", "", "| margin | chance of certifying | " + " | ".join(var) + f" | {inpool} |", "|---|---|" + "---|" * (len(var) + 1)]
    for m in (0.02, 0.03, 0.05):
        for power in (0.5, 0.8):
            zz = z + norm.ppf(power)
            L.append(f"| {m:.2f} | {power:.0%} | " + " | ".join(f"{math.ceil(zz ** 2 * v / (m - d) ** 2):,}" for v in var.values()) + f" | {pool_pairs(m, zz)} |")
    L += ["", "## Margin a sample can be expected to certify", "", "| pairs labelled | " + " | ".join(var) + f" | {inpool} |", "|---|" + "---|" * (len(var) + 1)]
    for n in (150, 300, 600, 1000):
        last = f"{d + z * math.sqrt(vp * (1 - RHO2 * (POOL - n) / POOL) / n):.3f}" if n <= POOL else f"more than {POOL} prompts"
        L.append(f"| {n:,} | " + " | ".join(f"{d + z * math.sqrt(v / n):.3f}" for v in var.values()) + f" | {last} |")
    L += ["", f"The two guard columns answer different questions. \"Guard's mean known\": the rate on the pool's own {POOL} prompts, with enough guard-only "
          "responses that the guard's mean carries no error (P9 draws 8 per policy per prompt). \"Other pool prompts only\": the rate on new prompts "
          f"from the same source, where the {POOL} - n unlabelled pool prompts are all the guard has. The betting bound that P9 uses for labels alone "
          "is wider than the normal-type limit in the \"paired by prompt\" column; `results/labels/p9/design_check.md` has the simulated margins."]
    os.makedirs(os.path.join(ROOT, "results", "paper"), exist_ok=True)
    open(os.path.join(ROOT, "results", "paper", "p9_budget.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
