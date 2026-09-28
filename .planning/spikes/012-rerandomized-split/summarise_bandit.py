"""Table for spike 012 part B (bandit.json from banditsplit.py).

    ../../../.venv/bin/python summarise_bandit.py bandit_m003.json

sd(comp) is the spread of the safety set's composition error (its prompts' exact rate under
the returned policy minus the population rate): what balancing can remove. sd(err) is the
whole error of the safety-set estimate (composition + response sampling).
"""
import json
import sys

import numpy as np

from summarise import wilson_hi

for p in sys.argv[1:]:
    rows = json.load(open(p))
    print(f"\n### {p}: margin {rows[0]['margin']}, {len(rows) // len({r['rule'] for r in rows})} seeds per rule\n")
    print("| rule | solution | miss (95% hi) | uncovered | sd(ref imbalance) | sd(comp) | sd(err) "
          "| mean err | agree | post-strat solution | post-strat miss | post-strat uncovered | reward if pass | tries |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for rule in dict.fromkeys(r["rule"] for r in rows):
        R = [r for r in rows if r["rule"] == rule]
        n = len(R)
        k = sum(r["miss"] for r in R)
        sol = [r for r in R if r["solution"]]
        agree = np.mean([r["predicted_feasible"] == r["solution"] for r in R])
        print(f"| {rule} | {len(sol) / n:.3f} | {k / n:.4f} ({wilson_hi(k, n):.4f}) "
              f"| {np.mean([r['uncovered'] for r in R]):.3f} "
              f"| {np.std([r['ref_imbalance'] for r in R]):.4f} "
              f"| {np.std([r['comp_err'] for r in R]):.4f} | {np.std([r['err_s'] for r in R]):.4f} "
              f"| {np.mean([r['err_s'] for r in R]):+.4f} | {agree:.3f} "
              f"| {np.mean([r['ps_pass'] for r in R]):.3f} | {np.mean([r['ps_miss'] for r in R]):.4f} "
              f"| {np.mean([r['ps_uncovered'] for r in R]):.3f} "
              f"| {np.mean([r['true_reward'] for r in sol]):.4f} | {np.mean([r['tries'] for r in R]):.1f} |")
