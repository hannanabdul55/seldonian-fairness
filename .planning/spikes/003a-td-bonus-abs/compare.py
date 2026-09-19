"""Spikes 003a/b/c head to head: what does an internal TD-error reward do to a
Seldonian run? Shared harness; 003b and 003c read the same results.

Arms: no bonus; |delta| (003a); max(delta, 0) (003b); learning progress (003c);
|A| (E1 of ../LITERATURE.md); a random bonus of matched size (control).
Environment: NoisyTVEnv, whose fifth action is safe, slightly worse than the best safe
action, and has 6x the reward noise: pure unlearnable surprise.

    ../../../.venv/bin/python compare.py [--seeds 60] [--betas 0.5 1 2]

Writes results.md / results.json here; 003b and 003c link to them.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "001-grpo-advantage-vs-td"))
import tdlab  # noqa: E402
from bonuses import ARMS  # noqa: E402


def one(job):
    arm, beta, pressure, seed = job
    bonus = (lambda: ARMS[arm](beta)) if arm != "none" else None
    row, log = tdlab.run(seed, method="seldonian_lag", pressure=pressure, env="noisy_tv",
                         steps=200, bonus=bonus)
    S = tdlab.stack
    dc, d, b = S(log, "delta_c"), S(log, "delta"), S(log, "bonus")
    share = np.array([s["action_share"] for s in log])       # population policy per step
    tr = S(log, "true_rate")
    h = len(log) // 2
    out = {
        "arm": arm, "beta": beta, "pressure": pressure, "seed": seed,
        "solution": row["solution"], "violates": row["violates"],
        "true_rate": row["true_rate"], "threshold": row["threshold"],
        "true_reward": row["true_reward"], "ref_reward": row["ref_reward"],
        "tv_share": row.get("tv_share"), "unsafe_share": float(share[-1, 2:4].sum()),
        "train_rate_mean": float(tr.mean()), "train_rate_max": float(tr.max()),
        "n_feasible": row["n_feasible"], "lam_end": float(S(log, "lam")[-1]),
        "bonus_mean_early": float(b[:h].mean()), "bonus_mean_late": float(b[h:].mean()),
        "critic_lag_late": float(np.abs(dc - d)[h:].mean()),   # |V - V_critic|
        "mean_delta_c_late": float(dc[h:].mean()),
        "tv_share_mid": float(share[h, 4]),
    }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=60)
    p.add_argument("--betas", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    p.add_argument("--pressure", type=float, nargs="+", default=[1.0, 4.0])
    a = p.parse_args()
    jobs = [("none", 0.0, pr, s) for pr in a.pressure for s in range(a.seeds)]
    jobs += [(arm, b, pr, s) for arm in ARMS for b in a.betas for pr in a.pressure
             for s in range(a.seeds)]
    with ProcessPoolExecutor(max(1, os.cpu_count() // 2)) as ex:
        rows = list(ex.map(one, jobs, chunksize=8))
    json.dump(rows, open(os.path.join(HERE, "results.json"), "w"), indent=1)

    def agg(rs, k):
        return float(np.mean([r[k] for r in rs]))

    L = [f"# Spikes 003a-c: internal TD-error rewards ({a.seeds} seeds per cell, "
         f"noisy-TV env, Seldonian Lagrangian, 200 steps)", "",
         "`solution` = safety test passed (a Seldonian solution was returned); `unsafe` = the "
         "returned policy's true violation rate is above the threshold; `reward` is the "
         "extrinsic (noise-free, bonus-free) population reward; `TV share` is the policy's "
         "probability on the noisy-TV action (unlearnable surprise, safe, slightly "
         "suboptimal); `train rate` is the true violation rate during training.", ""]
    for pr in a.pressure:
        L += [f"## pressure {pr:g}", "",
              "| arm | beta | solution | unsafe (of all) | unsafe given solution | reward | "
              "TV share end (mid) | unsafe-action share | train rate mean / max | lambda end | "
              "bonus early / late | critic lag late | mean delta late |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        cells = [("none", 0.0)] + [(arm, b) for arm in ARMS for b in a.betas]
        for arm, b in cells:
            rs = [r for r in rows if r["arm"] == arm and r["beta"] == b and r["pressure"] == pr]
            if not rs:
                continue
            sol = [r for r in rs if r["solution"]]
            L.append(
                f"| {arm} | {b:g} | {agg(rs, 'solution'):.2f} | {agg(rs, 'violates'):.3f} | "
                f"{(agg(sol, 'violates') if sol else float('nan')):.3f} | {agg(rs, 'true_reward'):.3f} | "
                f"{agg(rs, 'tv_share'):.3f} ({agg(rs, 'tv_share_mid'):.3f}) | {agg(rs, 'unsafe_share'):.3f} | "
                f"{agg(rs, 'train_rate_mean'):.3f} / {agg(rs, 'train_rate_max'):.3f} | "
                f"{agg(rs, 'lam_end'):.1f} | {agg(rs, 'bonus_mean_early'):.2f} / "
                f"{agg(rs, 'bonus_mean_late'):.2f} | {agg(rs, 'critic_lag_late'):.3f} | "
                f"{agg(rs, 'mean_delta_c_late'):+.3f} |")
        L.append("")
    txt = "\n".join(L) + "\n"
    open(os.path.join(HERE, "results.md"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
