"""Spike 004: a held-out forbidden task, its drift under training, and what stops it.

Arms (all train only the allowed task; F prompts carry base reward 0):
  grpo        no constraint, monitor only
  lag         Seldonian constraint on F + Lagrangian (lam0 0: nobody priced F in advance)
  lag_floor   + always-on floor lam >= 5 (the Round 6 fix)
  lag_early   + capability look-ahead: dual steps on willingness x (twin capability + slope)
  lag_random  control: the look-ahead's number and size of raises, at random checks
ratchet set (--set ratchet): the floor lam >= 5 armed by different triggers
  lag_armed           armed by a predicted breach (Round 6 B1)
  lag_early_ratchet   armed by the capability look-ahead
  lag_random_ratchet  control: armed at random checks with the look-ahead's arming hazard
at check intervals of 25 steps (as the LLM runs) and 10 steps.

    ../../../.venv/bin/python compare.py [--seeds 200] [--tau 0.12] [--set base|ratchet]

Writes results.json (base) or results_ratchet.json here; analyze.py makes results.md.
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import forbidlab as F

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = {"grpo": dict(method="grpo"), "lag": dict(method="lag"),
        "lag_floor": dict(method="lag", lam_floor=5.0, floor_always=True),
        "lag_early": dict(method="lag_early")}
RATCHET = {"lag_floor": dict(method="lag", lam_floor=5.0, floor_always=True),
           "lag_armed": dict(method="lag", lam_floor=5.0),
           "lag_early_ratchet": dict(method="lag_early", lam_floor=5.0, arm_floor=True)}


def one(job):
    arm, every, seed, tau, extra = job
    kw = {**ARMS.get(arm, {}), **RATCHET.get(arm, {}), **extra}
    if arm == "lag_random":
        kw["method"] = "lag_random"
    if arm == "lag_random_ratchet":
        kw.update(method="lag_random", lam_floor=5.0, arm_floor=True)
    row = F.run(seed, tau=tau, predict_every=every, **kw)
    row["arm"], row["every"] = arm, every
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--tau", type=float, default=0.12)
    ap.add_argument("--every", type=int, nargs="+", default=[25, 10])
    ap.add_argument("--set", choices=["base", "ratchet"], default="base")
    a = ap.parse_args()
    workers = max(os.cpu_count() // 2, 1)
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        arms, early_arm, ctrl = ((ARMS, "lag_early", "lag_random") if a.set == "base" else
                                 (RATCHET, "lag_early_ratchet", "lag_random_ratchet"))
        jobs = [(arm, e, s, a.tau, {}) for e in a.every for arm in arms for s in range(a.seeds)]
        rows += list(ex.map(one, jobs, chunksize=8))
        # size-matched random control, per check interval. Base set: the same raise
        # probability per check and mean raise as the look-ahead arm. Ratchet set: the
        # same hazard of the first trigger per check (the floor arms at the first one).
        for e in a.every:
            early = [r for r in rows if r["arm"] == early_arm and r["every"] == e]
            raises = [x for r in early for _, x in r["triggers"]]
            step = float(np.mean(raises)) if raises else 0.0
            if a.set == "base":
                n_upd = sum(len(r["lam_path"]) for r in early)
                p = len(raises) / max(n_upd, 1)
            else:
                every_steps = [e * (j + 1) for j in range(len(early[0]["lam_path"]))]
                armed = [r["triggers"][0][0] for r in early if r["triggers"]]
                at_risk = sum(sum(1 for s in every_steps if not r["triggers"] or s <= r["triggers"][0][0])
                              for r in early)
                p = len(armed) / max(at_risk, 1)
            print(f"every {e}: {early_arm} p={p:.3f} per check, mean +{step:.2f}")
            jobs = [(ctrl, e, s, a.tau, dict(random_p=p, random_step=step))
                    for s in range(a.seeds)]
            rows += list(ex.map(one, jobs, chunksize=8))
    name = "results.json" if a.set == "base" else "results_ratchet.json"
    with open(os.path.join(HERE, name), "w") as f:
        json.dump(dict(tau=a.tau, seeds=a.seeds, rows=rows), f)
    print(f"{len(rows)} runs written")


if __name__ == "__main__":
    main()
