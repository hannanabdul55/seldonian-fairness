"""Spike 010: why does an early price on F without a floor end up worse than no price?

Spike 004 (Results 4): the plain Lagrangian with lam0 = 5 and no floor entered U in
0.905 of runs at 25-step checks, against 0.685 with lam0 = 0, with lower task accuracy.
Its untested guess: the early penalty trains *wrong answers* on F rather than refusal,
and once dual descent has taken the multiplier away, nothing holds the policy back.

This traces both arms (and the always-on floor) step by step on the same seeds, from the
exact per-step log: the multiplier, F willingness (1 - refusal), capability on F and on
the twins, the harm rate, and the refuse logit's weights.

    ../../../.venv/bin/python trace.py [--seeds 200]
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "004-forbidden-capability"))
import forbidlab as F  # noqa: E402

ARMS = {"lam0_0": dict(method="lag", lam0=0.0),
        "lam0_5": dict(method="lag", lam0=5.0),
        "floor": dict(method="lag", lam_floor=5.0, floor_always=True),
        "grpo": dict(method="grpo")}
TAU, EVERY = 0.12, 25


def one(job):
    arm, seed = job
    # run() does not expose the backend; rebuild the per-step log from the row plus a
    # second instrumented pass is not needed: the row carries harm_traj and lam_traj, and
    # the per-check table carries willingness and twin capability. For per-step
    # willingness/capability we patch the backend's log through a subclass hook.
    rows = []
    orig = F.ForbiddenBackend.train

    def train(self, records, reward, on_step):
        def hook(step):
            ex = self.env.exact((self.W, self.c))
            rows.append(dict(step=step, will=ex["forbidden"]["answer"], cap_f=ex["forbidden"]["cap"],
                             cap_twin=ex["twin"]["cap"], harm=ex["harm"], task_acc=ex["task"]["acc"],
                             twin_acc=ex["twin"]["acc"],
                             refuse_flag_w=float(self.W[0, -1] - self.W[1:, -1].mean()),
                             lam=float(sum(getattr(reward, "lambdas", {}).values()))))
            on_step(step)
        return orig(self, records, reward, hook)

    F.ForbiddenBackend.train = train
    try:
        row = F.run(seed, tau=TAU, predict_every=EVERY, **ARMS[arm])
    finally:
        F.ForbiddenBackend.train = orig
    return dict(arm=arm, seed=seed, entered_U=row["entered_U"], first_U=row["first_U_step"],
                task_acc=row["final_task_acc"], trace=rows)


def summarise(res):
    steps = [1, 10, 25, 26, 40, 50, 75, 100, 150, 200]
    lines = ["# Spike 010 trace (means over seeds)", ""]
    keys = ["lam", "will", "cap_f", "cap_twin", "twin_acc", "task_acc", "harm", "refuse_flag_w"]
    for arm in ARMS:
        rs = [r for r in res if r["arm"] == arm]
        lines += [f"## {arm}: entered U {np.mean([r['entered_U'] for r in rs]):.3f}, "
                  f"final task acc {np.mean([r['task_acc'] for r in rs]):.3f}", "",
                  "| step | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
        for s in steps:
            vals = [np.mean([r["trace"][s - 1][k] for r in rs]) for k in keys]
            lines.append(f"| {s} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=200)
    a = ap.parse_args()
    js = [(arm, s) for arm in ARMS for s in range(a.seeds)]
    with ProcessPoolExecutor(4) as ex:
        res = list(ex.map(one, js, chunksize=8))
    json.dump([{k: v for k, v in r.items()} for r in res], open(os.path.join(HERE, "results.json"), "w"))
    text = summarise(res)
    open(os.path.join(HERE, "results.md"), "w").write(text)
    print(text)
