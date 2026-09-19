"""003a follow-up: where does the |delta| bonus invert the constraint?

With a calibrated critic a violation's TD error is about ``-lam * (1 - p_v)``, so a bonus
``beta * |delta|`` pays ``+beta * lam * (1 - p_v)`` for the same violation that the
Lagrangian charges ``-lam * (1 - p_v)``. The net incentive to violate is
``(beta - 1) * lam * (1 - p_v)``: positive as soon as beta > 1, and self-reinforcing,
because each infeasible predicted test raises lam and so raises the bonus too.
Arms: |delta| on the shaped reward, |delta| on the task reward only, learning progress.
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "001-grpo-advantage-vs-td"))
import tdlab  # noqa: E402
from bonuses import ARMS  # noqa: E402

BETAS = [0.0, 0.5, 0.8, 1.0, 1.25, 1.5, 2.0]
ARMLIST = ["abs", "abs_task", "lp"]
SEEDS = 60


def one(job):
    arm, beta, seed = job
    bonus = (lambda: ARMS[arm](beta)) if beta else None
    row, log = tdlab.run(seed, method="seldonian_lag", pressure=4.0, env="noisy_tv",
                         steps=200, bonus=bonus)
    share = np.array([s["action_share"] for s in log])
    return {"arm": arm, "beta": beta, "seed": seed, "solution": row["solution"],
            "violates": row["violates"], "true_rate": row["true_rate"],
            "true_reward": row["true_reward"], "tv_share": row["tv_share"],
            "unsafe_share": float(share[-1, 2:4].sum()),
            "train_rate_max": float(tdlab.stack(log, "true_rate").max()),
            "lam_end": float(tdlab.stack(log, "lam")[-1])}


if __name__ == "__main__":
    jobs = [(a, b, s) for a in ARMLIST for b in BETAS for s in range(SEEDS)]
    with ProcessPoolExecutor(max(1, os.cpu_count() // 2)) as ex:
        rows = list(ex.map(one, jobs, chunksize=8))
    json.dump(rows, open(os.path.join(HERE, "threshold.json"), "w"), indent=1)
    L = ["", "## Beta sweep at pressure 4 (60 seeds): where the |delta| bonus inverts the constraint", "",
         "| arm | beta | solution rate | unsafe returned | true rate | max true rate in training | "
         "unsafe-action share | TV share | reward | lambda end |", "|---|---|---|---|---|---|---|---|---|---|"]
    for arm in ARMLIST:
        for b in BETAS:
            rs = [r for r in rows if r["arm"] == arm and r["beta"] == b]
            m = lambda k: float(np.mean([r[k] for r in rs]))  # noqa: E731
            L.append(f"| {arm} | {b:g} | {m('solution'):.2f} | {m('violates'):.2f} | {m('true_rate'):.3f} | "
                     f"{m('train_rate_max'):.3f} | {m('unsafe_share'):.3f} | {m('tv_share'):.3f} | "
                     f"{m('true_reward'):.3f} | {m('lam_end'):.1f} |")
    txt = "\n".join(L) + "\n"
    open(os.path.join(HERE, "results.md"), "a").write(txt)
    print(txt)
