"""Spike 008: does a learning-progress bonus speed the drift into U?

003c found the learning-progress (LP) bonus the only safe internal-reward shape, with one
cost: it pays for visiting regions the critic is still learning, the unsafe ones included.
Spike 004's forbidden capability is exactly such a region: capability on F is being
learned (through A) while F is never rewarded. This adds 003c's bonus to 004's lab.

The bonus (003c, ``bonuses.learning_progress``): a linear critic ``Q(x, a)`` per action
learns the trained reward online; ``err = r - Q(x, a)``; per region, LP = EMA_slow |err| -
EMA_fast |err|; the bonus ``beta * max(LP, 0)`` is added to the trained reward before
GRPO's group normalisation. Regions: ``action`` (003c's choice; F and A share them) or
``flag_action`` (F answers are their own regions). Control: ``random``, a bonus of the
same running size drawn independently of the action.

Arms: grpo and lag_floor (004's winner), each without a bonus, with LP at beta 1 and 2
(both region choices), with the random control at beta 2, and with LP / random paid on
allowed-task prompts only (``_taskonly``). Checks every 25, tau 0.12.

    ../../../.venv/bin/python lpdrift.py [--seeds 100]
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
from seldonian.llm.synthetic import format_responses, parse_contexts  # noqa: E402

TAU, EVERY = 0.12, 25


class LPBackend(F.ForbiddenBackend):
    """ForbiddenBackend plus an online critic and an internal bonus on the trained reward."""

    bonus_kind = None          # None | "lp" | "random"
    region = "action"          # "action" | "flag_action"
    beta = 0.0
    task_only = False          # pay the bonus on allowed-task prompts only
    critic_lr = 0.2
    fast, slow = 0.2, 0.02

    def train(self, records, reward, on_step):
        env = self.env
        records = list(records)
        prompts_all = [r["prompt"] for r in records]
        idx_all = parse_contexts(prompts_all)
        groups_all = [r.get("group") for r in records]
        k, G = min(self.prompts_per_step, len(records)), self.group_size
        nA = env.n_actions
        Q = np.zeros((nA, env.d))
        nreg = nA * (2 if self.region == "flag_action" else 1)
        ema_f, ema_s = np.full(nreg, np.nan), np.full(nreg, np.nan)
        sd = None
        rng = np.random.default_rng(self.rng.integers(1 << 30))
        self.bonus_log = []
        self._monitor(0)
        for step in range(1, self.max_steps + 1):
            pick = np.repeat(self.rng.choice(len(records), size=k, replace=False), G)
            idx = idx_all[pick]
            actions, v = env.sample((self.W, self.c), idx, self.rng)
            r = np.asarray(reward([prompts_all[i] for i in pick], format_responses(actions, v),
                                  groups=[groups_all[i] for i in pick]), dtype=float)
            X = env.X[idx]
            err = r - (Q[actions] * X).sum(axis=1)
            flag = X[:, -1].astype(int)
            reg = actions + (nA * flag if self.region == "flag_action" else 0)
            b = np.zeros_like(r)
            if self.bonus_kind == "lp":
                lp = np.nan_to_num(ema_s - ema_f)
                b = self.beta * np.maximum(lp[reg], 0.0)
            elif self.bonus_kind == "random":
                # size-matched control: the running sd of |err| sets the scale, as in 003
                s = float(np.abs(err).std())
                sd = s if sd is None else 0.9 * sd + 0.1 * s
                b = self.beta * 0.1 * np.abs(rng.normal(0.0, sd, size=len(r)))
            if self.task_only:
                b = b * (flag == 0)
            for g in np.unique(reg):
                m = np.abs(err[reg == g]).mean()
                for arr, rate in ((ema_f, self.fast), (ema_s, self.slow)):
                    arr[g] = m if np.isnan(arr[g]) else (1 - rate) * arr[g] + rate * m
            # normalised LMS averaged per action (as tdlab.LinearQCritic): plain LMS at this
            # rate diverges (|x|^2 is 9-15), and summing per-sample steps overshoots
            cnt = np.bincount(actions, minlength=nA)[actions]
            np.add.at(Q, actions, self.critic_lr * (err / cnt)[:, None] * X
                      / (X * X).sum(1, keepdims=True))
            self.bonus_log.append(dict(step=step, mean=float(b.mean()),
                                       on_f=float(b[flag == 1].mean()) if (flag == 1).any() else 0.0,
                                       on_task=float(b[flag == 0].mean()) if (flag == 0).any() else 0.0))
            rr = (r + b).reshape(k, G)
            adv = (rr - rr.mean(axis=1, keepdims=True)) / (rr.std(axis=1, ddof=1, keepdims=True) + 1e-8)
            self._apply(*self.gradient(idx, actions, adv.ravel()))
            self.steps_done += 1
            ex = env.exact((self.W, self.c))
            self.log.append(dict(step=step, harm=ex["harm"], task_acc=ex["task"]["acc"],
                                 will=ex["forbidden"]["answer"], cap_f=ex["forbidden"]["cap"],
                                 cap_twin=ex["twin"]["cap"],
                                 lam=float(sum(getattr(reward, "lambdas", {}).values()))))
            if step % self.monitor_every == 0:
                self._monitor(step)
            on_step(step)


ARMS = {"none": (None, "action", 0.0)}
for beta in (1.0, 2.0):
    ARMS[f"lp{beta:g}"] = ("lp", "action", beta)
    ARMS[f"lp{beta:g}_flag"] = ("lp", "flag_action", beta)
ARMS["random2"] = ("random", "action", 2.0)
ARMS["lp2_taskonly"] = ("lp", "action", 2.0, True)
ARMS["random2_taskonly"] = ("random", "action", 2.0, True)
METHODS = {"grpo": dict(method="grpo"),
           "lag_floor": dict(method="lag", lam_floor=5.0, floor_always=True)}


def one(job):
    meth, arm, seed = job
    kind, region, beta, *rest = ARMS[arm]
    saved = F.ForbiddenBackend
    LPBackend.bonus_kind, LPBackend.region, LPBackend.beta = kind, region, beta
    LPBackend.task_only = bool(rest and rest[0])
    F.ForbiddenBackend = LPBackend
    bonus_log = []
    orig_train = LPBackend.train

    def train(self, *a, **k):
        out = orig_train(self, *a, **k)
        bonus_log.extend(self.bonus_log)
        return out
    LPBackend.train = train
    try:
        row = F.run(seed, tau=TAU, predict_every=EVERY, **METHODS[meth])
    finally:
        F.ForbiddenBackend = saved
        LPBackend.train = orig_train
    bl = bonus_log
    h = len(bl) // 2
    return dict(method=meth, arm=arm, seed=seed, entered_U=row["entered_U"],
                first_U=row["first_U_step"], peak_harm=row["peak_harm"],
                steps_in_U=row["steps_in_U"], final_harm=row["final_harm"],
                task_acc=row["final_task_acc"], solution=row["solution"],
                violates=row["violates"], lam_end=row["lam_traj"][-1],
                bonus_f_early=float(np.mean([x["on_f"] for x in bl[:h]])),
                bonus_task_early=float(np.mean([x["on_task"] for x in bl[:h]])),
                bonus_f_late=float(np.mean([x["on_f"] for x in bl[h:]])),
                bonus_task_late=float(np.mean([x["on_task"] for x in bl[h:]])))


def report(rows):
    lines = ["# Spike 008 results", "", f"tau {TAU}, checks every {EVERY}; means over seeds.", "",
             "| method | arm | entered U | first U step (median) | steps in U | peak harm | final harm | "
             "task acc | solution | lam end | bonus on F early/late | bonus on A early/late |",
             "|---" * 12 + "|"]
    for meth in METHODS:
        for arm in ARMS:
            rs = [r for r in rows if r["method"] == meth and r["arm"] == arm]
            if not rs:
                continue
            fu = [r["first_U"] for r in rs if r["first_U"] is not None]
            sol = [bool(r["solution"]) for r in rs if r["solution"] is not None]
            lines.append(
                f"| {meth} | {arm} | {np.mean([r['entered_U'] for r in rs]):.3f} | "
                f"{np.median(fu) if fu else '-'} | {np.mean([r['steps_in_U'] for r in rs]):.1f} | "
                f"{np.mean([r['peak_harm'] for r in rs]):.3f} | {np.mean([r['final_harm'] for r in rs]):.3f} | "
                f"{np.mean([r['task_acc'] for r in rs]):.3f} | {np.mean(sol) if sol else '-'} | "
                f"{np.mean([r['lam_end'] for r in rs]):.1f} | "
                f"{np.mean([r['bonus_f_early'] for r in rs]):.3f}/{np.mean([r['bonus_f_late'] for r in rs]):.3f} | "
                f"{np.mean([r['bonus_task_early'] for r in rs]):.3f}/{np.mean([r['bonus_task_late'] for r in rs]):.3f} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    js = [(m, arm, s) for m in METHODS for arm in ARMS for s in range(a.seeds)]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(one, js, chunksize=8))
    json.dump(rows, open(os.path.join(HERE, "results.json"), "w"))
    text = report(rows)
    open(os.path.join(HERE, "results.md"), "w").write(text)
    print(text)
