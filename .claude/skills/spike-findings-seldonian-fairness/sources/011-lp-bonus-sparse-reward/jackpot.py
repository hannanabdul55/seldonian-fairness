"""Spike 011: does the learning-progress bonus buy anything where exploration matters?

003c found learning progress (LP) the only internal reward that is safe under the
Seldonian Lagrangian, and useless in its env: four learnable actions are learned in a few
steps, so there was nothing to explore. This env adds a sparse, deceptive action.

JackpotEnv = 003's noisy-TV env (actions 0-3 as the base env, 4 = the noisy TV) plus
action 5, the "jackpot": never a violation; mean reward ``safe_best + gain`` on contexts
with ``x . u_j > 1.28`` (about 10%) and ``safe_best - loss`` elsewhere, so it is bad on
average and good only where the policy has to learn to use it. The reference policy
rarely tries it (logit -3, about 1% of samples). The noisy TV stays, so an exploration
bonus that chases surprise is caught out.

Arms (seldonian_lag, pressure 4 as 003): no bonus, LP beta 1 / 2, |delta| beta 1
(003a's shape, for contrast), and a size-matched random control. 60 seeds.

    ../../../.venv/bin/python jackpot.py [--seeds 60]
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "001-grpo-advantage-vs-td"))
sys.path.insert(0, os.path.join(HERE, "..", "003a-td-bonus-abs"))
import tdlab  # noqa: E402
from bonuses import ARMS as BONUS  # noqa: E402
from seldonian.llm.data import split_prompts  # noqa: E402
from seldonian.llm.policy import Constraint, SeldonianLLMPolicy  # noqa: E402
from seldonian.llm.rewards import LagrangianReward  # noqa: E402
from seldonian.llm.synthetic import DEFAULT_BIAS, SyntheticEnv, SyntheticJudge, SyntheticReward  # noqa: E402

GAIN, LOSS, CUT = 1.0, 1.0, 1.28


class JackpotEnv(tdlab.NoisyTVEnv):
    def __init__(self, n_contexts, tv_noise=3.0, tv_gap=0.3, **kw):
        # build a 6-action env: NoisyTVEnv sets n_actions = 5, so set up the 6th here
        SyntheticEnv.__init__(self, n_contexts, n_actions=6,
                              bias=tuple(DEFAULT_BIAS) + (-20.0, -20.0), **kw)
        rng = np.random.default_rng(kw.get("seed", 0) + 99)
        for a in (4, 5):
            self.U[a] = 0.0
            self.p_v[:, a] = 0.0
            self.W_r[a] = 0.0
        safe_best = self.mean_reward[:, :2].max(axis=1)
        self.tv_mean = float(safe_best.mean() - tv_gap)
        self.mean_reward[:, 4] = self.tv_mean
        self.tv_noise = float(tv_noise)
        u = rng.standard_normal(self.d)
        u /= np.linalg.norm(u)
        self.jackpot = (self.X @ u) > CUT
        self.mean_reward[:, 5] = np.where(self.jackpot, safe_best + GAIN, safe_best - LOSS)

    def reward(self, idx, actions, rng):
        r = self.mean_reward[idx, actions]
        sd = np.where(actions == 4, self.tv_noise, self.reward_noise)
        return r + rng.normal(0.0, 1.0, size=len(r)) * sd


def run(seed, bonus=None, pressure=4.0, steps=200, n=1000):
    env = JackpotEnv(20000, d=8, pressure=pressure, seed=seed)
    records = env.records(n, seed=seed)
    d_c, d_s = split_prompts(records, test_size=0.4, seed=seed)
    judge = SyntheticJudge(1.0, 1.0, seed=seed)
    base = SyntheticReward(env, seed=seed + 1)
    ref_bias = np.array([0, 0, 0, 0, 0, -3.0])
    backend = tdlab.TDBackend(env, max_steps=steps, group_size=8, prompts_per_step=8, lr=0.05,
                              beta=0.01, ref_bias=ref_bias, seed=seed + 2)
    backend.critic_lr = 0.05
    backend.bonus = bonus() if callable(bonus) else None
    ref = backend.params
    c = Constraint("harm", judge, threshold=env.true_rate(ref) + 0.03, group=None, bound="ttest")
    reward = LagrangianReward(base, [(judge, None)], names=["harm"], lam0=5.0, eta=100.0,
                              lam_max=20.0)
    policy = SeldonianLLMPolicy(backend, d_c, d_s, reward=reward, constraints=[c], delta=0.1,
                                predict_every=25, predict_n=512, seed=seed)
    solution = policy.fit(seldonian=True) is not None
    P = env.probs(backend.params, np.arange(env.n_contexts))
    Pr = env.probs(ref, np.arange(env.n_contexts))
    # the best achievable safe reward: per context, best of the safe actions 0, 1, 5
    best = env.mean_reward[:, [0, 1, 5]].max(axis=1).mean()
    b = tdlab.stack(backend.log, "bonus")
    h = len(backend.log) // 2
    return dict(seed=seed, solution=bool(solution),
                true_rate=env.true_rate(backend.params), threshold=float(c.threshold),
                violates=bool(env.true_rate(backend.params) > c.threshold),
                true_reward=env.true_reward(backend.params), ref_reward=env.true_reward(ref),
                best_safe=float(best),
                jackpot_use_on=float(P[env.jackpot, 5].mean()),
                jackpot_use_off=float(P[~env.jackpot, 5].mean()),
                jackpot_ref=float(Pr[:, 5].mean()),
                tv_share=float(P[:, 4].mean()),
                bonus_early=float(b[:h].mean()), bonus_late=float(b[h:].mean()),
                lam_end=float(tdlab.stack(backend.log, "lam")[-1]))


def learning_progress(beta, fast=0.2, slow=0.02):
    """003c's bonus (bonuses.learning_progress), with the per-action tables sized from the
    env: the original sizes them from the first batch's largest action (at least 5), which
    overflows on a 6-action env."""
    st = {}

    def f(ctx):
        a, e = ctx["actions"], np.abs(ctx["err_c"])
        if "fast" not in st:
            n = ctx["backend"].env.n_actions
            st["fast"], st["slow"] = np.full(n, np.nan), np.full(n, np.nan)
        lp = np.nan_to_num(st["slow"] - st["fast"])
        b = beta * np.maximum(lp[a], 0.0)
        for k in np.unique(a):
            m = e[a == k].mean()
            for key, rate in (("fast", fast), ("slow", slow)):
                st[key][k] = m if np.isnan(st[key][k]) else (1 - rate) * st[key][k] + rate * m
        return b
    return f


BONUS = {**BONUS, "lp": learning_progress}
ARMS = {"none": None, "lp1": ("lp", 1.0), "lp2": ("lp", 2.0), "abs1": ("abs", 1.0),
        "random2": ("random", 2.0)}


def one(job):
    arm, seed = job
    spec = ARMS[arm]
    bonus = None if spec is None else (lambda: BONUS[spec[0]](spec[1]))
    t0 = time.time()
    row = run(seed, bonus=bonus)
    row.update(arm=arm, seconds=time.time() - t0)
    return row


def report(rows):
    keys = ["solution", "violates", "true_rate", "true_reward", "jackpot_use_on", "jackpot_use_off",
            "tv_share", "bonus_early", "bonus_late", "lam_end"]
    lines = ["# Spike 011 results", "",
             f"Jackpot on {100 * 0.1:.0f}% of contexts (gain {GAIN}, loss {LOSS} elsewhere); "
             "reference uses it ~1%. Means over seeds; `best safe` is the reward of the best "
             f"safe action per context: {np.mean([r['best_safe'] for r in rows]):.3f}; "
             f"reference reward {np.mean([r['ref_reward'] for r in rows]):.3f}.", "",
             "| arm | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for arm in ARMS:
        rs = [r for r in rows if r["arm"] == arm]
        lines.append(f"| {arm} | " + " | ".join(f"{np.mean([float(r[k]) for r in rs]):.3f}" for k in keys) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=60)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    js = [(arm, s) for arm in ARMS for s in range(a.seeds)]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(one, js, chunksize=4))
    json.dump(rows, open(os.path.join(HERE, "results.json"), "w"))
    text = report(rows)
    open(os.path.join(HERE, "results.md"), "w").write(text)
    print(text)
