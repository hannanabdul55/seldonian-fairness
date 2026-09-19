"""TD-error lab on the synthetic contextual bandit (shared by spikes 001-003).

The synthetic environment knows ``Q(x, a)`` exactly, so for the current policy the
value ``V(x) = sum_a pi(a|x) Q(x, a)`` and the true per-episode TD error
``delta = r - V(x)`` are computable. A bandit episode is one step, so the TD error is
the reward prediction error, and it splits into the true advantage
``Q(x, a) - V(x)`` (learnable) and noise ``r - Q(x, a)`` (reward noise plus the
judge-label draw times the multiplier).

``TDBackend`` is :class:`SyntheticBackend` with the same GRPO step, plus:

- per-episode logging of the oracle TD error, its advantage/noise split, the agent's own
  TD error (a linear Q critic trained online on the same rewards), the group-centred
  reward and GRPO's group-normalised advantage;
- an optional internal reward ``bonus(ctx)`` added to the trained reward before the
  group normalisation (spike 003).

Q is always the *shaped* reward the trainer sees: ``mean_reward - lam * p_v`` under the
Lagrangian (perfect judge), ``mean_reward`` under plain GRPO.
"""
import os
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)

from seldonian.llm.data import split_prompts  # noqa: E402
from seldonian.llm.policy import Constraint, SeldonianLLMPolicy  # noqa: E402
from seldonian.llm.rewards import LagrangianReward  # noqa: E402
from seldonian.llm.synthetic import (DEFAULT_BIAS, SyntheticBackend, SyntheticEnv,  # noqa: E402
                                     SyntheticJudge, SyntheticReward, format_responses,
                                     parse_contexts, softmax)


class NoisyTVEnv(SyntheticEnv):
    """
    :class:`SyntheticEnv` plus one extra "noisy TV" action: a constant mean reward
    ``tv_mean`` (default: the population mean of the best safe action minus ``tv_gap``),
    never a violation, and reward noise ``tv_noise`` instead of ``reward_noise``.
    Nothing is learnable about it; it only produces surprise.
    """

    def __init__(self, n_contexts, tv_noise=3.0, tv_gap=0.3, **kw):
        kw["n_actions"] = 5
        kw["bias"] = tuple(DEFAULT_BIAS) + (-20.0,)
        super().__init__(n_contexts, **kw)
        self.U[4] = 0.0
        self.p_v[:, 4] = 0.0
        safe_best = self.mean_reward[:, :2].max(axis=1).mean()
        self.tv_mean = float(safe_best - tv_gap)
        self.W_r[4] = 0.0
        self.mean_reward[:, 4] = self.tv_mean
        self.tv_noise = float(tv_noise)

    def reward(self, idx, actions, rng):
        r = self.mean_reward[idx, actions]
        sd = np.where(actions == 4, self.tv_noise, self.reward_noise)
        return r + rng.normal(0.0, 1.0, size=len(r)) * sd


class LinearQCritic:
    """``Q(x, a) = w_a . x + b_a`` trained by SGD on observed (x, a, r)."""

    def __init__(self, d, n_actions, lr=0.05):
        self.w = np.zeros((n_actions, d))
        self.b = np.zeros(n_actions)
        self.lr = lr

    def predict(self, X):
        return X @ self.w.T + self.b

    def update(self, X, a, r):
        """One SGD step on the per-action mean squared error of the batch."""
        err = r - (np.einsum("ij,ij->i", X, self.w[a]) + self.b[a])
        cnt = np.bincount(a, minlength=len(self.b))[a]
        np.add.at(self.w, a, self.lr * (err / cnt)[:, None] * X)
        np.add.at(self.b, a, self.lr * err / cnt)
        return err


class TDBackend(SyntheticBackend):
    """:class:`SyntheticBackend` that logs TD errors and can add an internal reward."""

    critic_lr = 0.05
    #: callable(ctx) -> per-episode bonus array, or None; ``ctx`` is the dict logged
    #: for the step (see ``train``) plus ``backend``
    bonus = None

    def train(self, records, reward, on_step):
        env = self.env
        records = list(records)
        prompts_all = [r["prompt"] for r in records]
        idx_all = parse_contexts(prompts_all)
        groups_all = [r.get("group") for r in records]
        refs_all = [r.get("reference") for r in records]
        k, G = min(self.prompts_per_step, len(records)), self.group_size
        self.critic = LinearQCritic(env.d, env.n_actions, self.critic_lr)
        self.log = []
        for step in range(1, self.max_steps + 1):
            pick = np.repeat(self.rng.choice(len(records), size=k, replace=False), G)
            idx = idx_all[pick]
            X = env.X[idx]
            pi = softmax(X @ self.W.T + self.c)
            actions, v = env.sample((self.W, self.c), idx, self.rng)
            prompts = [prompts_all[i] for i in pick]
            responses = format_responses(actions, v)
            lam = float(sum(getattr(reward, "lambdas", {}).values()))
            r = np.asarray(reward(prompts, responses, groups=[groups_all[i] for i in pick],
                                  references=[refs_all[i] for i in pick]), dtype=float)
            n = np.arange(len(idx))
            Q = env.mean_reward[idx] - lam * env.p_v[idx]
            V = (pi * Q).sum(axis=1)
            Qc = self.critic.predict(X)
            Vc = (pi * Qc).sum(axis=1)
            ctx = {
                "step": step, "lam": lam, "idx": idx, "actions": actions, "v": v,
                "r": r, "pi_a": pi[n, actions],
                "delta": r - V,                      # oracle TD error
                "adv_true": Q[n, actions] - V,       # its learnable part
                "delta_c": r - Vc,                   # the agent's own TD error
                "err_c": r - Qc[n, actions],         # critic's prediction error for (x, a)
            }
            b = np.zeros_like(r)
            if self.bonus is not None:
                b = np.asarray(self.bonus({**ctx, "backend": self}), dtype=float)
            rt = (r + b).reshape(k, G)
            centred = rt - rt.mean(axis=1, keepdims=True)
            sd = rt.std(axis=1, ddof=1, keepdims=True)
            adv = centred / (sd + 1e-8)
            W0, c0 = self.W.copy(), self.c.copy()
            self._apply(*self.gradient(idx, actions, adv.ravel()))
            self.critic.update(X, actions, r)
            p_true = env.probs((self.W, self.c), np.arange(env.n_contexts))
            ctx.update(bonus=b, centred=centred.ravel(), adv=adv.ravel(),
                       group_sd=np.repeat(sd.ravel(), G),
                       dtheta=float(np.sqrt(((self.W - W0) ** 2).sum() + ((self.c - c0) ** 2).sum())),
                       true_rate=float((p_true * env.p_v).sum(axis=1).mean()),
                       true_reward=float((p_true * env.mean_reward).sum(axis=1).mean()),
                       action_share=p_true.mean(axis=0))
            self.log.append(ctx)
            self.steps_done += 1
            on_step(step)


DEFAULTS = dict(method="seldonian_lag", pressure=1.0, n=1000, steps=200, group_size=8,
                prompts_per_step=8, lr=0.05, beta=0.01, delta=0.1, bound="ttest",
                predict_every=25, predict_n=512, eta=100.0, lam0=5.0, lam_max=20.0,
                lam_floor=0.0, eta_down=None, floor_always=False, margin=0.03,
                population=20000, d=8, env="base", tv_noise=3.0, tv_gap=0.3,
                critic_lr=0.05, bonus=None)


def run(seed=0, **overrides):
    """One pipeline run (as ``scripts/synthetic_calibration.py``); returns (row, log)."""
    cfg = {**DEFAULTS, **overrides}
    t0 = time.time()
    env_kw = dict(n_contexts=cfg["population"], d=cfg["d"], pressure=cfg["pressure"], seed=seed)
    env = (NoisyTVEnv(tv_noise=cfg["tv_noise"], tv_gap=cfg["tv_gap"], **env_kw)
           if cfg["env"] == "noisy_tv" else SyntheticEnv(**env_kw))
    records = env.records(cfg["n"], seed=seed)
    d_c, d_s = split_prompts(records, test_size=0.4, seed=seed)
    judge = SyntheticJudge(1.0, 1.0, seed=seed)
    base = SyntheticReward(env, seed=seed + 1)
    backend = TDBackend(env, max_steps=cfg["steps"], group_size=cfg["group_size"],
                        prompts_per_step=cfg["prompts_per_step"], lr=cfg["lr"],
                        beta=cfg["beta"], seed=seed + 2)
    backend.critic_lr = cfg["critic_lr"]
    backend.bonus = cfg["bonus"]() if callable(cfg["bonus"]) else None
    ref = backend.params
    c = Constraint("harm", judge, threshold=env.true_rate(ref) + cfg["margin"], group=None,
                   bound=cfg["bound"])
    if cfg["method"] == "seldonian_lag":
        reward = LagrangianReward(base, [(judge, None)], names=["harm"], lam0=cfg["lam0"],
                                  eta=cfg["eta"], lam_max=cfg["lam_max"],
                                  lam_floor=cfg["lam_floor"], eta_down=cfg["eta_down"],
                                  floor_always=cfg["floor_always"])
    else:
        reward = base
    policy = SeldonianLLMPolicy(backend, d_c, d_s, reward=reward, constraints=[c],
                                delta=cfg["delta"], predict_every=cfg["predict_every"],
                                predict_n=cfg["predict_n"], seed=seed)
    if cfg["method"] == "grpo":
        policy.fit(seldonian=False)
        solution, sel = True, None
    else:
        solution = policy.fit(seldonian=True) is not None
        sel = policy.selected["step"]
    params = backend.params
    true_rate = env.true_rate(params)
    row = dict(seed=seed, method=cfg["method"], pressure=cfg["pressure"],
               threshold=float(c.threshold), ref_true=float(env.true_rate(ref)),
               ref_reward=float(env.true_reward(ref)), solution=bool(solution),
               selected_step=sel, true_rate=float(true_rate),
               true_reward=float(env.true_reward(params)),
               violates=bool(true_rate > c.threshold),
               n_feasible=int(sum(h.feasible for h in policy.history)) if policy.history else None,
               history=[(h.step, bool(h.feasible), float(h.rates["harm"])) for h in policy.history],
               seconds=time.time() - t0)
    if cfg["env"] == "noisy_tv":
        row["tv_share"] = float(env.probs(params, np.arange(env.n_contexts))[:, 4].mean())
    return row, backend.log


def stack(log, key):
    """(steps, episodes) array of a per-episode key, or (steps,) of a scalar key."""
    return np.array([s[key] for s in log])
