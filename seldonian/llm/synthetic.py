"""
Synthetic contextual bandit that stands in for an LLM, so the whole Seldonian pipeline
(:class:`seldonian.llm.policy.SeldonianLLMPolicy`, the judges, the reward shapers and
the bounds) can be run thousands of times against a *known* ground truth.

A prompt is a context ``x ~ N(0, I_d)`` named ``"ctx:<id>"``; a response is one of
``n_actions`` actions plus a sampled violation label, written ``"a=<a>;v=<v>"``. The
ground-truth violation probability is ``p_v(x, a) = sigmoid(u_a . x + b_a)`` and the
reward is ``r(x, a) = w_a . x + pressure * p_v(x, a) + noise``, so ``pressure`` is the
knob that makes the reward and the constraint conflict (0 = non-binding).

Because the policy is softmax-linear and the population is finite, the violation
rate and reward of any parameter vector are computed exactly
(:meth:`SyntheticEnv.true_rate`, :meth:`SyntheticEnv.true_reward`); that is what the
calibration harness (``scripts/synthetic_calibration.py``) compares the safety test
against. Everything is numpy, vectorised, and fast enough for hundreds of trials.
"""
import re

import numpy as np

from seldonian.llm.data import make_record
from seldonian.llm.judges import Judge
from seldonian.llm.policy import PolicyBackend
from seldonian.llm.rewards import Reward

#: action biases of the violation model; with ``u_scale=0.5`` the uniform policy
#: violates on ~13-14% of contexts (averaged over seeds), with two "safe" actions
#: (~2-5%) and two "unsafe" ones (~14% and ~30%)
DEFAULT_BIAS = (-4.5, -3.5, -2.2, -1.0)

_CTX_RE = re.compile(r"ctx:(\d+)")
_RESP_RE = re.compile(r"a=(\d+);v=(\d+)")


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def softmax(logits):
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def parse_contexts(prompts):
    """Context ids from ``"ctx:<id>"`` prompt strings (int array)."""
    out = np.empty(len(prompts), dtype=np.int64)
    for i, p in enumerate(prompts):
        m = _CTX_RE.match(p)
        if m is None:
            raise ValueError(f"not a synthetic prompt: {p!r}")
        out[i] = int(m.group(1))
    return out


def parse_responses(responses):
    """``(actions, violations)`` int arrays from ``"a=<a>;v=<v>"`` response strings."""
    a = np.empty(len(responses), dtype=np.int64)
    v = np.empty(len(responses), dtype=np.int64)
    for i, r in enumerate(responses):
        m = _RESP_RE.match(r)
        if m is None:
            raise ValueError(f"not a synthetic response: {r!r}")
        a[i], v[i] = int(m.group(1)), int(m.group(2))
    return a, v


def format_responses(actions, violations):
    return [f"a={a};v={v}" for a, v in zip(actions.tolist(), violations.tolist())]


def hash_uniform(ctx, action, label, seed):
    """
    Deterministic uniform(0, 1) per ``(context, action, label, seed)`` (splitmix64 on
    the packed integers), so a noisy judge gives the same verdict every time it sees
    the same (prompt, response) pair, like a real frozen judge.
    """
    with np.errstate(over="ignore"):
        z = (np.asarray(ctx, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15)
             + np.asarray(action, dtype=np.uint64) * np.uint64(0xC2B2AE3D27D4EB4F)
             + np.asarray(label, dtype=np.uint64) * np.uint64(0x165667B19E3779F9)
             + np.uint64(seed) * np.uint64(0xD6E8FEB86659FD93))
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))


class SyntheticEnv:
    """
    :param n_contexts: population size; the ground truth is the average over it
    :param d: context dimension
    :param n_actions: number of actions (responses)
    :param groups: group tags, assigned at random with ``group_frac`` in the first
    :param pressure: rho, the weight of ``p_v`` in the reward
    :param judge_noise: default ``(sensitivity, specificity)`` for :meth:`judge`
    :param bias: per-action violation bias ``b_a`` (default :data:`DEFAULT_BIAS`)
    :param u_scale: violation weights ``u_a ~ N(0, u_scale^2 I)``
    :param w_scale: reward weights ``w_a ~ N(0, w_scale^2 I)``; default ``0.7 / sqrt(d)``
    :param reward_noise: sd of the Gaussian reward noise
    :param seed: seed for the population and all parameters
    """

    def __init__(self, n_contexts, d=8, n_actions=4, groups=("adversarial", "benign"),
                 pressure=1.0, judge_noise=(1.0, 1.0), seed=0, group_frac=0.5, bias=None,
                 u_scale=0.5, w_scale=None, reward_noise=0.5):
        rng = np.random.default_rng(seed)
        self.n_contexts = int(n_contexts)
        self.d = d
        self.n_actions = n_actions
        self.groups = tuple(groups)
        self.pressure = float(pressure)
        self.judge_noise = tuple(judge_noise)
        self.seed = seed
        self.reward_noise = reward_noise
        self.w_scale = 0.7 / np.sqrt(d) if w_scale is None else w_scale
        if bias is None:
            if n_actions != len(DEFAULT_BIAS):
                bias = np.linspace(DEFAULT_BIAS[0], DEFAULT_BIAS[-1], n_actions)
            else:
                bias = DEFAULT_BIAS
        self.bias = np.asarray(bias, dtype=float)
        if self.bias.shape != (n_actions,):
            raise ValueError(f"bias needs {n_actions} entries")

        self.X = rng.standard_normal((self.n_contexts, d))
        if len(self.groups) == 1:
            self.group_idx = np.zeros(self.n_contexts, dtype=np.int64)
        else:
            first = rng.random(self.n_contexts) < group_frac
            self.group_idx = np.where(first, 0, 1 + rng.integers(0, len(self.groups) - 1,
                                                                 size=self.n_contexts))
        self.U = rng.standard_normal((n_actions, d)) * u_scale
        self.W_r = rng.standard_normal((n_actions, d)) * self.w_scale
        #: ground truth tables, (n_contexts, n_actions)
        self.p_v = sigmoid(self.X @ self.U.T + self.bias)
        self.mean_reward = self.X @ self.W_r.T + self.pressure * self.p_v

    # ------------------------------------------------------------ prompts

    def prompt(self, i):
        return f"ctx:{int(i)}"

    def records(self, n=None, seed=None):
        """
        Prompt records in the repo format (``prompt="ctx:<id>"``, ``task="synthetic"``),
        for the whole population or a random subset of ``n`` contexts.
        """
        if n is None or n >= self.n_contexts:
            idx = np.arange(self.n_contexts)
        else:
            rng = np.random.default_rng(self.seed if seed is None else seed)
            idx = np.sort(rng.choice(self.n_contexts, size=int(n), replace=False))
        return [make_record(self.prompt(i), self.groups[self.group_idx[i]], "synthetic")
                for i in idx]

    def indices(self, records=None, group=None):
        """Context ids of ``records`` (``None`` = the population), optionally one group."""
        if records is None:
            idx = np.arange(self.n_contexts)
        else:
            idx = parse_contexts([r["prompt"] if isinstance(r, dict) else r for r in records])
        if group is not None:
            idx = idx[self.group_idx[idx] == self.groups.index(group)]
        return idx

    # ------------------------------------------------------------ ground truth

    def probs(self, params, idx):
        """Policy ``pi(a | x)`` for the contexts ``idx``; ``params = (W, c)``."""
        W, c = params
        return softmax(self.X[idx] @ W.T + c)

    def true_rate(self, params, records=None, group=None):
        """Exact ``E_x sum_a pi(a|x) p_v(x, a)`` over ``records`` (``None`` = population)."""
        idx = self.indices(records, group)
        return float(np.mean(np.sum(self.probs(params, idx) * self.p_v[idx], axis=1)))

    def true_reward(self, params, records=None, group=None):
        """Exact expected reward (noise-free) over ``records`` (``None`` = population)."""
        idx = self.indices(records, group)
        return float(np.mean(np.sum(self.probs(params, idx) * self.mean_reward[idx], axis=1)))

    def judged_rate(self, params, judge, records=None, group=None):
        """
        Exact rate at which ``judge`` (a :class:`SyntheticJudge`) flags the policy: the
        violation probability passed through the judge's deterministic noise.
        """
        idx = self.indices(records, group)
        if judge.sensitivity == 1.0 and judge.specificity == 1.0:
            return self.true_rate(params, records, group)
        a = np.arange(self.n_actions)[None, :]
        ctx = idx[:, None]
        hit = hash_uniform(ctx, a, 1, judge.seed) < judge.sensitivity
        false_alarm = hash_uniform(ctx, a, 0, judge.seed) >= judge.specificity
        p = self.p_v[idx]
        q = p * hit + (1 - p) * false_alarm
        return float(np.mean(np.sum(self.probs(params, idx) * q, axis=1)))

    def uniform_params(self):
        return np.zeros((self.n_actions, self.d)), np.zeros(self.n_actions)

    # ------------------------------------------------------------ sampling

    def sample(self, params, idx, rng, temperature=1.0):
        """Sampled ``(actions, violation labels)`` for contexts ``idx``."""
        W, c = params
        pi = softmax((self.X[idx] @ W.T + c) / temperature)
        cdf = np.cumsum(pi, axis=1)
        actions = np.minimum((rng.random(len(idx))[:, None] > cdf).sum(axis=1),
                             self.n_actions - 1)
        v = (rng.random(len(idx)) < self.p_v[idx, actions]).astype(np.int64)
        return actions, v

    def reward(self, idx, actions, rng):
        """Noisy reward ``w_a . x + pressure * p_v(x, a) + N(0, reward_noise^2)``."""
        r = self.mean_reward[idx, actions]
        if self.reward_noise:
            r = r + rng.normal(0.0, self.reward_noise, size=len(r))
        return r

    def judge(self, cache_dir=None, seed=None):
        """A :class:`SyntheticJudge` with this environment's ``judge_noise``."""
        sens, spec = self.judge_noise
        return SyntheticJudge(sens, spec, seed=self.seed if seed is None else seed,
                              cache_dir=cache_dir)


class SyntheticJudge(Judge):
    """
    Reads the violation label ``v`` from a synthetic response, optionally through a
    noisy channel: a true violation is reported with probability ``sensitivity``, a
    non-violation is reported as one with probability ``1 - specificity``. The noise
    is a deterministic hash of (prompt, response, seed), so re-judging the same pair
    returns the same verdict.

    :param sensitivity: P(report 1 | v = 1)
    :param specificity: P(report 0 | v = 0)
    :param seed: seed of the noise hash
    """

    def __init__(self, sensitivity=1.0, specificity=1.0, seed=0, cache_dir=None):
        self.sensitivity = float(sensitivity)
        self.specificity = float(specificity)
        self.seed = int(seed)
        name = "synthetic_v"
        if self.sensitivity != 1.0 or self.specificity != 1.0:
            name += f"_sens{self.sensitivity:g}_spec{self.specificity:g}_seed{self.seed}"
        super().__init__(name, cache_dir)

    def _judge(self, prompts, responses, references=None):
        a, v = parse_responses(responses)
        if self.sensitivity == 1.0 and self.specificity == 1.0:
            return v.tolist()
        u = hash_uniform(parse_contexts(prompts), a, v, self.seed)
        out = np.where(v == 1, u < self.sensitivity, u >= self.specificity)
        return out.astype(int).tolist()


class SyntheticReward(Reward):
    """
    Noisy environment reward of a synthetic response: parses ``a=`` and returns
    ``w_a . x + pressure * p_v(x, a)`` plus fresh Gaussian noise.
    """

    name = "synthetic"

    def __init__(self, env, seed=0):
        self.env = env
        self.rng = np.random.default_rng(seed)

    def __call__(self, prompts, responses, groups=None, references=None):
        a, _ = parse_responses(responses)
        return self.env.reward(parse_contexts(prompts), a, self.rng)


class SyntheticBackend(PolicyBackend):
    """
    Softmax-linear policy ``pi(a|x) ~ exp(W_a . x + c_a)`` trained GRPO-style: every
    step samples ``prompts_per_step`` prompts and ``group_size`` actions per prompt,
    scores them through the reward object (so Lagrangian / composite shaping applies),
    normalises advantages within each group (``(r - mean) / (std + 1e-8)``) and takes
    an Adam (or SGD) step on the policy-gradient loss plus ``beta`` times the exact KL
    to the reference policy.

    :param env: a :class:`SyntheticEnv`
    :param max_steps: optimizer steps per :meth:`train`
    :param group_size: actions sampled per prompt
    :param prompts_per_step: prompts per step
    :param lr: learning rate
    :param beta: KL coefficient to the reference policy
    :param optimizer: ``"adam"`` or ``"sgd"``
    :param ref_bias: optional action logits of the reference policy (default uniform)
    :param seed: seed for sampling
    """

    def __init__(self, env, max_steps=200, group_size=8, prompts_per_step=8, lr=0.05,
                 beta=0.01, optimizer="adam", ref_bias=None, seed=0):
        if optimizer not in ("adam", "sgd"):
            raise ValueError(f"unknown optimizer {optimizer!r}")
        self.env = env
        self.max_steps = max_steps
        self.group_size = group_size
        self.prompts_per_step = prompts_per_step
        self.lr = lr
        self.beta = beta
        self.optimizer = optimizer
        self.rng = np.random.default_rng(seed)
        self.W = np.zeros((env.n_actions, env.d))
        self.c = np.zeros(env.n_actions) if ref_bias is None else np.asarray(ref_bias, float)
        self.ref_W, self.ref_c = self.W.copy(), self.c.copy()
        self._m = self._v = None
        self._t = 0
        self.steps_done = 0

    @property
    def params(self):
        return self.W.copy(), self.c.copy()

    @property
    def ref_params(self):
        return self.ref_W.copy(), self.ref_c.copy()

    def generate(self, prompts, max_new_tokens=256, temperature=1.0):
        idx = parse_contexts(prompts)
        actions, v = self.env.sample((self.W, self.c), idx, self.rng, temperature)
        return format_responses(actions, v)

    def gradient(self, idx, actions, advantages):
        """
        Gradient of ``-mean(A * log pi(a|x)) + beta * mean_x KL(pi(.|x) || pi_ref(.|x))``
        w.r.t. ``(W, c)``; ``idx`` repeats each prompt once per sampled action.
        """
        x = self.env.X[idx]
        logits = x @ self.W.T + self.c
        pi = softmax(logits)
        onehot = np.zeros_like(pi)
        onehot[np.arange(len(idx)), actions] = 1.0
        # d(-A log pi_a)/d logits = -A (e_a - pi)
        g_logits = -(advantages[:, None] * (onehot - pi)) / len(idx)
        if self.beta:
            ux, inv = np.unique(idx, return_inverse=True)
            xu = self.env.X[ux]
            logp = np.log(softmax(xu @ self.W.T + self.c))
            logq = np.log(softmax(xu @ self.ref_W.T + self.ref_c))
            p = np.exp(logp)
            diff = logp - logq
            # d KL / d logits_a = p_a (log p_a - log q_a - KL)
            g_kl = p * (diff - np.sum(p * diff, axis=1, keepdims=True)) / len(ux)
            g_W = g_logits.T @ x + self.beta * g_kl.T @ xu
            g_c = g_logits.sum(axis=0) + self.beta * g_kl.sum(axis=0)
        else:
            g_W = g_logits.T @ x
            g_c = g_logits.sum(axis=0)
        return g_W, g_c

    def _apply(self, g_W, g_c, b1=0.9, b2=0.999, eps=1e-8):
        if self.optimizer == "sgd":
            self.W -= self.lr * g_W
            self.c -= self.lr * g_c
            return
        g = np.concatenate([g_W.ravel(), g_c])
        if self._m is None:
            self._m, self._v = np.zeros_like(g), np.zeros_like(g)
        self._t += 1
        self._m = b1 * self._m + (1 - b1) * g
        self._v = b2 * self._v + (1 - b2) * g * g
        step = self.lr * (self._m / (1 - b1 ** self._t)) / (
            np.sqrt(self._v / (1 - b2 ** self._t)) + eps)
        self.W -= step[:self.W.size].reshape(self.W.shape)
        self.c -= step[self.W.size:]

    def train(self, records, reward, on_step):
        records = list(records)
        prompts_all = [r["prompt"] for r in records]
        idx_all = parse_contexts(prompts_all)
        groups_all = [r.get("group") for r in records]
        refs_all = [r.get("reference") for r in records]
        k, G = min(self.prompts_per_step, len(records)), self.group_size
        for step in range(1, self.max_steps + 1):
            pick = np.repeat(self.rng.choice(len(records), size=k, replace=False), G)
            idx = idx_all[pick]
            actions, v = self.env.sample((self.W, self.c), idx, self.rng)
            prompts = [prompts_all[i] for i in pick]
            responses = format_responses(actions, v)
            r = np.asarray(reward(prompts, responses, groups=[groups_all[i] for i in pick],
                                  references=[refs_all[i] for i in pick]), dtype=float)
            r = r.reshape(k, G)
            adv = (r - r.mean(axis=1, keepdims=True)) / (r.std(axis=1, ddof=1, keepdims=True)
                                                         + 1e-8)
            self._apply(*self.gradient(idx, actions, adv.ravel()))
            self.steps_done += 1
            on_step(step)

    def save_checkpoint(self, tag):
        return self.params

    def load_checkpoint(self, handle):
        W, c = handle
        self.W, self.c = W.copy(), c.copy()
