"""SyntheticEnv with tunable prompt heterogeneity (spike 013).

The base env's violation weights ``U_a`` are independent across actions, so under the
uniform reference policy the per-prompt rate averages four sigmoids and its ICC tops out
near 0.22 however large ``u_scale`` is. Here every action shares a common direction:
``U_a = sqrt(shared) u0 + sqrt(1 - shared) U_a`` (``u0`` a standard normal row), so a prompt
that is risky for one action is risky for all, and ``shared`` and ``u_scale`` set the ICC.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                "001-grpo-advantage-vs-td"))
from tdlab import SyntheticEnv  # noqa: E402
from seldonian.llm.synthetic import sigmoid  # noqa: E402


class HeteroEnv(SyntheticEnv):
    def __init__(self, n_contexts, shared=0.0, **kw):
        super().__init__(n_contexts, **kw)
        rng = np.random.default_rng([self.seed, 99])
        u_scale = kw.get("u_scale", 0.5)
        u0 = rng.standard_normal(self.d) * u_scale
        self.U = np.sqrt(shared) * u0[None, :] + np.sqrt(1 - shared) * self.U
        self.p_v = sigmoid(self.X @ self.U.T + self.bias)
        self.mean_reward = self.X @ self.W_r.T + self.pressure * self.p_v


def icc_ref(env):
    idx = np.arange(env.n_contexts)
    W, c = env.uniform_params()
    p = np.sum(env.probs((W, c), idx) * env.p_v, axis=1)
    return float(p.var() / (p.mean() * (1 - p.mean()))), float(p.mean())


if __name__ == "__main__":
    for u in (0.5, 1.0, 2.0, 3.0):
        for sh in (0.0, 0.5, 0.8, 1.0):
            e = HeteroEnv(20000, shared=sh, u_scale=u, seed=0)
            i, m = icc_ref(e)
            print(f"u_scale {u} shared {sh}: ICC_ref {i:.3f} rate {m:.3f}")
