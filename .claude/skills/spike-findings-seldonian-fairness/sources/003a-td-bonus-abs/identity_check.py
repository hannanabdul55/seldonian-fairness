"""Verify the identity behind 003a's beta > 1 collapse.

Within a GRPO group every sample shares the prompt, so the critic's value V(x) is a
constant of the group. With delta = r - V(x),

    r + beta*|delta| = (1 - beta) * r + 2*beta*max(delta, 0) + beta*V(x),

and group normalisation is invariant to a positive affine map of the group's rewards.
So for beta < 1 the |delta| bonus is exactly the positive-surprise objective at strength
2*beta/(1-beta); at beta = 1 the task reward drops out entirely; for beta > 1 its
coefficient is negative, i.e. the agent maximises the negative of the constrained reward.

    ../../../.venv/bin/python identity_check.py
"""
import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "001-grpo-advantage-vs-td"))
import tdlab  # noqa: E402
from bonuses import abs_td, pos_td  # noqa: E402

if __name__ == "__main__":
    print(f"{'beta(abs)':>9} {'beta(pos)':>9} {'seed':>4}  equal true rate / reward / TV share")
    for b in (0.25, 0.5, 0.8):
        p = 2 * b / (1 - b)
        for seed in (0, 1, 2):
            r1, _ = tdlab.run(seed, method="seldonian_lag", pressure=4.0, env="noisy_tv",
                              steps=200, bonus=lambda b=b: abs_td(b))
            r2, _ = tdlab.run(seed, method="seldonian_lag", pressure=4.0, env="noisy_tv",
                              steps=200, bonus=lambda p=p: pos_td(p))
            same = all(np.isclose(r1[k], r2[k], atol=1e-9) for k in
                       ("true_rate", "true_reward", "tv_share"))
            print(f"{b:>9g} {p:>9.3f} {seed:>4}  {same}  "
                  f"{r1['true_rate']:.6f}/{r1['true_reward']:.6f}/{r1['tv_share']:.6f}")
