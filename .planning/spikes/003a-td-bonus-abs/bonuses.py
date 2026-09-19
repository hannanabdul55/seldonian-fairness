"""Internal rewards built on the agent's own TD error (spikes 003a-c), plus controls.

Each factory returns a stateful ``bonus(ctx) -> array`` for ``tdlab.TDBackend.bonus``;
``ctx`` is the per-step dict TDBackend logs, computed *before* the policy update:
``delta_c`` (the agent's TD error against its online critic's value), ``err_c`` (the
critic's error for the taken action), ``r`` (shaped reward), ``actions``. The bonus is
added to the shaped reward before GRPO's group normalisation, the place a reward-model
term would go in the LLM pipeline.
"""
import numpy as np


def abs_td(beta):
    """003a: pay surprise, ``beta * |delta_c|`` (QXplore-style, curiosity by error)."""
    def f(ctx):
        return beta * np.abs(ctx["delta_c"])
    return f


def pos_td(beta):
    """003b: pay good news only, ``beta * max(delta_c, 0)`` ("joy")."""
    def f(ctx):
        return beta * np.maximum(ctx["delta_c"], 0.0)
    return f


def learning_progress(beta, fast=0.2, slow=0.02):
    """
    003c: pay the decrease of the critic's error, per action (the "region"; contexts
    rarely repeat): ``LP_a = EMA_slow |err| - EMA_fast |err|``, positive while the
    critic is getting better at action ``a``. Irreducible noise sits in both EMAs and
    cancels (Oudeyer et al. 2007; Kim et al. 2020 gamma-progress).
    """
    st = {}

    def f(ctx):
        a, e = ctx["actions"], np.abs(ctx["err_c"])
        n = int(a.max()) + 1 if "fast" not in st else len(st["fast"])
        n = max(n, 5)
        if "fast" not in st:
            st["fast"], st["slow"] = np.full(n, np.nan), np.full(n, np.nan)
        lp = np.nan_to_num(st["slow"] - st["fast"])
        b = beta * np.maximum(lp[a], 0.0)
        for k in np.unique(a):
            m = e[a == k].mean()
            for key, rate in (("fast", fast), ("slow", slow)):
                st[key][k] = m if np.isnan(st[key][k]) else (1 - rate) * st[key][k] + rate * m
        return b
    return f


def abs_adv(beta, G=8):
    """E1 of LITERATURE.md: ``beta * |A|`` from the pre-bonus group-normalised reward."""
    def f(ctx):
        r = ctx["r"].reshape(-1, G)
        A = (r - r.mean(1, keepdims=True)) / (r.std(1, ddof=1, keepdims=True) + 1e-8)
        return beta * np.abs(A).ravel()
    return f


def random_matched(beta):
    """Control: ``beta * |N(0, s)|`` with ``s`` the running sd of ``delta_c``; same
    size as the |delta| bonus, independent of the action (Spurious Rewards control)."""
    st = {"s": None, "rng": np.random.default_rng(12345)}

    def f(ctx):
        s = float(np.std(ctx["delta_c"]))
        st["s"] = s if st["s"] is None else 0.9 * st["s"] + 0.1 * s
        return beta * np.abs(st["rng"].normal(0.0, st["s"], size=len(ctx["r"])))
    return f


ARMS = {"abs": abs_td, "pos": pos_td, "lp": learning_progress, "absA": abs_adv,
        "random": random_matched}


def abs_task_td(beta, lr=0.05):
    """
    003a variant / the fix suggested by the literature (RND, EIPO: keep the bonus out of
    the constrained objective): surprise about the **task** reward only. A second critic
    is trained on ``r_task = r + lam * v`` (the base reward, penalty removed), so the
    bonus is ``beta * |r_task - V_task(x)|`` and never scales with the multiplier.
    """
    st = {}

    def f(ctx):
        X, pi, a = ctx["X"], ctx["pi"], ctx["actions"]
        if "critic" not in st:
            from tdlab import LinearQCritic
            st["critic"] = LinearQCritic(X.shape[1], pi.shape[1], lr)
        c = st["critic"]
        r_task = ctx["r"] + ctx["lam"] * ctx["v"]
        Q = c.predict(X)
        b = beta * np.abs(r_task - (pi * Q).sum(axis=1))
        c.update(X, a, r_task)
        return b
    return f


ARMS["abs_task"] = abs_task_td
