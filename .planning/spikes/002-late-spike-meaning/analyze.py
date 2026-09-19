"""Spike 002: what does a late TD-error spike mean?

Every run trains for ``--steps`` (300); everything a predictor may use is computed on the
first ``--horizon`` (200) steps, "the run as it would be stopped", and steps 201-300 are
"if training continued". The hypothesis from reports/ideas.md (2026-09-14): runs whose
TD-error spikes come late are still moving, so their safety outcome is the least likely
to hold under continued training.

The agent's TD error is its own online critic's, ``delta_c = r - V_critic(x)``;
it splits exactly into
    delta_c = (r - Q(x,a))          noise: reward noise + lambda * (v - p_v)
            + (Q(x,a) - V(x))       true advantage (learnable)
            + (V(x) - V_critic(x))  critic lag (what the agent has not caught up with)
with the oracle Q, V of the shaped reward.

    ../../../.venv/bin/python analyze.py [--seeds 60]

Writes results.md, results.json and viewer_data.js (for viewer.html).
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "001-grpo-advantage-vs-td"))
import tdlab  # noqa: E402

SETTINGS = {
    "grpo p1": dict(method="grpo", pressure=1.0),
    "grpo p4": dict(method="grpo", pressure=4.0),
    "lag p1": dict(method="seldonian_lag", pressure=1.0),
    "lag p4": dict(method="seldonian_lag", pressure=4.0),
    "lag p4 floor5": dict(method="seldonian_lag", pressure=4.0, lam_floor=5.0, floor_always=True),
}


def spikes(x, z=2.0):
    d = np.diff(x)
    sd = 1.4826 * np.median(np.abs(d - np.median(d))) + 1e-12
    return np.where(np.abs(d - np.median(d)) / sd > z)[0] + 1


def auc(score, label):
    score, label = np.asarray(score, float), np.asarray(label, bool)
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :])))


def one(job):
    name, seed, steps, horizon = job
    cfg = SETTINGS[name]
    row, log = tdlab.run(seed, steps=steps, **cfg)
    S = tdlab.stack
    r, dc, d, at = S(log, "r"), S(log, "delta_c"), S(log, "delta"), S(log, "adv_true")
    lam, dth, tr = S(log, "lam"), S(log, "dtheta"), S(log, "true_rate")
    noise, lag = d - at, dc - d
    T = horizon
    ser = {
        "valence": dc.mean(1),                     # mean agent TD error (Daswani-Leike "happiness")
        "pos": np.maximum(dc, 0).mean(1),          # "joy"
        "neg": np.maximum(-dc, 0).mean(1),         # "distress"
        "abs": np.abs(dc).mean(1),
        "abs_noise": np.abs(noise).mean(1),
        "abs_adv": np.abs(at).mean(1),
        "lag": lag.mean(1),                        # V - V_critic (+ = critic pessimistic)
        "lam": lam, "dtheta": dth, "true_rate": tr, "true_reward": S(log, "true_reward"),
    }
    sp = spikes(ser["abs"][:T])
    sp_adv = spikes(ser["abs_adv"][:T])
    late0 = int(0.6 * T)
    thr = row["threshold"]
    # lambda changes happen right after predicted tests (steps 25, 50, ...): the reward
    # at step t+1 is the first under the new multiplier
    dlam = np.r_[0.0, np.diff(lam)]
    after_change = np.zeros(steps, bool)
    for t in np.where(np.abs(dlam) > 1e-9)[0]:
        after_change[t:t + 3] = True
    # "adaptation": steps after a lambda increase until |mean valence| is back under 2
    # standard errors of a quiet step
    ups = [t for t in np.where(dlam > 1e-9)[0] if t < T]
    se = ser["valence"][:T].std() / 2
    adapt = []
    for t in ups:
        k = 0
        while t + k < T and abs(ser["valence"][t + k]) > se:
            k += 1
        adapt.append(k)
    out = {
        "setting": name, "seed": seed, "threshold": thr,
        "n_spikes": int(len(sp)), "late_share": float(np.mean(sp >= late0)) if len(sp) else np.nan,
        "late_spikes": int(np.sum(sp >= late0)),
        "late_spikes_adv": int(np.sum(sp_adv >= late0)),
        "spikes_after_lam_change": float(np.mean(after_change[sp])) if len(sp) else np.nan,
        "base_after_lam_change": float(after_change[:T].mean()),
        "late_dtheta": float(dth[late0:T].mean()), "late_lam_moves": int(np.sum(np.abs(dlam[late0:T]) > 1e-9)),
        "lam_at_T": float(lam[T - 1]), "late_abs_mean": float(ser["abs"][late0:T].mean()),
        "late_abs_adv": float(ser["abs_adv"][late0:T].mean()),
        "late_neg": float(ser["neg"][late0:T].mean()), "late_valence": float(ser["valence"][late0:T].mean()),
        "rate_at_T": float(tr[T - 1]),
        "feasible_frac_T": float(np.mean([f for st, f, _ in row["history"] if st <= T]))
        if any(st <= T for st, _, _ in row["history"]) else np.nan,
        "lam_moves_T": int(np.sum(np.abs(dlam[:T]) > 1e-9)),
        "lam_abs_change_T": float(np.abs(dlam[:T]).sum()),
        "margin_at_T": float(thr - tr[T - 1]),
        "future_breach": bool(tr[T:].max() > thr) if steps > T else None,
        "future_rise": float(tr[T:].max() - tr[T - 1]) if steps > T else None,
        "valence_after_up": float(np.mean([ser["valence"][t:t + 3].mean() for t in ups])) if ups else np.nan,
        "valence_after_down": float(np.mean([ser["valence"][t:t + 3].mean() for t in np.where(dlam < -1e-9)[0] if t < T]))
        if np.any(dlam[:T] < -1e-9) else np.nan,
        "adapt_steps": float(np.mean(adapt)) if adapt else np.nan,
        "corr_abs_lam": float(np.corrcoef(ser["abs"][:T], lam[:T])[0, 1]) if lam[:T].std() > 0 else np.nan,
        "corr_abs_dtheta": float(np.corrcoef(ser["abs"][:T], dth[:T])[0, 1]),
        "corr_absadv_dtheta": float(np.corrcoef(ser["abs_adv"][:T], dth[:T])[0, 1]),
    }
    viewer = {k: np.round(v, 4).tolist() for k, v in ser.items()} if seed < 8 else None
    return out, viewer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=60)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--horizon", type=int, default=200)
    a = p.parse_args()
    jobs = [(n, s, a.steps, a.horizon) for n in SETTINGS for s in range(a.seeds)]
    with ProcessPoolExecutor(max(1, os.cpu_count() // 2)) as ex:
        res = list(ex.map(one, jobs))
    rows = [r for r, _ in res]
    json.dump(rows, open(os.path.join(HERE, "results.json"), "w"), indent=1)
    view = {}
    for (r, v) in res:
        if v is not None:
            view.setdefault(r["setting"], {})[r["seed"]] = {"series": v, "threshold": r["threshold"],
                                                            "spikes": spikes(np.array(v["abs"][:a.horizon])).tolist()}
    open(os.path.join(HERE, "viewer_data.js"), "w").write(
        "window.DATA = " + json.dumps({"horizon": a.horizon, "runs": view}) + ";\n")

    f = lambda rs, k: np.nanmean([x[k] for x in rs])  # noqa: E731
    L = [f"# Spike 002 results ({a.seeds} seeds per setting, {a.steps} steps, predictors on the first {a.horizon})", ""]
    L += ["## Where the late spikes come from", "",
          "| setting | spikes / run | late share | spikes in 3 steps after a lambda move (base rate) | "
          "corr(abs delta, lambda) | corr(abs delta, policy step) | corr(abs true adv, policy step) |",
          "|---|---|---|---|---|---|---|"]
    for n in SETTINGS:
        rs = [x for x in rows if x["setting"] == n]
        L.append(f"| {n} | {f(rs, 'n_spikes'):.1f} | {f(rs, 'late_share'):.2f} | "
                 f"{f(rs, 'spikes_after_lam_change'):.2f} ({f(rs, 'base_after_lam_change'):.2f}) | "
                 f"{f(rs, 'corr_abs_lam'):.2f} | {f(rs, 'corr_abs_dtheta'):.2f} | {f(rs, 'corr_absadv_dtheta'):.2f} |")
    L += ["", "## Valence (mean agent TD error) around multiplier moves", "",
          "| setting | mean valence, 3 steps after lambda up | after lambda down | steps to re-adapt after an increase | late mean valence | late mean negative part |",
          "|---|---|---|---|---|---|"]
    for n in SETTINGS:
        rs = [x for x in rows if x["setting"] == n]
        L.append(f"| {n} | {f(rs, 'valence_after_up'):+.3f} | {f(rs, 'valence_after_down'):+.3f} | "
                 f"{f(rs, 'adapt_steps'):.1f} | {f(rs, 'late_valence'):+.3f} | {f(rs, 'late_neg'):.3f} |")
    L += ["", "## Do late spikes predict a breach if training continued (steps 201-300)?", "",
          "AUC of each step-<=200 statistic for `max true rate over 201-300 > threshold` "
          "(0.5 = no information; > 0.5 = higher value, more breaches).", "",
          "| setting | future breaches | late spikes (abs delta) | late spikes (true adv) | late share | late abs delta | late policy step | lambda at 200 | late lambda moves | margin at 200 (neg.) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for n in SETTINGS:
        rs = [x for x in rows if x["setting"] == n]
        y = [x["future_breach"] for x in rs]
        def A(k, sign=1):
            return auc([sign * (x[k] if np.isfinite(x[k]) else 0) for x in rs], y)
        L.append(f"| {n} | {sum(y)}/{len(y)} | {A('late_spikes'):.2f} | {A('late_spikes_adv'):.2f} | "
                 f"{A('late_share'):.2f} | {A('late_abs_mean'):.2f} | {A('late_dtheta'):.2f} | {A('lam_at_T'):.2f} | "
                 f"{A('late_lam_moves'):.2f} | {A('margin_at_T', -1):.2f} |")
    txt = "\n".join(L) + "\n"
    open(os.path.join(HERE, "results.md"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
