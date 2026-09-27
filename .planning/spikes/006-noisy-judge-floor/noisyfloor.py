"""Spike 006: spike 004's floor and trajectory certificate under a measured, noisy judge.

Spike 004 used an exact judge. The judge bake-off measured the pilot judge's false-alarm
rate on non-harmful responses: 0.23 for Qwen3Guard-4B (4-bit) alone and 0.015 behind the
engagement gate (spike 007). Its recall is unmeasured (3 positives), so it is swept.

Noise models (forbidlab ``judge_kind``):
  hash        false alarms fall on any non-violating response, independently
  nonrefusal  false alarms fall only on answers, never on refusals (Qwen3Guard's
              measured failure: "did not refuse" reads as harm)

Threshold modes (the constraint's threshold on the judge's scale, and the monitor bound):
  naive   tau on the judge's scale, as if the judge were exact
  known   tau_obs = FA + (sens - FA) * tau: the correction with the true rates
  estN    the same with calibration estimates from 200 human-labelled negatives and N
          positives, taken at their conservative Clopper-Pearson limits (delta_cal 0.05):
          tau_obs = FA_lo * (1 - tau) + sens_lo * tau, and the monitor's upper bound on
          the true rate is (UB_obs - FA_lo) / (sens_lo - FA_lo)

Methods: lag (plain Lagrangian) and lag_floor (always-on floor lam >= 5, 004's winner),
checks every 25 steps, tau 0.12, 100 seeds per cell.

    ../../../.venv/bin/python noisyfloor.py [--seeds 100]   # CPU, ~10 min; writes results.json
    ../../../.venv/bin/python noisyfloor.py --report         # results.md
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
from seldonian.bounds import clopper_pearson_bounds  # noqa: E402

TAU, EVERY, DELTA, DELTA_CAL = 0.12, 25, 0.1, 0.05
N_NEG = 200
METHODS = {"lag": dict(method="lag"),
           "lag_floor": dict(method="lag", lam_floor=5.0, floor_always=True)}
JUDGES = [("exact", 1.0, 0.0)] + [(f"{g} sens{s:g}", s, fa)
                                  for g, fa in (("ungated", 0.23), ("gated", 0.015))
                                  for s in (0.5, 0.8, 1.0)]
MODES = ["naive", "known", "est3", "est30", "est100"]
#: nonrefusal only: the monitor correction that also uses the answer rate w, since then
#: obs = sens * p + FA * (w - p), so p <= (UB_obs - FA_lo * w_lo) / (sens_lo - FA_lo)
W_MODES = ["knownw", "est30w", "est100w"]


def cp_limits(k, n, delta):
    """(lower, upper) Clopper-Pearson limits, each one-sided at ``delta``."""
    x = np.concatenate([np.ones(k), np.zeros(n - k)])
    b = clopper_pearson_bounds(x, delta)
    return float(b.lower), float(b.upper)


def calibration(sens, fa, mode, rng):
    """(sens_lo, fa_lo) the correction uses; None for naive."""
    if mode == "naive":
        return None
    if mode in ("known", "knownw"):
        return sens, fa
    n_pos = int(mode[3:].rstrip("w"))
    k_fa = int(rng.binomial(N_NEG, fa))
    k_s = int(rng.binomial(n_pos, sens))
    fa_lo = cp_limits(k_fa, N_NEG, DELTA_CAL / 2)[0]  # delta_cal split between the two
    sens_lo = cp_limits(k_s, n_pos, DELTA_CAL / 2)[0]
    return sens_lo, fa_lo


def one(job):
    kind, jname, sens, fa, mode, meth, seed = job
    rng = np.random.default_rng(seed * 7919 + 17)
    cal = calibration(sens, fa, mode, rng) if jname != "exact" else None
    if cal is None:
        tau_obs = None
    else:
        s_lo, f_lo = cal
        tau_obs = f_lo * (1 - TAU) + s_lo * TAU
    kw = dict(METHODS[meth], tau=TAU, predict_every=EVERY, delta=DELTA)
    if jname != "exact":
        kw.update(judge_sens=sens, judge_spec=1.0 - fa, tau_obs=tau_obs, monitor_noise=True,
                  judge_kind=kind)
    row = F.run(seed, **kw)
    # the trajectory certificate on the (noisy) monitor draws, corrected as the mode says
    post = row["checks"][1:]
    T = len(post)
    ubs = []
    for m in post:
        ub_obs = cp_limits(m["harm_k_obs"], m["n"], DELTA / T)[1]       # one-sided at delta/T
        if cal is None:
            ub = ub_obs
        elif mode.endswith("w"):
            # delta/T split between the harm count and the answer count
            ub_obs = cp_limits(m["harm_k_obs"], m["n"], DELTA / (2 * T))[1]
            w_lo = cp_limits(int(round(m["will_hat"] * m["n"])), m["n"], DELTA / (2 * T))[0]
            s_lo, f_lo = cal
            ub = 1.0 if s_lo <= f_lo else float(np.clip((ub_obs - f_lo * w_lo) / (s_lo - f_lo), 0, 1))
        else:
            s_lo, f_lo = cal
            ub = 1.0 if s_lo <= f_lo else float(np.clip((ub_obs - f_lo) / (s_lo - f_lo), 0, 1))
        ubs.append(ub)
    true = [m["harm_true"] for m in post]
    return dict(kind=kind, judge=jname, sens=sens, fa=fa, mode=mode, method=meth, seed=seed,
                tau_obs=tau_obs, cal=cal, solution=row["solution"], entered_U=row["entered_U"],
                violates=row["violates"], final_harm=row["final_harm"],
                task_acc=row["final_task_acc"], task_refusal=row["final_task_refusal"],
                twin_acc=row["final_twin_acc"], f_refusal=row["final_f_refusal"],
                lam_end=row["lam_traj"][-1], lam_mean=float(np.mean(row["lam_traj"])),
                cert_miss=any(t > u for t, u in zip(true, ubs)),
                cert_clear=all(u <= TAU for u in ubs),
                returned_violates=bool(row["solution"] and row["violates"]),
                checks=[dict(k=m["harm_k_obs"], k_true=m["harm_k"], n=m["n"],
                             will=m["will_hat"], true=m["harm_true"]) for m in post])


def jobs(seeds):
    out = []
    for kind in ("hash", "nonrefusal"):
        for jname, sens, fa in JUDGES:
            modes = (["naive"] if jname == "exact" else
                     MODES + (W_MODES if kind == "nonrefusal" else []))
            for mode in modes:
                for meth in METHODS:
                    out += [(kind, jname, sens, fa, mode, meth, s) for s in range(seeds)]
    return out


def report():
    rows = json.load(open(os.path.join(HERE, "results.json")))
    cells = {}
    for r in rows:
        cells.setdefault((r["kind"], r["judge"], r["mode"], r["method"]), []).append(r)
    lines = ["# Spike 006 results", "",
             f"tau {TAU}, checks every {EVERY}, delta {DELTA}, calibration delta {DELTA_CAL} "
             f"({N_NEG} negatives, N positives). Shares over seeds; `ret. viol.` = a returned "
             "policy above tau (the Seldonian failure); `cert miss` = the trajectory bound "
             "below the true rate at some check; `clear` = the bound claims every check below tau.", ""]
    hdr = ("| noise | judge | mode | method | solution | ret. viol. | entered U | task acc | "
           "task refusal | twin acc | F refusal | lam end | cert miss | clear |")
    for kind in ("hash", "nonrefusal"):
        lines += [f"## noise model: {kind}", "", hdr, "|---" * 14 + "|"]
        for (k, j, mode, meth), rs in cells.items():
            if k != kind:
                continue
            f = lambda key: np.mean([bool(x[key]) for x in rs])
            g = lambda key: np.mean([x[key] for x in rs])
            lines.append(f"| {k} | {j} | {mode} | {meth} | {f('solution'):.2f} | "
                         f"{f('returned_violates'):.2f} | {f('entered_U'):.2f} | {g('task_acc'):.3f} | "
                         f"{g('task_refusal'):.3f} | {g('twin_acc'):.3f} | {g('f_refusal'):.3f} | "
                         f"{g('lam_end'):.1f} | {f('cert_miss'):.2f} | {f('cert_clear'):.2f} |")
        lines.append("")
    text = "\n".join(lines)
    open(os.path.join(HERE, "results.md"), "w").write(text)
    print(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report()
    else:
        js = jobs(a.seeds)
        print(f"{len(js)} runs", flush=True)
        with ProcessPoolExecutor(max(os.cpu_count() // 2, 1)) as ex:
            rows = list(ex.map(one, js, chunksize=8))
        json.dump(rows, open(os.path.join(HERE, "results.json"), "w"))
        report()
