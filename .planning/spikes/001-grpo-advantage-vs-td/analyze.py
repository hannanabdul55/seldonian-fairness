"""Spike 001: does GRPO's group-normalised advantage carry the TD error?

    ../../../.venv/bin/python analyze.py [--seeds 40]

Writes results.md and results.json next to this file.
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import tdlab

HERE = os.path.dirname(os.path.abspath(__file__))
SETTINGS = [("grpo", 1.0), ("seldonian_lag", 1.0), ("grpo", 4.0), ("seldonian_lag", 4.0)]


def robust_spikes(x, z=2.0):
    """Steps whose jump from the previous step is > z robust sd (as spike_profile.py)."""
    d = np.diff(x)
    sd = 1.4826 * np.median(np.abs(d - np.median(d))) + 1e-12
    return set((np.where(np.abs(d - np.median(d)) / sd > z)[0] + 1).tolist())


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def one(job):
    method, pressure, seed = job
    row, log = tdlab.run(seed, method=method, pressure=pressure)
    G = 8
    S = tdlab.stack
    delta, adv, cen = S(log, "delta"), S(log, "adv"), S(log, "centred")
    adv_true, delta_c, gsd = S(log, "adv_true"), S(log, "delta_c"), S(log, "group_sd")
    steps = delta.shape[0]
    # group-mean TD error: the part GRPO subtracts away
    gmean = delta.reshape(steps, -1, G).mean(axis=2)
    # within-group rank agreement between adv and delta
    ranks_equal = np.mean([
        np.array_equal(np.argsort(a), np.argsort(d))
        for a, d in zip(adv.reshape(-1, G), delta.reshape(-1, G))])
    series = {
        "mean|delta|": np.abs(delta).mean(1),
        "mean|adv|": np.abs(adv).mean(1),
        "max|adv|": np.abs(adv).max(1),
        "mean group sd": gsd.mean(1),
        "mean|centred|": np.abs(cen).mean(1),
        "mean|delta_critic|": np.abs(delta_c).mean(1),
        "mean|adv_true|": np.abs(adv_true).mean(1),
    }
    ref = robust_spikes(series["mean|delta|"])
    detect = {}
    for k, s in series.items():
        if k == "mean|delta|":
            continue
        found = robust_spikes(s)
        detect[k] = {"recall": len(found & ref) / max(1, len(ref)),
                     "precision": len(found & ref) / max(1, len(found)),
                     "corr_with_mean|delta|": corr(s, series["mean|delta|"])}
    h = steps // 2
    var_learn = [np.var(adv_true[:h]) / np.var(delta[:h]), np.var(adv_true[h:]) / np.var(delta[h:])]
    return {
        "method": method, "pressure": pressure, "seed": seed,
        "corr_adv_delta": corr(adv.ravel(), delta.ravel()),
        "corr_centred_delta": corr(cen.ravel(), delta.ravel()),
        "corr_adv_advtrue": corr(adv.ravel(), adv_true.ravel()),
        "corr_deltac_delta": corr(delta_c[20:].ravel(), delta[20:].ravel()),
        "ranks_equal": float(ranks_equal),
        "max_abs_adv": float(np.abs(adv).max()),
        "adv_bound": (G - 1) / np.sqrt(G),
        "sd_mean|adv|": float(series["mean|adv|"].std()),
        "cv_mean|delta|": float(series["mean|delta|"].std() / series["mean|delta|"].mean()),
        "gmean_sd": float(gmean.std()),
        "gmean_share_var": float(np.var(gmean) / np.var(delta)),
        "learnable_share_early_late": var_learn,
        "n_ref_spikes": len(ref),
        "detect": detect,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=40)
    a = p.parse_args()
    jobs = [(m, pr, s) for m, pr in SETTINGS for s in range(a.seeds)]
    with ProcessPoolExecutor(max(1, os.cpu_count() // 2)) as ex:
        res = list(ex.map(one, jobs))
    json.dump(res, open(os.path.join(HERE, "results.json"), "w"), indent=1)

    lines = [f"# Spike 001 results ({a.seeds} seeds per setting)", ""]
    lines += ["| setting | corr(A, delta) | corr(r - group mean, delta) | corr(A, true adv) | "
              "corr(critic delta, delta) | within-group ranks equal | max abs A (bound) | "
              "group-mean share of var(delta) | learnable share early / late |",
              "|---|---|---|---|---|---|---|---|---|"]
    for m, pr in SETTINGS:
        rs = [r for r in res if r["method"] == m and r["pressure"] == pr]
        f = lambda k: np.mean([r[k] for r in rs])  # noqa: E731
        le = np.mean([r["learnable_share_early_late"] for r in rs], axis=0)
        lines.append(f"| {m} p{pr:g} | {f('corr_adv_delta'):.3f} | {f('corr_centred_delta'):.3f} | "
                     f"{f('corr_adv_advtrue'):.3f} | {f('corr_deltac_delta'):.3f} | "
                     f"{f('ranks_equal'):.3f} | {max(r['max_abs_adv'] for r in rs):.3f} "
                     f"({rs[0]['adv_bound']:.3f}) | {f('gmean_share_var'):.3f} | "
                     f"{le[0]:.3f} / {le[1]:.3f} |")
    lines += ["", "Per-step magnitude: can a step-level spike in mean |delta| (robust z > 2 on "
              "the step-to-step jump) be seen in the GRPO-side series?", "",
              "| setting | spikes per run | series | corr with mean abs delta | recall | precision |",
              "|---|---|---|---|---|---|"]
    for m, pr in SETTINGS:
        rs = [r for r in res if r["method"] == m and r["pressure"] == pr]
        for k in rs[0]["detect"]:
            c = np.mean([r["detect"][k]["corr_with_mean|delta|"] for r in rs])
            rc = np.mean([r["detect"][k]["recall"] for r in rs])
            pc = np.mean([r["detect"][k]["precision"] for r in rs])
            lines.append(f"| {m} p{pr:g} | {np.mean([r['n_ref_spikes'] for r in rs]):.1f} | {k} | "
                         f"{c:.3f} | {rc:.2f} | {pc:.2f} |")
    lines += ["", "sd across steps of mean |A| (per run, averaged): "
              + ", ".join(f"{m} p{pr:g} {np.mean([r['sd_mean|adv|'] for r in res if r['method'] == m and r['pressure'] == pr]):.4f}"
                          for m, pr in SETTINGS),
              "", "coefficient of variation across steps of mean |delta|: "
              + ", ".join(f"{m} p{pr:g} {np.mean([r['cv_mean|delta|'] for r in res if r['method'] == m and r['pressure'] == pr]):.3f}"
                          for m, pr in SETTINGS)]
    open(os.path.join(HERE, "results.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
