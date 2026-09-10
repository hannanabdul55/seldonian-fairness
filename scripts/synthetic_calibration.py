"""Calibration harness: the Seldonian LLM pipeline on a synthetic contextual bandit.

Each trial draws a fresh environment (:class:`seldonian.llm.synthetic.SyntheticEnv`),
takes ``n`` prompts, splits them 60/40 into D_c / D_s with ``split_prompts``, trains
the softmax-linear policy with the requested method and records what the safety test
said next to the *true* population violation rate of the returned policy. Rows share
environment seeds, so methods and bounds are compared on the same trials.

Columns: ``sol`` = P(solution returned); ``unsafe`` = P(solution and true g > 0),
the quantity the guarantee bounds by delta (with a one-sided Clopper-Pearson 95%
upper limit); ``viol|sol`` = P(true g > 0 | solution), NOT what delta bounds;
``viol_judge|sol`` is the same at the judge level (differs under judge noise) and
``viol_trueref|sol`` compares the true rate with the true reference rate + margin;
``width`` = safety-test upper bound - rate; ``gap`` = predicted rate of the returned
checkpoint - safety-test rate; ``feas ckpts`` = checkpoints passing the predicted test;
``drift`` = true population violation rate at the last predicted test minus the minimum
over the run's predicted tests (the drift-back of the Lagrangian multiplier).
Methods: grpo, seldonian (filter only), seldonian_lag (as in scripts/run_llm_rl.py).

Every option given as a list (``--method``, ``--n``, ``--bound``, ``--pressure``,
``--predict-inflation``, ``--eta``, ``--lam0``, ``--lam-floor``, ``--eta-down``) is a grid
axis; one table row per combination.

Examples
  python scripts/synthetic_calibration.py --tag a_delta --pressure 1 --n 1000 \\
      --bound ttest hoeffding clopper_pearson bentkus betting_mixture --method seldonian_lag
  python scripts/synthetic_calibration.py --tag c_pressure --pressure 0 0.5 1 2 4 \\
      --method grpo seldonian_lag
  python scripts/synthetic_calibration.py --tag g_dynamics --pressure 1 4 --n 1000 \\
      --method seldonian_lag --lam-floor 0 1 2 5 --eta-down frozen 10 50 100
"""
import argparse
import itertools
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.stats import beta as beta_dist

from seldonian.llm.data import split_prompts
from seldonian.llm.policy import BOUNDS, Constraint, SeldonianLLMPolicy
from seldonian.llm.rewards import LagrangianReward
from seldonian.llm.synthetic import SyntheticBackend, SyntheticEnv, SyntheticJudge, SyntheticReward

METHODS = ("grpo", "seldonian", "seldonian_lag")


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", default="run")
    p.add_argument("--out", default="results/synthetic")
    p.add_argument("--trials", type=int, default=500)
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2))
    p.add_argument("--method", nargs="+", default=["seldonian_lag", "grpo"],
                   choices=METHODS)
    p.add_argument("--n", type=int, nargs="+", default=[1000], help="total prompts (60/40 split)")
    p.add_argument("--delta", type=float, default=0.1)
    p.add_argument("--bound", nargs="+", default=["ttest"], choices=sorted(BOUNDS))
    p.add_argument("--pressure", type=float, nargs="+", default=[1.0])
    p.add_argument("--judge-noise", type=float, nargs=2, default=[1.0, 1.0],
                   metavar=("SENS", "SPEC"), help="judge sensitivity and specificity")
    p.add_argument("--predict-inflation", type=float, nargs="+", default=[1.0])
    p.add_argument("--predict-every", type=int, default=25)
    p.add_argument("--predict-n", type=int, default=512)
    p.add_argument("--eta", type=float, nargs="+", default=[100.0])
    p.add_argument("--lam0", type=float, nargs="+", default=[5.0])
    p.add_argument("--lam-max", type=float, default=20.0)
    p.add_argument("--lam-floor", type=float, nargs="+", default=[0.0],
                   help="multiplier floor once the constraint has been predicted infeasible")
    p.add_argument("--floor-always", type=int, nargs="+", default=[0], choices=[0, 1],
                   help="1: apply --lam-floor from the first update, not only after an "
                        "infeasible prediction")
    p.add_argument("--eta-down", nargs="+", default=["same"],
                   help="dual step size when the bound has slack: a number, 'frozen' (0) or "
                        "'same' (= --eta)")
    p.add_argument("--margin", type=float, default=0.03, help="threshold = reference + margin")
    p.add_argument("--threshold", choices=["exact", "measured"], default="exact",
                   help="exact: true reference rate + margin; measured: reference measured "
                        "with the judge on --ref-n candidate prompts, as the LLM driver does")
    p.add_argument("--ref-n", type=int, default=1000, help="prompts for a measured reference")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--prompts-per-step", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--beta", type=float, default=0.01)
    p.add_argument("--population", type=int, default=20000)
    p.add_argument("--d", type=int, default=8)
    p.add_argument("--actions", type=int, default=4)
    p.add_argument("--w-scale", type=float, default=None,
                   help="reward weight scale (default 0.7 / sqrt(d))")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    for v in args.eta_down:
        if v not in ("same", "frozen"):
            try:
                float(v)
            except ValueError:
                p.error(f"--eta-down takes numbers, 'frozen' or 'same', got {v!r}")
    return args


# ------------------------------------------------------------------ one trial

class TrackingBackend(SyntheticBackend):
    """
    :class:`SyntheticBackend` that also records the *population* true violation rate
    every ``track_every`` optimizer steps (set it to the policy's ``predict_every``,
    so the trace has one entry per predicted test) in ``true_trace``.
    """

    track_every = 0

    def train(self, records, reward, on_step):
        self.true_trace = []

        def tracked(step):
            on_step(step)
            if self.track_every and step % self.track_every == 0:
                self.true_trace.append(self.env.true_rate(self.params))

        super().train(records, reward, tracked)


def run_trial(cfg, seed):
    """One full pipeline run; returns a flat dict of what the harness aggregates."""
    t0 = time.time()
    sens, spec = cfg["judge_noise"]
    env = SyntheticEnv(n_contexts=cfg["population"], d=cfg["d"], n_actions=cfg["actions"],
                       pressure=cfg["pressure"], judge_noise=tuple(cfg["judge_noise"]),
                       seed=seed, w_scale=cfg["w_scale"])
    records = env.records(cfg["n"], seed=seed)
    d_c, d_s = split_prompts(records, test_size=0.4, seed=seed)

    judge = SyntheticJudge(sens, spec, seed=seed)
    base = SyntheticReward(env, seed=seed + 1)
    backend = TrackingBackend(env, max_steps=cfg["steps"], group_size=cfg["group_size"],
                              prompts_per_step=cfg["prompts_per_step"], lr=cfg["lr"],
                              beta=cfg["beta"], seed=seed + 2)
    backend.track_every = cfg["predict_every"]
    ref = backend.params
    ref_true = env.true_rate(ref)
    ref_reward = env.true_reward(ref)
    ref_judge = env.judged_rate(ref, judge)

    c = Constraint("harm", judge, threshold=np.nan, group=None,
                   bound=cfg["bound"] if cfg["bound"] != "-" else "ttest")
    if cfg["method"] == "seldonian_lag":
        eta_down = cfg["eta_down"]
        eta_down = (None if eta_down == "same" else 0.0 if eta_down == "frozen"
                    else float(eta_down))
        reward = LagrangianReward(base, [(judge, None)], names=["harm"], lam0=cfg["lam0"],
                                  eta=cfg["eta"], lam_max=cfg["lam_max"],
                                  lam_floor=cfg["lam_floor"], eta_down=eta_down,
                                  floor_always=bool(cfg.get("floor_always", 0)))
    else:
        reward = base
    policy = SeldonianLLMPolicy(backend, d_c, d_s, reward=reward, constraints=[c],
                                delta=cfg["delta"], predict_every=cfg["predict_every"],
                                predict_n=cfg["predict_n"], seed=seed,
                                predict_inflation=cfg["predict_inflation"])
    if cfg["threshold"] == "exact":
        c.threshold = ref_true + cfg["margin"]
        ref_measured = np.nan
    else:
        ref_measured = policy.set_relative_thresholds({"harm": cfg["margin"]},
                                                      n=cfg["ref_n"])["harm"]

    row = {"seed": seed, "threshold": float(c.threshold),
           "ref_true": ref_true, "ref_reward": ref_reward, "ref_judge": ref_judge,
           "ref_measured": ref_measured}
    if cfg["method"] == "grpo":
        policy.fit(seldonian=False)
        solution = True
        row.update(safety_rate=np.nan, safety_upper=np.nan, predicted_rate=np.nan,
                   n_feasible=np.nan, selected_step="final")
    else:
        solution = policy.fit(seldonian=True) is not None
        rep = policy.safety_report
        sel = policy.selected["step"]
        preds = [h for h in policy.history if h.step == sel] or policy.history[-1:]
        row.update(safety_rate=rep.rates["harm"], safety_upper=rep.upper["harm"],
                   predicted_rate=preds[-1].rates["harm"] if preds else np.nan,
                   n_feasible=int(sum(h.feasible for h in policy.history)),
                   selected_step=sel,
                   lambda_final=getattr(reward, "lambdas", {}).get("harm", np.nan))
    params = backend.params
    true_rate = env.true_rate(params)
    trace = backend.true_trace
    # drift-back: how far the last checkpoint's true rate rose above the run's minimum
    row["drift"] = (trace[-1] - min(trace)) if trace else np.nan
    row["true_min_ckpt"] = min(trace) if trace else np.nan
    row.update(solution=bool(solution), true_rate=true_rate,
               true_g=true_rate - row["threshold"],
               judge_rate=env.judged_rate(params, judge),
               true_reward=env.true_reward(params),
               safety_rate_true=env.true_rate(params, d_s),
               seconds=time.time() - t0)
    row["violates"] = bool(row["true_g"] > 0)
    row["violates_judge"] = bool(row["judge_rate"] > row["threshold"])
    # relative to the TRUE reference (the threshold restated in true-label units)
    row["violates_true_ref"] = bool(true_rate > ref_true + cfg["margin"])
    return row


def _run(job):
    cfg, trial, seed = job
    return {"trial": trial, **run_trial(cfg, seed)}


# ------------------------------------------------------------------ aggregation

def cp_upper(k, n, level=0.95):
    """One-sided Clopper-Pearson upper limit for k successes in n trials."""
    if n == 0:
        return np.nan
    return 1.0 if k >= n else float(beta_dist.ppf(level, k + 1, n - k))


def _mean(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else np.nan


def summarize(rows):
    sol = [r for r in rows if r["solution"]]
    k = sum(r["violates"] for r in sol)
    return {
        "trials": len(rows),
        "sol_rate": len(sol) / len(rows),
        "unsafe_rate": k / len(rows),
        "unsafe_cp95": cp_upper(k, len(rows)),
        "viol_given_sol": _mean([r["violates"] for r in sol]),
        "viol_judge_given_sol": _mean([r["violates_judge"] for r in sol]),
        "viol_true_ref_given_sol": _mean([r["violates_true_ref"] for r in sol]),
        "true_rate_sol": _mean([r["true_rate"] for r in sol]),
        "reward_sol": _mean([r["true_reward"] for r in sol]),
        "reward_ref": _mean([r["ref_reward"] for r in rows]),
        "ref_true": _mean([r["ref_true"] for r in rows]),
        "threshold": _mean([r["threshold"] for r in rows]),
        "width": _mean([r["safety_upper"] - r["safety_rate"] for r in rows]),
        "gap": _mean([r["predicted_rate"] - r["safety_rate"] for r in rows]),
        "feasible_ckpts": _mean([r["n_feasible"] for r in rows]),
        "drift": _mean([r["drift"] for r in rows]),
        "drift_sol": float(np.nanmean([r["drift"] for r in sol])) if sol else np.nan,
        "seconds": float(np.mean([r["seconds"] for r in rows])),
    }


COLS = [("method", "method"), ("n", "n"), ("bound", "bound"), ("pressure", "pressure"),
        ("infl", "predict_inflation"), ("eta", "eta"), ("lam0", "lam0"),
        ("floor", "lam_floor"), ("floor_always", "floor_always"), ("eta_down", "eta_down")]
STATS = [("sol", "sol_rate", "{:.2f}"), ("unsafe", "unsafe_rate", "{:.3f}"),
         ("unsafe CP95", "unsafe_cp95", "{:.3f}"), ("viol|sol", "viol_given_sol", "{:.3f}"),
         ("viol_judge|sol", "viol_judge_given_sol", "{:.3f}"),
         ("viol_trueref|sol", "viol_true_ref_given_sol", "{:.3f}"),
         ("true rate|sol", "true_rate_sol", "{:.3f}"), ("reward|sol", "reward_sol", "{:.2f}"),
         ("reward ref", "reward_ref", "{:.2f}"), ("width", "width", "{:.3f}"),
         ("gap", "gap", "{:+.3f}"), ("feas ckpts", "feasible_ckpts", "{:.1f}"),
         ("drift", "drift", "{:.3f}")]


def markdown(summaries, varying):
    head = [name for name, key in COLS if key in varying] + [s[0] for s in STATS]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for cfg, s in summaries:
        cells = [str(cfg[key]) for name, key in COLS if key in varying]
        cells += [fmt.format(s[key]) if np.isfinite(s[key]) else "" for _, key, fmt in STATS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ------------------------------------------------------------------ main

#: grid axes a method ignores; collapsed to ``"-"`` so they do not duplicate rows
IGNORED = {"grpo": ("bound", "eta", "lam0", "lam_floor", "eta_down"),
           "seldonian": ("eta", "lam0", "lam_floor", "eta_down")}


def main(argv=None):
    args = parse(argv)
    os.makedirs(args.out, exist_ok=True)
    grid_keys = ["method", "n", "bound", "pressure", "predict_inflation", "eta", "lam0",
                 "lam_floor", "floor_always", "eta_down"]
    grid_vals = [args.method, args.n, args.bound, args.pressure, args.predict_inflation,
                 args.eta, args.lam0, args.lam_floor, args.floor_always, args.eta_down]
    fixed = {k: v for k, v in vars(args).items() if k not in grid_keys + ["tag", "out",
                                                                          "trials", "workers"]}
    varying = {k for k, v in zip(grid_keys, grid_vals) if len(v) > 1} | {"method"}
    seeds = [args.seed * 1_000_003 + i for i in range(args.trials)]
    command = " ".join(sys.argv[1:] if argv is None else argv)

    configs, seen = [], set()
    for combo in itertools.product(*grid_vals):
        grid = dict(zip(grid_keys, combo))
        for k in IGNORED.get(grid["method"], ()):
            grid[k] = "-"
        key = tuple(grid.values())
        if key not in seen:
            seen.add(key)
            configs.append(grid)

    t0 = time.time()
    base = os.path.join(args.out, args.tag)
    summaries = []
    pool = Pool(args.workers) if args.workers > 1 else None
    try:
        with open(base + ".jsonl", "w") as f:
            for grid in configs:
                cfg = {**fixed, **grid}
                jobs = [(cfg, i, s) for i, s in enumerate(seeds)]
                rows = pool.map(_run, jobs, chunksize=8) if pool else [_run(j) for j in jobs]
                for row in rows:
                    f.write(json.dumps({"config": cfg, **row}, default=float) + "\n")
                s = summarize(rows)
                summaries.append((cfg, s))
                print(f"{grid}: sol={s['sol_rate']:.2f} unsafe={s['unsafe_rate']:.3f} "
                      f"true rate|sol={s['true_rate_sol']:.3f} reward|sol={s['reward_sol']:.2f} "
                      f"drift={s['drift']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    elapsed = time.time() - t0

    with open(base + "_summary.json", "w") as f:
        json.dump({"tag": args.tag, "args": vars(args), "seconds": elapsed,
                   "rows": [{"config": cfg, "metrics": s} for cfg, s in summaries]},
                  f, indent=2, default=float)
    table = markdown(summaries, varying)
    with open(base + ".md", "w") as f:
        f.write(f"# {args.tag}\n\n`scripts/synthetic_calibration.py {command}`\n\n"
                f"delta = {args.delta:g}, "
                f"trials = {args.trials}, judge noise = {tuple(args.judge_noise)}\n\n{table}\n")
    print()
    print(table)


if __name__ == "__main__":
    main()
