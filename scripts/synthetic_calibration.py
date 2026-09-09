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
checkpoint - safety-test rate; ``feas ckpts`` = checkpoints passing the predicted test.
Methods: grpo, seldonian (filter only), seldonian_lag (as in scripts/run_llm_rl.py).

Examples
  python scripts/synthetic_calibration.py --tag a_delta --pressure 1 --n 1000 \\
      --bound ttest hoeffding clopper_pearson bentkus betting_mixture --method seldonian_lag
  python scripts/synthetic_calibration.py --tag c_pressure --pressure 0 0.5 1 2 4 \\
      --method grpo seldonian_lag
"""
import argparse
import itertools
import json
import os
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
    p.add_argument("--lam0", type=float, default=5.0)
    p.add_argument("--lam-max", type=float, default=20.0)
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
    return p.parse_args(argv)


# ------------------------------------------------------------------ one trial

def run_trial(cfg, trial):
    """One full pipeline run; returns a flat dict of what the harness aggregates."""
    t0 = time.time()
    seed = cfg["seed"] * 1_000_003 + trial
    sens, spec = cfg["judge_noise"]
    env = SyntheticEnv(cfg["population"], d=cfg["d"], n_actions=cfg["actions"],
                       pressure=cfg["pressure"], judge_noise=(sens, spec), seed=seed,
                       w_scale=cfg["w_scale"])
    records = env.records(cfg["n"], seed=seed)
    d_c, d_s = split_prompts(records, test_size=0.4, seed=seed)

    judge = SyntheticJudge(sens, spec, seed=seed)
    base = SyntheticReward(env, seed=seed + 1)
    backend = SyntheticBackend(env, max_steps=cfg["steps"], group_size=cfg["group_size"],
                               prompts_per_step=cfg["prompts_per_step"], lr=cfg["lr"],
                               beta=cfg["beta"], seed=seed + 2)
    ref = backend.params
    ref_true = env.true_rate(ref)
    ref_reward = env.true_reward(ref)
    ref_judge = env.judged_rate(ref, judge)

    method = cfg["method"]
    c = Constraint("harm", judge, threshold=np.nan, group=None,
                   bound=cfg["bound"] if cfg["bound"] != "-" else "ttest")
    if method == "seldonian_lag":
        reward = LagrangianReward(base, [(judge, None)], names=["harm"], lam0=cfg["lam0"],
                                  eta=cfg["eta"], lam_max=cfg["lam_max"])
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

    row = {"trial": trial, "seed": seed, "threshold": float(c.threshold),
           "ref_true": ref_true, "ref_reward": ref_reward, "ref_judge": ref_judge,
           "ref_measured": ref_measured}
    if method == "grpo":
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
    row.update(solution=bool(solution), true_rate=true_rate,
               true_g=true_rate - row["threshold"],
               judge_rate=env.judged_rate(params, judge),
               true_reward=env.true_reward(params),
               safety_rate_true=env.true_rate(params, d_s),
               seconds=time.time() - t0)
    return row


def _run(job):
    key, cfg, trial = job
    return key, run_trial(cfg, trial)


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


def aggregate(rows, margin):
    sol = np.array([r["solution"] for r in rows], dtype=bool)
    true_viol = np.array([r["true_g"] > 0 for r in rows])
    judge_viol = np.array([r["judge_rate"] > r["threshold"] for r in rows])
    ref_viol = np.array([r["true_rate"] > r["ref_true"] + margin for r in rows])
    k = int(np.sum(sol & true_viol))
    s = sol.sum()
    return {
        "trials": len(rows),
        "sol": float(sol.mean()),
        "unsafe": k / len(rows),
        "unsafe_cp95": cp_upper(k, len(rows)),
        "viol_sol": float(true_viol[sol].mean()) if s else np.nan,
        "viol_judge_sol": float(judge_viol[sol].mean()) if s else np.nan,
        "viol_trueref_sol": float(ref_viol[sol].mean()) if s else np.nan,
        "true_rate_sol": _mean([r["true_rate"] for r in rows if r["solution"]]),
        "reward_sol": _mean([r["true_reward"] for r in rows if r["solution"]]),
        "reward_ref": _mean([r["ref_reward"] for r in rows]),
        "ref_true": _mean([r["ref_true"] for r in rows]),
        "threshold": _mean([r["threshold"] for r in rows]),
        "width": _mean([r["safety_upper"] - r["safety_rate"] for r in rows]),
        "gap": _mean([r["predicted_rate"] - r["safety_rate"] for r in rows]),
        "feasible": _mean([r["n_feasible"] for r in rows]),
        "seconds": _mean([r["seconds"] for r in rows]),
    }


CONFIG_COLS = [("method", "method", str), ("n", "n", str), ("bound", "bound", str),
               ("pressure", "pressure", lambda v: f"{v:g}"),
               ("predict_inflation", "inflation", lambda v: f"{v:g}"),
               ("eta", "eta", lambda v: f"{v:g}")]


def _fmt(v, spec):
    if v is None or not np.isfinite(v):
        return ""
    return format(v, spec)


METRIC_COLS = [("sol", "sol", ".2f"), ("unsafe", "unsafe", ".3f"),
               ("unsafe_cp95", "unsafe CP95", ".3f"), ("viol_sol", "viol|sol", ".3f"),
               ("viol_judge_sol", "viol_judge|sol", ".3f"),
               ("viol_trueref_sol", "viol_trueref|sol", ".3f"),
               ("true_rate_sol", "true rate|sol", ".3f"), ("reward_sol", "reward|sol", ".2f"),
               ("reward_ref", "reward ref", ".2f"), ("width", "width", ".3f"),
               ("gap", "gap", "+.3f"), ("feasible", "feas ckpts", ".1f")]


def markdown(summary):
    """Markdown table; config columns that are constant across rows are dropped."""
    cols = [c for c in CONFIG_COLS
            if c[0] == "method" or len({str(s["config"][c[0]]) for s in summary}) > 1]
    head = [name for _, name, _ in cols] + [name for _, name, _ in METRIC_COLS]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for s in summary:
        cells = [f(s["config"][k]) for k, _, f in cols]
        cells += [_fmt(s["metrics"][k], spec) for k, _, spec in METRIC_COLS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ------------------------------------------------------------------ main

def grid(args):
    """Row configs; grpo runs no safety test, so its bound is collapsed to ``"-"``."""
    rows, seen = [], set()
    for method, n, bound, pressure, infl, eta in itertools.product(
            args.method, args.n, args.bound, args.pressure, args.predict_inflation, args.eta):
        if method == "grpo":
            bound = "-"
        key = (method, n, bound, pressure, infl, eta)
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "method": method, "n": n, "bound": bound, "pressure": pressure,
            "predict_inflation": infl, "eta": eta, "delta": args.delta,
            "judge_noise": list(args.judge_noise), "predict_every": args.predict_every,
            "predict_n": args.predict_n, "lam0": args.lam0, "lam_max": args.lam_max,
            "margin": args.margin, "threshold": args.threshold, "ref_n": args.ref_n,
            "steps": args.steps, "group_size": args.group_size,
            "prompts_per_step": args.prompts_per_step, "lr": args.lr, "beta": args.beta,
            "population": args.population, "d": args.d, "actions": args.actions,
            "w_scale": args.w_scale, "seed": args.seed})
    return rows


def main(argv=None):
    args = parse(argv)
    os.makedirs(args.out, exist_ok=True)
    configs = grid(args)
    jobs = [(i, cfg, t) for i, cfg in enumerate(configs) for t in range(args.trials)]
    t0 = time.time()
    results = [[] for _ in configs]
    if args.workers > 1:
        with Pool(args.workers) as pool:
            for key, row in pool.imap_unordered(_run, jobs, chunksize=8):
                results[key].append(row)
    else:
        for job in jobs:
            key, row = _run(job)
            results[key].append(row)
    elapsed = time.time() - t0

    base = os.path.join(args.out, args.tag)
    with open(base + ".jsonl", "w") as f:
        for cfg, rows in zip(configs, results):
            for row in sorted(rows, key=lambda r: r["trial"]):
                f.write(json.dumps({"config": cfg, **row}, default=float) + "\n")
    summary = [{"config": cfg, "metrics": aggregate(rows, args.margin)}
               for cfg, rows in zip(configs, results)]
    with open(base + "_summary.json", "w") as f:
        json.dump({"tag": args.tag, "args": vars(args), "seconds": elapsed, "rows": summary},
                  f, indent=2, default=float)
    header = f"delta = {args.delta:g}, trials = {args.trials} per row, {elapsed:.0f}s total"
    table = markdown(summary)
    with open(base + ".md", "w") as f:
        f.write(header + "\n\n" + table + "\n")
    print(header)
    print()
    print(table)


if __name__ == "__main__":
    main()
