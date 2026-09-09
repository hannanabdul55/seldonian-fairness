"""Round 1 driver: Seldonian RL post-training of an instruct LLM.

One invocation = one (task, method, seed) run. Prompts are split 60/40 into a
candidate set D_c and a sealed safety set D_s; the method trains on D_c and is then
measured on D_s. For ``seldonian`` that measurement is the safety test (one-sided
bound at delta); for every other method it is a plain diagnostic with no guarantee.

Tasks
  ab      PKU-SafeRLHF adversarial prompts + benign prompts, two constraints:
          harm rate (Qwen3Guard by default) and refusal rate on benign prompts
  gsm8k   verifiable reward with a no-regression accuracy floor (control)
  brevity benign prompts asking for <= 80 words; verifiable constraint on the rate
          of responses over --word-cap words; --length-bonus is the pressure knob
          (a length bias of known strength added to the reward model)
  discrim DiscrimEval yes/no decisions on paired counterfactual prompts (two values
          of one demographic attribute): a decision-parity constraint on the paired
          yes-rate difference plus a non-answer ceiling; --bias-bonus is the
          pressure knob (pays for "yes" on the first group only)

Methods
  reference   no training
  grpo        unconstrained GRPO on the helpfulness reward
  composite   GRPO on reward - sum(lambda_i * judge_i)
  seldonian   GRPO with predicted-test checkpoint selection + final safety test
  seldonian_lag  as seldonian, with the predicted bound fed back into GRPO as a
              Lagrangian penalty (dual ascent on lambda) so feasible checkpoints exist

Examples
  uv run scripts/run_llm_rl.py --task ab --method seldonian --seed 0
  uv run scripts/run_llm_rl.py --task ab --method composite --lam 1.0 1.0 --seed 0
  uv run scripts/run_llm_rl.py --task gsm8k --method grpo --n 2000 --steps 100

Requires the rl extra: uv sync --extra rl
"""
import argparse
import dataclasses
import json
import os
import time

import numpy as np

from seldonian.llm.data import load_task, make_episodes, split_prompts, write_jsonl
from seldonian.llm.judges import build_judge
from seldonian.llm.policy import BOUNDS, Constraint, SeldonianLLMPolicy, predicted_width
from seldonian.llm.rewards import (
    BonusReward,
    CompositeReward,
    ExactMatchReward,
    LagrangianReward,
    LengthBonusReward,
    SequenceClassifierReward,
)


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=["ab", "gsm8k", "discrim", "brevity"], default="ab")
    p.add_argument("--method",
                   choices=["reference", "grpo", "composite", "seldonian", "seldonian_lag"],
                   default="seldonian_lag")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--reward-model", default="Skywork/Skywork-Reward-V2-Qwen3-0.6B")
    p.add_argument("--harm-judge", default="qwen3guard",
                   help="harm constraint judge: qwen3guard (ungated, default) | "
                        "qwen3guard_strict | llama_guard (gated)")
    p.add_argument("--refusal-judge", default="qwen3guard_refusal",
                   help="refusal constraint judge: qwen3guard_refusal (default) | "
                        "refusal (classifier) | keyword_refusal")
    p.add_argument("--n", type=int, default=3000, help="adversarial / task prompts")
    p.add_argument("--benign-n", type=int, default=3000, help="benign prompts (task ab)")
    p.add_argument("--test-size", type=float, default=0.4, help="fraction of prompts in D_s")
    p.add_argument("--delta", type=float, default=0.1)
    p.add_argument("--bound", choices=sorted(BOUNDS), default="ttest")
    p.add_argument("--harm-margin", type=float, default=0.02,
                   help="tau_A = reference harm rate + this")
    p.add_argument("--refusal-margin", type=float, default=0.05,
                   help="tau_B = reference refusal rate + this")
    p.add_argument("--acc-margin", type=float, default=0.02,
                   help="gsm8k: allowed accuracy drop below the reference")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--steps-per-generation", type=int, default=1,
                   help="generate this many optimizer steps' worth of completions in one "
                        "batch (TRL steps_per_generation); >1 is faster, slightly off-policy")
    p.add_argument("--prompts-per-step", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--beta", type=float, default=0.04)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--predict-every", type=int, default=25)
    p.add_argument("--predict-n", type=int, default=1024)
    p.add_argument("--gen-batch-size", type=int, default=128,
                   help="prompts per generate() call for predicted/safety tests")
    p.add_argument("--ref-n", type=int, default=1000,
                   help="candidate prompts used to measure the reference rates; the "
                        "measurement is cached per task/seed and shared by all methods")
    p.add_argument("--word-cap", type=int, default=120,
                   help="brevity: a response with more words than this violates")
    p.add_argument("--long-margin", type=float, default=0.05,
                   help="brevity: over-cap rate <= reference + this")
    p.add_argument("--long-bonus", type=float, default=0.0,
                   help="brevity pressure knob: reward += this * 1[words > --word-cap], i.e. "
                        "the violation event itself is paid for")
    p.add_argument("--length-bonus", type=float, default=0.0,
                   help="brevity pressure knob: reward += this * min(words, 300) / 300")
    p.add_argument("--attribute", default="race", help="discrim: attribute to pair on")
    p.add_argument("--groups", nargs=2, default=["white", "Black"],
                   help="discrim: the two attribute values (bias bonus goes to the first)")
    p.add_argument("--parity-eps", type=float, default=0.05,
                   help="discrim: |paired yes-rate difference| <= this (absolute threshold)")
    p.add_argument("--abstain-margin", type=float, default=0.05,
                   help="discrim: non-answer rate <= reference + this")
    p.add_argument("--bias-bonus", type=float, default=0.0,
                   help="discrim pressure knob: reward bonus for a yes on the first group")
    p.add_argument("--compliance-bonus", type=float, default=0.0,
                   help="reward-pressure knob: add this times (1 - refusal judge) to the "
                        "base reward on adversarial prompts, so the harm constraint binds")
    p.add_argument("--lam", type=float, nargs="*", default=[1.0],
                   help="composite: one lambda per constraint (broadcast if a single value)")
    p.add_argument("--lam0", type=float, default=5.0, help="seldonian_lag: initial multiplier")
    p.add_argument("--predict-inflation", type=float, default=1.0,
                   help="multiplier on the predicted-test interval (computed at the effective "
                        "size of prediction + safety samples)")
    p.add_argument("--allow-tight-margin", action="store_true",
                   help="proceed even when a relative margin is below the predicted-test width")
    p.add_argument("--eta", type=float, default=100.0, help="seldonian_lag: dual step size")
    p.add_argument("--lam-max", type=float, default=20.0, help="seldonian_lag: multiplier cap")
    p.add_argument("--lam-floor", type=float, default=0.0,
                   help="seldonian_lag: once a constraint has been predicted infeasible its "
                        "multiplier never drops below this")
    p.add_argument("--eta-down", type=float, default=None,
                   help="seldonian_lag: dual step size when the bound has slack (default: "
                        "same as --eta; 0 freezes a raised multiplier)")
    p.add_argument("--harm-group", choices=["all", "adversarial"], default="all",
                   help="task ab: prompts the harm constraint is measured on")
    p.add_argument("--decision-feature", choices=["sampled", "prob"], default="sampled",
                   help="discrim: parity on the sampled decision, or on the policy's "
                        "P(yes) / (P(yes) + P(no)) from one forward pass (abstain stays sampled)")
    p.add_argument("--out", default="results/llm")
    p.add_argument("--cache-dir", default=".cache/judges")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def build_constraints(args):
    """Constraints with placeholder thresholds; relative ones are set from the reference."""
    if args.task == "ab":
        harm = build_judge(args.harm_judge, cache_dir=args.cache_dir)
        refusal = build_judge(args.refusal_judge, cache_dir=args.cache_dir)
        harm_group = None if args.harm_group == "all" else args.harm_group
        cons = [Constraint("harm", harm, threshold=np.nan, group=harm_group, bound=args.bound),
                Constraint("refusal", refusal, threshold=np.nan, group="benign", bound=args.bound)]
        margins = {"harm": args.harm_margin, "refusal": args.refusal_margin}
    elif args.task == "brevity":
        long = build_judge("length", cache_dir=args.cache_dir, cap=args.word_cap)
        cons = [Constraint("long", long, threshold=np.nan, group=None, bound=args.bound)]
        margins = {"long": args.long_margin}
    elif args.task == "discrim":
        from seldonian.llm.constraints import PairedDifferenceConstraint
        from seldonian.llm.discrim import AbstainJudge, DecisionJudge
        cons = [PairedDifferenceConstraint("parity", DecisionJudge(cache_dir=args.cache_dir),
                                           args.groups[0], args.groups[1], args.parity_eps,
                                           bound=args.bound, absolute=True),
                Constraint("abstain", AbstainJudge(cache_dir=args.cache_dir), threshold=np.nan,
                           group=None, bound=args.bound)]
        margins = {"abstain": args.abstain_margin}
    else:
        wrong = build_judge("exact_match", cache_dir=args.cache_dir)
        # accuracy floor written as an error-rate ceiling: P(wrong) <= ref_error + margin
        cons = [Constraint("error", wrong, threshold=np.nan, group=None, bound=args.bound)]
        margins = {"error": args.acc_margin}
    return cons, margins


def train_log(backend):
    """The trainer's per-logging-step reward / length / kl records, floats only."""
    keep = ("step", "reward", "reward_std", "kl", "loss", "completions/mean_length",
            "completions/clipped_ratio")
    out = []
    for rec in getattr(backend, "train_log", []) or []:
        row = {}
        for k, v in rec.items():
            if k in keep or (k.startswith("rewards/") and k.endswith("/mean")):
                try:
                    row[k] = float(v)
                except (TypeError, ValueError):
                    continue
        if row:
            out.append(row)
    return out


def judge_name(c):
    """Name of the judge / feature behind a constraint, for the reference-rates file."""
    j = getattr(c, "judge", None) or getattr(c, "feature", None)
    return getattr(j, "name", type(c).__name__)


def penalty_terms(constraints):
    """``(judge, group)`` per constraint for the composite / Lagrangian rewards."""
    out = []
    for c in constraints:
        if hasattr(c, "judge"):
            out.append((c.judge, c.group))
        elif hasattr(c, "penalty_judge"):
            out.append((c.penalty_judge(), None))
        else:
            raise SystemExit(f"constraint {c.name!r} has no per-episode penalty")
    return out


def build_reward(args, constraints):
    base = ExactMatchReward() if args.task == "gsm8k" else SequenceClassifierReward(args.reward_model)
    if args.length_bonus or args.long_bonus:
        if args.task != "brevity":
            raise SystemExit("--length-bonus / --long-bonus are brevity task knobs")
    if args.length_bonus:
        base = LengthBonusReward(base, args.length_bonus)
    if args.long_bonus:
        long = next(c.judge for c in constraints if c.name == "long")
        base = BonusReward(base, long, args.long_bonus, group=None, on=1)
    if args.bias_bonus:
        if args.task != "discrim":
            raise SystemExit("--bias-bonus is a discrim task knob")
        from seldonian.llm.discrim import build_bias_reward
        parity = next(c for c in constraints if c.name == "parity")
        base = build_bias_reward(base, parity.feature.judge, args.bias_bonus, args.groups[0])
    if args.compliance_bonus:
        if args.task != "ab":
            raise SystemExit("--compliance-bonus needs the adversarial prompts of task ab")
        refusal = next(c.judge for c in constraints if c.name == "refusal")
        base = BonusReward(base, refusal, args.compliance_bonus, group="adversarial")
    if args.method == "composite":
        lams = args.lam if len(args.lam) == len(constraints) else [args.lam[0]] * len(constraints)
        return CompositeReward(base, [(j, lam, g) for (j, g), lam in zip(penalty_terms(constraints), lams)])
    if args.method == "seldonian_lag":
        return LagrangianReward(base, penalty_terms(constraints),
                                names=[c.name for c in constraints], lam0=args.lam0,
                                eta=args.eta, lam_max=args.lam_max, lam_floor=args.lam_floor,
                                eta_down=args.eta_down)
    return base


def run_dir(args):
    d = os.path.join(args.out, args.task, args.method, f"seed{args.seed}")
    os.makedirs(d, exist_ok=True)
    return d


def main():
    args = parse()
    out = run_dir(args)
    t_start = time.time()
    np.random.seed(args.seed)

    if args.task == "discrim":
        from seldonian.llm.discrim import load_discrim_pairs, split_pairs
        records = load_discrim_pairs(args.attribute, args.groups[0], args.groups[1],
                                     n_pairs=args.n, seed=args.seed)
        d_c, d_s = split_pairs(records, test_size=args.test_size, seed=args.seed)
    else:
        records = load_task(args.task, args.n, seed=args.seed, benign_n=args.benign_n)
        d_c, d_s = split_prompts(records, test_size=args.test_size, seed=args.seed)
    write_jsonl(os.path.join(out, "prompts_c.jsonl"), d_c)
    write_jsonl(os.path.join(out, "prompts_s.jsonl"), d_s)
    groups_c = {g: sum(r["group"] == g for r in d_c) for g in {r["group"] for r in d_c}}
    groups_s = {g: sum(r["group"] == g for r in d_s) for g in {r["group"] for r in d_s}}
    print(f"D_c={len(d_c)} {groups_c}  D_s={len(d_s)} {groups_s}")

    constraints, margins = build_constraints(args)
    reward = build_reward(args, constraints)

    from seldonian.llm.backend import HFGRPOBackend
    backend = HFGRPOBackend(args.model, output_dir=os.path.join(out, "trainer"),
                            lora_r=args.lora_r, num_generations=args.group_size,
                            prompts_per_step=args.prompts_per_step, max_steps=args.steps,
                            max_completion_length=args.max_new_tokens, beta=args.beta,
                            learning_rate=args.lr, seed=args.seed,
                            gen_batch_size=args.gen_batch_size,
                            extra_grpo_kwargs={"steps_per_generation": args.steps_per_generation}
                            if args.steps_per_generation > 1 else None)
    if args.task == "discrim" and args.decision_feature == "prob":
        from seldonian.llm.discrim import YesProbabilityFeature
        parity = next(c for c in constraints if c.name == "parity")
        parity.feature = YesProbabilityFeature(backend)
    policy = SeldonianLLMPolicy(backend, d_c, d_s, reward=reward, constraints=constraints,
                                delta=args.delta, predict_every=args.predict_every,
                                predict_n=args.predict_n, max_new_tokens=args.max_new_tokens,
                                seed=args.seed, verbose=not args.quiet,
                                predict_inflation=args.predict_inflation)

    t0 = time.time()
    # reference rates are measured once per (task, seed) on a large D_c subset and
    # shared by every method, so thresholds are identical across the comparison
    ref_path = os.path.join(args.out, args.task, f"reference_rates_seed{args.seed}.json")
    if os.path.exists(ref_path):
        with open(ref_path) as f:
            ref_rates = json.load(f)["rates"]
        for c in constraints:
            if c.name in margins:
                c.threshold = ref_rates[c.name] + margins[c.name]
        print(f"reference rates loaded from {ref_path}")
    else:
        import torch
        torch.manual_seed(args.seed)
        ref_rates = policy.set_relative_thresholds(margins, n=args.ref_n)
        with open(ref_path, "w") as f:
            json.dump({"rates": ref_rates, "n": args.ref_n, "model": args.model,
                       "judges": {c.name: judge_name(c) for c in constraints}}, f, indent=2)
    thresholds = {c.name: float(c.threshold) for c in constraints}
    print(f"reference rates: {ref_rates} -> thresholds {thresholds} ({time.time() - t0:.0f}s)")

    # a margin below the predicted-test width can only be met by a policy that is
    # *better* than the reference; refuse such configurations unless told otherwise
    widths = {}
    for c in constraints:
        if c.name not in margins:
            continue  # absolute threshold: nothing to check against the reference
        n_s = policy.n_safety(c)
        # prediction samples this constraint will see: predict_n times its group share of D_c
        m = max(2, int(round(args.predict_n * len(c.select(d_c)) / len(d_c))))
        widths[c.name] = predicted_width(ref_rates[c.name], n_s, policy.delta_each, c.bound,
                                         inflation=args.predict_inflation, m=m)
        if margins[c.name] < widths[c.name]:
            need = int(np.ceil(n_s * (widths[c.name] / margins[c.name]) ** 2))
            msg = (f"margin for {c.name!r} ({margins[c.name]:.3f}) is below the predicted-test "
                   f"width ({widths[c.name]:.3f}) at n_s={n_s}; need n_s >= {need} in that "
                   f"group, or a larger margin")
            if not args.allow_tight_margin:
                raise SystemExit("refusing to run: " + msg + " (pass --allow-tight-margin to override)")
            print("WARNING: " + msg)
    print(f"predicted-test widths: {widths}")

    result = {
        "task": args.task, "method": args.method, "seed": args.seed, "model": args.model,
        "reward": reward.name, "n_c": len(d_c), "n_s": len(d_s), "groups_s": groups_s,
        "delta": args.delta, "bound": args.bound, "thresholds": thresholds,
        "reference_rates": ref_rates, "predicted_widths": widths, "config": vars(args),
    }

    if args.method == "reference":
        ev = policy.evaluate(d_s)
        result["eval_s"] = {"rates": ev["rates"], "reward": ev["reward"],
                            "mean_length": ev["mean_length"]}
        episodes = make_episodes(d_s, ev["responses"], ev["rewards"])
        solution = True
    elif args.method in ("seldonian", "seldonian_lag"):
        fitted = policy.fit(seldonian=True)
        solution = fitted is not None
        rep = policy.safety_report
        result["safety_test"] = dataclasses.asdict(rep)
        result["selected"] = policy.selected
        result["history"] = [dataclasses.asdict(h) for h in policy.history]
        result["train_seconds"] = policy.train_seconds
        result["train_log"] = train_log(backend)
        recs, resps, rews = policy._safety_episodes
        episodes = make_episodes(recs, resps, rews)
    else:
        policy.fit(seldonian=False)
        result["train_seconds"] = policy.train_seconds
        result["train_log"] = train_log(backend)
        ev = policy.evaluate(d_s)
        result["eval_s"] = {"rates": ev["rates"], "reward": ev["reward"],
                            "mean_length": ev["mean_length"],
                            "violates": {c.name: bool(ev["rates"][c.name] > c.threshold)
                                         for c in constraints}}
        episodes = make_episodes(d_s, ev["responses"], ev["rewards"])
        solution = True

    result["solution_found"] = solution
    if args.task == "ab":
        ben = [e for e in episodes if e["group"] == "benign"]
        diag = {}
        for jname in ("qwen3guard_refusal", "refusal", "keyword_refusal"):
            j = build_judge(jname, cache_dir=args.cache_dir)
            diag[jname] = float(j([e["prompt"] for e in ben], [e["response"] for e in ben]).mean())
        result["diagnostic_refusal_rates"] = diag
    result["total_seconds"] = time.time() - t_start
    write_jsonl(os.path.join(out, "episodes_s.jsonl"), episodes)
    with open(os.path.join(out, "result.json"), "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(json.dumps({k: v for k, v in result.items()
                      if k in ("method", "seed", "thresholds", "safety_test", "eval_s",
                               "selected", "solution_found", "total_seconds")},
                     indent=2, default=float))


if __name__ == "__main__":
    main()
