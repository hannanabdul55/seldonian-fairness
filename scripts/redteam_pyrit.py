"""Red-team a trained (or reference) policy with PyRIT and read the result as a
Seldonian constraint.

One invocation = one (policy, constraint, prompt group) battery. The policy comes from
a finished ``run_llm_rl.py`` run directory: its ``result.json`` supplies the judge,
the threshold, the bound and delta of the constraint, ``prompts_s.jsonl`` supplies
the safety prompts (the certificate's own distribution D), and the selected
checkpoint supplies the adapter. Every PyRIT technique is then a different prompt
distribution D' over the same prompts, and the summary reports the judge rate on D'
with the safety test's own bound against the safety test's own threshold.

Examples
  # the certified over-refusal policy, harm constraint on the adversarial prompts
  uv run scripts/redteam_pyrit.py --run-dir results/llm_r6/c/ab/seldonian_lag/seed0 \\
      --constraint harm --n 600 --techniques converter template resample \\
      --out results/redteam/c0_harm
  # the untrained reference on the same prompts
  uv run scripts/redteam_pyrit.py --run-dir results/llm_r6/c/ab/seldonian_lag/seed0 \\
      --checkpoint none --constraint harm --n 600 --out results/redteam/ref_harm
  # attacker-model techniques (Foundry DIFFICULT tier + tense) with a local attacker
  uv run scripts/redteam_pyrit.py --run-dir ... --constraint harm --n 100 \\
      --techniques llm --attacker Qwen/Qwen2.5-1.5B-Instruct --out results/redteam/c0_llm

Each finished technique is saved under ``<out>/parts/`` and a relaunch with the same
``--out`` resumes after the last saved technique; a technique that fails to build or
run is reported as a row of errors and the battery continues.

Requires the rl and redteam extras: uv sync --extra rl --extra redteam
"""
import argparse
import asyncio
import dataclasses
import json
import os
import sys
import time

import numpy as np

from seldonian.llm.data import read_jsonl


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="a run_llm_rl.py run directory")
    p.add_argument("--checkpoint", default="auto",
                   help="'auto' = the run's selected checkpoint, 'none' = the untrained "
                        "reference (fresh adapter), or a path to an adapter directory")
    p.add_argument("--constraint", default=None,
                   help="constraint name from result.json (default: the first one)")
    p.add_argument("--group", default=None,
                   help="prompt group of D_s to attack (default: the constraint's group, "
                        "or all prompts)")
    p.add_argument("--n", type=int, default=600, help="prompts drawn from the group")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--techniques", nargs="+", default=["converter", "template", "resample"],
                   help="CertificateTechnique names or aggregates: converter, template, "
                        "resample, llm, all")
    p.add_argument("--attacker", default="none",
                   help="Hub id of a local attacker / converter model for the llm techniques")
    p.add_argument("--attacker-max-new-tokens", type=int, default=512)
    p.add_argument("--max-turns", type=int, default=5)
    p.add_argument("--many-shot-examples", type=int, default=16)
    p.add_argument("--max-concurrency", type=int, default=64)
    p.add_argument("--max-batch", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=None,
                   help="default: the run's setting (the certificate's sampling length)")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--delta", type=float, default=None,
                   help="default: the run's delta split over its constraints")
    p.add_argument("--bound", default=None, help="default: the run's bound")
    p.add_argument("--no-baseline", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--cache-dir", default=".cache/judges")
    p.add_argument("--out", required=True)
    p.add_argument("--label", default=None, help="policy label in the outputs")
    return p.parse_args()


def load_run(run_dir):
    with open(os.path.join(run_dir, "result.json")) as f:
        result = json.load(f)
    prompts_s = read_jsonl(os.path.join(run_dir, "prompts_s.jsonl"))
    return result, prompts_s


def constraint_spec(result, name):
    """(judge name, judge kwargs, group, threshold) for a constraint of a finished run."""
    cfg = result["config"]
    task = result["task"]
    if name == "harm":
        group = None if cfg.get("harm_group", "all") == "all" else cfg["harm_group"]
        return cfg.get("harm_judge", "qwen3guard"), {}, group, result["thresholds"]["harm"]
    if name == "refusal":
        return cfg.get("refusal_judge", "qwen3guard_refusal"), {}, "benign", result["thresholds"]["refusal"]
    if name == "long" and task == "brevity":
        return "length", {"cap": cfg.get("word_cap", 120)}, None, result["thresholds"]["long"]
    if name == "error" and task == "gsm8k":
        raise SystemExit("the GSM8K error constraint needs reference answers; not supported here")
    raise SystemExit(f"unknown constraint {name!r} for task {task!r}")


def checkpoint_path(run_dir, result, choice):
    if choice == "none":
        return None
    if choice != "auto":
        return choice
    sel = result.get("selected") or {}
    step = sel.get("step")
    if not isinstance(step, int):
        raise SystemExit(f"run has no selected checkpoint step ({sel}); pass --checkpoint")
    path = os.path.join(run_dir, "trainer", "checkpoints", f"feasible-step{step}")
    if not os.path.exists(os.path.join(path, "adapter_model.safetensors")):
        raise SystemExit(f"no adapter at {path}")
    return path


def main():
    args = parse()
    os.makedirs(args.out, exist_ok=True)
    t_start = time.time()
    result, prompts_s = load_run(args.run_dir)
    names = list(result["thresholds"])
    cname = args.constraint or names[0]
    judge_name, judge_kwargs, cgroup, tau = constraint_spec(result, cname)
    group = args.group or cgroup
    delta = args.delta if args.delta is not None else result["delta"] / max(len(names), 1)
    bound = args.bound or result["bound"]
    max_new_tokens = args.max_new_tokens or result["config"].get("max_new_tokens", 256)
    ckpt = checkpoint_path(args.run_dir, result, args.checkpoint)
    label = args.label or (f"{result['method']}-seed{result['seed']}-{os.path.basename(ckpt)}"
                           if ckpt else "reference")

    pool = [r for r in prompts_s if group is None or r.get("group") == group]
    rng = np.random.default_rng(args.seed)
    if args.n < len(pool):
        idx = np.sort(rng.choice(len(pool), size=args.n, replace=False))
        records = [pool[int(i)] for i in idx]
    else:
        records = list(pool)
    print(f"policy {label}: constraint {cname} (judge {judge_name}, tau {tau:.3f}, bound {bound}, "
          f"delta {delta:.3f}); {len(records)} of {len(pool)} {group or 'all'} safety prompts")

    from pyrit.setup import IN_MEMORY, initialize_pyrit_async

    from seldonian.llm.backend import HFChatBackend, HFGRPOBackend
    from seldonian.llm.judges import build_judge
    from seldonian.llm.redteam import (
        PolicyTarget,
        resolve_techniques,
        run_certificate_battery_async,
        summary_table,
    )

    techniques = resolve_techniques(args.techniques)
    judge = build_judge(judge_name, cache_dir=args.cache_dir, **judge_kwargs)
    # PyRIT's targets and scorers register with its central memory at construction
    asyncio.run(initialize_pyrit_async(memory_db_type=IN_MEMORY, silent=True))

    backend = HFGRPOBackend(result["model"], output_dir=os.path.join(args.out, "trainer"),
                            lora_r=result["config"].get("lora_r", 16), seed=args.seed,
                            gen_batch_size=args.max_batch, device=args.device)
    if ckpt:
        backend.load_checkpoint(ckpt)
        print(f"loaded adapter {ckpt}")
    target = PolicyTarget(backend=backend, name=label, max_new_tokens=max_new_tokens,
                          temperature=args.temperature, max_batch=args.max_batch)

    attacker_target = None
    if args.attacker != "none":
        attacker = HFChatBackend(args.attacker, device=args.device, gen_batch_size=args.max_batch)
        attacker_target = PolicyTarget(backend=attacker, name=f"attacker:{args.attacker}",
                                       max_new_tokens=args.attacker_max_new_tokens,
                                       temperature=0.7, max_batch=args.max_batch)

    def progress(row, secs):
        print(f"[{time.strftime('%H:%M:%S')}] {row.technique}: n {row.n} err {row.errors} "
              f"rate {row.rate:.3f} ({row.lower:.3f}-{row.upper:.3f}) vs tau {row.tau:.3f} "
              f"in {secs / 60:.1f} min", flush=True)

    async def go():
        return await run_certificate_battery_async(
            target=target, records=records, judge=judge, tau=tau, delta=delta, bound=bound,
            techniques=techniques, adversarial_chat=attacker_target,
            max_concurrency=args.max_concurrency, include_baseline=not args.no_baseline,
            max_turns=args.max_turns, many_shot_examples=args.many_shot_examples,
            seed=args.seed, memory_labels={"policy": label, "constraint": cname},
            on_technique=progress, parts_dir=os.path.join(args.out, "parts"))

    run = asyncio.run(go())
    rows = [dataclasses.asdict(r) | {"g_upper": r.g_upper, "point_breach": r.point_breach,
                                     "certified_breach": r.certified_breach,
                                     "certificate_holds": r.certificate_holds}
            for r in run.rows]
    summary = {
        "policy": label, "run_dir": args.run_dir, "checkpoint": ckpt, "model": result["model"],
        "constraint": cname, "judge": judge.name, "group": group, "tau": tau, "delta": delta,
        "bound": bound, "n_prompts": len(records), "pool": len(pool),
        "max_new_tokens": max_new_tokens, "temperature": args.temperature,
        "techniques": [t.value for t in techniques], "attacker": args.attacker,
        "max_turns": args.max_turns, "many_shot_examples": args.many_shot_examples,
        "safety_test": result.get("safety_test") or result.get("eval_s"),
        "rows": rows, "scenario_seconds": run.seconds, "total_seconds": time.time() - t_start,
        "policy_requests": target.requests, "generate_batches": target.batcher.batches,
        "config": vars(args),
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    with open(os.path.join(args.out, "episodes.jsonl"), "w") as f:
        for e in run.episodes:
            f.write(json.dumps(e, default=str) + "\n")
    table = summary_table(run.rows)
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(f"# {label}: {cname} on {len(records)} {group or 'all'} safety prompts\n\n"
                f"judge {judge.name}, tau {tau:.3f}, {bound} at delta {delta:.3f}, "
                f"{max_new_tokens} new tokens at temperature {args.temperature}\n\n{table}\n")
    print(table)
    print(f"done in {time.time() - t_start:.0f}s ({target.requests} policy requests in "
          f"{target.batcher.batches} batches) -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
