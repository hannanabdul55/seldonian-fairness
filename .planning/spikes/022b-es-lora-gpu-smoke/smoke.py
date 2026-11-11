"""Spike 022b, GPU part: memory, exactness and throughput of ``HFESBackend`` on Granite-2B
with 014's adapter and reward stack, and a 5-step GRPO reference in the same process.

Run through ``run.sh`` (GPU lock and time cap). Writes ``results/spikes/022b/smoke.json``
after every phase, so a kill at the cap keeps the phases that finished.
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "014-pushed-label-stratification"))
import gen014  # noqa: E402
from eslora import HFESBackend  # noqa: E402
from seldonian.llm.backend import HFGRPOBackend  # noqa: E402

OUT = os.path.join(REPO, "results", "spikes", "022b")
RUN_DIR = "/mnt/d/seldonian-runs/022b"
CKPT = "/mnt/d/seldonian-runs/014/s0/checkpoints/feasible-step100"
CONFIGS = [(30, 8), (8, 32)]          # (population, prompts per member)
GRPO_STEPS = 5
RES = {}


def save():
    json.dump(RES, open(os.path.join(OUT, "smoke.json"), "w"), indent=1, default=float)


def smi():
    try:
        return int(subprocess.run(["/usr/lib/wsl/lib/nvidia-smi", "--query-gpu=memory.used",
                                   "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split()[0])
    except Exception:
        return None


def mem(reset=False):
    import torch
    m = dict(alloc_peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30,
             reserved_peak_gb=torch.cuda.max_memory_reserved() / 2 ** 30,
             alloc_now_gb=torch.cuda.memory_allocated() / 2 ** 30, smi_used_mib=smi())
    if reset:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    return m


def phase(name, t0):
    RES[name]["seconds"] = time.time() - t0
    RES[name]["memory"] = mem(reset=True)
    save()
    print(f"[{name}] {RES[name]['seconds']:.0f}s  {RES[name]['memory']}", flush=True)


def main():
    import torch
    os.makedirs(OUT, exist_ok=True)
    T0 = time.time()

    # ---------------------------------------------------------------- setup: 014's stack
    t0 = time.time()
    be = HFESBackend(gen014.MODEL, RUN_DIR, pop=30, prompts_per_member=8, sigma=1e-3, es_lr=1e-3,
                     num_generations=8, prompts_per_step=8, max_steps=GRPO_STEPS, max_completion_length=128,
                     beta=0.04, learning_rate=3e-5, seed=0, gen_batch_size=64, logging_steps=1,
                     log_path=os.path.join(OUT, "events.jsonl"))
    be.load_checkpoint(CKPT)
    from seldonian.llm.judges import build_judge
    from seldonian.llm.rewards import LagrangianReward, SequenceClassifierReward
    refusal = build_judge("qwen3guard_refusal", model_name="Qwen/Qwen3Guard-Gen-4B", quant4=True)
    reward = LagrangianReward(SequenceClassifierReward(gen014.RM), [(refusal, "benign")], names=["refusal"],
                              lam0=5.0, eta=100.0, lam_max=50.0, lam_floor=5.0)
    d_c = gen014.training_records(gen014.pool_c1(), 600, 600)
    rng = np.random.default_rng(7)
    pick = lambda m: [d_c[int(i)] for i in rng.choice(len(d_c), m, replace=False)]
    warm = pick(8)
    tx, _ = be.sample_scored([r["prompt"] for r in warm], 1)
    reward([r["prompt"] for r in warm], tx, groups=[r["group"] for r in warm],
           references=[r.get("reference") for r in warm])          # loads the judge and the reward model
    dt = {str(p.dtype) for _, p in be.params}
    RES["setup"] = dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(0), adapter_dtypes=sorted(dt),
                        base_dtype=str(next(p for p in be.model.parameters() if not p.requires_grad).dtype),
                        n_tensors=len(be.params), n_params=be.n_params, d_c=len(d_c),
                        b_norm=float(sum(float(p.float().norm() ** 2) for n, p in be.params if "lora_B" in n) ** 0.5))
    phase("setup", t0)

    # ---------------------------------------------------------------- exactness on CUDA
    t0 = time.time()
    ex = {}
    a, b, c = list(be.noise(11)), list(be.noise(11)), list(be.noise(12))
    ex["noise_same_seed_identical"] = all(torch.equal(x, y) for x, y in zip(a, b))
    ex["noise_other_seed_differs"] = not all(torch.equal(x, y) for x, y in zip(a, c))
    del a, b, c
    h0 = be.state_hash()
    t1 = be._sync()
    for i in range(200):
        be.set_member(1000 + i, 1.0 if i % 2 else -1.0)
        be.restore()
    ex["perturb_restore_seconds_each"] = (be._sync() - t1) / 200
    ex["restore_200_hash_unchanged"] = be.state_hash() == h0
    be.set_member(5, 1.0)
    hm = be.state_hash()
    be.restore()
    be.set_member(5, 1.0)
    ex["member_rewrite_identical"] = be.state_hash() == hm
    ex["member_differs_from_master"] = hm != h0
    # completions: the same member and sampling seed, twice, and after a restore in between
    for m in (8, 32):
        prompts = [r["prompt"] for r in pick(m)]
        tx1, s1 = be.sample_scored(prompts, 99)
        tx2, s2 = be.sample_scored(prompts, 99)
        be.restore()
        be.set_member(5, 1.0)
        tx3, s3 = be.sample_scored(prompts, 99)
        same = lambda u, v: u["seq"].shape == v["seq"].shape and bool(torch.equal(u["seq"], v["seq"]))
        fw = be.forward_logps(s1)                      # the member is still in the model
        gap = (fw - s1["logp"]).abs()[s1["mask"]]
        ex[f"m{m}"] = dict(repeat_same_tokens=same(s1, s2), repeat_after_rewrite=same(s1, s3),
                           logp_repeat_max_gap=float((s1["logp"] - s2["logp"]).abs()[s1["mask"]].max())
                           if same(s1, s2) else None,
                           score_vs_forward_max=float(gap.max()), score_vs_forward_mean=float(gap.mean()),
                           kl_floor_same_model=be.kl_term(s1, fw), tokens=int(s1["mask"].sum()))
        with be.model.disable_adapter():
            ex[f"m{m}"]["kl_vs_reference"] = be.kl_term(s1, be.forward_logps(s1))
    be.restore()
    ex["restored_after_checks"] = be.state_hash() == h0
    RES["exact"] = ex
    phase("exact", t0)
    print(json.dumps(ex, indent=1), flush=True)

    # ---------------------------------------------------------------- generation time vs batch size
    t0 = time.time()
    curve = []
    be.set_member(5, 1.0)
    for m in (8, 16, 32, 64):
        prompts = [r["prompt"] for r in pick(m)]
        for want in (True, False):
            torch.cuda.reset_peak_memory_stats()
            t1 = be._sync()
            _, s = be.sample_scored(prompts, 3, want_logps=want)
            sec = be._sync() - t1
            curve.append(dict(m=m, logps=want, seconds=sec, tokens=int(s["mask"].sum()),
                              new_len=int(s["seq"].shape[1] - s["P"]), prompt_len=int(s["P"]),
                              completions_per_s=m / sec, peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30))
            print(curve[-1], flush=True)
            del s
    be.restore()
    RES["curve"] = dict(rows=curve)
    phase("curve", t0)

    # ---------------------------------------------------------------- one ES generation per config
    RES["es"] = {}
    for pop, m in CONFIGS:
        t0 = time.time()
        name = f"pop{pop}_m{m}"
        be.load_checkpoint(CKPT)                        # resets the master copy and Adam
        be.pop, be.m = pop, m
        start = [x.clone() for x in be.master]
        t1 = be._sync()
        seeds, fitness, ev = be.generation(d_c, reward)
        t2 = be._sync()
        be.apply_update(seeds, fitness)
        ev["timing"]["update"] = be._sync() - t2
        total = be._sync() - t1
        rebuilt = be.replay([json.loads(json.dumps(ev))], start)      # through JSON, as from the log file
        ev.update(total_seconds=total, completions_per_hour=3600 * ev["completions"] / total,
                  tokens_per_s=ev["tokens"] / total,
                  replay_identical=be.state_hash(rebuilt) == be.state_hash(be.master),
                  update_moved_master=be.state_hash(start) != be.state_hash(be.master),
                  fitness_sd=float(np.std(fitness)))
        with open(os.path.join(OUT, "events.jsonl"), "a") as fh:
            fh.write(json.dumps(dict(ev, config=name), default=float) + "\n")
        RES["es"][name] = ev
        del start, rebuilt
        phase_mem = mem(reset=True)
        ev["memory"], ev["phase_seconds"] = phase_mem, time.time() - t0
        save()
        print(f"[es {name}] {total:.0f}s for {ev['completions']} completions "
              f"({ev['completions_per_hour']:.0f}/h), timing {ev['timing']}, replay {ev['replay_identical']}, "
              f"{phase_mem}", flush=True)

    # ---------------------------------------------------------------- GRPO reference, 014's configuration
    t0 = time.time()
    be.load_checkpoint(CKPT)
    marks = [time.time()]
    HFGRPOBackend.train(be, d_c, reward, lambda step: marks.append(time.time()))
    wall = np.diff(marks)
    log = [h for h in be.train_log if "step_time" in h]
    RES["grpo"] = dict(steps=GRPO_STEPS, completions_per_step=64, wall_per_step=[float(x) for x in wall],
                       step_time=[h["step_time"] for h in log],
                       mean_length=[h.get("completions/mean_length") for h in log],
                       kl=[h.get("kl") for h in log],
                       # the first step carries warm-up; the rate uses the others
                       completions_per_hour=3600 * 64 / float(np.mean(wall[1:])),
                       tokens_per_s=float(np.mean([h["completions/mean_length"] for h in log[1:]])) * 64
                       / float(np.mean(wall[1:])) if len(log) > 1 else None)
    phase("grpo", t0)
    print(json.dumps(RES["grpo"], indent=1), flush=True)

    best = max(v["completions_per_hour"] for v in RES["es"].values())
    RES["summary"] = dict(total_seconds=time.time() - T0,
                          es_over_grpo={k: v["completions_per_hour"] / RES["grpo"]["completions_per_hour"]
                                        for k, v in RES["es"].items()},
                          best_es_over_grpo=best / RES["grpo"]["completions_per_hour"])
    save()
    print(json.dumps(RES["summary"], indent=1), flush=True)


if __name__ == "__main__":
    main()
