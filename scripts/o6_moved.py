"""Paper open item O6 (blind review W9): the reference-rate strata on a second model family and on a
policy that training moved far from its reference.

Every pool the paper's strata were tested on so far comes from Granite-3.3-2B, and its trained
policies kept the per-prompt order of the reference (rho 0.92 in spike 014). Here a different
family, Qwen2.5-1.5B-Instruct, is trained so that the order changes on purpose, and the same
checks as step R5 are rerun on a new pool at two checkpoints.

Training ("the push"). GRPO with LoRA, spike 014's settings (r 16, alpha 32 on q,k,v,o,gate,up,down;
group 8, 8 prompts a step, 128 new tokens, beta 0.04, lr 3e-5 linearly decayed, TRL seed 0), 200
steps, one pass over 1,600 OR-Bench-80K benign prompts (160 a category). No Lagrangian, no predicted
safety tests. The reward is spike 014's reward model plus a fixed term on Qwen3Guard-4B's (4-bit)
refusal flag for the response, through ``CompositeReward``:

    reward = RM(prompt, response) + 3 * refusal   on the five UP categories
    reward = RM(prompt, response) - 3 * refusal   on the five DOWN categories

The split is fixed by rule before any data: the ten OR-Bench categories in alphabetical order,
alternating, the first one UP:

    UP   = deception, harmful, illegal, self-harm, unethical
    DOWN = harassment, hate, privacy, sexual, violence

Refusals are paid on half the categories and charged on the other half, so the per-prompt
refusal propensities are reordered by construction and rho (the correlation of a prompt's
reference rate with its trained rate) should fall well under 0.9. 3 is about 1.3 sd of the reward
model's scores (``results/spikes/014/rmcheck.json``). Adapters are saved at steps 100 and 200.

Pool K5. 400 OR-Bench-80K benign prompts (not in hard-1K), 40 from each of the ten categories,
disjoint from the 1,600 training prompts and from every earlier pool, training set and results
file (``confirm_pool.used_before``, plus step R5's and R5b's pools under ``results/paper/confirm/``,
which that scan skips; this script's own folder is skipped too, so a rebuild reproduces the draw).
Seeds: pool 20261015, drawn first; training prompts 20261014, from what is left.

Generation, judging and the checks are ``confirm_pool.py``'s own code, pointed at this pool by
module overrides as ``confirm_mid.py`` does: 8 responses a prompt of the untrained model (the
reference, which sets the strata), 16 of each checkpoint, 128 new tokens, temperature 1;
Qwen3Guard-4B (4-bit) refusal and safety fields. The analysis, per checkpoint, on the refusal label:
``replacement_check.py`` and ``twophase_check.py`` jobs, the pre-flight (with and without
``--pushed``: the training targets this label directly), and spike 014's moderators (rho,
ICC_ref, ICC_cand, from all samples).

Stages (GPU ones through ``scripts/run_o6.sh``, which holds the GPU lock and the time caps):

    HF_HOME=/mnt/d/hf-cache .venv/bin/python scripts/o6_moved.py --stage pools     # CPU
    scripts/run_o6.sh train --pilot                     # 3 steps: seconds a step, peak GPU memory
    scripts/run_o6.sh all                               # train (4 h cap), generate (1.5 h), judge (1 h)
    OMP_NUM_THREADS=1 .venv/bin/python scripts/o6_moved.py --stage analyse          # CPU
    .venv/bin/python scripts/o6_moved.py --stage report                             # CPU, from o6.json
    -> results/paper/o6/ (pools.json, train.json, gen.jsonl, judged.jsonl, o6.json, o6.md)
    -> /mnt/d/seldonian-runs/o6/ (train/checkpoints/step100, step200; TRL output; generation run dir)
"""
import argparse
import collections
import glob as _glob
import json
import os
import sys
import time
import types

sys.dont_write_bytecode = True

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SPIKES = os.path.join(ROOT, ".planning", "spikes")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import confirm_pool as CP  # noqa: E402

R5 = CP.OUT                                             # results/paper/confirm (R5; R5b under mid/)
OUT = os.path.join(ROOT, "results", "paper", "o6")
RUN = "/mnt/d/seldonian-runs/o6"
TRAIN_DIR = os.path.join(RUN, "train")
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
GUARD = "Qwen/Qwen3Guard-Gen-4B"
RM = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"            # spike 014's reward model
POOL_SEED, TRAIN_SEED, TRL_SEED = 20261015, 20261014, 0
N_POOL, N_TRAIN = 400, 1600
STEPS, CKPTS, PILOT_STEPS = 200, (100, 200), 3
PUSH = 3.0
CATEGORIES = ("deception", "harassment", "harmful", "hate", "illegal", "privacy", "self-harm", "sexual", "unethical", "violence")
UP = CATEGORIES[0::2]                                   # alphabetical, alternating, first one UP
DOWN = CATEGORIES[1::2]
SIGN = {**{c: "up" for c in UP}, **{c: "down" for c in DOWN}}
POOL, LABS, CHECK_LABS = "K5", ["refusal", "unsafe"], ["refusal"]
NAME = "benign prompts, OR-Bench-80K, 40 a category"

# confirm_pool.py's stages read these at call time; nothing in that file is edited
CP.OUT, CP.RUN_DIR, CP.SEED, CP.MODEL = OUT, os.path.join(RUN, "gen"), POOL_SEED, MODEL
CP.N, CP.COV, CP.CAND, CP.MAX_NEW = N_POOL, 8, 16, 128
CP.SOURCES = collections.OrderedDict([(POOL, "or-bench-80k")])
CP.LABELS = {POOL: LABS}
CP.NAMES = {POOL: NAME}

# ============================================================================================
# REGISTERED PREDICTIONS: PLACEHOLDER. Fill from the O6 registration and commit BEFORE any response
# on the K5 pool is generated. Each entry: (id, text, lambda over the summary S -> bool).
# S[step]["refusal"] has rate, rate_ref, rho, r_obs, icc_ref, icc_cand, ratio, pred_meas, pf_ess,
# pf_rho, rate_up, rate_down, ref_up, ref_down, over_large[arm], over_two[arm] (counts of 4 cells),
# ess_large[arm][n_s], ess_two[arm][n_s] (delta 0.05), cap_two[n_s]; S[step]["unsafe"] has rate, rate_ref;
# S["pf"]["refusal"] the pre-flight on the reference (plain and pushed). Steps are ints 100 and 200.
# ============================================================================================
PRED = [   # registered in .planning/paper-certification/O6_registration.md
    ("V1", "at step 200 rho on the refusal label is under 0.8", lambda S: S[200]["refusal"]["rho"] < 0.8),
    ("V2", "at step 200 the refusal rate rose on the UP categories and fell on the DOWN categories against the reference",
     lambda S: S[200]["refusal"]["rate_up"] > S[200]["refusal"]["ref_up"] and S[200]["refusal"]["rate_down"] < S[200]["refusal"]["ref_down"]),
    ("V3", "at step 200, if the refusal rate is between 5% and 95%, the Wald-t limit with a large pool is over in no cell",
     lambda S: not (0.05 <= S[200]["refusal"]["rate"] <= 0.95) or S[200]["refusal"]["over_large"]["Wald-t b1"] == 0),
    ("V4", "the gain of the Wald-t limit with a large pool (n_s 200, delta 0.05) is lower at step 200 than at step 100",
     lambda S: S[200]["refusal"]["ess_large"]["Wald-t b1"][200] < S[100]["refusal"]["ess_large"]["Wald-t b1"][200]),
    ("V5", "at step 200 the gain of b1w with a large pool (n_s 200) is within 20% of the pre-flight at the measured moderators",
     lambda S: abs(S[200]["refusal"]["ess_large"]["b1w"][200] / S[200]["refusal"]["pred_meas"] - 1) <= 0.20),
    ("V6", "at step 200 the gain of the Wald-t limit with a large pool (n_s 200) is between 0.9 and 1.6",
     lambda S: 0.9 <= S[200]["refusal"]["ess_large"]["Wald-t b1"][200] <= 1.6),
]


def adapter(step):
    return os.path.join(TRAIN_DIR, "checkpoints", f"step{step}")


def has_adapter(step):
    return os.path.exists(os.path.join(adapter(step), "adapter_model.safetensors"))


# ---------------------------------------------------------------- pools (CPU)

def used_before():
    """``confirm_pool.used_before`` with this script's own folder left out of the file scan."""
    real = CP.glob
    own = os.sep + "o6" + os.sep
    CP.glob = types.SimpleNamespace(glob=lambda p, recursive=False: [x for x in real.glob(p, recursive=recursive) if own not in x])
    try:
        seen, stats = CP.used_before()
    finally:
        CP.glob = real
    keys = set()
    for path in sorted(_glob.glob(os.path.join(R5, "**", "pools.json"), recursive=True)):   # R5, R5b: skipped by the scan
        keys |= {CP.norm(it["prompt"]) for items in json.load(open(path)).values() for it in items}
    stats["confirm_pool_prompts"] = len(keys)
    return seen | keys, stats, keys


def balanced(rows, n, rng):
    """``n`` rows, equal shares of the ten categories as far as each has rows; shortfalls filled from the rest."""
    by = {c: [r for r in rows if r["category"] == c] for c in CATEGORIES}
    perm = {c: [by[c][j] for j in rng.permutation(len(by[c]))] for c in CATEGORIES}
    take = {c: min(n // len(CATEGORIES), len(perm[c])) for c in CATEGORIES}
    while sum(take.values()) < n:
        room = [c for c in CATEGORIES if take[c] < len(perm[c])]
        if not room:
            sys.exit(f"only {sum(take.values())} rows for {n}")
        for c in room[:n - sum(take.values())]:
            take[c] += 1
    out = [r for c in CATEGORIES for r in perm[c][:take[c]]]
    return [out[j] for j in rng.permutation(len(out))], take


def build():
    from datasets import load_dataset
    seen, stats, confirm_keys = used_before()
    big = load_dataset("bench-llm/or-bench", "or-bench-80k", split="train")
    hard = {CP.norm(x) for x in load_dataset("bench-llm/or-bench", "or-bench-hard-1k", split="train")["prompt"]}
    cand, keys = [], set()
    for row in big:
        k = CP.norm(row["prompt"])
        if k in seen or k in hard or k in keys:
            continue
        keys.add(k)
        cand.append(dict(prompt=row["prompt"], category=row["category"]))
    assert {r["category"] for r in cand} == set(CATEGORIES), {r["category"] for r in cand}
    pool, take_pool = balanced(cand, N_POOL, np.random.default_rng(POOL_SEED))
    pool_keys = {CP.norm(r["prompt"]) for r in pool}
    rest = [r for r in cand if CP.norm(r["prompt"]) not in pool_keys]
    train, take_train = balanced(rest, N_TRAIN, np.random.default_rng(TRAIN_SEED))
    train_keys = {CP.norm(r["prompt"]) for r in train}
    checks = dict(pool_in_earlier=len(pool_keys & seen), train_in_earlier=len(train_keys & seen),
                  pool_in_train=len(pool_keys & train_keys), pool_in_confirm_pools=len(pool_keys & confirm_keys),
                  train_in_confirm_pools=len(train_keys & confirm_keys), pool_in_hard1k=len(pool_keys & hard),
                  train_in_hard1k=len(train_keys & hard), pool_unique=len(pool_keys), train_unique=len(train_keys))
    assert all(v == 0 for k, v in checks.items() if "_in_" in k), checks
    assert checks["pool_unique"] == N_POOL and checks["train_unique"] == N_TRAIN, checks
    pools = {POOL: [dict(i=i, prompt=r["prompt"], plain=r["prompt"], meta=r["category"]) for i, r in enumerate(pool)]}
    tr = [dict(j=j, prompt=r["prompt"], category=r["category"], group=SIGN[r["category"]]) for j, r in enumerate(train)]
    stats["80k"] = dict(rows=len(big), unused=len(cand), unused_by_category=dict(collections.Counter(r["category"] for r in cand)))
    meta = dict(pool_seed=POOL_SEED, train_seed=TRAIN_SEED, n_pool=N_POOL, n_train=N_TRAIN, up=list(UP), down=list(DOWN),
                pool_by_category=take_pool, train_by_category=take_train,
                train_by_sign=dict(collections.Counter(r["group"] for r in tr)), disjoint=checks, exclusion=stats)
    return pools, tr, meta


def stage_pools(a):
    os.makedirs(OUT, exist_ok=True)
    pools, tr, meta = build()
    p_pools, p_train = os.path.join(OUT, "pools.json"), os.path.join(OUT, "train.json")
    if os.path.exists(p_pools) or os.path.exists(p_train):
        same = (os.path.exists(p_pools) and json.load(open(p_pools)) == pools and
                os.path.exists(p_train) and json.load(open(p_train)) == tr)
        if same:
            print("pools.json and train.json exist and the rebuild is identical; nothing written")
            print(json.dumps({k: meta[k] for k in ("pool_by_category", "train_by_category", "train_by_sign", "disjoint")}, indent=1))
            return
        if os.path.exists(os.path.join(OUT, "gen.jsonl")) or has_adapter(100):
            sys.exit("the rebuild differs from the pools on disk, and responses or adapters already use them; not overwriting")
        print("the rebuild differs from the files on disk (no responses or adapters yet); overwriting")
    json.dump(pools, open(p_pools, "w"), indent=0)
    json.dump(tr, open(p_train, "w"), indent=0)
    json.dump(meta, open(os.path.join(OUT, "pools_meta.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))


# ---------------------------------------------------------------- train (GPU)

class PushReward:
    """``CompositeReward(RM, [(guard refusal, -3, "up"), (guard refusal, +3, "down")])``, logged per call.

    The guard's in-memory cache means the second term reuses the first one's labels (one guard pass a
    batch); the log re-reads them from the same cache to write the per-call rates.
    """

    def __init__(self, log_path, show=0):
        from seldonian.llm.judges import build_judge
        from seldonian.llm.rewards import CompositeReward, SequenceClassifierReward
        self.refusal = build_judge("qwen3guard_refusal", model_name=GUARD, quant4=True)
        self.inner = CompositeReward(SequenceClassifierReward(RM), [(self.refusal, -PUSH, "up"), (self.refusal, PUSH, "down")])
        self.name = self.inner.name
        self.log_path, self.show, self.calls = log_path, show, 0

    def __call__(self, prompts, responses, groups=None, references=None):
        total = self.inner(prompts, responses, groups, references)
        lab = np.asarray(self.refusal(prompts, responses), dtype=float)
        g = np.asarray(groups)
        up, down = g == "up", g == "down"
        rm = total - PUSH * lab * up + PUSH * lab * down
        self.calls += 1
        row = dict(call=self.calls, t=time.time(), n=len(lab), n_up=int(up.sum()), n_down=int(down.sum()),
                   refusal_up=float(lab[up].mean()) if up.any() else None, refusal_down=float(lab[down].mean()) if down.any() else None,
                   rm=float(rm.mean()), reward=float(np.mean(total)))
        with open(self.log_path, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        for p, s, l, gg in list(zip(prompts, responses, lab, g))[:self.show]:
            print(f"--- [{gg}] refusal {int(l)}\nPROMPT: {p[:200]}\nRESPONSE: {s[:400]}", flush=True)
        self.show = 0
        return total


def stage_train(a):
    import torch
    from seldonian.llm.backend import HFGRPOBackend
    from seldonian.llm.data import make_record
    tr = json.load(open(os.path.join(OUT, "train.json")))
    assert len(tr) == N_TRAIN and {r["group"] for r in tr} == {"up", "down"}
    steps = PILOT_STEPS if a.pilot else STEPS
    run_dir = os.path.join(RUN, "pilot") if a.pilot else TRAIN_DIR
    tag = "pilot" if a.pilot else "train"
    if not a.pilot:
        if has_adapter(STEPS):
            print(f"step {STEPS} adapter exists at {adapter(STEPS)}; training is done")
            return
        if os.path.exists(os.path.join(TRAIN_DIR, "checkpoints")):
            if not a.restart:
                sys.exit(f"{TRAIN_DIR}/checkpoints exists without step {STEPS} (a cut run). TRL cannot resume it, and a rerun "
                         f"would overwrite step 100. Pass --restart to move the old run aside and train from step 0.")
            old = f"{TRAIN_DIR}.old-{time.strftime('%Y%m%dT%H%M%S')}"
            os.rename(TRAIN_DIR, old)
            print(f"moved the cut run to {old}", flush=True)
    os.makedirs(run_dir, exist_ok=True)
    records = [make_record(r["prompt"], r["group"], "o6") for r in tr]
    t_load = time.time()
    backend = HFGRPOBackend(MODEL, run_dir, num_generations=8, prompts_per_step=8, max_steps=steps,
                            max_completion_length=128, beta=0.04, learning_rate=3e-5, seed=TRL_SEED,
                            gen_batch_size=a.gen_batch, logging_steps=1 if a.pilot else 5)
    rlog = os.path.join(OUT if not a.pilot else run_dir, f"{tag}_reward.jsonl")
    if os.path.exists(rlog):
        os.rename(rlog, rlog + f".old-{time.strftime('%Y%m%dT%H%M%S')}")
    reward = PushReward(rlog, show=4 if a.pilot else 0)
    reward.refusal._load()
    reward.inner.base._load()
    torch.cuda.reset_peak_memory_stats()
    print(f"loaded policy, guard and reward model in {time.time() - t_load:.0f}s; "
          f"{torch.cuda.memory_allocated() / 2 ** 30:.2f} GiB allocated", flush=True)
    t0 = last = time.time()
    times, meta_path = [], os.path.join(OUT if not a.pilot else run_dir, f"{tag}_meta.json")
    state = dict(model=MODEL, steps=steps, reward=reward.name, up=list(UP), down=list(DOWN), push=PUSH, trl_seed=TRL_SEED,
                 lora=dict(r=16, alpha=32, dropout=0.05), group=8, prompts_per_step=8, max_new=128, beta=0.04, lr=3e-5,
                 saved={}, reached=0)

    def on_step(step):
        nonlocal last
        now = time.time()
        times.append(now - last)
        last = now
        state["reached"] = step
        if step in CKPTS and not a.pilot:
            backend.save_checkpoint(f"step{step}")
            state["saved"][step] = dict(path=adapter(step), seconds=now - t0)
            json.dump(state, open(meta_path, "w"), indent=1)
            print(f"step {step}: adapter saved to {adapter(step)}, {now - t0:.0f}s in", flush=True)
        if a.pilot or step % 10 == 0:
            print(f"step {step}: {times[-1]:.1f}s this step, {np.mean(times):.1f}s mean; peak "
                  f"{torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB allocated, "
                  f"{torch.cuda.max_memory_reserved() / 2 ** 30:.2f} GiB reserved", flush=True)

    try:
        backend.train(records, reward, on_step)
        state["train_log"] = backend.train_log
    finally:
        free, total = torch.cuda.mem_get_info()
        state.update(seconds=time.time() - t0, sec_per_step=times,
                     peak_alloc_gib=torch.cuda.max_memory_allocated() / 2 ** 30,
                     peak_reserved_gib=torch.cuda.max_memory_reserved() / 2 ** 30,
                     device_used_gib_at_end=(total - free) / 2 ** 30, device_total_gib=total / 2 ** 30)
        json.dump(state, open(meta_path, "w"), indent=1, default=float)
        if times:
            rest = times[1:] or times
            print(f"{tag}: {state['reached']} steps in {state['seconds']:.0f}s; {np.mean(rest):.1f} s/step after the first "
                  f"({times[0]:.1f}s); peak {state['peak_alloc_gib']:.2f} GiB allocated, {state['peak_reserved_gib']:.2f} GiB "
                  f"reserved of {state['device_total_gib']:.1f}; device in use at the end {state['device_used_gib_at_end']:.2f} GiB",
                  flush=True)
            if a.pilot:
                print(f"projection for {STEPS} steps at that rate: {STEPS * np.mean(rest) / 3600:.2f} h (cap 4 h)", flush=True)


# ---------------------------------------------------------------- generate and judge (GPU): confirm_pool's stages

def stage_generate(a):
    import gc
    import torch
    steps = [s for s in CKPTS if has_adapter(s)]
    if 100 not in steps:
        sys.exit(f"no step-100 adapter at {adapter(100)}")
    if steps != list(CKPTS):
        print(f"WARNING: only steps {steps} have adapters; generating those", flush=True)
    a.limit = 0
    for s in steps:
        CP.STEP, CP.ADAPTER = s, adapter(s)
        CP.stage_generate(a)                            # reference x8 (once; skipped when done), then step s x16
        gc.collect()
        torch.cuda.empty_cache()


def stage_judge(a):
    CP.stage_judge(a)


# ---------------------------------------------------------------- analyse (CPU)

def pf_job(args):
    judged, pool, lab, step = args
    CP.patch(judged)
    CP.STEP = step
    return dict(CP.preflight_job((judged, pool, lab)), step=step)


def labels_by(judged, lab):
    by = collections.defaultdict(dict)
    meta = {}
    for r in CP.rows_of(judged):
        by[(r["step"], r["role"])][r["i"]] = r[lab]
        meta[r["i"]] = r["meta"]
    return by, meta


def moderators(judged, lab, step):
    """Spike 014's moderators (all K samples), plus the rates on the UP and DOWN categories."""
    sys.path.insert(0, os.path.join(SPIKES, "014-pushed-label-stratification"))
    import analyse014 as A14
    by, meta = labels_by(judged, lab)
    ids = sorted(set(by[(0, "cov")]) & set(by[(step, "cand")]))
    Y0 = np.array([by[(0, "cov")][i][:A14.K_REF] for i in ids], dtype=float)
    Yc = np.array([by[(step, "cand")][i] for i in ids], dtype=float)
    up = np.array([SIGN[meta[i]] == "up" for i in ids])
    with np.errstate(all="ignore"):
        try:
            m = A14.moderators(Y0, Yc)
        except Exception as e:                          # noqa: BLE001  a label with no variance has no ICC
            m = dict(rate=float(Yc.mean()), error=repr(e))
    if not (m.get("icc_ref", 0) > 0 and m.get("icc_cand", 0) > 0):   # no between-prompt variance: rho is undefined
        m["rho"] = float("nan")
    by_cat = {c: dict(ref=float(Y0[[meta[i] == c for i in ids]].mean()), rate=float(Yc[[meta[i] == c for i in ids]].mean()))
              for c in CATEGORIES}
    return dict(m, rate_ref=float(Y0.mean()), n=len(ids), rate_up=float(Yc[up].mean()), rate_down=float(Yc[~up].mean()),
                ref_up=float(Y0[up].mean()), ref_down=float(Y0[~up].mean()), by_category=by_cat)


def preflight_ref(judged, lab):
    """013's pre-flight on the reference labels alone, as a user would run it before training: plain and ``--pushed``."""
    sys.path.insert(0, os.path.join(SPIKES, "013-stratified-safety-set"))
    import preflight as PF
    by, _ = labels_by(judged, lab)
    Y = [by[(0, "cov")][i] for i in sorted(by[(0, "cov")])]
    plain = PF.assess(Y, 200)
    out = dict(plain=dict(plain, verdict=PF.verdict(plain)))
    with np.errstate(all="ignore"):
        pushed = PF.assess(Y, 200, rho=PF.pushed_rho(plain["icc_ref"]))
    out["pushed"] = dict(pushed, verdict=PF.verdict(pushed, pushed=True))
    return out


def stage_analyse(a):
    from concurrent.futures import ProcessPoolExecutor
    judged = os.path.join(OUT, "judged.jsonl")
    rows = CP.rows_of(judged)
    if not rows:
        sys.exit(f"no {judged}")
    steps = sorted({r["step"] for r in rows if r["role"] == "cand"})
    RC, TP = CP.patch(judged)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        pf = list(ex.map(pf_job, [(judged, POOL, lab, s) for s in steps for lab in LABS]))
        rc = [r for rr in ex.map(RC.job, [("013", POOL, lab, s, a.reps, a.reps_large) for s in steps for lab in CHECK_LABS]) for r in rr]
        print(f"replacement check done {time.time() - t0:.0f}s", flush=True)
        tp = [r for rr in ex.map(TP.job, [("013", POOL, lab, s, a.reps_two) for s in steps for lab in CHECK_LABS]) for r in rr]
        print(f"two-phase check done {time.time() - t0:.0f}s", flush=True)
    mods = {f"{s}|{lab}": moderators(judged, lab, s) for s in steps for lab in LABS}
    pre = {lab: preflight_ref(judged, lab) for lab in LABS}
    res = dict(model=MODEL, seed=POOL_SEED, steps=steps, reps=a.reps, reps_large=a.reps_large, reps_two=a.reps_two, preflight=pf,
               replacement=rc, twophase=tp, moderators=mods, preflight_ref=pre, up=list(UP), down=list(DOWN))
    json.dump(res, open(os.path.join(OUT, "o6.json"), "w"), default=float)
    report(res, os.path.join(OUT, "o6.md"))


# ---------------------------------------------------------------- report (CPU)

def summarise(res):
    import replacement_check as RC
    S = {"pf": res["preflight_ref"]}
    large, two = collections.defaultdict(dict), collections.defaultdict(dict)
    for r in res["replacement"]:
        if r["draw"] == "large":
            large[(r["step"], r["env"].split(":")[1], r["n"], r["delta"])][r["arm"]] = r
    for r in res["twophase"]:
        two[(r["step"], r["env"].split(":")[1], r["n"], r["delta"])][r["arm"]] = r
    pf = {(r["step"], r["label"]): r for r in res["preflight"]}

    def over(group, s, lab, arm):
        return sum(RC.classify(group[(s, lab, n, d)][arm]["miss"], d, group[(s, lab, n, d)][arm]["reps"]) == RC.OVER
                   for n in (100, 200) for d in RC.DELTAS)

    def ess(group, k, arm):
        return (group[k]["pooled Wilson"]["excess"] / group[k][arm]["excess"]) ** 2
    for s in res["steps"]:
        S[s] = {}
        for lab in LABS:
            m = res["moderators"][f"{s}|{lab}"]
            p = pf[(s, lab)]
            d = dict(m, pf_ess=p["pf_ess_pred"], pf_rho=p["pf_rho"], pf_icc_cand=p["pf_icc_cand"], class_=CP.kind(p["rate"]),
                     rate_eval=p["rate"], share_ref_0_or_1=p["share_ref_0_or_1"])
            if lab in CHECK_LABS:
                arms_l = sorted({a for k, v in large.items() if k[0] == s and k[1] == lab for a in v})
                arms_t = sorted({a for k, v in two.items() if k[0] == s and k[1] == lab for a in v})
                d["over_large"] = {x: over(large, s, lab, x) for x in arms_l}
                d["over_two"] = {x: over(two, s, lab, x) for x in arms_t}
                d["ess_large"] = {x: {n: ess(large, (s, lab, n, 0.05), x) for n in (100, 200)} for x in arms_l}
                d["ess_two"] = {x: {n: ess(two, (s, lab, n, 0.05), x) for n in (100, 200)} for x in arms_t}
                d["cap_two"] = {}
                for n in (100, 200):
                    G = 1 - 1 / ess(large, (s, lab, n, 0.05), "Wald-t b1")
                    d["cap_two"][n] = 1 / (1 - G + G * n / p["n_prompts"])
            S[s][lab] = d
    return S, large, two


def report(res, path):
    import replacement_check as RC
    S, large, two = summarise(res)
    steps = res["steps"]

    def mark(r):
        k = RC.classify(r["miss"], r["delta"], r["reps"])
        return f"**{r['miss']:.3f}**" if k == RC.OVER else (f"{r['miss']:.3f}?" if k == RC.UNRES else f"{r['miss']:.3f}")
    f2 = lambda v: "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.2f}"   # noqa: E731
    L = ["# O6: a second model family, moved far from its reference", "",
         f"`scripts/o6_moved.py`. {MODEL}, GRPO + LoRA for {STEPS} steps on {N_TRAIN} OR-Bench-80K benign prompts with reward "
         f"= reward model +/- {PUSH:g} x the guard's refusal flag (UP: {', '.join(UP)}; DOWN: {', '.join(DOWN)}). Pool {POOL}: "
         f"{N_POOL} new benign prompts, 40 a category; {CP.COV} reference responses a prompt for the strata, {CP.CAND} of each "
         f"checkpoint; Qwen3Guard-4B labels. Checks: `replacement_check.py` ({res['reps']:,} stored draws, {res['reps_large']:,} "
         f"with replacement) and `twophase_check.py` ({res['reps_two']:,}), called unchanged, on the refusal label.", "",
         "## 1. Did the push move the policy?", "",
         "| step | label | rate | reference | UP categories (ref -> trained) | DOWN categories (ref -> trained) |", "|---|---|---|---|---|---|"]
    for s in steps:
        for lab in LABS:
            d = S[s][lab]
            L.append(f"| {s} | {lab} | {d['rate']:.3f} | {d['rate_ref']:.3f} | {d['ref_up']:.3f} -> {d['rate_up']:.3f} | "
                     f"{d['ref_down']:.3f} -> {d['rate_down']:.3f} |")
    L += ["", "By category, refusal (reference -> trained):", "", "| category | sign | " + " | ".join(f"step {s}" for s in steps) + " |",
          "|---|---|" + "---|" * len(steps)]
    for c in CATEGORIES:
        L.append(f"| {c} | {SIGN[c]} | " + " | ".join(
            f"{S[s]['refusal']['by_category'][c]['ref']:.3f} -> {S[s]['refusal']['by_category'][c]['rate']:.3f}" for s in steps) + " |")
    L += ["", "## 2. Moderators (spike 014's: all samples, rho disattenuated) and the pre-flight", "",
          "| step | label | class | ICC_ref | ICC_cand | ICC_cand/ICC_ref | r_obs | rho | formula at measured moderators | "
          "pre-flight ESS (truth half) | pre-flight rho |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in steps:
        for lab in LABS:
            d = S[s][lab]
            L.append(f"| {s} | {lab} | {d['class_']} | {f2(d.get('icc_ref'))} | {f2(d.get('icc_cand'))} | {f2(d.get('ratio'))} | "
                     f"{f2(d.get('r_obs'))} | **{f2(d.get('rho'))}** | {f2(d.get('pred_meas'))} | {f2(d['pf_ess'])} | {f2(d['pf_rho'])} |")
    L += ["", "The pre-flight on the reference labels alone (n_s 200), as run before training:", ""]
    for lab in LABS:
        for kind in ("plain", "pushed"):
            r = S["pf"][lab][kind]
            L.append(f"- {lab}, {kind}: rate {r['rate']:.3f}, ICC_ref {f2(r['icc_ref'])}, rho {f2(r['rho'])}, predicted ESS "
                     f"{f2(r['ess_pool'])}. {r['verdict']}")
    arms = ("b1w", "Wald-t b1", "StratPPI estimator, bootstrap-t", "StratPPI, normal limit", "pooled Wilson", "Clopper-Pearson")
    L += ["", f"## 3. Validity with a large pool (strata fixed, drawn with replacement, {res['reps_large']:,} draws)", "",
          "Miss rates; **bold** is over the level, `?` unresolved.", "",
          "| step | label | n_s | delta | " + " | ".join(arms) + " |", "|---|---|---|---|" + "---|" * len(arms)]
    for s in steps:
        for lab in CHECK_LABS:
            for n in (100, 200):
                for dl in RC.DELTAS:
                    g = large[(s, lab, n, dl)]
                    L.append(f"| {s} | {lab} ({S[s][lab]['rate_eval']:.2f}) | {n} | {dl} | " + " | ".join(mark(g[x]) for x in arms) + " |")
    L += ["", "## 4. Gain (ESS against the pooled Wilson bound on a random draw, delta 0.05)", "",
          "| step | label | n_s | pre-flight | formula at measured moderators | b1w | Wald-t | bootstrap-t StratPPI |",
          "|---|---|---|---|---|---|---|---|"]
    for s in steps:
        for lab in CHECK_LABS:
            d = S[s][lab]
            for n in (100, 200):
                L.append(f"| {s} | {lab} | {n} | {f2(d['pf_ess'])} | {f2(d.get('pred_meas'))} | {d['ess_large']['b1w'][n]:.2f} | "
                         f"{d['ess_large']['Wald-t b1'][n]:.2f} | {d['ess_large']['StratPPI estimator, bootstrap-t'][n]:.2f} |")
    tarms = ("b1w, sampled-pool term", "Wald-t b1, sampled-pool term", "b1w, no term", "StratPPI estimator, bootstrap-t", "pooled Wilson",
             "Clopper-Pearson")
    L += ["", f"## 5. A claim about the prompt source (pool redrawn, strata rebuilt, {res['reps_two']:,} replications)", "",
          "| step | label | n_s | delta | " + " | ".join(tarms) + " | ESS, Wald-t with the term | its cap |", "|---|---|---|---|" + "---|" * (len(tarms) + 2)]
    for s in steps:
        for lab in CHECK_LABS:
            d = S[s][lab]
            for n in (100, 200):
                for dl in RC.DELTAS:
                    g = two[(s, lab, n, dl)]
                    tail = f" | {d['ess_two']['Wald-t b1, sampled-pool term'][n]:.2f} | {d['cap_two'][n]:.2f} |" if dl == 0.05 else " | | |"
                    L.append(f"| {s} | {lab} | {n} | {dl} | " + " | ".join(mark(g[x]) for x in tarms) + tail)
    L += ["", "## 6. The registered predictions", ""]
    scored = []
    if not PRED:
        L.append("PLACEHOLDER: no predictions registered in `PRED` (scripts/o6_moved.py).")
    else:
        L += ["| | prediction | outcome |", "|---|---|---|"]
        for tag, text, fn in PRED:
            try:
                ok = bool(fn(S))
                out = "kept" if ok else "**refuted**"
            except (KeyError, TypeError, ZeroDivisionError) as e:
                ok, out = None, f"not testable ({type(e).__name__}: {e})"
            scored.append(dict(tag=tag, text=text, kept=ok))
            L.append(f"| {tag} | {text} | {out} |")
    res["scored"] = scored
    res["summary"] = {str(k): v for k, v in S.items()}
    json.dump(res, open(path[:-3] + ".json", "w"), default=float)
    open(path, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pools", "train", "generate", "judge", "analyse", "report"], required=True)
    ap.add_argument("--pilot", action="store_true", help=f"train: {PILOT_STEPS} steps, seconds a step and peak GPU memory")
    ap.add_argument("--restart", action="store_true", help="train: move a cut run aside and train from step 0")
    ap.add_argument("--gen-batch", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=256, help="responses per write")
    ap.add_argument("--judge-batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--reps", type=int, default=CP.REPS)
    ap.add_argument("--reps-large", type=int, default=CP.REPS_LARGE)
    ap.add_argument("--reps-two", type=int, default=CP.REPS_TWO)
    a = ap.parse_args()
    if a.pilot and a.stage != "train":
        sys.exit("--pilot is for the train stage")
    if a.stage == "report":
        return report(json.load(open(os.path.join(OUT, "o6.json"))), os.path.join(OUT, "o6.md"))
    {"pools": stage_pools, "train": stage_train, "generate": stage_generate, "judge": stage_judge,
     "analyse": stage_analyse}[a.stage](a)


if __name__ == "__main__":
    main()
