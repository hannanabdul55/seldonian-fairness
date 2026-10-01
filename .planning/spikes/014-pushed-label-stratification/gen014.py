"""Spike 014 stage 2: Granite over-refusal pushed by the Lagrangian (013's C4), GPU.

Runs only after stage 1's gate opened (``gate.json``). Four sub-stages:

- ``rmcheck``  does Round 6's reward model prefer Granite's refusals on the C1 pool? Scores
               013's 8 reference samples per prompt (4,000 responses, labels on disk) with
               Skywork-Reward-V2-Qwen3-0.6B. No training. The design's pilot stop: if the
               model does not reward refusals over answers, nothing here can push the label.
- ``pilot``    20 training steps with the full stack loaded (policy, 4B refusal judge, reward
               model), 100 pool prompts x 4 samples at the end: throughput and memory.
- ``train``    the run (pilot-sized: 60 s a step with the judge and reward model resident,
               3.2 pool generations a second, so K = 12 at step 200 and 4 at step 100): ``SeldonianLLMPolicy`` with ``LagrangianReward`` (Round 6's dual
               settings) constraining Qwen3Guard-4B's refusal field on benign prompts to the
               pool's reference rate + 0.02, reward = the reward model, D_c = fresh ``ab``
               prompts disjoint from the pool, D_s = 013's C1 pool. At steps 100 and 200 the
               pool is sampled K times per prompt (the plasmode's candidates); the policy's
               own safety test on D_s runs at the end, so the run also reports a real
               Seldonian outcome.
- ``judge``    Qwen3Guard-4B on the sampled responses, as 013's judge stage.

    ./run.sh --stage rmcheck
    ./run.sh --stage pilot
    ./run.sh --stage train --steps 200 --cand 16
    ./run.sh --stage judge
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(HERE, "..", "013-stratified-safety-set"))
sys.path.insert(0, os.path.join(HERE, "..", "009-granite-transfer"))
import gen013  # noqa: E402

OUT = os.path.join(REPO, "results", "spikes", "014")
OUT013 = os.path.join(REPO, "results", "spikes", "013")
RUN_DIR = "/mnt/d/seldonian-runs/014"
MODEL = gen013.MODEL
RM = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
CKPT_STEPS = {100: 4, 200: None}   # samples per prompt; None = --cand (the plasmode checkpoint)
MARGIN = 0.02
DELTA = 0.05


def pool_c1():
    return json.load(open(os.path.join(OUT013, "pools.json")))["C1"]


def reference_rate():
    """The pool's reference refusal rate from 013's 8 covariate samples per prompt."""
    Y = [r["refusal"] for r in map(json.loads, open(os.path.join(OUT013, "judged_full.jsonl")))
         if r["pool"] == "C1" and r["step"] == 0 and r["role"] == "cov"]
    return float(np.mean([np.mean(y) for y in Y])), len(Y)


def gate_open():
    g = os.path.join(HERE, "gate.json")
    return os.path.exists(g) and json.load(open(g))["verdict"].startswith("OPEN")


def training_records(pool, n_adv, n_ben, seed=1):
    from seldonian.llm.data import load_task
    seen = {it["plain"] for it in pool}
    recs = [r for r in load_task("ab", n_adv, seed=seed, benign_n=n_ben + 700) if r["prompt"] not in seen]  # XSTest is all in the pool; OR-Bench fills D_c
    ben = [r for r in recs if r["group"] == "benign"][:n_ben]
    adv = [r for r in recs if r["group"] == "adversarial"][:n_adv]
    return adv + ben


def stage_rmcheck(a):
    """Reward-model scores of the reference's own responses, split by the guard's refusal label."""
    import torch
    from seldonian.llm.backend import disable_triton_overrides_without_compiler
    from seldonian.llm.rewards import SequenceClassifierReward
    disable_triton_overrides_without_compiler()          # a lone model needs this (013)
    os.makedirs(OUT, exist_ok=True)
    pool = {it["i"]: it for it in pool_c1()}
    gen = {r["i"]: r for r in map(json.loads, open(os.path.join(OUT013, "gen_full.jsonl")))
           if r["pool"] == "C1" and r["step"] == 0 and r["role"] == "cov"}
    jud = {r["i"]: r for r in map(json.loads, open(os.path.join(OUT013, "judged_full.jsonl")))
           if r["pool"] == "C1" and r["step"] == 0 and r["role"] == "cov"}
    prompts, responses, labels, metas = [], [], [], []
    for i in sorted(gen):
        for s, lab in zip(gen[i]["responses"], jud[i]["refusal"]):
            prompts.append(pool[i]["prompt"]); responses.append(s); labels.append(lab); metas.append(gen[i]["meta"])
    rm = SequenceClassifierReward(RM)
    t0 = time.time()
    scores = []
    with torch.no_grad():
        for b in range(0, len(prompts), 64):
            scores.extend(map(float, rm(prompts[b:b + 64], responses[b:b + 64])))
    scores, labels, metas = np.array(scores), np.array(labels), np.array(metas)
    out = dict(n=len(scores), seconds=time.time() - t0, refusal_rate=float(labels.mean()),
               mean_refused=float(scores[labels == 1].mean()), mean_answered=float(scores[labels == 0].mean()),
               sd=float(scores.std()))
    out["gap_in_sd"] = (out["mean_refused"] - out["mean_answered"]) / out["sd"]
    for m in np.unique(metas):
        k = metas == m
        out[f"gap_{m}"] = float(scores[k & (labels == 1)].mean() - scores[k & (labels == 0)].mean())
    # within-prompt: does the reward prefer the refusal when the same prompt has both?
    per = []
    for i in sorted(gen):
        rows = [(s, l) for s, l, p in zip(scores, labels, prompts) if p == pool[i]["prompt"]]
        r1 = [s for s, l in rows if l == 1]; r0 = [s for s, l in rows if l == 0]
        if r1 and r0:
            per.append(np.mean(r1) - np.mean(r0))
    out["within_prompt_gap_mean"] = float(np.mean(per)) if per else None
    out["within_prompt_prefers_refusal"] = float(np.mean([g > 0 for g in per])) if per else None
    out["prompts_with_both"] = len(per)
    json.dump(out, open(os.path.join(OUT, "rmcheck.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


def make_backend(steps, gen_batch, seed=0, tag="s0"):
    from seldonian.llm.backend import HFGRPOBackend
    return HFGRPOBackend(MODEL, os.path.join(RUN_DIR, tag), num_generations=8, prompts_per_step=8,
                         max_steps=steps, max_completion_length=128, beta=0.04, learning_rate=3e-5,
                         seed=seed, gen_batch_size=gen_batch, logging_steps=5)


def build_policy(a, backend, pool):
    from seldonian.llm.judges import build_judge
    from seldonian.llm.policy import Constraint, SeldonianLLMPolicy
    from seldonian.llm.rewards import LagrangianReward, SequenceClassifierReward
    ref_rate, n_ref = reference_rate()
    refusal = build_judge("qwen3guard_refusal", model_name="Qwen/Qwen3Guard-Gen-4B", quant4=True)
    c = Constraint("refusal", refusal, threshold=ref_rate + MARGIN, group="benign", bound="clopper_pearson")
    reward = LagrangianReward(SequenceClassifierReward(RM), [(refusal, "benign")], names=["refusal"],
                              lam0=5.0, eta=100.0, lam_max=50.0, lam_floor=5.0)
    d_c = training_records(pool, a.n_adv, a.n_ben)
    from seldonian.llm.data import make_record
    d_s = [dict(make_record(it["prompt"], "benign", "c1"), i=it["i"]) for it in pool]
    policy = SeldonianLLMPolicy(backend, d_c, d_s, reward=reward, constraints=[c], delta=DELTA,
                                predict_every=a.predict_every, predict_n=a.predict_n, seed=a.seed)
    print(f"reference refusal rate {ref_rate:.4f} on {n_ref} prompts -> threshold {c.threshold:.4f}; "
          f"D_c {len(d_c)} ({sum(r['group'] == 'benign' for r in d_c)} benign), D_s {len(d_s)}", flush=True)
    return policy, c, reward


def run_training(a, steps, ckpts, cand, tag):
    """Train under the Lagrangian; sample the pool at the checkpoints; run the safety test."""
    os.makedirs(OUT, exist_ok=True)
    pool = pool_c1()
    if a.limit:
        pool = pool[:a.limit]
    backend = make_backend(steps, a.gen_batch, a.seed, tag)
    policy, c, reward = build_policy(a, backend, pool)
    path = os.path.join(OUT, f"gen_{tag}.jsonl")
    pools = {"C1": pool}
    t0 = time.time()
    orig_train = backend.train

    def train_with_sampling(records, rw, on_step):
        def both(step):
            on_step(step)
            if step in ckpts:
                k = ckpts[step] or cand
                print(f"step {step}: sampling the pool x{k}, {time.time() - t0:.0f}s in", flush=True)
                gen013.generate_state(backend, pools, step, {"cand": k}, path)
        return orig_train(records, rw, both)

    backend.train = train_with_sampling
    solution = policy.fit(seldonian=True)
    hist = [dict(step=h.step, feasible=bool(h.feasible), rates=h.rates, upper=h.upper, reward=h.reward,
                 lambdas=getattr(h, "lambdas", None)) for h in policy.history]
    rep = policy.safety_report
    json.dump(dict(steps=steps, threshold=c.threshold, selected=policy.selected, solution=solution is not None,
                   safety_test=dict(g=rep.g, rates=rep.rates, upper=rep.upper, n=rep.n, passed=rep.passed,
                                    reward=rep.reward),
                   history=hist, train_log=backend.train_log, seconds=time.time() - t0),
              open(os.path.join(OUT, f"run_{tag}.json"), "w"), indent=1, default=float)
    recs, responses, _ = policy._safety_episodes
    with open(os.path.join(OUT, f"safety_{tag}.jsonl"), "w") as fh:
        for r, s in zip(recs, responses):
            fh.write(json.dumps(dict(i=r["i"], response=s)) + "\n")
    print(f"done: solution {solution is not None}, safety test {rep.rates} <= {c.threshold:.3f}: {rep.passed}, "
          f"{time.time() - t0:.0f}s", flush=True)


def stage_pilot(a):
    a.limit = 100
    run_training(a, 20, {20: 4}, 4, "pilot")


def stage_train(a):
    if not gate_open():
        sys.exit("gate.json does not say OPEN; stage 1 must open the gate first")
    run_training(a, a.steps, CKPT_STEPS, a.cand, f"s{a.seed}")


def stage_judge(a):
    import torch
    from seldonian.llm.backend import disable_triton_overrides_without_compiler
    from seldonian.llm.judges import build_judge
    disable_triton_overrides_without_compiler()
    tag = "pilot" if a.pilot else f"s{a.seed}"
    plain = {it["i"]: it["plain"] for it in pool_c1()}
    src = os.path.join(OUT, f"gen_{tag}.jsonl")
    dst = os.path.join(OUT, f"judged_{tag}.jsonl")
    done = gen013.done_keys(dst)
    rows = [r for r in map(json.loads, open(src)) if (r["pool"], r["step"], r["role"], r["i"]) not in done]
    judge = build_judge("qwen3guard_4b", cache_dir=None)
    judge._load()
    t0, n = time.time(), 0
    for c in range(0, len(rows), 32):
        part = rows[c:c + 32]
        pairs = [(plain[r["i"]], s) for r in part for s in r["responses"]]
        verdicts = []
        with torch.no_grad():
            for b in range(0, len(pairs), a.judge_batch):
                convs = [[{"role": "user", "content": p}, {"role": "assistant", "content": s}]
                         for p, s in pairs[b:b + a.judge_batch]]
                e = judge._tok.apply_chat_template(convs, return_tensors="pt", padding=True,
                                                   return_dict=True).to(judge.device)
                g = judge._model.generate(**e, max_new_tokens=32, do_sample=False,
                                          pad_token_id=judge._tok.pad_token_id)
                verdicts += [judge.parse(t) for t in judge._tok.batch_decode(
                    g[:, e["input_ids"].shape[1]:], skip_special_tokens=True)]
        with open(dst, "a") as fh:
            k = 0
            for r in part:
                v = verdicts[k:k + len(r["responses"])]
                k += len(r["responses"])
                fh.write(json.dumps(dict(pool=r["pool"], step=r["step"], role=r["role"], i=r["i"], meta=r["meta"],
                                         unsafe=[int(x.get("safety", "") == "unsafe") for x in v],
                                         refusal=[int(x.get("refusal", "").startswith("yes")) for x in v])) + "\n")
        n += len(pairs)
        print(f"judged {n} in {time.time() - t0:.0f}s", flush=True)
    print("DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["rmcheck", "pilot", "train", "judge"], required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--cand", type=int, default=16)
    ap.add_argument("--n-adv", type=int, default=600)
    ap.add_argument("--n-ben", type=int, default=600)
    ap.add_argument("--predict-every", type=int, default=25)
    ap.add_argument("--predict-n", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gen-batch", type=int, default=64)
    ap.add_argument("--judge-batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    {"rmcheck": stage_rmcheck, "pilot": stage_pilot, "train": stage_train, "judge": stage_judge}[a.stage](a)


if __name__ == "__main__":
    main()
