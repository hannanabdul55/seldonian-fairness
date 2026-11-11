"""Spike 022b, CPU checks of ``eslora.HFESBackend`` before any GPU time.

A random two-layer model with Granite's architecture and tokenizer stands in for Granite
(built once in ``--tiny``; nothing is downloaded beyond the cached tokenizer and config).

    ../../../.venv/bin/python check.py            # all checks, writes check.json

C1  a seed regenerates the same noise, bit for bit
C2  perturb and put back 200 times: the adapter in the model is unchanged
C3  log-probs read off the generation scores equal a teacher-forced forward pass, alone and
    in a padded batch; the KL term is 0 while the adapter is a no-op
C4  the ES estimate (raw fitness) points along the backprop gradient as theory says
C5  the whole Seldonian loop runs on the backend; on_step fires at the equivalent steps;
    a checkpoint round-trips; the (seeds, fitness) log alone rebuilds the trained adapter
C6  014's real adapter (fp32 and bf16 copies): what in-place perturb-and-restore loses at
    sigma 1e-3, the case the master copy is there to avoid
"""
import argparse
import json
import os
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eslora import HFESBackend  # noqa: E402

MODEL = "ibm-granite/granite-3.3-2b-instruct"
CKPT014 = "/mnt/d/seldonian-runs/014/s0/checkpoints/feasible-step100/adapter_model.safetensors"
RESULTS = {}


def tiny_model(path):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    if os.path.exists(os.path.join(path, "config.json")):
        return path
    cfg = AutoConfig.from_pretrained(MODEL)
    cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers = 64, 128, 2
    cfg.num_attention_heads, cfg.num_key_value_heads = 4, 2
    torch.manual_seed(0)
    AutoModelForCausalLM.from_config(cfg).save_pretrained(path)
    AutoTokenizer.from_pretrained(MODEL).save_pretrained(path)
    return path


def backend(path, out, **kw):
    base = dict(lora_r=2, lora_alpha=4, max_steps=4, max_completion_length=12, beta=0.04,
                bf16=False, device="cpu", pop=8, prompts_per_member=4, sigma=1e-2, es_lr=1e-2,
                gens_per_step=16, seed=0)
    base.update(kw)
    return HFESBackend(path, out, **base)


def randomise_b(be, scale=0.5, seed=1):
    """LoRA's B starts at zero (the adapter is a no-op); give it values so A matters too."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in be.params:
            if "lora_B" in n:
                p.copy_(scale * torch.randn(p.shape, generator=g))
    be.sync_master()


def record(name, ok, **info):
    RESULTS[name] = dict(ok=bool(ok), **info)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {json.dumps(info, default=float)}", flush=True)


PROMPTS = ["Name a colour.", "Write one sentence about the sea and about why sailors fear it.",
           "Count to three.", "What is a judge?"]


def c1_c2(be):
    a = [e.clone() for e in be.noise(123)]
    b = list(be.noise(123))
    c = list(be.noise(124))
    record("C1 noise replays from its seed", all(torch.equal(x, y) for x, y in zip(a, b))
           and not all(torch.equal(x, y) for x, y in zip(a, c)),
           tensors=len(a), n_params=be.n_params, adapter_dtype=str(be.params[0][1].dtype),
           base_dtype=str(next(be.model.base_model.model.parameters()).dtype))
    h0 = be.state_hash()
    be.set_member(5, 1.0)
    h1 = be.state_hash()
    be.restore()
    for i in range(200):
        be.set_member(i, 1.0 if i % 2 else -1.0)
        be.restore()
    be.set_member(5, 1.0)
    h1b = be.state_hash()
    be.restore()
    record("C2 perturb and put back, 200 times", h0 == be.state_hash() and h1 != h0 and h1 == h1b)


def c3(be, tag):
    texts, s = be.sample_scored(PROMPTS, 7)
    fwd = be.forward_logps(s)
    gap = float((fwd - s["logp"])[s["mask"]].abs().max())
    # each row alone, no padding: the padded batch must give the same numbers
    solo = 0.0
    for i, p in enumerate(PROMPTS):
        enc = be.tokenizer.apply_chat_template([be.messages(p)], add_generation_prompt=True,
                                               return_tensors="pt", return_dict=True)
        new = s["seq"][i:i + 1, s["P"]:]
        one = dict(seq=torch.cat([enc["input_ids"], new], 1), P=enc["input_ids"].shape[1],
                   attn=torch.ones(1, enc["input_ids"].shape[1] + new.shape[1], dtype=torch.long))
        solo = max(solo, float((be.forward_logps(one)[0] - s["logp"][i])[s["mask"][i]].abs().max()))
    with be.model.disable_adapter():
        ref = be.forward_logps(s)
    record(f"C3 generation scores = forward pass ({tag})", gap < 2e-4 and solo < 2e-4,
           max_gap=gap, max_gap_unpadded=solo, kl_term=be.kl_term(s, ref),
           tokens=int(s["mask"].sum()), sample=texts[0][:40])
    return be.kl_term(s, ref)


def c4(be, pairs=1500):
    """f = mean log-prob of a fixed continuation; ES with raw fitness against autograd."""
    _, s = be.sample_scored(PROMPTS, 3, want_logps=False)

    def f(grad=False):
        with torch.set_grad_enabled(grad):
            logits = be.model(input_ids=s["seq"], attention_mask=s["attn"]).logits[:, s["P"] - 1:-1]
            lp = torch.log_softmax(logits.float(), -1).gather(-1, s["seq"][:, s["P"]:, None]).squeeze(-1)
            return lp.mean()

    be.model.zero_grad()
    f(True).backward()
    true = torch.cat([p.grad.reshape(-1) for _, p in be.params]).double()
    be.rank, be.pop, sig = False, 2 * pairs, be.sigma
    be.sigma = 2e-2          # large enough that f+ - f- clears fp32 rounding in the forward pass
    seeds, fit = list(range(1000, 1000 + pairs)), []
    for sd in seeds:
        for sign in (1.0, -1.0):
            be.set_member(sd, sign)
            fit.append(float(f()))
    be.restore()
    est = torch.zeros_like(true)
    w = np.asarray(fit).reshape(-1, 2)
    for sd, c in zip(seeds, (w[:, 0] - w[:, 1]) / (2 * pairs * be.sigma)):
        est += c * torch.cat([e.reshape(-1) for e in be.noise(sd)]).double()
    cos = float(est @ true / (est.norm() * true.norm()))
    n = be.n_params
    expect = float(np.sqrt(pairs / (pairs + n + 1)))        # cosine of an N-sample Gaussian estimate
    scale = float(est.norm() / true.norm())
    be.sigma, be.rank = sig, True
    record("C4 ES estimate against backprop", abs(cos - expect) < 0.08 and cos > 0,
           cosine=cos, expected=expect, pairs=pairs, n_params=n, norm_ratio=scale,
           expected_norm_ratio=float(np.sqrt((pairs + n + 1) / pairs)))


class ShortReward:
    """Toy base reward: fewer characters is better (the model can learn to stop early)."""
    name = "toy:short"

    def __call__(self, prompts, responses, groups=None, references=None):
        return np.asarray([-len(r) / 50.0 for r in responses])


def c5(path, out):
    from seldonian.llm.data import make_record
    from seldonian.llm.judges import LengthJudge
    from seldonian.llm.policy import Constraint, SeldonianLLMPolicy
    from seldonian.llm.rewards import LagrangianReward
    log = os.path.join(out, "events.jsonl")
    be = backend(path, out, max_steps=20, pop=8, prompts_per_member=4, gens_per_step=16, log_path=log)
    randomise_b(be)
    start = [x.clone() for x in be.master]
    words = "sea judge colour number river stone window garden".split()
    recs = [make_record(f"Say something about the {w} ({i}).", "benign", "toy")
            for i, w in enumerate(words * 6)]
    judge = LengthJudge(cap=3, unit="words")
    reward = LagrangianReward(ShortReward(), [(judge, "benign")], names=["long"], lam0=1.0, eta=5.0)
    policy = SeldonianLLMPolicy(be, recs[:24], recs[24:], reward=reward, delta=0.1, predict_every=5,
                                predict_n=16, max_new_tokens=12, verbose=False,
                                constraints=[Constraint("long", judge, threshold=0.99, group="benign",
                                                        bound="clopper_pearson")])
    steps = []
    orig = be.train

    def spy(records, rw, on_step):
        return orig(records, rw, lambda k: (steps.append(k), on_step(k))[1])
    be.train = spy
    policy.fit(seldonian=True)
    trained = [x.clone() for x in be.master] if policy.selected["step"] == "final" else None
    logged = [json.loads(line) for line in open(log)]
    final = be.replay(logged, start)                     # from the file, not from memory
    # the live end-of-training master: rebuild it from the in-memory events as the reference
    live = be.replay(be.events, start)
    same = all(torch.equal(a, b) for a, b in zip(final, live))
    if trained is not None:
        same = same and all(torch.equal(a, b) for a, b in zip(final, trained))
    be.load_checkpoint(be.save_checkpoint("roundtrip"))
    h = be.state_hash()
    be.set_member(1, 1.0)
    be.load_checkpoint("roundtrip")
    lam = [e["lambdas"]["long"] for e in logged]
    r = [e["reward_mean"] for e in logged]
    record("C5 Seldonian loop, on_step, checkpoint, replay from the log",
           steps == list(range(1, 21)) and same and h == be.state_hash() and len(logged) == 10
           and be.gens == 320,
           on_steps=len(steps), generations=len(logged), completions=be.gens,
           predicted_tests=len(policy.history), selected=str(policy.selected["step"]),
           replay_bit_identical=bool(same), lambdas_first_last=[lam[0], lam[-1]],
           shaped_reward_first3_last3=[float(np.mean(r[:3])), float(np.mean(r[-3:]))],
           moved=float(sum(((a - b) ** 2).sum() for a, b in zip(final, start)) ** 0.5))


def c6():
    from safetensors.torch import load_file
    if not os.path.exists(CKPT014):
        record("C6 in-place perturbation on 014's adapter", False, error="checkpoint not found")
        return
    sd = load_file(CKPT014)
    x32 = torch.cat([v.float().reshape(-1) for v in sd.values()])
    g = torch.Generator().manual_seed(0)
    info = dict(tensors=len(sd), n_params=int(x32.numel()), saved_dtype=str(next(iter(sd.values())).dtype),
                mb_fp32=round(x32.numel() * 4 / 2 ** 20, 1), abs_median=float(x32.abs().median()))
    eps = torch.randn(x32.shape, generator=g)
    for sigma in (1e-3, 1e-2):
        for name, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
            x = x32.to(dt)
            s = (sigma * eps).to(dt)
            back = (x + s) - s
            real = ((x.float() + sigma * eps).to(dt).float() - x.float())      # what the model sees
            lost = float((real == 0).float().mean())
            corr = float(torch.corrcoef(torch.stack([real, sigma * eps]))[0, 1])
            info[f"{name}_sigma{sigma:g}"] = dict(not_restored=float((back != x).float().mean()),
                                                  perturbation_lost=lost, corr_with_intended=corr)
    record("C6 in-place perturbation on 014's adapter", True, **info)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", default=os.path.join(tempfile.gettempdir(), "022b-tiny-granite"))
    a = ap.parse_args()
    torch.set_num_threads(4)
    path = tiny_model(a.tiny)
    out = tempfile.mkdtemp(prefix="022b-check-")
    be = backend(path, out)
    kl0 = c3(be, "fresh adapter")
    record("C3b KL term is 0 while B = 0", abs(kl0) < 1e-6, kl_term=kl0)
    randomise_b(be)
    c1_c2(be)
    kl1 = c3(be, "B randomised")
    record("C3c KL term is positive once the adapter acts", kl1 > 1e-6, kl_term=kl1)
    c4(be)
    be16 = backend(path, tempfile.mkdtemp(prefix="022b-check-"), bf16=True)
    randomise_b(be16)
    record("C1b dtypes under the GPU loading path (bf16 base)", True,
           adapter_dtype=str(be16.params[0][1].dtype),
           base_dtype=str(next(be16.model.base_model.model.parameters()).dtype))
    c1_c2(be16)
    c5(path, tempfile.mkdtemp(prefix="022b-check-"))
    c6()
    json.dump(RESULTS, open(os.path.join(HERE, "check.json"), "w"), indent=1, default=float)
    bad = [k for k, v in RESULTS.items() if not v["ok"]]
    print(f"\n{len(RESULTS) - len(bad)} of {len(RESULTS)} checks passed" + (f"; FAILED: {bad}" if bad else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
