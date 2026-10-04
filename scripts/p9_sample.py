"""Fresh responses for the human-terms certificate (paper plan P9).

For each of the 490 safety prompts (spike 013's C1 pool minus the ten prompts quoted as examples in
the labelling guideline) and each of two policies, the reference (Granite-3.3-2B, untrained) and
spike 014's returned policy (its step-175 adapter): one response for the labelling sheet (role
``sheet``) and ``--extra`` more that only the guard reads (role ``extra``). Temperature 1.0, 256 new
tokens, through the training backend's own ``generate``, as 013 and 014 sampled. The reference is
the backend's freshly initialised adapter (its B matrices are zero, so it is the base model), which
is how 013 sampled it. Prompts already in the output are skipped, so a killed run resumes.

    scripts/run_p9_sample.sh [--extra 8]            -> results/labels/p9/gen.jsonl      (GPU)
    .venv/bin/python scripts/p9_sample.py --flat    -> results/labels/p9/guard_in.jsonl (CPU)

``--flat`` writes one row per response, ``{id, prompt, response}`` with id
``policy|role|i|k``, the input of ``guard_refusal_score.py``.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from refusal_sheet_build import EXAMPLE_PROMPTS  # noqa: E402

OUT = os.path.join(ROOT, "results", "labels", "p9")
MODEL = "ibm-granite/granite-3.3-2b-instruct"
ADAPTER = "/mnt/d/seldonian-runs/014/s0/checkpoints/feasible-step175"
RUN_DIR = "/mnt/d/seldonian-runs/p9"
MAX_NEW, TEMPERATURE, SEED = 256, 1.0, 2026
POLICIES = ("ref", "trained")


def pool():
    items = json.load(open(os.path.join(ROOT, "results", "spikes", "013", "pools.json")))["C1"]
    return [it for it in items if it["i"] not in EXAMPLE_PROMPTS]


def rows(path):
    """Rows of a jsonl file, or of its ``.xz`` copy (what the repo holds), or none."""
    if os.path.exists(path):
        return [json.loads(line) for line in open(path)]
    if os.path.exists(path + ".xz"):
        import lzma
        return [json.loads(line) for line in lzma.open(path + ".xz", "rt")]
    return []


def lora_b_norm(model):
    import torch
    return float(torch.sqrt(sum((p.detach().float() ** 2).sum() for n, p in model.named_parameters()
                                if "lora_B" in n and ".default" in n)))


def file_b_norm(adapter):
    import torch
    from safetensors.torch import load_file
    w = load_file(os.path.join(adapter, "adapter_model.safetensors"))
    return float(torch.sqrt(sum((v.float() ** 2).sum() for k, v in w.items() if "lora_B" in k)))


def generate(a):
    import torch
    from seldonian.llm.backend import HFGRPOBackend
    os.makedirs(OUT, exist_ok=True)
    items = pool()[:a.limit] if a.limit else pool()
    path = os.path.join(OUT, "gen.jsonl")
    done = {(r["policy"], r["role"], r["i"]) for r in rows(path)}
    backend = HFGRPOBackend(MODEL, RUN_DIR, max_completion_length=MAX_NEW, seed=SEED, gen_batch_size=a.gen_batch)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    t0, n_gen = time.time(), 0
    for policy in POLICIES:
        if policy == "trained":
            backend.load_checkpoint(ADAPTER)
            want = file_b_norm(ADAPTER)
        else:
            want = 0.0
        got = lora_b_norm(backend.model)
        print(f"{policy}: adapter B norm {got:.4f} (expected {want:.4f})", flush=True)
        if abs(got - want) > 1e-3 * max(want, 1.0):
            sys.exit(f"{policy}: the adapter in the model is not the one expected")
        for role, k in (("sheet", 1), ("extra", a.extra)):
            todo = [it for it in items if (policy, role, it["i"]) not in done]
            step = max(1, a.chunk // k)
            for c in range(0, len(todo), step):
                part = todo[c:c + step]
                resp = backend.generate([it["prompt"] for it in part for _ in range(k)],
                                        max_new_tokens=MAX_NEW, temperature=TEMPERATURE)
                ntok = [len(x) for x in backend.tokenizer(resp, add_special_tokens=False)["input_ids"]]
                with open(path, "a") as fh:
                    for j, it in enumerate(part):
                        fh.write(json.dumps(dict(policy=policy, role=role, i=it["i"], meta=it["meta"],
                                                 responses=resp[j * k:(j + 1) * k], ntok=ntok[j * k:(j + 1) * k])) + "\n")
                n_gen += len(resp)
                dt = time.time() - t0
                print(f"{policy} {role}: {c + len(part)} of {len(todo)} prompts x {k}; {n_gen} gens, {dt:.0f}s, "
                      f"{n_gen / max(dt, 1e-9):.2f} gen/s", flush=True)
    print("DONE", flush=True)


def flat(a):
    plain = {it["i"]: it["plain"] for it in pool()}
    src = rows(os.path.join(OUT, "gen.jsonl"))
    with open(os.path.join(OUT, "guard_in.jsonl"), "w") as fh:
        for r in src:
            for k, s in enumerate(r["responses"]):
                fh.write(json.dumps(dict(id=f"{r['policy']}|{r['role']}|{r['i']}|{k}", prompt=plain[r["i"]], response=s)) + "\n")
    print(sum(len(r["responses"]) for r in src), "responses ->", os.path.join(OUT, "guard_in.jsonl"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", type=int, default=8, help="guard-only responses per prompt and policy")
    ap.add_argument("--gen-batch", type=int, default=96)
    ap.add_argument("--chunk", type=int, default=288, help="responses per write")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--flat", action="store_true")
    a = ap.parse_args()
    (flat if a.flat else generate)(a)


if __name__ == "__main__":
    main()
