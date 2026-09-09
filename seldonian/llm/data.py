"""
Task loaders, prompt de-duplication and the candidate / safety split.

A *prompt record* is a dict::

    {"prompt_id": str, "prompt": str, "group": str, "reference": str | None, "task": str}

``group`` is what constraints filter on (``"adversarial"``, ``"benign"``, ``"math"``);
``reference`` carries a gold answer for verifiable tasks.

An *episode* adds what the policy produced on that prompt::

    {..., "response": str, "reward": float, "judge": {name: 0/1}}

Dataset ids live in module constants so a swap is a one-line change.
"""
import hashlib
import json
import re

import numpy as np

PKU_SAFE_RLHF = "PKU-Alignment/PKU-SafeRLHF"
XSTEST = ("Paul/XSTest", "train")  # ungated copy; label column is "safe"/"unsafe"
OR_BENCH = ("bench-llm/or-bench", "or-bench-80k")
GSM8K = ("openai/gsm8k", "main")


def _load_dataset(*args, **kwargs):
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover
        raise ImportError("datasets is required - install with `uv sync --extra rl`") from e
    return load_dataset(*args, **kwargs)


def _norm(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def _pid(prefix, text):
    return f"{prefix}:{hashlib.sha1(_norm(text).encode('utf-8')).hexdigest()[:12]}"


def make_record(prompt, group, task, reference=None):
    return {"prompt_id": _pid(task, prompt), "prompt": prompt, "group": group,
            "reference": reference, "task": task}


def dedupe(records):
    """Drop records whose whitespace/case-normalised prompt was already seen."""
    seen, out = set(), []
    for r in records:
        key = _norm(r["prompt"])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def subsample(records, n, seed):
    if n is None or n >= len(records):
        return list(records)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(records), size=n, replace=False)
    return [records[int(i)] for i in np.sort(idx)]


def split_prompts(records, test_size=0.4, seed=0):
    """
    Stratified (by ``group``) split into candidate and safety sets. Records are
    de-duplicated first so no normalised prompt appears on both sides.
    """
    records = dedupe(records)
    rng = np.random.default_rng(seed)
    groups = sorted({r["group"] for r in records})
    d_c, d_s = [], []
    for g in groups:
        members = [r for r in records if r["group"] == g]
        perm = rng.permutation(len(members))
        n_s = int(round(test_size * len(members)))
        s_idx = set(perm[:n_s].tolist())
        for i, r in enumerate(members):
            (d_s if i in s_idx else d_c).append(r)
    rng.shuffle(d_c)
    rng.shuffle(d_s)
    return d_c, d_s


# --------------------------------------------------------------------------------------
# task loaders
# --------------------------------------------------------------------------------------

def load_pku_prompts(n=None, seed=0, split="train"):
    ds = _load_dataset(PKU_SAFE_RLHF, split=split)
    recs = dedupe(make_record(p, "adversarial", "pku") for p in ds["prompt"])
    return subsample(recs, n, seed)


def load_benign_prompts(n=None, seed=0):
    """XSTest safe prompts (all of them) topped up with OR-Bench-80K."""
    recs = []
    xs = _load_dataset(XSTEST[0], split=XSTEST[1])
    for row in xs:
        if str(row.get("label", "")).lower() == "safe":
            recs.append(make_record(row["prompt"], "benign", "xstest"))
    if n is None or len(recs) < n:
        ob = _load_dataset(*OR_BENCH, split="train")
        extra = dedupe(make_record(p, "benign", "orbench") for p in ob["prompt"])
        need = None if n is None else n - len(recs)
        recs.extend(subsample(extra, need, seed))
    return dedupe(recs)


def load_gsm8k(n=None, seed=0, split="train"):
    ds = _load_dataset(*GSM8K, split=split)
    recs = []
    for row in ds:
        gold = row["answer"].split("####")[-1].strip()
        recs.append(make_record(row["question"], "math", "gsm8k", reference=gold))
    return subsample(dedupe(recs), n, seed)


BREVITY_SUFFIX = " Answer in at most 80 words."


def load_brevity_prompts(n=None, seed=0, suffix=BREVITY_SUFFIX):
    """Benign prompts with an explicit brevity instruction (group ``"benign"``)."""
    recs = []
    for r in load_benign_prompts(n, seed):
        recs.append(make_record(r["prompt"].rstrip() + suffix, "benign", "brevity"))
    return dedupe(recs)


def load_task(task, n, seed=0, benign_n=None):
    """
    ``ab``      -> PKU adversarial prompts (n) + benign prompts (benign_n), Tasks A and B
    ``gsm8k``   -> GSM8K train questions with gold answers, Task D
    ``brevity`` -> benign prompts (n) asking for an answer in at most 80 words
    """
    if task == "ab":
        return load_pku_prompts(n, seed) + load_benign_prompts(benign_n, seed)
    if task == "gsm8k":
        return load_gsm8k(n, seed)
    if task == "brevity":
        return load_brevity_prompts(n, seed)
    raise ValueError(f"unknown task {task!r}")


# --------------------------------------------------------------------------------------
# episodes on disk
# --------------------------------------------------------------------------------------

def make_episodes(records, responses, rewards=None, judge_labels=None):
    """Zip prompt records with what the policy produced on them."""
    eps = []
    for i, (rec, resp) in enumerate(zip(records, responses)):
        ep = dict(rec)
        ep["response"] = resp
        ep["reward"] = None if rewards is None else float(rewards[i])
        ep["judge"] = {} if judge_labels is None else {
            k: int(v[i]) for k, v in judge_labels.items()}
        eps.append(ep)
    return eps


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
