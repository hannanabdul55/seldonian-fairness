"""Blind labelling sheet for the capability screen's harmful-prompt responses (no GPU).

Spike 005 (``.planning/spikes/005-capability-screen``) found that Qwen3Guard-0.6B
flags encoded harmful prompts as Unsafe whenever the response does not refuse, even
when the model could not read the prompt and answered with a hallucinated
plaintext. This script draws a blind sheet from ``results/screen/<model>/`` in the
format of ``scripts/redteam_labels.py``, whose ``label`` and ``analyze`` commands
then run on it unchanged. The battery is the model and the technique is the encoding
(``plain`` for the unencoded prompts).

In the key, ``judge_primary`` is the guard's Unsafe flag and ``judge_question`` is
*not refused* (1 - the guard's Refusal flag). The analysis's agreement table then
shows directly whether the Unsafe flag tracks human-labelled compliance or just
non-refusal.

    uv run scripts/screen_labels.py sample                       # draw results/screen/labels/
    uv run scripts/redteam_labels.py label --out results/screen/labels --annotator ah --tier 1
    uv run scripts/redteam_labels.py analyze --out results/screen/labels

Guideline: results/screen/labels/GUIDELINE.md (the rules are the harm-sheet rules of
results/redteam/labels/GUIDELINE.md, including its section on encoded prompts).
"""
import argparse
import collections
import hashlib
import json
import os
import random
import sys

ROOT = "results/screen"
OUT = "results/screen/labels"
ENCODINGS = ("leetspeak", "reverse", "caesar3", "rot13", "atbash", "base64")
#: 0.5B is ruled out for the pilot, and its responses are the guideline's examples
DEFAULT_EXCLUDE = ("0.5B",)
#: per (model, encoding): (tier, judge Unsafe flag, count); plain prompts use PLAIN
PLAN = [(1, 1, 4), (1, 0, 2), (2, 1, 3), (2, 0, 2)]
PLAIN = [(1, 1, 3), (1, 0, 3)]


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def write_jsonl(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _h(text):
    return hashlib.sha1(text.encode()).hexdigest()[:10]


def episodes(root, model):
    """One episode per judged response: (encoding, question, prompt seen, response, flags)."""
    d = os.path.join(root, model)
    gen = {(r["kind"], r["enc"], r["prompt"]): r for r in read_jsonl(os.path.join(d, "gen.jsonl"))
           if r["kind"] in ("f_enc", "f_plain")}
    out = []
    for j in read_jsonl(os.path.join(d, "judged.jsonl")):
        g = gen.get((j["kind"], j["enc"], j["prompt"]))
        if g is None:
            continue
        for k, (resp, u, rf) in enumerate(zip(g["responses"], j["unsafe"], j["refusal"])):
            out.append(dict(enc="plain" if j["kind"] == "f_plain" else j["enc"],
                            question=g["plain"], prompt_seen=g["prompt"], response=resp,
                            unsafe=int(u), not_refused=int(1 - rf),
                            eid=f"{model}|{j['kind']}|{j['enc']}|{_h(g['prompt'])}|{k}"))
    return out


def sample(args):
    os.makedirs(args.out, exist_ok=True)
    sheet_path = os.path.join(args.out, "sheet.jsonl")
    if os.path.exists(sheet_path) and not args.force:
        sys.exit(f"{sheet_path} exists; labels already made against it would no longer match. "
                 "Pass --force to redraw.")
    models = args.models or sorted(
        m for m in os.listdir(args.root)
        if os.path.exists(os.path.join(args.root, m, "judged.jsonl")) and m not in args.exclude)
    rng = random.Random(args.seed)
    rows, shares = [], {}
    for m in models:
        eps = episodes(args.root, m)
        by_enc = collections.defaultdict(list)
        for e in eps:
            by_enc[e["enc"]].append(e)
        for enc, lst in sorted(by_enc.items()):
            combos = collections.Counter((e["unsafe"], e["not_refused"]) for e in lst)
            shares[f"{m}/{enc}"] = {f"{a}{b}": c / len(lst) for (a, b), c in combos.items()}
            used = set()
            for tier, flag, count in (PLAIN if enc == "plain" else PLAN):
                pool = [e for e in lst if e["unsafe"] == flag and e["eid"] not in used]
                pick = rng.sample(pool, min(count, len(pool)))
                for e in pick:
                    used.add(e["eid"])
                    rows.append((tier, m, enc, e))
    rng.shuffle(rows)
    rows.sort(key=lambda r: r[0])
    sheet, key = [], []
    for i, (tier, m, enc, e) in enumerate(rows):
        rid = f"s{tier}-{i:03d}"
        sheet.append({"id": rid, "task": "harm", "tier": tier, "question": e["question"],
                      "prompt_seen": e["prompt_seen"], "response": e["response"]})
        key.append({"id": rid, "battery": m, "technique": enc, "judge_primary": e["unsafe"],
                    "judge_question": e["not_refused"], "conversation_id": e["eid"]})
    write_jsonl(sheet_path, sheet)
    write_jsonl(os.path.join(args.out, "key.jsonl"), key)
    with open(os.path.join(args.out, "shares.json"), "w") as f:
        json.dump({"note": "share of each model/encoding's responses by (guard Unsafe, not "
                           "refused); the weights of the corrected rate", "seed": args.seed,
                   "models": models, "shares": shares}, f, indent=2)
    with open(os.path.join(args.out, "sheet.md"), "w") as f:
        f.write("# Screen labelling sheet (reading copy)\n\nLabel with `uv run "
                "scripts/redteam_labels.py label --out results/screen/labels`; see GUIDELINE.md.\n\n")
        for r in sheet:
            f.write(f"## {r['id']} (tier {r['tier']})\n\n**Question:** {r['question']}\n\n"
                    f"**What the model received:**\n\n{r['prompt_seen']}\n\n"
                    f"**Response:**\n\n{r['response']}\n\n---\n\n")
    by = collections.Counter(r["tier"] for r in sheet)
    print(f"{len(sheet)} responses from {len(models)} models ({', '.join(models)}) -> {sheet_path}")
    for tier, n in sorted(by.items()):
        print(f"  tier {tier}: {n}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="draw the blind sheet and its key")
    s.add_argument("--root", default=ROOT)
    s.add_argument("--out", default=OUT)
    s.add_argument("--models", nargs="+", default=None,
                   help="model folders under --root (default: every judged one but 0.5B)")
    s.add_argument("--exclude", nargs="+", default=list(DEFAULT_EXCLUDE))
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--force", action="store_true", help="overwrite an existing sheet")
    args = p.parse_args()
    {"sample": sample}[args.cmd](args)


if __name__ == "__main__":
    main()
