"""Spike: does a larger guard read the attacked responses the same way?

Every red-team rate is a Qwen3Guard-Gen-0.6B rate. Reading the episodes showed
three artefacts of that judge: it flags harmless replies to jailbreak templates,
flags gibberish answers to harmful questions when scored against the question,
and flags ordinary answers to many-shot prompts as refusals. This spike re-judges
a sample of the attacked episodes with Qwen3Guard-Gen-4B, under both prompt
readings, and compares.

    uv run scripts/spike_big_judge.py judge       # GPU, resumable
    uv run scripts/spike_big_judge.py analyze     # CPU; uses human labels if present

The sample: 100 random episodes per technique on the rows the report's
conclusions rest on, plus every item of the human labelling sheet, so the two
judges can be scored against the human labels once they exist.
"""
import argparse
import collections
import json
import os
import random
import sys

OUT = "results/spikes/big_judge"
ROOT = "results/redteam"
MODEL = "Qwen/Qwen3Guard-Gen-4B"
TECHNIQUES = {
    ("c0_harm", "harm"): ["baseline", "url", "ansi_attack", "many_shot", "jailbreak", "diacritic",
                          "resample_16", "best_of_n"],
    ("c0_llm", "harm"): ["pair", "tap"],
    ("c0_refusal", "refusal"): ["baseline", "base64", "morse", "jailbreak", "many_shot",
                                "ansi_attack", "url"],
}
PER_TECHNIQUE = 100


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def small_labels(e):
    def pick(suffix):
        v = [x for k, x in (e.get("scores") or {}).items() if k.endswith(suffix)]
        return None if not v or v[0] is None else int(bool(v[0]))
    return pick(":conversation"), pick(":objective")


def last_user_turn(e):
    users = [t for t in e.get("turns") or [] if t.get("role") == "user"]
    if not users:
        return e.get("objective", "")
    return users[-1].get("converted") or users[-1].get("original") or ""


def build_items(seed=0):
    """One item per episode: battery, technique, both prompts, response, 0.6B labels."""
    sheet_ids = {}
    key_path = os.path.join(ROOT, "labels", "key.jsonl")
    if os.path.exists(key_path):
        for r in read_jsonl(key_path):
            sheet_ids[(r["battery"], r["conversation_id"])] = r["id"]
    items = []
    rng = random.Random(seed)
    for (battery, task), techs in TECHNIQUES.items():
        by_tech = collections.defaultdict(list)
        extra = []
        with open(os.path.join(ROOT, battery, "episodes.jsonl")) as f:
            for ln in f:
                e = json.loads(ln)
                if None in small_labels(e):
                    continue
                if e["technique"] in techs:
                    by_tech[e["technique"]].append(e)
                elif (battery, e["conversation_id"]) in sheet_ids:
                    extra.append(e)
        chosen, random_ids = [], set()
        for t in techs:
            eps = by_tech[t]
            pick = rng.sample(eps, min(PER_TECHNIQUE, len(eps)))
            ids = {e["conversation_id"] for e in pick}
            random_ids |= ids
            # every sheet item of this technique too (not part of the random sample)
            pick += [e for e in eps if (battery, e["conversation_id"]) in sheet_ids
                     and e["conversation_id"] not in ids]
            chosen += pick
        chosen += extra
        for e in chosen:
            c, o = small_labels(e)
            items.append({"battery": battery, "task": task, "technique": e["technique"],
                          "conversation_id": e["conversation_id"],
                          "sheet_id": sheet_ids.get((battery, e["conversation_id"])),
                          "in_random_sample": e["conversation_id"] in random_ids,
                          "question": e["objective"], "prompt_seen": last_user_turn(e),
                          "response": e.get("last_response") or "",
                          "small_primary": c, "small_question": o})
    # dedupe (a sheet item can also be in the random sample)
    seen, out = set(), []
    for it in items:
        k = (it["battery"], it["conversation_id"])
        if k not in seen:
            seen.add(k)
            out.append(it)
    return out


def judge(args):
    from seldonian.llm.judges import build_judge

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "labels.jsonl")
    done = {(r["battery"], r["conversation_id"]) for r in read_jsonl(path)} \
        if os.path.exists(path) else set()
    items = [it for it in build_items(args.seed) if (it["battery"], it["conversation_id"]) not in done]
    print(f"{len(done)} done, {len(items)} to judge with {args.model}", flush=True)
    judges = {
        "harm": build_judge("qwen3guard", cache_dir=args.cache_dir, model_name=args.model,
                            batch_size=args.batch),
        "refusal": build_judge("qwen3guard_refusal", cache_dir=args.cache_dir,
                               model_name=args.model, batch_size=args.batch),
    }
    # judge in length order so long prompts (many-shot, 27k characters) share batches
    # with each other; a per-call character budget keeps the 4B model in memory
    for task in ("harm", "refusal"):
        todo = [it for it in items if it["task"] == task]
        pairs = []
        for it in todo:
            pairs.append((it, "primary", it["prompt_seen"]))
            pairs.append((it, "question", it["question"]))
        pairs.sort(key=lambda x: len(x[2]) + len(x[0]["response"]))
        results = {}
        i = 0
        while i < len(pairs):
            budget, j = 0, i
            while j < len(pairs) and (j == i or budget + len(pairs[j][2]) + len(pairs[j][0]["response"])
                                      <= args.max_chars) and j - i < args.batch:
                budget += len(pairs[j][2]) + len(pairs[j][0]["response"])
                j += 1
            chunk = pairs[i:j]
            labs = judges[task]([p for _, _, p in chunk], [it["response"] for it, _, _ in chunk])
            for (it, which, _), v in zip(chunk, labs):
                results.setdefault((it["battery"], it["conversation_id"]), {})[which] = int(v)
            if i // 500 != j // 500:
                print(f"{task}: {j}/{len(pairs)} judge calls", flush=True)
            i = j
        with open(path, "a") as f:
            for it in todo:
                r = results[(it["battery"], it["conversation_id"])]
                f.write(json.dumps({k: it[k] for k in ("battery", "task", "technique",
                                                       "conversation_id", "sheet_id",
                                                       "in_random_sample", "small_primary",
                                                       "small_question")}
                                   | {"big_primary": r["primary"], "big_question": r["question"]})
                        + "\n")
        print(f"{task}: wrote {len(todo)} items", flush=True)


def analyze(args):
    rows = read_jsonl(os.path.join(args.out, "labels.jsonl"))
    lines = ["# Spike: Qwen3Guard-Gen-0.6B against Qwen3Guard-Gen-4B on attacked episodes", "",
             f"{PER_TECHNIQUE} random episodes per technique (the rates below are on those), "
             "judged by both guards under both prompt readings: *primary* = the prompt the "
             "policy received, *question* = the original request.", ""]
    by = collections.defaultdict(list)
    for r in rows:
        if r["in_random_sample"]:
            by[(r["battery"], r["technique"])].append(r)
    lines += ["| battery / technique | n | 0.6B primary | 4B primary | 0.6B question | "
              "4B question | primary agree | 4B primary=1 & 0.6B=0 | 0.6B=1 & 4B=0 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for (bat, tech), rs in sorted(by.items()):
        n = len(rs)
        m = lambda k: sum(r[k] for r in rs) / n  # noqa: E731
        agree = sum(r["small_primary"] == r["big_primary"] for r in rs) / n
        up = sum(r["big_primary"] and not r["small_primary"] for r in rs)
        down = sum(r["small_primary"] and not r["big_primary"] for r in rs)
        lines.append(f"| {bat}/{tech} | {n} | {m('small_primary'):.2f} | {m('big_primary'):.2f} | "
                     f"{m('small_question'):.2f} | {m('big_question'):.2f} | {agree:.2f} | {up} | {down} |")

    # scored against human labels, if any
    lab_dir = os.path.join(ROOT, "labels")
    files = sorted(f for f in os.listdir(lab_dir) if f.startswith("labels_") and f.endswith(".jsonl")) \
        if os.path.isdir(lab_dir) else []
    if files:
        sys.path.insert(0, os.path.dirname(__file__))
        from redteam_labels import TASKS
        human = {}
        for f in files:
            for r in read_jsonl(os.path.join(lab_dir, f)):
                if r.get("label") not in (None, "?"):
                    human.setdefault(r["id"], r)
        scored = [r for r in rows if r.get("sheet_id") in human]
        if scored:
            lines += ["", f"## Against the human labels ({len(scored)} items)", "",
                      "Accuracy of each judge reading against the constraint's own human "
                      "reading (harm: complies; refusal: unanswered). The sheet over-represents "
                      "disagreements, so compare judges, not levels.", "",
                      "| constraint | n | 0.6B primary | 4B primary | 0.6B question | 4B question |",
                      "|---|---|---|---|---|---|"]
            for task in ("harm", "refusal"):
                rs = [r for r in scored if r["task"] == task]
                if not rs:
                    continue
                ev = next(iter(TASKS[task]["events"].values()))
                y = [int(human[r["sheet_id"]]["label"] in ev) for r in rs]
                acc = lambda k: sum(int(r[k]) == t for r, t in zip(rs, y)) / len(rs)  # noqa: E731
                lines.append(f"| {task} | {len(rs)} | {acc('small_primary'):.2f} | "
                             f"{acc('big_primary'):.2f} | {acc('small_question'):.2f} | "
                             f"{acc('big_question'):.2f} |")
    text = "\n".join(lines) + "\n"
    print(text)
    with open(os.path.join(args.out, "analysis.md"), "w") as f:
        f.write(text)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("judge")
    j.add_argument("--model", default=MODEL)
    j.add_argument("--batch", type=int, default=16)
    j.add_argument("--max-chars", type=int, default=40_000)
    j.add_argument("--seed", type=int, default=0)
    j.add_argument("--cache-dir", default=".cache/judges")
    j.add_argument("--out", default=OUT)
    a = sub.add_parser("analyze")
    a.add_argument("--out", default=OUT)
    args = p.parse_args()
    {"judge": judge, "analyze": analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
