"""Spike 017, GPU stage: the compiled judge's scores on the populations PPI needs.

015 scored only the labelled items. PPI's first term is the judge's mean over the whole
scored population, so this pass scores (with 015's scorer and rubrics, unchanged):

- ``harm_pop``     every harmful-prompt response of the capability screen (the 4,800 the
                   human sheet was sampled from), harm wordings 0, 2 and 4;
- ``ref_c1_200``   013's over-refusal pool after 200 training steps, the same (prompt, k)
                   pairs 015 scored at step 0, six refusal wordings;
- ``ref_c3_0``     013's plain harmful prompts at step 0 (refusal rate 0.67), 500 responses,
                   six refusal wordings.

No rubric text is written here: everything comes from results/spikes/015/wordings.json.
Rows are appended in chunks and the run resumes.

    ./run.sh                       # about 40 min GPU
    ./run.sh --sets ref_c1_200     # one set
"""
import argparse
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(HERE, "..", "015-prompted-judge-fidelity"))

import rubric015 as r15  # noqa: E402

OUT = os.path.join(REPO, "results", "spikes", "017")
SCREEN = os.path.join(REPO, "results", "screen")
HARM_WORDINGS = (0, 2, 4)
REF_WORDINGS = (0, 1, 2, 3, 4, 5)
CHUNK = 256


def harm_pop():
    import screen_labels as sl
    models = json.load(open(os.path.join(SCREEN, "labels", "shares.json")))["models"]
    rows = []
    for m in models:
        for e in sl.episodes(SCREEN, m):
            rows.append(dict(id=e["eid"], request=e["question"], response=e["response"],
                             model=m, enc=e["enc"], unsafe06=e["unsafe"],
                             not_refused=e["not_refused"]))
    return rows


def pool013(pool, step, n=500, ids=None, seed=0):
    pools = json.load(open(os.path.join(r15.GEN013, "pools.json")))
    prompts = {r["i"]: r["plain"] for r in pools[pool]}
    sel = lambda r: r["pool"] == pool and r["step"] == step and r["role"] == "cand"  # noqa: E731
    gen = {r["i"]: r for r in r15.read_jsonl(os.path.join(r15.GEN013, "gen_full.jsonl")) if sel(r)}
    jud = {r["i"]: r for r in r15.read_jsonl(os.path.join(r15.GEN013, "judged_full.jsonl")) if sel(r)}
    rows = []
    for i in sorted(gen):
        for k in (0, 1):
            resp = gen[i]["responses"][k]
            rows.append(dict(id=f"{pool.lower()}-{i}-{k}", request=prompts[i], response=resp,
                             refusal=int(jud[i]["refusal"][k]), unsafe=int(jud[i]["unsafe"][k]),
                             words=len(resp.split()), meta=gen[i]["meta"]))
    if ids is not None:
        keep = set(ids)
        return [r for r in rows if r["id"] in keep]
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:n]


def sets():
    ids015 = sorted({r["id"] for r in r15.read_jsonl(os.path.join(r15.OUT, "scores.jsonl"))
                     if r["task"] == "refusal"})
    return {
        "ref_c1_200": ("refusal", REF_WORDINGS, lambda: pool013("C1", 200, ids=ids015)),
        "ref_c3_0": ("refusal", REF_WORDINGS, lambda: pool013("C3", 0)),
        "harm_pop": ("harm", HARM_WORDINGS, harm_pop),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="ref_c1_200,ref_c3_0,harm_pop")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N items per set")
    ap.add_argument("--out", default="scores_pop.jsonl")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, a.out)
    done = {(r["set"], r["wording"], r["id"]) for r in r15.read_jsonl(path)}
    spec = json.load(open(os.path.join(r15.OUT, "wordings.json")))
    todo = []
    for name in a.sets.split(","):
        task, wordings, loader = sets()[name]
        rows = loader()
        if a.limit:
            rows = rows[:a.limit]
        for w in wordings:
            pend = [r for r in rows if (name, w, r["id"]) not in done]
            if pend:
                todo.append((name, task, w, pend))
        print(f"{name}: {len(rows)} items x {len(wordings)} wordings", flush=True)
    total = sum(len(t[3]) for t in todo)
    print(f"to score: {total} forward passes", flush=True)
    if not total:
        print("DONE", flush=True)
        return
    sc = r15.Scorer(batch=a.batch)
    t0, n_done = time.time(), 0
    for name, task, w, pend in todo:
        instr = spec[task][w]["rubric"]
        pend = sorted(pend, key=lambda r: len(r["request"]) + len(r["response"]))
        for c in range(0, len(pend), CHUNK):
            chunk = pend[c:c + CHUNK]
            texts = [r15.JUDGE_PROMPT.format(instructions=instr, request=r["request"],
                                             response=r["response"]) for r in chunk]
            p = sc.p_yes(texts)
            with open(path, "a") as fh:
                for r, pv in zip(chunk, p):
                    row = {k: v for k, v in r.items() if k not in ("request", "response")}
                    row.update(set=name, wording=w, p=round(pv, 5))
                    fh.write(json.dumps(row) + "\n")
            n_done += len(chunk)
            el = time.time() - t0
            print(f"{name} w{w}: {c + len(chunk)}/{len(pend)}  total {n_done}/{total}  "
                  f"{el:.0f}s  eta {el / n_done * (total - n_done):.0f}s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
