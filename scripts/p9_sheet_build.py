"""Build the P9 labelling sheet: 300 prompt pairs from the two policies (paper plan P9).

The 300 prompts are a simple random sample without replacement (seed 2026) of the 490 safety prompts
that are not guideline examples; the draw uses the prompt ids only, so it does not depend on any
response. Each sampled prompt contributes the reference's and the trained policy's ``sheet``
response from ``p9_sample.py``: 600 items, shuffled, with blind ids. The key holds the policy.

Outputs in ``results/labels/p9/``: ``sheet.jsonl`` (blind), ``key.jsonl`` (hidden fields),
``design.json``. ``--page`` also renders the labelling page from the refusal desk's template.

    .venv/bin/python scripts/p9_sheet_build.py [--page /path/p9_desk.html]
"""
import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from p9_sample import ADAPTER, MAX_NEW, MODEL, OUT, POLICIES, TEMPERATURE, pool, rows  # noqa: E402
from refusal_sheet_build import EXAMPLE_PROMPTS, TERMINAL  # noqa: E402

SEED, PAIRS = 2026, 300


def sampled_prompts():
    """The 300 prompt ids, a function of the seed and the pool only."""
    return sorted(random.Random(SEED).sample(sorted(it["i"] for it in pool()), PAIRS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", default=None, help="write the labelling page (template + sheet) here")
    a = ap.parse_args()
    P = {it["i"]: it for it in pool()}
    G = {(r["policy"], r["i"]): r for r in rows(os.path.join(OUT, "gen.jsonl")) if r["role"] == "sheet"}
    missing = [(p, i) for p in POLICIES for i in P if (p, i) not in G]
    if missing:
        sys.exit(f"{len(missing)} sheet responses are missing from gen.jsonl; finish the sampling first")
    picked = sampled_prompts()
    items = [dict(i=i, policy=p, text=G[p, i]["responses"][0], ntok=G[p, i]["ntok"][0]) for i in picked for p in POLICIES]
    random.Random(SEED + 1).shuffle(items)            # blind ids and file order leak nothing
    sheet, key, page = [], [], []
    for n, x in enumerate(items):
        iid = f"p{n:03d}"
        sheet.append(dict(id=iid, prompt=P[x["i"]]["plain"], response=x["text"]))
        page.append(dict(id=iid, p=P[x["i"]]["plain"], r=x["text"], s=1, d=0))
        key.append(dict(id=iid, i=x["i"], policy=x["policy"], meta=P[x["i"]]["meta"], ntok=x["ntok"],
                        cut_off=int(x["ntok"] >= MAX_NEW and x["text"].rstrip()[-1:] not in TERMINAL)))
    for name, data in (("sheet.jsonl", sheet), ("key.jsonl", key)):
        with open(os.path.join(OUT, name), "w") as fh:
            for r in data:
                fh.write(json.dumps(r) + "\n")
    json.dump(dict(seed=SEED, pairs=PAIRS, items=len(sheet), prompts_in_frame=len(P), excluded_prompts=list(EXAMPLE_PROMPTS),
                   frame="013 C1 pool minus the guideline's example prompts; one fresh response per policy per prompt",
                   policies=dict(ref=MODEL, trained=ADAPTER), max_new_tokens=MAX_NEW, temperature=TEMPERATURE,
                   sampled_prompts=picked, cut_off=sum(k["cut_off"] for k in key),
                   builder="scripts/p9_sheet_build.py", sampler="scripts/p9_sample.py"),
              open(os.path.join(OUT, "design.json"), "w"), indent=1)
    print(f"{len(sheet)} items from {len(picked)} prompts of {len(P)}; cut off at {MAX_NEW} tokens: {sum(k['cut_off'] for k in key)}")
    if a.page:
        t = open(os.path.join(ROOT, "results", "labels", "refusal", "desk_template.html")).read()
        for old, new in (("<title>Refusal Label Desk</title>", "<title>Policy Pair Sheet</title>"),
                         ("<h1>Refusal Label Desk</h1>", "<h1>Policy Pair Sheet</h1>"),
                         ("Read the guideline once. Then enter your initials and your slot. Everyone labels the same 40 shared items first; the rest are split by slot so two people do not label the same item.",
                          "600 responses to benign prompts, from two versions of one model, in a mixed order. Label them exactly as you labelled the refusal sheet. Enter your initials and press Start; the slot and K fields do not matter here. You can stop and come back: your place and labels are kept.")):
            assert t.count(old) == 1, old
            t = t.replace(old, new)
        open(a.page, "w").write(t.replace("__SHEET_JSON__", json.dumps(page).replace("</", "<\\/")))
        print("page ->", a.page, f"({os.path.getsize(a.page)} bytes)")


if __name__ == "__main__":
    main()
