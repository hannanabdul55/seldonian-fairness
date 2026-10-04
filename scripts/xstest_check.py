"""An agreement number for the paper's single annotator (plan step P16).

The annotator labels 60 of XSTest's published completions (safe prompts only, 20 per published
class, models and labels hidden) on a copy of the Refusal Label Desk; their labels are compared
with XSTest's two-annotator labels. Our `h` (a refusal statement, then the content) is narrower
than XSTest's partial refusal (any mix of refusal and compliance), so agreement is reported on
the three classes and on each two-way cut.

    .venv/bin/python scripts/xstest_check.py build --page /path/xstest_check.html
    .venv/bin/python scripts/xstest_check.py analyze <store document .json>

The completions stay outside the repo (model owners' licences): the key written here holds ids
and published labels only, and the built page is not committed.
"""
import argparse
import collections
import csv
import glob
import json
import os
import random

SRC = "/mnt/d/seldonian-runs/p15/xstest"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "results", "labels", "xstest")
CLS = {"1_full_compliance": "a", "2_full_refusal": "r", "3_partial_refusal": "p"}
OURS = {"a": "a", "r": "r", "h": "p"}          # our label -> XSTest's class
# prompts the labelling guideline uses as worked examples (its answers would leak)
EXAMPLES = ("smash a pi", "blow up a balloon", "black olives", "princess mononoke")
PER_CLASS, SEED = 20, 2026


def kappa(a, b):
    cats = sorted(set(a) | set(b)); n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def build(a):
    rows = []
    for f in sorted(glob.glob(os.path.join(SRC, "*.csv"))):
        model = os.path.basename(f)[len("xstest_v2_completions_"):-4]
        for r in csv.DictReader(open(f, encoding="utf-8")):
            if r["type"].startswith("contrast_") or any(e in r["prompt"].lower() for e in EXAMPLES):
                continue
            rows.append(dict(model=model, pid=r["id"], type=r["type"], prompt=r["prompt"], response=r["completion"],
                             a1=CLS[r["annotation_1"]], a2=CLS[r["annotation_2"]], final=CLS[r["final_label"]]))
    rng = random.Random(SEED)
    pick = [x for c in "arp" for x in rng.sample([r for r in rows if r["final"] == c], PER_CLASS)]
    rng.shuffle(pick)
    page, key = [], []
    for n, r in enumerate(pick):
        page.append(dict(id=f"x{n:03d}", p=r["prompt"], r=r["response"], s=1, d=0))
        key.append(dict(id=f"x{n:03d}", **{k: r[k] for k in ("model", "pid", "type", "a1", "a2", "final")}))
    with open(os.path.join(OUT, "check60_key.jsonl"), "w") as fh:
        for k in key:
            fh.write(json.dumps(k) + "\n")
    t = open(os.path.join(ROOT, "results", "labels", "refusal", "desk_template.html")).read()
    for old, new in (("<title>Refusal Label Desk</title>", "<title>XSTest Agreement Check</title>"), ("<h1>Refusal Label Desk</h1>", "<h1>XSTest Agreement Check</h1>"),
                     ("Read the guideline once. Then enter your initials and your slot. Everyone labels the same 40 shared items first; the rest are split by slot so two people do not label the same item.",
                      "60 responses from other models to benign prompts. Label them exactly as you labelled the refusal sheet. Enter your initials and press Start; the slot and K fields do not matter here.")):
        assert t.count(old) == 1, old
        t = t.replace(old, new)
    open(a.page, "w").write(t.replace("__SHEET_JSON__", json.dumps(page).replace("</", "<\\/")))
    print(len(page), "items;", dict(collections.Counter(k["final"] for k in key)), "by model", dict(collections.Counter(k["model"] for k in key)), "->", a.page)


def analyze(a):
    key = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(OUT, "check60_key.jsonl"))}
    d = json.load(open(a.doc)); d = d.get("data", d) if "labels" not in d else d
    lab = {i: v["label"] for i, v in d["labels"].items() if i in key}
    ids = [i for i in key if lab.get(i) in OURS]
    mine = [OURS[lab[i]] for i in ids]
    L = ["# The annotator against XSTest's published labels", "",
         f"{len(lab)} of {len(key)} items labelled; {sum(v == 'u' for v in lab.values())} marked unsure and left out. "
         "Classes: a full compliance, r full refusal, p partial refusal (our refuse-then-answer is narrower than XSTest's partial refusal). "
         "The sample holds 20 of each published class, so raw agreement is on a balanced set, not on XSTest's mix.", "",
         "| compared with | raw agreement | kappa, three classes | kappa, full refusal vs the rest | kappa, compliance vs the rest |", "|---|---|---|---|---|"]
    for name, f in (("XSTest final label", "final"), ("XSTest annotator 1", "a1"), ("XSTest annotator 2", "a2")):
        th = [key[i][f] for i in ids]
        L.append(f"| {name} | {sum(x == y for x, y in zip(mine, th)) / len(ids):.2f} | {kappa(mine, th):.2f} | "
                 f"{kappa([x == 'r' for x in mine], [x == 'r' for x in th]):.2f} | {kappa([x == 'a' for x in mine], [x == 'a' for x in th]):.2f} |")
    t1, t2 = [key[i]["a1"] for i in ids], [key[i]["a2"] for i in ids]
    L.append(f"| (XSTest's two annotators with each other, same items) | {sum(x == y for x, y in zip(t1, t2)) / len(ids):.2f} | {kappa(t1, t2):.2f} | "
             f"{kappa([x == 'r' for x in t1], [x == 'r' for x in t2]):.2f} | {kappa([x == 'a' for x in t1], [x == 'a' for x in t2]):.2f} |")
    conf = collections.Counter((key[i]["final"], OURS[lab[i]]) for i in ids)
    L += ["", "Confusion (rows XSTest's final label, columns the annotator):", "", "| | a | r | p |", "|---|---|---|---|"]
    L += [f"| {c} | " + " | ".join(str(conf[(c, k)]) for k in "arp") + " |" for c in "arp"]
    open(os.path.join(OUT, "check60_analysis.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--page", required=True)
    an = sub.add_parser("analyze"); an.add_argument("doc")
    a = ap.parse_args()
    build(a) if a.cmd == "build" else analyze(a)
