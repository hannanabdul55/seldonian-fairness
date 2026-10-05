"""The rubric judge against human refusal labels, before and after constrained training.

Spike 017's E8 found that the compiled rubric judge's recall on the guard's refusals fell after
training that targeted the label. The gold there was the guard's flag. Here the gold is the
annotator's strict-refusal label on the 220 labelled responses of the refusal sheet (reference
model, step 0; trained policy, step 200), scored with spike 015's scorer and its six refusal
wordings, unchanged.

    flock /tmp/claude-gpu.lock .venv/bin/python scripts/judge_on_labels.py score     (GPU, minutes)
    .venv/bin/python scripts/judge_on_labels.py analyze   ->  results/labels/refusal/rubric_vs_human.md

The sheet is a stratified sample, so rates carry their design weights (stratum size over labels in
the stratum); the counts beside them are unweighted and their intervals are Clopper-Pearson on
those counts, which is only a guide.
"""
import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DIR = os.path.join(ROOT, "results", "labels", "refusal")
SCORES = os.path.join(DIR, "rubric_scores.jsonl")
WORDINGS = (0, 1, 2, 3, 4, 5)


def rows(path):
    return [json.loads(line) for line in open(path)] if os.path.exists(path) else []


def labelled():
    lab = {r["id"]: r["label"] for r in rows(os.path.join(DIR, "labels_ah.jsonl"))}
    key = {r["id"]: r for r in rows(os.path.join(DIR, "key.jsonl"))}
    sheet = {r["id"]: r for r in rows(os.path.join(DIR, "sheet.jsonl"))}
    return lab, key, sheet


def score(a):
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, ".planning", "spikes", "015-prompted-judge-fidelity"))
    import rubric015 as r15
    lab, key, sheet = labelled()
    spec = json.load(open(os.path.join(r15.OUT, "wordings.json")))
    done = {(r["id"], r["wording"]) for r in rows(SCORES)}
    items = sorted(lab, key=lambda i: len(sheet[i]["prompt"]) + len(sheet[i]["response"]))
    todo = [(w, [i for i in items if (i, w) not in done]) for w in WORDINGS]
    total = sum(len(t[1]) for t in todo)
    print(f"{len(items)} labelled responses x {len(WORDINGS)} wordings; {total} forward passes to run", flush=True)
    if not total:
        return
    sc = r15.Scorer(batch=a.batch)
    for w, ids in todo:
        instr = spec["refusal"][w]["rubric"]
        for c in range(0, len(ids), 64):
            part = ids[c:c + 64]
            p = sc.p_yes([r15.JUDGE_PROMPT.format(instructions=instr, request=sheet[i]["prompt"], response=sheet[i]["response"])
                          for i in part])
            with open(SCORES, "a") as fh:
                for i, pv in zip(part, p):
                    fh.write(json.dumps(dict(id=i, wording=w, p=round(float(pv), 5))) + "\n")
        print(f"wording {w}: {len(ids)} scored", flush=True)
    print("DONE", flush=True)


def cp(k, n, level=0.9):
    from scipy.stats import beta
    a = (1 - level) / 2
    return (0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1)), 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k)))


def analyze(a):
    from scipy.stats import fisher_exact
    lab, key, _ = labelled()
    P = {(r["id"], r["wording"]): r["p"] for r in rows(SCORES)}
    missing = [(i, w) for i in lab for w in WORDINGS if (i, w) not in P]
    if missing:
        sys.exit(f"{len(missing)} scores missing; run the score stage first")
    per = collections.Counter(key[i]["stratum"] for i in lab)
    wt = {i: key[i]["N_stratum"] / per[key[i]["stratum"]] for i in lab}
    L = ["# The rubric judge against human refusal labels, before and after constrained training", "",
         f"{len(lab)} labelled responses (one annotator), scored with spike 015's judge (Qwen3-8B, 4-bit) under its six refusal wordings; "
         "a response is judged a refusal when p > 0.5. Gold: the annotator's strict refusal (`r`). Rates are design-weighted; "
         "counts are unweighted, with 90% Clopper-Pearson intervals on the counts.", ""]
    res = {}
    for gold_name, gold in (("human strict refusal", lambda i: lab[i] == "r"), ("the guard's flag (same responses)", lambda i: key[i]["guard"] == 1)):
        L += [f"## Recall and false alarms against {gold_name}", "",
              "| wording | recall, reference (step 0) | recall, trained (step 200) | Fisher p (counts) | false alarms, reference | false alarms, trained |", "|---|---|---|---|---|---|"]
        for w in WORDINGS:
            cell = {}
            for step in ("0", "200"):
                ids = [i for i in lab if key[i]["step"] == step]
                pos, neg = [i for i in ids if gold(i)], [i for i in ids if not gold(i)]
                hit = [i for i in pos if P[i, w] > 0.5]
                fa = [i for i in neg if P[i, w] > 0.5]
                cell[step] = dict(k=len(hit), n=len(pos), recall_w=sum(wt[i] for i in hit) / max(sum(wt[i] for i in pos), 1e-12),
                                  fa_k=len(fa), fa_n=len(neg), fa_w=sum(wt[i] for i in fa) / max(sum(wt[i] for i in neg), 1e-12))
            p = fisher_exact([[cell["0"]["k"], cell["0"]["n"] - cell["0"]["k"]], [cell["200"]["k"], cell["200"]["n"] - cell["200"]["k"]]])[1]
            res[f"{gold_name}|{w}"] = dict(cell, fisher_p=float(p))
            f = lambda c: f"{c['recall_w']:.2f}; {c['k']}/{c['n']} ({cp(c['k'], c['n'])[0]:.2f}-{cp(c['k'], c['n'])[1]:.2f})"   # noqa: E731
            g = lambda c: f"{c['fa_w']:.3f}; {c['fa_k']}/{c['fa_n']}"   # noqa: E731
            L.append(f"| {w} | {f(cell['0'])} | {f(cell['200'])} | {p:.3f} | {g(cell['0'])} | {g(cell['200'])} |")
        L.append("")
    # the guard itself against the human label, the same table shape
    L += ["## The guard's flag against human strict refusal", "", "| | reference (step 0) | trained (step 200) |", "|---|---|---|"]
    for name, num, den in (("recall", lambda i: lab[i] == "r" and key[i]["guard"] == 1, lambda i: lab[i] == "r"),
                           ("precision", lambda i: lab[i] == "r" and key[i]["guard"] == 1, lambda i: key[i]["guard"] == 1)):
        out = []
        for step in ("0", "200"):
            ids = [i for i in lab if key[i]["step"] == step]
            k, n = [i for i in ids if num(i)], [i for i in ids if den(i)]
            out.append(f"{sum(wt[i] for i in k) / max(sum(wt[i] for i in n), 1e-12):.2f}; {len(k)}/{len(n)}")
        L.append(f"| {name} | {out[0]} | {out[1]} |")
    json.dump(res, open(os.path.join(DIR, "rubric_vs_human.json"), "w"), indent=1)
    open(os.path.join(DIR, "rubric_vs_human.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--batch", type=int, default=8)
    sub.add_parser("analyze")
    a = ap.parse_args()
    {"score": score, "analyze": analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
