"""The guard against published human labels (paper plan P15).

XSTest (Röttger et al., NAACL 2024; github.com/paul-rottger/exaggerated-safety, CC-BY-4.0) ships
completions from five model variants on its 450 prompts (250 safe, 200 unsafe contrasts), each
with two human annotations and a final label: full compliance, full refusal, partial refusal.
``guard_refusal_score.py`` scored every completion with Qwen3Guard-4B. This script compares the
guard's refusal flag and logit with the human labels, on models this project did not train.

The completions stay outside the repo (they are under the model owners' licences); the output
keeps ids, human labels and guard scores only.

    .venv/bin/python scripts/xstest_guard.py
    -> results/labels/xstest/analysis.md, guard_scores.jsonl
"""
import collections
import csv
import glob
import json
import os
import re
import sys

import numpy as np
from scipy.stats import beta

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from refusal_sheet_build import HYB, HYB_MIN_WORDS  # noqa: E402

SRC = "/mnt/d/seldonian-runs/p15"
OUT = os.path.join(HERE, "..", "results", "labels", "xstest")
MODELS = {"gpt4": "GPT-4", "llama2orig": "Llama 2, original system prompt", "llama2new": "Llama 2, no system prompt",
          "mistralinstruct": "Mistral instruct", "mistralguard": "Mistral with guardrail prompt"}
CLS = {"1_full_compliance": "a", "2_full_refusal": "r", "3_partial_refusal": "p"}
EVENTS = (("full refusal (strict)", lambda l: l == "r"), ("full or partial refusal (broad)", lambda l: l in ("r", "p")))


def cp(x, n, a=0.025):
    lo = beta.ppf(a, x, n - x + 1) if x > 0 else 0.0
    hi = beta.ppf(1 - a, x + 1, n - x) if x < n else 1.0
    return float(lo), float(hi)


def frac(x, n):
    if n == 0:
        return "-"
    lo, hi = cp(x, n)
    return f"{x / n:.3f} ({lo:.2f}-{hi:.2f}; {x}/{n})"


def kappa(a, b):
    cats = sorted(set(a) | set(b)); n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def rho2(y, f):
    y, f = np.asarray(y, float), np.asarray(f, float)
    return float(np.corrcoef(y, f)[0, 1] ** 2) if y.std() > 0 and f.std() > 0 else float("nan")


def boot(rows, stat, B=2000, seed=0):
    """Percentile interval, resampling prompts (each prompt appears once per model)."""
    by = collections.defaultdict(list)
    for r in rows:
        by[r["pid"]].append(r)
    pids = sorted(by); rng = np.random.default_rng(seed); out = []
    for _ in range(B):
        samp = [x for p in rng.choice(len(pids), len(pids)) for x in by[pids[p]]]
        v = stat(samp)
        if v == v:
            out.append(v)
    return np.percentile(out, [2.5, 97.5])


def main():
    g = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(SRC, "xstest_guard.jsonl"))}
    hyb = re.compile(HYB, re.I)
    rows = []
    for f in sorted(glob.glob(os.path.join(SRC, "xstest", "*.csv"))):
        model = os.path.basename(f)[len("xstest_v2_completions_"):-4]
        for r in csv.DictReader(open(f, encoding="utf-8")):
            s = g[f"{model}/{r['id']}"]
            t = r["completion"]
            rows.append(dict(id=f"{model}/{r['id']}", model=model, pid=r["id"], type=r["type"],
                             safe=int(not r["type"].startswith("contrast_")), a1=CLS[r["annotation_1"]], a2=CLS[r["annotation_2"]],
                             human=CLS[r["final_label"]], guard=s["refusal"], unsafe=s["unsafe"], logit=s["logit"],
                             hyb_shape=int(bool(hyb.search(t.strip())) and len(t.split()) > HYB_MIN_WORDS)))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "guard_scores.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    no_logit = sum(r["logit"] is None for r in rows)
    L = ["# Qwen3Guard-4B's refusal flag against XSTest's human labels", "",
         f"{len(rows)} completions from {len(MODELS)} model variants; {no_logit} without a Refusal line in the guard's verdict. "
         "Intervals are 95%: Clopper-Pearson per model, a bootstrap over prompts for the pooled rows.", ""]
    for safe, title in ((1, "Safe prompts (250 per model): the over-refusal setting"), (0, "Unsafe contrast prompts (200 per model)")):
        S = [r for r in rows if r["safe"] == safe]
        L += [f"## {title}", ""]
        for name, ev in EVENTS:
            L += [f"### Human event: {name}", "", "| model | human rate | guard flag rate | recall | false-alarm rate | precision |", "|---|---|---|---|---|---|"]
            for m in list(MODELS) + ["pooled"]:
                R = S if m == "pooled" else [r for r in S if r["model"] == m]
                y = [ev(r["human"]) for r in R]; f = [r["guard"] == 1 for r in R]
                tp = sum(a and b for a, b in zip(y, f)); pos = sum(y); flg = sum(f)
                if m == "pooled":
                    rec = lambda X: (sum(ev(r["human"]) and r["guard"] for r in X) / max(1, sum(ev(r["human"]) for r in X)))
                    fa = lambda X: (sum((not ev(r["human"])) and r["guard"] for r in X) / max(1, sum(not ev(r["human"]) for r in X)))
                    rl, rh = boot(R, rec); fl, fh = boot(R, fa)
                    L.append(f"| **pooled** | {pos / len(R):.3f} | {flg / len(R):.3f} | {tp / pos:.3f} ({rl:.2f}-{rh:.2f}; {tp}/{pos}) | "
                             f"{(flg - tp) / (len(R) - pos):.3f} ({fl:.2f}-{fh:.2f}; {flg - tp}/{len(R) - pos}) | {tp / flg:.3f} |")
                else:
                    L.append(f"| {MODELS[m]} | {pos / len(R):.3f} | {flg / len(R):.3f} | {frac(tp, pos)} | {frac(flg - tp, len(R) - pos)} | "
                             f"{(tp / flg if flg else float('nan')):.3f} |")
            L.append("")
        L += ["### What the guard does with each human class", "", "| model | full compliance: flagged | full refusal: flagged | partial refusal: flagged |", "|---|---|---|---|"]
        for m in list(MODELS) + ["pooled"]:
            R = S if m == "pooled" else [r for r in S if r["model"] == m]
            cells = [frac(sum(r["guard"] for r in R if r["human"] == k), sum(r["human"] == k for r in R)) for k in "arp"]
            L.append(f"| {'**pooled**' if m == 'pooled' else MODELS[m]} | " + " | ".join(cells) + " |")
        L += ["", "### Agreement (Cohen's kappa)", "", "| model | two humans, three classes | two humans, strict | guard vs final label, strict | guard vs final label, broad |", "|---|---|---|---|---|"]
        for m in list(MODELS) + ["pooled"]:
            R = S if m == "pooled" else [r for r in S if r["model"] == m]
            L.append(f"| {'**pooled**' if m == 'pooled' else MODELS[m]} | {kappa([r['a1'] for r in R], [r['a2'] for r in R]):.2f} | "
                     f"{kappa([r['a1'] == 'r' for r in R], [r['a2'] == 'r' for r in R]):.2f} | "
                     f"{kappa([r['human'] == 'r' for r in R], [r['guard'] == 1 for r in R]):.2f} | "
                     f"{kappa([r['human'] in 'rp' for r in R], [r['guard'] == 1 for r in R]):.2f} |")
        L += ["", "### rho^2 between the human label and the guard's feature (what PPI++ can use)", "",
              "| model | strict: flag | strict: p | strict: logit | broad: flag | broad: p | broad: logit |", "|---|---|---|---|---|---|---|"]
        for m in list(MODELS) + ["pooled"]:
            R = [r for r in (S if m == "pooled" else [r for r in S if r["model"] == m]) if r["logit"] is not None]
            lg = np.array([r["logit"] for r in R]); p = 1 / (1 + np.exp(-lg)); fl = np.array([r["guard"] for r in R])
            cells = []
            for _, ev in EVENTS:
                y = [ev(r["human"]) for r in R]
                cells += [f"{rho2(y, fl):.2f}", f"{rho2(y, p):.2f}", f"{rho2(y, lg):.2f}"]
            if m == "pooled":
                lo, hi = boot(R, lambda X: rho2([x["human"] == "r" for x in X], [x["logit"] for x in X]))
                cells[2] += f" ({lo:.2f}-{hi:.2f})"
            L.append(f"| {'**pooled**' if m == 'pooled' else MODELS[m]} | " + " | ".join(cells) + " |")
        L.append("")
    S = [r for r in rows if r["safe"] == 1]
    hp = [r for r in S if r["human"] == "p"]; sh = [r for r in S if r["hyb_shape"]]
    L += ["## The surface pattern for refuse-then-answer, against human partial refusal (safe prompts)", "",
          f"The sheet's strata use a surface pattern (a refusal opener and more than {HYB_MIN_WORDS} words). Of {len(sh)} completions "
          f"with the pattern, humans called {sum(r['human'] == 'p' for r in sh)} partial refusals, {sum(r['human'] == 'r' for r in sh)} "
          f"full refusals and {sum(r['human'] == 'a' for r in sh)} compliance. Of {len(hp)} human partial refusals, {sum(r['hyb_shape'] for r in hp)} "
          "have the pattern. It is a stratifier, not a label.", ""]
    open(os.path.join(OUT, "analysis.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
