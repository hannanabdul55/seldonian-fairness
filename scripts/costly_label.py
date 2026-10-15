"""O3: the reference-rate design where it can pay, a costly label stratified on a cheap one.

Registered in `.planning/paper-certification/O3_registration.md` before any outcome was
computed. Pool: XSTest's 250 safe prompts (Röttger et al., 2024). Costly label: the published
human label (full refusal = strict; full or partial = broad) on one target model's completion,
for each of the five released model variants. Cheap stratifiers, which never see a human
label:

- ``B``  the paper's design: Qwen3Guard-4B's refusal flag on 8 Granite-3.3-2B reference
         responses per prompt (spike 013, pool C1, step 0), a reference model that is not the
         target;
- ``A``  the guard's mean refusal probability (sigmoid of its logit) on the completions of the
         other model families' variants (GPT-4; Llama 2 x 2; Mistral x 2), the target's own
         family left out.

8 equal rank strata with random ties, proportional allocation, safety sets of 50 and 100
(20% and 40% of the pool), deltas 0.05 and 0.10. Two draws: ``fixed`` (without replacement
from the 250; the claim is the pool's rate) and ``large`` (with replacement within strata, the
limit of a pool far larger than the safety set with the same strata and rate; 40,000 draws).
Arms: the stratified Wald-t limit ``b1`` and ``b1w`` on the stratified draw; the pooled Wilson
bound and Clopper-Pearson on a simple random draw of the same size. Gain = (excess of the
pooled Wilson bound over the truth / the arm's excess)^2 on the same kind of draw, as in
``scripts/replacement_check.py``.

    OMP_NUM_THREADS=1 .venv/bin/python scripts/costly_label.py run
    .venv/bin/python scripts/costly_label.py report
    -> results/paper/costly_label.json, results/paper/costly_label.md
"""
import argparse
import collections
import csv
import json
import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, ".planning", "spikes", "013-stratified-safety-set"))
import plasmode as PM            # noqa: E402
import stratbounds as SB         # noqa: E402
sys.path.insert(0, ROOT)
from seldonian.bounds_eval import cp_upper  # noqa: E402

OUT = os.path.join(ROOT, "results", "paper")
XS = "/mnt/d/seldonian-runs/p15/xstest"
MODELS = ("gpt4", "llama2orig", "llama2new", "mistralinstruct", "mistralguard")
FAMILY = {"gpt4": "gpt", "llama2orig": "llama", "llama2new": "llama", "mistralinstruct": "mistral",
          "mistralguard": "mistral"}
EVENTS = {"strict": ("r",), "broad": ("r", "p")}
STRATIFIERS = ("B", "A")
NS = (50, 100)
DELTAS = (0.05, 0.1)
H = 8
REPS = {"fixed": 20000, "large": 40000}
Z = 2.0
RARE = 0.05                      # a label is rare under 5% (the paper's class limit)


def load():
    """Per safe prompt: human labels and guard probabilities for every model; B's reference rate."""
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "results/labels/xstest/guard_scores.jsonl"))]
    rows = [r for r in rows if r["safe"] == 1]
    pids = sorted({r["pid"] for r in rows}, key=lambda p: int(p.split("-")[1]))
    ix = {p: i for i, p in enumerate(pids)}
    human = {m: np.empty(len(pids), dtype=object) for m in MODELS}
    prob = {m: np.zeros(len(pids)) for m in MODELS}
    for r in rows:
        human[r["model"]][ix[r["pid"]]] = r["human"]
        prob[r["model"]][ix[r["pid"]]] = 1.0 / (1.0 + np.exp(-r["logit"]))
    text = {}
    with open(os.path.join(XS, "xstest_v2_completions_gpt4.csv")) as f:
        for d in csv.DictReader(f):
            text[d["prompt"]] = d["id"]
    pools = json.load(open(os.path.join(ROOT, "results/spikes/013/pools.json")))["C1"]
    ref = np.full(len(pids), np.nan)
    for l in open(os.path.join(ROOT, "results/spikes/013/judged_full.jsonl")):
        d = json.loads(l)
        if d["pool"] == "C1" and d["step"] == 0 and d["role"] == "cov" and d["meta"] == "xstest":
            ref[ix[text[pools[int(d["i"])]["plain"]]]] = np.mean(d["refusal"][:8])
    assert not np.isnan(ref).any(), "a safe prompt has no Granite reference rate"
    return pids, human, prob, ref


def stratifier(name, target, prob, ref):
    if name == "B":
        return ref
    others = [m for m in MODELS if FAMILY[m] != FAMILY[target]]
    return np.mean([prob[m] for m in others], axis=0)


def draw_split(strata, n_s, rng, replace):
    if not replace:
        return PM._draw_split(strata, n_s, rng)
    N = len(strata)
    idx, st = [], []
    for h in np.unique(strata):
        mem = np.flatnonzero(strata == h)
        k = max(2, int(round(n_s * len(mem) / N)))
        idx.append(rng.choice(mem, size=k, replace=True))
        st.append(np.full(k, h))
    return np.concatenate(idx), np.concatenate(st)


def job(args):
    target, event, strat = args
    pids, human, prob, ref = load()
    y_all = np.array([l in EVENTS[event] for l in human[target]], dtype=float)
    f = stratifier(strat, target, prob, ref)
    N = len(y_all)
    truth = float(y_all.mean())
    srng = np.random.default_rng(zlib.crc32(f"O3:{strat}:{target}".encode()))
    strata = PM.quantile_strata(f, H, srng, ties="random")
    W = np.bincount(strata, minlength=H) / N
    rows = []
    for n_s in NS:
        for draw, R in REPS.items():
            replace = draw == "large"
            acc = collections.defaultdict(list)
            for r in range(R):
                rr = np.random.default_rng([zlib.crc32(f"{target}:{event}:{strat}:{draw}".encode()), n_s, r])
                ids, st = draw_split(strata, n_s, rr, replace)
                y = y_all[ids]
                s = np.array([y[st == h].sum() for h in range(H)])
                n = np.array([(st == h).sum() for h in range(H)])
                ids_r = rr.choice(N, size=n_s, replace=replace)
                yr = y_all[ids_r]
                for d in DELTAS:
                    acc[("Wald-t b1", d)].append(SB.b1(s, n, W, d))
                    acc[("b1w", d)].append(SB.b1w(s, n, W, d))
                    acc[("pooled Wilson", d)].append(SB.b1w(yr.sum(), n_s, 1.0, d))
                    acc[("Clopper-Pearson", d)].append(cp_upper(int(yr.sum()), n_s, d))
            for (arm, d), ub in acc.items():
                ub = np.asarray(ub)
                k = int((~(ub >= truth)).sum())
                rows.append(dict(target=target, event=event, stratifier=strat, n_s=n_s, draw=draw, delta=d,
                                 arm=arm, R=R, truth=truth, k=k, miss=k / R, excess=float(ub.mean() - truth)))
    return rows


def classify(m, delta, R):
    se = np.sqrt(delta * (1 - delta) / R)
    return "over" if m > delta + Z * se + 1e-12 else ("unresolved" if m > delta + 1e-12 else "at or under")


def run(a):
    jobs = [(t, e, s) for t in MODELS for e in EVENTS for s in STRATIFIERS]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = [r for rs in ex.map(job, jobs) for r in rs]
    json.dump(dict(rows=rows), open(os.path.join(OUT, "costly_label.json"), "w"), indent=0)
    print(len(rows), "rows")


def report(a):
    rows = json.load(open(os.path.join(OUT, "costly_label.json")))["rows"]
    for r in rows:
        r["cls"] = classify(r["miss"], r["delta"], r["R"])
    ex = {(r["target"], r["event"], r["stratifier"], r["n_s"], r["draw"], r["delta"], r["arm"]): r for r in rows}

    def gain(t, e, s, n_s, draw, d, arm="Wald-t b1"):
        return (ex[(t, e, s, n_s, draw, d, "pooled Wilson")]["excess"] / ex[(t, e, s, n_s, draw, d, arm)]["excess"]) ** 2

    labels = sorted({(r["target"], r["event"], r["truth"]) for r in rows})
    mid = [(t, e) for t, e, p in labels if RARE <= p <= 1 - RARE]
    rare = [(t, e) for t, e, p in labels if p < RARE]
    ge10 = [(t, e) for t, e, p in labels if p >= 0.10]
    wald = [r for r in rows if r["arm"] == "Wald-t b1"]
    y1 = [r for r in wald if (r["target"], r["event"]) in ge10 and r["n_s"] == 100 and r["draw"] == "large"]
    y2 = [r for r in wald if (r["target"], r["event"]) in mid and r["draw"] == "fixed"]
    gB = [gain(t, e, "B", 100, "large", 0.05) for t, e in mid]
    gA = [gain(t, e, "A", 100, "large", 0.05) for t, e in mid]
    y5 = [r for r in rows if r["arm"] == "Clopper-Pearson" and (r["target"], r["event"]) in rare and r["draw"] == "large"]
    pred = [
        ("Y1", f"Wald-t on strata, labels at 10% or more ({len(ge10)}), n_s 100, with replacement: over in no cell "
               f"({len(y1)} cells)", sum(r["cls"] == "over" for r in y1) == 0),
        ("Y2", f"Wald-t on strata, mid-rate labels ({len(mid)}), without replacement: over in no cell ({len(y2)} cells)",
         sum(r["cls"] == "over" for r in y2) == 0),
        ("Y3", "stratifier B, n_s 100, with replacement, delta 0.05: median gain over the mid-rate labels at least 1.1",
         float(np.median(gB)) >= 1.1),
        ("Y4", "the same: no mid-rate label gains more than 2.0", max(gB) <= 2.0),
        ("Y5", "stratifier A gains more than B for at least 6 of the mid-rate labels",
         sum(a_ > b_ for a_, b_ in zip(gA, gB)) >= 6),
        ("Y6", f"Clopper-Pearson on a random draw, rare labels ({len(rare)}), with replacement: over in no cell",
         sum(r["cls"] == "over" for r in y5) == 0),
    ]
    L = ["# O3: a costly label stratified on a cheap one (XSTest, 250 safe prompts)", "",
         "Registered in `.planning/paper-certification/O3_registration.md`. Over: the miss exceeds delta by more "
         f"than {Z:g} Monte Carlo SE.", "", "## Labels", "", "| target | event | rate | class |", "|---|---|---|---|"]
    L += [f"| {t} | {e} | {p:.3f} | {'rare' if p < RARE else 'mid'} |" for t, e, p in labels]
    L += ["", "## Predictions", ""]
    L += [f"- **{q}** {t}: **{'kept' if ok else 'refuted'}**" for q, t, ok in pred]
    L += ["", "## Gains (Wald-t on strata against the pooled Wilson bound; with replacement)", "",
          "| target | event | stratifier | n_s | delta | Wald-t | b1w |", "|---|---|---|---|---|---|---|"]
    for t, e, _ in labels:
        for s in STRATIFIERS:
            for n_s in NS:
                for d in DELTAS:
                    L.append(f"| {t} | {e} | {s} | {n_s} | {d} | {gain(t, e, s, n_s, 'large', d):.2f} | "
                             f"{gain(t, e, s, n_s, 'large', d, 'b1w'):.2f} |")
    L += ["", "## Cells", "", "| target | event | stratifier | n_s | draw | delta | arm | R | miss | cell |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    L += [f"| {r['target']} | {r['event']} | {r['stratifier']} | {r['n_s']} | {r['draw']} | {r['delta']} | {r['arm']} | "
          f"{r['R']} | {r['miss']:.4f} | {r['cls']} |" for r in rows]
    open(os.path.join(OUT, "costly_label.md"), "w").write("\n".join(L) + "\n")
    out = json.load(open(os.path.join(OUT, "costly_label.json")))
    out["predictions"] = [dict(id=q, text=t, kept=bool(ok)) for q, t, ok in pred]
    json.dump(out, open(os.path.join(OUT, "costly_label.json"), "w"), indent=0)
    print("\n".join(L[:40]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "report", "check"])
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()
    if a.stage == "check":                      # data plumbing only: no label is read against a stratifier
        pids, human, prob, ref = load()
        print(len(pids), "safe prompts;", {m: int(sum(l is not None for l in human[m])) for m in MODELS})
        print("B distinct values:", len(np.unique(ref)), " A distinct (target gpt4):",
              len(np.unique(stratifier("A", "gpt4", prob, ref))))
    else:
        {"run": run, "report": report}[a.stage](a)
