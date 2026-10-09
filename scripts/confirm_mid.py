"""A mid-rate confirmation pool for the reference-rate strata (paper plan, section 9, step R5b).

The confirmation pool of step R5 (``scripts/confirm_pool.py``) gave refusal rates of 12%, 82% and
98%: no label fell between 18% and 65%, the range no earlier cell tested either. No slice of
OR-Bench sits there for this policy, so this pool is a fixed mixture: every hard-1K prompt that
nothing has used yet, topped up to 400 with unused benign prompts of the 80K set. The
registration, ``.planning/paper-certification/R5b_registration.md``, fixes the pool, labels,
sizes, checks and predictions; it, this script and the pool were committed before any response on
the new prompts was generated.

Generation, judging and the two checks are ``confirm_pool.py``'s and the checks' own code, called
unchanged; this file only builds the pool, points that code at it, and scores the new
predictions.

Stages:

- ``build``     the pool K4 (CPU, no model).
- ``preview``   the same analysis on a re-mix of step R5's responses (207 prompts of its K1 and 193
                of its K2): a test of the code path and the source of the registered expectations.
                Seen data; not a confirmation.
- ``generate``, ``judge``   ``confirm_pool.py``'s stages. GPU.
- ``analyse``   ``replacement_check.py`` and ``twophase_check.py`` on K4, the pre-flight, and the
                registered predictions scored. CPU.

    .venv/bin/python scripts/confirm_mid.py --stage build
    OMP_NUM_THREADS=1 .venv/bin/python scripts/confirm_mid.py --stage preview
    scripts/run_confirm_mid.sh generate
    scripts/run_confirm_mid.sh judge
    OMP_NUM_THREADS=1 .venv/bin/python scripts/confirm_mid.py --stage analyse
    -> results/paper/confirm/mid/ (pools.json, gen.jsonl, judged.jsonl, confirm.json, confirm.md)
"""
import argparse
import collections
import json
import os
import sys
import time

sys.dont_write_bytecode = True

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import confirm_pool as CP  # noqa: E402

R5 = CP.OUT                                           # step R5's files: its pools are excluded, its responses previewed
OUT = os.path.join(R5, "mid")                         # under confirm/, which used_before() does not scan
SEED = 20261009
N = 400
POOL, LABS = "K4", ["refusal", "unsafe"]
NAME = "hard-1K and 80K benign prompts, mixed"
WALD_GAIN = 1.7                                       # prediction M5's floor, fixed in the registration

# confirm_pool.py's stages read these at call time; nothing in that file is edited
CP.OUT, CP.RUN_DIR, CP.SEED = OUT, "/mnt/d/seldonian-runs/r5b", SEED
CP.SOURCES = collections.OrderedDict([(POOL, "or-bench-hard-1k + or-bench-80k")])
CP.LABELS = {POOL: LABS}
CP.NAMES = {POOL: NAME}


# ---------------------------------------------------------------- build

def stage_build(a):
    from datasets import load_dataset
    os.makedirs(OUT, exist_ok=True)
    seen, stats = CP.used_before()
    r5 = json.load(open(os.path.join(R5, "pools.json")))
    r5_keys = {CP.norm(it["prompt"]) for items in r5.values() for it in items}
    stats["r5_pool_prompts"] = len(r5_keys)
    seen |= r5_keys
    hard = load_dataset("bench-llm/or-bench", "or-bench-hard-1k", split="train")
    big = load_dataset("bench-llm/or-bench", "or-bench-80k", split="train")
    hard_keys = {CP.norm(x) for x in hard["prompt"]}

    def unused(rows, skip=()):
        out, keys = [], set()
        for row in rows:
            k = CP.norm(row["prompt"])
            if k in seen or k in keys or k in skip:
                continue
            keys.add(k)
            out.append(row)
        return out
    c_hard, c_big = unused(hard), unused(big, hard_keys)
    n_big = N - len(c_hard)
    assert 0 < n_big < N, len(c_hard)
    rng = np.random.default_rng(SEED)
    pick = sorted(rng.permutation(len(c_big))[:n_big].tolist())
    items = [dict(prompt=r["prompt"], meta=r["category"], src="hard-1k") for r in c_hard]
    items += [dict(prompt=c_big[j]["prompt"], meta=c_big[j]["category"], src="80k") for j in pick]
    items = [items[j] for j in rng.permutation(N)]
    pool = [dict(i=i, prompt=it["prompt"], plain=it["prompt"], meta=it["meta"], src=it["src"]) for i, it in enumerate(items)]
    stats[POOL] = {"hard-1k": dict(rows=len(hard), unused=len(c_hard), taken=len(c_hard)),
                   "80k": dict(rows=len(big), unused=len(c_big), taken=n_big)}
    json.dump({POOL: pool}, open(os.path.join(OUT, "pools.json"), "w"), indent=0)
    json.dump(dict(seed=SEED, n=N, exclusion=stats), open(os.path.join(OUT, "pools_meta.json"), "w"), indent=1)
    print(stats)


def stage_preview(a):
    """Step R5's judged responses, re-mixed in this pool's proportions and analysed as this pool will be."""
    meta = json.load(open(os.path.join(OUT, "pools_meta.json")))["exclusion"][POOL]
    take = {"K2": meta["hard-1k"]["taken"], "K1": meta["80k"]["taken"]}
    rng = np.random.default_rng(SEED)
    rows = CP.rows_of(os.path.join(R5, "judged.jsonl"))
    new = {}
    for p in ("K2", "K1"):
        ids = sorted({r["i"] for r in rows if r["pool"] == p})
        for i in sorted(rng.permutation(len(ids))[:take[p]].tolist()):
            new[(p, ids[i])] = len(new)
    path = os.path.join(OUT, "preview_judged.jsonl")
    with open(path, "w") as fh:
        for r in rows:
            if (r["pool"], r["i"]) in new:
                fh.write(json.dumps(dict(r, pool=POOL, i=new[(r["pool"], r["i"])], src="hard-1k" if r["pool"] == "K2" else "80k")) + "\n")
    a.judged, a.out = path, os.path.join(OUT, "preview")
    stage_analyse(a, preview=True)


# ---------------------------------------------------------------- analyse (CPU)

def describe(judged, lab):
    """What the mixture is made of: rates by source, and how much of the strata's gain the source alone gives."""
    import replacement_check as RC
    rows = CP.rows_of(judged)
    src = {r["i"]: r["src"] for r in rows if "src" in r}
    if not src:
        src = {it["i"]: it["src"] for it in json.load(open(os.path.join(OUT, "pools.json")))[POOL]}
    cov = {r["i"]: r for r in rows if r["role"] == "cov"}
    cand = {r["i"]: r for r in rows if r["role"] == "cand"}
    ids = sorted(set(cov) & set(cand))
    f = np.array([np.mean(cov[i][lab][:RC.K_REF]) for i in ids])
    p = np.array([np.mean(cand[i][lab]) for i in ids])
    s = np.array([src[i] for i in ids])
    out = dict(n=len(ids), rate=float(p.mean()), by_source={})
    for k in ("hard-1k", "80k"):
        m = s == k
        out["by_source"][k] = dict(n=int(m.sum()), rate=float(p[m].mean()), rate_ref=float(f[m].mean()),
                                   at0=float((p[m] == 0).mean()), at1=float((p[m] == 1).mean()))
    out["at0"], out["at1"] = float((p == 0).mean()), float((p == 1).mean())

    def ratio(strata):                                 # variance of a random draw over that of a proportional stratified one
        within = sum((strata == h).mean() * p[strata == h].mean() * (1 - p[strata == h].mean()) for h in np.unique(strata))
        return float(p.mean() * (1 - p.mean()) / within) if within > 0 else float("inf")
    out["var_ratio_source"] = ratio((s == "hard-1k").astype(int))
    rng = np.random.default_rng(SEED)
    st = [RC.PM.quantile_strata(f, RC.H, rng, ties="random") for _ in range(200)]
    out["var_ratio_strata"] = float(np.mean([ratio(x) for x in st]))
    out["hard_share_by_stratum"] = [float(np.mean([(s[x == h] == "hard-1k").mean() for x in st])) for h in range(RC.H)]
    return out


def stage_analyse(a, preview=False):
    from concurrent.futures import ProcessPoolExecutor
    judged = a.judged or os.path.join(OUT, "judged.jsonl")
    RC, TP = CP.patch(judged)
    cells = [(POOL, lab) for lab in LABS]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        pf = list(ex.map(CP.preflight_job, [(judged, p, lab) for p, lab in cells]))
        rc = [r for rows in ex.map(RC.job, [("013", p, lab, CP.STEP, a.reps, a.reps_large) for p, lab in cells]) for r in rows]
        print(f"replacement check done {time.time() - t0:.0f}s", flush=True)
        tp = [r for rows in ex.map(TP.job, [("013", p, lab, CP.STEP, a.reps_two) for p, lab in cells]) for r in rows]
        print(f"two-phase check done {time.time() - t0:.0f}s", flush=True)
    res = dict(seed=SEED, step=CP.STEP, preview=preview, reps=a.reps, reps_large=a.reps_large, reps_two=a.reps_two, preflight=pf,
               replacement=rc, twophase=tp, describe={lab: describe(judged, lab) for lab in LABS})
    out = a.out or os.path.join(OUT, "confirm")
    json.dump(res, open(out + ".json", "w"))
    report(res, out + ".md")


def report(res, path):
    import replacement_check as RC
    pf = {r["label"]: r for r in res["preflight"]}
    key = lambda r: (r["env"].split(":")[1], r["n"], r["delta"])           # noqa: E731
    large, stored, two = collections.defaultdict(dict), collections.defaultdict(dict), collections.defaultdict(dict)
    for r in res["replacement"]:
        if r["draw"] in ("large", "stored"):
            (large if r["draw"] == "large" else stored)[key(r)][r["arm"]] = r
    for r in res["twophase"]:
        two[key(r)][r["arm"]] = r

    def cls(r):
        return RC.classify(r["miss"], r["delta"], r["reps"])

    def mark(r):
        k = cls(r)
        return f"**{r['miss']:.3f}**" if k == RC.OVER else (f"{r['miss']:.3f}?" if k == RC.UNRES else f"{r['miss']:.3f}")

    def ess(group, k, arm, base="pooled Wilson"):
        return (group[k][base]["excess"] / group[k][arm]["excess"]) ** 2

    def over(group, lab, arm, deltas=RC.DELTAS):
        return sum(cls(group[(lab, n_s, d)][arm]) == RC.OVER for n_s in (100, 200) for d in deltas)

    what = ("A PREVIEW on seen data: step R5's judged responses, re-mixed in this pool's proportions. It tests the code and gives "
            "the registration its expectations; it confirms nothing." if res.get("preview") else
            "Registered in `.planning/paper-certification/R5b_registration.md` before any response on these prompts was generated.")
    L = ["# The mid-rate confirmation pool (plan step R5b)" + (": preview" if res.get("preview") else ""), "",
         f"`scripts/confirm_mid.py`. {what} One pool of {N} OR-Bench prompts, every unused hard-1K prompt topped up with unused benign "
         f"prompts of the 80K set; Granite-3.3-2B, {CP.COV} reference responses a prompt for the strata and {CP.CAND} of the trained "
         f"policy; Qwen3Guard-4B labels. Checks: `replacement_check.py` ({res['reps']:,} stored draws, {res['reps_large']:,} with "
         f"replacement) and `twophase_check.py` ({res['reps_two']:,}), called unchanged.", "",
         "## 1. The labels", "",
         "| label | rate, trained policy | rate, reference | class | ICC of the reference | prompts with a reference rate of 0 or 1 | "
         "pre-flight ESS |", "|---|---|---|---|---|---|---|"]
    for lab in LABS:
        r = pf[lab]
        L.append(f"| {lab} | {r['rate']:.3f} | {r['rate_ref']:.3f} | {CP.kind(r['rate'])} | {r['pf_icc_ref']:.2f} | "
                 f"{100 * r['share_ref_0_or_1']:.0f}% | {r['pf_ess_pred']:.2f} |")
    L += ["", "What the mixture is made of (the trained policy's 16 responses a prompt):", "",
          "| label | source | prompts | rate | reference rate | prompts at a rate of 0 | at 1 |", "|---|---|---|---|---|---|---|"]
    for lab in LABS:
        for k, v in res["describe"][lab]["by_source"].items():
            L.append(f"| {lab} | {k} | {v['n']} | {v['rate']:.3f} | {v['rate_ref']:.3f} | {100 * v['at0']:.0f}% | {100 * v['at1']:.0f}% |")
    d0 = res["describe"]["refusal"]
    L += ["", f"The source alone explains much of what the strata gain on the refusal label. The variance of a random draw over that "
          f"of a proportional stratified draw is {d0['var_ratio_source']:.2f} with two strata, the two sources, and "
          f"{d0['var_ratio_strata']:.2f} with the 8 strata of the reference rate. Share of hard-1K prompts in the 8 strata, lowest "
          f"reference rate first: " + ", ".join(f"{100 * x:.0f}%" for x in d0["hard_share_by_stratum"]) + "."]
    arms = ("b1w", "Wald-t b1", "StratPPI estimator, bootstrap-t", "StratPPI, normal limit", "pooled Wilson", "Clopper-Pearson")
    L += ["", f"## 2. Validity with a large pool (strata fixed, drawn with replacement, {res['reps_large']:,} draws)", "",
          "Miss rates; **bold** is over the level, `?` unresolved.", "",
          "| label | n_s | delta | " + " | ".join(arms) + " |", "|---|---|---|" + "---|" * len(arms)]
    for lab in LABS:
        for n_s in (100, 200):
            for d in RC.DELTAS:
                g = large[(lab, n_s, d)]
                L.append(f"| {lab} ({pf[lab]['rate']:.2f}) | {n_s} | {d} | " + " | ".join(mark(g[x]) for x in arms) + " |")
    L += ["", "## 3. Gain (ESS against the pooled Wilson bound on a random draw, from the truth to the limit, delta 0.05)", "",
          "| label | n_s | pre-flight | b1w, large pool | ratio to pre-flight | b1w, stored design | Wald-t, large pool | bootstrap-t "
          "StratPPI, large pool |", "|---|---|---|---|---|---|---|---|"]
    gains = {}
    for lab in LABS:
        for n_s in (100, 200):
            k = (lab, n_s, 0.05)
            e, pred = ess(large, k, "b1w"), pf[lab]["pf_ess_pred"]
            gains[(lab, n_s)] = dict(ess=e, pred=pred, ratio=e / pred, wald=ess(large, k, "Wald-t b1"))
            L.append(f"| {lab} | {n_s} | {pred:.2f} | {e:.2f} | {e / pred:.2f} | {ess(stored, k, 'b1w'):.2f} | "
                     f"{gains[(lab, n_s)]['wald']:.2f} | {ess(large, k, 'StratPPI estimator, bootstrap-t'):.2f} |")
    tarms = ("b1w, sampled-pool term", "Wald-t b1, sampled-pool term", "b1w, no term", "StratPPI estimator, bootstrap-t", "pooled Wilson",
             "Clopper-Pearson")
    L += ["", f"## 4. A claim about the prompt source (pool redrawn, strata rebuilt, {res['reps_two']:,} replications)", "",
          "| label | n_s | delta | " + " | ".join(tarms) + " | ESS, b1w with the term | its cap | ESS, Wald-t with the term | its cap |",
          "|---|---|---|" + "---|" * (len(tarms) + 4)]
    caps = {}
    for lab in LABS:
        for n_s in (100, 200):
            for d in RC.DELTAS:
                k = (lab, n_s, d)
                c = {}
                for tag, arm, base in (("b1w", "b1w, sampled-pool term", "b1w"), ("wald", "Wald-t b1, sampled-pool term", "Wald-t b1")):
                    G = 1 - 1 / ess(large, k, base)
                    c[tag] = dict(ess=ess(two, k, arm), cap=1 / (1 - G + G * n_s / pf[lab]["n_prompts"]))
                caps[k] = c
                L.append(f"| {lab} | {n_s} | {d} | " + " | ".join(mark(two[k][x]) for x in tarms) +
                         f" | {c['b1w']['ess']:.2f} | {c['b1w']['cap']:.2f} | {c['wald']['ess']:.2f} | {c['wald']['cap']:.2f} |")

    # ---- the registered predictions (on the refusal label; M9 on a rare label)
    lab, rate = "refusal", pf["refusal"]["rate"]
    mid = CP.kind(rate) == "mid"
    S = []
    S.append(("M1", "with a large pool the stratified Wald-t limit is over its level in no cell", over(large, lab, "Wald-t b1") == 0,
              f"{over(large, lab, 'Wald-t b1')} of 4 over"))

    def gap(arm):                                      # the largest and smallest miss minus the level, over the four cells
        v = [large[(lab, n_s, d)][arm]["miss"] - d for n_s in (100, 200) for d in RC.DELTAS]
        return min(v), max(v)
    s_over, s_gap = over(large, lab, "StratPPI estimator, bootstrap-t"), gap("StratPPI estimator, bootstrap-t")
    S.append(("M2", "with a large pool the bootstrap-t StratPPI limit is over its level in at most one of its four cells, and its miss "
              "exceeds the level by no more than 0.01 in any", s_over <= 1 and s_gap[1] <= 0.01,
              f"{s_over} of 4 over; largest miss minus level {s_gap[1]:+.3f}"))
    b, b_gap = over(large, lab, "b1w"), gap("b1w")
    S.append(("M3a", "with a large pool `b1w` is over its level in no cell if the rate is under 0.45, and in at least one of its four "
              "cells if it is above 0.55 (step R5's rule)",
              (b == 0) if rate < 0.45 else (b > 0) if rate > 0.55 else None, f"rate {rate:.3f}; {b} of 4 over"))
    S.append(("M3b", "if the rate is between 0.45 and 0.55, the miss of `b1w` with a large pool is within 0.01 of its level in every cell",
              (b_gap[0] >= -0.01 and b_gap[1] <= 0.01) if 0.45 <= rate <= 0.55 else None,
              f"rate {rate:.3f}; miss minus level from {b_gap[0]:+.3f} to {b_gap[1]:+.3f}"))
    S.append(("M4", "the gain of `b1w` with a large pool is within 20% of the pre-flight's prediction at both safety-set sizes",
              all(abs(gains[(lab, n)]["ratio"] - 1) <= 0.20 for n in (100, 200)),
              "; ".join(f"n_s {n}: {gains[(lab, n)]['ess']:.2f} against {gains[(lab, n)]['pred']:.2f}" for n in (100, 200))))
    S.append(("M5", f"the gain of the Wald-t limit with a large pool is at least {WALD_GAIN} at both safety-set sizes",
              all(gains[(lab, n)]["wald"] >= WALD_GAIN for n in (100, 200)),
              "; ".join(f"n_s {n}: {gains[(lab, n)]['wald']:.2f}" for n in (100, 200))))
    w = over(two, lab, "Wald-t b1, sampled-pool term")
    S.append(("M6", "for the prompt source, the Wald-t limit with the sampled-pool term is over in no cell", w == 0, f"{w} of 4 over"))
    t0, ts = over(two, lab, "b1w, no term", (0.05,)), over(two, lab, "StratPPI estimator, bootstrap-t", (0.05,))
    S.append(("M7", "for the prompt source, `b1w` without the term and the bootstrap-t StratPPI limit are each over in at least one of "
              "the two cells at delta 0.05", t0 >= 1 and ts >= 1, f"no term: {t0} of 2; bootstrap-t StratPPI: {ts} of 2"))
    cw = [caps[(lab, n, 0.05)]["wald"] for n in (100, 200)]
    S.append(("M8", "for the prompt source, the gain of the Wald-t limit with the term is above 1 and within 15% of section 6.4's cap "
              "at both safety-set sizes", all(v["ess"] > 1 and abs(v["ess"] / v["cap"] - 1) <= 0.15 for v in cw),
              "; ".join(f"n_s {n}: {v['ess']:.2f} against {v['cap']:.2f}" for n, v in zip((100, 200), cw))))
    rare = [x for x in LABS if CP.kind(pf[x]["rate"]) == "rare"]
    cp_over = sum(over(large, x, "Clopper-Pearson") for x in rare)
    S.append(("M9", "on a rare label the strata gain nothing (ESS of `b1w` between 0.9 and 1.15) and Clopper-Pearson on a random draw is "
              "over in no cell", (all(0.9 <= gains[(x, n)]["ess"] <= 1.15 for x in rare for n in (100, 200)) and cp_over == 0)
              if rare else None,
              ("; ".join(f"{x} n_s {n}: {gains[(x, n)]['ess']:.2f}" for x in rare for n in (100, 200)) +
               f"; Clopper-Pearson {cp_over} of {4 * len(rare)} over") if rare else "no rare label"))
    if not mid:                                        # M1 to M8 are about a mid-rate label
        S = [(t, x, ok if t == "M9" else None, f) for t, x, ok, f in S]
    L += ["", "## 5. The registered predictions", "", "| | prediction | outcome | what was found |", "|---|---|---|---|"]
    for tag, text, ok, found in S:
        L.append(f"| {tag} | {text} | {'no label to test it' if ok is None else 'kept' if ok else '**refuted**'} | {found} |")
    b4 = over(two, lab, "b1w, sampled-pool term")
    in_gap = 0.18 < rate < 0.65
    L += ["", f"The refusal rate is {rate:.3f}, {'inside' if in_gap else 'OUTSIDE'} the range of 18% to 65% that no earlier cell tested.",
          "", f"Reported without a prediction: `b1w` with the sampled-pool term is over in {b4} of 4 cells for the prompt source."]
    res["scored"] = [dict(tag=t, text=x, kept=ok, found=f) for t, x, ok, f in S]
    res["in_gap"], res["b1w_term_over"] = in_gap, b4
    res["gains"] = {f"{k[0]}|{k[1]}": v for k, v in gains.items()}
    res["caps"] = {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in caps.items()}
    json.dump(res, open(path[:-3] + ".json", "w"))
    open(path, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["build", "preview", "generate", "judge", "analyse", "report"], required=True)
    ap.add_argument("--gen-batch", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=256, help="responses per write")
    ap.add_argument("--judge-batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="never for the registered run")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--reps", type=int, default=CP.REPS)
    ap.add_argument("--reps-large", type=int, default=CP.REPS_LARGE)
    ap.add_argument("--reps-two", type=int, default=CP.REPS_TWO)
    ap.add_argument("--judged", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.limit:
        sys.exit("--limit is not for the registered run")
    if a.stage == "report":
        out = a.out or os.path.join(OUT, "confirm")
        return report(json.load(open(out + ".json")), out + ".md")
    {"build": stage_build, "preview": stage_preview, "generate": CP.stage_generate, "judge": CP.stage_judge,
     "analyse": stage_analyse}[a.stage](a)


if __name__ == "__main__":
    main()
