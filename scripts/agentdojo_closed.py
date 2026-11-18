"""Q1 (2026-11-17): two checks on the AgentDojo certificate. Registered in
``.planning/paper-certification/Q1_registration.md`` before either was run.

A. Closed-form limits by user task. The paper's limit is a cluster bootstrap-t, which returns no
   limit on some tables. Two closed forms never abstain unless the resample has no variance:

       ct  = est + t_{m-1}(1 - delta) * se
       cts = ct + max(0, k3 / V) * (2 t^2 + 1) / 6

   with m user tasks, sizes n_j, sums s_j, est = sum s / sum n, e_j = s_j - est n_j,
   V = se^2 = m / (m - 1) * sum e^2 / n^2 (the bootstrap-t's own standard error) and
   k3 = sum e^3 / n^3 (third cumulant of the linearised estimate). ``cts`` is the one-term
   skew correction of ``waldt_skew.py`` carried to clusters. Checked under the scheme the
   certificate claims: user tasks drawn with replacement, injection tasks fixed, each pipeline's
   table the population (``agentdojo_recheck.py`` scheme a). A resample with V = 0 gives no limit
   (reported as 1.0, not a miss), as the bootstrap-t does.

B. Leave one injection task out. For a pipeline, a suite and one of the suite's injection tasks,
   the cluster bootstrap-t by user task is computed on the suite's other injection tasks and
   compared with the left-out task's own success rate over the same user tasks.

    .venv/bin/python scripts/agentdojo_closed.py --selftest
    .venv/bin/python scripts/agentdojo_closed.py --time      # timings only, prints no outcome
    OMP_NUM_THREADS=1 .venv/bin/python scripts/agentdojo_closed.py
    -> results/paper/agentdojo_closed.json, agentdojo_closed.md
"""
import argparse
import collections
import json
import os
import sys
import time

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
from scipy.stats import t as student  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import agentdojo_recheck as AR  # noqa: E402

DELTA, TAU = 0.05, 0.05
REPS = 20000
BOOTS = 4000          # the paper's inner bootstrap size
BIG = 200000          # the real-data bootstrap-t beside the closed forms
SEED = 20261117
CERT = "claude-3-5-sonnet-20241022"
OUT = os.path.join(AR.OUT_DIR, "agentdojo_closed")


def closed(sizes, sums, delta=DELTA):
    """(est, ct, cts, no_limit) for each row of cluster sizes and sums."""
    sizes = np.atleast_2d(np.asarray(sizes, dtype=float))
    sums = np.atleast_2d(np.asarray(sums, dtype=float))
    m = sizes.shape[1]
    n = sizes.sum(1)
    est = sums.sum(1) / n
    e = sums - est[:, None] * sizes
    V = m / (m - 1) * (e ** 2).sum(1) / n ** 2
    k3 = (e ** 3).sum(1) / n ** 3
    tq = student.ppf(1 - delta, m - 1)
    none = V <= 0
    Vs = np.where(none, 1.0, V)
    ct = est + tq * np.sqrt(Vs)
    cts = ct + np.maximum(0.0, k3 / Vs) * (2 * tq ** 2 + 1) / 6
    ct = np.where(none, 1.0, np.clip(ct, 0, 1))
    cts = np.where(none, 1.0, np.clip(cts, 0, 1))
    return est, ct, cts, none


def classify(miss, reps):
    se = np.sqrt(DELTA * (1 - DELTA) / reps)
    return "over" if miss > DELTA + 2 * se else ("unresolved" if miss > DELTA else "under")


def part_a(tables, reps):
    rng = np.random.default_rng(SEED)
    out = {}
    for p, (M, _, _) in tables.items():
        Y0, V = AR.prep(M)
        sizes, sums = V.sum(1), Y0.sum(1)
        truth = sums.sum() / sizes.sum()
        idx = rng.integers(0, len(sizes), (reps, len(sizes)))
        _, ct, cts, none = closed(sizes[idx], sums[idx])
        _, rct, rcts, _ = closed(sizes, sums)
        bt = AR.cluster_t_arr(sizes, sums, DELTA, np.random.default_rng(SEED + 1), BIG)
        out[p] = dict(truth=float(truth), m=int(len(sizes)), pairs=int(sizes.sum()), successes=int(sums.sum()),
                      tasks_with_success=int((sums > 0).sum()),
                      miss_ct=float(np.mean(ct < truth)), miss_cts=float(np.mean(cts < truth)),
                      no_limit=float(none.mean()), median_ct=float(np.median(ct)), median_cts=float(np.median(cts)),
                      real_ct=float(rct[0]), real_cts=float(rcts[0]), real_boot_t=float(bt))
    return out


def part_b(R_by, boots):
    rng = np.random.default_rng(SEED + 2)
    rows = []
    for p, R in R_by.items():
        by_suite = collections.defaultdict(list)
        for r in R:
            by_suite[r["suite"]].append(r)
        for suite, S in by_suite.items():
            for l in sorted({r["injection_task_key"] for r in S}):
                held = [r["y"] for r in S if r["injection_task_key"] == l]
                rest = collections.defaultdict(list)
                for r in S:
                    if r["injection_task_key"] != l:
                        rest[r["user_task_key"]].append(r["y"])
                if len(rest) < 2:
                    continue
                sizes = np.array([len(v) for v in rest.values()])
                sums = np.array([sum(v) for v in rest.values()], dtype=float)
                ub, est, se, q = AR.cluster_t_arr(sizes, sums, DELTA, rng, boots, detail=True)
                none = not (np.isfinite(q) and se > 0)
                _, _, cts, cnone = closed(sizes, sums)
                rows.append(dict(pipeline=p, suite=suite, task=l, held_rate=float(np.mean(held)), held_n=len(held),
                                 rest_rate=float(est), limit=float(ub), no_limit=bool(none),
                                 limit_cts=float(cts[0]), no_limit_cts=bool(cnone[0])))
    return rows


def selftest():
    rng = np.random.default_rng(0)
    sizes = rng.integers(3, 12, 40)
    sums = rng.binomial(sizes, 0.2).astype(float)
    est, ct, cts, none = closed(sizes, sums)
    n, m = sizes.sum(), len(sizes)
    e = sums - sums.sum() / n * sizes
    se = np.sqrt(m / (m - 1) * (e ** 2).sum()) / n
    tq = student.ppf(0.95, m - 1)
    assert abs(ct[0] - (sums.sum() / n + tq * se)) < 1e-12
    assert cts[0] >= ct[0] and not none[0]
    assert abs(cts[0] - ct[0] - max(0, (e ** 3).sum() / n ** 3 / se ** 2) * (2 * tq ** 2 + 1) / 6) < 1e-12
    _, z1, z2, zn = closed(sizes, np.zeros(40))
    assert zn[0] and z1[0] == 1.0 and z2[0] == 1.0
    # a left-skewed sample gets no correction
    left = sizes.astype(float).copy()
    left[:2] = 0
    _, a, b, _ = closed(sizes, left)
    assert abs(a[0] - b[0]) < 1e-12
    print("selftest ok")


def score(A, B):
    n_ct = sum(classify(v["miss_ct"], REPS) == "over" for v in A.values())
    n_cts = sum(classify(v["miss_cts"], REPS) == "over" for v in A.values())
    max_cts = max(v["miss_cts"] for v in A.values())
    passing = sorted(p for p, v in A.items() if v["real_cts"] <= TAU)
    lim = [r for r in B if not r["no_limit"]]
    exc = [r for r in lim if r["held_rate"] > r["limit"]]
    pipes = {r["pipeline"] for r in exc}
    share = len(exc) / len(lim) if lim else float("nan")
    return dict(
        C1=(n_ct >= 8, f"closed-form cluster-t over in {n_ct} of {len(A)} pipelines"),
        C2=(n_cts < n_ct and n_cts <= 4, f"skew-corrected over in {n_cts} of {len(A)} (plain {n_ct})"),
        C3=(max_cts <= 0.08, f"largest skew-corrected miss {max_cts:.4f}"),
        C4=(passing == [CERT], f"at or under {TAU} under the skew-corrected limit: {passing}"),
        C5=(0.020 <= A[CERT]["real_cts"] <= 0.030, f"{CERT}: skew-corrected {A[CERT]['real_cts']:.4f}, "
            f"plain {A[CERT]['real_ct']:.4f}, bootstrap-t {A[CERT]['real_boot_t']:.4f}"),
        L1=(share > 0.15, f"left-out rate above the limit in {len(exc)} of {len(lim)} task-and-pipeline pairs with a "
            f"limit ({share:.3f}); {len(B) - len(lim)} of {len(B)} had no limit"),
        L2=(len(pipes) >= 20, f"at least one such task in {len(pipes)} of {len(A)} pipelines"),
    )


def report(res):
    A, B, S = res["A"], res["B"], res["score"]
    se = np.sqrt(DELTA * (1 - DELTA) / res["reps"])
    L = ["# AgentDojo: closed-form limits by user task, and leaving one injection task out", "",
         "Registered in `.planning/paper-certification/Q1_registration.md`. Written by `scripts/agentdojo_closed.py`.", "",
         f"Level {DELTA}. Part A: {res['reps']:,} resamples of user tasks per pipeline (injection tasks fixed); over if "
         f"miss > {DELTA + 2 * se:.4f}. Part B: inner bootstrap {res['boots']:,}.", "",
         "## Registered predictions", "", "| | verdict | what was found |", "|---|---|---|"]
    for k, (ok, text) in S.items():
        L.append(f"| {k} | {'kept' if ok else 'REFUTED'} | {text} |")
    L += ["", "## A. Miss rates and real-data limits", "",
          "| pipeline | truth | tasks with a success | miss, closed-form t | miss, skew-corrected | no limit | real: closed-form t "
          "| real: skew-corrected | real: bootstrap-t (200,000) |", "|---|---|---|---|---|---|---|---|---|"]
    for p, v in A.items():
        f = lambda x: f"**{x:.4f}**" if classify(x, res["reps"]) == "over" else f"{x:.4f}"  # noqa: E731
        L.append(f"| {p} | {v['truth']:.3f} | {v['tasks_with_success']} of {v['m']} | {f(v['miss_ct'])} | {f(v['miss_cts'])} "
                 f"| {v['no_limit']:.4f} | {v['real_ct']:.4f} | {v['real_cts']:.4f} | {v['real_boot_t']:.4f} |")
    cls = lambda k: collections.Counter(classify(v[k], res["reps"]) for v in A.values())  # noqa: E731
    for k, name in (("miss_ct", "closed-form t"), ("miss_cts", "skew-corrected")):
        c = cls(k)
        L += ["", f"{name}: over {c['over']}, unresolved above delta {c['unresolved']}, at or under {c['under']}; "
              f"miss {min(v[k] for v in A.values()):.4f} to {max(v[k] for v in A.values()):.4f}; "
              f"largest no-limit share {max(v['no_limit'] for v in A.values()):.4f}."]
    L += ["", "## B. Leave one injection task out (within suite)", "",
          "| pipeline | tasks | no limit | left-out rate above the limit | largest left-out rate | above the skew-corrected closed form |",
          "|---|---|---|---|---|---|"]
    by = collections.defaultdict(list)
    for r in B:
        by[r["pipeline"]].append(r)
    for p, rs in by.items():
        lim = [r for r in rs if not r["no_limit"]]
        limc = [r for r in rs if not r["no_limit_cts"]]
        L.append(f"| {p} | {len(rs)} | {len(rs) - len(lim)} | {sum(r['held_rate'] > r['limit'] for r in lim)} of {len(lim)} "
                 f"| {max(r['held_rate'] for r in rs):.3f} | {sum(r['held_rate'] > r['limit_cts'] for r in limc)} of {len(limc)} |")
    limc = [r for r in B if not r["no_limit_cts"]]
    L += ["", f"Skew-corrected closed form, all pipelines: left-out rate above the limit in "
          f"{sum(r['held_rate'] > r['limit_cts'] for r in limc)} of {len(limc)} pairs with a limit.", "",
          f"### {CERT}, task by task", "", "| suite | left-out task | its rate | rate of the rest | limit from the rest |",
          "|---|---|---|---|---|"]
    for r in by[CERT]:
        L.append(f"| {r['suite']} | {r['task'].split('/')[1]} | {r['held_rate']:.3f} ({round(r['held_rate'] * r['held_n'])} of "
                 f"{r['held_n']}) | {r['rest_rate']:.3f} | {'no limit' if r['no_limit'] else format(r['limit'], '.3f')} |")
    L += ["", "The limit bounds the mean rate over the suite's other injection tasks. A single task's rate above it is "
          "not a miss of that limit; the count shows how far the limit is from a statement about a task it did not see."]
    open(OUT + ".md", "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--time", action="store_true", help="timings only; prints no outcome and writes nothing")
    ap.add_argument("--render", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.render:
        return report(json.load(open(OUT + ".json")))
    R_by, _ = AR.load_rows()
    tables = {p: AR.table(R) for p, R in R_by.items()}
    if a.time:
        t0 = time.time()
        part_a(dict(list(tables.items())[:2]), 2000)
        t1 = time.time()
        part_b(dict(list(R_by.items())[:1]), 500)
        print(f"pipelines {len(tables)}; part A 2 pipelines x 2000: {t1 - t0:.1f}s; part B 1 pipeline x 500 boots: {time.time() - t1:.1f}s")
        return
    t0 = time.time()
    A = part_a(tables, REPS)
    B = part_b(R_by, BOOTS)
    S = score(A, B)
    res = dict(reps=REPS, boots=BOOTS, big=BIG, seed=SEED, delta=DELTA, A=A, B=B,
               score={k: [bool(ok), text] for k, (ok, text) in S.items()}, seconds=time.time() - t0)
    json.dump(res, open(OUT + ".json", "w"), indent=1)
    report(res)
    print(open(OUT + ".md").read().split("## A.")[0])


if __name__ == "__main__":
    main()
