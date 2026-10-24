"""Per-family guarantee, rule F (finish one family before the next) vs rule A (APT) vs uniform.

Registration: .planning/harm-twin/REGISTRATION-PER-FAMILY-F.md (2026-10-23). Same plasmode, decision
rule and bounds as scripts/per_family_cert.py; adds families resolved at a fixed label budget.
Writes results/per_family_f/{per_family_f.json,report.md}.

    .venv/bin/python -I scripts/per_family_rule_f.py [--smoke]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import per_family_cert as P  # noqa: E402
from twin_stage_a import load  # noqa: E402

OUT = "results/per_family_f"
BUDGETS = (300, 600, 1000, 1500, 2500)
PRIMARY = ("gated", 0.05)
N_GO, N_NOGO = 1.10, 1.50
M_STEP = 0.002
NSTAR_CAP = 10 ** 6


def nstar_table(tau, eps, c):
    """n*(m) on a grid of m: labels until a family whose rate stays at m is resolved."""
    m = np.arange(0, 1 + M_STEP / 2, M_STEP)[:, None]
    t = np.unique(np.concatenate([np.arange(1, 20001), np.geomspace(20001, NSTAR_CAP, 400).astype(int)]))[None, :]
    s = m * t
    lo, hi = max(tau - eps, 1e-9), min(tau + eps, 1 - 1e-9)
    ok = P._logM(s, t, tau) >= c
    ok |= (m > lo) & (m < hi) & (P._logM(s, t, lo) >= c) & (P._logM(s, t, hi) >= c)
    first = np.where(ok.any(1), t[0, ok.argmax(1)], NSTAR_CAP)
    first[np.abs(m[:, 0] - tau) < 1e-12] = NSTAR_CAP
    return first


def sequential(pool, tau, rule, seed, nrun):
    rng = np.random.default_rng(seed)
    K, eps = pool.K, P.EPS_FRAC * tau
    c = np.log(K / P.DELTA)
    lo, hi = max(tau - eps, 1e-9), min(tau + eps, 1 - 1e-9)
    in_zone = (pool.B >= tau - eps) & (pool.B <= tau + eps)
    nstar = nstar_table(tau, eps, c) if rule == "F" else None
    T = np.zeros((nrun, K))
    S = np.zeros((nrun, K))
    d = np.full((nrun, K), P.OPEN)
    used = np.zeros(nrun, int)
    at_budget = {b: np.full(nrun, -1) for b in BUDGETS}

    def draw(r, k):
        off = (rng.random(len(r)) * pool.sizes[k]).astype(int)
        return rng.random(len(r)) < pool.ph[pool.flat[pool.starts[k] + off]]

    def update(r, k):
        t, s = T[r, k], S[r, k]
        mean = s / t
        excl = P._logM(s, t, tau) >= c
        indiff = (in_zone[k] & (P._logM(s, t, lo) >= c) & (mean > lo)
                  & (P._logM(s, t, hi) >= c) & (mean < hi))
        new = np.where(excl & (mean < tau), P.SAFE, np.where(excl & (mean > tau), P.UNSAFE,
                                                             np.where(indiff, P.INDIFF, P.OPEN)))
        d[r, k] = np.where(d[r, k] == P.OPEN, new, d[r, k])

    allr = np.arange(nrun)
    for k in range(K):
        for _ in range(P.BURN):
            kk = np.full(nrun, k)
            T[allr, kk] += 1
            S[allr, kk] += draw(allr, kk)
            used += 1
            for b in BUDGETS:
                hit = used == b
                at_budget[b][hit] = (d[hit] != P.OPEN).sum(1)
        update(allr, np.full(nrun, k))
    active = (d == P.OPEN).any(1)
    stop = np.where(active, -1, used)
    while active.any():
        r = np.where(active)[0]
        mean = S[r] / T[r]
        if rule == "A":
            idx = np.sqrt(T[r]) * (np.abs(mean - tau) + eps)
        else:
            R = np.maximum(nstar[np.rint(mean / M_STEP).astype(int)] - T[r], 1)
            idx = R - T[r] / 2e5 + np.arange(K)[None, :] / 1e9
        idx = np.where(d[r] == P.OPEN, idx, np.inf)
        k = idx.argmin(1)
        T[r, k] += 1
        S[r, k] += draw(r, k)
        update(r, k)
        used[r] += 1
        for b in BUDGETS:
            hit = used[r] == b
            at_budget[b][r[hit]] = (d[r[hit]] != P.OPEN).sum(1)
        done = ~(d[r] == P.OPEN).any(1)
        stop[r[done]] = used[r[done]]
        active[r[done | (used[r] >= P.CAP)]] = False
    for b in BUDGETS:
        at_budget[b][at_budget[b] < 0] = K   # resolved everything before reaching this budget
    resolved = stop >= 0
    return dict(p_resolve=float(resolved.mean()),
                p_mis=float(P.misclassified(d, tau, pool.B).any(1).mean()),
                n80=(float(np.percentile(np.where(resolved, used, np.inf), 80))
                     if resolved.mean() >= 0.8 else None),
                resolved_at_budget={b: float(at_budget[b].mean()) for b in BUDGETS},
                median_labels_per_family=np.median(T, 0).tolist())


def uniform_at_budgets(pool, tau, seed, nrun):
    rng = np.random.default_rng(seed)
    a, eps = P.DELTA / pool.K, P.EPS_FRAC * tau
    out = {}
    for b in BUDGETS:
        nk = rng.multinomial(b, pool.W, size=nrun)
        s = np.array([[pool.draw(rng, k, nk[r, k]) for k in range(pool.K)] for r in range(nrun)])
        d = P.decide(P.cp_upper(s, nk, a), P.cp_lower(s, nk, a), tau, eps, pool.B)
        out[b] = dict(resolved=float((d != P.OPEN).sum(1).mean()),
                      p_mis=float(P.misclassified(d, tau, pool.B).any(1).mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny run; prints timings and key names only")
    args = ap.parse_args()
    nrun = 20 if args.smoke else P.NRUN
    fam, n, nh_g, nh_u = load({"C2", "C3"})
    cells = {}
    t0 = time.time()
    for lab, nh in (("gated", nh_g), ("ungated", nh_u)):
        pool = P.Pool(fam, nh / n)
        for tau in P.TAUS:
            if args.smoke and (lab, tau) != PRIMARY:
                continue
            cells[f"{lab}|{tau}"] = dict(
                label=lab, tau=tau, families=pool.groups, B=pool.B.tolist(),
                F=sequential(pool, tau, "F", seed=21, nrun=nrun),
                A=sequential(pool, tau, "A", seed=22, nrun=nrun),
                U=uniform_at_budgets(pool, tau, seed=23, nrun=nrun))
            if not args.smoke:
                print(f"{lab} tau={tau} done at {time.time() - t0:.0f}s", flush=True)
    if args.smoke:
        print(f"smoke ok in {time.time() - t0:.1f}s; cell keys {list(cells)}; "
              f"design keys {list(cells[next(iter(cells))])}")
        return
    os.makedirs(OUT, exist_ok=True)

    p = cells[f"{PRIMARY[0]}|{PRIMARY[1]}"]
    V = all(c["F"]["p_mis"] <= P.DELTA for c in cells.values())
    g600 = p["F"]["resolved_at_budget"][600] >= p["U"][600]["resolved"]
    g1000 = p["F"]["resolved_at_budget"][1000] >= p["U"][1000]["resolved"]
    vG = "GO" if (g600 and g1000) else ("NO-GO" if not g1000 else "INCONCLUSIVE")
    nF, nA = p["F"]["n80"], p["A"]["n80"]
    rN = nF / nA if (nF and nA) else None
    vN = "NO-GO" if rN is None else ("GO" if rN <= N_GO else ("NO-GO" if rN > N_NOGO else "INCONCLUSIVE"))
    verdict = "NO-GO" if (not V or "NO-GO" in (vG, vN)) else ("GO" if vG == vN == "GO" else "INCONCLUSIVE")
    preds = dict(V=dict(holds=V, worst=max(c["F"]["p_mis"] for c in cells.values())),
                 G=dict(F_600=p["F"]["resolved_at_budget"][600], U_600=p["U"][600]["resolved"],
                        F_1000=p["F"]["resolved_at_budget"][1000], U_1000=p["U"][1000]["resolved"],
                        verdict=vG),
                 N=dict(n80_F=nF, n80_A=nA, ratio=rN, verdict=vN))
    json.dump(dict(registration=".planning/harm-twin/REGISTRATION-PER-FAMILY-F.md", cells=cells,
                   predictions=preds, verdict=verdict),
              open(os.path.join(OUT, "per_family_f.json"), "w"), indent=2, default=str)

    def fmt(x, nd=2):
        return "-" if x is None else f"{x:.{nd}f}"

    L = ["# Per-family guarantee, rule F (finish one family at a time)", "",
         "Registration: `.planning/harm-twin/REGISTRATION-PER-FAMILY-F.md`. Plasmode on judge labels, "
         f"{P.NRUN} runs per rule and cell.", "", f"**Verdict: {verdict}**", "",
         "| ID | value | verdict |", "|---|---|---|",
         f"| V F misclassifies <= {P.DELTA} | worst {preds['V']['worst']:.3f} | {'holds' if V else 'FAILS'} |",
         f"| G F vs U families resolved at 600 / 1,000 | {preds['G']['F_600']:.1f} vs {preds['G']['U_600']:.1f} / "
         f"{preds['G']['F_1000']:.1f} vs {preds['G']['U_1000']:.1f} | {vG} |",
         f"| N n80(F) / n80(A) | {fmt(nF, 0)} / {fmt(nA, 0)} = {fmt(rN)} | {vN} |", "",
         "## Families resolved (of 7) when labelling stops at a budget", "",
         "| label | tau | rule | " + " | ".join(str(b) for b in BUDGETS) + " | n80 | P(mis) |",
         "|---|---|---|" + "---|" * len(BUDGETS) + "---|---|"]
    for c in cells.values():
        for rule in ("F", "A"):
            x = c[rule]
            L.append(f"| {c['label']} | {c['tau']} | {rule} | "
                     + " | ".join(f"{x['resolved_at_budget'][b]:.1f}" for b in BUDGETS)
                     + f" | {fmt(x['n80'], 0)} | {x['p_mis']:.3f} |")
        L.append(f"| {c['label']} | {c['tau']} | U | "
                 + " | ".join(f"{c['U'][b]['resolved']:.1f}" for b in BUDGETS)
                 + f" | (fixed n) | {max(v['p_mis'] for v in c['U'].values()):.3f} |")
    open(os.path.join(OUT, "report.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
