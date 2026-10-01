"""Spike 014 stage 3: the real-data moderators, the three predictions, and H3/H4.

Reads 013's reference labels (``results/spikes/013/judged_full.jsonl``: 8 ``cov`` samples per
C1 prompt at step 0, and 013's own side-effect checkpoints at steps 100/200), 014's judged
checkpoints (``results/spikes/014/judged_<tag>.jsonl``), the training record
(``run_<tag>.json``), the bandit gate (``gate.json``) and the plasmode (``plasmode.json``).

Writes ``results.md``: the rate / ICC / rho trajectory for the pushed run beside 013's
side-effect run, the realised ESS of 013's final rule (S2, k 8, H 8, ``b1w``) with its miss
rate, and the three predictions of it: the bandit-calibrated one (the gate's cells whose
compression matches the real checkpoint, their rho carried over), the formula at the
measured real moderators, and the naive pre-flight (ICC_ref, rho 0.8). Then H3 and H4.

    ../../../.venv/bin/python analyse014.py --tag s0 --plasmode plasmode.json
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "013-stratified-safety-set"))
from plasmode import C_H, icc_from_samples, reliability  # noqa: E402

K_REF, H_RULE, N_S, BOUND = 8, 8, 200, "b1w"


def load_labels(tag):
    cov, side, cand = {}, defaultdict(dict), defaultdict(dict)
    for r in map(json.loads, open(os.path.join(REPO, "results", "spikes", "013", "judged_full.jsonl"))):
        if r["pool"] != "C1":
            continue
        if r["step"] == 0 and r["role"] == "cov":
            cov[r["i"]] = r["refusal"][:K_REF]
        elif r["role"] == "cand":
            side[r["step"]][r["i"]] = r["refusal"]
    for r in map(json.loads, open(os.path.join(REPO, "results", "spikes", "014", f"judged_{tag}.jsonl"))):
        cand[r["step"]][r["i"]] = r["refusal"]
    ids = sorted(cov)
    Y0 = np.array([cov[i] for i in ids], dtype=float)
    S = {s: np.array([side[s][i] for i in ids], dtype=float) for s in sorted(side)}
    C = {s: np.array([cand[s][i] for i in ids], dtype=float) for s in sorted(cand)}
    return Y0, S, C


def moderators(Y0, Yc):
    """ICC_ref, ICC_cand (all K samples), rho disattenuated for both sides' sampling noise."""
    icc_ref, icc_cand = icc_from_samples(Y0), icc_from_samples(Yc)
    r_obs = np.corrcoef(Y0.mean(axis=1), Yc.mean(axis=1))[0, 1]
    rel = reliability(icc_ref, Y0.shape[1]) * reliability(icc_cand, Yc.shape[1])
    rho = float(np.clip(r_obs / np.sqrt(max(rel, 1e-12)), -1, 1))
    g = icc_cand * rho ** 2 * reliability(icc_ref, K_REF) * C_H[H_RULE]
    g_naive = icc_ref * 0.8 ** 2 * reliability(icc_ref, K_REF) * C_H[H_RULE]
    return dict(rate=float(Yc.mean()), K=int(Yc.shape[1]), icc_ref=float(icc_ref), icc_cand=float(icc_cand),
                ratio=float(icc_cand / icc_ref), r_obs=float(r_obs), rho=rho, G=float(g),
                pred_meas=float(1 / (1 - g)), pred_naive=float(1 / (1 - g_naive)))


def bandit_prediction(gate, ratio, icc_ref, tol=0.15):
    """The gate's cells whose compression is within `tol` of the real one; their rho carried."""
    cells = [c for c in gate["gate"] if abs(c["icc_ratio"] - ratio) <= tol]
    if not cells:
        near = min(gate["gate"], key=lambda c: abs(c["icc_ratio"] - ratio))
        cells, note = [near], f"no cell within {tol}; nearest ratio {near['icc_ratio']:.2f}"
    else:
        note = f"{len(cells)} cells within {tol}"
    rho = float(np.mean([c["rho"] for c in cells]))
    g = icc_ref * ratio * rho ** 2 * reliability(icc_ref, K_REF) * C_H[H_RULE]
    return dict(rho=rho, pred=float(1 / (1 - g)), cells=[(c["env"], c["pressure"], c["margin"]) for c in cells],
                bandit_ess=float(np.mean([c["ess"] for c in cells])), note=note)


def realised(plas, env, cand, delta, k=K_REF, H=H_RULE, n_s=N_S, bound=BOUND):
    """Realised ESS of S2 (k, H) against R at the same n_s, delta, bound; both miss rates."""
    def pick(arm, kk, HH):
        rows = [r for r in plas if r["env"] == env and r["cand"] == cand and r["arm"] == arm and r["k"] == kk
                and r["H"] == HH and r["n_s"] == n_s and r["delta"] == delta and r["bound"] == bound]
        if not rows:
            return None
        return dict(miss=np.mean([r["miss"] for r in rows]), width=np.mean([r["width"] for r in rows]),
                    reps=sum(r["reps"] for r in rows), pf_ess=np.mean([r["pf_ess_pred"] for r in rows]),
                    pf_rho=np.mean([r["pf_rho"] for r in rows]), pf_icc_cand=np.mean([r["pf_icc_cand"] for r in rows]))
    R, S2 = pick("R", 8, 4), pick("S2", k, H)
    if R is None or S2 is None:
        return None
    se = np.sqrt(delta * (1 - delta) / S2["reps"])
    return dict(ess=(R["width"] / S2["width"]) ** 2, miss_R=R["miss"], miss_S2=S2["miss"],
                valid=S2["miss"] <= delta + 2 * se, width_R=R["width"], width_S2=S2["width"],
                pf_ess=S2["pf_ess"], pf_rho=S2["pf_rho"], pf_icc_cand=S2["pf_icc_cand"], reps=S2["reps"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s0")
    ap.add_argument("--plasmode", default="plasmode.json")
    ap.add_argument("--out", default="results.md")
    a = ap.parse_args()
    Y0, S, C = load_labels(a.tag)
    run = json.load(open(os.path.join(REPO, "results", "spikes", "014", f"run_{a.tag}.json")))
    gate = json.load(open(os.path.join(HERE, "gate.json")))
    plas = json.load(open(os.path.join(HERE, a.plasmode)))

    L = ["# Spike 014 stage 2-3: Granite over-refusal under the Lagrangian", ""]
    # 1. trajectory
    L += ["## Rate, ICC and rho across the run", "",
          f"Reference (013's 8 `cov` samples per C1 prompt): rate {Y0.mean():.3f}, ICC_ref "
          f"{icc_from_samples(Y0):.2f}. Threshold {run['threshold']:.3f}. Safety test on D_s (one "
          f"response per prompt, n {run['safety_test']['n']['refusal']}): rate "
          f"{run['safety_test']['rates']['refusal']:.3f}, Clopper-Pearson upper "
          f"{run['safety_test']['upper']['refusal']:.3f}, passed {run['safety_test']['passed']}; "
          f"selected step {run['selected']['step']} ({run['selected']['reason']}).", "",
          "Predicted-test trajectory during training (D_c subsample, `predict_every` steps):", "",
          "| step | rate | upper | feasible | lambda | reward |", "|---|---|---|---|---|---|"]
    for h in run["history"]:
        L.append(f"| {h['step']} | {h['rates']['refusal']:.3f} | {h['upper']['refusal']:.3f} | {h['feasible']} "
                 f"| {h['lambdas']['refusal']:.1f} | {h['reward']:.2f} |")
    L += ["", "Checkpoints sampled on the pool (all K samples; rho disattenuated for both sides' "
          "sampling noise, as 013's pre-flight does):", "",
          "| run | step | K | rate | moved | ICC_cand | ICC_cand/ICC_ref | r_obs | rho | pred meas | pred naive |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    mods = {}
    for name, D in (("014 pushed (Lagrangian, C4)", C), ("013 side-effect (GRPO, C1)", S)):
        for s, Y in D.items():
            m = moderators(Y0, Y)
            mods[(name, s)] = m
            L.append(f"| {name} | {s} | {m['K']} | {m['rate']:.3f} | {m['rate'] - Y0.mean():+.3f} | {m['icc_cand']:.2f} "
                     f"| {m['ratio']:.2f} | {m['r_obs']:.2f} | {m['rho']:.2f} | {m['pred_meas']:.2f} | {m['pred_naive']:.2f} |")

    # 2. realised ESS and the three predictions
    L += ["", f"## Realised ESS of 013's rule (S2, k {K_REF}, H {H_RULE}, n_s {N_S}, `{BOUND}`) and the predictions", "",
          "`bandit` = the gate's cells whose ICC_cand/ICC_ref is within 0.15 of the real checkpoint's, their rho "
          "carried into the formula at the real ICC_ref; `meas` = the formula at the real checkpoint's measured "
          "ICC_cand and rho (all K samples); `naive` = ICC_ref, rho 0.8. Miss against delta; valid = miss <= "
          "delta + 2 MC se.", "",
          "| run | step | delta | miss R | miss S2 | valid | width R | width S2 | realised ESS | bandit (rho, cells) | meas | naive |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    score = {}
    for (name, s), m in mods.items():
        role = "cand" if name.startswith("014") else "side"
        bp = bandit_prediction(gate, m["ratio"], m["icc_ref"])
        for d in (0.05, 0.1):
            r = realised(plas, f"C1:refusal:{role}", f"step{s}", d)
            if r is None:
                continue
            score[(role, s, d)] = dict(r, bandit=bp["pred"], meas=m["pred_meas"], naive=m["pred_naive"])
            L.append(f"| {name} | {s} | {d} | {r['miss_R']:.3f} | {r['miss_S2']:.3f} | {r['valid']} | {r['width_R']:.4f} "
                     f"| {r['width_S2']:.4f} | **{r['ess']:.2f}** | {bp['pred']:.2f} ({bp['rho']:.2f}, {bp['note']}) "
                     f"| {m['pred_meas']:.2f} | {m['pred_naive']:.2f} |")

    # 3. H3 / H4
    final = max(C)
    m = mods[("014 pushed (Lagrangian, C4)", final)]
    bp = bandit_prediction(gate, m["ratio"], m["icc_ref"])
    matched = [c for c in gate["cells"] if c["method"] == "seldonian_lag" and c["eta"] == 400 and c["steps"] == 200
               and abs(c["icc_ratio"] - m["ratio"]) <= 0.15 and c["feasible"] >= 0.5]
    L += ["", "## H3 and H4", ""]
    if matched:
        ess_m = [c["ess@0.1"] for c in matched]
        miss_m = [c["miss_strat_pool@0.1"] for c in matched]
        n_seeds = min(c["n"] for c in matched)
        h3 = min(ess_m) >= 1.2 and max(miss_m) <= 0.1 + 2 * np.sqrt(0.09 / n_seeds)
        L.append(f"- **H3** (bandit cells matching the real compression {m['ratio']:.2f} +- 0.15, constrained, eta 400, "
                 f"200 steps: {len(matched)} cells): realised bandit ESS {min(ess_m):.2f}-{max(ess_m):.2f}, "
                 f"stratified miss {min(miss_m):.3f}-{max(miss_m):.3f} at delta 0.1. H3 asks ESS >= 1.2 and valid: "
                 f"**{'holds' if h3 else 'fails'}**.")
    else:
        L.append(f"- **H3**: no bandit cell within 0.15 of the real compression {m['ratio']:.2f} "
                 f"(the bandit's ratios run {min(c['icc_ratio'] for c in gate['cells']):.2f}-"
                 f"{max(c['icc_ratio'] for c in gate['cells']):.2f}); scored on the nearest cell below.")
    for d in (0.1, 0.05):
        r = score.get(("cand", final, d))
        if r is None:
            continue
        err = abs(r["ess"] - r["bandit"])
        h4 = err <= 0.3 and r["valid"]
        L.append(f"- **H4** at delta {d}: realised ESS {r['ess']:.2f} at step {final}, bandit-calibrated prediction "
                 f"{r['bandit']:.2f} (|error| {err:.2f}; asks <= 0.3), `b1w` miss {r['miss_S2']:.3f} "
                 f"({'valid' if r['valid'] else 'INVALID'}): **{'holds' if h4 else 'fails'}**. "
                 f"Formula at the measured moderators {r['meas']:.2f} (|error| {abs(r['ess'] - r['meas']):.2f}); "
                 f"naive {r['naive']:.2f} (|error| {abs(r['ess'] - r['naive']):.2f}).")
    side_final = max(S)
    rs, rc = score.get(("side", side_final, 0.1)), score.get(("cand", final, 0.1))
    if rs and rc:
        L.append(f"- Like for like at delta 0.1, step {final}: pushed ESS {rc['ess']:.2f} against 013's side-effect "
                 f"ESS {rs['ess']:.2f} on the same prompts and reference labels.")
    json.dump(dict(moderators={f"{k[0]}|{k[1]}": v for k, v in mods.items()},
                   score={f"{k[0]}|{k[1]}|{k[2]}": v for k, v in score.items()},
                   bandit_final=bp, matched=len(matched)),
              open(os.path.join(HERE, "stage3.json"), "w"), indent=1, default=float)
    open(os.path.join(HERE, a.out), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
