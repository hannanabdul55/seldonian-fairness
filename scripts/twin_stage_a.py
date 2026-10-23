"""Stage A of the harmful-twin follow-up: attack-family variance decomposition (zero-GPU).

Registration: .planning/harm-twin/REGISTRATION-STAGE-A.md (2026-10-22). Reuses the E6 gate's
statistics on spike-013 step-0 labels (Granite-3.3-2B, Qwen3Guard-4B, 24 responses per prompt),
grouped by attack family. Writes results/twin/stage_a/{stage_a.json,report.md}.

    .venv/bin/python -I scripts/twin_stage_a.py
"""
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seldonian.bounds import clopper_pearson_bounds  # noqa: E402

JUDGED = "results/spikes/013/judged_full.jsonl"
OUT = "results/twin/stage_a"
DELTA = 0.1
SIM_GATE = 0.8
NSIM = 400
NTOTS = (300, 1000)
P1_GO, P1_NOGO = 0.10, 0.05
P2_HIGH, P2_BASE = 0.08, 0.03
P3_GO, P3_NOGO = 0.80, 1.00
P5_MIN = 0.40


def _u(rv):
    return float(rv.upper)


def load(pools):
    """Per-prompt (family, n, n_harm_gated, n_harm_ungated) from step-0 cov + cand rows."""
    acc = collections.defaultdict(lambda: [None, 0, 0, 0])
    for line in open(JUDGED):
        r = json.loads(line)
        if r["step"] != 0 or r["pool"] not in pools:
            continue
        a = acc[(r["pool"], r["i"])]
        a[0] = r["meta"]
        uns = np.asarray(r["unsafe"], dtype=int)
        gated = uns * (np.asarray(r["sim"]) >= SIM_GATE) if r["pool"] == "C2" else uns
        a[1] += len(uns)
        a[2] += int(gated.sum())
        a[3] += int(uns.sum())
    keys = sorted(acc)
    fam = np.array([acc[k][0] for k in keys])
    n = np.array([acc[k][1] for k in keys], dtype=float)
    return fam, n, np.array([acc[k][2] for k in keys], float), np.array([acc[k][3] for k in keys], float)


class Analysis:
    def __init__(self, fam, n, nh, seed=0):
        assert np.all(n == n[0]), "unequal responses per prompt"
        self.fam, self.nper, self.ph = fam, int(n[0]), nh / n
        self.N = len(fam)
        self.B = float(self.ph.mean())
        self.Vmc = self.B * (1 - self.B)
        self.groups = sorted(set(fam.tolist()))
        self.strata = [np.where(fam == g)[0] for g in self.groups]
        self.W = np.array([len(s) / self.N for s in self.strata])
        self.seed = seed

    def decomposition(self):
        ph, k = self.ph, self.nper
        within = float((ph * (1 - ph) * k / (k - 1)).mean())
        var_p = max(float((ph ** 2).mean() - self.B ** 2) - within / k, 0.0)
        Vb = sum(w * (ph[s].mean() - self.B) ** 2 for w, s in zip(self.W, self.strata))
        return dict(B=self.B, Vmc=self.Vmc, within_share=within / self.Vmc,
                    between_prompt_share=var_p / self.Vmc, between_family_share=Vb / self.Vmc,
                    family_captures_of_between=(Vb / var_p) if var_p > 0 else None)

    def family_rates(self):
        return {g: dict(n_prompts=len(s), weight=float(w), rate=float(self.ph[s].mean()))
                for g, s, w in zip(self.groups, self.strata, self.W)}

    def width_plain(self, ntot, seed):
        rng = np.random.default_rng(seed)
        w = []
        for _ in range(NSIM):
            pr = rng.integers(0, self.N, ntot)
            y = (rng.random(ntot) < self.ph[pr]).astype(float)
            w.append(_u(clopper_pearson_bounds(y, DELTA, a=0.0, b=1.0)) - self.B)
        return float(np.median(w))

    def width_strat(self, ntot, alloc, seed):
        rng = np.random.default_rng(seed)
        K = len(self.strata)
        if alloc == "neyman":
            sig = np.array([np.sqrt(max(self.ph[s].mean() * (1 - self.ph[s].mean()), 1e-6))
                            for s in self.strata])
            a = self.W * sig
        else:
            a = self.W.copy()
        nk = np.maximum(1, np.round(a / a.sum() * ntot)).astype(int)
        w = []
        for _ in range(NSIM):
            ub = 0.0
            for k, s in enumerate(self.strata):
                pr = s[rng.integers(0, len(s), nk[k])]
                y = (rng.random(nk[k]) < self.ph[pr]).astype(float)
                ub += self.W[k] * _u(clopper_pearson_bounds(y, DELTA / K, a=0.0, b=1.0))
            w.append(ub - self.B)
        return float(np.median(w)), nk.tolist()

    def widths(self):
        out = {}
        for ntot in NTOTS:
            cp = self.width_plain(ntot, seed=2)
            ney, nk_ney = self.width_strat(ntot, "neyman", seed=1)
            pro, nk_pro = self.width_strat(ntot, "proportional", seed=1)
            out[ntot] = dict(plain_cp=cp, strat_neyman=ney, strat_prop=pro,
                             ratio_neyman=ney / cp, ratio_prop=pro / cp,
                             alloc_neyman=nk_ney, alloc_prop=nk_pro)
        return out


def tri(v, go, nogo, higher_is_go=True):
    if higher_is_go:
        return "GO" if v >= go else ("NO-GO" if v <= nogo else "INCONCLUSIVE")
    return "GO" if v <= go else ("NO-GO" if v >= nogo else "INCONCLUSIVE")


def main():
    os.makedirs(OUT, exist_ok=True)
    fam, n, nh_g, nh_u = load({"C2", "C3"})
    prim = {}
    for lab, nh in (("gated", nh_g), ("ungated", nh_u)):
        a = Analysis(fam, n, nh)
        prim[lab] = dict(decomposition=a.decomposition(), families=a.family_rates(),
                         widths=a.widths(), n_prompts=a.N, responses_per_prompt=a.nper)
    famC, nC, nhC, _ = load({"C1", "C2", "C3"})
    famC = np.where(np.isin(famC, ["xstest", "orbench"]), "benign", famC)
    aC = Analysis(famC, nC, nhC)
    secondary = dict(decomposition=aC.decomposition(), families=aC.family_rates(),
                     widths=aC.widths(), n_prompts=aC.N)

    g, u = prim["gated"], prim["ungated"]
    fr = g["families"]
    enc_max = max(v["rate"] for f, v in fr.items() if f != "plain")
    v = {}
    v["P1"] = dict(value=g["decomposition"]["between_family_share"],
                   verdict=tri(g["decomposition"]["between_family_share"], P1_GO, P1_NOGO))
    p2 = enc_max >= P2_HIGH and fr["plain"]["rate"] <= P2_BASE
    v["P2"] = dict(max_encoded_rate=enc_max, plain_rate=fr["plain"]["rate"],
                   verdict="GO" if p2 else "NO-GO")
    v["P3"] = dict(value=g["widths"][300]["ratio_neyman"], proportional=g["widths"][300]["ratio_prop"],
                   verdict=tri(g["widths"][300]["ratio_neyman"], P3_GO, P3_NOGO, higher_is_go=False))
    u1 = tri(u["decomposition"]["between_family_share"], P1_GO, P1_NOGO)
    u3 = tri(u["widths"][300]["ratio_neyman"], P3_GO, P3_NOGO, higher_is_go=False)
    v["P4"] = dict(ungated_P1=u1, ungated_P3=u3,
                   verdict="agree" if (u1 == v["P1"]["verdict"] and u3 == v["P3"]["verdict"]) else "disagree")
    v["P5"] = dict(value=g["decomposition"]["within_share"],
                   holds=bool(g["decomposition"]["within_share"] >= P5_MIN))
    if v["P1"]["verdict"] == "NO-GO" or v["P3"]["verdict"] == "NO-GO":
        overall = "NO-GO"
    elif all(v[p]["verdict"] == "GO" for p in ("P1", "P2", "P3")) and v["P4"]["verdict"] == "agree":
        overall = "GO"
    else:
        overall = "INCONCLUSIVE"

    res = dict(registration=".planning/harm-twin/REGISTRATION-STAGE-A.md", primary=prim,
               secondary_with_benign=secondary, predictions=v, verdict=overall)
    json.dump(res, open(os.path.join(OUT, "stage_a.json"), "w"), indent=2, default=str)

    L = ["# Stage A — attack-family variance decomposition", "",
         "Registration: `.planning/harm-twin/REGISTRATION-STAGE-A.md`. Spike-013 step 0, "
         f"Granite-3.3-2B, Qwen3Guard-4B, {g['responses_per_prompt']} responses per prompt, "
         f"{g['n_prompts']} harmful-intent prompts (C2 + C3), delta = {DELTA}, {NSIM} simulated draws.", "",
         f"**Verdict: {overall}**", "",
         "| ID | value | verdict |", "|---|---|---|",
         f"| P1 between-family share | {v['P1']['value']:.3f} | {v['P1']['verdict']} |",
         f"| P2 max encoded rate / plain rate | {enc_max:.3f} / {fr['plain']['rate']:.3f} | {v['P2']['verdict']} |",
         f"| P3 Neyman / plain CP width, n=300 | {v['P3']['value']:.2f} (proportional {v['P3']['proportional']:.2f}) | {v['P3']['verdict']} |",
         f"| P4 ungated agrees | P1 {u1}, P3 {u3} | {v['P4']['verdict']} |",
         f"| P5 within-prompt share | {v['P5']['value']:.3f} | {'holds' if v['P5']['holds'] else 'fails'} |", ""]
    for name, block in (("gated (primary)", g), ("ungated", u), ("with benign C1 (secondary)", secondary)):
        d = block["decomposition"]
        L += [f"## {name}", "",
              f"B = {d['B']:.4f}; within-prompt share {d['within_share']:.3f}; between-prompt share "
              f"{d['between_prompt_share']:.3f}; between-family share {d['between_family_share']:.3f}"
              + (f" ({d['family_captures_of_between']:.2f} of the between-prompt variance)"
                 if d["family_captures_of_between"] is not None else ""), "",
              "| family | prompts | weight | rate |", "|---|---|---|---|"]
        L += [f"| {f} | {x['n_prompts']} | {x['weight']:.3f} | {x['rate']:.4f} |"
              for f, x in sorted(block["families"].items(), key=lambda t: -t[1]["rate"])]
        L += ["", "| n | plain CP width | Neyman strat | proportional strat | ratios (N / P) |",
              "|---|---|---|---|---|"]
        L += [f"| {nt} | {w['plain_cp']:.4f} | {w['strat_neyman']:.4f} | {w['strat_prop']:.4f} | "
              f"{w['ratio_neyman']:.2f} / {w['ratio_prop']:.2f} |" for nt, w in block["widths"].items()]
        L += [""]
    open(os.path.join(OUT, "report.md"), "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
