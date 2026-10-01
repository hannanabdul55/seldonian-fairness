"""Spike 017, stage C: what a harm constraint can certify, and with how many human positives.

Two ways to a certificate on the *human* harm rate of a candidate policy:

1. **Label the candidate's own responses.** Clopper-Pearson on n labels; the judge enters
   only through PPI++, whose gain is ``1 / (1 - rho^2)``. The table gives the labels needed
   to certify ``tau`` with probability 0.8, and the gain each judge would add at that rate.
2. **Carry a calibration** (recall from k human positives, false alarms from m human
   negatives, measured elsewhere) and correct the candidate's judged rate: Youden, or spike
   006's answer-rate-aware form where false alarms fall on answers only. Simulated in the
   best case, where the carried rates really are the candidate's (stage E says when they
   are not).

Judges at their measured operating points on the human sheet: the compiled rubric (recall
2/3, false alarms 0.216; 015), Qwen3Guard-4B (1/3, 0.230), and the gated two-stage label
(false alarms 0.015 on answers; recall unmeasured, so 0.5 and 0.8).

    ../../../.venv/bin/python harmcert017.py        # CPU, about 1 min; writes harmcert.md, harmcert.json
"""
import os

import numpy as np
from scipy.stats import binom

import cert017 as c

HERE = os.path.dirname(os.path.abspath(__file__))
DELTA, REPS, BIG_N, M_NEG = 0.05, 4000, 2000, 222
JUDGES = (("compiled rubric", 2 / 3, 0.216), ("Qwen3Guard-4B", 1 / 3, 0.230),
          ("gated, recall 0.5", 0.5, 0.015), ("gated, recall 0.8", 0.8, 0.015))


def n_needed(tau, r, power=0.8):
    """Smallest n with P(CP upper <= tau) >= power when K ~ Bin(n, r)."""
    for n in np.unique(np.round(np.geomspace(10, 200000, 900)).astype(int)):
        k = np.arange(0, int(n * tau) + 2)
        ok = c.cp_upper(k, n, DELTA) <= tau
        if ok.any() and binom.cdf(k[ok].max(), n, r) >= power:
            return int(n)
    return None


def carried(rng, sens, fa, k_pos, r_c, answer_rate=None):
    """Bounds on a candidate at true rate r_c from a calibration with k positives, M_NEG negatives."""
    tp = rng.binomial(k_pos, sens, REPS)
    if answer_rate is None:
        fp = rng.binomial(M_NEG, fa, REPS)
        q = sens * r_c + fa * (1 - r_c)
        f = (rng.random((REPS, BIG_N)) < q).astype(float)
        return c.youden(None, None, f, DELTA, cal=(tp, np.full(REPS, k_pos), fp, np.full(REPS, M_NEG)))
    m_ans = int(round(M_NEG * answer_rate))
    fp = rng.binomial(m_ans, fa, REPS)                       # false alarms among answered negatives
    ans = (rng.random((REPS, BIG_N)) < answer_rate).astype(float)
    q = sens * r_c + fa * (answer_rate - r_c)
    f = (rng.random((REPS, BIG_N)) < q).astype(float)
    return c.answer_aware(f, ans, DELTA, (tp, np.full(REPS, k_pos), fp, np.full(REPS, m_ans)))


def main():
    rng = np.random.default_rng(17)
    L = ["# Spike 017, stage C: what a harm constraint can certify", "",
         "## 1. Labelling the candidate's own responses", "",
         "Labels needed for the Clopper-Pearson bound to certify `tau` with probability 0.8 "
         "(delta 0.05), and the PPI++ gain `1 / (1 - rho^2)` each judge would add at that true "
         "rate (a gain of 1.02 saves 2% of the labels).", "",
         "| tau | true rate | labels needed | expected positives among them | "
         + " | ".join(f"gain: {j[0]}" for j in JUDGES) + " |",
         "|---|---|---|---|" + "---|" * len(JUDGES)]
    for tau in (0.01, 0.02, 0.05):
        for r in (0.0, tau / 4, tau / 2):
            n = n_needed(tau, r)
            g = [1 / (1 - c.rho2_binary(r, s, a)) if r > 0 else 1.0 for _, s, a in JUDGES]
            L.append(f"| {tau} | {r:.4f} | {n} | {n * r:.1f} | " + " | ".join(f"{x:.2f}" for x in g) + " |")
    L += ["", "The rule of three: with no positive among n labels the bound is `1 - 0.05^(1/n)`, "
          "about 3 / n: 0.0132 at the sheet's 225, 0.0100 at 299.", "",
          "## 2. Carrying a calibration with k human positives", "",
          f"Candidate: {BIG_N} judged responses at true harm rate `r`; calibration: k positives "
          f"and {M_NEG} negatives, the carried rates exactly right (best case). Median bound, "
          "and the share of draws that certify tau = 0.05. `Youden` for the ungated judges; "
          "`answer-aware` (answer rate 0.33, the screen's) for the gated label.", "",
          "| judge | route | true rate | k = 3 | k = 10 | k = 30 | k = 100 | k = 300 |",
          "|---|---|---|---|---|---|---|---|"]
    curves = []
    for name, sens, fa in JUDGES:
        gated = name.startswith("gated")
        for r_c in (0.0, 0.013):
            cells = []
            for k in (3, 10, 30, 100, 300):
                u = carried(rng, sens, fa, k, r_c, 0.33 if gated else None)
                cells.append(f"{np.median(u):.3f} ({(u <= 0.05).mean():.2f})")
                curves.append(dict(judge=name, rate=r_c, k=k, median=float(np.median(u)),
                                   certify05=float((u <= 0.05).mean())))
            L.append(f"| {name} | {'answer-aware' if gated else 'Youden'} | {r_c} | " + " | ".join(cells) + " |")
    L += ["", "Misses (bound below the true rate) over every cell above: see `max miss` printed "
          "by the script; all are valid by construction when the carried rates are right."]
    worst = 0.0
    for name, sens, fa in JUDGES:
        for k in (3, 30, 300):
            u = carried(rng, sens, fa, k, 0.013, 0.33 if name.startswith("gated") else None)
            worst = max(worst, float((u < 0.013).mean()))
    L[-1] = f"Largest miss rate (bound below the true rate 0.013) over judges and k: {worst:.3f}."
    import json
    json.dump(curves, open(os.path.join(HERE, "harmcert.json"), "w"), indent=1)
    open(os.path.join(HERE, "harmcert.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
