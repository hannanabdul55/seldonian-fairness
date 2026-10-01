"""Spike 017: assemble results.md from the stage outputs.

    ../../../.venv/bin/python report017.py
"""
import json
import os

import cert017 as c

HERE = c.HERE
ROUTES = [("classical", "-", "labels alone, Clopper-Pearson (exact)"),
          ("naive", "f01", "judge alone, 0/1"),
          ("youden", "f01", "Youden, recall and false alarms from the same labels"),
          ("exact3", "f01", "PPI, three exact limits"),
          ("strat", "f01", "post-stratified on the 0/1 judge, exact"),
          ("block", "f01", "block PPI, betting, 0/1 judge (finite-sample)"),
          ("block", "logit", "block PPI, betting, logit (finite-sample)"),
          ("ppi", "p", "PPI, normal limit, p (`E[p]` plus rectifier)"),
          ("ppi++", "f01", "PPI++, normal limit, 0/1 judge"),
          ("ppi++", "p", "PPI++, normal limit, p"),
          ("ppi++", "logit", "PPI++, normal limit, logit"),
          ("ppi++", "platt", "PPI++, normal limit, cross-fitted Platt"),
          ("ppi++w", "logit", "PPI++, score (Wilson-type) limit, logit"),
          ("boot", "f01", "PPI++, bootstrap-t, 0/1 judge"),
          ("boot", "p", "PPI++, bootstrap-t, p"),
          ("boot", "logit", "PPI++, bootstrap-t, logit"),
          ("boot", "platt", "PPI++, bootstrap-t, cross-fitted Platt")]


def cells(rows, **kw):
    return {(r["method"], r["feat"]): r for r in rows if all(r[k] == v for k, v in kw.items())}


def mb(r):
    return f"{r['miss']:.3f} ({r['bound']:.3f})"


def ess(cl, r):
    """Effective-label gain read off the mean bounds: (classical excess / route excess)^2."""
    a, b = cl["bound"] - cl["rate"], r["bound"] - r["rate"]
    return (a / b) ** 2 if b > 0 else float("nan")


def main():
    d = json.load(open(os.path.join(HERE, "plasmode.json")))
    s = json.load(open(os.path.join(HERE, "plasmode_shift.json")))
    rows, rho = d["rows"], d["rho2"]
    L = ["# Spike 017 results", "",
         "Miss = share of draws whose upper bound falls below the truth (target <= 0.05); the mean "
         "bound is in brackets. 4,000 draws a cell unless stated (block PPI 1,000).", "",
         "## B1. Every route on the canonical compiled refusal judge", "",
         "015's 500 scored responses, gold rate 0.200; N = 2,000 judged, n labelled at random.", "",
         "| route | n = 50 | n = 100 | n = 225 | n = 500 |", "|---|---|---|---|---|"]
    by_n = {n: cells(rows, task="refusal", variant="rubric", wording=0, n=n, N=2000) for n in (50, 100, 225, 500)}
    for m, f, name in ROUTES:
        L.append(f"| {name} | " + " | ".join(mb(by_n[n][(m, f)]) for n in (50, 100, 225, 500)) + " |")

    L += ["", "## B2. The judge's worth in labels: formula against measurement", "",
          "`formula` = 1 / (1 - rho^2 Nu / (Nu + n)) from the pool's rho^2; `variance` = measured "
          "variance ratio of the PPI++ estimate to the labels-alone mean; `bound` = the same read "
          "off the bootstrap-t bound's mean excess over the truth, against Clopper-Pearson's.", "",
          "| judge | feature | rho^2 | n | formula | variance | bound |", "|---|---|---|---|---|---|---|"]
    for variant in ("rubric", "raw"):
        for feat in ("f01", "p", "logit", "platt"):
            for n in (100, 225, 500):
                cc = cells(rows, task="refusal", variant=variant, wording=0, n=n, N=2000)
                if not cc:
                    continue
                r2 = rho[f"refusal|{variant}|0"][feat]
                L.append(f"| refusal, {variant} | {feat} | {r2:.3f} | {n} | {c.gain_ppipp(r2, n, 2000 - n):.2f} "
                         f"| {cc[('ppi++', feat)]['gain']:.2f} | {ess(cc[('classical', '-')], cc[('boot', feat)]):.2f} |")

    L += ["", "## B3. Wording changes the width, not the certified quantity", "",
          "n = 225, N = 2,000. `judged rate` is what the uncorrected constraint would measure.", "",
          "| constraint | wording | judged rate | rho^2 (logit) | PPI++ estimate (truth) | bootstrap-t: miss (bound) "
          "| labels alone | plain PPI's worth in labels | PPI++ lam |", "|---|---|---|---|---|---|---|---|---|"]
    for task in ("refusal", "brevity"):
        for variant, w in [("rubric", i) for i in range(6)] + [("raw", 0)]:
            cc = cells(rows, task=task, variant=variant, wording=w, n=225, N=2000)
            cl = cc[("classical", "-")]
            L.append(f"| {task} | {variant} {w} | "
                     f"{judged(task, variant, w):.3f} | {rho[f'{task}|{variant}|{w}']['logit']:.3f} "
                     f"| {cc[('ppi++', 'logit')]['est']:.3f} ({cl['rate']:.3f}) | {mb(cc[('boot', 'logit')])} "
                     f"| {mb(cl)} | {cc[('ppi', 'logit')]['gain']:.2f} | {cc[('ppi++', 'logit')]['lam']:.2f} |")

    L += ["", "## B4. The same judge at rarer rates", "",
          "Prevalence-shifted draws from the refusal pool (N = 4,000).", "",
          "| judge | rate | n | labels alone | PPI++ normal, logit | PPI++ bootstrap-t, logit "
          "| post-stratified, exact | block PPI, logit |", "|---|---|---|---|---|---|---|---|"]
    for variant in ("rubric", "raw"):
        for rate in (0.2, 0.05, 0.013):
            for n in (225, 1000):
                cc = cells(s["rows"], variant=variant, rate=rate, n=n)
                L.append(f"| {variant} | {rate} | {n} | {mb(cc[('classical', '-')])} | {mb(cc[('ppi++', 'logit')])} "
                         f"| {mb(cc[('boot', 'logit')])} | {mb(cc[('strat', 'f01')])} | {mb(cc[('block', 'logit')])} |")

    L += ["", "## B5. A large unlabelled set (N = 20,000; 1,000 draws, block 500)", "",
          "| judge | n | labels alone | PPI++ bootstrap-t, logit | PPI++ bootstrap-t, Platt | block PPI, 0/1 "
          "| block PPI, logit |", "|---|---|---|---|---|---|---|"]
    for variant in ("rubric", "raw"):
        for n in (225, 1000):
            cc = cells(rows, task="refusal", variant=variant, wording=0, n=n, N=20000)
            if cc:
                L.append(f"| {variant} | {n} | {mb(cc[('classical', '-')])} | {mb(cc[('boot', 'logit')])} "
                         f"| {mb(cc[('boot', 'platt')])} | {mb(cc[('block', 'f01')])} | {mb(cc[('block', 'logit')])} |")

    worst = {}
    for r in rows + s["rows"]:
        k = (r["method"], r["feat"])
        worst[k] = max(worst.get(k, 0.0), r["miss"])
    L += ["", "## B6. Largest miss of each route over every cell above", "",
          "| route | largest miss |", "|---|---|"]
    for m, f, name in ROUTES:
        L.append(f"| {name} | {worst[(m, f)]:.3f} |")

    for title, fname in (("Stage A: the sheet's sampling design", "design.md"),
                         ("Routing rule", "route.md"), ("Stage C", "harmcert.md"),
                         ("Stage D", "harm.md"), ("Stage E", "transfer.md"),
                         ("The certificates", "cards.md"), ("Checks on known truth", "check_cert.md")):
        path = os.path.join(HERE, fname)
        if os.path.exists(path):
            body = open(path).read().strip().split("\n")
            L += ["", "---", "", f"<!-- {fname} -->"] + ["#" + ln if ln.startswith("#") else ln for ln in body]
    open(os.path.join(HERE, "results.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[:95]))


_POOLS = None


def judged(task, variant, w):
    global _POOLS
    if _POOLS is None:
        import plasmode017 as pm
        _POOLS = pm.load_pools()
    return float((_POOLS[(task, variant, w)][1] > 0.5).mean())


if __name__ == "__main__":
    main()
