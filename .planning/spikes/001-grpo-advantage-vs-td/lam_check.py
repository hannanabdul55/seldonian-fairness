"""Spike 001 follow-up: in Seldonian runs, is the step-level |delta| series the multiplier?"""
import numpy as np
import tdlab

out = []
for p in (1.0, 4.0):
    cs, ns, ls = [], [], []
    for s in range(20):
        _, log = tdlab.run(s, pressure=p)
        d, at = tdlab.stack(log, "delta"), tdlab.stack(log, "adv_true")
        lam = tdlab.stack(log, "lam")
        noise = np.abs(d - at).mean(1)
        cs.append(np.corrcoef(np.abs(d).mean(1), lam)[0, 1])
        ns.append(np.corrcoef(noise, lam)[0, 1])
        ls.append(np.corrcoef(np.abs(at).mean(1), lam)[0, 1])
    out.append(f"| seldonian_lag p{p:g} | {np.nanmean(cs):.3f} | {np.nanmean(ns):.3f} | {np.nanmean(ls):.3f} |")
txt = ("\nPer-step correlation with the multiplier lambda (20 seeds):\n\n"
       "| setting | corr(mean abs delta, lambda) | corr(mean abs noise part, lambda) | corr(mean abs learnable part, lambda) |\n"
       "|---|---|---|---|\n" + "\n".join(out) + "\n")
open("results.md", "a").write(txt)
print(txt)
