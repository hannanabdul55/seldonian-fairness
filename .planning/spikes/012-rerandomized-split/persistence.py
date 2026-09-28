"""Do per-prompt rates survive training? (spike 012 follow-up)

Spike 009's evals (``results/spikes/009/evals_lr3e-5.jsonl``) sample every eval prompt
several times at steps 0, 25, 50 and 100 of Granite-3.3-2B GRPO. Labels: capability
(``ok``) on the decode/twin sets; ``refusal``, ``unsafe`` and the gated harm label
(``sim >= 0.8`` AND unsafe, spike 007) on the forbidden sets. The question is how well a
prompt's step-0 (reference) rate predicts its rate later in training. Reported per set: the raw correlation of per-prompt means, and the same corrected for
the sampling noise in both (Spearman's disattenuation, reliabilities from the within-prompt
split-half correlation).

    ../../../.venv/bin/python persistence.py
"""
import collections
import json
import os

import numpy as np

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                    "results", "spikes", "009", "evals_lr3e-5.jsonl")


def reliability(samples):
    """Split-half reliability of a per-prompt mean of k samples (Spearman-Brown)."""
    a = np.array([np.mean(s[::2]) for s in samples])
    b = np.array([np.mean(s[1::2]) for s in samples])
    if a.std() == 0 or b.std() == 0:
        return np.nan
    r = np.corrcoef(a, b)[0, 1]
    return 2 * r / (1 + r)


rows = [json.loads(l) for l in open(PATH)]
S = collections.defaultdict(list)
for r in rows:
    if "ok" in r:
        S[r["set"] + ":ok", r["item"], r["step"]].append(r["ok"])
    else:
        for lab in ("refusal", "unsafe"):
            S[r["set"] + ":" + lab, r["item"], r["step"]].append(r[lab])
        S[r["set"] + ":gated", r["item"], r["step"]].append(int(r["sim"] >= 0.8 and r["unsafe"]))
print("| set:label | prompts | samples/prompt | step | rate | corr with step 0 | disattenuated |")
print("|---|---|---|---|---|---|---|")
for st in sorted({k[0] for k in S}):
    items = sorted({k[1] for k in S if k[0] == st})
    k = len(S[st, items[0], 0])
    base = [S[st, i, 0] for i in items]
    m0 = np.array([np.mean(s) for s in base])
    r0 = reliability(base) if k >= 2 else np.nan
    for step in (25, 50, 100):
        cur = [S[st, i, step] for i in items]
        m = np.array([np.mean(s) for s in cur])
        c = np.corrcoef(m0, m)[0, 1] if m0.std() > 0 and m.std() > 0 else np.nan
        rt = reliability(cur) if k >= 2 else np.nan
        dis = c / np.sqrt(r0 * rt) if (r0 > 0 and rt > 0) else np.nan
        print(f"| {st} | {len(items)} | {k} | {step} | {m.mean():.3f} | {c:.2f} | {dis:.2f} |")
