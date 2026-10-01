"""Spike 017: bundle the data for ``viewer.html`` into ``viewer_data.js``.

Scores and gold labels only (no prompt or response text): 015's refusal and brevity pools
for the live certificate, the plasmode miss rates, the harm stage tables and the carried
calibration curves.

    ../../../.venv/bin/python make_viewer.py
"""
import json
import os

import cert017 as c
import plasmode017 as pm


def main():
    pools = {}
    for (task, variant, w), (y, p) in pm.load_pools().items():
        if task != "harm":
            pools[f"{task}|{variant}|{w}"] = dict(y=[int(v) for v in y], p=[round(float(v), 5) for v in p])
    words = json.load(open(os.path.join(c.REPO, "results", "spikes", "015", "wordings.json")))
    text = {t: [e["w"] for e in words[t]] for t in ("refusal", "brevity")}
    keep = ("classical", "naive", "ppi++", "boot")
    plas = [dict(task=r["task"], variant=r["variant"], wording=r["wording"], n=r["n"],
                 method=r["method"], feat=r["feat"], miss=round(r["miss"], 4), bound=round(r["bound"], 4))
            for r in json.load(open(os.path.join(c.HERE, "plasmode.json")))["rows"]
            if r["N"] == 2000 and r["method"] in keep and r["feat"] in ("-", "f01", "p", "logit")]
    load = lambda f: json.load(open(os.path.join(c.HERE, f)))      # noqa: E731
    data = dict(pools=pools, text=text, plasmode=plas, harm=load("harm.json"),
                carried=load("harmcert.json"), cards=load("cards.json"), k0=c.K0)
    with open(os.path.join(c.HERE, "viewer_data.js"), "w") as fh:
        fh.write("window.DATA = " + json.dumps(data) + ";\n")
    print({k: len(v) for k, v in data.items() if hasattr(v, "__len__")})


if __name__ == "__main__":
    main()
