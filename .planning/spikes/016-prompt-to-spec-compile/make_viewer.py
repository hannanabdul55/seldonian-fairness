"""Spike 016: bundle the scored compiles into ``viewer_data.js`` for ``viewer.html``.

    ../../../.venv/bin/python make_viewer.py
"""
import json
import os

import speclab as sl
from gold import CONSTRAINTS, EDGE, HELDOUT

NAMES = {"v1": "v1 (pre-registered prompt)", "v2": "v2 (flat groups, REF only, more examples)",
         "v1b": "v1b (v1's registry, v2's examples)", "v2a": "v2a (v2's registry, v1's examples)"}


def main():
    rows = {}
    for ver, tag in (("v2", "_v2"), ("v1", ""), ("v1b", "_v1b"), ("v2a", "_v2a")):
        path = os.path.join(sl.HERE, f"results{tag}.json")
        if os.path.exists(path):
            rows[ver] = json.load(open(path))
    gold = {c["key"]: c["gold"][0] for c in CONSTRAINTS + EDGE + HELDOUT if "gold" in c}
    data = dict(rows=rows, gold=gold, versions={v: NAMES[v] for v in rows})
    with open(os.path.join(sl.HERE, "viewer_data.js"), "w") as fh:
        fh.write("window.DATA = " + json.dumps(data) + ";\n")
    print({v: len(r) for v, r in rows.items()})


if __name__ == "__main__":
    main()
