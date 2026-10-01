"""Spike 016: one headline table across prompt versions and arms, from results*.json.

    ../../../.venv/bin/python summary016.py        # writes summary.md
"""
import json
import os

import speclab as sl

VERSIONS = [("v1", "", "v1: attribute=value, 3 routes to the reference, 3 examples"),
            ("v1b", "_v1b", "v1b: v1's registry, v2's 10 examples"),
            ("v2a", "_v2a", "v2a: v2's registry, v1's 3 examples"),
            ("v2", "_v2", "v2: flat groups, REF only, 10 examples")]
ARMS = ["dsl-plain", "json-plain", "dsl-think", "json-think"]
GOOD = ("same_g", "same_point")


def frac(rs, levels):
    return f"{sum(r['level'] in levels for r in rs)}/{len(rs)}"


def main():
    out = ["# Spike 016 headline\n",
           "same_g = the gold certificate; req = same_g or same_point (the same requirement, "
           "the bound left to the deterministic rule); silent = parsed, built, wrong.\n",
           "| prompt | arm | canonical 8: same_g / req | faithful wordings (39): same_g / req / "
           "silent / no spec | held-out 10: same_g | traps 6 | under-specified 8: asked | "
           "outside the registry 6: JUDGE, right prompts and limit | valid first reply (89) |",
           "|---|---|---|---|---|---|---|---|---|"]
    for ver, tag, _ in VERSIONS:
        path = os.path.join(sl.HERE, f"results{tag}.json")
        if not os.path.exists(path):
            continue
        rows = json.load(open(path))
        for arm in ARMS:
            rs = [r for r in rows if r["arm"] == arm]
            if len(rs) < 89:
                continue
            canon = [r for r in rs if r["set"] == "main" and r["wording"] == 0]
            faith = [r for r in rs if r["set"] == "main" and r["fidelity"] == "F"]
            held = [r for r in rs if r["set"] == "heldout"]
            trap = [r for r in rs if r["set"] == "trap"]
            under = [r for r in rs if r["set"] in ("underspecified", "heldout_under")]
            unreg = [r for r in rs if r["set"] in ("unregistered", "heldout_unreg")]
            out.append(
                f"| {ver} | {arm} | {frac(canon, ('same_g',))} / {frac(canon, GOOD)} | "
                f"{sum(r['level'] == 'same_g' for r in faith)} / "
                f"{sum(r['level'] in GOOD for r in faith)} / "
                f"{sum(r['level'] == 'wrong' for r in faith)} / "
                f"{sum(r['level'] in ('fail', 'ask') for r in faith)} | "
                f"{frac(held, ('same_g',))} | {frac(trap, ('same_g',))} | "
                f"{frac(under, ('asked',))} | {frac(unreg, ('same_g',))} | "
                f"{sum(r['first_ok'] for r in rs)} |")
    out.append("")
    out += [f"- {label}" for _, _, label in VERSIONS]
    text = "\n".join(out) + "\n"
    open(os.path.join(sl.HERE, "summary.md"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
