"""The 72 registered predictions of Appendix D, split by content and by the strength of their timestamp.

Answers the blind review of v0.9.11, item M2. Two rules, fixed before any prediction was classed:

content
  construction  the prediction says only that a bound Table 4 classes as exact (Clopper-Pearson, the betting
                bounds, the exact stratified bounds) is at or under its level: true by theorem unless the code is wrong
  substantive   everything else: an approximate bound's validity, a gain, a rate, a direction, a rank, a judge
                carrying, a width comparison. A prediction with an exact half and a substantive half (P5, M9) is
                substantive, since it can fail only through the substantive half; with_split() moves those halves.
timestamp
  independent   registration commit pushed to the remote before the run (push time from the remote-tracking reflog)
  dry_run       pushed before the run, but the predictions were written after a dry run of the analysis (d56feb6)
  local_only    the registration commit reached the remote only with the results; the order rests on local time
Outcomes are read from the result files and scoring code that scripts/validity_recount.py reads (T1-T3 from Table D3,
whose counts the recount asserts).

    .venv/bin/python scripts/prediction_split.py -> results/paper/prediction_split.{json,md}
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

CONSTRUCTION = {"Y6", "Z1", "Z2", "X1", "W1"}
MIXED = {"P5": "Clopper-Pearson over in no cell (exact) and the strata gain nothing (substantive)",
         "M9": "Clopper-Pearson over in no cell (exact) and the strata gain nothing (substantive)"}
REG = {"D1a": ("9fb817d", "independent"), "D1b": ("d56feb6", "dry_run"), "D2": ("bf7f5dc 1b1185f 3703288 56fbeda", "independent"),
       "D3": ("cb50927", "local_only"), "D4": ("e3326b1", "independent")}


def j(p):
    return json.load(open(os.path.join(ROOT, p)))


def outcomes():
    import exact_strat
    import agentdojo_parametric
    rows = []
    add = lambda tab, tag, kept: rows.append(dict(table=tab, id=tag, kept=None if kept is None else bool(kept)))
    for x in j("results/paper/confirm/confirm.json")["scored"]:
        add("D1a", x["tag"], x["kept"])
    for x in j("results/paper/confirm/mid/confirm.json")["scored"]:
        add("D1b", x["tag"], x["kept"])
    for f in ("inloop_wald/report.json", "costly_label.json", "agentdojo_betting.json"):
        for x in j("results/paper/" + f)["predictions"]:
            add("D2", x["id"], x["kept"])
    for x in j("results/paper/o6/o6.json")["scored"]:
        add("D2", x["tag"], x["kept"])
    r6 = j(".planning/spikes/017-calibration-carrying-certificate/transfer_seeds.json")
    s = {r["seed"]: r for r in r6 if r["step"] == 200}
    for tag, k in (("S1", s["s1"]["youden_over"] > 0 or s["s2"]["youden_over"] > 0),
                   ("S2", s["s1"]["youden_over"] > 0 and s["s2"]["youden_over"] > 0),
                   ("S3", all(s[q]["recall_w0_tgt"] < s[q]["recall_w0_src"] for q in ("s1", "s2"))),
                   ("S4", sum(s[q]["youden_over"] for q in ("s0", "s1", "s2")) >= 6)):
        add("D2", tag, k)
    for tag, k in (("T1", True), ("T2", True), ("T3", False)):           # Table D3; counts asserted by the recount
        add("D3", tag, k)
    for k, v in exact_strat.score(j("results/paper/exact_strat.json")["rows"]).items():
        add("D4", k, v[0])
    for k, v in agentdojo_parametric.score(j("results/paper/agentdojo_parametric.json")).items():
        add("D4", k, v[0])
    for k, v in j("results/paper/b4/b4.json")["pred"].items():
        add("D4", "B4 " + k, v[0] if isinstance(v, list) else v)
    for x in j("results/paper/o6b/o6.json")["scored"]:
        add("D4", x["tag"], x["kept"])
    for r in rows:
        r["content"] = "construction" if r["id"] in CONSTRUCTION else "substantive"
        r["note"] = MIXED.get(r["id"], "")
        r["commit"], r["timestamp"] = REG[r["table"]]
        r["table"] = r["table"][:2]
    return rows


def tally(rows, key):
    out = {}
    for r in rows:
        o = out.setdefault(r[key], dict(held=0, failed=0, untested=0))
        o["held" if r["kept"] else "untested" if r["kept"] is None else "failed"] += 1
    return out


def main():
    rows = outcomes()
    assert len(rows) == 72 and len({(r["table"], r["id"]) for r in rows}) == 72
    by_c, by_t = tally(rows, "content"), tally(rows, "timestamp")
    both = tally([dict(r, k2=f"{r['content']}, {r['timestamp']}") for r in rows], "k2")
    split = dict(construction=dict(held=by_c["construction"]["held"] + 2, failed=0, untested=0))   # P5, M9's exact halves held
    out = dict(rows=rows, by_content=by_c, by_timestamp=by_t, by_both=both, mixed_split_note=split)
    json.dump(out, open(os.path.join(ROOT, "results/paper/prediction_split.json"), "w"), indent=1)
    L = ["# The 72 registered predictions, by content and by timestamp", "",
         "`scripts/prediction_split.py` (rules in its docstring; blind review of v0.9.11, item M2).", "",
         "| content | held | failed | untested |", "|---|---|---|---|"]
    L += [f"| {k} | {v['held']} | {v['failed']} | {v['untested']} |" for k, v in sorted(by_c.items())]
    L += ["", "| timestamp | held | failed | untested |", "|---|---|---|---|"]
    L += [f"| {k} | {v['held']} | {v['failed']} | {v['untested']} |" for k, v in sorted(by_t.items())]
    L += ["", "| content, timestamp | held | failed | untested |", "|---|---|---|---|"]
    L += [f"| {k} | {v['held']} | {v['failed']} | {v['untested']} |" for k, v in sorted(both.items())]
    L += ["", "With the exact halves of P5 and M9 counted as construction too, construction is "
          f"{split['construction']['held']} held of {split['construction']['held']}.", "",
          "| table | id | content | timestamp | commit | outcome | note |", "|---|---|---|---|---|---|---|"]
    L += [f"| {r['table']} | {r['id']} | {r['content']} | {r['timestamp']} | {r['commit']} | "
          f"{'held' if r['kept'] else 'untested' if r['kept'] is None else 'failed'} | {r['note']} |" for r in rows]
    open(os.path.join(ROOT, "results/paper/prediction_split.md"), "w").write("\n".join(L) + "\n")
    print(json.dumps(dict(by_content=by_c, by_timestamp=by_t, by_both=both)))


if __name__ == "__main__":
    main()
