"""Spike 016 analysis (CPU): score every compile against the gold constraint.

A compiled spec is scored by what it computes, not by how it is written. On five data sets
(two checkpoints, three prompt sub-samples) it is evaluated beside each gold alternative:

- ``same_g``      identical g, bound included: the same certificate
- ``same_point``  identical point value of g (statistic minus threshold) but a different
                  bound: the same requirement, certified another way
- ``wrong``       anything else that parsed and built: a silent error
- ``ask`` / ``fail``  the compiler asked a question / no valid spec after one repair turn

    ../../../.venv/bin/python analyze016.py [--tag v1]
"""
import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np

import speclab as sl
from check_builder import datasets
from gold import CONSTRAINTS, EDGE, HELDOUT

TOL = 1e-9
ARM = lambda r: f"{r['fmt']}-{'think' if r['think'] else 'plain'}"   # noqa: E731
ARMS = ["dsl-plain", "json-plain", "dsl-think", "json-think"]
RELATIVE = ["harm_rel", "refusal_rel", "brevity_rel", "harm_times", "paired_ref"]


class Scorer:
    def __init__(self):
        self.cache = sl.Cache()
        self.data = datasets(self.cache)
        self.memo = {}
        self.gold = {}
        for c in CONSTRAINTS + [e for e in EDGE + HELDOUT if "gold" in e]:
            self.gold[c["key"]] = [self.vector(sl.parse_dsl(g)) for g in c["gold"]]

    def vector(self, spec):
        """(g, g_point) on every data set, or None when the spec cannot be evaluated."""
        key = json.dumps(sl.canonical(spec), sort_keys=True)
        if key not in self.memo:
            try:
                bound = sl.choose_bound(spec, self.cache)["bound"]
                ev = [sl.evaluate(spec, self.cache, ds, ref, bound=bound)
                      for _, ds, ref in self.data]
                v = (np.array([e["g"] for e in ev]), np.array([e["g_point"] for e in ev]))
                self.memo[key] = v if np.isfinite(v[1]).all() else None
            except (sl.SpecError, ZeroDivisionError, KeyError):
                self.memo[key] = None
        return self.memo[key]

    def level(self, spec, key):
        v = self.vector(spec)
        if v is None:
            return "wrong"
        best = "wrong"
        for g in self.gold[key]:
            if np.allclose(v[0], g[0], atol=TOL, rtol=0, equal_nan=True):
                return "same_g"
            if np.allclose(v[1], g[1], atol=TOL, rtol=0):
                best = "same_point"
        return best

    def ref_violates(self, spec):
        """Lint for a relative constraint: does the reference model itself break it?

        A requirement stated relative to the reference ("no more than 5 points above it") is
        one the reference meets by construction, so a compiled spec that the reference's own
        point estimate violates has its direction or its sign wrong. No labels needed."""
        if reference_style(spec) == "none":
            return False
        key = "lint:" + json.dumps(sl.canonical(spec), sort_keys=True)
        if key not in self.memo:
            try:
                ds0 = self.ds0 = getattr(self, "ds0", None) or self.cache.dataset(0)
                self.memo[key] = sl.evaluate(spec, self.cache, ds0, ds0)["g_point"] > 1e-12
            except (sl.SpecError, ZeroDivisionError, KeyError):
                self.memo[key] = False
        return bool(self.memo[key])

    def same_point(self, a, b):
        """Do two compiled specs state the same requirement (whatever the bound)?"""
        va, vb = self.vector(a), self.vector(b)
        if va is None or vb is None:
            return self.same(a, b)
        return bool(np.allclose(va[1], vb[1], atol=TOL, rtol=0))

    def same(self, a, b):
        """Do two compiled specs give the same certificate?"""
        va, vb = self.vector(a), self.vector(b)
        if va is None or vb is None:
            return json.dumps(sl.canonical(a), sort_keys=True) == \
                json.dumps(sl.canonical(b), sort_keys=True)
        return bool(np.allclose(va[0], vb[0], atol=TOL, rtol=0, equal_nan=True))


def reference_style(spec):
    """How the reference model enters: through REF, a two-sample or a paired difference."""
    named = any(m["where"].get("model") == "reference" for m in spec["measures"].values())
    ref = spec["threshold"]["form"] != "absolute"
    if spec["paired"]:
        return "paired" + ("+REF" if ref else "")
    if named:
        return "two-sample" + ("+REF" if ref else "")
    return "REF" if ref else "none"


def diagnose(spec, gold):
    """Which parts of a wrong compile differ from the gold spec (several may)."""
    def feats(sp):
        return sorted(json.dumps([m["feature"], m["args"]]) for m in sp["measures"].values())

    def pops(sp):
        return sorted(json.dumps({k: v for k, v in m["where"].items() if k != "model"},
                                 sort_keys=True) for m in sp["measures"].values())

    def shape(node):
        if node[0] == "m":
            return "m"
        return [node[0]] + [shape(c) if isinstance(c, list) else c for c in node[1:]]

    tags = []
    if feats(spec) != feats(gold):
        tags.append("measurement")
    if pops(spec) != pops(gold):
        tags.append("prompts")
    if reference_style(spec) != reference_style(gold):
        tags.append("reference")
    if shape(spec["expr"]) != shape(gold["expr"]):
        tags.append("expression")
    if (spec["threshold"]["form"], round(spec["threshold"]["value"], 9)) != \
            (gold["threshold"]["form"], round(gold["threshold"]["value"], 9)):
        tags.append("limit")
    return tags or ["order of terms"]


def score_unregistered(spec, e):
    """A property outside the registry: one JUDGE measure, right prompts, right limit."""
    ms = list(spec["measures"].values())
    if len(ms) != 1 or ms[0]["feature"] != "JUDGE":
        return "wrong", None
    text = ms[0]["args"][0]
    ok = (ms[0]["where"] == e["where"] and spec["threshold"]["form"] == "absolute"
          and abs(spec["threshold"]["value"] - e["value"]) < TOL and spec["expr"][0] == "m")
    return ("same_g" if ok else "wrong"), text


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def pct(a, b):
    return f"{a}/{b}" if b else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="suffix of compiles/verify files, e.g. _v2")
    a = ap.parse_args()
    sc = Scorer()
    audit = json.load(open(os.path.join(sl.HERE, "paraphrase_audit.json")))
    edge = {e["key"]: e for e in EDGE + HELDOUT}
    rows = sl.read_jsonl(os.path.join(sl.OUT, f"compiles{a.tag}.jsonl"))
    vpath = os.path.join(sl.OUT, f"verify{a.tag}.jsonl")
    verify = {(r["fmt"], r["think"], r["set"], r["key"], r["wording"]): r["p"]
              for r in (sl.read_jsonl(vpath) if os.path.exists(vpath) else [])}

    for r in rows:
        r["arm"] = ARM(r)
        r["first_ok"] = r["error1"] is None
        r["p_verify"] = verify.get((r["fmt"], r["think"], r["set"], r["key"], r["wording"]))
        r["fidelity"] = ("F" if r["wording"] == 0 else audit[r["key"]][r["wording"] - 1]) \
            if r["set"] == "main" else None
        if r["status"] == "ok":
            r["dsl"] = sl.render_dsl(r["spec"])
            r["english"] = sl.render_english(r["spec"])
            r["routes"] = sl.routes(r["spec"])
            r["ref_style"] = reference_style(r["spec"])
            r["lint"] = sc.ref_violates(r["spec"])
            r["ungrounded"] = sl.ungrounded(r["spec"], r["text"])
            if r["set"] in ("unregistered", "heldout_unreg"):
                r["level"], r["judge_text"] = score_unregistered(r["spec"], edge[r["key"]])
            elif r["set"] in ("underspecified", "heldout_under"):
                r["level"] = "invented"
            else:
                r["level"] = sc.level(r["spec"], r["key"])
        else:
            r["level"] = "ask" if r["status"] == "clarify" else "fail"
        if r["set"] in ("underspecified", "heldout_under") and r["status"] == "clarify":
            r["level"] = "asked"

    out = []
    w = out.append
    arms = [x for x in ARMS if any(r["arm"] == x for r in rows)]
    by = defaultdict(list)
    for r in rows:
        by[r["arm"]].append(r)

    def cell(sel, arm, levels):
        rs = [r for r in by[arm] if sel(r)]
        return pct(sum(r["level"] in levels for r in rs), len(rs))

    w(f"# Spike 016 results{' (' + a.tag.strip('_') + ')' if a.tag else ''}\n")
    w(f"{len(rows)} compiles, {len(arms)} arms. Levels: `same_g` = the gold certificate; "
      "`same_point` = the same requirement with a different bound; `wrong` = a silent error "
      "(parsed, built, means something else).\n")

    w("## 1. The eight canonical sentences\n")
    w("| constraint | " + " | ".join(arms) + " |")
    w("|---|" + "---|" * len(arms))
    for c in CONSTRAINTS:
        cells = []
        for arm in arms:
            r = [x for x in by[arm] if x["set"] == "main" and x["key"] == c["key"]
                 and x["wording"] == 0]
            cells.append(r[0]["level"] + ("" if r[0]["first_ok"] else " (repaired)")
                         if r else "-")
        w(f"| {c['key']} | " + " | ".join(cells) + " |")
    w("| **same_g** | " + " | ".join(
        cell(lambda r: r["set"] == "main" and r["wording"] == 0, arm, {"same_g"})
        for arm in arms) + " |")
    w("| **same_g or same_point** | " + " | ".join(
        cell(lambda r: r["set"] == "main" and r["wording"] == 0, arm, {"same_g", "same_point"})
        for arm in arms) + " |\n")

    w("## 2. All wordings, by the fidelity of the paraphrase\n")
    w("Fidelity was judged by reading the paraphrases before any compile ran "
      "(`paraphrase_audit.json`): F faithful (incl. the canonical sentence), A ambiguous, "
      "D drifted.\n")
    w("| fidelity | arm | n | same_g | same_point | wrong | ask | fail |")
    w("|---|---|---|---|---|---|---|---|")
    for fid in "FAD":
        for arm in arms:
            rs = [r for r in by[arm] if r["set"] == "main" and r["fidelity"] == fid]
            n = Counter(r["level"] for r in rs)
            w(f"| {fid} | {arm} | {len(rs)} | {n['same_g']} | {n['same_point']} | "
              f"{n['wrong']} | {n['ask']} | {n['fail']} |")
    w("")
    w("Faithful wordings, Round 6 three against the harder five (same_g / same_g or "
      "same_point / n):\n")
    w("| group | " + " | ".join(arms) + " |")
    w("|---|" + "---|" * len(arms))
    for label, keys in (("Round 6 three", [c["key"] for c in CONSTRAINTS if c["round6"]]),
                        ("harder five", [c["key"] for c in CONSTRAINTS if not c["round6"]])):
        cells = []
        for arm in arms:
            rs = [r for r in by[arm] if r["set"] == "main" and r["fidelity"] == "F"
                  and r["key"] in keys]
            cells.append(f"{sum(r['level'] == 'same_g' for r in rs)} / "
                         f"{sum(r['level'] in ('same_g', 'same_point') for r in rs)} / {len(rs)}")
        w(f"| {label} | " + " | ".join(cells) + " |")
    w("")

    w("## 3. Per constraint, faithful wordings (same_g + same_point + wrong + ask/fail)\n")
    w("| constraint | n | " + " | ".join(arms) + " | distinct certificates (best arm) |")
    w("|---|---|" + "---|" * len(arms) + "---|")
    best_arm = max(arms, key=lambda arm: sum(
        r["level"] == "same_g" for r in by[arm] if r["set"] == "main" and r["fidelity"] == "F"))
    for c in CONSTRAINTS:
        cells, n = [], 0
        for arm in arms:
            rs = [r for r in by[arm] if r["set"] == "main" and r["key"] == c["key"]
                  and r["fidelity"] == "F"]
            k = Counter(r["level"] for r in rs)
            n = len(rs)
            cells.append(f"{k['same_g']}+{k['same_point']}+{k['wrong']}+{k['ask'] + k['fail']}")
        oks = [r for r in by[best_arm] if r["set"] == "main" and r["key"] == c["key"]
               and r["fidelity"] == "F" and r["status"] == "ok"]
        classes = []
        for r in oks:
            if not any(sc.same(r["spec"], q["spec"]) for q in classes):
                classes.append(r)
        w(f"| {c['key']} | {n} | " + " | ".join(cells) + f" | {len(classes)} |")
    w(f"\nBest arm by same_g on faithful wordings: **{best_arm}**.\n")

    w("## 4. How the reference model enters a relative constraint\n")
    w("The gold uses the project's convention (`REF`, the reference value as a constant), "
      "except `paired_ref`, whose sentence asks for a paired difference. Counts over all "
      "wordings that compiled.\n")
    styles = sorted({r["ref_style"] for r in rows if r["status"] == "ok"
                     and r["set"] == "main" and r["key"] in RELATIVE})
    w("| constraint | arm | " + " | ".join(styles) + " |")
    w("|---|---|" + "---|" * len(styles))
    for key in RELATIVE:
        for arm in arms:
            k = Counter(r["ref_style"] for r in by[arm] if r["set"] == "main"
                        and r["key"] == key and r["status"] == "ok")
            w(f"| {key} | {arm} | " + " | ".join(str(k[s]) for s in styles) + " |")
    w("")

    w("## 5. Edge probes\n")
    w("| probe | want | " + " | ".join(arms) + " |")
    w("|---|---|" + "---|" * len(arms))
    for kind, want, good in (("trap", "same_g", {"same_g"}),
                             ("verifiable", "same_g", {"same_g"}),
                             ("underspecified", "asked", {"asked"}),
                             ("unregistered", "JUDGE, right prompts and limit", {"same_g"})):
        w(f"| {kind} | {want} | " + " | ".join(
            cell(lambda r: r["set"] == kind, arm, good) for arm in arms) + " |")
    w("")
    w("Per probe (level per arm):\n")
    w("| probe | " + " | ".join(arms) + " |")
    w("|---|" + "---|" * len(arms))
    for e in EDGE:
        cells = []
        for arm in arms:
            r = [x for x in by[arm] if x["key"] == e["key"] and x["set"] == e["kind"]]
            cells.append(r[0]["level"] if r else "-")
        w(f"| {e['key']} | " + " | ".join(cells) + " |")
    w("")
    w("What the compiler wrote when it should have asked:\n")
    for e in EDGE:
        if e["kind"] != "underspecified":
            continue
        w(f"- *{e['text']}*")
        for arm in arms:
            r = [x for x in by[arm] if x["key"] == e["key"]]
            if r:
                got = r[0].get("dsl") or ("ASKED: " + (r[0]["question"] or "")
                                           if r[0]["status"] == "clarify" else "no valid spec")
                w(f"  - {arm}: `{got}`")
    w("")
    w("Routing: specs whose gold uses only counted features (brevity_rel, length_score, "
      "t_onein5, v_*), and what the compile routed to:\n")
    verif = {"brevity_rel", "length_score", "t_onein5", "v_longwinded", "v_budget", "v_rambling"}
    w("| arm | compiled | used a judged feature | used JUDGE(\"...\") |")
    w("|---|---|---|---|")
    for arm in arms:
        rs = [r for r in by[arm] if r["key"] in verif and r["status"] == "ok"
              and r["set"] != "roundtrip"]
        w(f"| {arm} | {len(rs)} | {sum('judged' in r['routes'] for r in rs)} | "
          f"{sum('prompted' in r['routes'] for r in rs)} |")
    w("")
    w("JUDGE wording for the unregistered properties:\n")
    for e in EDGE:
        if e["kind"] != "unregistered":
            continue
        w(f"- *{e['text']}*")
        for arm in arms:
            r = [x for x in by[arm] if x["key"] == e["key"]]
            if r:
                w(f"  - {arm}: `{r[0].get('dsl') or r[0]['level']}`")
    w("")

    if any(r["set"].startswith("heldout") for r in rows):
        w("## 5b. Held-out sentences (written before the second prompt)\n")
        w("| sentence | want | " + " | ".join(arms) + " |")
        w("|---|---|" + "---|" * len(arms))
        for e in HELDOUT:
            want = {"heldout": "same_g", "heldout_under": "asked",
                    "heldout_unreg": "JUDGE"}[e["kind"]]
            cells = []
            for arm in arms:
                r = [x for x in by[arm] if x["key"] == e["key"] and x["set"] == e["kind"]]
                cells.append(r[0]["level"] if r else "-")
            w(f"| {e['key']} | {want} | " + " | ".join(cells) + " |")
        w("| **translation: same_g** | | " + " | ".join(
            cell(lambda r: r["set"] == "heldout", arm, {"same_g"}) for arm in arms) + " |")
        w("| **same_g or same_point** | | " + " | ".join(
            cell(lambda r: r["set"] == "heldout", arm, {"same_g", "same_point"})
            for arm in arms) + " |")
        w("| **all 14 as wanted** | | " + " | ".join(
            cell(lambda r: r["set"].startswith("heldout"), arm, {"same_g", "asked"})
            for arm in arms) + " |\n")
        w("Compiled lines for the held-out sentences:\n")
        for e in HELDOUT:
            w(f"- *{e['text']}*")
            for arm in arms:
                r = [x for x in by[arm] if x["key"] == e["key"] and x["set"] == e["kind"]]
                if r:
                    got = r[0].get("dsl") or ("ASKED: " + (r[0]["question"] or "")
                                               if r[0]["status"] == "clarify" else "no valid spec")
                    w(f"  - {arm} [{r[0]['level']}]: `{got}`")
        w("")

    w("## 6. Round trip: the English rendering of the gold spec, compiled again\n")
    w("| constraint | " + " | ".join(arms) + " |")
    w("|---|" + "---|" * len(arms))
    for c in CONSTRAINTS:
        cells = []
        for arm in arms:
            r = [x for x in by[arm] if x["set"] == "roundtrip" and x["key"] == c["key"]]
            cells.append(r[0]["level"] if r else "-")
        w(f"| {c['key']} | " + " | ".join(cells) + " |")
    w("| **same_g** | " + " | ".join(
        cell(lambda r: r["set"] == "roundtrip", arm, {"same_g"}) for arm in arms) + " |\n")

    w("## 7. Parse and repair\n")
    w("| arm | items | valid on the first reply | valid or asked after one repair turn | "
      "no valid spec |")
    w("|---|---|---|---|---|")
    for arm in arms:
        rs = by[arm]
        w(f"| {arm} | {len(rs)} | {sum(r['first_ok'] for r in rs)} | "
          f"{sum(r['status'] != 'fail' for r in rs)} | {sum(r['status'] == 'fail' for r in rs)} |")
    errs = Counter()
    for r in rows:
        if r["error1"]:
            errs[r["error1"].split(";")[0].split(":")[0][:60]] += 1
    w("\nMost common first-reply errors: " + "; ".join(f"{k} ({v})" for k, v in
                                                      errs.most_common(6)) + ".\n")

    w("## 8. Catching silent errors\n")
    scored = [r for r in rows if r["set"] in ("main", "trap", "verifiable", "heldout")
              and r["status"] == "ok" and r.get("fidelity") in (None, "F")]
    w("Compiles that parsed and built, on items with a gold and a faithful wording. "
      "An *error* is `wrong`; `same_point` counts as right here.\n")
    w("| arm | built | wrong (silent) | verify AUC | accepted at P(Yes) >= 0.5 | "
      "wrong among accepted |")
    w("|---|---|---|---|---|---|")
    for arm in arms:
        rs = [r for r in scored if r["arm"] == arm]
        good = [r["p_verify"] for r in rs if r["level"] != "wrong" and r["p_verify"] is not None]
        bad = [r["p_verify"] for r in rs if r["level"] == "wrong" and r["p_verify"] is not None]
        acc = [r for r in rs if r["p_verify"] is not None and r["p_verify"] >= 0.5]
        w(f"| {arm} | {len(rs)} | {sum(r['level'] == 'wrong' for r in rs)} | "
          f"{auc(good, bad):.3f} | {len(acc)} | {sum(r['level'] == 'wrong' for r in acc)} |")
    w("")
    w("Lint, no model and no labels: a relative constraint that the reference model itself "
      "violates has its direction or sign wrong.\n")
    w("| arm | relative specs built | flagged | flagged and wrong | flagged but right | "
      "wrong and not flagged |")
    w("|---|---|---|---|---|---|")
    for arm in arms:
        rs = [r for r in scored if r["arm"] == arm and r["ref_style"] != "none"]
        fl = [r for r in rs if r["lint"]]
        w(f"| {arm} | {len(rs)} | {len(fl)} | {sum(r['level'] == 'wrong' for r in fl)} | "
          f"{sum(r['level'] != 'wrong' for r in fl)} | "
          f"{sum(r['level'] == 'wrong' and not r['lint'] for r in rs)} |")
    w("")
    w("Lint, no model and no labels: every number in the spec must occur in the sentence "
      "(as written, as a percentage, as a number word, or as one plus or minus such a "
      "fraction). A number that does not is an invented limit.\n")
    w("| arm | built, should have asked | of those, flagged | built with a gold | flagged "
      "though right | flagged and wrong |")
    w("|---|---|---|---|---|---|")
    for arm in arms:
        inv = [r for r in by[arm] if r["level"] == "invented"]
        gold = [r for r in by[arm] if r["status"] == "ok" and r["key"] in sc.gold
                and r["set"] != "roundtrip"]
        w(f"| {arm} | {len(inv)} | {sum(bool(r['ungrounded']) for r in inv)} | {len(gold)} | "
          f"{sum(bool(r['ungrounded']) and r['level'] != 'wrong' for r in gold)} | "
          f"{sum(bool(r['ungrounded']) and r['level'] == 'wrong' for r in gold)} |")
    w("")
    w("Agreement between two independent compiles of the same sentence as a filter. "
      "*certificate*: accept when both built and give the same g; *requirement*: accept when "
      "both state the same requirement (same point value of g), leaving the bound to the "
      "deterministic rule. `+lint` also drops anything the lint flags.\n")
    w("| pair | level | items | accepted | wrong among accepted | wrong among "
      "rejected-but-built |")
    w("|---|---|---|---|---|---|")
    idx = {(r["arm"], r["set"], r["key"], r["wording"]): r for r in rows}
    items = sorted({(r["set"], r["key"], r["wording"]) for r in scored})
    pairs = [(x, y) for j, x in enumerate(arms) for y in arms[j + 1:]]
    for x, y in pairs:
        for label, fn, lint in (("certificate", sc.same, False),
                                ("requirement", sc.same_point, False),
                                ("requirement +lint", sc.same_point, True)):
            acc = rej = wa = wr = 0
            for it in items:
                rx, ry = idx.get((x,) + it), idx.get((y,) + it)
                if not rx or not ry:
                    continue
                ok = (rx["status"] == "ok" and ry["status"] == "ok"
                      and fn(rx["spec"], ry["spec"])
                      and not (lint and (rx["lint"] or ry["lint"])))
                if ok:
                    acc += 1
                    wa += rx["level"] == "wrong"
                else:
                    for r in (rx, ry):
                        if r["status"] == "ok":
                            rej += 1
                            wr += r["level"] == "wrong"
            w(f"| {x} + {y} | {label} | {len(items)} | {acc} | {wa} | {wr}/{rej} |")
    w("")

    w("## 9. The guarded pipeline, end to end\n")
    w("Compile in the primary form; drop anything a lint flags (reference violates it, "
      "invented number); compile again in a second form and accept only when the two state "
      "the same requirement; otherwise go back to the developer with a question. Items: "
      "every sentence with a known right answer (faithful wordings, traps, verifiable and "
      "unregistered probes, held-out) plus the under-specified ones, which must be sent back. "
      "Ambiguous and drifted paraphrases are left out: they have no right answer. *Right* = "
      "same_g or same_point (the bound is set by the deterministic rule either way).\n")
    w("| primary + check | items | accepted | accepted and right | accepted and wrong | "
      "sent back: under-specified | sent back: a clear sentence |")
    w("|---|---|---|---|---|---|---|")
    combos = [c for c in (("dsl-plain", "json-plain"), ("dsl-think", "dsl-plain"),
                          ("dsl-think", "json-think"))
              if c[0] in arms and c[1] in arms]
    all_items = sorted({(r["set"], r["key"], r["wording"]) for r in rows
                        if r["set"] != "roundtrip" and r.get("fidelity") in (None, "F")})
    pipeline = {}
    for x, y in combos:
        n = acc = right = wrong = back_ok = back_clear = 0
        for it in all_items:
            rx, ry = idx.get((x,) + it), idx.get((y,) + it)
            if not rx or not ry:
                continue
            n += 1
            unclear = rx["set"] in ("underspecified", "heldout_under")
            ok = (rx["status"] == "ok" and ry["status"] == "ok"
                  and not rx["lint"] and not rx["ungrounded"]
                  and sc.same_point(rx["spec"], ry["spec"]))
            if ok:
                acc += 1
                good = rx["level"] in ("same_g", "same_point") and not unclear
                right += good
                wrong += not good
                pipeline[(x, y) + it] = "accepted, right" if good else "accepted, WRONG"
            else:
                back_ok += unclear
                back_clear += not unclear
        w(f"| {x} + {y} | {n} | {acc} | {right} | {wrong} | {back_ok} | {back_clear} |")
    w("")
    for (x, y, st, key, wd), v in sorted(pipeline.items()):
        if v.endswith("WRONG"):
            r = idx[(x, st, key, wd)]
            w(f"- accepted and wrong, {x} + {y}: `{key}` w{wd} ({st}): `{r['dsl']}`")
    w("")

    w("## 10. What is wrong in a wrong compile\n")
    w("Wrong compiles on faithful wordings, traps, verifiable probes and held-out sentences, "
      "compared part by part with the gold spec (a compile can differ in several parts).\n")
    parts = ["measurement", "prompts", "reference", "expression", "limit", "order of terms"]
    golds = {c["key"]: sl.parse_dsl(c["gold"][0])
             for c in CONSTRAINTS + EDGE + HELDOUT if "gold" in c}
    w("| arm | wrong | " + " | ".join(parts) + " |")
    w("|---|---|" + "---|" * len(parts))
    for arm in arms:
        rs = [r for r in scored if r["arm"] == arm and r["level"] == "wrong"]
        k = Counter(t for r in rs for t in diagnose(r["spec"], golds[r["key"]]))
        w(f"| {arm} | {len(rs)} | " + " | ".join(str(k[p]) for p in parts) + " |")
    w("")

    text = "\n".join(out) + "\n"
    name = f"results{a.tag}.md"
    open(os.path.join(sl.HERE, name), "w").write(text)
    keep = ("arm", "set", "key", "wording", "text", "status", "level", "first_ok", "repaired",
            "fidelity", "dsl", "english", "question", "error1", "error2", "ref_style",
            "routes", "p_verify", "judge_text", "lint", "ungrounded")
    slim = [{k: r.get(k) for k in keep} for r in rows]
    json.dump(slim, open(os.path.join(sl.HERE, f"results{a.tag}.json"), "w"), indent=0)
    print(text)


if __name__ == "__main__":
    main()
