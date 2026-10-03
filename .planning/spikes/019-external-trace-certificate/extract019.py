"""Spike 019: events per trial from the RoboDojo transcript pages, by code only.

Each page renders the trial twice (a pretty view and a ``<details class="raw-transcript">``
block); only the raw block is parsed, so nothing is counted twice. Per trial:

- ``termination`` from ``trials.csv`` (done / give_up / overheat) and the last tool call;
- ``approver``: counts by variant of ``approver: N step(s) modified (<variant>)`` lines, the
  variant with any ``xK`` multiplicity stripped (``delta_clamped``, and whatever else the
  enumeration pass shows);
- ``operator``: lines mentioning the operator outside the system prompt;
- ``errors``: tool results that report an error;
- ``joint_eff``: peak absolute joint effort over observations, excluding the gripper slots
  (6 and 13 of the 14-slot layout), and the share of observations whose peak exceeds the
  pooled 95th percentile (computed in a second pass over all trials);
- ``risk_notes``: number of ``note`` fields matching drop / collision / spill / damage /
  force / risk (the policy's own narration; a feature, never a label);
- ``moves``, ``calls``, ``steps``.

    ../../../.venv/bin/python extract019.py --enumerate     # distinct templates, nothing written
    ../../../.venv/bin/python extract019.py                 # -> results/spikes/019/events.jsonl
"""
import argparse
import collections
import csv
import html
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(REPO, "results", "spikes", "019")
LOGS = "/mnt/d/seldonian-runs/019/logs"
GRIPPER_SLOTS = (6, 13)
# exploratory, after the scored pass showed "drop" matching intended releases ("drop the fruit into the bowl")
RISK_STRICT = re.compile(r"\b(collision|collide|collid|spill|damage|damaging|risk|unsafe|danger)", re.I)
RISK = re.compile(r"\b(drop|dropp|collision|collide|spill|damage|damaging|force|forcing|risk)", re.I)


def raw_text(path):
    s = open(path, encoding="utf-8", errors="replace").read()
    blocks = re.findall(r'<details class="raw-transcript">(.*?)</details>', s, re.S)
    m = bool(blocks)
    block = "\n".join(blocks) if blocks else s
    block = re.sub(r'<img[^>]*>|<figure.*?</figure>', '', block, flags=re.S)
    txt = html.unescape(re.sub(r'<[^>]+>', '\n', block))
    header = html.unescape(re.sub(r'<[^>]+>', '\n', re.sub(r'<script.*?</script>|<style.*?</style>', '', s[:20000], flags=re.S)))
    return txt, header, bool(m)


def split_messages(txt):
    """The raw block renders roles as bare lines: system / user / assistant / tool."""
    msgs, cur, role = [], [], None
    for line in txt.split("\n"):
        t = line.strip()
        if t in ("system", "user", "assistant", "tool") and not cur:
            role = t
            continue
        if t in ("system", "user", "assistant", "tool"):
            msgs.append((role, "\n".join(cur)))
            cur, role = [], t
            continue
        if t:
            cur.append(t)
    if cur:
        msgs.append((role, "\n".join(cur)))
    return msgs


def parse(path):
    txt, header, has_raw = raw_text(path)
    msgs = split_messages(txt)
    ev = dict(has_raw=has_raw, n_msgs=len(msgs))
    sys_text = "\n".join(c for r, c in msgs if r == "system")
    appr = collections.Counter()
    operator, errors, calls, notes = [], [], [], []
    peaks = []
    for role, c in msgs:
        if role == "system":
            continue
        for n, variant in re.findall(r"approver: (\d+) step\(s\) modified \((.*?)\)", c):
            v = re.sub(r"\s*[x×]\s*\d+", "", variant).strip()
            appr[v] += int(n)
        for line in c.split("\n"):
            if role in ("user", "tool") and "operator" in line.lower() and line not in sys_text \
                    and not line.startswith(("give_up", "done")):
                operator.append(line[:200])
            if role == "tool" and re.search(r"\berror|failed|exception|refused|rejected", line, re.I) \
                    and not line.startswith(("give_up", "done")):
                errors.append(line[:200])
        for m in re.finditer(r"state\[joint_eff\]: \[([^\]]*)\]", c):
            vals = np.array([float(x) for x in m.group(1).split(",")])
            keep = [i for i in range(len(vals)) if i not in GRIPPER_SLOTS]
            peaks.append(float(np.abs(vals[keep]).max()))
        if role == "assistant":
            for m in re.finditer(r"\b(move_to|give_up|done|wait|observe|reset)\(\{", c):
                calls.append(m.group(1))
            for m in re.finditer(r'"note"\s*:\s*"((?:[^"\\]|\\.)*)"', c):
                notes.append(m.group(1))
    ev.update(approver={k: v for k, v in appr.items()}, approver_lines=sum(appr.values()),
              operator_lines=operator, n_operator=len(operator), errors=errors[:5], n_errors=len(errors),
              calls=collections.Counter(calls), n_moves=calls.count("move_to"), n_calls=len(calls),
              last_call=calls[-1] if calls else None, n_notes=len(notes),
              ended_mid_move=bool(calls) and calls[-1] == "move_to",
              risk_notes=sum(1 for n in notes if RISK.search(n)),
              risk_notes_strict=sum(1 for n in notes if RISK_STRICT.search(n)),
              risk_terms=collections.Counter(t.lower() for n in notes for t in RISK.findall(n)),
              eff_peak=max(peaks) if peaks else None, eff_peaks=peaks, n_obs=len(peaks))
    m = re.search(r"total steps\n(\d+)", header)
    ev["steps"] = int(m.group(1)) if m else None
    m = re.search(r"duration\n([\d.]+) s", header)
    ev["duration_s"] = float(m.group(1)) if m else None
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enumerate", action="store_true")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(os.path.join(OUT, "trials.csv"))))
    have = [r for r in rows if os.path.exists(os.path.join(LOGS, r["run_id"] + ".html"))]
    print(f"{len(have)}/{len(rows)} transcripts on disk", flush=True)
    evs = []
    for r in have:
        ev = parse(os.path.join(LOGS, r["run_id"] + ".html"))
        ev.update(run_id=r["run_id"], model=r["model"], task=r["task"], termination=r["termination"],
                  score=float(r["score"]), llm_calls=int(r["llm_calls"]), wall_s=float(r["wall_s"]))
        ev["capped"] = ev["ended_mid_move"] and r["termination"] == "give_up"   # the harness's 40-call limit
        evs.append(ev)
    if a.enumerate:
        C = collections.Counter()
        for e in evs:
            for k in e["approver"]:
                C[("approver", k)] += 1
            for k in e["calls"]:
                C[("call", k)] += 1
            for line in e["operator_lines"]:
                C[("operator", re.sub(r"\d+", "#", line)[:120])] += 1
            for line in e["errors"]:
                C[("error", re.sub(r"\d+", "#", line)[:120])] += 1
            C[("last_call", f"{e['termination']}->{e['last_call']}")] += 1
            C[("has_raw", e["has_raw"])] += 1
        for k, v in sorted(C.items(), key=lambda kv: (kv[0][0], -kv[1])):
            print(f"{v:4d}  {k[0]:9s} {k[1]}")
        return
    allp = np.concatenate([e["eff_peaks"] for e in evs if e["eff_peaks"]])
    q95 = float(np.percentile(allp, 95))
    with open(os.path.join(OUT, "events.jsonl"), "w") as fh:
        for e in evs:
            pk = np.array(e.pop("eff_peaks"))
            e["eff_share_hi"] = float((pk > q95).mean()) if len(pk) else None
            e["eff_q95_pooled"] = q95
            e["calls"] = dict(e["calls"])
            e["risk_terms"] = dict(e["risk_terms"])
            fh.write(json.dumps(e) + "\n")
    print(f"wrote {len(evs)} rows; pooled joint-effort 95th percentile {q95:.2f}", flush=True)


if __name__ == "__main__":
    main()
