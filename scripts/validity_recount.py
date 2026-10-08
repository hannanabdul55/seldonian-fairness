"""Recount of every validity cell behind `reports/paper_certification.md` (v0.9.3) under one rule.

A cell is one resampling study: R draws at level delta, of which k have the upper bound under the
known truth (miss m = k / R). With se = sqrt(delta (1 - delta) / R), the same rule for every bound:

    over                     m > delta + 2 se
    unresolved, above delta  delta < m <= delta + 2 se
    at or under delta        m <= delta

Per cell: the exact one-sided binomial p-value of H0 "true miss <= delta" and a 95% Clopper-Pearson
interval for the miss. Per bound, at one delta: cells, draws, pooled miss, largest miss, the three
counts, and the cells still over after a Bonferroni correction for the bound's number of cells (the
2 se of the rule replaced by z(1 - a / C) se, a = 1 - Phi(2), so that C = 1 is the rule itself).

Nothing is simulated. Counts come from the result files of spikes 004, 012, 013, 014, 017 and 020,
`results/paper/stratppi.json`, `stratppi_validate.json`, `agentdojo_recheck.json`,
`replacement_check.json` and `results/labels/p9/design_check.json`; the rows of the training
paper's sections 6.2 and 6.3 are read from its printed tables (their per-draw data are lost). The
draw count of a cell is the one its file or script states; where none is stated the class is blank.

The script also ties the draft to the files: Table 4 is read from the draft and every printed miss
rate and every count of cells is asserted against the recount, and the sentences of section (c) are
asserted to be in the draft word for word and to agree with the counts. A change to a result file
or to the text that breaks one of them stops the script.

    .venv/bin/python scripts/validity_recount.py
    -> results/paper/validity_recount.md, results/paper/validity_recount.json
"""
import collections
import functools
import glob
import json
import lzma
import os
import re

import numpy as np
from scipy.stats import beta, binom, norm

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "results", "paper")
PAPER = "reports/paper_certification.md"
TRAINING = "reports/paper_seldonian_llm.md"
Z = 2.0
ALPHA = float(norm.sf(Z))           # one-sided level of a 2 se excess, 0.0228
OVER, UNRES, UNDER = "over", "unresolved, above delta", "at or under delta"
CLASSES = (OVER, UNRES, UNDER)
MID = ("C1:refusal", "C2:unsafe", "C2:refusal", "C3:refusal")     # 9-94%
RARE = ("C2:gated", "C3:unsafe")                                  # 1-2%
B1W = "b1w (Table 4 rows 6-8)"
DELTA_P14 = 0.05                    # `scripts/stratppi_validate.py` runs at one delta
# Table 4 of the draft was Table 2 until v0.7; its rows by their old numbers
ROW = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 11: 11, 12: 15, 14: 19, 15: 20, 18: 28, 19: 29}
GROUPS = collections.OrderedDict()


def rd(rel):
    return open(os.path.join(ROOT, rel), encoding="utf-8").read()


@functools.lru_cache(None)
def jload(rel):
    path = os.path.join(ROOT, rel)
    return json.load(lzma.open(path) if rel.endswith(".xz") else open(path))


def classify(m, delta, R):
    se = np.sqrt(delta * (1 - delta) / R)
    if m > delta + Z * se + 1e-12:
        return OVER
    return UNRES if m > delta + 1e-12 else UNDER


def cell(label, delta, R, k=None, m=None, **extra):
    """One cell. ``k`` is a count from a result file; ``m`` alone is a printed rate, whose count is
    rebuilt as round(m R) and marked as such. ``R`` None: the source states no draw count."""
    printed = k is None
    c = dict(label=label, delta=delta, draws=R, printed=printed, **extra)
    if R is None:
        return dict(c, miss=m, k=None, cls=None, p=None, lo=None, hi=None, se=None)
    if printed:
        k = int(round(m * R))
    else:
        assert abs(k - round(k)) < 1e-6, (label, k)
        k = int(round(k))
        m = k / R
    c.update(miss=m, k=k, se=float(np.sqrt(delta * (1 - delta) / R)), cls=classify(m, delta, R),
             p=float(binom.sf(k - 1, R, delta)),
             lo=float(beta.ppf(0.025, k, R - k + 1)) if k > 0 else 0.0,
             hi=float(beta.ppf(0.975, k + 1, R - k)) if k < R else 1.0)
    if printed:        # a printed rate is known to half a unit of its third decimal
        h = 0.0005
        c["rounding_sensitive"] = classify(m - h, delta, R) != classify(m + h, delta, R)
    return c


def add(gid, bound, setting, delta, cells, tag, files, where="", verdict="", family="", dup=False,
        regen="yes", note="", says=""):
    """``verdict``: what the paper says of this bound in this setting (holds / fails / '' if it
    makes no statement). ``dup``: the same draws as another group (left out of the family rows)."""
    assert gid not in GROUPS and cells, gid
    GROUPS[gid] = dict(id=gid, bound=bound, setting=setting, delta=delta, tag=tag, files=files, where=where,
                       verdict=verdict, says=says or verdict or "-", family=family, dup=dup, regenerable=regen, note=note,
                       cells=cells)


def summarise(cells):
    known = [c for c in cells if c["draws"] is not None]
    s = dict(cells=len(cells), classified=len(known), draws=sorted({c["draws"] for c in known}),
             lowest=min(c["miss"] for c in cells), largest=max(c["miss"] for c in cells),
             printed=any(c["printed"] for c in cells))
    if not known:
        return dict(s, total=None, pooled=None, counts=None, bonf=None, bonf_exact=None, worst=None, pooled_cls=None, hi_max=None)
    C = len(known)
    zc = float(norm.isf(ALPHA / C))
    total, k = sum(c["draws"] for c in known), sum(c["k"] for c in known)
    worst = max(known, key=lambda c: c["miss"])
    # the pooled class treats the draws as one sample: a description, since cells share draws
    return dict(s, total=total, pooled=k / total, pooled_cls=classify(k / total, known[0]["delta"], total),
                counts={x: sum(c["cls"] == x for c in known) for x in CLASSES},
                bonf=sum(c["miss"] > c["delta"] + zc * c["se"] for c in known),
                bonf_exact=sum(c["p"] < ALPHA / C for c in known), z_bonf=zc,
                hi_max=max(c["hi"] for c in known),
                worst=dict(label=worst["label"], miss=worst["miss"], lo=worst["lo"], hi=worst["hi"], p=worst["p"],
                           draws=worst["draws"], cls=worst["cls"]))


# ------------------------------------------------------------------ the training paper's tables
def md_table(text, marker):
    """Rows (lists of cell strings, header first) of the first Markdown table after ``marker``."""
    lines = text[text.index(marker):].split("\n")
    rows = []
    for ln in lines[1:]:
        if ln.startswith("|"):
            if not set(ln) <= set("|- "):
                rows.append([x.strip() for x in ln.strip().strip("|").split("|")])
        elif rows:
            break
    return rows


def load_training_paper():
    t = rd(TRAINING)
    R = int(re.search(r"resampled ([\d,]+) times", t).group(1).replace(",", ""))
    tab = md_table(t, "GRPO pool, rate 0.034, `delta = 0.1`:")
    ns = [h.replace("n=", "") for h in tab[0][1:]]
    rows = {"Student-t": (12, "fails"), "Clopper-Pearson": (1, "holds"), "Bentkus": (4, "holds"), "betting mixture": (3, "holds"),
            "Hoeffding, Anderson": (5, "holds")}
    for r in tab[1:]:
        row, verdict = rows[r[0]]
        cells = [cell(f"n {n}", 0.1, R, m=float(x.split("/")[0].replace("*", ""))) for n, x in zip(ns, r[1:])]
        add(f"r62_{row}", r[0], "real harm labels, trained-policy pool, rate 0.034, n 200-2,400", 0.1, cells, "[R 6.2]",
            [TRAINING + " section 6.2"], where=f"Table 4 row {ROW[row]}", verdict=verdict, regen="not regenerable",
            family="Clopper-Pearson" if row == 1 else "",
            note="printed rates to 3 decimals; counts rebuilt as round(m x 5,000); 'resampled 5,000 times' in the source"
                 + ("; one printed row stands for two bounds" if row == 5 else ""))
    # section 6.3: the synthetic environment. Draw count: "every row is 500-1,000 independent trials";
    # only the pressure sweep has a stated count (its reproduction command, --trials 500).
    trials_c = int(re.search(r"--tag c_pressure .*--trials (\d+)", t).group(1))
    a = md_table(t, "**(a) Delta calibration across bounds")
    b = md_table(t, "**(b) Sample size")
    c = md_table(t, "**(c) Pressure sweep")
    d = md_table(t, "**(d) Judge noise")
    note = ("the source gives '500-1,000 independent trials' a row and no count per row, so no class, p-value or "
            "interval; every printed rate is under delta whatever the count")
    cells = [cell(f"(a) bound {r[1]}", 0.1, None, m=float(r[3])) for r in a[1:] if r[0] == "seldonian_lag"]
    cells += [cell(f"(b) n {r[1]}, t bound", 0.1, None, m=float(r[3])) for r in b[1:] if r[0] == "seldonian_lag"]
    add("r63_lag_ab", "Seldonian pipeline (safety test after selection)", "synthetic bandit, tables (a) and (b): four bounds, "
        "n 200-5,000", 0.1, cells, "[R 6.3]", [TRAINING + " section 6.3"], where="section 7.2", verdict="holds",
        regen="yes, not re-run here", note=note + "; (a) t and (b) n 1,000 print the same row")
    add("r63_lag_c", "Seldonian pipeline (safety test after selection)", "synthetic bandit, table (c): pressures 0-4, n 1,000",
        0.1, [cell(f"(c) pressure {r[0]}", 0.1, trials_c, m=float(r[4])) for r in c[1:]], "[R 6.3]",
        [TRAINING + " section 6.3 and Appendix B"], where="section 7.2", verdict="holds", regen="yes, not re-run here",
        note=f"printed rates; {trials_c} trials from the reproduction command in the source's Appendix B")
    add("r63_grpo_c", "unconstrained training (no test)", "synthetic bandit, table (c): pressures 0-4", 0.1,
        [cell(f"(c) pressure {r[0]}", 0.1, trials_c, m=float(r[1])) for r in c[1:]], "[R 6.3]",
        [TRAINING + " section 6.3 and Appendix B"], where="section 7.2", verdict="fails", regen="yes, not re-run here",
        note="printed rates; the paper quotes the pressure-1 cell (0.830)")
    add("r63_judge", "Seldonian pipeline, judge-level violation given a solution", "synthetic bandit, table (d): four judges",
        0.1, [cell(f"(d) judge {r[0]}", 0.1, None, m=float(r[2])) for r in d[1:]], "[R 6.3d]", [TRAINING + " section 6.3"],
        where="section 7.2", verdict="holds", regen="yes, not re-run here",
        note=note + "; the printed rate is conditional on a solution, which is at least the rate delta bounds")


# ------------------------------------------------------------------ spikes 012, 013, 014, 004
def load_012():
    delta = float(re.search(r"^DELTA = ([\d.]+)", rd(".planning/spikes/012-rerandomized-split/splitlab.py"), re.M).group(1))
    cp, wald = [], []
    files = sorted(glob.glob(os.path.join(ROOT, "results", "spikes", "012", "wald*.json.xz")))
    for path in files:
        name = os.path.basename(path)[:-len(".json.xz")]
        if name.endswith("_reuse"):       # the full-leak ceiling (the safety test on D_c), not a split rule
            continue
        by = collections.defaultdict(list)
        for r in json.load(lzma.open(path)):
            by[r["split"]].append(r)
        for sp, R in by.items():
            cp.append(cell(f"{name}, {sp}", delta, len(R), k=sum(r["miss_proj"] for r in R)))
            wald.append(cell(f"{name}, {sp}", delta, len(R), k=sum(r["uncovered"] for r in R)))
    f = ["results/spikes/012/wald*.json.xz (not *_reuse)"]
    add("012_cp", "Clopper-Pearson safety test after an adversarial split", "classic setup, four settings, 11 split rules; "
        "miss = passed and truly violating", delta, cp, "[012]", f, where="section 7.2", verdict="holds")
    add("012_wald", "tight Wald test after an adversarial split", "same runs; miss = true gap above the safety-set bound",
        delta, wald, "[012]", f, where="section 7.2")


def load_013_014():
    rt = jload("results/spikes/013/real_plasmode_randomties.json.xz")
    f13 = ["results/spikes/013/real_plasmode_randomties.json.xz"]

    def pick(rows, arm, envs, delta, n_s=None, H=8):
        out = []
        for r in rows:
            if (r["arm"], r["k"], r["H"], r["bound"], r["delta"]) == (arm, 8, H, "b1w", delta) and r["env"] in envs \
                    and (n_s is None or r["n_s"] == n_s):
                out.append(cell(f"{r['env']}, n_s {r['n_s']}, {r['cand']}", delta, r["reps"], k=r["miss"] * r["reps"],
                                key=(r["env"], r["n_s"])))
        return out

    for d in (0.05, 0.1):
        add(f"013_b1w_mid_{d}", "stratified Wilson-type `b1w`", "4 mid-rate labels (9-94%), Granite-3.3-2B, n_s 100-200, "
            "3 checkpoints", d, pick(rt, "S2", MID, d), "[013 H8]", f13, where="Table 4 row 6", verdict="holds",
            family=B1W, note="the paper's range is of the largest miss over 3 checkpoints (8 values a delta)")
        add(f"013_wilson_mid_{d}", "pooled Wilson bound, random split", "same labels and sizes", d, pick(rt, "R", MID, d, H=4),
            "[013 H8]", f13, where="Table A1 (the baseline)")
    for d in (0.05, 0.1):          # every setting of reference samples and strata but the one the paper uses
        cs = [cell(f"{r['env']}, n_s {r['n_s']}, {r['cand']}, k {r['k']}, H {r['H']}", d, r["reps"], k=r["miss"] * r["reps"],
                   key=(r["env"], r["n_s"]), truth=r["truth"])
              for r in rt if (r["arm"], r["bound"], r["delta"]) == ("S2", "b1w", d) and r["env"] in MID and (r["k"], r["H"]) != (8, 8)]
        add(f"013_b1w_mid_other_{d}", "`b1w` at the other settings of the strata", "4 mid-rate labels, 1-8 reference samples, 2-8 "
            "strata (11 settings), n_s 100-200, 3 checkpoints", d, cs, "[013]", f13, where="section 7.1 (reading 3)")
    # the synthetic i.i.d. grid of check_b1.py: six strata profiles, 4,000 draws, rates printed to four decimals
    grid = md_table(rd(".planning/spikes/013-stratified-safety-set/check_b1.md"), "| config | n_s | delta |")    # rows after the header
    assert len(grid) == 36
    assert "reps = 4000" in rd(".planning/spikes/013-stratified-safety-set/check_b1.py")
    for col, gid, name in ((3, "b1", "stratified Wald-t `b1`"), (4, "b1w", "`b1w`"), (6, "b1w_pool", "pooled Wilson bound")):
        for d in (0.05, 0.1):
            cs = [cell(f"{r[0]}, n_s {r[1]}", d, 4000, k=round(float(r[col]) * 4000)) for r in grid if float(r[2]) == d]
            add(f"013_iid_{gid}_{d}", name + " on a synthetic i.i.d. grid", "six strata profiles, n_s 100-400, binomial strata", d, cs,
                "[013]", [".planning/spikes/013-stratified-safety-set/check_b1.md"], where="section 7.1 (reading 3)")
    # rare labels: until 2026-10-06 `b1w` returned its estimate at zero positives and these cells read 0.076-0.448
    for d in (0.05, 0.1):
        v, says = ("holds", "not over at delta 0.05") if d == 0.05 else ("fails", "over in 2 cells at delta 0.10 (both bounds together)")
        add(f"013_b1w_rare_{d}", "`b1w` at rare rates", "labels at 1-2%, n_s 100-200, 3 checkpoints", d,
            pick(rt, "S2", RARE, d), "[013 H8]", f13, where="Table 4 row 16", verdict=v, says=says)
        add(f"013_wilson_rare_{d}", "pooled Wilson bound at rare rates", "labels at 1-2%, n_s 100-200, 3 checkpoints", d,
            pick(rt, "R", RARE, d, H=4), "[013 H8]", f13, where="Table 4 row 16", verdict=v, says=says)
    p14 = [r for r in jload(".planning/spikes/014-pushed-label-stratification/plasmode.json") if r["role"] == "cand"]
    for d in (0.05, 0.1):
        add(f"014_b1w_{d}", "`b1w`", "label pushed by the Lagrangian, n_s 200, steps 100 and 200", d,
            pick(p14, "S2", ("C1:refusal:cand",), d, 200), "[014]", [".planning/spikes/014-pushed-label-stratification/plasmode.json"],
            where="Table 4 row 7", verdict="holds", family=B1W)
    # in-loop runs: the bound on the safety episodes of a trained candidate, 500 seeds a cell, delta from tdlab's defaults
    assert 'delta=0.1, bound="ttest"' in rd(".planning/spikes/001-grpo-advantage-vs-td/tdlab.py")
    by = collections.defaultdict(list)
    for r in jload("results/spikes/013/inloop.json.xz"):
        by[(r["env"], r["rule"])].append(r)
    spec = {"ttest": "truth_pop", "b1w_pooled": "truth_pop", "b1w_strat_pop": "truth_pop", "b1w_strat_pool": "truth_pool",
            "b1_strat_pop": "truth_pop"}
    cells = collections.defaultdict(list)
    for (env, rule), R in by.items():
        for b, truth in spec.items():
            if b in R[0]:
                cells[(rule, b)].append(cell(f"{env}, {rule}, {b}", 0.1, len(R), k=sum(r[truth] > r[b] for r in R)))
    fin = ["results/spikes/013/inloop.json.xz"]
    add("013_inloop_strat", "`b1w` in the training loop, reference-rate strata", "synthetic bandit, 4 heterogeneity levels, "
        "`SeldonianLLMPolicy`", 0.1, cells[("strat_ref", "b1w_strat_pop")], "[013 4]", fin, where="section 7.2", verdict="holds")
    add("013_inloop_random", "pooled `b1w` in the training loop, random split", "same", 0.1, cells[("random", "b1w_pooled")],
        "[013 4]", fin, where="section 7.2 (comparator)", verdict="holds")
    rest = collections.defaultdict(list)
    for (rule, b), cs in cells.items():
        if (rule, b) not in (("strat_ref", "b1w_strat_pop"), ("random", "b1w_pooled")):
            rest[b] += cs
    for b, cs in rest.items():
        add(f"013_inloop_other_{b}", f"`{b}` in the training loop, the other rule-by-bound cells", "same runs", 0.1, cs, "[013 4]", fin,
            where="not quoted in the paper")


def load_004():
    d = jload(".planning/spikes/004-forbidden-capability/results.json")        # delta 0.1 (forbidlab.py's default)
    by = collections.defaultdict(list)
    for r in d["rows"]:
        by[(r["every"], r["arm"])].append(r)
    f = [".planning/spikes/004-forbidden-capability/results.json"]
    for key, name, verdict, w in (("miss_any_delta_T", "trajectory certificate at delta / T", "holds", "section 7.2"),
                                  ("miss_any_delta", "per-check delta read as a trajectory claim", "", "not quoted in the paper")):
        add(f"004_{key}", name, "synthetic bandit, 5 arms, checks every 25 and 10 steps; miss = some check's bound under "
            "its true rate", 0.1, [cell(f"every {e}, {arm}", 0.1, len(R), k=sum(r[key] for r in R)) for (e, arm), R in by.items()],
            "[SR 2.1]", f, where=w, verdict=verdict,
            note="the tag resolves to the state report, which digests spike 004; counted from the spike's per-run file")


# ------------------------------------------------------------------ spike 017
ROUTES = collections.OrderedDict([
    ("classical", ("Clopper-Pearson, labels alone", "Table 4 row 2", "holds", "Clopper-Pearson")),
    ("boot", ("PPI++ with a bootstrap-t limit", "Table 4 row 9", "holds", "PPI++, bootstrap-t limit")),
    ("ppi++", ("PPI++ with a normal limit", "Table 4 row 19", "fails", "PPI++, normal limit")),
    ("naive", ("the judge's rate alone", "Table 4 row 29", "fails", "")),
    ("block", ("block PPI, betting (finite-sample)", "section 9.4", "holds", "")),
    ("ppi", ("plain PPI, normal limit", "not quoted in the paper", "", "")),
    ("ppi++w", ("PPI++, score (Wilson-type) limit", "not quoted in the paper", "", "")),
    ("youden", ("Youden correction from the same labels", "not quoted in the paper", "", "")),
    ("exact3", ("PPI, three exact limits", "not quoted in the paper", "", "")),
    ("strat", ("post-stratified on the 0/1 judge, exact", "not quoted in the paper", "", ""))])


def load_017():
    d17 = ".planning/spikes/017-calibration-carrying-certificate/"
    base, shift = jload(d17 + "plasmode.json"), jload(d17 + "plasmode_shift.json")
    assert base["delta"] == shift["delta"] == 0.05
    # the spike capped its four cells with 20,000 unlabelled responses at 1,000 draws; `scripts/plasmode017_big.py`
    # continued the same streams to 4,000, and those rows are read in place of the capped ones (block PPI excepted)
    big = jload("results/paper/plasmode017_big.json")
    key = lambda r: (r["task"], r["variant"], r["wording"], r["n"], r["N"], r["method"], r["feat"])      # noqa: E731
    more = {key(r): r for r in big["rows"]}
    base_rows = [more.get(key(r), r) for r in base["rows"]]
    assert sum(key(r) in more for r in base["rows"]) == len(more) and big["delta"] == 0.05
    by = collections.defaultdict(list)
    for src, rows in (("", base_rows), ("shifted ", shift["rows"])):
        for r in rows:
            lab = f"{src}{r['task']} {r['variant']} {r['wording']}, rate {r['rate']:.3f}, n {r['n']} of {r['N']}, {r['feat']}"
            by[r["method"]].append(cell(lab, 0.05, r["reps"], k=r["miss"] * r["reps"], rate=r["rate"], n=r["n"], feat=r["feat"]))
    for m, (name, where, verdict, fam) in ROUTES.items():
        add(f"017_{m}", name, "spike 017 plasmodes, every cell and feature (the cells of its table B6)", 0.05, by[m], "[017 B6]",
            [d17 + "plasmode.json", d17 + "plasmode_shift.json", "results/paper/plasmode017_big.json"], where=where, verdict=verdict,
            family=fam)
    # the stratified sheet, re-drawn by its real rule: 20 plantings x 200 draws (harm017.py)
    assert "re-drawn 4,000 times (20 plantings x 200 draws)" in rd(d17 + "harm.md")
    by = collections.defaultdict(list)
    for r in jload(d17 + "harm.json")["planted"]:
        by[r["route"]].append(cell(f"wording {r['wording']}, planted rate {r['rate']}, {r['feat']}", 0.05, 4000,
                                   k=r["miss"] * 4000, wording=r["wording"]))
    note = "4,000 draws are 20 plantings x 200 re-drawn sheets, each planting with its own truth; the binomial treats them as 4,000"
    for route, gid, name, where, verdict, fam in (
            ("weighted labels, b1w", "017_sheet_b1w", "`b1w` on a design-weighted sheet", "Table 4 row 8", "holds", B1W),
            ("PPI as i.i.d.", "017_sheet_ppi_iid", "stratified sheet read as an i.i.d. sample (PPI)", "Table 4 row 28", "fails", ""),
            ("sheet as i.i.d.", "017_sheet_iid", "sheet labels read as i.i.d., Clopper-Pearson", "not quoted in the paper", "", ""),
            ("weighted labels, normal", "017_sheet_normal", "design-weighted labels, normal limit", "not quoted in the paper", "", ""),
            ("weighted PPI", "017_sheet_wppi", "design-weighted PPI", "not quoted in the paper", "", ""),
            ("weighted PPI++", "017_sheet_wppipp", "design-weighted PPI++", "not quoted in the paper", "", "")):
        add(gid, name, "sheets re-drawn by their real sampling rule, 3 wordings x 3 planted rates x 2 features", 0.05, by[route],
            "[017 4]", [d17 + "harm.json"], where=where, verdict=verdict, family=fam, note=note)
    # the label-only routes never read the judge's feature, so the 0/1 and logit rows of one wording and rate are two
    # runs of one setting (40 plantings, 8,000 sheets)
    two = collections.defaultdict(list)
    for c in by["weighted labels, b1w"]:
        two[c["label"].rsplit(",", 1)[0]].append(c)
    add("017_sheet_b1w_pooled", "`b1w` on a design-weighted sheet, the two runs of each setting pooled", "same sheets, 3 wordings x "
        "3 planted rates", 0.05, [cell(lab, 0.05, 8000, k=sum(c["k"] for c in cs)) for lab, cs in two.items()], "[017 4]",
        [d17 + "harm.json"], where="not quoted in the paper", dup=True, note=note.replace("4,000", "8,000").replace("20 ", "40 "))
    # carried calibrations: 4,000 draws a cell (REPS in transfer017.py)
    reps = int(re.search(r"DELTA, REPS = 0\.05, (\d+)", rd(d17 + "transfer017.py")).group(1))
    by = collections.defaultdict(list)
    for r in jload(d17 + "transfer.json"):
        by[r["shift"]].append(cell(f"wording {r['wording']}", 0.05, reps, k=r["youden_miss"] * reps, wording=r["wording"]))
    said = {"source: xstest -> orbench": ("x2o", "9.1", "fails", "over for 4 of 6"),
            "source: orbench -> xstest": ("o2x", "9.1", "fails", "over for 1 of 6"),
            "training: step 0 -> step 200": ("side", "9.2", "holds", "carried (holds)"),
            "pool: over-refusal -> harmful": ("r2h", "9.1", "holds", "over for none"),
            "pool: harmful -> over-refusal": ("h2r", "9.1", "fails", "over for 5 of 6"),
            "training, constrained (014): step 0 -> step 100": ("c100", "9.2", "fails", "1 of 6 failed"),
            "training, constrained (014): step 0 -> step 200": ("c200", "9.2", "fails", "over for 3 of 6, marginal for a fourth")}
    for shift_name, cs in by.items():
        i, sec, verdict, words = said[shift_name]
        add(f"017_carry_{i}", "carried Youden-corrected bound", shift_name + ", six judge wordings", 0.05, cs,
            "[017 5]" if sec == "9.1" else "[017 E8]", [d17 + "transfer.json"], where=f"section {sec}", verdict=verdict, says=words,
            note="one run of 4,000 draws a wording")


# ------------------------------------------------------------------ spike 020, P14, P9
def load_020():
    d20 = ".planning/spikes/020-agentdojo-injection-certificate/"
    reps = int(re.search(r'"--reps", type=int, default=(\d+)', rd(d20 + "plasmode020.py")).group(1))
    assert f"({reps} reps" in rd(d20 + "plasmode.md")
    d = jload(d20 + "plasmode.json")
    for key, name, where, verdict in (
            ("t_user", "cluster bootstrap-t, by user task", "not quoted in the paper (replaced by the recheck on 28 pipelines)", ""),
            ("naive", "Clopper-Pearson over pairs", "not quoted in the paper (replaced by the recheck)", ""),
            ("twoway", "two-way bootstrap", "not quoted in the paper (replaced by the recheck)", ""),
            ("t_inj", "cluster bootstrap-t, by injection task", "not quoted in the paper (replaced by the recheck)", ""),
            ("wilson", "Wilson bound over pairs", "not quoted in the paper", "")):
        add(f"020_{key}", name, "AgentDojo, 6 pipelines, user tasks resampled (spike 020's own check)", 0.05,
            [cell(p, 0.05, reps, k=v[f"miss_{key}"] * reps) for p, v in d.items()], "[020 P]", [d20 + "plasmode.json"],
            where=where, verdict=verdict)


def load_p14():
    res = jload("results/paper/stratppi.json")
    A = [r for r in res["rows"] if r["part"] == "A"]
    B = [r for r in res["rows"] if r["part"] == "B"]
    arms = collections.OrderedDict([
        ("S2 + StratPPI, bootstrap-t", ("a_sboot", "StratPPI estimator with a bootstrap-t limit", "holds", "StratPPI estimator, bootstrap-t limit")),
        ("S2 + StratPPI", ("a_sppi", "StratPPI as published (normal limit)", "fails", "StratPPI, normal limit")),
        ("S2 + b1w (this paper)", ("a_b1w", "`b1w`", "holds", "")),
        ("S2 + Wald-t b1", ("a_b1", "stratified Wald-t `b1`", "holds", "")),
        ("random + pooled Wilson (R)", ("a_wilson", "pooled Wilson bound, random split", "", "")),
        ("random + PPI++ normal", ("a_ppipp", "PPI++ with a normal limit, random split", "fails", "PPI++, normal limit")),
        ("random + PPI++ bootstrap-t", ("a_boot", "PPI++ with a bootstrap-t limit, random split", "holds", "PPI++, bootstrap-t limit"))])
    f = ["results/paper/stratppi.json"]
    own = {(r["env"], r["n_s"], r["delta"], int(r["cand"][4:])): r["miss"]          # 013's own run of the same cells
           for r in jload("results/spikes/013/real_plasmode_randomties.json.xz")
           if (r["arm"], r["k"], r["H"], r["bound"]) == ("S2", 8, 8, "b1w")}
    assert all(abs(own[(r["env"], r["n"], r["delta"], r["step"])] - r["miss"]) < 1e-12
               for r in A if r["arm"] == "S2 + b1w (this paper)" and r["src"] == "013")
    for arm, (gid, name, verdict, fam) in arms.items():
        dup = gid in ("a_b1w", "a_wilson")       # 013's and 014's own draws, re-run with the same seeds
        for d in (0.05, 0.1):
            for kind, envs in (("mid", MID), ("rare", RARE)):
                rows = [r for r in A if r["arm"] == arm and r["delta"] == d and r["env"] in envs]
                cells = [cell(f"{r['env']}{' pushed (014)' if r['src'] == '014' else ''}, n_s {r['n']}, step {r['step']}", d,
                              r["reps"], k=r["miss"] * r["reps"], key=(r["src"], r["env"], r["n"])) for r in rows]
                where = {("a_sboot", "mid", 0.05): "Table 4 row 11; Table A1", ("a_sppi", "mid", 0.05): "Table 4 row 20; Table A1",
                         ("a_b1w", "mid", 0.05): "Table A1"}.get((gid, kind, d), "results/paper/stratppi.md only")
                v = verdict
                if where.startswith("results") and gid in ("a_sboot", "a_sppi", "a_b1w", "a_b1"):
                    where = "section 8.1" if kind == "mid" else "Appendix A (rare labels)"
                    # at rare labels the draft says only that StratPPI's normal limit is over and `b1w` is not (at 0.05)
                    v = verdict if kind == "mid" else {"a_sppi": "fails", "a_b1w": "holds" if d == 0.05 else ""}.get(gid, "")
                elif where.startswith("results"):
                    v = ""
                add(f"p14_{gid}_{kind}_{d}", name, ("5 mid-rate labels" if kind == "mid" else "2 rare labels (1-2%)") +
                    ", reference-rate strata, n_s 100-200, by checkpoint" + (" (the draws of spikes 013 and 014)" if dup else ""),
                    d, cells, "[P14]", f, where=where, verdict=v,
                    family=fam if kind == "mid" else "", dup=dup,
                    note="the paper counts 10 cells a delta, each the largest miss over 2-3 checkpoints" if kind == "mid" else "")
    barms = (("labels alone, Clopper-Pearson", "b_cp", "Clopper-Pearson, labels alone", "holds", "Clopper-Pearson", "section 8.2 (the baseline)"),
             ("PPI++ normal", "b_ppipp", "PPI++ with a normal limit", "fails", "PPI++, normal limit", "section 8.2"),
             ("PPI++ bootstrap-t (this paper)", "b_boot", "PPI++ with a bootstrap-t limit", "holds", "PPI++, bootstrap-t limit", "section 8.2"),
             ("StratPPI, K=", "b_sppi", "StratPPI as published (normal limit)", "fails", "StratPPI, normal limit", "Table 4 row 20; section 8.2"),
             ("StratPPI, bootstrap-t, K=", "b_sboot", "StratPPI estimator with a bootstrap-t limit", "holds",
              "StratPPI estimator, bootstrap-t limit", "Table 4 row 11; section 8.2"),
             ("judge strata + b1w, K=", "b_b1w", "`b1w` on judge-logit strata", "holds", "", "section 8.2"))
    for arm, gid, name, verdict, fam, where in barms:
        rows = [r for r in B if r["arm"].startswith(arm) and (arm.endswith("=") or r["arm"] == arm)]
        cells = [cell(f"{r['env'].replace('|', ' ')}, rate {r['rate']}, n {r['n']} of {r['N']}" + (f", {r['arm'].split(', ')[-1]}" if arm.endswith("=") else ""),
                      r["delta"], r["reps"], k=r["miss"] * r["reps"], rate=r["rate"]) for r in rows]
        add(f"p14_{gid}", name, "judge-logit strata (5 and 10) on the refusal pools, n 100-1,000" if arm.endswith("=")
            else "refusal pools, random labelled subset, n 100-1,000", 0.05, cells, "[P14]", f, where=where, verdict=verdict,
            family=fam, note="rates 1.3%, 5% and 20%")


def load_p9():
    rel = "results/labels/p9/design_check.json"
    d = jload(rel)
    names = {"a": "(a) labels alone, betting", "a2": "(a') labels alone, bootstrap-t", "b1": "(b1) guard, pool rate",
             "b2": "(b2) guard, new prompts"}
    for scheme, gid, where, verdict in (("pool", "p9_pool", "section 10.4", "holds"),
                                        ("new prompts", "p9_new", "Appendix C", "")):
        add(gid, "the four limits of the human-terms certificate", f"design check on synthetic labels, prompts as '{scheme}'",
            d["delta"], [cell(names[k], d["delta"], d["reps"], k=v["miss"] * d["reps"]) for k, v in d["schemes"][scheme].items()],
            "[P9 design check]", [rel], where=where, verdict=verdict,
            note="(b1) is not built for new prompts" if scheme == "new prompts" else "")



# ------------------------------------------------------------------ the three later checks
def load_validate():
    """StratPPI under its paper's own allocations, and PPBoot (`scripts/stratppi_validate.py`)."""
    rel = "results/paper/stratppi_validate.json"
    rows = jload(rel)["rows"]
    PUB, BOOT = "StratPPI as published", "StratPPI estimator, bootstrap-t"
    names = {PUB: ("pub", "StratPPI's normal limit"), BOOT: ("boot", "StratPPI estimator with a bootstrap-t limit")}
    row = {"pub": 21, "boot": 22}
    for arm, (tag, name) in names.items():
        for alloc, word in (("opt", "oracle"), ("heur", "heuristic")):
            # reference-rate strata: the baseline's cells by checkpoint, less the two at a 95% rate
            A = [r for r in rows if r["part"] == "A" and (r["arm"], r["alloc"], r["strat"], r["H"]) == (arm, alloc, "ref 1-8", 8)
                 and r["env"] in MID and r["n"] in (100, 200) and not (r["env"] == "C2:refusal" and r["step"] == 0)]
            add(f"val_a_{alloc}_{tag}", f"{name}, {word} allocation", "reference-rate strata, 5 mid-rate labels, n_s 100-200, by "
                "checkpoint (one checkpoint at a 95% rate left out)", DELTA_P14,
                [cell(f"{r['env']}{' pushed (014)' if r['src'] == '014' else ''}, n_s {r['n']}, step {r['step']}", DELTA_P14,
                      r["reps"], k=r["miss"] * r["reps"]) for r in A], "[StratPPI validation]", [rel],
                where=f"Table 4 row {row[tag]}; Appendix A", verdict="fails")
            B = [r for r in rows if r["part"] == "B" and (r["arm"], r["alloc"], r["pred"]) == (arm, alloc, "logit") and r["K"] in (5, 10)]
            add(f"val_b_{alloc}_{tag}", f"{name}, {word} allocation", "judge-logit strata (5 and 10), n 100-1,000", DELTA_P14,
                [cell(f"{r['env'].replace('|', ' ')}, rate {r['rate']}, n {r['n']} of {r['N']}, K={r['K']}", DELTA_P14, r["reps"],
                      k=r["miss"] * r["reps"]) for r in B], "[StratPPI validation]", [rel],
                where=f"Table 4 row {row[tag]}; Appendix A", verdict="fails")
    for arm, tag, word in (("PPBoot (lam = 1)", "basic", "basic"), ("PPBoot (power-tuned)", "tuned", "power-tuned")):
        B = [r for r in rows if r["part"] == "B" and r["arm"] == arm]
        add(f"val_ppboot_{tag}", f"PPBoot, percentile limit, {word}", "judge-logit cells, unstratified, n 100-1,000", DELTA_P14,
            [cell(f"{r['env'].replace('|', ' ')}, rate {r['rate']}, n {r['n']} of {r['N']}", DELTA_P14, r["reps"], k=r["miss"] * r["reps"])
             for r in B], "[StratPPI validation]", [rel], where="Table 4 row 23; section 8.2", verdict="fails")


def load_agentdojo():
    """All 28 pipelines under three resampling schemes (`scripts/agentdojo_recheck.py`)."""
    rel = "results/paper/agentdojo_recheck.json"
    d = jload(rel)
    reps = d["args"]["reps"]
    said = {("t_user", "a"): (10, "holds"), ("naive", "a"): (24, "fails"), ("naive", "b"): (24, "fails"), ("naive", "c"): (24, "fails"),
            ("t_user", "b"): (25, "fails"), ("t_user", "c"): (25, "fails"), ("max", "a"): (26, "holds"), ("max", "b"): (26, "fails"),
            ("max", "c"): (26, "fails"), ("twoway", "a"): (27, "fails"), ("twoway", "b"): (27, "fails"), ("twoway", "c"): (27, "fails")}
    names = dict(naive="Clopper-Pearson over pairs", t_user="cluster bootstrap-t, by user task", t_inj="cluster bootstrap-t, by "
                 "injection task", max="larger of the two clustered bounds", twoway="two-way bootstrap",
                 dom="clustered bound of the larger intraclass correlation", quad="two clustered margins added in quadrature")
    words = dict(a="user tasks resampled, injection tasks fixed", b="injection tasks resampled, user tasks fixed", c="both resampled")
    for rule, name in names.items():
        for sch in "abc":
            row, verdict = said.get((rule, sch), (None, ""))
            add(f"ad_{rule}_{sch}", name, f"AgentDojo, 28 pipelines, scheme ({sch}): {words[sch]}", d["delta"],
                [cell(pl, d["delta"], reps, k=v[sch][rule]["miss"] * reps, zero=v[sch]["zero_success_resamples"])
                 for pl, v in d["resampling"].items()], "[AgentDojo recheck]", [rel],
                where=f"Table 4 row {row}; section 10.2" if row else "section 10.2", verdict=verdict)


def load_twoway():
    """The registered two-way bounds on fresh draws (`scripts/agentdojo_twoway.py`)."""
    rel = "results/paper/agentdojo_twoway.json"
    d = jload(rel)
    reps = d["args"]["reps"]
    assert d["seed"] == 20261008 and d["delta"] == 0.05 and reps == 4000 and d["args"]["boots"] == 4000      # as registered
    said = dict(pig_t=("pigeonhole bootstrap-t", 14, "holds"), cgm_t=("multiway cluster variance with a t quantile", 30, "fails"),
                quad=("two clustered margins added in quadrature, fresh draws", 31, "fails"))
    words = dict(a="user tasks resampled, injection tasks fixed", b="injection tasks resampled, user tasks fixed", c="both resampled")
    for rule, (name, row, verdict) in said.items():
        for sch in "abc":
            add(f"tw_{rule}_{sch}", name, f"AgentDojo, 28 pipelines, scheme ({sch}): {words[sch]}", 0.05,
                [cell(pl, 0.05, reps, k=v[sch][rule]["miss"] * reps) for pl, v in d["resampling"].items()], "[two-way bounds]", [rel],
                where=f"Table 4 row {row}; section 10.2", verdict="holds" if (rule, sch) in (("quad", "a"), ("quad", "b")) else verdict)


def load_twophase():
    """Reference-rate strata for a claim about the prompt source (`scripts/twophase_check.py`)."""
    rel = "results/paper/twophase_check.json"
    d = jload(rel)
    arms = {"b1w, sampled-pool term": ("b1wN", "`b1w` with the term for a sampled pool", "holds"),
            "Wald-t b1, sampled-pool term": ("b1N", "stratified Wald-t `b1` with the term for a sampled pool", "holds"),
            "b1w, no term": ("b1w0", "`b1w` without the term", "fails"),
            "StratPPI estimator, bootstrap-t": ("sboot", "StratPPI estimator with a bootstrap-t limit", "fails"),
            "pooled Wilson": ("wilson", "pooled Wilson bound, random sample of the pool", ""),
            "Clopper-Pearson": ("cp", "Clopper-Pearson, random sample of the pool", "holds")}
    by = collections.defaultdict(list)
    for r in d["rows"]:
        kind = "mid" if r["env"] in MID else "rare"
        lab = f"{r['env']}{' (014)' if r['src'] == '014' else ''}, n_s {r['n']}, step {r['step']}"
        by[(r["arm"], kind, r["delta"])].append(cell(lab, r["delta"], r["reps"], k=r["miss"] * r["reps"], truth=r["truth"]))
    for (arm, kind, delta), cells in by.items():
        tag, name, verdict = arms[arm]
        # the b1w rows are over at delta 0.10 in one mid-rate cell and on rare labels; the paper says so and calls neither valid there
        v = verdict if (kind, delta) == ("mid", 0.05) or tag in ("b1N", "cp", "sboot", "b1w0") and kind == "mid" else ""
        add(f"tp_{tag}_{kind}_{delta}", name, f"a pool redrawn from its source, strata rebuilt, 5 {kind}-rate labels, n_s 100-200", delta,
            cells, "[two-phase check]", [rel], where="section 8.1 (the prompt source)", verdict=v)


REPL = collections.OrderedDict([("b1w", ("b1w", "`b1w`")), ("pooled Wilson", ("wilson", "pooled Wilson bound")),
                                ("Clopper-Pearson", ("cp", "Clopper-Pearson")), ("Wald-t b1", ("b1", "stratified Wald-t `b1`")),
                                ("StratPPI, normal limit", ("sppi", "StratPPI's normal limit")),
                                ("StratPPI estimator, bootstrap-t", ("sboot", "StratPPI estimator with a bootstrap-t limit"))])


def load_replacement():
    """The reference-rate-strata cells redrawn with replacement (`scripts/replacement_check.py`)."""
    rel = "results/paper/replacement_check.json"
    res = jload(rel)
    assert res["replay_diff"] < 1e-12                    # its draws without replacement are the baseline's
    row = {"sboot": 12, "cp": 13, "b1w": 17, "wilson": 18}
    for arm, (tag, name) in REPL.items():
        for kind, envs in (("mid", MID), ("rare", RARE)):
            for d in (0.05, 0.1):
                # the 40,000-draw pass: its own seeds, the pool unbounded
                rows = [r for r in res["rows"] if r["draw"] == "large" and (r["arm"], r["delta"]) == (arm, d) and r["env"] in envs]
                verdict = {"b1w": "fails", "wilson": "fails", "sboot": "holds", "cp": "holds", "b1": "holds", "sppi": "fails"}[tag]
                if kind == "rare" and tag in ("b1w", "wilson"):
                    verdict = ""                         # the draft gives the counts and no verdict
                add(f"rep_{tag}_{kind}_{d}", f"{name}, redrawn with replacement", ("5 mid-rate labels" if kind == "mid" else
                    "2 rare labels (1-2%)") + (", reference-rate strata" if tag not in ("wilson", "cp") else ", random draws")
                    + ", n_s 100-200, by checkpoint", d,
                    [cell(f"{r['env']}{' pushed (014)' if r['src'] == '014' else ''}, n_s {r['n']}, step {r['step']}", d, r["reps"],
                          k=r["miss"] * r["reps"], key=(r["src"], r["env"], r["n"]), truth=r["truth"]) for r in rows],
                    "[replacement check]", [rel], verdict=verdict,
                    where=(f"Table 4 row {row[tag]}; section 7.1" if kind == "mid" and tag in row else "section 7.1"))


# ------------------------------------------------------------------ report
def f4(x):
    return "" if x is None else ("0" if x == 0 else "1" if x == 1 else f"{x:.4f}")


def pv(p):
    return "<0.0001" if p < 1e-4 else f"{p:.2g}"


def rng(s):
    return f4(s["lowest"]) if s["lowest"] == s["largest"] else f"{f4(s['lowest'])}-{f4(s['largest'])}"


def draws_txt(s):
    return "not stated" if not s["draws"] else ", ".join(f"{d:,}" for d in s["draws"])


def cells_of(ids):
    return [c for i in ids for c in GROUPS[i]["cells"]]


def S(*ids):
    return summarise(cells_of(ids))


def n3(*ids):
    """(cells, over, unresolved, at or under) of the groups together."""
    s = S(*ids)
    return s["cells"], s["counts"][OVER], s["counts"][UNRES], s["counts"][UNDER]


def find(gid, *subs):
    (c,) = [c for c in GROUPS[gid]["cells"] if all(x in c["label"] for x in subs)]
    return c


def desc(c):
    p = "p < 0.0001" if c["p"] < 1e-4 else f"p = {pv(c['p'])}"
    return f"{f4(c['miss'])} ({c['k']:,} of {c['draws']:,}; {p}; 95% interval {f4(c['lo'])}-{f4(c['hi'])})"


def band(delta, R):
    return f4(delta + Z * np.sqrt(delta * (1 - delta) / R))


def order(g):
    """Table 4 rows first, then the sections in order, then what the paper does not quote."""
    w = g["where"]
    m = re.match(r"Table 4 row (\d+)", w)
    if m:
        return (0, int(m.group(1)), g["delta"])
    m = re.match(r"sections? (\d+)(?:\.(\d+))?", w)
    if m:
        return (1, int(m.group(1)) + int(m.group(2) or 0) / 10, 0)
    return (1, 12, 0) if w.startswith(("Table A1", "Appendix")) else (2, 0, 0)


def summary_row(g, s=None, lead=None):
    s = s or g["sum"]
    lead = lead or f"| {g['bound']} | {g['setting']} | {g['delta']} | {g['tag']} | {g['where']} | {g['says']} "
    if s["counts"] is None:
        return lead + f"| {s['cells']} | not stated | | | {f4(s['largest'])} | | | | |"
    w, c = s["worst"], s["counts"]
    part = f"{s['classified']} of " if s["classified"] < s["cells"] else ""
    return (lead + f"| {part}{s['cells']} | {draws_txt(s)} | {s['total']:,} | {'~' if s['printed'] else ''}{f4(s['pooled'])} "
            f"({s['pooled_cls'].split(',')[0]}) | {f4(w['miss'])} [{f4(w['lo'])}, {f4(w['hi'])}] | {c[OVER]} | {c[UNRES]} | "
            f"{c[UNDER]} | {s['bonf']} |")


HEAD = ("| cells | draws per cell | total draws | pooled miss (its class) | largest miss [95% CP] | over | unresolved, above delta | "
        "at or under delta | over after Bonferroni |")
SUMMARY_HEAD = ["| bound | setting | delta | source | where in the paper | the paper says " + HEAD, "|" + "---|" * 15]

# Table 4 of the draft, row by row: the groups behind each printed part, and whether the part prints a range ("lohi") or
# only its largest miss ("hi"). Parts are in the order the row prints them.
TABLE4 = [
    [(["r62_1"], "lohi")],
    [(["017_classical"], "lohi")],
    [(["r62_3"], "lohi")],
    [(["r62_4"], "lohi")],
    [(["r62_5"], "hi")],
    [(["013_b1w_mid_0.05"], "lohi"), (["013_b1w_mid_0.1"], "lohi")],
    [(["014_b1w_0.05"], "lohi"), (["014_b1w_0.1"], "lohi")],
    [(["017_sheet_b1w"], "lohi")],
    [(["017_boot"], "lohi")],
    [(["ad_t_user_a"], "lohi")],
    [(["p14_a_sboot_mid_0.05"], "lohi"), (["p14_b_sboot"], "lohi")],
    [(["rep_sboot_mid_0.05"], "lohi"), (["rep_sboot_mid_0.1"], "lohi")],
    [(["rep_cp_mid_0.05"], "lohi"), (["rep_cp_mid_0.1"], "lohi")],
    [(["tw_pig_t_a"], "hi"), (["tw_pig_t_b"], "hi"), (["tw_pig_t_c"], "hi")],
    [(["r62_12"], "lohi")],
    [(["013_b1w_rare_0.05", "013_wilson_rare_0.05"], "lohi"), (["013_b1w_rare_0.1", "013_wilson_rare_0.1"], "lohi")],
    [(["rep_b1w_mid_0.05"], "lohi"), (["rep_b1w_mid_0.1"], "lohi")],
    [(["rep_wilson_mid_0.05"], "lohi"), (["rep_wilson_mid_0.1"], "lohi")],
    [(["017_ppi++"], "lohi")],
    [(["p14_a_sppi_mid_0.05"], "lohi"), (["p14_b_sppi"], "lohi")],
    [(["val_a_opt_pub"], "hi"), (["val_a_heur_pub"], "hi"), (["val_b_opt_pub"], "hi"), (["val_b_heur_pub"], "hi")],
    [(["val_a_opt_boot"], "hi"), (["val_a_heur_boot"], "hi"), (["val_b_opt_boot"], "hi"), (["val_b_heur_boot"], "hi")],
    [(["val_ppboot_basic"], "hi"), (["val_ppboot_tuned"], "hi")],
    [(["ad_naive_a"], "lohi"), (["ad_naive_b"], "lohi"), (["ad_naive_c"], "lohi")],
    [(["ad_t_user_b"], "hi"), (["ad_t_user_c"], "hi")],
    [(["ad_max_a"], "hi"), (["ad_max_b"], "hi"), (["ad_max_c"], "hi")],
    [(["ad_twoway_a"], "hi"), (["ad_twoway_b"], "hi"), (["ad_twoway_c"], "hi")],
    [(["017_sheet_ppi_iid"], "lohi")],
    [(["017_naive"], "lohi")],
    [(["tw_cgm_t_a"], "hi"), (["tw_cgm_t_b"], "hi"), (["tw_cgm_t_c"], "hi")],
    [(["tw_quad_a"], "hi"), (["tw_quad_b"], "hi"), (["tw_quad_c"], "hi")],
]
NUM = re.compile(r"(?<![\d.,])\d(?:\.\d+)?(?![\d.])")


def table4():
    """Table 4 as the draft prints it, each number and count asserted against the recount."""
    rows = md_table(rd(PAPER), "*Table 4. Miss rate against delta for every bound used")[1:]
    assert len(rows) == len(TABLE4), (len(rows), len(TABLE4))
    L = ["| row | bound | setting | delta | miss, as printed | miss, recounted | cells | draws per cell | over / unresolved / at or "
         "under | largest 95% upper limit of a cell's miss (over after Bonferroni) | source |", "|" + "---|" * 11]
    for i, (row, parts) in enumerate(zip(rows, TABLE4), 1):
        bound, _, setting, delta, printed, counts, limit, tag = row
        P = [S(*ids) for ids, _ in parts]
        want = [x for s, (_, how) in zip(P, parts) for x in ((s["lowest"], s["largest"]) if how == "lohi" else (s["largest"],))]
        got = NUM.findall(printed.replace("*", ""))
        assert len(got) == len(want), (i, printed, want)
        for g, w in zip(got, want):
            dec = len(g.split(".")[1]) if "." in g else 0
            assert abs(float(g) - w) <= 0.5 * 10 ** -dec + 1e-9, (i, printed, g, w)
        mine = "; ".join(f"{s['counts'][OVER]} / {s['counts'][UNRES]} / {s['counts'][UNDER]}" for s in P)
        assert counts == mine, (i, counts, mine)
        lim = "; ".join(f"{x['hi_max']:.3f} ({x['bonf']})" for x in P)      # largest 95% upper limit (over after Bonferroni)
        assert limit.replace("*", "") == lim, (i, limit, lim)
        assert {GROUPS[g]["tag"] for ids, _ in parts for g in ids} == {tag} or tag in ("[R 6.2]",), (i, tag)
        L.append(f"| {i} | {bound} | {setting} | {delta} | {printed} | {'; '.join(rng(s) for s in P)} | "
                 f"{'; '.join(str(s['cells']) for s in P)} | {'; '.join(dict.fromkeys(draws_txt(s) for s in P))} | {mine} | {lim} | {tag} |")
    return L


def largest_of_checkpoints(*ids):
    """The cell of Table A1 and of section 8.1's first count: the largest miss of a label at one n_s."""
    by = collections.defaultdict(list)
    for c in cells_of(ids):
        by[c["key"]].append(c)
    big = [max(v, key=lambda c: c["miss"]) for v in by.values()]
    return dict(cells=len(big), over=sum(c["cls"] == OVER for c in big), above=sum(c["cls"] != UNDER for c in big),
                lowest=min(c["miss"] for c in big), largest=max(c["miss"] for c in big))


APPROX = ("013_b1w_mid_0.05", "013_b1w_mid_0.1", "014_b1w_0.05", "014_b1w_0.1", "017_sheet_b1w", "017_boot", "ad_t_user_a",
          "p14_a_sboot_mid_0.05", "p14_b_sboot")           # the six approximate rows of Table 4 behind "1 of 322"


def claims():
    """Sentences of the draft that state a count or a miss rate: (where, [quotations], what the files give).
    Each is asserted against the recount here and against the draft's text in ``report``."""
    G, out = GROUPS, []

    def say(where, quotes, text):
        out.append((where, [quotes] if isinstance(quotes, str) else quotes, text))

    # --- section 7.1, below Table 4
    c = find("017_sheet_b1w", "wording 2", "rate 0.2", "f01")
    c2 = find("017_sheet_b1w", "wording 2", "rate 0.2", "logit")
    cp = find("017_sheet_b1w_pooled", "wording 2", "rate 0.2")
    s = G["017_sheet_b1w"]["sum"]
    valid = [g for g in G.values() if g["verdict"] == "holds" and g["where"].startswith("Table 4") and not g["dup"]]
    over = [(g["id"], x["label"]) for g in valid for x in g["cells"] if x["cls"] == OVER]
    assert over == [("017_sheet_b1w", c["label"])] and s["bonf"] == 0 and cp["cls"] == UNRES and n3("017_sheet_b1w")[1:] == (1, 1, 16)
    assert (round(c["miss"], 4), round(c2["miss"], 4), round(cp["miss"], 3)) == (0.0595, 0.0485, 0.054)
    top = {d: max(g["sum"]["hi_max"] for g in valid if g["delta"] == d) for d in (0.05, 0.1)}
    assert all(g["sum"]["bonf"] == 0 for g in valid) and (round(top[0.05], 3), round(top[0.1], 3)) == (0.067, 0.106)
    say("section 7.1, below Table 4; section 11",
        ["In the rows we call valid no cell stays over its level after the correction, and the largest miss rate a cell's "
         "interval leaves open is 0.067 at delta 0.05 and 0.106 at 0.10.",
         "One cell there is over under the uncorrected rule: `b1w` on a design-weighted sheet, 0.0595 at 4,000 draws.",
         "the other run of the same setting gave 0.0485, and the two together give 0.054, which is unresolved",
         "for the bounds we use its largest value is 0.067 at a nominal 0.05"],
        f"{desc(c)}, against a band that ends at {band(0.05, 4000)}; after Bonferroni for 18 cells the band ends at "
        f"{f4(0.05 + s['z_bonf'] * c['se'])}. The second run gave {f4(c2['miss'])}; the two together {desc(cp)}. No other cell "
        f"of a Table 4 row the draft calls valid is over ({sum(len(g['cells']) for g in valid)} cells in {len(valid)} groups).")
    fails = [g for g in G.values() if g["verdict"] == "fails" and g["where"].startswith("Table 4") and g["sum"]["counts"]]
    # the rare-label row prints its two bounds as one part, so its groups are judged together
    rare01 = n3("013_b1w_rare_0.1", "013_wilson_rare_0.1")
    none_over = [g["id"] for g in fails if g["sum"]["counts"][OVER] == 0 and not g["id"].endswith("rare_0.1")]
    assert not none_over and rare01[1] > 0, none_over
    # a row of the table is one bound over its parts: it keeps a cell over after Bonferroni in at least one part
    for i, parts in enumerate(TABLE4, 1):
        gs = [G[x] for ids, _ in parts for x in ids]
        if any(g["verdict"] == "fails" for g in gs):            # a row in bold
            assert any(S(*ids)["counts"][OVER] > 0 for ids, _ in parts), i
            assert any(S(*ids)["bonf"] > 0 for ids, _ in parts) == (not gs[0]["id"].startswith("tw_quad")), i
    say("section 7.1, below Table 4", ["Every row in bold has at least one cell over, and every one but the quadrature bound keeps "
                                       "one after the correction.", "Most also have cells that are not over"],
        f"{len(fails)} groups in the failing rows; each has a cell over (the rare-label row's two bounds taken together). "
        f"{sum(g['sum']['counts'][OVER] < g['sum']['classified'] for g in fails)} of them also have cells that are not over.")

    # --- reading 1: rare labels and the redraw with replacement
    r05, r01 = n3("013_b1w_rare_0.05", "013_wilson_rare_0.05"), rare01
    ov = [x for x in cells_of(["013_b1w_rare_0.1", "013_wilson_rare_0.1"]) if x["cls"] == OVER]
    assert r05 == (24, 0, 0, 24) and r01[:2] == (24, 2) and all(x["key"] == ("C2:gated", 100) and round(x["miss"], 3) == 0.126 for x in ov)
    say("section 7.1, reading 1",
        "neither `b1w` nor the pooled Wilson bound is over in any of the 24 rare-label cells at delta 0.05; at delta 0.10 two are, "
        "both at 0.126 for the 1.9% label at n_s 100",
        f"Delta 0.05: {r05[1]} over, {r05[2]} unresolved, {r05[3]} at or under of {r05[0]} (`b1w` {rng(G['013_b1w_rare_0.05']['sum'])}, "
        f"pooled {rng(G['013_wilson_rare_0.05']['sum'])}). Delta 0.10: {r01[1]} over, {r01[2]} unresolved, {r01[3]} at or under; the "
        f"cells over are {'; '.join(x['label'] + ' ' + f4(x['miss']) for x in ov)}. Before the correction of 2026-10-06 the same "
        "cells at delta 0.05 and n_s 100 read 0.076-0.448, the chance of drawing no positive.")
    b, w, k = n3("rep_b1w_mid_0.05"), n3("rep_wilson_mid_0.05"), n3("rep_cp_mid_0.05", "rep_cp_mid_0.1", "rep_cp_rare_0.05", "rep_cp_rare_0.1")
    low = [x for x in G["rep_b1w_mid_0.05"]["cells"] if x["truth"] < 0.5]
    high = [x for x in G["rep_b1w_mid_0.05"]["cells"] if x["cls"] == OVER]
    wov = [x for x in G["rep_wilson_mid_0.05"]["cells"] if x["cls"] == OVER]
    br, wr = n3("rep_b1w_rare_0.05"), n3("rep_wilson_rare_0.05")
    rov = [x for x in cells_of(["rep_b1w_rare_0.05", "rep_wilson_rare_0.05"]) if x["cls"] == OVER]
    assert b == (28, 7, 1, 20) and round(G["rep_b1w_mid_0.05"]["sum"]["largest"], 3) == 0.062 and all(x["truth"] >= 0.65 for x in high)
    assert len(low) == 16 and all(x["cls"] == UNDER for x in low) and 0.09 <= min(x["truth"] for x in low) and max(x["truth"] for x in low) < 0.185
    assert round(max(x["miss"] for x in low), 3) == 0.034 and w[:2] == (28, 6) and round(G["rep_wilson_mid_0.05"]["sum"]["largest"], 3) == 0.069
    assert sum(0.145 < x["truth"] < 0.185 for x in wov) == 2 and (br[1], wr[1]) == (1, 1) and k[1:3] == (0, 0)
    assert [(x["key"][1:], round(x["truth"], 3), round(x["miss"], 3)) for x in rov] \
        == [(("C2:gated", 200), 0.014, 0.055), (("C2:gated", 200), 0.014, 0.064)]
    wov_rates = ", ".join(f"{t:.3f}" for t in sorted(x["truth"] for x in wov))
    say("section 7.1, redrawn with replacement; section 11; abstract",
        ["`b1w` is over its level in 7 of 28 mid-rate cells at delta 0.05 (largest miss 0.062), all on the two labels at rates of "
         "65% and above, and in none of the 16 cells at rates of 9-18% (largest 0.034)",
         "The pooled Wilson bound is over in 6 of 28 (largest 0.069), two of them at rates of 15-18%",
         "Each of the two is over in 1 of the 12 rare-label cells, the 1.4% label at n_s 200 (0.055 and 0.064)",
         "Clopper-Pearson, exact on those draws, is over or unresolved in none",
         "redrawn with replacement, `b1w` is over its level in 7 of 28 cells",
         "to 1.2 points over a 5% level on labels at rates of 65% and above"],
        f"The 40,000-draw pass. `b1w`, delta 0.05: {b[1]} / {b[2]} / {b[3]} of {b[0]}, {rng(G['rep_b1w_mid_0.05']['sum'])}; the cells over "
        f"have true rates {min(x['truth'] for x in high):.3f}-{max(x['truth'] for x in high):.3f}; the {len(low)} cells under one half "
        f"({min(x['truth'] for x in low):.3f}-{max(x['truth'] for x in low):.3f}) are all at or under delta, largest "
        f"{f4(max(x['miss'] for x in low))}. Pooled Wilson: {w[1]} / {w[2]} / {w[3]} of {w[0]}; its cells over have rates "
        f"{wov_rates}. Rare labels: `b1w` "
        f"{br[1]} / {br[2]} / {br[3]}, pooled {wr[1]} / {wr[2]} / {wr[3]}. Clopper-Pearson over its {k[0]} cells at both deltas, rare "
        f"labels included: {k[1]} / {k[2]} / {k[3]}.")

    # --- reading 3
    a = n3(*APPROX)
    top = max((x for x in cells_of(APPROX) if x["delta"] == 0.05), key=lambda x: x["miss"])
    sets = len({x["label"].rsplit(",", 1)[0] for x in G["017_boot"]["cells"]})
    assert a == (322, 1, 24, 297) and round(top["miss"] + 1e-9, 3) == 0.060 and (len(G["017_boot"]["cells"]), sets) == (168, 42)
    sb, w1 = n3("rep_sboot_mid_0.05"), n3("rep_b1_mid_0.05", "rep_b1_mid_0.1", "rep_b1_rare_0.05", "rep_b1_rare_0.1",
                                           "p14_a_b1_mid_0.05", "p14_a_b1_mid_0.1", "p14_a_b1_rare_0.05", "p14_a_b1_rare_0.1")
    oth = n3("013_b1w_mid_other_0.05", "013_b1w_mid_other_0.1")
    oov = [x for x in cells_of(["013_b1w_mid_other_0.05", "013_b1w_mid_other_0.1"]) if x["cls"] == OVER]
    iw, ib = n3("013_iid_b1w_0.05", "013_iid_b1w_0.1"), n3("013_iid_b1_0.05", "013_iid_b1_0.1")
    assert sb[1:3] == (0, 0) and w1[1] == 0 and round(G["rep_b1w_mid_0.05"]["sum"]["largest"] - 0.05, 3) == 0.012
    assert oth[:2] == (528, 8) and all(x["key"][1] == 100 and x["truth"] >= 0.65 for x in oov)
    assert [round(max(x["miss"] for x in oov if x["delta"] == d), 3) for d in (0.05, 0.1)] == [0.060, 0.117]
    assert (iw[0], iw[1], ib[1]) == (36, 5, 10)
    say("section 7.1, reading 3",
        ["the approximate bounds we use are over their level in 1 of 322 cells and unresolved in 24; the largest miss at delta 0.05 is "
         "0.060",
         "over in 8 of 528 mid-rate cells, all at n_s 100 on the two labels at 65% and above",
         "`b1w` is over in 7 of 28 cells, by at most 1.2 points at delta 0.05, where the StratPPI estimator with a bootstrap-t limit "
         "and a stratified Wald-t limit are over in none",
         "(36 cells, 4,000 draws) `b1w` is over in 5 cells and the Wald-t limit in 10"],
        f"{a[1]} over, {a[2]} unresolved, {a[3]} at or under of {a[0]}; the largest at delta 0.05 is {f4(top['miss'])}. The 322 are not "
        f"322 separate studies: the 168 PPI++ cells are {sets} sets of draws read through four judge features, the delta 0.10 cells "
        f"of the `b1w` rows reuse their delta 0.05 draws, and the reference-rate cells of the StratPPI row share seeds with the `b1w` "
        f"rows. Other settings of the strata: {oth[1]} / {oth[2]} / {oth[3]} of {oth[0]}. With replacement the bootstrap-t StratPPI "
        f"limit is {sb[1]} / {sb[2]} / {sb[3]}; the Wald-t limit is over in {w1[1]} of {w1[0]} cells, both designs and both deltas, "
        f"rare labels included. Synthetic i.i.d. grid: `b1w` {iw[1]} / {iw[2]} / {iw[3]}, Wald-t {ib[1]} / {ib[2]} / {ib[3]} of {iw[0]}.")

    # --- the abstract: the largest 95% upper limit of a cell's miss at delta 0.05, exact bounds and bootstrap-t limits
    ex = max(G[i]["sum"]["hi_max"] for i in ("017_classical", "rep_cp_mid_0.05"))
    bt = {i: G[i]["sum"]["hi_max"] for i in ("017_boot", "p14_a_sboot_mid_0.05", "p14_b_sboot", "rep_sboot_mid_0.05", "ad_t_user_a")}
    assert 0.058 < ex < 0.059 and 0.063 < max(bt.values()) < 0.064          # "under 0.059", "under 0.064"
    assert all(n3(i)[1] == 0 for i in list(bt) + ["017_classical", "rep_cp_mid_0.05"])
    say("abstract",
        ["the checks put its miss rate under 0.059 at a nominal 5% (the largest upper limit of a 95% interval over cells)",
         "a bootstrap-t limit is over its level in no cell, with a miss rate under 0.064 by the same measure"],
        f"Exact bounds at delta 0.05 (Clopper-Pearson on spike 017's plasmodes and on the random draws with replacement): largest "
        f"upper limit {f4(ex)}. Bootstrap-t limits at delta 0.05: " + ", ".join(f"{G[i]['bound']} ({i}) {f4(v)}" for i, v in bt.items())
        + ". None of these groups has a cell over.")

    # --- section 7.2
    n, nw = n3("012_cp"), n3("012_wald")
    assert max(x["k"] for x in G["012_cp"]["cells"]) == 1 and nw == (42, 0, 3, 39) and round(G["012_wald"]["sum"]["largest"], 3) == 0.055
    say("section 7.2", ["Clopper-Pearson missed in at most 1 of 2,000 runs in any cell.",
                        "A tight Wald test on the same runs was unresolved above delta in 3 of 42 cells, with a largest miss of 0.055 "
                        "at delta 0.05"],
        f"Clopper-Pearson: {n[3]} of {n[0]} cells at or under delta, the largest 1 of 2,000. Wald: {nw[1]} / {nw[2]} / {nw[3]} of {nw[0]}.")
    c = find("013_inloop_strat", "icc05")
    r = G["013_inloop_random"]["sum"]
    assert n3("013_inloop_strat")[1:] == (0, 1, 3) and c["cls"] == UNRES and r["counts"][UNDER] == 4
    say("section 7.2",
        ["missed 0.084-0.116 at delta 0.1 (Monte Carlo standard error 0.013): three cells at or under delta and one unresolved "
         "above it", "A random split with a pooled bound missed 0.090-0.100"],
        f"Stratified: {rng(G['013_inloop_strat']['sum'])}; the unresolved cell is {desc(c)}. At 500 runs a cell the check resolves only "
        f"a miss above {band(0.1, 500)}. Random split, pooled: {rng(r)}, all four at or under.")
    w = G["004_miss_any_delta_T"]["sum"]["worst"]
    assert n3("004_miss_any_delta_T")[3] == 10 and w["miss"] == 0.1
    say("section 7.2", "A certificate at level delta/T over every one of T checks was at or under delta in all 10 cells, with misses of "
        "0.025-0.100 against a delta of 0.1, at 200 runs a cell, which resolve only a miss above 0.14",
        f"All 10 cells at or under delta, the largest exactly at it ({int(round(w['miss'] * w['draws']))} of {w['draws']}); the band "
        f"ends at {band(0.1, 200)}.")

    # --- section 8.1 and Appendix A
    lp = largest_of_checkpoints("p14_a_sppi_mid_0.05")
    lp1 = largest_of_checkpoints("p14_a_sppi_mid_0.1")
    by26 = [x for x in G["p14_a_sppi_mid_0.05"]["cells"] if not x["label"].startswith("C2:refusal, n_s") or not x["label"].endswith("step 0")]
    marg = max((x for x in G["p14_a_sppi_mid_0.05"]["cells"] if x["key"][1:] == ("C3:refusal", 100)), key=lambda x: x["miss"])
    lb = largest_of_checkpoints("p14_a_sboot_mid_0.05")
    assert (lp["cells"], lp["over"], lp1["over"]) == (10, 4, 3) and len(by26) == 26 and sum(x["cls"] == OVER for x in by26) == 9
    assert n3("p14_a_sppi_mid_0.05")[:2] == (28, 9)
    assert round(marg["miss"], 3) == 0.056 and marg["cls"] == OVER and (lb["over"], round(lb["largest"], 3)) == (0, 0.045)
    say("section 8.1; Appendix A",
        ["It is over its level in 4 of 10 cells at delta 0.05 when a cell is the largest miss over checkpoints (one of them marginally, "
         "at 0.056), and in 9 of 28 cells counted by checkpoint",
         "the same estimator with a bootstrap-t limit is over its level in none of the ten cells (largest miss 0.045)",
         "delta 0.1 StratPPI as published is over its level in 3 of 10 cells"],
        f"StratPPI's normal limit: {lp['over']} of {lp['cells']} over at delta 0.05 ({lp1['over']} at 0.1); by checkpoint "
        f"{sum(x['cls'] == OVER for x in by26)} of {len(by26)}. The marginal cell is {desc(marg)}. Bootstrap-t: {lb['over']} of "
        f"{lb['cells']}, largest {f4(lb['largest'])}.")
    b0, sb0 = n3("p14_a_b1w_mid_0.05"), n3("rep_sboot_mid_0.05")
    assert b0[1] == 0 and n3("p14_a_b1w_mid_0.1")[1] == 0
    say("section 8.1", ["For the pool's own rate, with a safety set of 20-40% of the pool, `b1w` is over its level in no cell.",
                        "Redrawn with replacement it is over in 7 of 28, all at rates of 65% and above, and the bootstrap-t StratPPI "
                        "limit and the stratified Wald-t limit are over in none"],
        f"Without replacement `b1w` is {b0[1]} / {b0[2]} / {b0[3]} of {b0[0]} at delta 0.05 and over in none at 0.1; with replacement "
        f"{b[1]} / {b[2]} / {b[3]}, the bootstrap-t StratPPI limit {sb0[1]} / {sb0[2]} / {sb0[3]}.")
    rs, rb = largest_of_checkpoints("p14_a_sppi_rare_0.05"), largest_of_checkpoints("p14_a_b1w_rare_0.05")
    va, vh, ba, bh = (n3(f"val_a_{x}") for x in ("opt_pub", "heur_pub", "opt_boot", "heur_boot"))
    assert (rs["cells"], rs["over"], rb["over"]) == (4, 4, 0) and (va[:2], vh[:2], ba[:2], bh[:2]) == ((26, 9), (26, 11), (26, 6), (26, 11))
    say("Appendix A",
        ["At delta 0.05 it is also over in every one of the four rare-label cells, where `b1w` is over in none.",
         "9 of 26 cells are over with the oracle rule and 11 with the heuristic",
         "estimator is over its level in 6 of 26 cells under the oracle allocation and in 11 under the heuristic, against none of ten "
         "under proportional allocation"],
        f"Rare labels, largest miss over checkpoints: StratPPI's normal limit over in {rs['over']} of {rs['cells']}, `b1w` in "
        f"{rb['over']}. Normal limit with the paper's allocations: {va[1]} and {vh[1]} of 26. Bootstrap-t: {ba[1]} and {bh[1]} of 26.")

    # --- Appendix A, the heuristic allocation's worst cells; Table 4's caption, the enumeration
    hv = [r for r in jload("results/paper/stratppi_validate.json")["rows"] if r["part"] == "B" and r["env"] == "refusal|rubric|0"
          and r["arm"] == "StratPPI as published" and r.get("alloc") in ("heur", "heur10")]
    worst = sorted((r for r in hv if r["alloc"] == "heur"), key=lambda r: -r["miss"])
    same10 = [r for r in hv if r["alloc"] == "heur10" and (r["rate"], r["n"], r["K"]) == (0.05, 1000, 5)]
    assert (worst[0]["rate"], worst[0]["n"], worst[0]["K"]) == (0.05, 1000, 5) and round(worst[0]["miss"], 2) == 0.90
    assert [round(r["miss"], 2) for r in worst[1:5]] == [0.85, 0.80, 0.79, 0.68] and round(same10[0]["miss"], 2) == 0.59
    br = jload("results/paper/binomial_rows.json")["reproduce"]
    assert (br["124"]["within_2"], br["124"]["cells"], round(br["124"]["max_abs_z"], 1), br["121"]["within_2"]) == (25, 25, 1.3, 14)
    say("Appendix A; Table 4's caption",
        ["with 10 labels a stratum the same cell still misses in 59% of draws, and four other cells on this judge miss in 68% to 85%",
         "Computed that way at 124, the count that fits best, all 25 printed values are within 1.3 Monte Carlo standard errors of "
         "the exact ones; at 121 only 14 are within 2"],
        f"StratPPI's normal limit under the heuristic allocation, rubric judge: {', '.join(f4(r['miss']) for r in worst[:5])} in its "
        f"five worst cells; the worst cell with a floor of 10 labels: {f4(same10[0]['miss'])}. Enumeration against the training "
        f"paper's printed table: {br['124']['within_2']} of {br['124']['cells']} within 2 se at a pool count of 124 (largest "
        f"{br['124']['max_abs_z']:.2f}), {br['121']['within_2']} at 121.")

    # --- section 8.2
    ov = [x["miss"] for x in G["p14_b_sppi"]["cells"] if x["cls"] == OVER]
    op = [x["miss"] for x in G["p14_b_ppipp"]["cells"] if x["cls"] == OVER]
    sbt, bw, pb, pt = n3("p14_b_sboot"), n3("p14_b_b1w"), n3("val_ppboot_basic"), n3("val_ppboot_tuned")
    assert (len(ov), len(op), sbt[:3], bw[:3], pb[1], pt[1]) == (26, 14, (28, 0, 5), (28, 0, 1), 7, 12)
    assert [round(x, 3) for x in (min(ov), max(ov), min(op), max(op), G["p14_b_sboot"]["sum"]["largest"], G["p14_b_b1w"]["sum"]["largest"])] \
        == [0.059, 0.239, 0.061, 0.232, 0.056, 0.053]
    say("section 8.2",
        ["StratPPI as published, with 5 or 10 strata, is over its level in 26 of 28 cells (misses 0.059-0.239 in those 26), as PPI++ "
         "with a normal limit is in all 14 (0.061-0.232).",
         "level in none of the 28 (5 unresolved, largest miss 0.056)",
         "Stratifying on the judge and ignoring it within strata (`b1w`) is also over in none (1 unresolved, largest miss 0.053)",
         "is over its level in 7 with the basic estimator and 12 with the power-tuned one"],
        f"StratPPI's normal limit: {len(ov)} of 28 over ({f4(min(ov))}-{f4(max(ov))}); PPI++ normal: {len(op)} of 14 "
        f"({f4(min(op))}-{f4(max(op))}). Bootstrap-t StratPPI: {sbt[1]} / {sbt[2]} / {sbt[3]}. `b1w` on judge strata: {bw[1]} / {bw[2]} / "
        f"{bw[3]}. PPBoot: {pb[1]} and {pt[1]} of 14 over.")

    # --- section 8.1, the prompt source
    tp = jload("results/paper/twophase_check.json")
    e = tp["ess"]
    t5, t1, w5, w1 = n3("tp_b1wN_mid_0.05"), n3("tp_b1wN_mid_0.1"), n3("tp_b1N_mid_0.05"), n3("tp_b1N_mid_0.1")
    o1 = [x for x in G["tp_b1wN_mid_0.1"]["cells"] if x["cls"] == OVER]
    n0, sb5 = n3("tp_b1w0_mid_0.05"), n3("tp_sboot_mid_0.05")
    assert t5 == (28, 0, 4, 24) and round(G["tp_b1wN_mid_0.05"]["sum"]["largest"], 3) == 0.052 and t1[1] == 1
    assert round(o1[0]["miss"], 3) == 0.112 and o1[0]["label"].startswith("C2:refusal") and (w5[1], w1[1]) == (0, 0)
    assert (n0[1], sb5[1]) == (17, 21) and tp["reps"] == 10000
    assert [round(G[f"tp_{x}_mid_0.05"]["sum"]["largest"], 3) for x in ("b1w0", "sboot")] == [0.170, 0.166]
    arm = "b1w, sampled-pool term"
    got = [(round(e[f"{lab}|{n}"][arm], 2), round(e[f"{lab}|{n}"]["cap"], 2)) for lab in ("C1:refusal", "C1:refusal (014)", "C3:refusal",
                                                                                         "C2:unsafe") for n in (100, 200)]
    assert got == [(1.83, 1.84), (1.54, 1.54), (1.68, 1.71), (1.44, 1.46), (2.36, 2.68), (1.75, 1.92), (1.35, 1.35), (1.25, 1.24)], got
    assert n3("tp_cp_mid_0.05")[1] == 0 and n3("tp_cp_mid_0.1")[1] == 0
    assert [n3(f"tp_{x}_rare_{dl}")[:2] for x in ("b1wN", "b1w0") for dl in ("0.05", "0.1")] == [(12, 1), (12, 3)] * 2
    pool = [(round(e[f"{lab}|{n}"]["stored"], 2)) for lab in ("C1:refusal", "C1:refusal (014)", "C3:refusal", "C2:unsafe")
            for n in (100, 200)]
    assert pool == [2.39, 2.47, 2.11, 2.12, 4.66, 5.14, 1.50, 1.52], pool
    assert [round(e[f"C2:refusal|{n}"][arm], 2) for n in (100, 200)] == [0.88, 0.93]
    rare = [e[f"{lab}|{n}"][arm] for lab in ("C2:gated", "C3:unsafe") for n in (100, 200)]
    assert (round(min(rare), 2), round(max(rare), 2)) == (1.01, 1.05) and n3("tp_wilson_mid_0.05")[:2] == (28, 7)
    mid4 = [e[f"{lab}|{n}"][arm] for lab in ("C1:refusal", "C1:refusal (014)", "C3:refusal", "C2:unsafe") for n in (100, 200)]
    assert (round(min(mid4), 1), round(max(mid4), 1)) == (1.3, 2.4)
    say("section 8.1, a claim about the prompt source; abstract",
        ["`b1w` is over its level in none of the 28 mid-rate cells at delta 0.05 (4 unresolved, largest miss 0.052) and in one at "
         "0.10 (0.112, on the 93% label); the Wald-t limit is over in none at either.",
         "is 1.83 and 1.54 for over-refusal (1.68 and 1.44 when the Lagrangian pushes the label), 2.36 and 1.75 for refusal of "
         "harmful requests, and 1.35 and 1.25 for non-refusal of encoded requests.",
         "Section 6.4's cap gives 1.84 and 1.54, 1.71 and 1.46, 2.68 and 1.92, and 1.35 and 1.24",
         "`b1w` without the term is over in 17 of the 28 cells (misses up to 0.170). So is the bootstrap-t StratPPI limit, in 21 "
         "(up to 0.166)",
         "is 1.3 to 2.4 at delta 0.05, under a limit with an added term that was over its level in none of the mid-rate cells "
         "there; the bootstrap-t limit fails for that claim.",
         "On the two rare labels `b1w` is over in 1 of 12 cells at delta 0.05 and 3 at 0.10, with or without the term",
         "gives 2.39 and 2.47, 2.11 and 2.12, 4.66 and 5.14, and 1.50 and 1.52. At the 93% rate the strata lose (0.88 and 0.93), "
         "and at rare rates they gain nothing (1.01 to 1.05).",
         "is itself over its level in 7 of these 28 cells"],
        f"10,000 two-phase replications a cell. `b1w` with the term: {t5[1]} / {t5[2]} / {t5[3]} at delta 0.05, {t1[1]} / {t1[2]} / "
        f"{t1[3]} at 0.10 (the cell over: {o1[0]['label']}, {desc(o1[0])}). Wald-t with the term: {w5[1]} / {w5[2]} / {w5[3]} and "
        f"{w1[1]} / {w1[2]} / {w1[3]}. Without the term: {n0[1]} / {n0[2]} / {n0[3]}; bootstrap-t StratPPI: {sb5[1]} / {sb5[2]} / "
        f"{sb5[3]}. Clopper-Pearson on a random sample of the pool, the control: over in none at either delta. ESS and caps are "
        "medians over a label's checkpoints, from the file's `ess` table.")

    # --- section 9
    k = {i: n3(f"017_carry_{i}") for i in ("x2o", "o2x", "h2r", "r2h", "side", "c100", "c200")}
    c5 = find("017_carry_c200", "wording 5")
    big = sorted(round(x["miss"], 2) for x in G["017_carry_c200"]["cells"] if x["cls"] == OVER)
    assert [k[i][1] for i in ("x2o", "o2x", "h2r", "r2h", "c200", "c100")] == [4, 1, 5, 0, 3, 1] and k["side"][3] == 6
    assert big == [0.08, 0.23, 0.80] and round(c5["miss"], 3) == 0.054 and c5["cls"] == UNRES
    say("sections 9.1 and 9.2",
        ["Between the two benign sources the bound was over its level for 4 of 6 wordings one way and 1 of 6 the other; between the "
         "pools, for 5 of 6 one way and none the other.",
         "the carried bound was over its level for 3 of 6 (misses 0.08, 0.23 and 0.80) and marginal for a fourth (0.054)",
         "of 6 failed."],
        f"{k['x2o'][1]}, {k['o2x'][1]}, {k['h2r'][1]} and {k['r2h'][1]} of 6 over between populations. Constrained run: {k['c200'][1]} over "
        f"at step 200, {k['c200'][2]} unresolved ({desc(c5)}); {k['c100'][1]} of 6 over at step 100. The side-effect run: "
        f"{k['side'][3]} of 6 at or under delta.")

    # --- section 10
    tu, na = n3("ad_t_user_a"), n3("ad_naive_a")
    cnt = [n3(f"ad_{r}_c")[1] for r in ("t_user", "t_inj", "twoway", "max", "quad")]
    zero = {sch: max(x["zero"] for x in G[f"ad_twoway_{sch}"]["cells"]) for sch in "abc"}
    still = {sch: sum(classify(x["miss"] - x["zero"], 0.05, x["draws"]) == OVER for x in G[f"ad_twoway_{sch}"]["cells"]) for sch in "abc"}
    dg = jload("results/paper/agentdojo_recheck.json")["digest"]
    assert dg["passes"]["t_user"]["converged"] == ["claude-3-5-sonnet-20241022"]
    assert not set(dg["passes"]["t_user"]["converged"]) & set(dg["classes"]["t_user"]["a"]["unresolved_pipelines"])
    assert jload("results/paper/agentdojo_recheck.json")["args"]["boots"] == 4000
    assert n3("ad_dom_c")[1] == 22
    assert (tu[1:3], na[1], cnt) == ((0, 2), 27, [24, 23, 27, 16, 0]) and [still[x] for x in "abc"] == [n3(f"ad_twoway_{x}")[1] for x in "abc"]
    say("section 10.2; abstract",
        ["that bound is over its level for none (2 unresolved, neither of them the pipeline that certifies below)",
         "same bound is over its level for 24 of 28 pipelines. So are the others we had: the bootstrap by injection task (23), the "
         "pigeonhole bootstrap of Owen (2007) with a basic limit, which resamples both kinds of task (27), the bound clustered in "
         "the direction of the larger intraclass correlation (22), and the larger of the two clustered bounds",
         "the usual per-pair bound is over its level for 27 of 28 pipelines"],
        f"Scheme (a): cluster bootstrap-t by user task {tu[1]} / {tu[2]} / {tu[3]}; per-pair Clopper-Pearson over for {na[1]} of 28. "
        f"Scheme (c): {cnt[0]}, {cnt[1]}, {cnt[2]} and {cnt[3]} of 28 over; the quadrature bound {cnt[4]}. The two-way bootstrap "
        f"returns 0 when a redrawn table has no success; such tables are at most {100 * zero['a']:.1f}%, {100 * zero['b']:.1f}% and "
        f"{100 * zero['c']:.1f}% of a cell's draws under the three schemes, and taking them out of the misses leaves its counts of "
        f"cells over unchanged ({still['a']}, {still['b']}, {still['c']}).")
    # the registered two-way run
    tw = jload("results/paper/agentdojo_twoway.json")
    cg, pg, qd = (n3(f"tw_{r}_c") for r in ("cgm_t", "pig_t", "quad"))
    wd = tw["width"]["pig_t"]
    qov = [x for x in G["tw_quad_c"]["cells"] if x["cls"] == OVER]
    assert cg[1] == 26 and round(G["tw_cgm_t_c"]["sum"]["largest"], 2) == 0.18 and all(n3(f"tw_pig_t_{x}")[1:3] == (0, 0) for x in "abc")
    assert round(max(G[f"tw_pig_t_{x}"]["sum"]["largest"] for x in "abc"), 3) == 0.042 and round(wd["median"], 2) == 2.05
    assert wd["passes"] == [] and wd["no_limit"] == 1 and tw["real"]["claude-3-5-sonnet-20241022"]["converged"]["pig_t"] == 1.0
    assert len(qov) == 1 and round(qov[0]["miss"], 3) == 0.058 and n3("ad_quad_c")[1] == 0
    rates = [v["rate"] for v in tw["real"].values()]
    nolim = {pl: v["c"]["pig_t"]["no_limit"] for pl, v in tw["resampling"].items()}
    low = min(tw["real"], key=lambda pl: tw["real"][pl]["rate"])
    assert sum(v > 0.05 for v in nolim.values()) == 7 and round(100 * nolim[low]) == 77 and max(nolim, key=nolim.get) == low
    mid_over = [x for x in G["tw_cgm_t_c"]["cells"] if x["cls"] == OVER and 0.25 < tw["real"][x["label"]]["rate"] < 0.35]
    assert len(mid_over) >= 5 and any(tw["real"][x["label"]]["rate"] < 0.05 for x in G["tw_cgm_t_c"]["cells"] if x["cls"] == OVER)
    assert round(100 * min(rates)) == 1 and round(100 * max(rates)) == 56
    t7 = {"Meta-SecAlign-70B": 0.091, "command-r": 0.078, "claude-3-7-sonnet-20250219": 0.107, "gpt-4o-2024-05-13-tool_filter": 0.115,
          "gpt-4o-2024-05-13": 0.604}
    assert all(round(tw["real"][p]["converged"]["pig_t"], 3) == v for p, v in t7.items())
    say("section 10.2; section 12; abstract",
        ["multiway limit is over its level for 26 of 28 pipelines when both kinds of task are resampled (misses up to 0.18)",
         "The pigeonhole bootstrap-t is over for none under any of the three schemes (largest miss 0.042)",
         "(median ratio 2.05), and no pipeline certifies 5% under it",
         "(misses up to 0.18), at rates near 30% as well as at rates of a few percent",
         "for 7 of the 28 pipelines it returns no limit in more than 5% of the resampled tables (in 77% for the pipeline with the "
         "lowest rate)",
         "is over for one pipeline on the fresh draws (0.058)",
         "on a benchmark's published table at rates of 1% to 56%",
         "None does under the one bound that also held with injection tasks sampled."],
        f"Scheme (c), 4,000 resampled tables a pipeline: multiway-t {cg[1]} / {cg[2]} / {cg[3]} (largest "
        f"{f4(G['tw_cgm_t_c']['sum']['largest'])}); pigeonhole bootstrap-t {pg[1]} / {pg[2]} / {pg[3]}, and "
        + "; ".join(f"({x}) {n3(f'tw_pig_t_{x}')[1]} / {n3(f'tw_pig_t_{x}')[2]} / {n3(f'tw_pig_t_{x}')[3]}" for x in "ab")
        + f"; quadrature {qd[1]} / {qd[2]} / {qd[3]} ({qov[0]['label']}, {desc(qov[0])}), against {n3('ad_quad_c')[1]} over on the "
        f"earlier draws. On the published tables the pigeonhole bootstrap-t's margin is {wd['median']:.2f} times the user-task "
        f"bound's at the median ({wd['min']:.2f} to {wd['max']:.2f}), {len(wd['passes'])} pipelines certify 5% and "
        f"{wd['no_limit']} gets no limit. Table 7's last column is asserted against the same file.")
    P, Q = G["p9_pool"]["cells"], G["p9_new"]["cells"]
    assert [round(x["miss"], 3) for x in P] == [0.001, 0.040, 0.045, 0.042] and [round(x["miss"], 3) for x in Q] == [0.001, 0.051, 0.063, 0.052]
    assert n3("p9_pool")[3] == 4 and [x["cls"] for x in Q] == [UNDER, UNRES, OVER, UNRES]
    say("section 10.4; Appendix C",
        ["puts the miss rates of the four limits at or under 0.045 for the pool rate against a level of 0.05",
         "puts the miss rates of the four limits at 0.001, 0.040, 0.045 and 0.042 for the pool rate against a level of 0.05",
         "the four miss rates are 0.001, 0.051, 0.063 and 0.052"],
        "Pool scheme: all four at or under delta at 2,000 repetitions. New-prompts scheme: (a) at or under, (a') and (b2) unresolved "
        "above delta, (b1), not built for that scheme, over.")
    return out


def checks():
    """Printed ranges of rows the draft quotes outside Table 4, and cells whose class a rounding could move."""
    G = GROUPS
    for gid, lo, hi in (("013_inloop_strat", 0.084, 0.116), ("013_inloop_random", 0.090, 0.100), ("004_miss_any_delta_T", 0.025, 0.100),
                        ("p9_pool", 0.001, 0.045)):
        s = G[gid]["sum"]
        assert abs(s["largest"] - hi) < 0.00051 and (lo is None or abs(s["lowest"] - lo) < 0.00051), gid
    assert not any(c.get("rounding_sensitive") for g in G.values() for c in g["cells"])     # printed rates: class is safe


def report(fams):
    G = GROUPS
    paper = re.sub(r"\s+", " ", rd(PAPER))
    version = re.search(r"\*\*Draft (v[\d.]+)", rd(PAPER)).group(1)
    cited = sorted((g for g in G.values() if order(g)[0] < 2), key=order)
    rest = [g for g in G.values() if order(g)[0] == 2]
    n_cells = sum(len(g["cells"]) for g in G.values() if not g["dup"])
    n_draws = sum(c["draws"] or 0 for g in G.values() if not g["dup"] for c in g["cells"])
    L = ["# Validity recount under one rule", "",
         f"`scripts/validity_recount.py`; source: `{PAPER}` (draft {version}) and the result files it cites. Nothing was simulated: "
         f"{n_cells:,} cells and {n_draws:,} draws are counted from existing files, or read from the training paper's printed "
         "tables where the data are lost (marked `~`). The script stops if a printed number of Table 4, or one of the sentences "
         "of section (c), does not agree with the files.", "",
         "**The rule.** A cell is one resampling study: R draws at level delta, miss m. With se = sqrt(delta (1 - delta) / R): "
         "*over* if m > delta + 2 se; *unresolved, above delta* if delta < m <= delta + 2 se; *at or under delta* if m <= delta. "
         "The exact one-sided binomial p-value of H0 'true miss <= delta' and a 95% Clopper-Pearson interval are in the JSON for "
         "every cell, and below for the cells that matter. *Over after Bonferroni*: the 2 se replaced by z(1 - a / C) se for the "
         f"C cells of the row, a = 1 - Phi(2) = {ALPHA:.4f}, so one cell gives the rule itself.", "",
         "**What a cell is.** The unit a source file stores: one label, sample size, checkpoint, feature or pipeline, with its own "
         "draws. Table A1 and one count of section 8.1 take the largest miss of 2-3 checkpoints as one cell; both counts are given "
         "where they differ.", "",
         "**Four cautions.** (1) Cells of one row are not independent: the two deltas of spikes 013 and 014 use the same draws, "
         "the four judge features of spike 017 share a cell's draws, cells of one checkpoint and size share their seeds across "
         "labels, and arms are paired. Bonferroni is then conservative, and the pooled miss is a description, not a test. (2) The "
         "4,000 draws of a sheet cell are 20 plantings x 200 sheets; the per-planting counts were not stored, so the binomial "
         "treats them as 4,000. (3) *At or under delta* is a statement about the estimate; with 200-500 draws its interval still "
         "reaches well above delta (see the largest-miss column). (4) The reference-rate-strata cells of spikes 013 and 014 and of "
         "`stratppi.json` draw 20-40% of a pool of about 500 prompts without replacement and take the pool's rate as the truth, "
         "which makes every bound there look more conservative than it is for a large pool; the rows marked *redrawn with "
         "replacement* are the same cells without that help.", "",
         "**The correction of 2026-10-06.** Until then `b1w` (and the pooled Wilson bound, which is `b1w` at one stratum) returned "
         "its estimate when the sample held no positive. The rare-label cells of spike 013 then read 0.076-0.448 at delta 0.05 and "
         "n_s 100, the chance of drawing no positive, and earlier versions of this file classed all twelve as over. Every caller "
         "was rerun with its original seeds (`reports/b1w_fix_and_audit_2026-10-06.md`); the counts below are from the corrected "
         "files.", "",
         "## (a) Per-bound summary", "", "Rows the paper quotes, in the order of Table 4 and then by section.", ""]
    L += SUMMARY_HEAD + [summary_row(g) for g in cited]
    L += ["", "Bounds in the same files that the paper does not quote.", ""] + SUMMARY_HEAD + [summary_row(g) for g in rest]
    L += ["", "The same bound over all its settings at one delta (cells that repeat another row's draws left out).", "",
          "| bound | delta | rows pooled " + HEAD, "|" + "---|" * 12]
    for (fam, d), gs in fams.items():
        L.append(summary_row(None, S(*[g["id"] for g in gs]), f"| {fam} | {d} | {len(gs)} "))
    diff = [g for g in G.values() if g["sum"]["counts"] and g["sum"]["bonf"] != g["sum"]["bonf_exact"]]
    L += ["", "Bonferroni by exact p-values (p < a / C) in place of the widened band gives the same count in every row except: "
          + "; ".join(f"{g['bound']}, {g['setting'].split(',')[0]} ({g['sum']['bonf_exact']} against {g['sum']['bonf']})" for g in diff) + "."]

    L += ["", "## (b) Table 4 of the draft against the files", "",
          "Every printed miss rate agrees with the recount to its printed digits and every count of cells is equal (asserted). "
          "Bold and the `kind` column are the paper's and are not derived from the counts.", ""] + table4()
    t = G["r62_12"]["cells"]
    lo6, lo61 = largest_of_checkpoints("013_b1w_mid_0.05"), largest_of_checkpoints("013_b1w_mid_0.1")
    L += ["", "- Rows 1, 3, 4, 5 and 15: counts rebuilt as round(m x 5,000) from rates printed to three decimals (`~` in (a)); no "
          "class changes within the rounding. Row 5 is one printed row for two bounds. Row 15, Student-t: over at n 200, 400 and "
          f"800 ({', '.join(f4(x['miss']) for x in t[:3])}), unresolved at 1,200 and 2,400 ({f4(t[3]['miss'])} each).",
          f"- Row 6 by the largest miss over three checkpoints, the cell of earlier drafts: {lo6['cells']} such cells, {lo6['over']} "
          f"over, {f4(lo6['lowest'])}-{f4(lo6['largest'])} at delta 0.05 and {f4(lo61['lowest'])}-{f4(lo61['largest'])} at 0.10.",
          "- Rows 12, 13, 17 and 18 are the cells of rows 6, 7 and 11 redrawn with replacement (caution 4). Row 13 is the control: "
          "Clopper-Pearson is exact on those draws, so its unresolved cells are Monte Carlo noise."]

    cl = claims()
    L += ["", "## (c) Sentences of the draft that state a count, against the files", "",
          "Each quotation is in the draft word for word and each count is asserted in the script.", ""]
    missing = []
    for i, (where, quotes, text) in enumerate(cl, 1):
        L.append(f"{i}. **{where}.**")
        for q in quotes:
            if re.sub(r"\s+", " ", q) not in paper:
                missing.append(q)
            L.append(f"   > {q}")
        L += ["", f"   {text}", ""]
    assert not missing, missing        # every quotation is in the draft, word for word

    L += ["## (d) The two lists the rule produces", "",
          "**Bounds the paper says hold, with a cell over.**", ""]
    own = [g for g in cited if not g["dup"]]             # rows on another row's draws are not listed twice
    hit = [(g, c) for g in own if g["verdict"] == "holds" for c in g["cells"] if c["cls"] == OVER]
    L += [f"- {g['bound']} ({g['where']}, {g['tag']}): {c['label']}, {desc(c)}; over after Bonferroni for {g['sum']['classified']} "
          f"cells: {'yes' if c['miss'] > c['delta'] + g['sum']['z_bonf'] * c['se'] else 'no'}." for g, c in hit] or ["- none"]
    L += ["", "**Bounds the paper says fail, with their cells that are not over.**", ""]
    only = []
    for g in own:
        s = g["sum"]
        if g["verdict"] == "fails" and s["counts"] and s["counts"][OVER] < s["classified"]:
            if s["counts"][OVER] == 0:
                only.append(g)
            L.append(f"- {g['bound']}, {g['setting']}, delta {g['delta']} ({g['where']}): {s['counts'][OVER]} of {s['classified']} "
                     f"over, {s['counts'][UNRES]} unresolved, {s['counts'][UNDER]} at or under; pooled miss {f4(s['pooled'])}.")
    L += ["", "Of these, with no cell over (only unresolved or under): "
          + ("; ".join(f"{g['bound']}, {g['setting']}" for g in only) if only else "none") + "."]

    lost = G["r63_lag_ab"]["sum"]["cells"] + G["r63_judge"]["sum"]["cells"]
    top63 = max(G["r63_lag_ab"]["sum"]["largest"], G["r63_judge"]["sum"]["largest"])
    L += ["", "## (e) Not recounted, and why", "",
          "- **[R 6.2], Table 4 rows 1, 3, 4, 5 and 15.** The cached labels were lost on 2026-09-19; the printed rates are reproduced by "
          "enumeration at the pool's rate (`results/paper/binomial_rows.md`). Classified here from "
          "the printed rates and the 5,000 resamples the training paper states; counts are round(m x 5,000).",
          "- **[R 6.2], the Student-t bound at delta 0.05.** The training paper's sentence (0.084, 0.067, 0.083, 0.074, 0.067) "
          "repeats its Clopper-Pearson row in four of five values (the draft's open check 2). Not classified; Table 4 does not use "
          "it.",
          f"- **[R 6.3], section 7.2, tables (a), (b) and (d).** {lost} printed cells with no class: the training paper gives "
          f"'500-1,000 independent trials' a row and no count per row. Every printed rate (at most {f4(top63)}) is under delta 0.1 "
          "whatever the count; the p-values and intervals need it. Table (c) is classified at the 500 trials of the reproduction "
          "command in that paper's Appendix B. Regenerable (`scripts/synthetic_calibration.py`), not re-run here.",
          "- **Section 7.2, language-model policies.** 0 breaches in 3 seeds and in 10 seeds: the paper declines to read these as a "
          "miss rate, and three or ten draws give no class worth the name.",
          "- **The exact Wilson calculation (`results/paper/wilson_exact.md`).** An enumeration, not a resampling study; it has no "
          "cells to classify. `scripts/wilson_exact.py` asserts its own closed forms.",
          "- **Clustering of the sheet draws.** `harm.json` stores one rate per cell, not per planting, so a planting-level "
          "standard error cannot be formed without re-running `harm017.py`.",
          "- **`plasmode_n500.json` (spike 017).** Not part of the spike's table B6 and not quoted in the paper; left out."]

    L += ["", "## Appendix. The cells behind (d)", "",
          "Every cell that is not at or under delta in a row the paper says holds, and every cell that is not over in a row it "
          "says fails (rows with more than 12 such cells give the count; all cells are in the JSON).", "",
          "| bound | setting | the paper says | cell | delta | draws | misses | miss | class | exact p | 95% CP |", "|" + "---|" * 11]
    for g in own:
        if not g["verdict"] or g["sum"]["counts"] is None:
            continue
        odd = [c for c in g["cells"] if c["cls"] != (UNDER if g["verdict"] == "holds" else OVER)]
        if len(odd) > 12:
            L.append(f"| {g['bound']} | {g['setting']} | {g['verdict']} | {len(odd)} cells, {rng(summarise(odd))} | {g['delta']} | "
                     f"{draws_txt(summarise(odd))} | | | {sum(c['cls'] == UNRES for c in odd)} unresolved, "
                     f"{sum(c['cls'] == UNDER for c in odd)} at or under | | |")
            continue
        for c in odd:
            L.append(f"| {g['bound']} | {g['setting']} | {g['verdict']} | {c['label']} | {c['delta']} | {c['draws']:,} | "
                     f"{'~' if c['printed'] else ''}{c['k']:,} | {f4(c['miss'])} | {c['cls']} | {pv(c['p'])} | "
                     f"[{f4(c['lo'])}, {f4(c['hi'])}] |")
    open(os.path.join(OUT, "validity_recount.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


def main():
    for load in (load_training_paper, load_012, load_013_014, load_004, load_017, load_020, load_p14, load_p9, load_validate,
                 load_agentdojo, load_replacement, load_twoway, load_twophase):
        load()
    for g in GROUPS.values():
        g["sum"] = summarise(g["cells"])
    fams = collections.OrderedDict()
    for g in sorted(GROUPS.values(), key=order):
        if g["family"] and not g["dup"]:
            fams.setdefault((g["family"], g["delta"]), []).append(g)
    os.makedirs(OUT, exist_ok=True)
    checks()
    report(fams)
    cl = claims()
    rows4 = md_table(rd(PAPER), "*Table 4. Miss rate against delta for every bound used")[1:]
    json.dump(dict(rule=dict(z=Z, alpha_one_sided=ALPHA, classes=CLASSES), groups=list(GROUPS.values()),
                   families=[dict(family=k[0], delta=k[1], groups=[g["id"] for g in v], sum=S(*[g["id"] for g in v]))
                             for k, v in fams.items()],
                   table4=[dict(row=i, bound=r[0], kind=r[1], setting=r[2], delta=r[3], paper_miss=r[4], paper_counts=r[5], source=r[6],
                                parts=[dict(groups=ids, prints=how, sum=S(*ids)) for ids, how in parts])
                           for i, (r, parts) in enumerate(zip(rows4, TABLE4), 1)],
                   claims=[dict(where=w, quotes=q, files=t) for w, q, t in cl]),
              open(os.path.join(OUT, "validity_recount.json"), "w"))
    cited = [g for g in GROUPS.values() if order(g)[0] < 2 and g["sum"]["counts"] and not g["dup"]]
    print(f"{len(GROUPS)} bound-by-setting rows, {sum(len(g['cells']) for g in GROUPS.values())} cells")
    for v in ("holds", "fails"):
        gs = [g for g in cited if g["verdict"] == v]
        print(f"the paper says {v}: {len(gs)} rows; cells over {sum(g['sum']['counts'][OVER] for g in gs)}, unresolved "
              f"{sum(g['sum']['counts'][UNRES] for g in gs)}, at or under {sum(g['sum']['counts'][UNDER] for g in gs)}")
    print("-> results/paper/validity_recount.md, results/paper/validity_recount.json")


if __name__ == "__main__":
    main()
