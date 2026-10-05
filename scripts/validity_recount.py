"""Recount of every validity cell behind `reports/paper_certification.md` (v0.5) under one rule.

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
`results/paper/stratppi.json` and `results/labels/p9/design_check.json`; the rows of the training
paper's sections 6.2 and 6.3 are read from its printed tables (their per-draw data are lost). The
draw count of a cell is the one its file or script states; where none is stated the class is blank.

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
B1W = "b1w (Table 2 rows 6-8)"
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
        return dict(s, total=None, pooled=None, counts=None, bonf=None, bonf_exact=None, worst=None, pooled_cls=None)
    C = len(known)
    zc = float(norm.isf(ALPHA / C))
    total, k = sum(c["draws"] for c in known), sum(c["k"] for c in known)
    worst = max(known, key=lambda c: c["miss"])
    # the pooled class treats the draws as one sample: a description, since cells share draws
    return dict(s, total=total, pooled=k / total, pooled_cls=classify(k / total, known[0]["delta"], total),
                counts={x: sum(c["cls"] == x for c in known) for x in CLASSES},
                bonf=sum(c["miss"] > c["delta"] + zc * c["se"] for c in known),
                bonf_exact=sum(c["p"] < ALPHA / C for c in known), z_bonf=zc,
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
            [TRAINING + " section 6.2"], where=f"Table 2 row {row}", verdict=verdict, regen="not regenerable",
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
        "n 200-5,000", 0.1, cells, "[R 6.3]", [TRAINING + " section 6.3"], where="section 4", verdict="holds",
        regen="yes, not re-run here", note=note + "; (a) t and (b) n 1,000 print the same row")
    add("r63_lag_c", "Seldonian pipeline (safety test after selection)", "synthetic bandit, table (c): pressures 0-4, n 1,000",
        0.1, [cell(f"(c) pressure {r[0]}", 0.1, trials_c, m=float(r[4])) for r in c[1:]], "[R 6.3]",
        [TRAINING + " section 6.3 and Appendix B"], where="section 4", verdict="holds", regen="yes, not re-run here",
        note=f"printed rates; {trials_c} trials from the reproduction command in the source's Appendix B")
    add("r63_grpo_c", "unconstrained training (no test)", "synthetic bandit, table (c): pressures 0-4", 0.1,
        [cell(f"(c) pressure {r[0]}", 0.1, trials_c, m=float(r[1])) for r in c[1:]], "[R 6.3]",
        [TRAINING + " section 6.3 and Appendix B"], where="section 4", verdict="fails", regen="yes, not re-run here",
        note="printed rates; the paper quotes the pressure-1 cell (0.830)")
    add("r63_judge", "Seldonian pipeline, judge-level violation given a solution", "synthetic bandit, table (d): four judges",
        0.1, [cell(f"(d) judge {r[0]}", 0.1, None, m=float(r[2])) for r in d[1:]], "[R 6.3d]", [TRAINING + " section 6.3"],
        where="section 4", verdict="holds", regen="yes, not re-run here",
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
        "miss = passed and truly violating", delta, cp, "[012]", f, where="section 4", verdict="holds")
    add("012_wald", "tight Wald test after an adversarial split", "same runs; miss = true gap above the safety-set bound",
        delta, wald, "[012]", f, where="not quoted in the paper")


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
            "3 checkpoints", d, pick(rt, "S2", MID, d), "[013 H8]", f13, where="Table 2 row 6", verdict="holds",
            family=B1W, note="the paper's range is of the largest miss over 3 checkpoints (8 values a delta)")
        add(f"013_wilson_mid_{d}", "pooled Wilson bound, random split", "same labels and sizes", d, pick(rt, "R", MID, d, H=4),
            "[013 H8]", f13, where="section 4 (the comparator of the in-loop sentence); Table 3 baseline")
    for n_s in (100, 200):
        w = "Table 2 row 13" if n_s == 100 else "sections 3 and 5 (rare labels)"
        add(f"013_b1w_rare{n_s}", "`b1w` at rare rates", f"labels at 1-2%, n_s {n_s}, 3 checkpoints", 0.05,
            pick(rt, "S2", RARE, 0.05, n_s), "[013 H8]", f13, where=w, verdict="fails")
        add(f"013_wilson_rare{n_s}", "pooled Wilson bound at rare rates", f"labels at 1-2%, n_s {n_s}, 3 checkpoints", 0.05,
            pick(rt, "R", RARE, 0.05, n_s, H=4), "[013 H8]", f13, where=w, verdict="fails")
    p14 = [r for r in jload(".planning/spikes/014-pushed-label-stratification/plasmode.json") if r["role"] == "cand"]
    for d in (0.05, 0.1):
        add(f"014_b1w_{d}", "`b1w`", "label pushed by the Lagrangian, n_s 200, steps 100 and 200", d,
            pick(p14, "S2", ("C1:refusal:cand",), d, 200), "[014]", [".planning/spikes/014-pushed-label-stratification/plasmode.json"],
            where="Table 2 row 7", verdict="holds", family=B1W)
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
        "`SeldonianLLMPolicy`", 0.1, cells[("strat_ref", "b1w_strat_pop")], "[013 4]", fin, where="section 4", verdict="holds")
    add("013_inloop_random", "pooled `b1w` in the training loop, random split", "same", 0.1, cells[("random", "b1w_pooled")],
        "[013 4]", fin, where="section 4 (comparator)", verdict="holds")
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
    for key, name, verdict, w in (("miss_any_delta_T", "trajectory certificate at delta / T", "holds", "section 4"),
                                  ("miss_any_delta", "per-check delta read as a trajectory claim", "", "not quoted in the paper")):
        add(f"004_{key}", name, "synthetic bandit, 5 arms, checks every 25 and 10 steps; miss = some check's bound under "
            "its true rate", 0.1, [cell(f"every {e}, {arm}", 0.1, len(R), k=sum(r[key] for r in R)) for (e, arm), R in by.items()],
            "[SR 2.1]", f, where=w, verdict=verdict,
            note="the tag resolves to the state report, which digests spike 004; counted from the spike's per-run file")


# ------------------------------------------------------------------ spike 017
ROUTES = collections.OrderedDict([
    ("classical", ("Clopper-Pearson, labels alone", "Table 2 row 2", "holds", "Clopper-Pearson")),
    ("boot", ("PPI++ with a bootstrap-t limit", "Table 2 row 9", "holds", "PPI++, bootstrap-t limit")),
    ("ppi++", ("PPI++ with a normal limit", "Table 2 row 14", "fails", "PPI++, normal limit")),
    ("naive", ("the judge's rate alone", "Table 2 row 19", "fails", "")),
    ("block", ("block PPI, betting (finite-sample)", "section 7.4", "holds", "")),
    ("ppi", ("plain PPI, normal limit", "not quoted in the paper", "", "")),
    ("ppi++w", ("PPI++, score (Wilson-type) limit", "not quoted in the paper", "", "")),
    ("youden", ("Youden correction from the same labels", "not quoted in the paper", "", "")),
    ("exact3", ("PPI, three exact limits", "not quoted in the paper", "", "")),
    ("strat", ("post-stratified on the 0/1 judge, exact", "not quoted in the paper", "", ""))])


def load_017():
    d17 = ".planning/spikes/017-calibration-carrying-certificate/"
    base, shift = jload(d17 + "plasmode.json"), jload(d17 + "plasmode_shift.json")
    assert base["delta"] == shift["delta"] == 0.05
    by = collections.defaultdict(list)
    for src, rows in (("", base["rows"]), ("shifted ", shift["rows"])):
        for r in rows:
            lab = f"{src}{r['task']} {r['variant']} {r['wording']}, rate {r['rate']:.3f}, n {r['n']} of {r['N']}, {r['feat']}"
            by[r["method"]].append(cell(lab, 0.05, r["reps"], k=r["miss"] * r["reps"], rate=r["rate"], n=r["n"], feat=r["feat"]))
    for m, (name, where, verdict, fam) in ROUTES.items():
        add(f"017_{m}", name, "spike 017 plasmodes, every cell and feature (the cells of its table B6)", 0.05, by[m], "[017 B6]",
            [d17 + "plasmode.json", d17 + "plasmode_shift.json"], where=where, verdict=verdict, family=fam)
    # the stratified sheet, re-drawn by its real rule: 20 plantings x 200 draws (harm017.py)
    assert "re-drawn 4,000 times (20 plantings x 200 draws)" in rd(d17 + "harm.md")
    by = collections.defaultdict(list)
    for r in jload(d17 + "harm.json")["planted"]:
        by[r["route"]].append(cell(f"wording {r['wording']}, planted rate {r['rate']}, {r['feat']}", 0.05, 4000,
                                   k=r["miss"] * 4000, wording=r["wording"]))
    note = "4,000 draws are 20 plantings x 200 re-drawn sheets, each planting with its own truth; the binomial treats them as 4,000"
    for route, gid, name, where, verdict, fam in (
            ("weighted labels, b1w", "017_sheet_b1w", "`b1w` on a design-weighted sheet", "Table 2 row 8", "holds", B1W),
            ("PPI as i.i.d.", "017_sheet_ppi_iid", "stratified sheet read as an i.i.d. sample (PPI)", "Table 2 row 18", "fails", ""),
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
    said = {"source: xstest -> orbench": ("x2o", "7.1", "fails", "over for 4 of 6"),
            "source: orbench -> xstest": ("o2x", "7.1", "fails", "over for 1 of 6"),
            "training: step 0 -> step 200": ("side", "7.2", "holds", "carried (holds)"),
            "pool: over-refusal -> harmful": ("r2h", "7.1", "holds", "over for none"),
            "pool: harmful -> over-refusal": ("h2r", "7.1", "fails", "over for 5 of 6"),
            "training, constrained (014): step 0 -> step 100": ("c100", "7.2", "fails", "1 of 6 failed"),
            "training, constrained (014): step 0 -> step 200": ("c200", "7.2", "fails", "over for 3 of 6, marginal for a fourth")}
    for shift_name, cs in by.items():
        i, sec, verdict, words = said[shift_name]
        add(f"017_carry_{i}", "carried Youden-corrected bound", shift_name + ", six judge wordings", 0.05, cs,
            "[017 5]" if sec == "7.1" else "[017 E8]", [d17 + "transfer.json"], where=f"section {sec}", verdict=verdict, says=words,
            note="one run of 4,000 draws a wording")


# ------------------------------------------------------------------ spike 020, P14, P9
def load_020():
    d20 = ".planning/spikes/020-agentdojo-injection-certificate/"
    reps = int(re.search(r'"--reps", type=int, default=(\d+)', rd(d20 + "plasmode020.py")).group(1))
    assert f"({reps} reps" in rd(d20 + "plasmode.md")
    d = jload(d20 + "plasmode.json")
    for key, name, where, verdict in (
            ("t_user", "cluster bootstrap-t, by user task", "Table 2 row 10", "holds"),
            ("naive", "Clopper-Pearson over pairs", "Table 2 row 16", "fails"),
            ("twoway", "two-way bootstrap", "Table 2 row 17", "fails"),
            ("t_inj", "cluster bootstrap-t, by injection task", "section 8.2 (half of the larger-of-two rule)", ""),
            ("wilson", "Wilson bound over pairs", "not quoted in the paper", "")):
        add(f"020_{key}", name, "AgentDojo, 6 pipelines, user tasks resampled", 0.05,
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
                where = {("a_sboot", "mid", 0.05): "Table 2 row 11; Table 3", ("a_sppi", "mid", 0.05): "Table 2 row 15; Table 3",
                         ("a_b1w", "mid", 0.05): "Table 3"}.get((gid, kind, d), "results/paper/stratppi.md only")
                v = verdict
                if where.startswith("results") and gid in ("a_sboot", "a_sppi", "a_b1w", "a_b1"):
                    where = "section 5" if kind == "mid" or gid in ("a_sppi", "a_b1w") else "section 3 (reading 1)"
                    v = verdict if kind == "mid" or gid in ("a_sboot", "a_b1") else "fails"
                elif where.startswith("results"):
                    v = ""
                add(f"p14_{gid}_{kind}_{d}", name, ("5 mid-rate labels" if kind == "mid" else "2 rare labels (1-2%)") +
                    ", reference-rate strata, n_s 100-200, by checkpoint" + (" (the draws of spikes 013 and 014)" if dup else ""),
                    d, cells, "[P14]", f, where=where, verdict=v,
                    family=fam if kind == "mid" else "", dup=dup,
                    note="the paper counts 10 cells a delta, each the largest miss over 2-3 checkpoints" if kind == "mid" else "")
    barms = (("labels alone, Clopper-Pearson", "b_cp", "Clopper-Pearson, labels alone", "holds", "Clopper-Pearson", "section 6 (the baseline)"),
             ("PPI++ normal", "b_ppipp", "PPI++ with a normal limit", "fails", "PPI++, normal limit", "section 6"),
             ("PPI++ bootstrap-t (this paper)", "b_boot", "PPI++ with a bootstrap-t limit", "holds", "PPI++, bootstrap-t limit", "section 6"),
             ("StratPPI, K=", "b_sppi", "StratPPI as published (normal limit)", "fails", "StratPPI, normal limit", "Table 2 row 15; section 6"),
             ("StratPPI, bootstrap-t, K=", "b_sboot", "StratPPI estimator with a bootstrap-t limit", "holds",
              "StratPPI estimator, bootstrap-t limit", "Table 2 row 11; section 6"),
             ("judge strata + b1w, K=", "b_b1w", "`b1w` on judge-logit strata", "holds", "", "section 6"))
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
    for scheme, gid, where, verdict in (("pool", "p9_pool", "section 8.4", "holds"),
                                        ("new prompts", "p9_new", "not quoted in the paper", "")):
        add(gid, "the four limits of the human-terms certificate", f"design check on synthetic labels, prompts as '{scheme}'",
            d["delta"], [cell(names[k], d["delta"], d["reps"], k=v["miss"] * d["reps"]) for k, v in d["schemes"][scheme].items()],
            "[P9 design check]", [rel], where=where, verdict=verdict,
            note="(b1) is not built for new prompts" if scheme == "new prompts" else "")




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
    """Table 2 rows first, then the sections in order, then what the paper does not quote."""
    w = g["where"]
    m = re.match(r"Table 2 row (\d+)", w)
    if m:
        return (0, int(m.group(1)), g["delta"])
    m = re.match(r"sections? (\d+)(?:\.(\d+))?", w)
    if m:
        return (1, int(m.group(1)) + int(m.group(2) or 0) / 10, 0)
    return (1, 5, 0) if w.startswith("Table 3") else (2, 0, 0)


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

# Table 2 of the paper, row by row: (bound, kind, setting, delta, miss as printed, source, joiner, parts). A part is the
# list of groups behind one printed value; parts are joined as the paper joins them (" / " two deltas, "; " two settings).
TABLE2 = [
    ("Clopper-Pearson", "exact (binary labels)", "real harm labels, trained-policy pool, rate 0.034, n 200-2,400", "0.10",
     "0.067-0.084", "[R 6.2]", " / ", [["r62_1"]]),
    ("Clopper-Pearson", "exact", "spike 017 plasmodes, every cell", "0.05", "at most 0.051", "[017 B6]", " / ", [["017_classical"]]),
    ("betting mixture", "exact (bounded)", "same pool as row 1", "0.10", "0.007-0.016", "[R 6.2]", " / ", [["r62_3"]]),
    ("Bentkus", "exact (bounded)", "same pool", "0.10", "0.023-0.035", "[R 6.2]", " / ", [["r62_4"]]),
    ("Hoeffding, Anderson", "exact (bounded)", "same pool", "0.10", "0.000", "[R 6.2]", " / ", [["r62_5"]]),
    ("stratified Wilson-type `b1w`", "approximate", "4 mid-rate labels (9-94%), real Granite-3.3-2B responses, n_s 100-200, "
     "3 checkpoints [c]", "0.05 / 0.10", "0.023-0.054 / 0.069-0.097", "[013 H8]", " / ", [["013_b1w_mid_0.05"], ["013_b1w_mid_0.1"]]),
    ("`b1w`", "approximate", "label pushed by the Lagrangian, n_s 200", "0.05 / 0.10", "0.011-0.023 / 0.040-0.064", "[014]", " / ",
     [["014_b1w_0.05"], ["014_b1w_0.1"]]),
    ("`b1w` on a design-weighted sheet", "approximate", "sheets re-drawn by their real sampling rule", "0.05", "0.001-0.059",
     "[017 4]", " / ", [["017_sheet_b1w"]]),
    ("PPI++ with a bootstrap-t limit", "approximate (second order)", "spike 017 plasmodes, every cell and feature", "0.05",
     "at most 0.053", "[017 B6]", " / ", [["017_boot"]]),
    ("cluster bootstrap-t, by user task", "approximate", "AgentDojo, 6 pipelines, user tasks resampled", "0.05", "0.020-0.060",
     "[020 P]", " / ", [["020_t_user"]]),
    ("StratPPI estimator with a bootstrap-t limit", "approximate", "reference-rate strata, 5 mid-rate labels, n_s 100-200, "
     "2-3 checkpoints [c]; judge-logit strata, 14 cells x 2 strata counts, n 100-1,000", "0.05", "at most 0.045; at most 0.056",
     "[P14]", "; ", [["p14_a_sboot_mid_0.05"], ["p14_b_sboot"]]),
    ("**Student-t**", "fails at low rates", "same pool as row 1, n 200-2,400 (the paper's row: n 200-800) [a]", "0.10",
     "**0.115-0.184**", "[R 6.2]", " / ", [["r62_12"]]),
    ("**`b1w` and the pooled Wilson bound at rare rates**", "fail", "harm labels at 1-2%, n_s 100, any design, 3 checkpoints [c]",
     "0.05", "**0.24-0.44**", "[013 H8]", " / ", [["013_b1w_rare100", "013_wilson_rare100"]]),
    ("**PPI++ with a normal limit**", "fails", "spike 017 plasmodes", "0.05", "**up to 0.241**", "[017 B6]", " / ", [["017_ppi++"]]),
    ("**StratPPI as published (normal limit)**", "fails at these sizes", "reference-rate strata, 5 mid-rate labels, 2-3 "
     "checkpoints [c]; judge-logit strata", "0.05", "**up to 0.086; up to 0.239**", "[P14]", "; ",
     [["p14_a_sppi_mid_0.05"], ["p14_b_sppi"]]),
    ("**Clopper-Pearson over pairs**", "fails on a crossed design", "AgentDojo, user tasks resampled", "0.05", "**0.048-0.275**",
     "[020 P]", " / ", [["020_naive"]]),
    ("**two-way bootstrap**", "fails where positives sit in few clusters", "AgentDojo", "0.05", "**up to 0.170**", "[020 P]", " / ",
     [["020_twoway"]]),
    ("**stratified sheet read as an i.i.d. sample**", "fails", "PPI on the sheet, three judge wordings (the paper's row: one) [b]",
     "0.05", "**up to 0.98**", "[017 4]", " / ", [["017_sheet_ppi_iid"]]),
    ("**the judge's rate alone**", "fails", "spike 017 plasmodes", "0.05", "**1.000**", "[017 B6]", " / ", [["017_naive"]]),
]


def table2():
    L = ["| bound | kind | setting | delta | miss (paper) | miss (recount, per cell) | cells | draws per cell | pooled miss | over | "
         "unresolved, above delta | at or under delta | source |", "|" + "---|" * 13]
    for bound, kind, setting, delta, printed, tag, sep, parts in TABLE2:
        P = [S(*p) for p in parts]
        col = lambda fn: sep.join(fn(s) for s in P)                                     # noqa: E731
        L.append(f"| {bound} | {kind} | {setting} | {delta} | {printed} | {col(rng)} | {col(lambda s: str(s['cells']))} | "
                 f"{col(draws_txt)} | {'~' if P[0]['printed'] else ''}{col(lambda s: f4(s['pooled']))} | "
                 f"{col(lambda s: str(s['counts'][OVER]))} | {col(lambda s: str(s['counts'][UNRES]))} | "
                 f"{col(lambda s: str(s['counts'][UNDER]))} | {tag} |")
    return L


def largest_of_checkpoints(*ids):
    """The paper's own cell for the rows built on checkpoints: the largest miss of a label at one n_s."""
    by = collections.defaultdict(list)
    for c in cells_of(ids):
        by[c["key"]].append(c)
    big = [max(v, key=lambda c: c["miss"]) for v in by.values()]
    return dict(cells=len(big), over=sum(c["cls"] == OVER for c in big), above=sum(c["cls"] != UNDER for c in big),
                lowest=min(c["miss"] for c in big), largest=max(c["miss"] for c in big))


def sentences():
    """(changes, stands): each entry is (where, [quotes], what the rule gives)."""
    G = GROUPS
    ch, st = [], []
    # --- Table 2 as printed
    x12 = [c for c in G["r62_12"]["cells"] if c["label"] in ("n 1200", "n 2400")]
    n13 = n3("013_b1w_rare200", "013_wilson_rare200")
    x18 = [c for c in G["017_sheet_ppi_iid"]["cells"] if c["wording"] != 2]
    w2 = [c for c in G["017_sheet_ppi_iid"]["cells"] if c["wording"] == 2]
    n16, n17, n18, n19 = n3("020_naive"), n3("020_twoway"), n3("017_sheet_ppi_iid"), n3("017_naive")
    ch.append(("section 3, above Table 2", ["One rule is applied to every bound, favoured or not."],
               "The rule is stated, and the table's `kind` column and bold type are not its output. "
               f"Row 8 is plain type and has a cell over its level. Rows 12, 13 and 18 print only the sizes or the wording where "
               f"the bound fails: the source has {len(x12)} more Student-t cells ({sum(c['cls'] != OVER for c in x12)} not over), "
               f"{n13[0]} more rare-label cells at n_s 200 ({n13[2] + n13[3]} not over) and {len(x18)} more sheet cells "
               f"({sum(c['cls'] != OVER for c in x18)} not over). Rows 16 to 19 are bold with {n16[3]} of {n16[0]}, {n17[3]} of "
               f"{n17[0]}, {n18[3]} of {n18[0]} and {n19[3]} of {n19[0]} cells at or under delta. Rows 6, 13 and 15 take "
               "the largest miss of 2-3 checkpoints before the threshold, a stricter test than the rule and one the other rows do "
               "not get. The replacement in (b) counts every cell the source has and leaves the verdict to the three count columns."))
    ch.append(("section 3, caption of Table 2", ["Monte Carlo standard errors are 0.003-0.004 for 5,000 resamples and 0.011 for 400."],
               "Right for those two counts, and five other counts are in the table or the text: 4,000 draws (se 0.0034 at delta 0.05; "
               "rows 2, 8, 9, 11, 14, 15, 18, 19 and section 7), 1,000 (0.0069; block PPI, and spike 017's cells with 20,000 judged responses, where "
               "row 9's largest miss of 0.053 sits), 500 (0.0097; block PPI at N 20,000, and 0.0134 at delta 0.1 in section 4), "
               "2,000 (0.0049; sections 4 and 8.4) and 200 (0.0212 at delta 0.1; the trajectory certificate)."))
    assert {x["draws"] for x in G["017_boot"]["cells"] if "of 20000" in x["label"]} == {1000}
    assert G["017_boot"]["sum"]["worst"]["draws"] == 1000
    # --- b1w
    c = find("017_sheet_b1w", "wording 2", "rate 0.2", "f01")
    c2 = find("017_sheet_b1w", "wording 2", "rate 0.2", "logit")
    cp = find("017_sheet_b1w_pooled", "wording 2", "rate 0.2")
    s = G["017_sheet_b1w"]["sum"]
    ch.append(("section 3, Table 2 row 8; section 6, Table 4; section 9",
               ["| `b1w` on a design-weighted sheet | approximate | sheets re-drawn by their real sampling rule | 0.05 | 0.001-0.059 | [017 4] |",
                "| gold labels from a stratified sheet | design-weighted labels, `b1w` | approximate |",
                "`b1w`, the bootstrap-t limits and the cluster bootstrap are checked by resampling, not proved at finite n. Table 2 is "
                "the evidence, with its Monte Carlo error."],
               f"One of 18 cells is over its level: judge wording 2, planted rate 0.2, {desc(c)}, against a band that ends at "
               f"{band(0.05, 4000)}. One more is unresolved ({f4(find('017_sheet_b1w', 'wording 0', 'rate 0.2', 'f01')['miss'])}), "
               f"{s['counts'][UNDER]} are at or under delta. After a Bonferroni correction for 18 cells none is over (the band "
               f"then ends at {f4(0.05 + s['z_bonf'] * c['se'])}). The route does not read the judge's feature, so the source's "
               f"second row for the same wording and rate is a second run of the same setting: it gave {f4(c2['miss'])}, and the "
               f"two together {desc(cp)}, {cp['cls']}. This is the only bound the paper lists as usable that has a cell over."))
    a = G["013_b1w_mid_0.05"]["cells"]
    hot = [x for x in a if x["cls"] != UNDER]
    assert all(x["key"] == ("C2:refusal", 100) and x["cls"] == UNRES for x in hot)
    kk, rr = sum(x["k"] for x in hot), sum(x["draws"] for x in hot)
    lo = largest_of_checkpoints("013_b1w_mid_0.05")
    ch.append(("section 3, Table 2 row 6; section 5",
               ["| stratified Wilson-type `b1w` | approximate | 4 mid-rate labels (9-94%), real Granite-3.3-2B responses, n_s 100-200 | "
                "0.05 / 0.10 | 0.023-0.054 / 0.069-0.097 | [013 H8] |", "with coverage as in Table 2"],
               f"No cell over. At delta 0.05, {len(hot)} of {len(a)} checkpoint cells are unresolved above delta, all three the "
               f"94% label at n_s 100 ({', '.join(f4(x['miss']) for x in hot)}); together {kk:,} of {rr:,} = {f4(kk / rr)}, still "
               f"inside the band for that many draws ({band(0.05, rr)}; p = {pv(float(binom.sf(kk - 1, rr, 0.05)))}, picked after "
               f"the fact as the worst label). At delta 0.1 all {n3('013_b1w_mid_0.1')[0]} are at or under. The printed ranges are of "
               f"the largest miss over three checkpoints ({f4(lo['lowest'])}-{f4(lo['largest'])} at 0.05); per cell the range is "
               f"{rng(G['013_b1w_mid_0.05']['sum'])}."))
    # --- exact bounds
    n_cp = n3("r62_1", "017_classical", "p14_b_cp")
    un = [x for x in cells_of(["017_classical", "p14_b_cp"]) if x["cls"] == UNRES]
    assert all(x["draws"] == 4000 for x in un) and n3("r62_1")[2] == 0
    n_ex = n3("r62_3", "r62_4", "r62_5", "012_cp", "017_block")
    ch.append(("abstract; introduction; section 3, Table 2 rows 1-5",
               ["Exact bounds hold their level.", "Exact bounds hold everywhere we test."],
               f"No exact-bound cell is over. Clopper-Pearson on i.i.d. labels: {n_cp[3]} of {n_cp[0]} cells at or under delta, "
               f"{n_cp[2]} unresolved above it ({', '.join(f4(x['miss']) for x in un)}, each at 4,000 draws). The other exact rows "
               f"and checks (betting, Bentkus, Hoeffding and Anderson, the split study, block PPI): {n_ex[3]} of {n_ex[0]} at or "
               "under. For an exact bound a miss above delta can only be Monte Carlo noise, so 'hold' is true by the proof; the "
               "resampling shows no cell over, which is what the sentence can cite. It also calibrates the rule: a bound known "
               f"to be valid lands in the unresolved class in {n_cp[2]} of {n_cp[0]} cells."))
    # --- bootstrap-t
    nb = n3("017_boot", "p14_a_boot_mid_0.05", "p14_a_boot_rare_0.05", "p14_b_boot")
    ns = n3("p14_a_sboot_mid_0.05", "p14_a_sboot_rare_0.05", "p14_b_sboot")
    ch.append(("abstract; introduction; section 6 (Position); section 10",
               ["The normal-quantile intervals of PPI++ and StratPPI, in our implementation, miss in up to 24% of draws at a nominal "
                "5%, and a bootstrap-t limit restores the level.",
                "a bootstrap-t limit on the same estimators holds (sections 3 to 6)",
                "a studentised bootstrap that does",
                "their estimators with a bootstrap-t limit do"],
               f"'Holds' and 'restores the level' are stronger than the check. At delta 0.05 PPI++ with a bootstrap-t limit is over "
               f"in {nb[1]} of {nb[0]} cells, unresolved above delta in {nb[2]} and at or under in {nb[3]}; StratPPI's estimator "
               f"with the same limit is over in {ns[1]} of {ns[0]}, unresolved in {ns[2]}, at or under in {ns[3]}. By section 3's "
               "own definition an unresolved cell is weaker than holding. The supported wording is 'is over its level in no cell'."))
    c = find("p14_b_sboot", "rubric", "n 100 of 2000", "K=5")
    n = n3("p14_b_sboot")
    ch.append(("section 6", ["The StratPPI estimator with a bootstrap-t limit holds in all 28 (largest miss 0.056)"],
               f"Over in {n[1]} of {n[0]}; {n[2]} are unresolved above delta "
               f"({', '.join(f4(x['miss']) for x in G['p14_b_sboot']['cells'] if x['cls'] == UNRES)}) and {n[3]} at or under. The "
               f"largest is {desc(c)}; the band ends at {band(0.05, 4000)}."))
    n = n3("p14_b_b1w")
    c = max(G["p14_b_b1w"]["cells"], key=lambda x: x["miss"])
    ch.append(("section 6", ["Stratifying on the judge and ignoring it within strata (`b1w`) also holds (largest miss 0.053)"],
               f"Over in {n[1]} of {n[0]}, unresolved in {n[2]} ({desc(c)}), at or under in {n[3]}."))
    # --- normal limits
    n17, na, nbb, nA1 = n3("017_ppi++"), n3("p14_a_sppi_mid_0.05"), n3("p14_b_sppi"), n3("p14_a_sppi_mid_0.1")
    under17 = [x for x in G["017_ppi++"]["cells"] if x["cls"] == UNDER]
    brev = sum("brevity" in x["label"] for x in under17)
    low = [x for x in G["017_ppi++"]["cells"] if x["rate"] <= 0.2]
    mid = [x for x in G["017_ppi++"]["cells"] if x["rate"] > 0.2]
    ch.append(("section 3, Table 2 rows 14 and 15; section 6 (Position); section 10",
               ["evidence that the published normal-quantile intervals, stratified or not, do not hold their level at the sample "
                "sizes and rates of a safety test",
                "(the normal-quantile intervals of PPI++ and StratPPI do not; their estimators with a bootstrap-t limit do)"],
               f"True of most cells, not all, and the sentence reads as all. PPI++ with a normal limit on spike 017's plasmodes: over "
               f"in {n17[1]} of {n17[0]}, unresolved in {n17[2]}, at or under delta in {n17[3]} ({brev} of those on the brevity "
               f"label, where the judge carries nothing; {G['017_ppi++']['sum']['bonf']} over after Bonferroni). StratPPI as "
               f"published on judge-logit strata: over in {nbb[1]} of {nbb[0]}, unresolved in {nbb[2]}. On reference-rate strata at "
               f"mid rates: over in {na[1]} of {na[0]} checkpoint cells, unresolved in {na[2]}, at or under in {na[3]}, pooled miss "
               f"{f4(G['p14_a_sppi_mid_0.05']['sum']['pooled'])}; at delta 0.1, {nA1[1]} of {nA1[0]} over. By rate, PPI++ with a "
               f"normal limit is over in {sum(x['cls'] == OVER for x in low)} of {len(low)} cells at rates of 20% and below and in "
               f"{sum(x['cls'] == OVER for x in mid)} of {len(mid)} at 40%. 'Over its level in most cells at rates of 20% and "
               "below, in fewer at mid rates' is what the counts support."))
    lp, lp1 = largest_of_checkpoints("p14_a_sppi_mid_0.05"), largest_of_checkpoints("p14_a_sppi_mid_0.1")
    c = max((x for x in G["p14_a_sppi_mid_0.05"]["cells"] if x["key"][1:] == ("C3:refusal", 100)), key=lambda x: x["miss"])
    rb = largest_of_checkpoints("p14_a_b1w_rare_0.05")
    rs = largest_of_checkpoints("p14_a_sppi_rare_0.05")
    ch.append(("section 5, after Table 3",
               ["It exceeds delta in 4 of 10 cells at delta 0.05 (one of them marginally, at 0.056; 3 of 10 at delta 0.1), most at "
                "the 9% label, and in every rare-label cell; `b1w` fails in three of those four."],
               f"The counts are of cells over the band, not over delta: taking the paper's cell (the largest miss of a label's "
               f"checkpoints at one n_s), {lp['over']} of {lp['cells']} are over at 0.05 and {lp['above']} are above delta; "
               f"{lp1['over']} of {lp1['cells']} and {lp1['above']} at 0.1. 'Exceeds delta' should read 'is over its level'. The "
               f"marginal cell is {desc(c)}, over by {c['miss'] - 0.05 - Z * c['se']:.4f}. By checkpoint, which is the rule's cell, "
               f"{na[1]} of {na[0]} are over ({G['p14_a_sppi_mid_0.05']['sum']['bonf']} after Bonferroni), {na[2]} unresolved, "
               f"{na[3]} at or under. Rare labels: StratPPI over in {rs['over']} of {rs['cells']}, `b1w` in {rb['over']} of "
               f"{rb['cells']}, as printed."))
    # --- rare labels
    r100 = S("013_b1w_rare100", "013_wilson_rare100")
    lb, lw = largest_of_checkpoints("013_b1w_rare100"), largest_of_checkpoints("013_wilson_rare100")
    g2 = [x for x in G["013_b1w_rare200"]["cells"] if x["key"][0] == "C2:gated"]
    u2 = [x for x in G["013_b1w_rare200"]["cells"] if x["key"][0] == "C3:unsafe"]
    assert all(x["cls"] == OVER for x in u2) and all(x["cls"] == UNDER for x in g2) and r100["counts"][OVER] == r100["cells"]
    ch.append(("section 3, Table 2 row 13 and reading 1; section 5",
               ["At 1-2% and n_s 100 the Wilson-type bounds, pooled or stratified, missed in 24-44% of draws",
                "Rare labels (the approximate bound is invalid there and exact stratified bounds did not beat pooling)"],
               f"All {r100['cells']} cells at n_s 100 are over, and stay over after Bonferroni. By checkpoint they run "
               f"{rng(r100)}; 24-44% is the stratified bound's largest checkpoint for each label ({f4(lb['lowest'])} and "
               f"{f4(lb['largest'])}; the pooled bound's are {f4(lw['lowest'])} and {f4(lw['largest'])}). At n_s 200, which the row "
               f"leaves out, `b1w` is over for the 1% label ({', '.join(f4(x['miss']) for x in u2)}) and at or under delta for the "
               f"2% label ({', '.join(f4(x['miss']) for x in g2)}): 'invalid there' holds at n_s 100 and for one of two labels at 200."))
    # --- Student-t
    t = G["r62_12"]["cells"]
    assert [x["cls"] for x in t] == [OVER] * 3 + [UNRES] * 2
    ch.append(("section 3, Table 2 row 12 and reading 2",
               ["| **Student-t** | fails at low rates | same pool as row 1, n 200-800 | 0.10 | **0.115-0.184** | [R 6.2] |",
                "It is anti-conservative exactly where trained policies sit (harm rates of 3-6%)"],
               f"Over at n 200, 400 and 800 ({', '.join(f4(x['miss']) for x in t[:3])}). At n 1,200 and 2,400, in the same source "
               f"table and left out of the row, it is {f4(t[3]['miss'])} and {f4(t[4]['miss'])}: unresolved above delta (p = "
               f"{pv(t[3]['p'])} each). 'Over its level at n up to 800; unresolved at 1,200 and 2,400' is the rule's reading. "
               "Printed rates, not regenerable."))
    # --- reading 3
    c = find("013_inloop_strat", "icc05")
    cu = max(G["020_t_user"]["cells"], key=lambda x: x["miss"])
    cb = find("017_sheet_b1w", "wording 2", "rate 0.2", "f01")
    assert (cu["cls"], cb["cls"]) == (UNRES, OVER)
    ch.append(("section 3, reading 3",
               ["In Table 2 the approximate bounds we use miss at most one percentage point over delta (the largest is 0.060 at "
                "delta 0.05)."],
               f"True of the printed values. Under the rule the two largest fall in different classes: {f4(cu['miss'])} at 400 "
               f"draws is unresolved (band to {band(0.05, 400)}), {f4(cb['miss'])} at 4,000 draws is over (band to "
               f"{band(0.05, 4000)}). A distance from delta is not a class without the draw count."))
    r = G["013_inloop_random"]["sum"]
    assert n3("013_inloop_strat")[1:] == (0, 1, 3) and c["cls"] == UNRES and r["counts"][UNDER] == 4
    ch.append(("section 3, reading 3; section 4",
               ["Inside the full training loop the stratified test reached 0.116 at delta 0.1, with a Monte Carlo standard error of "
                "0.013 (section 4).",
                "the stratified `b1w` test missed 0.084-0.116 at delta 0.1 (Monte Carlo standard error 0.013), the same as a random "
                "split with a pooled bound [013 4]"],
               f"Three of four cells at or under delta, one unresolved above it: {desc(c)}. At 500 runs a cell the check resolves "
               f"only a miss above {band(0.1, 500)}. The random split with a pooled bound is {rng(r)}, all four at or under: "
               "'the same' is 'not distinguishable at 500 runs', and the stratified test has the one cell above delta."))
    w = G["004_miss_any_delta_T"]["sum"]["worst"]
    assert n3("004_miss_any_delta_T")[3] == 10 and w["miss"] == 0.1
    ch.append(("section 4", ["A certificate at level delta/T over every one of T checks held, with misses of 0.025-0.100 against a "
                             "delta of 0.1 [SR 2.1]."],
               f"All 10 cells at or under delta, the largest exactly at it ({int(round(w['miss'] * w['draws']))} of {w['draws']}; "
               f"95% interval {f4(w['lo'])}-{f4(w['hi'])}). At 200 runs a cell the check resolves only a miss above "
               f"{band(0.1, 200)}, so 'held' is 'at or under delta in every cell, at low resolution'."))
    # --- AgentDojo
    nv = sorted(G["020_naive"]["cells"], key=lambda x: x["miss"])
    assert [x["cls"] for x in nv] == [UNDER] + [OVER] * 5
    ch.append(("abstract; introduction; section 3, Table 2 row 16; section 7.3",
               ["On AgentDojo's published runs the usual per-pair bound misses in 5-28% of resamples",
                "The per-pair Clopper-Pearson bound missed in 5-28% of resamples at delta 0.05 (Table 2).",
                "Independent-sample bounds fail on crossed designs"],
               f"Over in {n16[1]} of {n16[0]} pipelines ({f4(nv[1]['miss'])}-{f4(nv[-1]['miss'])}; {G['020_naive']['sum']['bonf']} "
               f"after Bonferroni by the widened band, {G['020_naive']['sum']['bonf_exact']} by exact p-values). The low end of "
               f"the printed range is the sixth: {desc(nv[0])}, at or under delta. 'Over its level in 5 of 6 pipelines, "
               f"{100 * nv[1]['miss']:.0f}-{100 * nv[-1]['miss']:.0f}%' is "
               "the rule's reading; '5-28%' counts a cell under delta as a miss of the level."))
    tw, ntw = sorted(G["020_twoway"]["cells"], key=lambda x: -x["miss"]), n3("020_twoway")
    ch.append(("section 3, Table 2 row 17",
               ["| **two-way bootstrap** | fails where positives sit in few clusters | AgentDojo | 0.05 | **up to 0.170** | [020 P] |"],
               f"Over in {ntw[1]} of {ntw[0]} pipelines ({f4(tw[0]['miss'])} and "
               f"{f4(tw[1]['miss'])}), at or under delta in {ntw[3]}; pooled over the six it is "
               f"{f4(G['020_twoway']['sum']['pooled'])}, {G['020_twoway']['sum']['pooled_cls'].split(',')[0]}. The `kind` column carries the qualifier; the count belongs "
               "beside it."))
    nu = n3("020_t_user")
    ci = max(G["020_t_inj"]["cells"], key=lambda x: x["miss"])
    assert ci["label"] == "Meta-SecAlign-70B" and ci["cls"] == OVER and n3("020_t_inj")[1] == 1
    ch.append(("section 3, Table 2 row 10; section 8.2; section 11",
               ["The check behind this is narrow: 6 of the 28 pipelines, 400 resamples each (standard error 0.011)",
                "Where we could check against a known truth, exact bounds and design-respecting resampling held"],
               f"Cluster bootstrap-t by user task: over in {nu[1]} of {nu[0]}, unresolved above delta in {nu[2]} "
               f"({', '.join(f4(x['miss']) for x in G['020_t_user']['cells'] if x['cls'] == UNRES)}), at or under in {nu[3]}. At 400 "
               f"draws the check resolves only a miss above {band(0.05, 400)}, so 'held' is 'was not over its level in six "
               f"pipelines'. The other half of section 8.2's certificate, the bootstrap-t by injection task, is over for one "
               f"pipeline in the same file ({ci['label']}: {desc(ci)}). For that pipeline the user-task bound is the larger of "
               "the two in Table 6 (0.104 against 0.035), so the rule would use it; the rule itself was not resampled, as the "
               "paper says."))
    # --- sheet as iid, judge alone
    ov2 = [x for x in w2 if x["cls"] == OVER]
    w4 = [x for x in G["017_sheet_ppi_iid"]["cells"] if x["wording"] == 4 and x["cls"] == OVER]
    assert all(x["cls"] == UNDER for x in G["017_sheet_ppi_iid"]["cells"] if x["wording"] == 0)
    assert all((x["cls"] == UNDER) == ("rate 0.2" in x["label"]) for x in w2)
    ch.append(("section 3, Table 2 row 18; section 7.5",
               ["Read as i.i.d., it certified a negative harm rate under one judge wording, and missed in 98% of re-drawn sheets "
                "[017 4]."],
               f"Over in {n18[1]} of {n18[0]} cells and at or under delta in {n18[3]}: {len(ov2)} of {len(w2)} for the wording the "
               f"paper names ({f4(min(x['miss'] for x in ov2))}-{f4(max(x['miss'] for x in ov2))}; its two cells at a 20% planted "
               f"rate are at or under), {len(w4)} of 6 for a second wording "
               f"({f4(min(x['miss'] for x in w4))}-{f4(max(x['miss'] for x in w4))}), none of 6 for the third. The sentence says "
               f"'one judge wording'; the row's 'fails' needs the same words. The largest miss is "
               f"{f4(G['017_sheet_ppi_iid']['sum']['largest'])} (the logit feature) and {f4(max(x['miss'] for x in w2 if 'f01' in x['label']))} "
               "(the 0/1 verdict); '98%' is the second."))
    jr = G["017_naive"]["cells"]
    one, nz = [x for x in jr if x["miss"] == 1], [x for x in jr if x["miss"] == 0]
    part = [x for x in jr if 0 < x["miss"] < 1]
    assert all("rubric" in x["label"] for x in one) and all("refusal raw" in x["label"] for x in nz)
    assert all("brevity raw" in x["label"] and x["cls"] == OVER for x in part) and len(one) + len(nz) + len(part) == len(jr)
    ch.append(("section 3, Table 2 row 19",
               ["| **the judge's rate alone** | fails | spike 017 plasmodes | 0.05 | **1.000** | [017 B6] |"],
               f"1.000 is the largest miss, reached in {len(one)} of {len(jr)} cells: every wording of the compiled rubric, whose "
               f"judged rate is far under the truth. The raw wording gives {rng(summarise(part))} in its {len(part)} cells on the "
               f"brevity label (over) and 0 in its {len(nz)} cells on the refusal label, where its judged rate (0.30 on the pool, "
               "against a truth of 0.20) is above the truth and the bound is never under it. At or under delta there says the "
               "judge over-reports, not that the route is valid. The row needs 'up to 1.000, where the judge under-reports'."))
    # --- section 8.4
    b2, a2, b1 = find("p9_new", "(b2)"), find("p9_new", "(a')"), find("p9_new", "(b1)")
    assert (b2["cls"], a2["cls"], b1["cls"]) == (UNRES, UNRES, OVER) and n3("p9_pool")[3] == 4
    ch.append(("section 8.4",
               ["puts the miss rates of the four limits at 0.001, 0.040, 0.045 and 0.042 for the pool rate against a level of 0.05"],
               f"All four at or under delta at 2,000 repetitions, as printed. The qualifier is the scheme: limit (b2) is built for "
               f"new prompts, and in the same check's new-prompts scheme, which the paper does not quote, it is {desc(b2)} and "
               f"the labels-alone bootstrap-t is {f4(a2['miss'])}, both unresolved above delta; (b1), not built for that scheme, "
               f"is {f4(b1['miss'])}, over."))
    wn, wc = G["017_ppi++"]["sum"]["worst"], G["017_carry_c200"]["sum"]["worst"]
    nc = n3("017_carry_c200")
    ch.append(("section 11",
               ["three common shortcuts failed by wide margins: a normal quantile at small rates, a per-pair bound on a crossed "
                "design, and a judge calibration carried to a policy it was not measured on"],
               f"Wide in the worst cells ({f4(wn['miss'])}, {f4(nv[-1]['miss'])}, {f4(wc['miss'])}), and each shortcut also has "
               f"cells that are not over: {n17[2] + n17[3]} of {n17[0]} for PPI++ with a normal limit, {n16[3]} of {n16[0]} for the "
               f"per-pair bound, {nc[2] + nc[3]} of {nc[0]} wordings for the calibration carried across the constrained run. "
               "'Failed in most cells, by wide margins in the worst' is what the counts carry."))
    # --- sentences the rule leaves as written
    n = n3("012_cp")
    assert max(x["k"] for x in G["012_cp"]["cells"]) == 1
    st.append(("section 4", "Clopper-Pearson missed in at most 1 of 2,000 runs in any cell [012].",
               f"{n[3]} of {n[0]} cells at or under delta 0.05; the largest is 1 of 2,000."))
    n, n1 = n3("p14_a_sboot_mid_0.05"), n3("p14_a_sboot_mid_0.1")
    st.append(("section 5", "The same estimator with a bootstrap-t limit holds in all ten cells (largest miss 0.045)",
               f"Stronger by checkpoint: {n[3]} of {n[0]} cells at or under delta at 0.05 and {n1[3]} of {n1[0]} at 0.1, none unresolved."))
    n = n3("p14_a_b1_mid_0.05")
    st.append(("section 5", "A stratified Wald-t limit on the same strata also held in all ten cells (largest miss 0.036",
               f"{n[3]} of {n[0]} checkpoint cells at or under delta."))
    ov = [x["miss"] for x in G["p14_b_sppi"]["cells"] if x["cls"] == OVER]
    op = [x["miss"] for x in G["p14_b_ppipp"]["cells"] if x["cls"] == OVER]
    st.append(("section 6", "StratPPI as published, with 5 or 10 strata, is over its level in 26 of 28 cells (misses 0.059-0.239 in "
               "those 26), as PPI++ with a normal limit is in all 14 (0.061-0.232).",
               f"{len(ov)} of 28 over ({f4(min(ov))}-{f4(max(ov))}), {G['p14_b_sppi']['sum']['bonf']} after Bonferroni; "
               f"{len(op)} of 14 over ({f4(min(op))}-{f4(max(op))}), {G['p14_b_ppipp']['sum']['bonf']} after Bonferroni."))
    k = {i: n3(f"017_carry_{i}") for i in ("x2o", "o2x", "h2r", "r2h", "side", "c100", "c200")}
    st.append(("section 7.1", "Between the two benign sources the bound was over its level for 4 of 6 wordings one way and 1 of 6 the "
               "other; between the pools, for 5 of 6 one way and none the other.",
               f"{k['x2o'][1]}, {k['o2x'][1]}, {k['h2r'][1]} and {k['r2h'][1]} of 6 over; every other cell is at or under delta, and "
               "the counts are the same after Bonferroni for six wordings."))
    c5 = find("017_carry_c200", "wording 5")
    st.append(("section 7.2", "the carried bound was over its level for 3 of 6 (misses 0.08, 0.23 and 0.80) and marginal for a fourth "
               "(0.054)",
               f"{k['c200'][1]} over, {k['c200'][2]} unresolved above delta ({desc(c5)}), {k['c200'][3]} at or under. 'Marginal' is "
               "the rule's 'unresolved'. The spike's own summary counts 4 of 6, by misses above 0.05; the paper's count is the rule's."))
    st.append(("section 7.2", "At step 100, where the rate had moved by 3.4 points, 1 of 6 failed.",
               f"{k['c100'][1]} of 6 over, {k['c100'][3]} at or under."))
    st.append(("section 7.2", "A calibration carried across 200 steps of one training run that moved the label only as a side effect",
               f"{k['side'][3]} of 6 at or under delta; the largest is {f4(G['017_carry_side']['sum']['largest'])}."))
    n = n3("017_block")
    st.append(("section 7.4", "Finite-sample judge-assisted bounds (betting on blocks) were valid and bought nothing",
               f"{n[3]} of {n[0]} cells at or under delta; the largest is {f4(G['017_block']['sum']['largest'])}."))
    n = n3("014_b1w_0.05", "014_b1w_0.1")
    st.append(("section 3, Table 2 rows 1, 3, 4, 5 and 7", "(rows as printed)",
               f"Every cell at or under delta: 5 each for Clopper-Pearson, the betting mixture, Bentkus, and Hoeffding and Anderson "
               f"(printed rates, 5,000 draws), and {n[3]} of {n[0]} for `b1w` on the pushed label."))
    return ch, st


def checks():
    """The printed values and counts that the report calls reproduced."""
    G = GROUPS
    for gid, lo, hi in (("r62_1", 0.067, 0.084), ("r62_3", 0.007, 0.016), ("r62_4", 0.023, 0.035), ("r62_5", 0, 0),
                        ("017_classical", None, 0.051), ("014_b1w_0.05", 0.011, 0.023), ("014_b1w_0.1", 0.040, 0.064),
                        ("017_sheet_b1w", 0.001, 0.059), ("017_boot", None, 0.053), ("020_t_user", 0.020, 0.060),
                        ("p14_a_sboot_mid_0.05", None, 0.045), ("p14_b_sboot", None, 0.056), ("017_ppi++", None, 0.241),
                        ("p14_a_sppi_mid_0.05", None, 0.086), ("p14_b_sppi", None, 0.239), ("020_naive", 0.048, 0.275),
                        ("020_twoway", None, 0.170), ("013_inloop_strat", 0.084, 0.116), ("004_miss_any_delta_T", 0.025, 0.100),
                        ("p9_pool", 0.001, 0.045)):
        s = G[gid]["sum"]
        assert abs(s["largest"] - hi) < 0.00051 and (lo is None or abs(s["lowest"] - lo) < 0.00051), gid
    assert largest_of_checkpoints("p14_a_sppi_mid_0.05")["over"] == 4 and largest_of_checkpoints("p14_a_sppi_mid_0.1")["over"] == 3
    assert n3("p14_b_sppi")[1] == 26 and n3("p14_b_ppipp")[1] == 14
    assert [n3(f"017_carry_{i}")[1] for i in ("x2o", "o2x", "h2r", "r2h", "c200", "c100")] == [4, 1, 5, 0, 3, 1]
    assert not any(c.get("rounding_sensitive") for g in G.values() for c in g["cells"])     # printed rates: class is safe


def report(fams):
    G = GROUPS
    paper = re.sub(r"\s+", " ", rd(PAPER))
    cited = sorted((g for g in G.values() if order(g)[0] < 2), key=order)
    rest = [g for g in G.values() if order(g)[0] == 2]
    n_cells = sum(len(g["cells"]) for g in G.values() if not g["dup"])
    n_draws = sum(c["draws"] or 0 for g in G.values() if not g["dup"] for c in g["cells"])
    L = ["# Validity recount under one rule", "",
         f"`scripts/validity_recount.py`; source: `{PAPER}` (draft v0.5) and the result files it cites. Nothing was simulated: "
         f"{n_cells:,} cells and {n_draws:,} draws are counted from existing files, or read from the training paper's printed "
         "tables where the data are lost (marked `~`).", "",
         "**The rule.** A cell is one resampling study: R draws at level delta, miss m. With se = sqrt(delta (1 - delta) / R): "
         "*over* if m > delta + 2 se; *unresolved, above delta* if delta < m <= delta + 2 se; *at or under delta* if m <= delta. "
         "The exact one-sided binomial p-value of H0 'true miss <= delta' and a 95% Clopper-Pearson interval are in the JSON for "
         "every cell, and below for the cells that matter. *Over after Bonferroni*: the 2 se replaced by z(1 - a / C) se for the "
         f"C cells of the row, a = 1 - Phi(2) = {ALPHA:.4f}, so one cell gives the rule itself.", "",
         "**What a cell is.** The unit a source file stores: one label, sample size, checkpoint, feature or pipeline, with its own "
         "draws. The paper counts some rows differently (the largest miss of 2-3 checkpoints as one cell); both counts are given "
         "where they differ.", "",
         "**Three cautions.** (1) Cells of one row are not independent: the two deltas of spikes 013 and 014 use the same draws, "
         "the four judge features of spike 017 share a cell's draws, and arms are paired. Bonferroni is then conservative, and the "
         "pooled miss is a description, not a test. (2) The 4,000 draws of a sheet cell are 20 plantings x 200 sheets; the "
         "per-planting counts were not stored, so the binomial treats them as 4,000. (3) *At or under delta* is a statement about "
         "the estimate; with 200-500 draws its interval still reaches well above delta (see the largest-miss column).", "",
         "## (a) Per-bound summary", "", "Rows the paper quotes, in the order of Table 2 and then by section.", ""]
    L += SUMMARY_HEAD + [summary_row(g) for g in cited]
    L += ["", "Bounds in the same files that the paper does not quote.", ""] + SUMMARY_HEAD + [summary_row(g) for g in rest]
    L += ["", "The same bound over all its settings at one delta (cells that repeat another row's draws left out).", "",
          "| bound | delta | rows pooled " + HEAD, "|" + "---|" * 12]
    for (fam, d), gs in fams.items():
        L.append(summary_row(None, S(*[g["id"] for g in gs]), f"| {fam} | {d} | {len(gs)} "))
    diff = [g for g in G.values() if g["sum"]["counts"] and g["sum"]["bonf"] != g["sum"]["bonf_exact"]]
    L += ["", "Bonferroni by exact p-values (p < a / C) in place of the widened band gives the same count in every row except: "
          + "; ".join(f"{g['bound']}, {g['setting'].split(',')[0]} ({g['sum']['bonf_exact']} against {g['sum']['bonf']})" for g in diff) + "."]

    L += ["", "## (b) Table 2, recounted", "",
          "*Table 2. Miss rate against delta for every bound used, one rule for every row. A cell is over its level when its miss "
          "exceeds delta by more than two Monte Carlo standard errors, unresolved when it is above delta inside that band, and at "
          "or under delta otherwise. `~`: printed rates, not regenerable. Bold and the `kind` column are the paper's and are not "
          "derived from the counts.*", ""] + table2()
    t, sh = G["r62_12"]["cells"], [c for c in G["017_sheet_ppi_iid"]["cells"] if c["wording"] == 2]
    L += ["", f"- [a] The paper's row covers n 200-800: 3 cells, all over ({rng(summarise(t[:3]))}). The source table has n 1,200 "
          f"and 2,400 as well ({f4(t[3]['miss'])} each, unresolved).",
          f"- [b] The paper's row is one wording: 6 cells, {sum(c['cls'] == OVER for c in sh)} over and "
          f"{sum(c['cls'] == UNDER for c in sh)} at or under ({rng(summarise(sh))}).",
          "- [c] Cells are label x size x checkpoint. The paper's printed range is of the largest miss over the checkpoints: "
          + "; ".join(f"{name} {x['cells']} such cells, {x['over']} over, {x['above']} above delta ({f4(x['lowest'])}-{f4(x['largest'])})"
                      for name, x in (("row 6 at 0.05,", largest_of_checkpoints("013_b1w_mid_0.05")),
                                      ("row 6 at 0.10,", largest_of_checkpoints("013_b1w_mid_0.1")),
                                      ("row 11, reference-rate strata,", largest_of_checkpoints("p14_a_sboot_mid_0.05")),
                                      ("row 13, `b1w`,", largest_of_checkpoints("013_b1w_rare100")),
                                      ("row 13, pooled Wilson,", largest_of_checkpoints("013_wilson_rare100")),
                                      ("row 15, reference-rate strata,", largest_of_checkpoints("p14_a_sppi_mid_0.05")))) + ".",
          "- Rows 1, 3, 4, 5 and 12: counts rebuilt as round(m x 5,000) from rates printed to three decimals; no class changes "
          "within the rounding. Row 5 is one printed row for two bounds."]

    ch, st = sentences()
    L += ["", "## (c) Sentences whose hold/fail wording changes or needs a qualifier", ""]
    missing = []
    for i, (where, quotes, text) in enumerate(ch, 1):
        L.append(f"{i}. **{where}.**")
        for q in quotes:
            if re.sub(r"\s+", " ", q) not in paper:
                missing.append(q)
            L.append(f"   > {q}")
        L += ["", f"   {text}", ""]
    L += ["Sentences checked that stand as written:", ""]
    for where, q, text in st:
        if not q.startswith("(") and re.sub(r"\s+", " ", q) not in paper:
            missing.append(q)
        L.append(f"- **{where}.** \"{q}\" {text}")
    assert not missing, missing        # every quotation is in the draft, word for word

    L += ["", "## (d) The two lists the rule produces", "",
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

    lo6, lo61 = largest_of_checkpoints("013_b1w_mid_0.05"), largest_of_checkpoints("013_b1w_mid_0.1")
    lb, lw = largest_of_checkpoints("013_b1w_rare100"), largest_of_checkpoints("013_wilson_rare100")
    jr = G["017_naive"]["cells"]
    nz = sum(c["miss"] == 0 for c in jr)
    fig_low = re.search(r"the judge.s rate alone,[\d.]+,([\d.]+)", rd("reports/figs/fig1_validity.csv")).group(1)
    L += ["", "## (e) Printed numbers against their sources", "",
          "| where | the paper | the source, per cell | how the printed value arises |", "|---|---|---|---|",
          f"| Table 2 row 6 | 0.023-0.054 / 0.069-0.097 | {rng(G['013_b1w_mid_0.05']['sum'])} / {rng(G['013_b1w_mid_0.1']['sum'])} | "
          f"the largest of 3 checkpoints per label and size: {f4(lo6['lowest'])}-{f4(lo6['largest'])} / {f4(lo61['lowest'])}-{f4(lo61['largest'])} |",
          f"| Table 2 row 13; section 3 reading 1 | 0.24-0.44 | {rng(S('013_b1w_rare100', '013_wilson_rare100'))} | the stratified "
          f"bound's largest checkpoint per label, {f4(lb['lowest'])} and {f4(lb['largest'])}; the pooled bound's are "
          f"{f4(lw['lowest'])} and {f4(lw['largest'])} |",
          f"| Table 2 row 12 | n 200-800, 0.115-0.184 | n 200-2,400, {rng(G['r62_12']['sum'])} | the three cells the source prints in bold |",
          f"| Table 2 row 18; section 7.5 | up to 0.98; 98% | largest {f4(G['017_sheet_ppi_iid']['sum']['largest'])} | the 0/1-verdict "
          f"cell ({f4(max(c['miss'] for c in sh if 'f01' in c['label']))}); the logit cell is higher |",
          f"| Table 2 row 19 | 1.000 | 1.000 in {sum(c['miss'] == 1 for c in jr)} cells, {rng(summarise([c for c in jr if 0 < c['miss'] < 1]))} "
          f"in {sum(0 < c['miss'] < 1 for c in jr)}, 0 in {nz} | the largest over cells (spike 017's table B6). "
          f"`reports/figs/fig1_validity.csv` prints a lowest miss of {fig_low} for this row |",
          f"| Table 2 row 16; abstract; section 7.3 | 0.048-0.275; 5-28% | {rng(G['020_naive']['sum'])} | reproduces; the low end is at or under delta |",
          "| Table 2 rows 2 and 8 | at most 0.051; 0.001-0.059 | " + f"{f4(G['017_classical']['sum']['largest'])}; {rng(G['017_sheet_b1w']['sum'])} | "
          "reproduces; 0.0515 and 0.0595 print as 0.051 and 0.059 in the sources' three decimals |",
          "| Table 2 caption | se 0.003-0.004 (5,000), 0.011 (400) | also 4,000, 2,000, 1,000, 500 and 200 draws | see (c) item 2 |",
          "", "Reproduced to the printed digits (asserted in the script): Table 2 rows 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 14, 15, 16 "
          "and 17; the counts '4 of 10', '3 of 10', '26 of 28' and 'all 14' of sections 5 and 6 under the paper's own cells; "
          "section 4's 0.084-0.116 and 0.025-0.100; section 7's 4, 1, 5, 0, 3 and 1 of 6; section 8.4's four rates."]

    lost = G["r63_lag_ab"]["sum"]["cells"] + G["r63_judge"]["sum"]["cells"]
    top63 = max(G["r63_lag_ab"]["sum"]["largest"], G["r63_judge"]["sum"]["largest"])
    L += ["", "## (f) Not recounted, and why", "",
          "- **[R 6.2], Table 2 rows 1, 3, 4, 5, 12.** Not regenerable: the cached labels were lost on 2026-09-19. Classified from the "
          "printed rates and the 5,000 resamples the training paper states; counts are round(m x 5,000).",
          "- **[R 6.2], the Student-t bound at delta 0.05.** The training paper's sentence (0.084, 0.067, 0.083, 0.074, 0.067) "
          "repeats its Clopper-Pearson row in four of five values (the paper's Appendix A, open check 2). Not classified; Table 2 "
          "does not use it.",
          f"- **[R 6.3], section 4, tables (a), (b) and (d).** {lost} printed cells with no class: the training paper gives '500-1,000 "
          f"independent trials' a row and no count per row. Every printed rate (at most {f4(top63)}) is under delta 0.1 whatever the "
          "count; the p-values and intervals need it. Table (c) is classified at the 500 trials of the reproduction command in "
          "that paper's Appendix B. Regenerable (`scripts/synthetic_calibration.py`), not re-run here.",
          "- **Section 4, language-model policies.** 0 breaches in 3 seeds and in 10 seeds: the paper already declines to read "
          "these as a miss rate, and three or ten draws give no class worth the name.",
          "- **Section 8.2, the larger-of-two clustered bound.** The certificate the paper uses was not resampled in the source; "
          "its two halves were, one at a time, with user tasks resampled.",
          "- **Clustering of the sheet draws.** `harm.json` stores one rate per cell, not per planting, so a planting-level "
          "standard error cannot be formed without re-running `harm017.py`.",
          "- **[SR 2.4].** No row of Table 2 in v0.5 carries this tag; the state report's section 2.4 digests spike 017, whose "
          "cells are counted from its own files above.",
          "- **`plasmode_n500.json` (spike 017).** Not part of the spike's table B6 and not quoted in the paper; left out."]

    L += ["", "## Appendix. The cells behind (c) and (d)", "",
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
    for load in (load_training_paper, load_012, load_013_014, load_004, load_017, load_020, load_p14, load_p9):
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
    ch, st = sentences()
    json.dump(dict(rule=dict(z=Z, alpha_one_sided=ALPHA, classes=CLASSES), groups=list(GROUPS.values()),
                   families=[dict(family=k[0], delta=k[1], groups=[g["id"] for g in v], sum=S(*[g["id"] for g in v]))
                             for k, v in fams.items()],
                   table2=[dict(bound=r[0], kind=r[1], setting=r[2], delta=r[3], paper_miss=r[4], source=r[5],
                                parts=[dict(groups=p, sum=S(*p)) for p in r[7]]) for r in TABLE2],
                   sentences=dict(change=[dict(where=w, quotes=q, rule=t) for w, q, t in ch],
                                  stand=[dict(where=w, quote=q, rule=t) for w, q, t in st])),
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
