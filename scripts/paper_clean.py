"""A clean, venue-agnostic copy of the certification paper (paper plan P13).

Reads the working draft, ``reports/paper_certification.md``, and writes
``reports/paper_certification_clean.md``: no draft header, no source tags, no gap or check
markers, no references to the plan or to spike numbers, figures placed where they are first
cited, and a short appendix on reproducing the numbers. The working draft stays the source: edit it and run this again.

    python3 scripts/paper_clean.py
"""
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SRC = os.path.join(ROOT, "reports", "paper_certification.md")
DST = os.path.join(ROOT, "reports", "paper_certification_clean.md")

TAG = r"(?:\n[ \t]*|[ \t]+)?\[(?:R |SR |P\d|0\d\d|StratPPI validation|AgentDojo recheck|validity recount|judge on labels|replacement check|Wilson exact|robot sampling|binomial rows|StratPPI heuristic cell|two-way bounds|two-phase check)[^\]]*\]"
# old figure number -> (file, caption); Figure 1 of the draft (Table 2 redrawn) is left out
FIGS = {2: ("fig2_strata_ess", "Gain from reference-rate strata against the reference model's intraclass correlation."),
        3: ("fig3_stratppi", "StratPPI's normal limit and the same estimator with a bootstrap-t limit, with labels allocated in "
                             "proportion to stratum size. Left: reference-rate strata. Right: strata on a judge's logit."),
        4: ("fig4_carrying", "A carried calibration: miss rate of the carried bound and the judge's recall, by shift."),
        5: ("fig5_robodojo", "What a benchmark of 120 trials per model can certify."),
        6: ("fig6_agentdojo", "AgentDojo: the per-pair bound and the bound clustered by user task for all 28 pipelines, "
                              "from 200,000 bootstrap draws."),
        7: ("fig7_guard_xstest", "What the guard flags, by human class, on XSTest's published completions."),
        8: ("fig8_refusal_sheet", "Refusal rates of the reference model and the trained policy, by the guard's flag and by human label.")}
PHRASES = [
    (r"spike 017 plasmodes", "the plasmodes of section 8.2"),
    (r"the refusal pools\s+of spike 017", "the refusal pools of those plasmodes"),
    (r"spike 013's figures", "an earlier measurement's figures"),
    (r"\(spike 014\)", "(the constrained run of section 9.2)"),
    (r"\(Round 6, seed 1\)", "(one run of the companion paper)"),
    (r"the project's `SeldonianLLMPolicy`", "our `SeldonianLLMPolicy`"),
    (r",\s+which an earlier draft used here \((\d+)\)", r" (\1)"),
    (r", and an earlier draft reported 0\.104", ""),
    (r"the training\s+paper's", "the companion paper's"),
    (r"the training\s+paper", "the companion paper"),
    (r"\(Draft notes 1\)", "(Appendix E)"),
]
APPENDIX = """## Appendix E. Reproducing the numbers

The numbers in this paper come from the scripts below, apart from those cited from the companion
paper and a few recomputed from the data files. The working draft carries a source tag on each
number and a table that resolves each tag to a file.

| what | script | output |
|---|---|---|
| Table 4, every validity cell under one rule; the table and the counts in the text asserted against the files | `scripts/validity_recount.py` | `results/paper/validity_recount.md` |
| section 7.1, the reference-rate-strata cells redrawn with replacement | `scripts/replacement_check.py` | `results/paper/replacement_check.md` |
| section 7.1, the exact miss probability of the Wilson limit | `scripts/wilson_exact.py` | `results/paper/wilson_exact.md` |
| Table 4, the labels-alone rows by enumeration | `scripts/binomial_rows.py` | `results/paper/binomial_rows.md` |
| Table 4, the four judge-plasmode cells with 20,000 unlabelled responses, at 4,000 draws | `scripts/plasmode017_big.py` | `results/paper/plasmode017_big.json` |
| section 8.1, the strata for a claim about the prompt source | `scripts/twophase_check.py` | `results/paper/twophase_check.md` |
| end-point tests of every bound used | `tests/test_bound_endpoints.py` | run with `pytest` |
| sections 8.1 and 8.2, the StratPPI comparison | `scripts/stratppi_baseline.py` | `results/paper/stratppi.md` |
| sections 8.1 and 8.2, implementation check, allocations, PPBoot, sweeps | `scripts/stratppi_validate.py` | `results/paper/stratppi_validate.md` |
| section 9.2, the judge against human labels | `scripts/judge_on_labels.py` | `results/labels/refusal/rubric_vs_human.md` |
| Appendix A, the heuristic allocation in its worst cell | `scripts/stratppi_heur_cell.py` | `results/paper/stratppi_heur_cell.md` |
| section 10.1, the robot benchmark's sampling unit | `scripts/robodojo_sampling.py` | `results/paper/robodojo_sampling.md` |
| section 10.2, AgentDojo | `scripts/agentdojo_recheck.py` | `results/paper/agentdojo_recheck.md` |
| section 10.2, the two-way bounds (registered) | `scripts/agentdojo_twoway.py` | `results/paper/agentdojo_twoway.md` |
| section 10.3, the guard on XSTest | `scripts/xstest_guard.py` | `results/labels/xstest/analysis.md` |
| section 10.3, the refusal sheet | `scripts/refusal_sheet_build.py`, `scripts/refusal_labels.py` | `results/labels/refusal/analysis.md` |
| section 10.4, the label budget | `scripts/p9_budget.py` | `results/paper/p9_budget.md` |
| section 10.4, the prepared certificate and its design check | `scripts/p9_sample.py`, `scripts/p9_certificate.py` | `results/labels/p9/design_check.md` |
| figures | `scripts/paper_figures.py` | `reports/figs/` |
| references checked against arXiv and CrossRef | `scripts/bib_check.py` | printed report |

Results cited from the companion paper on training (parts of section 7.2) are printed rates whose
per-draw data were lost; they cannot be regenerated without retraining. The rows of Table 4 on a
trained-policy harm pool are the exception: they are reproduced by enumeration.
"""


def drop_last_column(block):
    return "\n".join("|".join(line.rstrip().rstrip("|").split("|")[:-1]) + "|" for line in block.split("\n"))


def main():
    s = open(SRC).read()
    title = s[:s.index("\n")]
    s = s[s.index("## Abstract"):s.index("## Draft notes 1.")]
    # the validity table: its caption names sources, and its last column is the source tag
    cap = re.search(r"Draws per cell are 5,000 for .*?plasmodes\.", s, re.S)
    s = s.replace(cap.group(0), "Draws per cell are 4,000 or 5,000, and 40,000 for the rows redrawn with replacement.")
    for m in list(re.finditer(r"(?:^\|.*\n)+", s, re.M)):
        if m.group(0).split("\n")[0].rstrip().endswith("| source |"):
            s = s.replace(m.group(0), drop_last_column(m.group(0).rstrip("\n")) + "\n")
    s = re.sub(r"[ \t]*`\[GAP: P\d+\]`[^.]*\.[ \t]*", "", s)
    s = re.sub(TAG, "", s)
    for old, new in PHRASES:
        s = re.sub(old, new, s)
    # figures: drop the draft's Figure 1, renumber, and place each where it is first cited
    s = re.sub(r"Figure 1 draws the table below as miss rate over delta\.\n\n", "", s)
    s = re.sub(r"Figure (\d)", lambda m: f"Figure {int(m.group(1)) - 1}", s)
    blocks = s.split("\n\n")
    placed = set()
    out = []
    for blk in blocks:
        out.append(blk)
        for n in sorted({int(x) + 1 for x in re.findall(r"Figure (\d)", blk)}):
            if n in FIGS and n not in placed:
                placed.add(n)
                name, caption = FIGS[n]
                out.append(f"![Figure {n - 1}](figs/{name}.png)\n\n*Figure {n - 1}. {caption}*")
    s = "\n\n".join(out)
    head = (f"{title}\n\n*Draft of 2026-10-07, not formatted for any venue. Generated from the working draft by\n"
            "`scripts/paper_clean.py`; to change the text, edit the working draft and run the script again.*\n\n")
    s = head + s.rstrip() + "\n\n" + APPENDIX
    s = re.sub(r"\n{3,}", "\n\n", s)
    left = [w for w in ("[GAP", "[CHECK", "spike", "PLAN.md", "earlier draft", "training paper", "Draft notes")
            if w in re.sub(r"\s+", " ", s)]
    left += sorted(set(re.findall(TAG, s)))
    missing = sorted(set(FIGS) - placed)
    open(DST, "w").write(s)
    print(f"{DST}: {len(s.split())} words, {len(placed)} figures placed")
    if left or missing:
        for w in left:
            for line in s.split("\n"):
                if w.strip() in line:
                    print("  LEFT:", w.strip(), "|", line[:140])
        print("  figures not placed:", missing)
        sys.exit(1)


if __name__ == "__main__":
    main()
