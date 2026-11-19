"""Check the workshop version's numbers against the long draft.

Every number written in ``reports/workshop/trust4rsi_aaai27.tex`` and ``reports/workshop/appendix.tex`` (the generated
tables are copied cell for cell and are not checked here) must occur in the long draft's clean copy (or in one of the RESULTS files it quotes directly),
``reports/paper_certification_clean.md``, whose numbers ``scripts/validity_recount.py`` asserts against the result files.
This catches a mistyped or invented figure; it does not check that a number is used with the right meaning, which is
the fresh-agent source check's job. Numbers the workshop text derives on its own are listed in ALLOWED with a reason.

    python3 scripts/workshop_check.py
"""
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SRC = os.path.join(ROOT, "reports", "paper_certification_clean.md")
TEX = [os.path.join(ROOT, "reports", "workshop", f) for f in ("trust4rsi_aaai27.tex", "appendix.tex")]
# result files the workshop text quotes directly, where the long draft gives a derived figure instead
RESULTS = [os.path.join(ROOT, "results", "paper", f) for f in
           ("b4/b4.md", "c1_human_certificate.json", "m1_abstention_mc.md", "m1_conditional.md")] + \
          [os.path.join(ROOT, "results", f) for f in                          # per-family section (2026-11-16)
           ("per_family/report.md", "per_family_f/report.md", "twin/stage_a/report.md", "twin/e6/report.md", "per_family_equal/report.md", "paper/waldt_fix/confirm.md", "paper/waldt_fix/dev.md", "paper/waldt_fix/gains.md", "paper/agentdojo_closed.md", "paper/trunc/trunc.md", "paper/rubric7/rubric7.md")]

ALLOWED = {
    "0.62": "60 of 97 = 0.619, rubric7/rubric7.md", "648": "4 shapes x 3 strata counts x 9 rates x 2 weightings x 3 sizes, W2_registration.md", "232": "116 cells x 2 levels, waldt_fix/dev.md", "196": "real and costly pairs in waldt_fix/dev.md: (18+24+16+8+28+4) x 2",
    "2027": "template version", "12": "GPU memory in GB, and section-free counts", "100": "percent and sizes",
    "0.0": "axis", "1.0": "axis", "2027.1": "template version", "3.5": "column spacing", "3.65": "column width", "4.25": "column width", "2.5": "column spacing", "3.12": "Python version", "4070": "GPU model", "5.16": "Transformers version", "7800": "CPU model", "0.4": "2,444/6,500 = 0.38, per_family_equal/report.md", "0.6": "2,444/4,000 = 0.61", "250": "fine grid step, scripts/per_family_equal_budget.py", "85": "600 // 7 labels per family", "0.9535": "results/paper/validity_recount.md: Clopper-Pearson at p 0.9535, n 100", "187.2": "median forward KL, /mnt/d/seldonian-runs/twin/e1/forward_kl.json (41 prompts), off-repo", "1.5": "per-family F vs A n80 max gap 1.3%, results/per_family_f/report.md", "0.8": "similarity gate, REGISTRATION-STAGE-A.md", "108": "83 + 25 guard flags of seed 1 at step 200, b4.md table D",
}


def numbers(text):
    text = re.sub(r"%.*", "", text)                                            # comments
    text = re.sub(r"\\(label|ref|input|includegraphics|cite[pt]?|bibliography|setcounter|setlength)(\[[^]]*\])?\{[^}]*\}", " ", text)
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = text.replace("{,}", ",").replace("--", "-")
    return re.findall(r"(?<![\w.])\d[\d,]*(?:\.\d+)?", text)


def canon(n):
    return n.replace(",", "")


def main():
    src = open(SRC).read() + "\n".join(open(f).read() for f in RESULTS)
    have = {canon(n) for n in re.findall(r"(?<![\w.])\d[\d,]*(?:\.\d+)?", src)}
    have |= {canon(n).lstrip("0") or "0" for n in have}
    missing = []
    for path in TEX:
        for n in numbers(open(path).read()):
            c = canon(n).rstrip(",")
            if c in have or c in ALLOWED or (c.startswith("0.") and c.rstrip("0") in have):
                continue
            missing.append((os.path.basename(path), n))
    for f, n in sorted(set(missing)):
        print(f"not in the long draft: {f}: {n}")
    print(f"{len(set(missing))} numbers not found")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
