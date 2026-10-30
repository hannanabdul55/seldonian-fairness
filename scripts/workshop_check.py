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
           ("b4/b4.md", "c1_human_certificate.json", "m1_abstention_mc.md")]

ALLOWED = {
    "2027": "template version", "12": "GPU memory in GB, and section-free counts", "100": "percent and sizes",
    "0.0": "axis", "1.0": "axis", "2027.1": "template version", "3.5": "column spacing", "3.65": "column width", "3.12": "Python version", "4070": "GPU model", "5.16": "Transformers version", "7800": "CPU model", "108": "83 + 25 guard flags of seed 1 at step 200, b4.md table D",
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
