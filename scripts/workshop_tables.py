"""LaTeX tabulars for the workshop version's appendices, taken from the long draft's clean copy.

The appendix tables of ``reports/workshop/trust4rsi_aaai27.tex`` are the long draft's tables, carried over cell for
cell so that no number is retyped: this script reads ``reports/paper_certification_clean.md`` (run
``scripts/paper_clean.py`` first), finds each table by the start of its caption, converts its Markdown cells to LaTeX,
rewrites the long draft's section references to the workshop version's, strips commit hashes, and writes one tabular
per file into ``reports/workshop/tables/``. Captions are written in the .tex, not here.

    python3 scripts/workshop_tables.py
"""
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SRC = os.path.join(ROOT, "reports", "paper_certification_clean.md")
OUT = os.path.join(ROOT, "reports", "workshop", "tables")

# long draft's section -> the workshop version's place for it
SECTIONS = {
    "7.1": "App.~\\ref{app:validity}", "7.2": "App.~\\ref{app:pipeline}", "8.1": "App.~\\ref{app:strata}",
    "8.1.1": "App.~\\ref{app:strata}", "8.1.3": "App.~\\ref{app:strata}", "8.1.4": "App.~\\ref{app:strata}",
    "8.2": "App.~\\ref{app:judge}", "8.1, 9.1, 9.4": "Apps.~\\ref{app:strata}, \\ref{app:judge}",
    "9.2": "App.~\\ref{app:judge}", "10.1": "App.~\\ref{app:bench}", "10.2": "App.~\\ref{app:bench}",
    "10.3": "App.~\\ref{app:human}", "10.4": "App.~\\ref{app:human}", "8.2, 9.1, 9.4": "App.~\\ref{app:judge}",
    "7.2, 8.1.4": "Apps.~\\ref{app:pipeline}, \\ref{app:strata}",
}

# table id -> (caption start in the clean copy, column spec, rows to keep or None, output name)
TABLES = [
    ("Table 4.", r"@{}r p{2.8cm} p{2.0cm} p{3.9cm} l p{2.1cm} p{2.0cm} p{1.9cm}@{}", range(1, 19), "validity_a"),
    ("Table 4.", r"@{}r p{2.8cm} p{2.0cm} p{3.9cm} l p{2.1cm} p{2.0cm} p{1.9cm}@{}", range(19, 32), "validity_b"),
    ("Table 4.", r"@{}r p{2.8cm} p{2.0cm} p{3.9cm} l p{2.1cm} p{2.0cm} p{1.9cm}@{}", range(32, 44), "validity_c"),
    ("Table 4.", r"@{}r p{2.8cm} p{2.0cm} p{3.9cm} l p{2.1cm} p{2.0cm} p{1.9cm}@{}", range(44, 56), "validity_d"),
    ("Table A1.", r"@{}p{4.2cm} r p{2.4cm} p{2.6cm} p{3.4cm}@{}", None, "stratppi"),
    ("Table 6.", r"@{}l r r r r r p{2.6cm} r@{}", None, "robot"),
    ("Table 7.", r"@{}p{3.0cm} r r r r r l r p{1.6cm}@{}", None, "agentdojo"),
    ("Table 8.", r"@{}p{4.0cm} p{2.6cm} r p{2.6cm} p{2.6cm}@{}", None, "xstest"),
    ("Table 9.", r"@{}p{5.6cm} p{3.0cm} p{4.6cm}@{}", None, "human"),
    ("Table 10.", r"@{}r r r r r p{3.2cm}@{}", None, "cost"),
    ("Table D0.", r"@{}p{8.6cm} r r r@{}", None, "d0"),
    ("Table D1.", r"@{}l p{7.4cm} p{7.4cm}@{}", None, "d1"),
    ("Table D2.", r"@{}l p{7.4cm} p{7.4cm}@{}", None, "d2"),
    ("Table D3.", r"@{}l p{7.4cm} p{7.4cm}@{}", None, "d3"),
    ("Table D4.", r"@{}l p{7.4cm} p{7.4cm}@{}", None, "d4"),
    ("## Appendix D.", r"@{}p{5.6cm} l p{9.2cm}@{}", None, "record"),
]

WORDS = [
    (r"stratified Wilson-type `b1w`", "stratified Wilson-type limit"),
    (r"`b1w`", "the Wilson-type limit"),
    (r"\brho\^2\b", r"$\\rho^2$"),
    (r"`ESS_truth`", r"$\\mathrm{ESS}_{\\mathrm{truth}}$"), (r"`ESS_est`", r"$\\mathrm{ESS}_{\\mathrm{est}}$"),
    (r"`ICC_cand`", r"$\\mathrm{ICC}_{\\mathrm{cand}}$"), (r"`omega`", r"$\\omega$"), (r"`rho\^2`", r"$\\rho^2$"),
    (r"\bn_s\b", r"$n_s$"), (r"\bdelta\b", r"$\\delta$"), (r"(?<!\\)\brho\b(?!\^)", r"$\\rho$"),
    (r"the plasmodes of section 8\.2", "the judge plasmodes (App.~\\\\ref{app:judge})"),
]


def find_table(lines, start):
    i = next((k for k, l in enumerate(lines) if l.startswith(start) or l.startswith("*" + start)), None)
    assert i is not None, f"caption not found: {start}"
    j = next(k for k in range(i, len(lines)) if lines[k].startswith("|"))
    rows = []
    while j < len(lines) and lines[j].startswith("|"):
        rows.append(lines[j])
        j += 1
    head, body = rows[0], rows[2:]
    split = lambda r: [c.strip() for c in r.strip().strip("|").split("|")]
    return split(head), [split(r) for r in body]


def cell(s, section_col=False):
    s = re.sub(r"\(`[0-9a-f]{7}`, (?=[a-z])", "(", s)                          # commit hashes
    s = re.sub(r"\s*\(`[0-9a-f]{7}`(?:, `[0-9a-f]{7}`)*\)", "", s)
    assert not re.search(r"`[0-9a-f]{7}`", s), s
    for a, b in WORDS:
        s = re.sub(a, b, s)
    for k in sorted(SECTIONS, key=len, reverse=True):
        s = re.sub(rf"\bsections? {re.escape(k)}\b(?!\.\d)", lambda m, k=k: SECTIONS[k], s)
    s = re.sub(r"\bTable 5\b", lambda m: "Table~\\ref{tab:routing}", s)
    s = re.sub(r"\bTable D([0-4])\b", lambda m: "Table~\\ref{tab:d" + m.group(1) + "}", s)
    assert not re.search(r"\bTable [A-Z]?\d", s), f"unmapped table reference: {s}"
    if section_col:
        assert s in SECTIONS, f"unmapped section: {s}"
        s = SECTIONS[s]
    assert not re.search(r"\bsections? \d", s), f"unmapped section reference: {s}"
    parts = re.split(r"(\$[^$]*\$|\\ref\{[^}]*\})", s)                      # escape outside math and refs only
    for i in range(0, len(parts), 2):
        parts[i] = parts[i].replace("%", r"\%").replace("&", r"\&").replace("#", r"\#").replace("_", r"\_")
    s = "".join(parts)
    s = s.replace("†", r"$^\dagger$").replace("~", "~")
    s = re.sub(r"`([^`]+)`", lambda m: r"\texttt{" + m.group(1) + "}", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\\emph{\1}", s)
    s = re.sub(r"(?<![\w.-])(\d[\d.,]*)-(?=\d[\d.,]*(?![\w-]))(?!\d\d-\d\d)", r"\1--", s)
    s = s.replace('"', "''").replace(" ''", " ``")
    if s.startswith("''"):
        s = "``" + s[2:]
    return s


def main():
    lines = open(SRC).read().split("\n")
    os.makedirs(OUT, exist_ok=True)
    for start, spec, keep, name in TABLES:
        head, body = find_table(lines, start)
        if keep is not None:
            body = [r for r in body if r[0].isdigit() and int(r[0]) in keep]
            assert [int(r[0]) for r in body] == list(keep), name
        ncol = len(spec.replace("@{}", "").split()) if "p{" not in spec else len(re.findall(r"[lrc]|p\{[^}]+\}", spec.replace("@{}", "")))
        assert all(len(r) == len(head) == ncol for r in body), (name, len(head), ncol, {len(r) for r in body})
        out = [r"\begin{tabular}{" + spec + "}", r"\toprule",
               " & ".join(r"\textbf{" + cell(h) + "}" if h else "" for h in head) + r" \\", r"\midrule"]
        for r in body:
            if name == "d4" and re.fullmatch(r"Q\d", r[0]):              # the long draft reuses Q for two registrations
                r = ["J" + r[0][1:]] + r[1:]
            out.append(" & ".join(cell(c, name == "record" and k == 1) for k, c in enumerate(r)) + r" \\")
        out += [r"\bottomrule", r"\end{tabular}"]
        open(os.path.join(OUT, name + ".tex"), "w").write("\n".join(out) + "\n")
        print(f"{name}: {len(body)} rows")


if __name__ == "__main__":
    sys.exit(main())
