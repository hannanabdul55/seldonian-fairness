"""A PDF of a plain Markdown report (not the paper), reusing the paper's LaTeX helpers.

Handles what the reports use: one ``#`` title, ``##``/``###`` headings (kept as written, unnumbered),
paragraphs, bullet and numbered lists, pipe tables and inline code. Writes ``reports/latex/<name>.tex``
and builds ``reports/<name>.pdf`` with latexmk from TinyTeX (in ~/.local/bin).

    python3 scripts/report_pdf.py reports/harm_twin_what_did_not_work.md
"""
import os
import re
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_latex import PREAMBLE, inline, table  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
LIST = re.compile(r"^(\s*)([-*]|\d+\.)\s+(.*)$")


def convert(src):
    lines = src.splitlines()
    out, para, title, i = [], [], "", 0

    def flush():
        if para:
            out.append(inline(" ".join(para)) + "\n")
            para.clear()

    while i < len(lines):
        line = lines[i]
        m = re.match(r"^(#{1,3})\s+(.*)$", line)
        if m:
            flush()
            if len(m.group(1)) == 1:
                title = m.group(2).strip()
            else:
                cmd = "section" if len(m.group(1)) == 2 else "subsection"
                out.append(rf"\{cmd}*{{{inline(m.group(2).strip(), heading=True)}}}")
            i += 1
            continue
        if line.lstrip().startswith("|"):
            flush()
            j = i
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                j += 1
            out.append(table(lines[i:j], None))
            i = j
            continue
        m = LIST.match(line)
        if m:
            flush()
            ordered = m.group(2)[0].isdigit()
            items, j = [], i
            while j < len(lines):
                mm = LIST.match(lines[j])
                if mm and mm.group(2)[0].isdigit() == ordered:
                    items.append(mm.group(3))
                elif lines[j].startswith("  ") and items and lines[j].strip():
                    items[-1] += " " + lines[j].strip()
                else:
                    break
                j += 1
            env = "enumerate" if ordered else "itemize"
            out.append(rf"\begin{{{env}}}" + "\n" + "\n".join(r"\item " + inline(x) for x in items)
                       + "\n" + rf"\end{{{env}}}")
            i = j
            continue
        if line.strip():
            para.append(line.strip())
        else:
            flush()
        i += 1
    flush()
    head = (PREAMBLE + r"\setcounter{secnumdepth}{0}" + "\n" + rf"\title{{{inline(title)}}}" + "\n\\author{}\n\\date{}\n"
            + "\\begin{document}\n\\maketitle\n")
    return head + "\n\n".join(out) + "\n\\end{document}\n"


def main():
    src = os.path.abspath(sys.argv[1])
    name = os.path.splitext(os.path.basename(src))[0]
    out_dir = os.path.join(ROOT, "reports", "latex")
    tex_path = os.path.join(out_dir, name + ".tex")
    os.makedirs(out_dir, exist_ok=True)
    open(tex_path, "w").write(convert(open(src).read()))
    env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + os.pathsep + os.environ["PATH"])
    r = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", name + ".tex"],
                       cwd=out_dir, env=env, capture_output=True, text=True)
    log_path = os.path.join(out_dir, name + ".log")
    log = open(log_path, errors="replace").read() if os.path.exists(log_path) else r.stdout
    if r.returncode:
        print("\n".join(l for l in log.splitlines() if l.startswith("!") or l.startswith("l."))[:3000])
        sys.exit(1)
    subprocess.run(["latexmk", "-c", name + ".tex"], cwd=out_dir, env=env, capture_output=True)
    pdf = os.path.join(ROOT, "reports", name + ".pdf")
    shutil.move(os.path.join(out_dir, name + ".pdf"), pdf)
    pages = re.search(r"Output written on .*?\((\d+) pages", log)
    over = re.findall(r"^Overfull \\hbox \(([\d.]+)pt", log, re.M)
    print(f"{pdf}: {pages.group(1) if pages else '?'} pages, {len(over)} overfull lines"
          + (f" (largest {max(map(float, over)):.0f}pt)" if over else ""))


if __name__ == "__main__":
    main()
