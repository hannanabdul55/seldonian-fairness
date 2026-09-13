"""Render a report's Markdown as a self-contained HTML page (the artifact format).

    uv run scripts/md2html.py reports/paper_seldonian_llm.md out.html "Tab title" "Summary box text" "eyebrow"

Handles headings (with ids), paragraphs, bullet and numbered lists, pipe tables
(numeric cells right-aligned), fenced code, inline code / bold / italic. The
first H1 becomes the page header; the summary is shown in a box under it.
"""
import html
import re
import sys

CSS = """
:root{--paper:#f6f7f4;--panel:#eceee9;--ink:#1d2530;--muted:#5d6874;--rule:#cfd4cc;--accent:#1f6f78;--accent-ink:#155158;--code-bg:#e7eae5}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--paper:#131a20;--panel:#1b242c;--ink:#e3e7ea;--muted:#9aa5ae;--rule:#324049;--accent:#62b6bd;--accent-ink:#8fd0d5;--code-bg:#1f2a32}}
:root[data-theme="dark"]{--paper:#131a20;--panel:#1b242c;--ink:#e3e7ea;--muted:#9aa5ae;--rule:#324049;--accent:#62b6bd;--accent-ink:#8fd0d5;--code-bg:#1f2a32}
html{color-scheme:light dark}
body{margin:0;background:var(--paper);color:var(--ink);font-family:"Source Sans 3","Segoe UI",system-ui,sans-serif;font-size:17px;line-height:1.55}
main{max-width:76ch;margin:0 auto;padding:2.5rem 1.25rem 5rem}
header{border-bottom:2px solid var(--accent);padding-bottom:1rem;margin-bottom:1.5rem}
h1,h2,h3{font-family:"Source Serif 4",Georgia,serif;font-weight:600;line-height:1.2;text-wrap:balance;color:var(--ink)}
h1{font-size:2rem;margin:0 0 .5rem}
h2{font-size:1.45rem;margin:2.6rem 0 .8rem;padding-top:.4rem;border-top:1px solid var(--rule)}
h3{font-size:1.15rem;margin:2rem 0 .6rem;color:var(--accent-ink)}
p{margin:0 0 1rem}
.eyebrow{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.78rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:0 0 .6rem}
code{font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace;font-size:.86em;background:var(--code-bg);padding:.05em .3em;border-radius:3px}
pre{background:var(--code-bg);padding:.8rem 1rem;overflow-x:auto;border-radius:4px}
pre code{background:none;padding:0}
ul,ol{padding-left:1.4rem;margin:0 0 1rem}
li{margin:.3rem 0}
li::marker{color:var(--accent)}
.tablewrap{overflow-x:auto;margin:0 0 1.2rem;border:1px solid var(--rule);border-radius:4px}
table{border-collapse:collapse;width:100%;font-size:.9rem;font-variant-numeric:tabular-nums}
th,td{padding:.45rem .6rem;border-bottom:1px solid var(--rule);text-align:left;vertical-align:top}
th{background:var(--panel);font-weight:600;font-size:.8rem;letter-spacing:.03em;text-transform:uppercase;color:var(--muted)}
td.num,th.num{text-align:right;font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.84rem;white-space:nowrap}
tbody tr:last-child td{border-bottom:none}
strong{font-weight:600}
td strong{color:var(--accent-ink)}
a{color:var(--accent)}
.summary{background:var(--panel);border-left:4px solid var(--accent);padding:1rem 1.2rem;margin:0 0 1.5rem;border-radius:0 4px 4px 0}
.summary p{margin:0}
@media (max-width:640px){body{font-size:16px} h1{font-size:1.6rem}}
"""
FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">\n'
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,500;8..60,600'
         '&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap">')


def inline(t):
    t = html.escape(t, quote=False)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<em>\1</em>", t)
    t = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<a href="\2">\1</a>', t)
    return t


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40]


def is_num(c):
    c = re.sub(r"[*`]", "", c).strip()
    return bool(re.match(r"^[-+]?\d", c)) or c in ("", "-")


def table(rows):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    cells = [r for r in cells if not all(re.fullmatch(r":?-{2,}:?", c) for c in r)]
    head, body = cells[0], cells[1:]
    ncol = len(head)
    numeric = [all(is_num(r[i]) for r in body if i < len(r)) and any(r[i].strip() for r in body if i < len(r))
               for i in range(ncol)]
    out = ["<div class='tablewrap'><table><thead><tr>"]
    out += [f"<th class='{'num' if numeric[i] else ''}'>{inline(c)}</th>" for i, c in enumerate(head)]
    out.append("</tr></thead><tbody>")
    for r in body:
        out.append("<tr>" + "".join(
            f"<td class='{'num' if i < ncol and numeric[i] else ''}'>{inline(c)}</td>"
            for i, c in enumerate(r)) + "</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def convert(src):
    lines = src.splitlines()
    out, para, i, title = [], [], 0, None

    def flush():
        nonlocal para
        if para:
            out.append("<p>" + inline(" ".join(para)) + "</p>")
            para = []

    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            flush()
            j = i + 1
            buf = []
            while j < len(lines) and not lines[j].startswith("```"):
                buf.append(lines[j])
                j += 1
            out.append("<pre><code>" + html.escape("\n".join(buf)) + "</code></pre>")
            i = j + 1
            continue
        m = re.match(r"^(#{1,3})\s+(.*)$", line)
        if m:
            flush()
            level, text = len(m.group(1)), m.group(2).strip()
            if level == 1 and title is None:
                title = text
                out.append(f"<header><h1>{inline(text)}</h1></header>")
                out.append("@@SUMMARY@@")
            else:
                out.append(f"<h{level} id='{slug(text)}'>{inline(text)}</h{level}>")
            i += 1
            continue
        if line.lstrip().startswith("|"):
            flush()
            j = i
            rows = []
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                rows.append(lines[j])
                j += 1
            out.append(table(rows))
            i = j
            continue
        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", line)
        if m:
            flush()
            ordered = m.group(2)[0].isdigit()
            items = []
            j = i
            while j < len(lines):
                mm = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", lines[j])
                if mm and (mm.group(2)[0].isdigit()) == ordered:
                    items.append(mm.group(3))
                    j += 1
                elif lines[j].startswith("  ") and items and lines[j].strip():
                    items[-1] += " " + lines[j].strip()
                    j += 1
                else:
                    break
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{inline(x)}</li>" for x in items) + f"</{tag}>")
            i = j
            continue
        if not line.strip():
            flush()
        else:
            para.append(line.strip())
        i += 1
    flush()
    return out, title


def main():
    src = sys.argv[1]
    dst = sys.argv[2]
    tab = sys.argv[3] if len(sys.argv) > 3 else "Report"
    summary = sys.argv[4] if len(sys.argv) > 4 else ""
    eyebrow = sys.argv[5] if len(sys.argv) > 5 else "seldonian-fairness &middot; branch llm-seldonian-rl"
    body, title = convert(open(src).read())
    box = f'<div class="summary"><p>{inline(summary)}</p></div>' if summary else ""
    page = [f"<title>{html.escape(tab)}</title>", FONTS, "<style>" + CSS + "</style>", "<main>",
            f'<p class="eyebrow">{eyebrow}</p>']
    for chunk in body:
        page.append(box if chunk == "@@SUMMARY@@" else chunk)
    page.append("</main>")
    with open(dst, "w") as f:
        f.write("\n".join(page))
    print(f"{dst}: {sum(len(p) for p in page)} bytes, title {title!r}")


if __name__ == "__main__":
    main()
