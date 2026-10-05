"""Check every entry of a .bib file against arXiv and CrossRef (paper plan P13).

For each entry: the record is fetched by its arXiv id or DOI when the entry has one, otherwise
found by title on CrossRef and arXiv. Title, author surnames in order, and year are compared with
what the entry says. Nothing is rewritten; the output is a report to act on by hand.

    python3 scripts/bib_check.py reports/paper_certification.bib [-o report.md]
"""
import argparse
import json
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

UA = {"User-Agent": "bib-check/0.1 (mailto:noreply@example.org)"}
ATOM = "{http://www.w3.org/2005/Atom}"


def get(url, tries=3):
    for k in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40).read()
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(3 * (k + 1))
    return ("ERROR " + repr(err)).encode()


def norm(s):
    s = re.sub(r"\{?\\[`'\"^~cv]\s*\{?(\w)\}?\}?", r"\1", s)            # LaTeX accents: \`e, \`{e}, {\`e}
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = s.replace("'", "")                                              # Don't and Don’t compare equal
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def entries(path):
    text = open(path).read()
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\}\s*(?=\n@|\n*\Z)", text, re.S):
        body = m.group(3)
        f = {k.lower(): v for k, v in re.findall(r"(\w+)\s*=\s*\{((?:[^{}]|\{[^{}]*\})*)\}", body)}
        blob = " ".join(f.values())
        ax = re.search(r"arXiv:(\d{4}\.\d{4,5})", blob)
        authors = [a.strip() for a in re.split(r"\s+and\s+", f.get("author", ""))]
        sur = [norm(a.split(",")[0] if "," in a else a.split()[-1]) for a in authors if a]
        yield dict(key=m.group(2), title=f.get("title", ""), surnames=sur, year=f.get("year", ""), doi=f.get("doi"),
                   arxiv=ax.group(1) if ax else None, venue=f.get("journal") or f.get("booktitle") or "")


def from_arxiv(xml):
    out = []
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return out
    for e in root.findall(ATOM + "entry"):
        names = [a.find(ATOM + "name").text for a in e.findall(ATOM + "author")]
        out.append(dict(title=" ".join(e.find(ATOM + "title").text.split()), surnames=[norm(n.split()[-1]) for n in names],
                        year=e.find(ATOM + "published").text[:4], where="arXiv " + e.find(ATOM + "id").text.rsplit("/", 1)[-1]))
    return out


def from_crossref(item):
    yr = (item.get("published") or item.get("issued") or {}).get("date-parts", [[None]])[0][0]
    return dict(title=" ".join((item.get("title") or [""])[0].split()), surnames=[norm(a.get("family", "")) for a in item.get("author", [])],
                year=str(yr), where=f"CrossRef {item.get('DOI')} ({(item.get('container-title') or [''])[0]})")


def lookup(e):
    found = []
    if e["arxiv"]:
        found += from_arxiv(get("http://export.arxiv.org/api/query?id_list=" + e["arxiv"]))
        time.sleep(3)
    if e["doi"]:
        raw = get("https://api.crossref.org/works/" + urllib.parse.quote(e["doi"]))
        if not raw.startswith(b"ERROR"):
            found.append(from_crossref(json.loads(raw)["message"]))
    if not found:
        raw = get("https://api.crossref.org/works?rows=3&query.bibliographic=" + urllib.parse.quote(e["title"] + " " + " ".join(e["surnames"][:2])))
        if not raw.startswith(b"ERROR"):
            found += [from_crossref(i) for i in json.loads(raw)["message"]["items"]]
        found += from_arxiv(get("http://export.arxiv.org/api/query?max_results=3&search_query=" + urllib.parse.quote('ti:"' + norm(e["title"]) + '"')))
        time.sleep(3)
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bib")
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()
    L = ["| key | found at | title | authors | year | note |", "|---|---|---|---|---|---|"]
    bad = 0
    for e in entries(a.bib):
        hits = [h for h in lookup(e) if norm(h["title"]) == norm(e["title"])]
        if not hits:
            bad += 1
            L.append(f"| {e['key']} | **not found by title** | | | | check by hand |")
            continue
        notes, where = [], []
        for h in hits:
            where.append(h["where"])
            if h["surnames"] and h["surnames"] != e["surnames"]:
                notes.append(f"{h['where'].split()[0]} authors: {', '.join(h['surnames'])}")
            if h["year"] != e["year"]:
                notes.append(f"{h['where'].split()[0]} year {h['year']}")
        ok_auth = any(h["surnames"] == e["surnames"] for h in hits)
        bad += not ok_auth
        L.append(f"| {e['key']} | {'; '.join(where)} | match | {'match (' + str(len(e['surnames'])) + ')' if ok_auth else '**differs**'} | {e['year']} | {'; '.join(notes)} |")
        print(L[-1], flush=True)
    L.append("")
    L.append(f"{bad} entries need a look." if bad else "Every entry's title and author list match a fetched record.")
    if a.out:
        open(a.out, "w").write("\n".join(L) + "\n")
    print(L[-1])
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
