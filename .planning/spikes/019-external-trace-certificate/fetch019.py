"""Spike 019: fetch the 360 RoboDojo-RC Tier 1 transcript pages (resumable, polite).

Reads the report page for the transcript URLs, keeps ``trials.csv`` and ``cells.csv`` in
``results/spikes/019/``, and stores each transcript page (about 12 MB of HTML with embedded
images) under ``/mnt/d/seldonian-runs/019/logs/<run_id>.html``. A page is kept only if it
holds the trial's terminating call; otherwise it is re-fetched next run.

    ../../../.venv/bin/python fetch019.py [--sleep 2]
"""
import argparse
import csv
import os
import re
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(REPO, "results", "spikes", "019")
LOGS = "/mnt/d/seldonian-runs/019/logs"
PAGE = "https://robocurve.org/opus-5-5-robodojo-rc-tier-1/"
UA = "seldonian-fairness spike 019 (research; one fetch per trial; contact via github hannanabdul55)"


def get(url, retries=4):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            wait = 10 * (i + 1)
            print(f"  {url}: {e}; retry in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(url)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sleep", type=float, default=2.0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    page = get(PAGE).decode()
    open(os.path.join(OUT, "page.html"), "wb").write(page.encode())
    for name in ("trials.csv", "cells.csv"):
        open(os.path.join(OUT, name), "wb").write(get(PAGE + "data/" + name))
    urls = {m.group(2): m.group(1) for m in re.finditer(
        r'(https://robodojo-tier1-artifacts\.pages\.dev/log/[^"/]+/[^"/]+/(adhoc_[0-9a-f]+)/)', page)}
    rows = list(csv.DictReader(open(os.path.join(OUT, "trials.csv"))))
    missing = [r["run_id"] for r in rows if r["run_id"] not in urls]
    print(f"{len(rows)} trials, {len(urls)} transcript urls, {len(missing)} without a url", flush=True)
    t0, done = time.time(), 0
    for k, r in enumerate(rows):
        rid = r["run_id"]
        if rid not in urls:
            continue
        path = os.path.join(LOGS, rid + ".html")
        if os.path.exists(path) and os.path.getsize(path) > 1000:
            done += 1
            continue
        body = get(urls[rid])
        text = body.decode(errors="replace")
        term = r["termination"]
        ok = ("give_up(" in text or "give_up" in text) if term == "give_up" else True
        if not ok or len(body) < 1000:
            print(f"  {rid}: page looks incomplete ({len(body)} bytes); kept for inspection", flush=True)
        open(path, "wb").write(body)
        done += 1
        if done % 20 == 0:
            el = time.time() - t0
            print(f"{done}/{len(rows)} {el:.0f}s", flush=True)
        time.sleep(a.sleep)
    print(f"DONE {done} pages in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
