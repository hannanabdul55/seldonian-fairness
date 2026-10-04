"""Browser test of the Refusal Label Desk before it goes to annotators.

Runs the built page in headless Chrome against a mock of the artifact runtime whose stored
snapshots are deep-frozen, as the real runtime's are (``db.d.ts``: "delivered snapshots and their
data() are frozen"). The first version of the page reused the stored ``labels`` object on a return
visit, every later label was silently dropped, and 220 labels were lost on 2026-10-04; that path
had never been run. Each scenario presses a label key N times and checks what reached the store.

    .venv/bin/python scripts/refusal_sheet_build.py --page /tmp/desk.html
    python3 scripts/refusal_desk_test.py /tmp/desk.html        # exits 1 if any scenario fails

Needs Chrome (the Windows one under WSL by default; set CHROME to another binary).
"""
import html as H
import json
import os
import re
import subprocess
import sys
import tempfile

CHROME = os.environ.get("CHROME", "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe")
MOCK = """<script>
(function(){
  window.__errors = [];
  window.addEventListener('error', e => window.__errors.push(String(e.message)));
  window.addEventListener('unhandledrejection', e => window.__errors.push('rejection: ' + String(e.reason && (e.reason.message || e.reason.code) || e.reason)));
  const deepFreeze = o => { if (o && typeof o === 'object') { Object.values(o).forEach(deepFreeze); Object.freeze(o); } return o; };
  const store = { doc: __REMOTE__ }; const writes = []; const FAIL = __FAIL__;
  localStorage.clear();
  const local = __LOCAL__;
  if (local) localStorage.setItem('__LS_KEY__', JSON.stringify(local));
  const docRef = {
    get: async () => ({ exists: !!store.doc, data: () => store.doc ? deepFreeze(JSON.parse(JSON.stringify(store.doc))) : undefined }),
    set: async d => { if (FAIL) throw { code: 'unavailable', message: 'mock failure' }; store.doc = JSON.parse(JSON.stringify(d)); writes.push(1); },
  };
  window.__store = store; window.__writes = writes;
  window.claude = { use: async name => name === 'db' ? { doc: () => docRef } : name === 'user' ? { id: async () => 'u_test' } : null };
})();
</script>"""
DRIVER = """<script>
setTimeout(async () => {
  const $ = id => document.getElementById(id);
  const setupShown = !$('setup').hidden;
  if (setupShown) { $('handle').value = 'tt'; $('slot').value = 1; $('kk').value = 2; $('start').click(); }
  for (let i = 0; i < __PRESSES__; i++) { document.dispatchEvent(new KeyboardEvent('keydown', { key: 'a', bubbles: true })); await new Promise(r => setTimeout(r, 30)); }
  await new Promise(r => setTimeout(r, 2500));
  const doc = window.__store.doc || {};
  const out = { setupShown, status: $('status').textContent, store: Object.keys(doc.labels || {}).length, sheet: doc.sheet,
                aside: Object.keys(doc.labels_v1 || {}).length, browser: Object.keys((JSON.parse(localStorage.getItem('__LS_KEY__')) || {}).labels || {}).length,
                errors: window.__errors, done: $('donetext').textContent, alert: ($('alert') && !$('alert').hidden) ? $('alert').textContent : '' };
  const pre = document.createElement('pre'); pre.id = 'result'; pre.textContent = JSON.stringify(out); document.body.appendChild(pre);
}, 1200);
</script>"""


def shuffled(arr, seed):
    """The page's seeded shuffle (a 32-bit linear congruential generator), step for step."""
    a, s = list(arr), seed & 0xFFFFFFFF
    for i in range(len(a) - 1, 0, -1):
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        j = int(s / 4294967296 * (i + 1))
        a[i], a[j] = a[j], a[i]
    return a


def lab(ids, label, t):
    return {i: dict(label=label, note="", t=t) for i in ids}


def scenarios(order):
    """name -> (stored document, browser copy, store fails, key presses, check on the result)."""
    doc = lambda n, **kw: dict(handle="tt", slot=1, K=2, pos=n, labels=lab(order[:n], "a", "2026-10-05T00:00:00Z"), updated="2026-10-05T00:00:01Z", **kw)
    return {
        "first visit": (None, None, False, 10, lambda r: r["store"] == 10 and r.get("sheet") == 2 and r["browser"] == 10),
        "return visit, labels in the store": (doc(5, sheet=2), None, False, 10, lambda r: r["store"] == 15 and not r["setupShown"]),
        "document from an earlier sheet": (doc(28), None, False, 10, lambda r: r["store"] == 10 and r["aside"] == 28 and r["setupShown"]),
        "this browser ahead of the store": (doc(5, sheet=2), dict(handle="tt", slot=1, K=2, pos=8, sheet=2, labels=lab(order[:8], "r", "2026-10-05T01:00:00Z")),
                                            False, 10, lambda r: r["store"] == 18),
        "the store rejects writes": (None, None, True, 10, lambda r: r["store"] == 0 and r["browser"] == 10 and "store did not accept" in r["alert"]),
        "a full pass": (None, None, False, len(order), lambda r: r["store"] == len(order) and f"store holds {len(order)} of them" in r["done"]),
    }


def main():
    page = open(sys.argv[1]).read()
    sheet = json.loads(re.search(r'<script id="sheet" type="application/json">(.*?)</script>', page, re.S).group(1).replace("<\\/", "</"))
    ls_key = re.search(r"const LS_KEY = '([^']+)'", page).group(1)
    # slot 1 of 2 in the order the page presents it, so stored labels can sit at the head of the list
    order = [x["id"] for x in shuffled([x for x in sheet if x["s"]], 40) + [x for x in shuffled([x for x in sheet if not x["s"]], 41) if x["d"] % 2 == 0]]
    bad = 0
    with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.abspath(sys.argv[1]))) as tmp:
        for name, (remote, local, fail, presses, ok) in scenarios(order).items():
            mock = MOCK.replace("__REMOTE__", json.dumps(remote)).replace("__LOCAL__", json.dumps(local)).replace("__FAIL__", "true" if fail else "false")
            path = os.path.join(tmp, "t.html")
            open(path, "w").write("<!doctype html><meta charset=utf8><body>" + (mock + page + DRIVER.replace("__PRESSES__", str(presses))).replace("__LS_KEY__", ls_key))
            url = "file://" + path
            if CHROME.startswith("/mnt/"):       # Windows Chrome reads the WSL file through its UNC path
                url = "file:///" + subprocess.run(["wslpath", "-w", path], capture_output=True, text=True).stdout.strip().replace("\\", "/")
            r = subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--virtual-time-budget=20000", "--dump-dom", url],
                               capture_output=True, text=True, timeout=180, cwd="/mnt/c" if CHROME.startswith("/mnt/") else None)
            m = re.search(r'<pre id="result">(.*?)</pre>', r.stdout, re.S)
            res = json.loads(H.unescape(m.group(1))) if m else None
            good = bool(res) and ok(res) and not res["errors"]
            bad += not good
            print(("ok  " if good else "FAIL"), name, "->", res if res else "no result: " + r.stderr[-300:])
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
