"""Spike 017's four large-unlabelled-set cells at 4,000 draws (plan step R3).

``plasmode017.py`` caps the cells with 20,000 unlabelled responses at 1,000 draws, the only cells
of the paper's Table 4 under 4,000. A cell draws in batches of 250 from one generator, so a
4,000-draw run starts with the stored 1,000 draws and adds 3,000. This script first reruns the
four cells at 1,000 draws and asserts every stored row is reproduced, then runs them at 4,000 and
writes those rows to ``results/paper/plasmode017_big.json``. The spike's own file is left as it
is; ``scripts/validity_recount.py`` reads the 4,000-draw rows in place of the capped ones. The
block-PPI rows stay at their 500 draws.

    OMP_NUM_THREADS=1 .venv/bin/python scripts/plasmode017_big.py      # about 6 minutes on 4 cores
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
SPIKE = os.path.join(HERE, "..", ".planning", "spikes", "017-calibration-carrying-certificate")
sys.path.insert(0, SPIKE)
import plasmode017 as P  # noqa: E402

PATH = os.path.join(SPIKE, "plasmode.json")
OUT = os.path.join(HERE, "..", "results", "paper", "plasmode017_big.json")
BIG_N, REPS = 20000, 4000


def jobs_at(reps, pools):
    """The large-set jobs with the seeds ``plasmode017.main`` gives them."""
    seed, out = 0, []
    for key in sorted(pools):
        if key[0] == "harm":
            continue
        seed += len((50, 100, 225, 500) if key[2] == 0 else (225,))
    for n in (225, 1000):
        for key in (("refusal", "rubric", 0), ("refusal", "raw", 0)):
            seed += 1
            out.append((key, *pools[key], n, BIG_N, reps, None, seed, 500))
    return out


def ident(r):
    return (r["task"], r["variant"], r["wording"], r["n"], r["N"], r["method"], r["feat"])


def main():
    pools = P.load_pools()
    d = json.load(open(PATH))
    stored = {ident(r): r for r in d["rows"] if r["N"] == BIG_N}
    with ProcessPoolExecutor(4) as ex:
        again = [r for rows in ex.map(P.cell, jobs_at(1000, pools)) for r in rows]
        assert len(again) == len(stored), (len(again), len(stored))
        for r in again:
            s = stored[ident(r)]
            assert s["reps"] == r["reps"] and abs(s["miss"] - r["miss"]) < 1e-12 and abs(s["bound"] - r["bound"]) < 1e-9, (s, r)
        print(f"the {len(again)} stored rows at 1,000 draws are reproduced", flush=True)
        new = {ident(r): r for rows in ex.map(P.cell, jobs_at(REPS, pools)) for r in rows}
    assert set(new) == set(stored)
    note = ("the four cells of spike 017's plasmode with 20,000 unlabelled responses, at 4,000 draws; the first 1,000 are "
            "the spike's stored draws (scripts/plasmode017_big.py)")
    json.dump(dict(delta=d["delta"], reps=REPS, note=note, rows=[new[ident(r)] for r in d["rows"] if r["N"] == BIG_N]),
              open(OUT, "w"))
    for k in sorted(new):
        if k[5] in ("classical", "boot", "ppi++", "naive"):
            print(k, "reps", new[k]["reps"], "miss", round(stored[k]["miss"], 4), "->", round(new[k]["miss"], 4))


if __name__ == "__main__":
    main()
