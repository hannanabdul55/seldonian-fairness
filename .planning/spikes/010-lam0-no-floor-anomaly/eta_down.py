"""Spike 010 check: does a slow downward dual step remove the lam0 = 5 anomaly?

    ../../../.venv/bin/python eta_down.py
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "004-forbidden-capability"))
import forbidlab as F  # noqa: E402

CELLS = {"lam0 0, eta_down 100": dict(lam0=0.0), "lam0 5, eta_down 100": dict(lam0=5.0),
         "lam0 5, eta_down 10": dict(lam0=5.0, eta_down=10.0),
         "lam0 5, eta_down 1": dict(lam0=5.0, eta_down=1.0),
         "lam0 0, eta_down 1": dict(lam0=0.0, eta_down=1.0)}


def one(job):
    name, seed = job
    kw = dict(CELLS[name])
    eta_down = kw.pop("eta_down", None)
    orig = F.LagrangianReward.__init__

    def init(self, *a, **k):
        orig(self, *a, **k)
        if eta_down is not None:
            self.eta_down = eta_down
    F.LagrangianReward.__init__ = init
    try:
        r = F.run(seed, method="lag", tau=0.12, predict_every=25, **kw)
    finally:
        F.LagrangianReward.__init__ = orig
    return name, r["entered_U"], r["final_task_acc"], r["solution"], r["violates"]


if __name__ == "__main__":
    js = [(n, s) for n in CELLS for s in range(200)]
    with ProcessPoolExecutor(4) as ex:
        res = list(ex.map(one, js, chunksize=8))
    print("| cell | entered U | task acc | solution | final violates |\n|---|---|---|---|---|")
    for n in CELLS:
        rs = [r for r in res if r[0] == n]
        print(f"| {n} | {np.mean([r[1] for r in rs]):.3f} | {np.mean([r[2] for r in rs]):.3f} | "
              f"{np.mean([bool(r[3]) for r in rs]):.2f} | {np.mean([r[4] for r in rs]):.3f} |")
