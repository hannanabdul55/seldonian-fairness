"""Where StratPPI's heuristic allocation puts the labels in the cells with the largest misses (paper Appendix A; plan step R4).

    .venv/bin/python scripts/stratppi_heur_cell.py > results/paper/stratppi_heur_cell.md
"""
import sys

import numpy as np
sys.dont_write_bytecode = True
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stratppi_validate as V
PL = V.PL
pools = PL.load_pools()
y_pool, p_pool = pools[("refusal", "rubric", 0)]
for (n, big_n, rate, seed, K) in ((1000, 4000, 0.05, 7005, 5), (1000, 4000, 0.013, 7007, 10), (225, 4000, 0.05, 7004, 5)):
    rng = np.random.default_rng(seed)
    x_pool = PL.logit01(p_pool); M = len(y_pool)
    q = np.where(y_pool == 1, rate / (y_pool == 1).sum(), (1 - rate) / (y_pool == 0).sum())
    order = np.lexsort((rng.random(M), x_pool))
    cum = np.cumsum(q[order]) - q[order] / 2
    st = np.empty(M, dtype=int); st[order] = np.minimum((cum * K).astype(int), K - 1)
    W = np.array([q[st == h].sum() for h in range(K)])
    sh, ph, mp = [], [], []
    for h in range(K):
        m = np.flatnonzero(st == h); w = q[m] / q[m].sum(); pp = p_pool[m]
        sh.append(np.sqrt(w @ (pp * (1 - pp)) + w @ pp ** 2 - (w @ pp) ** 2)); ph.append(w @ y_pool[m]); mp.append(w @ pp)
    heur = V.allocate(W * np.array(sh), n); prop = np.maximum(2, np.rint(n * W).astype(int))
    ph = np.array(ph)
    print(f"n {n} rate {rate} K {K}")
    print(" W        ", np.round(W, 3)); print(" true rate", np.round(ph, 4)); print(" share of positives", np.round(W * ph / (W * ph).sum(), 3))
    print(" mean P(Yes)", np.round(mp, 4)); print(" heur sd  ", np.round(sh, 4)); print(" prop n_k ", prop); print(" heur n_k ", heur)
    starved = heur <= 2
    print(" P(no positive among the labels of strata at the floor) =", round(float(np.prod((1 - ph[starved]) ** heur[starved])), 3),
          "; share of all positives in those strata", round(float((W * ph)[starved].sum() / (W * ph).sum()), 3))
