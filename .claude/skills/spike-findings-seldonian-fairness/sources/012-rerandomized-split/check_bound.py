"""Check splitlab's vectorised predicted bound against seldonian.objectives.ghat_tpr_diff.

    ../../../.venv/bin/python check_bound.py
"""
import numpy as np
import splitlab as L
from seldonian.objectives import ghat_tpr_diff

for bound in ("ttest", "clopper_pearson"):
  L.BOUND = bound
  g = ghat_tpr_diff(L.A_IDX, method=bound, threshold=L.TAU)
  worst = 0
  for seed in range(20):
    X, y, _ = L.make_synthetic(1000, L.D, A_idx=L.A_IDX, seed=seed)
    m, _ = L.split_random(X, y, np.random.default_rng(seed))
    Xc, yc, n_s = X[~m], y[~m], int(m.sum())
    w, b, *_ = L.candidate(Xc, yc, n_s, 2.0)
    s = L.logit(Xc, w, b)
    pa, pb, npa, npb = L.group_rates(s, yc, Xc[:, L.A_IDX], L.GRID)
    na, nb = max(2, int(n_s * npa / len(yc))), max(2, int(n_s * npb / len(yc)))
    la, ha = L.rate_interval(pa, npa, na, L.DELTA / 4, 2.0)
    lb, hb = L.rate_interval(pb, npb, nb, L.DELTA / 4, 2.0)
    rng = np.random.default_rng(seed)
    for _ in range(20):
        i, j = rng.integers(0, len(L.GRID), 2)
        ub = max(abs(lb[j] - ha[i]), abs(hb[j] - la[i]))
        ref = g(Xc, yc, L.predict(Xc, w, b, np.array([[L.GRID[j]] * L.BINS, [L.GRID[i]] * L.BINS])), L.DELTA, n_s,
                predict=True, ub=True) + L.TAU
        worst = max(worst, abs(ub - ref))
  print(f"{bound}: max |vectorised - project| predicted UB over 400 candidates: {worst:.2e}")
