"""Test evaluation: constrained Seldonian models vs unconstrained baseline.

Data: make_synthetic with different base rates per group (tp_a=0.4, tp_b=0.8).
Constraint: TPR difference between sensitive groups <= 0.2, delta = 0.05.
Held-out test set is separate from both the candidate and safety sets.
"""
import io
import contextlib
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split

from seldonian.objectives import ghat_tpr_diff, tpr_rate
from seldonian.seldonian import (
    LogisticRegressionSeldonianModel,
    SeldonianAlgorithmLogRegCMAES,
)
from seldonian.synthetic import make_synthetic

THRESHOLD = 0.2
DELTA = 0.05

np.random.seed(1)
X, y, A_idx = make_synthetic(12000, 5, tp_a=0.4, tp_b=0.8, seed=1)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=1)
ghats = [{"fn": ghat_tpr_diff(A_idx, threshold=THRESHOLD), "delta": DELTA}]


def tpr_gap(X, y_true, y_pred):
    a = tpr_rate(A_idx, 1)(X, y_true, y_pred).mean()
    b = tpr_rate(A_idx, 0)(X, y_true, y_pred).mean()
    return a, b, abs(a - b)


def report(name, model_predict, safety=None, seconds=None, solution=None):
    preds = model_predict(X_te)
    tpr_a, tpr_b, gap = tpr_gap(X_te, y_te, preds)
    g_test = ghat_tpr_diff(A_idx, threshold=THRESHOLD)(
        X_te, y_te, preds, delta=DELTA, ub=False)
    print(f"{name}")
    print(f"  accuracy={accuracy_score(y_te, preds):.3f}  "
          f"balanced_accuracy={balanced_accuracy_score(y_te, preds):.3f}")
    print(f"  TPR[A=1]={tpr_a:.3f}  TPR[A=0]={tpr_b:.3f}  gap={gap:.3f}  "
          f"g(theta) on test={g_test:+.3f} ({'violates' if g_test > 0 else 'satisfies'} <= {THRESHOLD})")
    if solution is not None:
        print(f"  candidate selection: {'solution found' if solution else 'No Solution Found'}")
    if safety is not None:
        print(f"  internal safety test: {'PASS' if safety else 'FAIL'}")
    if seconds is not None:
        print(f"  train time: {seconds:.1f}s")
    print()


# --- unconstrained baseline ---
t0 = time.time()
base = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
report("Unconstrained sklearn LogisticRegression", base.predict, seconds=time.time() - t0)

# --- Seldonian scipy (Powell) ---
t0 = time.time()
m1 = LogisticRegressionSeldonianModel(X_tr, y_tr, g_hats=ghats, verbose=False)
with contextlib.redirect_stdout(io.StringIO()):
    r1 = m1.fit(opt="Powell")
report("Seldonian LogisticRegression (scipy Powell)", m1.predict,
       safety=m1.safetyTest(), seconds=time.time() - t0, solution=r1 is not None)

# --- Seldonian CMA-ES (native) ---
t0 = time.time()
m2 = SeldonianAlgorithmLogRegCMAES(X_tr, y_tr, g_hats=ghats, random_seed=1,
                                   maxiter=3000, optimizer="native")
with contextlib.redirect_stdout(io.StringIO()):
    m2.fit()
report("Seldonian LogisticRegression (CMA-ES native)", m2.predict,
       safety=m2.safetyTest(), seconds=time.time() - t0)

# --- Seldonian CMA-ES (pycma) ---
t0 = time.time()
m3 = SeldonianAlgorithmLogRegCMAES(X_tr, y_tr, g_hats=ghats, random_seed=1,
                                   maxiter=3000, optimizer="pycma")
with contextlib.redirect_stdout(io.StringIO()):
    m3.fit()
report("Seldonian LogisticRegression (CMA-ES pycma)", m3.predict,
       safety=m3.safetyTest(), seconds=time.time() - t0)
