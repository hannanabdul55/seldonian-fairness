"""Spike 002 follow-up: do late spikes add information beyond the multiplier and margin?

Cross-validated AUC (logistic regression, 10-fold, repeated) for future breach from
the step-<=200 run state, with and without the TD-spike features.
"""
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

rows = json.load(open("results.json"))
SETS = {
    "state (lambda at 200, margin at 200)": ["lam_at_T", "margin_at_T"],
    "state + late policy step": ["lam_at_T", "margin_at_T", "late_dtheta"],
    "state + late TD spikes (count, share, mean abs delta)": ["lam_at_T", "margin_at_T", "late_spikes", "late_share", "late_abs_mean"],
    "state + late TD spikes in the learnable part": ["lam_at_T", "margin_at_T", "late_spikes_adv", "late_abs_adv"],
    "state + late valence (mean, negative part)": ["lam_at_T", "margin_at_T", "late_valence", "late_neg"],
    "TD spikes alone": ["late_spikes", "late_share", "late_abs_mean"],
}
out = ["", "## Incremental value over the run state (5x10-fold CV logistic regression, AUC for future breach)", "",
       "| features | lag p1 | lag p4 | pooled lag p1 + p4 |", "|---|---|---|---|"]
def cvauc(rs, feats, reps=5):
    X = np.array([[np.nan_to_num(r[f]) for f in feats] for r in rs])
    y = np.array([r["future_breach"] for r in rs], int)
    aucs = []
    for rep in range(reps):
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=rep)
        s = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=1.0)), X, y,
                              cv=list(cv.split(X, y)), method="predict_proba")[:, 1]
        aucs.append(roc_auc_score(y, s))
    return np.mean(aucs)
groups = {"lag p1": [r for r in rows if r["setting"] == "lag p1"],
          "lag p4": [r for r in rows if r["setting"] == "lag p4"]}
groups["pooled"] = groups["lag p1"] + groups["lag p4"]
for name, feats in SETS.items():
    out.append(f"| {name} | " + " | ".join(f"{cvauc(g, feats):.2f}" for g in groups.values()) + " |")
txt = "\n".join(out) + "\n"
open("results.md", "a").write(txt)
print(txt)

# E3 of LITERATURE.md: the LLM runs' statistic, Spearman(late share, fraction of feasible
# checkpoints) was -0.25 to -0.36. Does it survive controlling for the multiplier's movement?
from scipy.stats import spearmanr, rankdata
def partial_spearman(x, y, zs):
    rx, ry = rankdata(x), rankdata(y)
    Z = np.column_stack([np.ones(len(x))] + [rankdata(z) for z in zs])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])
out = ["", "## E3: Spearman(late share of spikes, fraction of feasible predicted tests in steps 1-200)", "",
       "| setting | raw | partial, controlling multiplier moves and total abs change | partial, also lambda at 200 |",
       "|---|---|---|---|"]
for name, g in groups.items():
    g = [r for r in g if np.isfinite(r["late_share"]) and np.isfinite(r["feasible_frac_T"])]
    x = [r["late_share"] for r in g]; y = [r["feasible_frac_T"] for r in g]
    z1 = [[r["lam_moves_T"] for r in g], [r["lam_abs_change_T"] for r in g]]
    z2 = z1 + [[r["lam_at_T"] for r in g]]
    out.append(f"| {name} | {spearmanr(x, y)[0]:+.2f} | {partial_spearman(x, y, z1):+.2f} | {partial_spearman(x, y, z2):+.2f} |")
txt = "\n".join(out) + "\n"
open("results.md", "a").write(txt)
print(txt)
