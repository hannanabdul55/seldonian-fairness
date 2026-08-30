"""Seldonian fairness on Bias in Bios (De-Arteaga et al. 2019).

Task: classify real professional biographies as surgeon (positive) vs nurse from a
frozen-LLM embedding. The occupations are heavily gender-skewed (surgeons 15%
female, nurses 91% female in the corpus), so an unconstrained classifier uses
gender as a shortcut and its true positive rate differs sharply by gender - the
canonical "gender TPR gap" metric for this dataset. A Seldonian neural-net head is
trained with that gap bounded.

Requires the llm extra: uv sync --extra llm
"""
import argparse
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score

try:
    from datasets import load_dataset
except ImportError as e:
    raise SystemExit("datasets not installed - run: uv sync --extra llm") from e

from seldonian.encoders import embed_texts, normalize_embeddings
from seldonian.objectives import ghat_tpr_diff, ghat_tpr_diff_t, tpr_rate
from seldonian.seldonian import NeuralNetSeldonianGD

POSITIVE_PROF = 25  # surgeon
NEGATIVE_PROF = 13  # nurse
THRESHOLD = 0.1
DELTA = 0.05


def load_pair(split, max_samples, rng):
    ds = load_dataset("LabHC/bias_in_bios", split=split)
    prof = np.array(ds["profession"])
    keep = np.flatnonzero((prof == POSITIVE_PROF) | (prof == NEGATIVE_PROF))
    if max_samples and len(keep) > max_samples:
        keep = rng.choice(keep, size=max_samples, replace=False)
    texts = [ds[int(i)]["hard_text"] for i in keep]
    y = (prof[keep] == POSITIVE_PROF).astype(int)
    gender = np.array(ds["gender"])[keep]  # 1 = female
    return texts, y, gender


def report(name, preds, y, X, A_idx, safety=None, seconds=None, solution=None):
    tpr_f = tpr_rate(A_idx, 1)(X, y, preds).mean()
    tpr_m = tpr_rate(A_idx, 0)(X, y, preds).mean()
    g_test = ghat_tpr_diff(A_idx, threshold=THRESHOLD)(X, y, preds, delta=DELTA,
                                                       ub=False)
    print(f"{name}")
    print(f"  accuracy={accuracy_score(y, preds):.3f}  "
          f"balanced_accuracy={balanced_accuracy_score(y, preds):.3f}")
    print(f"  TPR[female]={tpr_f:.3f}  TPR[male]={tpr_m:.3f}  gap={abs(tpr_f - tpr_m):.3f}  "
          f"g(theta) on test={g_test:+.3f} "
          f"({'violates' if g_test > 0 else 'satisfies'} <= {THRESHOLD})")
    if solution is not None:
        print(f"  candidate selection: "
              f"{'solution found' if solution else 'No Solution Found'}")
    if safety is not None:
        print(f"  internal safety test: {'PASS' if safety else 'FAIL'}")
    if seconds is not None:
        print(f"  time: {seconds:.1f}s")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="prajjwal1/bert-tiny",
                        help="HF model id for the frozen encoder")
    parser.add_argument("--max-samples", type=int, default=20000,
                        help="cap on training-pair samples (0 = all)")
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lambda-lr", type=float, default=8e-2)
    args = parser.parse_args()
    rng = np.random.default_rng(0)

    texts_tr, y_tr, gen_tr = load_pair("train", args.max_samples, rng)
    texts_te, y_te, gen_te = load_pair("test", 0, rng)
    print(f"train: {len(y_tr)} bios ({y_tr.mean():.2f} surgeon), "
          f"surgeon female frac={gen_tr[y_tr == 1].mean():.2f}, "
          f"nurse female frac={gen_tr[y_tr == 0].mean():.2f}")
    print(f"test:  {len(y_te)} bios\n")

    t0 = time.time()
    emb_all = embed_texts(texts_tr + texts_te, args.model, verbose=True)
    emb_all = normalize_embeddings(emb_all)
    print(f"Embedded with {args.model} ({emb_all.shape[1]}-dim) "
          f"in {time.time() - t0:.1f}s\n")

    gender_all = np.concatenate([gen_tr, gen_te])
    X_all = np.hstack([emb_all, gender_all[:, None]]).astype(np.float32)
    A_idx = X_all.shape[1] - 1
    X_tr, X_te = X_all[:len(y_tr)], X_all[len(y_tr):]

    t0 = time.time()
    base = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
    report("Unconstrained LogisticRegression head", base.predict(X_te), y_te, X_te,
           A_idx, seconds=time.time() - t0)

    ghats = [{"fn": ghat_tpr_diff_t(A_idx, threshold=THRESHOLD), "delta": DELTA}]
    t0 = time.time()
    np.random.seed(0)
    model = NeuralNetSeldonianGD(X_tr, y_tr, g_hats=ghats, random_seed=0,
                                 margin=args.margin, epochs=args.epochs,
                                 lambda_lr=args.lambda_lr)
    result = model.fit()
    report("Seldonian NeuralNet head (gradient-based Adam)", model.predict(X_te),
           y_te, X_te, A_idx, safety=model.safetyTest(), seconds=time.time() - t0,
           solution=result is not None)


if __name__ == "__main__":
    main()
