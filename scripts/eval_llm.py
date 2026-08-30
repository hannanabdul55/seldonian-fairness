"""Seldonian fairness constraint on top of a small LLM.

A frozen bert-tiny (~4M params) encodes short reviews into embeddings; a Seldonian
neural-net head is trained on [embedding, gender] features with a TPR-parity
constraint. The synthetic corpus is deliberately biased: the positive rate differs
by the gender of the sentence subject, and 30% of sentences carry no sentiment
words at all, so an unconstrained classifier learns to use gender as a shortcut.

Requires the llm extra: uv sync --extra llm
"""
import argparse
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split

try:
    from seldonian.encoders import embed_texts, normalize_embeddings
except ImportError as e:
    raise SystemExit("transformers not installed - run: uv sync --extra llm") from e

from seldonian.objectives import ghat_tpr_diff, ghat_tpr_diff_t, tpr_rate
from seldonian.seldonian import NeuralNetSeldonianGD

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
THRESHOLD = 0.15
DELTA = 0.05
N = 4000
POS_RATE = {1: 0.4, 0: 0.8}  # positive-label rate per gender group
SENTIMENT_PROB = 0.7         # chance a sentence contains actual sentiment words

SUBJECTS = {
    1: ["He", "My brother", "Mr. Smith", "The waiter", "My uncle", "The salesman"],
    0: ["She", "My sister", "Mrs. Smith", "The waitress", "My aunt", "The saleswoman"],
}
POSITIVE = ["loved the movie", "had a wonderful time", "thought it was excellent",
            "was thrilled with the service", "found the food delightful"]
NEGATIVE = ["hated the movie", "had a terrible time", "thought it was awful",
            "was disappointed with the service", "found the food bland"]
NEUTRAL = ["watched the movie yesterday", "visited the place downtown",
           "attended the event on Sunday", "used the service last week",
           "ordered the food to go"]


def make_corpus(n, rng):
    texts, labels, genders = [], [], []
    for _ in range(n):
        g = int(rng.integers(2))
        y = int(rng.random() < POS_RATE[g])
        if rng.random() < SENTIMENT_PROB:
            phrase = rng.choice(POSITIVE if y else NEGATIVE)
        else:
            phrase = rng.choice(NEUTRAL)
        texts.append(f"{rng.choice(SUBJECTS[g])} {phrase}.")
        labels.append(y)
        genders.append(g)
    return texts, np.array(labels), np.array(genders)


def report(name, preds, y_te, X_te, A_idx, safety=None, seconds=None, solution=None):
    tpr_a = tpr_rate(A_idx, 1)(X_te, y_te, preds).mean()
    tpr_b = tpr_rate(A_idx, 0)(X_te, y_te, preds).mean()
    g_test = ghat_tpr_diff(A_idx, threshold=THRESHOLD)(X_te, y_te, preds,
                                                       delta=DELTA, ub=False)
    print(f"{name}")
    print(f"  accuracy={accuracy_score(y_te, preds):.3f}  "
          f"balanced_accuracy={balanced_accuracy_score(y_te, preds):.3f}")
    print(f"  TPR[male]={tpr_a:.3f}  TPR[female]={tpr_b:.3f}  gap={abs(tpr_a - tpr_b):.3f}  "
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
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="HF model id to use as the frozen encoder")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    texts, y, gender = make_corpus(N, rng)
    print(f"Corpus: {N} sentences, positive rate male={y[gender == 1].mean():.2f} "
          f"female={y[gender == 0].mean():.2f}")

    t0 = time.time()
    emb = embed_texts(texts, args.model, max_length=32)
    print(f"Embedded with {args.model} ({emb.shape[1]}-dim) in {time.time() - t0:.1f}s\n")

    emb = normalize_embeddings(emb)
    # gender appended as the last feature column so the g-hat functions can mask on
    # it; it must stay exactly 0/1
    X = np.hstack([emb, gender[:, None]]).astype(np.float32)
    A_idx = X.shape[1] - 1
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=1)

    t0 = time.time()
    base = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
    report("Unconstrained LogisticRegression head", base.predict(X_te), y_te, X_te,
           A_idx, seconds=time.time() - t0)

    ghats = [{"fn": ghat_tpr_diff_t(A_idx, threshold=THRESHOLD), "delta": DELTA}]
    t0 = time.time()
    np.random.seed(0)
    model = NeuralNetSeldonianGD(X_tr, y_tr, g_hats=ghats, random_seed=0)
    result = model.fit()
    report("Seldonian NeuralNet head (gradient-based Adam)", model.predict(X_te),
           y_te, X_te, A_idx, safety=model.safetyTest(), seconds=time.time() - t0,
           solution=result is not None)


if __name__ == "__main__":
    main()
