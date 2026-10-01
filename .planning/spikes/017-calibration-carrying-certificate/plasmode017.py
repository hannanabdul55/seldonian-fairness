"""Spike 017, stage B: plasmode of every certificate route on 015's real judge scores.

The pool is 015's 500 scored responses with a gold label on every one (refusal: the
Qwen3Guard-4B refusal field; brevity: words > 80, exact). A draw is N responses i.i.d. from
the pool (with replacement, so the pool's mean is the exact truth: 013's rule), of which the
first n carry the gold label. Every route in ``cert017`` gives its upper bound; the table
reports the miss rate against delta, the mean bound, and the effective-label gain over the
labels alone (variance ratio of the estimates), beside the gain the formula predicts from
the pool's ``rho^2``. ``boot`` is PPI++ with the studentised bootstrap limit.

Judge features: ``f01`` (p > 0.5), ``p``, ``logit`` (the logit clipped to +-11.5 and mapped
to [0, 1]; scores were stored to 5 decimals), ``platt`` (a two-fold cross-fitted logistic
recalibration of the logit: each half of the labels fits the map used on the other half,
and the unlabelled responses get the average of the two maps).

``--shift`` runs the prevalence-shifted version: Y ~ Bernoulli(r), the score drawn from the
pool's scores with that label, so the same real judge is seen at r = 0.2, 0.05 and 0.013.

    OMP_NUM_THREADS=1 ../../../.venv/bin/python plasmode017.py --reps 4000      # ~10 min
    OMP_NUM_THREADS=1 ../../../.venv/bin/python plasmode017.py --shift --reps 4000
"""
import argparse
import collections
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import cert017 as c

HERE = os.path.dirname(os.path.abspath(__file__))
S015 = os.path.join(c.REPO, "results", "spikes", "015", "scores.jsonl")
DELTA = 0.05
LCLIP = 11.5
FEATS = ("f01", "p", "logit", "platt")


def load_pools():
    d = collections.defaultdict(list)
    for line in open(S015):
        r = json.loads(line)
        if r["variant"] in ("rubric", "raw"):
            d[(r["task"], r["variant"], r["wording"])].append(r)
    out = {}
    for k, rows in d.items():
        rows = sorted(rows, key=lambda r: r["id"])
        out[k] = (np.array([r["ref"] for r in rows], dtype=float),
                  np.array([r["p"] for r in rows], dtype=float))
    return out


def logit01(p):
    p = np.clip(p, np.exp(-LCLIP) / (1 + np.exp(-LCLIP)), 1 - np.exp(-LCLIP) / (1 + np.exp(-LCLIP)))
    return (np.log(p / (1 - p)) + LCLIP) / (2 * LCLIP)


def platt_fit(x, y, iters=30, ridge=1e-2):
    """Vectorised logistic regression y ~ a + b (2x - 1), one fit per row. Returns (a, b)."""
    z = 2 * x - 1
    a = np.zeros(x.shape[0])
    b = np.zeros(x.shape[0])
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-(a[:, None] + b[:, None] * z)))
        w = mu * (1 - mu)
        ga = (y - mu).sum(1) - ridge * a
        gb = ((y - mu) * z).sum(1) - ridge * b
        haa = w.sum(1) + ridge
        hab = (w * z).sum(1)
        hbb = (w * z * z).sum(1) + ridge
        det = haa * hbb - hab ** 2
        da = (hbb * ga - hab * gb) / det
        db = (haa * gb - hab * ga) / det
        step = np.minimum(1.0, 5.0 / np.maximum(np.abs(da) + np.abs(db), 1e-12))
        a, b = a + step * da, b + step * db
    return a, b


def platt_apply(ab, x):
    a, b = ab
    return 1 / (1 + np.exp(-(a[:, None] + b[:, None] * (2 * x - 1))))


def features(p_lab, p_unl, y_lab):
    """Dict feature -> (f_lab, f_unl). ``platt`` is cross-fitted on two halves of the labels."""
    out = {"f01": ((p_lab > 0.5).astype(float), (p_unl > 0.5).astype(float)),
           "p": (p_lab, p_unl)}
    xl, xu = logit01(p_lab), logit01(p_unl)
    out["logit"] = (xl, xu)
    h = xl.shape[1] // 2
    fa = platt_fit(xl[:, :h], y_lab[:, :h])
    fb = platt_fit(xl[:, h:], y_lab[:, h:])
    out["platt"] = (np.concatenate([platt_apply(fb, xl[:, :h]), platt_apply(fa, xl[:, h:])], axis=1),
                    0.5 * (platt_apply(fa, xu) + platt_apply(fb, xu)))
    return out


def sample(rng, y_pool, p_pool, reps, size, rate=None):
    if rate is None:
        idx = rng.integers(0, len(y_pool), size=(reps, size))
        return y_pool[idx], p_pool[idx]
    pos, neg = p_pool[y_pool == 1], p_pool[y_pool == 0]
    y = (rng.random((reps, size)) < rate).astype(float)
    p = np.where(y == 1, pos[rng.integers(0, len(pos), size=(reps, size))],
                 neg[rng.integers(0, len(neg), size=(reps, size))])
    return y, p


def cell(job):
    key, y_pool, p_pool, n, big_n, reps, rate, seed, block_reps = job
    rng = np.random.default_rng(seed)
    truth = float(y_pool.mean()) if rate is None else rate
    acc = collections.defaultdict(list)
    batch = 250 if big_n > 5000 else 500
    done = 0
    while done < reps:
        b = min(batch, reps - done)
        y, p = sample(rng, y_pool, p_pool, b, big_n, rate)
        yl, pl, pu = y[:, :n], p[:, :n], p[:, n:]
        feats = features(pl, pu, yl)
        acc["classical|-"].append((c.classical(yl, DELTA), yl.mean(1), np.zeros(b)))
        f01l, f01u = feats["f01"]
        f01all = np.concatenate([f01l, f01u], axis=1)
        for name, u in (("naive", c.naive(f01all, DELTA)), ("youden", c.youden(yl, f01l, f01all, DELTA)),
                        ("exact3", c.ppi_exact3(yl, f01l, f01all, DELTA)),
                        ("strat", c.strat_exact(yl, f01l, f01all, DELTA))):
            acc[f"{name}|f01"].append((u, np.full(b, np.nan), np.zeros(b)))
        for fname, (fl, fu) in feats.items():
            e1, _, _ = c.ppi_point(yl, fl, fu, 1.0)
            e2, _, lam = c.ppi_point(yl, fl, fu, None)
            acc[f"ppi|{fname}"].append((c.ppi_clt(yl, fl, fu, DELTA), e1, np.ones(b)))
            acc[f"ppi++|{fname}"].append((c.ppipp_clt(yl, fl, fu, DELTA), e2, lam))
            acc[f"ppi++w|{fname}"].append((c.ppipp_wilson(yl, fl, fu, DELTA), e2, lam))
            acc[f"boot|{fname}"].append((c.ppipp_boot(yl, fl, fu, DELTA, seed=seed + done), e2, lam))
        if done < block_reps:
            k = min(b, block_reps - done)
            yp, pp = sample(rng, y_pool, p_pool, k, n, rate)            # independent pilot for lam
            for fname in ("f01", "logit"):
                fl, fu = feats[fname]
                fp_ = (pp > 0.5).astype(float) if fname == "f01" else logit01(pp)
                lam = np.clip(c.lam_hat(yp, fp_, n / (big_n - n)), 0, 3)
                u = c.ppi_block(yl[:k], fl[:k], fu[:k], DELTA, lam)
                acc[f"block|{fname}"].append((u, np.full(k, np.nan), lam))
        done += b
    rows = []
    v0 = np.concatenate([a[1] for a in acc["classical|-"]]).var()
    for name, parts in acc.items():
        u = np.concatenate([a[0] for a in parts])
        e = np.concatenate([a[1] for a in parts])
        lam = np.concatenate([a[2] for a in parts])
        method, feat = name.split("|")
        rows.append(dict(task=key[0], variant=key[1], wording=key[2], n=n, N=big_n, rate=truth,
                         method=method, feat=feat, reps=len(u), miss=float((u < truth).mean()),
                         bound=float(u.mean()), est=float(np.nanmean(e)) if not np.isnan(e).all() else None,
                         gain=float(v0 / e.var()) if not np.isnan(e).any() and e.var() > 0 else None,
                         lam=float(lam.mean())))
    return rows


def pool_stats(y, p):
    """rho^2 of the gold label with each feature on the whole pool (Platt fitted in-sample)."""
    x = logit01(p)
    ab = platt_fit(x[None, :], y[None, :])
    return {"f01": c.rho2(y, p > 0.5), "p": c.rho2(y, p), "logit": c.rho2(y, x),
            "platt": c.rho2(y, platt_apply(ab, x[None, :])[0])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=4000)
    ap.add_argument("--block-reps", type=int, default=1000)
    ap.add_argument("--shift", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    pools = load_pools()
    jobs, seed = [], 0
    if not a.shift:
        for key in sorted(pools):
            if key[0] == "harm":
                continue
            ns = (50, 100, 225, 500) if key[2] == 0 else (225,)
            for n in ns:
                seed += 1
                jobs.append((key, *pools[key], n, 2000, a.reps, None, seed, a.block_reps))
        for n, big_n in ((225, 20000), (1000, 20000)):          # a large unlabelled set
            for key in (("refusal", "rubric", 0), ("refusal", "raw", 0)):
                seed += 1
                jobs.append((key, *pools[key], n, big_n, min(a.reps, 1000), None, seed,
                             min(a.block_reps, 500)))
        out = "plasmode.json"
    else:
        for key in (("refusal", "rubric", 0), ("refusal", "raw", 0)):
            for rate in (0.2, 0.05, 0.013):
                for n in (225, 1000):
                    seed += 1
                    jobs.append((key, *pools[key], n, 4000, a.reps, rate, 1000 + seed, a.block_reps))
        out = "plasmode_shift.json"
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(cell, jobs)):
            rows += r
            print(f"{i + 1}/{len(jobs)} cells, {time.time() - t0:.0f}s", flush=True)
    stats = {"|".join(map(str, k)): pool_stats(*v) for k, v in pools.items() if k[0] != "harm"}
    json.dump(dict(delta=DELTA, rows=rows, rho2=stats), open(os.path.join(HERE, out), "w"))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
