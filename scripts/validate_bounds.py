"""
Validation suite for the mean bounds in ``seldonian.bounds``.

Stages (``--stage`` may be given several times; default runs all):

* ``exact``     exact (enumerated) coverage and slack on two-point laws, adversarial sweep
* ``three``     exact coverage on three-point laws (small n)
* ``mc``        Monte Carlo coverage / slack on continuous laws
* ``diff``      exact coverage / slack of the fairness-constraint bound on |rate_a - rate_b|
* ``shuffle``   order dependence of the sequential betting bound ("shuffle hacking")
* ``seldonian`` end-to-end: solution rate / true-violation rate of a Seldonian classifier

Results are written as JSON + Markdown + PNG to ``reports/bounds/``.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import binom

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from seldonian import bounds_eval  # noqa: E402
from seldonian.bounds import BOUNDS  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), '..', 'reports', 'bounds')
os.makedirs(OUT, exist_ok=True)

ALL_METHODS = list(BOUNDS)
GENERAL = [m for m in ALL_METHODS if m != 'clopper_pearson']
DELTA = 0.05
KW = {'learned_miller_thomas': {'draws': 2000}}


def kw(method):
    return KW.get(method, {})


def dump(name, obj):
    with open(os.path.join(OUT, name + '.json'), 'w') as f:
        json.dump(obj, f, indent=1, default=float)


def md_table(header, rows, fmt='{:.4f}'):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    for r in rows:
        out.append('| ' + ' | '.join(fmt.format(c) if isinstance(c, float) else str(c)
                                     for c in r) + ' |')
    return '\n'.join(out)


# --------------------------------------------------------------------------------------
# stage: exact two-point
# --------------------------------------------------------------------------------------

EXACT_NS = [5, 10, 20, 50, 100, 200, 500, 1000]
PAIRS = [(0.0, 1.0), (0.0, 0.5), (0.5, 1.0), (0.0, 0.1), (0.9, 1.0), (0.25, 0.75)]
P_GRID = np.concatenate((np.geomspace(1e-3, 0.05, 40), np.linspace(0.05, 0.95, 361),
                         1 - np.geomspace(1e-3, 0.05, 40)[::-1]))
SLACK_PS = [0.01, 0.05, 0.2, 0.5]


def _exact_task(args):
    method, n, v0, v1, delta = args
    t0 = time.time()
    table = bounds_eval.two_point_table(method, n, delta, v0, v1, **kw(method))
    res = bounds_eval.exact_two_point(method, n, delta, v0, v1, P_GRID, table=table)
    slack = bounds_eval.exact_two_point(method, n, delta, v0, v1, np.array(SLACK_PS),
                                        table=table)
    return {'method': method, 'n': n, 'pair': [v0, v1], 'delta': delta,
            'min_cov_lower': float(res['cov_lower'].min()),
            'min_cov_upper': float(res['cov_upper'].min()),
            'argmin_p_lower': float(P_GRID[int(np.argmin(res['cov_lower']))]),
            'argmin_p_upper': float(P_GRID[int(np.argmin(res['cov_upper']))]),
            'cov_lower_curve': res['cov_lower'].tolist(),
            'cov_upper_curve': res['cov_upper'].tolist(),
            'slack_ps': SLACK_PS, 'slack_upper': slack['slack_upper'].tolist(),
            'slack_lower': slack['slack_lower'].tolist(),
            'seconds': time.time() - t0, 'ms_per_call': 1000 * (time.time() - t0) / (n + 1)}


def load_existing(name):
    path = os.path.join(OUT, name + '.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def merge(existing, fresh, keys):
    def key(r):
        return tuple(tuple(r[k]) if isinstance(r[k], list) else r[k] for k in keys)
    fresh_keys = {key(r) for r in fresh}
    return [r for r in existing if key(r) not in fresh_keys] + fresh


def stage_exact(workers, methods=None):
    tasks = []
    for method in methods or ALL_METHODS:
        for n in EXACT_NS:
            for v0, v1 in PAIRS:
                if method == 'clopper_pearson' and (v0, v1) != (0.0, 1.0):
                    continue
                tasks.append((method, n, v0, v1, DELTA))
    # slowest first for load balance
    tasks.sort(key=lambda t: -t[1] * (30 if t[0] == 'learned_miller_thomas' else 1))
    with ProcessPoolExecutor(workers) as ex:
        results = list(ex.map(_exact_task, tasks, chunksize=1))
    results = merge(load_existing('exact_two_point'), results, ('method', 'n', 'pair'))
    dump('exact_two_point', results)
    write_exact_report(results)


def write_exact_report(results):
    lines = ['# Exact coverage on two-point laws', '',
             f'delta = {DELTA} per endpoint (one-sided). Coverage computed exactly by '
             'enumerating the binomial count for every mixing probability on a grid of '
             f'{len(P_GRID)} points and atom pairs {PAIRS}. Entries below 1 - delta = '
             f'{1 - DELTA} mean the bound is **invalid** for that sample size.', '',
             '## Worst-case one-sided coverage (min over atoms and mixing probability)', '']
    header = ['method'] + [f'n={n}' for n in EXACT_NS]
    rows = []
    for method in ALL_METHODS:
        row = [method]
        for n in EXACT_NS:
            rs = [r for r in results if r['method'] == method and r['n'] == n]
            row.append(min(min(r['min_cov_lower'], r['min_cov_upper']) for r in rs))
        rows.append(row)
    lines += [md_table(header, rows), '']
    lines += ['## Worst case location (atoms, p) for the upper endpoint at n = 20 and n = 200', '']
    rows = []
    for method in ALL_METHODS:
        row = [method]
        for n in (20, 200):
            rs = [r for r in results if r['method'] == method and r['n'] == n]
            r = min(rs, key=lambda r: r['min_cov_upper'])
            row.append(f"{r['min_cov_upper']:.4f} @ {tuple(r['pair'])}, p={r['argmin_p_upper']:.3f}")
        rows.append(row)
    lines += [md_table(['method', 'n=20', 'n=200'], rows), '']
    for v0, v1 in PAIRS:
        lines += [f'## Expected upper slack E[upper - mean], atoms ({v0}, {v1})', '',
                  'Rows: method; columns: n and P(X = v1). Smaller is tighter; compare only '
                  'among rows whose coverage is valid above.', '']
        header = ['method'] + [f'n={n}, p={p}' for n in (20, 100, 1000) for p in SLACK_PS]
        rows = []
        for method in ALL_METHODS:
            if method == 'clopper_pearson' and (v0, v1) != (0.0, 1.0):
                continue
            row = [method]
            for n in (20, 100, 1000):
                r = next(r for r in results if r['method'] == method and r['n'] == n
                         and r['pair'] == [v0, v1])
                row += [float(s) for s in r['slack_upper']]
            rows.append(row)
        lines += [md_table(header, rows), '']
    lines += ['## Cost (ms per bound evaluation, n = 1000)', '']
    rows = [[m, np.mean([r['ms_per_call'] for r in results if r['method'] == m and r['n'] == 1000])]
            for m in ALL_METHODS]
    lines += [md_table(['method', 'ms/call'], rows, '{:.2f}'), '']
    with open(os.path.join(OUT, 'exact_two_point.md'), 'w') as f:
        f.write('\n'.join(lines))
    plot_exact(results)


def plot_exact(results):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for method in ALL_METHODS:
        ys = []
        for n in EXACT_NS:
            rs = [r for r in results if r['method'] == method and r['n'] == n]
            ys.append(min(min(r['min_cov_lower'], r['min_cov_upper']) for r in rs))
        axes[0].plot(EXACT_NS, ys, marker='o', label=method)
    axes[0].axhline(1 - DELTA, color='k', ls='--', lw=1)
    axes[0].set_xscale('log')
    axes[0].set_ylim(0.3, 1.01)
    axes[0].set_xlabel('n')
    axes[0].set_ylabel('worst-case one-sided coverage')
    axes[0].set_title('Exact worst-case coverage over two-point laws')
    for method in ALL_METHODS:
        ys = []
        for n in EXACT_NS:
            r = next(r for r in results if r['method'] == method and r['n'] == n
                     and r['pair'] == [0.0, 1.0])
            ys.append(r['slack_upper'][SLACK_PS.index(0.05)])
        axes[1].plot(EXACT_NS, ys, marker='o', label=method)
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('n')
    axes[1].set_ylabel('E[upper - mean]')
    axes[1].set_title('Expected upper slack, Bernoulli(0.05)')
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'exact_two_point.png'), dpi=120)


# --------------------------------------------------------------------------------------
# stage: exact three-point
# --------------------------------------------------------------------------------------

THREE_NS = [10, 25, 40]
THREE_VALUES = [(0.0, 0.5, 1.0), (0.0, 0.1, 1.0), (0.0, 0.9, 1.0), (0.0, 0.05, 0.2)]


def _simplex(step):
    out = []
    for i in range(int(round(1 / step)) + 1):
        for j in range(int(round(1 / step)) + 1 - i):
            p1, p2 = i * step, j * step
            out.append([1 - p1 - p2, p1, p2])
    return [np.clip(np.array(p), 0, 1) for p in out if min(p) >= -1e-12]


def _three_task(args):
    method, n, values = args
    probs = _simplex(0.1)
    res = bounds_eval.exact_three_point(method, n, DELTA, values, probs, **kw(method))
    return {'method': method, 'n': n, 'values': list(values),
            'min_cov_lower': min(r['cov_lower'] for r in res),
            'min_cov_upper': min(r['cov_upper'] for r in res),
            'mean_slack_upper': float(np.mean([r['slack_upper'] for r in res]))}


def stage_three(workers):
    tasks = [(m, n, v) for m in GENERAL for n in THREE_NS for v in THREE_VALUES]
    tasks.sort(key=lambda t: -t[1] * (30 if t[0] == 'learned_miller_thomas' else 1))
    with ProcessPoolExecutor(workers) as ex:
        results = list(ex.map(_three_task, tasks, chunksize=1))
    dump('exact_three_point', results)
    lines = ['# Exact coverage on three-point laws', '',
             f'delta = {DELTA}; atoms {THREE_VALUES}; 66 probability vectors on a 0.1 simplex '
             'grid; exact multinomial enumeration. Minimum one-sided coverage:', '']
    header = ['method'] + [f'n={n}' for n in THREE_NS]
    rows = []
    for m in GENERAL:
        rows.append([m] + [min(min(r['min_cov_lower'], r['min_cov_upper'])
                               for r in results if r['method'] == m and r['n'] == n)
                           for n in THREE_NS])
    lines += [md_table(header, rows), '']
    with open(os.path.join(OUT, 'exact_three_point.md'), 'w') as f:
        f.write('\n'.join(lines))


# --------------------------------------------------------------------------------------
# stage: Monte Carlo on continuous laws
# --------------------------------------------------------------------------------------

MC_NS = [10, 30, 100, 300, 1000]
MC_DISTS = list(bounds_eval.DISTRIBUTIONS)


def _mc_task(args):
    method, dist, n, reps, mean = args
    r = bounds_eval.mc_coverage(method, dist, n, DELTA, reps=reps, seed=n * 31 + 7,
                                mean=mean, **kw(method))
    r.update({'method': method, 'dist': dist, 'n': n})
    return r


def stage_mc(workers, reps):
    means = {d: bounds_eval.true_mean(d) for d in MC_DISTS}
    tasks = [(m, d, n, reps, means[d]) for m in GENERAL for d in MC_DISTS for n in MC_NS]
    tasks.sort(key=lambda t: -t[2] * (30 if t[0] == 'learned_miller_thomas' else 1))
    with ProcessPoolExecutor(workers) as ex:
        results = list(ex.map(_mc_task, tasks, chunksize=1))
    dump('mc_continuous', results)
    lines = ['# Monte Carlo coverage on continuous laws', '',
             f'delta = {DELTA}, {reps} replications per cell. Cells show the estimated '
             'one-sided miscoverage max(lower, upper) with its 99% Clopper-Pearson upper '
             'confidence limit in brackets; a cell whose limit is below 0.05 is certified '
             '(at 99%) not to exceed delta.', '']
    for dist in MC_DISTS:
        lines += [f'## {dist}  (mean = {means[dist]:.4f})', '',
                  '### miscoverage', '']
        header = ['method'] + [f'n={n}' for n in MC_NS]
        rows = []
        for m in GENERAL:
            row = [m]
            for n in MC_NS:
                r = next(r for r in results if r['method'] == m and r['dist'] == dist and r['n'] == n)
                miss = max(r['miss_lower'], r['miss_upper'])
                ucl = max(r['miss_lower_ucl'], r['miss_upper_ucl'])
                row.append(f'{miss:.3f} [{ucl:.3f}]')
            rows.append(row)
        lines += [md_table(header, rows), '', '### expected slack E[upper - mean]', '']
        rows = []
        for m in GENERAL:
            row = [m]
            for n in MC_NS:
                r = next(r for r in results if r['method'] == m and r['dist'] == dist and r['n'] == n)
                row.append(float(r['slack_upper']))
            rows.append(row)
        lines += [md_table(header, rows), '']
    with open(os.path.join(OUT, 'mc_continuous.md'), 'w') as f:
        f.write('\n'.join(lines))


# --------------------------------------------------------------------------------------
# stage: two-sample difference bound
# --------------------------------------------------------------------------------------

DIFF_METHODS = ['ttest', 'hoeffdings', 'clopper_pearson', 'bentkus', 'betting_mixture',
                'bentkus_diff', 'convex_order_diff']
# the convex-order difference bound costs ~0.1 s per cell; skip the largest exact table
DIFF_SKIP = {('convex_order_diff', 200, 200)}
DIFF_SIZES = [(20, 20), (50, 50), (100, 100), (200, 200), (30, 150)]
DIFF_P = np.linspace(0.02, 0.98, 25)
DIFF_SLACK_AT = [(0.5, 0.5), (0.4, 0.6), (0.1, 0.3), (0.05, 0.05), (0.02, 0.5)]


def _diff_task(args):
    method, n_a, n_b = args
    t0 = time.time()
    table = bounds_eval.diff_upper_table(method, n_a, n_b, DELTA)
    cov = np.array([[bounds_eval.exact_diff(method, n_a, n_b, DELTA, pa, pb, table)['cov']
                     for pb in DIFF_P] for pa in DIFF_P])
    slack = {f'{pa},{pb}': bounds_eval.exact_diff(method, n_a, n_b, DELTA, pa, pb, table)['slack']
             for pa, pb in DIFF_SLACK_AT}
    return {'method': method, 'n_a': n_a, 'n_b': n_b, 'min_cov': float(cov.min()),
            'argmin': [float(DIFF_P[i]) for i in np.unravel_index(cov.argmin(), cov.shape)],
            'slack': slack, 'ms_per_call': 1000 * (time.time() - t0) / table.size}


# fixed classifiers whose true gap exceeds the threshold: the safety test must pass with
# probability <= delta. (p_a, p_b, n_a, n_b) with threshold 0.2 and |p_a - p_b| = 0.25/0.3
FALSE_PASS_CASES = [(0.75, 1.0, 8, 40), (0.75, 1.0, 15, 60), (0.7, 0.95, 15, 60),
                    (0.6, 0.85, 30, 100), (0.5, 0.75, 60, 60), (0.02, 0.27, 40, 40),
                    (0.0, 0.3, 10, 100)]
FALSE_PASS_THRESHOLD = 0.2


def _false_pass_task(args):
    method, pa, pb, na, nb = args
    table = bounds_eval.diff_upper_table(method, na, nb, DELTA)
    wa = binom.pmf(np.arange(na + 1), na, pa)
    wb = binom.pmf(np.arange(nb + 1), nb, pb)
    w = wa[:, None] * wb[None, :]
    return {'method': method, 'case': [pa, pb, na, nb],
            'false_pass': float((w * (table <= FALSE_PASS_THRESHOLD)).sum())}


def stage_false_pass(workers):
    tasks = [(m, *c) for m in DIFF_METHODS for c in FALSE_PASS_CASES]
    with ProcessPoolExecutor(workers) as ex:
        results = list(ex.map(_false_pass_task, tasks, chunksize=1))
    dump('false_pass', results)
    lines = ['# Safety-test false-pass probability for unsafe classifiers', '',
             f'A fixed classifier with true group TPRs (p_a, p_b) violates the constraint '
             f'|TPR_a - TPR_b| <= {FALSE_PASS_THRESHOLD}. The safety set has n_a and n_b '
             'positives in the two groups. The table gives the exact probability that the '
             f'safety test (delta = {DELTA}) nevertheless certifies the classifier; a '
             f'Seldonian method must stay <= {DELTA}.', '']
    header = ['method'] + [f'p=({pa},{pb}) n=({na},{nb})' for pa, pb, na, nb in FALSE_PASS_CASES]
    rows = [[m] + [next(r['false_pass'] for r in results if r['method'] == m and r['case'] == list(c))
                   for c in FALSE_PASS_CASES] for m in DIFF_METHODS]
    lines += [md_table(header, rows), '']
    with open(os.path.join(OUT, 'false_pass.md'), 'w') as f:
        f.write('\n'.join(lines))


def stage_diff(workers, methods=None):
    tasks = [(m, na, nb) for m in (methods or DIFF_METHODS) for na, nb in DIFF_SIZES
             if (m, na, nb) not in DIFF_SKIP]
    tasks.sort(key=lambda t: -(t[1] * t[2]) * (3 if t[0].endswith('diff') else 1))
    with ProcessPoolExecutor(workers) as ex:
        results = list(ex.map(_diff_task, tasks, chunksize=1))
    results = merge(load_existing('diff_exact'), results, ('method', 'n_a', 'n_b'))
    dump('diff_exact', results)
    lines = ['# Fairness-constraint bound on |rate_a - rate_b|', '',
             f'The constraint objective certifies an upper bound on the absolute TPR gap at '
             f'delta = {DELTA}. One-sample methods build a rectangle from two intervals '
             '(delta/4 per tail); `bentkus_diff` bounds the difference directly '
             '(delta/2 per tail). Coverage is exact (enumeration of both binomial counts) '
             f'and minimised over a {len(DIFF_P)}x{len(DIFF_P)} grid of (p_a, p_b).', '',
             '## Minimum coverage of the certified upper bound', '']
    header = ['method'] + [f'n=({na},{nb})' for na, nb in DIFF_SIZES]
    def cell(m, na, nb, key):
        r = next((r for r in results if r['method'] == m and (r['n_a'], r['n_b']) == (na, nb)), None)
        return 'n/a' if r is None else (r['min_cov'] if key == 'min_cov' else r['slack'][key])
    rows = [[m] + [cell(m, na, nb, 'min_cov') for na, nb in DIFF_SIZES] for m in DIFF_METHODS]
    lines += [md_table(header, rows), '']
    for pa, pb in DIFF_SLACK_AT:
        lines += [f'## Expected slack E[upper - |p_a - p_b|] at (p_a, p_b) = ({pa}, {pb})', '']
        rows = [[m] + [cell(m, na, nb, f'{pa},{pb}') for na, nb in DIFF_SIZES]
                for m in DIFF_METHODS]
        lines += [md_table(header, rows), '']
    rows = [[m, np.mean([r['ms_per_call'] for r in results if r['method'] == m])] for m in DIFF_METHODS]
    lines += ['## Cost (ms per constraint evaluation)', '', md_table(['method', 'ms/call'], rows, '{:.2f}'), '']
    with open(os.path.join(OUT, 'diff_exact.md'), 'w') as f:
        f.write('\n'.join(lines))


# --------------------------------------------------------------------------------------
# stage: shuffle hacking
# --------------------------------------------------------------------------------------

def stage_shuffle(workers):
    results = []
    for dist in ['beta(2,8)', 'spike-slab', 'uniform']:
        for n in [50, 200]:
            r = bounds_eval.shuffle_hack('betting', dist, n, DELTA, tries=20, reps=300)
            r.update({'method': 'betting', 'dist': dist, 'n': n})
            results.append(r)
            r2 = bounds_eval.shuffle_hack('betting', dist, n, DELTA, tries=20, reps=100,
                                          permutations=16)
            r2.update({'method': 'betting(permutations=16)', 'dist': dist, 'n': n})
            results.append(r2)
            r3 = bounds_eval.shuffle_hack('betting_mixture', dist, n, DELTA, tries=5, reps=100)
            r3.update({'method': 'betting_mixture', 'dist': dist, 'n': n})
            results.append(r3)
    dump('shuffle_hack', results)
    lines = ['# Order dependence of the sequential betting bound', '',
             'An adversary re-shuffles the safety data until the lower bound clears the '
             'threshold. `miss_single` is the honest one-sided miscoverage (one ordering); '
             '`miss_best_of_tries` is the miscoverage of the best of `tries` orderings; '
             '`mean_spread` is the average max - min of the lower bound across orderings.', '']
    header = ['method', 'dist', 'n', 'tries', 'miss_single', 'miss_best_of_tries', 'mean_spread']
    rows = [[r['method'], r['dist'], r['n'], r['tries'], r['miss_single'],
             r['miss_best_of_tries'], r['mean_spread']] for r in results]
    lines += [md_table(header, rows), '']
    with open(os.path.join(OUT, 'shuffle_hack.md'), 'w') as f:
        f.write('\n'.join(lines))


# --------------------------------------------------------------------------------------
# stage: end-to-end Seldonian classifier
# --------------------------------------------------------------------------------------

SELD_METHODS = ['ttest', 'hoeffdings', 'clopper_pearson', 'bentkus', 'betting_mixture',
                'bentkus_diff', 'convex_order_diff']
SELD_NS = [150, 300, 600, 1200, 2500, 5000]
SELD_SEEDS = 40
SELD_THRESHOLD = 0.2
SELD_D = 5
SELD_TP = (0.4, 0.8)


def _seld_task(args):
    import io
    import contextlib
    from sklearn.linear_model import LogisticRegression
    from seldonian.objectives import ghat_tpr_diff_t, tpr_rate
    from seldonian.seldonian import LogisticRegressionSeldonianGD
    from seldonian.synthetic import make_synthetic
    method, n, seed = args
    A_idx = 2
    X, y, _ = make_synthetic(n, SELD_D, tp_a=SELD_TP[0], tp_b=SELD_TP[1], A_idx=A_idx,
                             seed=seed)
    X_true, y_true, _ = make_synthetic(200_000, SELD_D, tp_a=SELD_TP[0], tp_b=SELD_TP[1],
                                       A_idx=A_idx, seed=10_000 + seed)
    ghats = [{'fn': ghat_tpr_diff_t(A_idx, method=method, threshold=SELD_THRESHOLD),
              'delta': DELTA}]
    t0 = time.time()
    model = LogisticRegressionSeldonianGD(X, y, g_hats=ghats, random_seed=seed)
    with contextlib.redirect_stdout(io.StringIO()):
        result = model.fit()
    seconds = time.time() - t0
    preds = model.predict(X_true)
    gap = abs(tpr_rate(A_idx, 1)(X_true, y_true, preds).mean()
              - tpr_rate(A_idx, 0)(X_true, y_true, preds).mean())
    acc = float((preds == y_true).mean())
    base = LogisticRegression(max_iter=1000).fit(X, y)
    bp = base.predict(X_true)
    base_gap = abs(tpr_rate(A_idx, 1)(X_true, y_true, bp).mean()
                   - tpr_rate(A_idx, 0)(X_true, y_true, bp).mean())
    return {'method': method, 'n': n, 'seed': seed, 'solution': result is not None,
            'true_gap': float(gap), 'violation': bool(gap > SELD_THRESHOLD),
            'accuracy': acc, 'base_gap': float(base_gap),
            'base_accuracy': float((bp == y_true).mean()), 'seconds': seconds}


def stage_seldonian(workers, methods=None):
    tasks = [(m, n, s) for m in (methods or SELD_METHODS) for n in SELD_NS
             for s in range(SELD_SEEDS)]
    with ProcessPoolExecutor(workers) as ex:
        results = list(ex.map(_seld_task, tasks, chunksize=4))
    results = merge(load_existing('seldonian_end_to_end'), results, ('method', 'n', 'seed'))
    dump('seldonian_end_to_end', results)
    lines = ['# End-to-end Seldonian classification', '',
             f'`LogisticRegressionSeldonianGD` on `make_synthetic` (d={SELD_D}, group base '
             f'rates {SELD_TP}), constraint |TPR_a - TPR_b| <= {SELD_THRESHOLD} at delta = '
             f'{DELTA}, {SELD_SEEDS} seeds per cell. The true gap of each returned model is '
             'measured on 200k fresh samples. "failure" = a solution was returned whose true '
             f'gap exceeds {SELD_THRESHOLD} (must stay <= delta for a Seldonian method); '
             '"solution rate" = fraction of runs that passed the safety test.', '']
    for key, title in [('solution', 'Solution rate'), ('failure', 'Failure rate (unsafe solution returned)'),
                       ('accuracy', 'Accuracy of returned solutions (200k test)'),
                       ('gap', 'True TPR gap of returned solutions')]:
        lines += [f'## {title}', '']
        header = ['method'] + [f'n={n}' for n in SELD_NS]
        rows = []
        for m in SELD_METHODS:
            row = [m]
            for n in SELD_NS:
                rs = [r for r in results if r['method'] == m and r['n'] == n]
                sol = [r for r in rs if r['solution']]
                if key == 'solution':
                    row.append(len(sol) / len(rs))
                elif key == 'failure':
                    row.append(sum(r['violation'] for r in sol) / len(rs))
                elif key == 'accuracy':
                    row.append(float(np.mean([r['accuracy'] for r in sol])) if sol else float('nan'))
                else:
                    row.append(float(np.mean([r['true_gap'] for r in sol])) if sol else float('nan'))
            rows.append(row)
        lines += [md_table(header, rows, '{:.3f}'), '']
    base = [r for r in results if r['method'] == SELD_METHODS[0]]
    lines += ['## Unconstrained logistic regression reference', '']
    rows = [[n, float(np.mean([r['base_accuracy'] for r in base if r['n'] == n])),
             float(np.mean([r['base_gap'] for r in base if r['n'] == n])),
             float(np.mean([r['base_gap'] > SELD_THRESHOLD for r in base if r['n'] == n]))]
            for n in SELD_NS]
    lines += [md_table(['n', 'accuracy', 'true gap', 'P(gap > threshold)'], rows, '{:.3f}'), '']
    with open(os.path.join(OUT, 'seldonian_end_to_end.md'), 'w') as f:
        f.write('\n'.join(lines))


def stage_confirm(workers):
    """
    Targeted re-checks of the two cells of the exact stage that rely on Monte Carlo inside
    the bound or inside the enumeration: the Learned-Miller-Thomas quantile (2000 draws)
    and the order average of the sequential betting bound (32 orderings).
    """
    rows = []
    p_grid = np.linspace(0.01, 0.99, 99)
    for n in (20, 100):
        for draws in (2000, 20000):
            lo, hi, al, ah = bounds_eval.worst_case_two_point(
                'learned_miller_thomas', n, DELTA, [(0.0, 1.0)], p_grid, draws=draws)
            rows.append(['learned_miller_thomas', f'draws={draws}', n, min(lo, hi)])
    for n in (50, 100):
        for orderings in (32, 256):
            lo, hi, al, ah = bounds_eval.worst_case_two_point(
                'betting', n, DELTA, [(0.0, 1.0)], p_grid, orderings=orderings)
            rows.append(['betting', f'orderings={orderings}', n, min(lo, hi)])
    # direct Monte Carlo with fresh random orderings at the reported argmin (n=100, p=0.152)
    rng = np.random.default_rng(5)
    from seldonian.bounds import betting_bounds
    miss = 0
    reps = 20000
    for _ in range(reps):
        x = (rng.random(100) < 0.152).astype(float)
        rv = betting_bounds(x, DELTA)
        miss += (rv.upper < 0.152) or (rv.lower > 0.152)
    rows.append(['betting', f'MC {reps} reps, p=0.152, both sides', 100, 1 - miss / reps])
    dump('confirm', rows)
    lines = ['# Confirmation of Monte-Carlo-affected cells', '',
             'Worst-case one-sided coverage on Bernoulli laws (99-point p grid) with more '
             'Monte Carlo effort inside the bound / the enumeration. The last row is a plain '
             'Monte Carlo estimate of the two-sided coverage at the previously reported '
             'worst case (so its target is 1 - 2 delta = 0.90).', '',
             md_table(['method', 'setting', 'n', 'min coverage'], rows), '']
    with open(os.path.join(OUT, 'confirm.md'), 'w') as f:
        f.write('\n'.join(lines))


STAGES = {'exact': stage_exact, 'confirm': stage_confirm, 'false_pass': stage_false_pass, 'three': stage_three, 'mc': None, 'diff': stage_diff,
          'shuffle': stage_shuffle, 'seldonian': stage_seldonian}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', action='append', choices=list(STAGES))
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--mc-reps', type=int, default=2000)
    ap.add_argument('--methods', nargs='*', help='exact/diff/seldonian stages: rerun only '
                    'these methods and merge into the existing results')
    args = ap.parse_args()
    for stage in args.stage or list(STAGES):
        t0 = time.time()
        print(f'=== stage {stage}', flush=True)
        if stage == 'mc':
            stage_mc(args.workers, args.mc_reps)
        elif stage in ('exact', 'diff', 'seldonian'):
            STAGES[stage](args.workers, args.methods)
        else:
            STAGES[stage](args.workers)
        print(f'=== stage {stage} done in {time.time() - t0:.0f}s', flush=True)
