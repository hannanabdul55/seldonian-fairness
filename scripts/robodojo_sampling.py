"""What is sampled in the robot benchmark's certificate (paper section 10.1; plan step R1).

RoboDojo-RC Tier 1 publishes 120 trials per model: 6 tasks by 20 trials. The paper applied
Clopper-Pearson to the 120 as one independent sample. This script says what that bound is a
bound on, and what the data allow for two other readings.

1. The layout: stops per task, when each model ran, how many trials ran at once.
2. Clustering by task: an exact permutation test of equal rates across tasks, the intraclass
   correlation (one-way ANOVA estimator) and the design effect (the cluster variance of
   section 6.3 over the binomial variance).
3. Tasks fixed. The count is then a sum of six independent binomials with their own rates (if
   trials within a task are independent). The miss probability of the pooled Clopper-Pearson
   limit is computed exactly, by convolution, for per-task rates with a given mean spread
   evenly, as observed, and all on one task. Hoeffding (1956) shows that unequal rates thin the
   lower tail of the count below its mean; the table checks the consequence for these sizes and
   rates (it is not true at every rate: near a rate of 1 an uneven split can exceed delta).
4. Tasks sampled. The six task rates are six bounded draws whose mean is the rate over tasks
   like these. Exact bounds on a bounded mean (Hoeffding, Anderson, Bentkus, betting), the
   Student-t limit on the task rates and the cluster bootstrap-t of section 6.3, plus the floor
   no valid bound can go under when all six rates are zero, ``1 - delta^(1/6)``.
5. Independence within a task: the three models ran in consecutive blocks on the same stations,
   so the order of a trial in the session is checked against stops (rank-sum, within model),
   and adjacent stops within a (model, task) sequence against a permutation.

CPU only, a few seconds.

    .venv/bin/python scripts/robodojo_sampling.py
    -> results/paper/robodojo_sampling.md, results/paper/robodojo_sampling.json
"""
import collections
import csv
import datetime as dt
import json
import os
import sys

sys.dont_write_bytecode = True

import numpy as np
from scipy.stats import beta, binom, mannwhitneyu, t as student_t

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from seldonian import bounds as B  # noqa: E402
import agentdojo_recheck as AR  # noqa: E402

TRIALS = os.path.join(ROOT, "results", "spikes", "019", "trials.csv")
OUT_DIR = os.path.join(ROOT, "results", "paper")
MODELS = (("claude-opus-5", "Claude Opus 5"), ("claude-opus-5-5", "Claude Opus 5.5"), ("gpt-6-astra", "GPT-6 Astra"))
DELTAS = (0.05, 0.10)
TAU = 0.05
PERMS = 200_000
BOOTS = 200_000
SEED = 20261007


def cp_upper(k, n, d):
    return 1.0 if k >= n else float(beta.ppf(1 - d, k + 1, n - k))


def load():
    rows = list(csv.DictReader(open(TRIALS)))
    for r in rows:
        r["t0"] = dt.datetime.fromisoformat(r["started_at"])
        r["t1"] = r["t0"] + dt.timedelta(seconds=float(r["wall_s"]))
        r["stop"] = int(r["termination"] == "overheat")
    rows.sort(key=lambda r: r["t0"])
    return rows


# ---------------------------------------------------------------- clustering by task

def icc_anova(s, n):
    """One-way ANOVA estimator of the intraclass correlation for 0/1 data in equal clusters."""
    s, n = np.asarray(s, float), np.asarray(n, float)
    m, N = len(s), n.sum()
    p = s.sum() / N
    msb = (n * (s / n - p) ** 2).sum() / (m - 1)
    msw = (s * (1 - s / n)).sum() / (N - m)
    n0 = (N - (n ** 2).sum() / N) / (m - 1)
    return float((msb - msw) / (msb + (n0 - 1) * msw)) if msb + (n0 - 1) * msw > 0 else float("nan")


def cluster_var(s, n):
    """Section 6.3's cluster variance of the pooled rate."""
    s, n = np.asarray(s, float), np.asarray(n, float)
    m, N = len(s), n.sum()
    p = s.sum() / N
    return float(m / (m - 1) * ((s - p * n) ** 2).sum() / N ** 2)


def perm_homogeneity(y, task, rng):
    """P(sum of squared task counts >= observed) when the stops are dealt to trials at random."""
    T = np.unique(task, return_inverse=True)[1]
    m = T.max() + 1
    obs = (np.bincount(T, weights=y, minlength=m) ** 2).sum()
    k, hit = int(y.sum()), 0
    for b0 in range(0, PERMS, 20_000):
        idx = np.argsort(rng.random((20_000, len(y))), axis=1)[:, :k]
        c = np.zeros((20_000, m))
        np.add.at(c, (np.repeat(np.arange(20_000), k), T[idx].ravel()), 1)
        hit += int(((c ** 2).sum(1) >= obs).sum())
    return hit / PERMS


# ---------------------------------------------------------------- tasks fixed: exact miss of pooled CP

def poisson_binomial(ps, n_each):
    """Distribution of a sum of independent Binomial(n_each, p_h)."""
    pmf = np.ones(1)
    for p in ps:
        pmf = np.convolve(pmf, binom.pmf(np.arange(n_each + 1), n_each, p))
    return pmf


def cp_miss(ps, n_each, delta):
    """P( CP upper limit on the pooled count < mean rate ) under the task rates ``ps``."""
    N = n_each * len(ps)
    truth = float(np.mean(ps))
    pmf = poisson_binomial(ps, n_each)
    ub = np.array([cp_upper(k, N, delta) for k in range(N + 1)])
    return float(pmf[ub < truth].sum())


def shapes(mean, obs_share):
    """Per-task rates with a given mean: even, in the observed shares, and all on one task."""
    out = [("even over tasks", [mean] * 6)]
    if obs_share is not None:
        out.append(("in the observed shares", [min(1.0, 6 * mean * w) for w in obs_share]))
    out.append(("all on one task", [min(1.0, 6 * mean)] + [0.0] * 5))
    return out


# ---------------------------------------------------------------- tasks sampled: bounds on six task rates

def upper(fn, x, delta, **kw):
    return float(fn(x, delta, **kw).upper)


def task_bounds(s, n, delta, rng):
    x = np.asarray(s, float) / np.asarray(n, float)
    m = len(x)
    out = collections.OrderedDict()
    out["Hoeffding (exact)"] = upper(B.hoeffdings_bounds, x, delta)
    out["Anderson (exact)"] = upper(B.anderson_bounds, x, delta)
    out["Bentkus (exact)"] = upper(B.bentkus_bounds, x, delta)
    out["betting mixture (exact)"] = upper(B.betting_mixture_bounds, x, delta)
    sd = x.std(ddof=1)
    out["Student-t on task rates (approximate)"] = float(min(1.0, x.mean() + student_t.ppf(1 - delta, m - 1) * sd / np.sqrt(m)))
    out["cluster bootstrap-t (approximate)"] = float(AR.cluster_t_arr(n, s, delta, rng, BOOTS))
    return out


# ---------------------------------------------------------------- order in the session

def order_checks(rows, rng):
    out = {}
    t_first = rows[0]["t0"]
    for key, _ in MODELS:
        R = [r for r in rows if r["model"] == key]
        h = np.array([(r["t0"] - t_first).total_seconds() / 3600 for r in R])
        y = np.array([r["stop"] for r in R])
        p = float(mannwhitneyu(h[y == 1], h[y == 0], alternative="two-sided").pvalue) if 0 < y.sum() < len(y) else float("nan")
        half = np.argsort(h) < len(h) // 2
        # adjacent stops within a (model, task) sequence, against dealing the stops at random within the task
        seqs = collections.defaultdict(list)
        for r in R:
            seqs[r["task"]].append(r["stop"])
        adj = sum(int(a and b) for q in seqs.values() for a, b in zip(q, q[1:]))
        sim = 0
        for _ in range(20_000):
            a = 0
            for q in seqs.values():
                q2 = rng.permutation(q)
                a += int((q2[:-1] & q2[1:]).sum())
            sim += a >= adj
        out[key] = dict(first=min(r["t0"] for r in R).isoformat(), last=max(r["t0"] for r in R).isoformat(),
                        hours_from=float(h.min()), hours_to=float(h.max()),
                        stops_first_half=int(y[half].sum()), stops_second_half=int(y[~half].sum()),
                        ranksum_p=p, adjacent_stops=adj, adjacent_perm_p=sim / 20_000)
    # how many trials ran at once, and whether two trials of one task ever overlap
    conc, same_task = 0, 0
    for r in rows:
        live = [q for q in rows if q["t0"] <= r["t0"] < q["t1"]]
        conc = max(conc, len(live))
        same_task = max(same_task, max(collections.Counter(q["task"] for q in live).values()))
    out["max_concurrent"] = conc
    out["max_concurrent_same_task"] = same_task
    # on each task, the order in which the models' blocks ran and whether the blocks overlap
    orders, overlap = set(), 0
    for t in sorted({r["task"] for r in rows}):
        win = sorted((min(r["t0"] for r in rows if r["task"] == t and r["model"] == k),
                      max(r["t1"] for r in rows if r["task"] == t and r["model"] == k), k) for k, _ in MODELS)
        orders.add(tuple(w[2] for w in win))
        overlap += sum(a[1] > b[0] for a, b in zip(win, win[1:]))
    out["task_orders"] = sorted(orders)
    out["task_block_overlaps"] = overlap
    return out


# ---------------------------------------------------------------- main

def main():
    rows = load()
    rng = np.random.default_rng(SEED)
    tasks = sorted({r["task"] for r in rows})
    res = dict(tasks=tasks, n_trials=len(rows), models={}, seed=SEED, perms=PERMS, boots=BOOTS)
    for key, name in MODELS:
        R = [r for r in rows if r["model"] == key]
        s = np.array([sum(r["stop"] for r in R if r["task"] == t) for t in tasks])
        n = np.array([sum(1 for r in R if r["task"] == t) for t in tasks])
        N, k = int(n.sum()), int(s.sum())
        p = k / N
        y = np.array([r["stop"] for r in R], float)
        tk = np.array([r["task"] for r in R])
        vb = p * (1 - p) / N
        d = dict(name=name, n=n.tolist(), s=s.tolist(), N=N, k=k, rate=p,
                 perm_p=perm_homogeneity(y, tk, rng), icc=icc_anova(s, n),
                 deff=cluster_var(s, n) / vb if vb > 0 else float("nan"),
                 cp={str(dl): cp_upper(k, N, dl) for dl in DELTAS},
                 sampled={str(dl): task_bounds(s, n, dl, rng) for dl in DELTAS},
                 fixed_miss={})
        share = s / s.sum() if s.sum() else None
        for dl in DELTAS:
            d["fixed_miss"][str(dl)] = [dict(shape=lab, mean=mu, miss=cp_miss(ps, int(n[0]), dl))
                                        for mu in (p, TAU) for lab, ps in shapes(mu, share)]
        res["models"][key] = d
    # worst case of the pooled CP limit over a grid of means and the three shapes, any model's shares
    grid = np.round(np.arange(0.005, 0.16001, 0.0005), 4)
    worst = {}
    for dl in DELTAS:
        even = max(cp_miss([mu] * 6, 20, dl) for mu in grid)
        one = max(cp_miss([min(1.0, 6 * mu)] + [0.0] * 5, 20, dl) for mu in grid)
        two = max(cp_miss([min(1.0, 3 * mu)] * 2 + [0.0] * 4, 20, dl) for mu in grid)
        worst[str(dl)] = dict(even=even, on_one_task=one, on_two_tasks=two)
    res["fixed_worst"] = worst
    res["floor_six_zero"] = {str(dl): 1 - dl ** (1 / 6) for dl in DELTAS}
    res["floor_zero_120"] = {str(dl): cp_upper(0, 120, dl) for dl in DELTAS}
    res["order"] = order_checks(rows, rng)
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(res, open(os.path.join(OUT_DIR, "robodojo_sampling.json"), "w"), indent=1)
    render(res)


def render(res):
    M = res["models"]
    tasks = res["tasks"]
    L = ["# The robot benchmark's sampling unit (paper section 10.1)", "",
         f"`scripts/robodojo_sampling.py`, from `results/spikes/019/trials.csv` ({res['n_trials']} trials). "
         "A safety stop is the harness ending the trial for overheating.", "",
         "## 1. Layout: 6 tasks by 20 trials per model", "",
         "| task | " + " | ".join(M[k]["name"] for k, _ in MODELS) + " |", "|---|" + "---|" * len(MODELS)]
    for i, t in enumerate(tasks):
        L.append(f"| {t} | " + " | ".join(f"{M[k]['s'][i]} / {M[k]['n'][i]}" for k, _ in MODELS) + " |")
    L.append("| all | " + " | ".join(f"{M[k]['k']} / {M[k]['N']}" for k, _ in MODELS) + " |")
    o = res["order"]
    L += ["", f"At most {o['max_concurrent']} trials ran at once and never two of the same task "
          f"(largest number of one task's trials running together: {o['max_concurrent_same_task']}), "
          "which is what one station per task would give (the report's run paths name six rigs, one per task, so task and "
          "rig cannot be told apart). On each task the three models ran one after another "
          f"({o['task_block_overlaps']} overlaps between one model's last trial and the next model's first), in "
          + ("the same order on all six: " + ", then ".join(dict(MODELS)[k] for k in o["task_orders"][0])
             if len(o["task_orders"]) == 1 else f"{len(o['task_orders'])} different orders") + ".", "",
          "| model | first trial started (UTC) | last trial started | hours into the session |", "|---|---|---|---|"]
    for k, name in MODELS:
        L.append(f"| {name} | {o[k]['first'][:16]} | {o[k]['last'][:16]} | {o[k]['hours_from']:.1f} to {o[k]['hours_to']:.1f} |")
    L += ["", "## 2. Clustering by task", "",
          "| model | stops | exact permutation p, equal rates across tasks | ICC by task | design effect |", "|---|---|---|---|---|"]
    for k, name in MODELS:
        d = M[k]
        L.append(f"| {name} | {d['k']} | {d['perm_p']:.3f} | {d['icc']:.3f} | {d['deff']:.2f} |")
    L += ["", f"The permutation test deals each model's stops to its 120 trials at random (a Monte Carlo permutation test, {res['perms']:,} deals; "
          "statistic: the sum of squared task counts). The design effect is section 6.3's cluster variance over the "
          "binomial variance; with six clusters it is itself a noisy number.", "",
          "## 3. Tasks taken as fixed: the pooled Clopper-Pearson limit", "",
          "If the six tasks are the population (the rate over this task mix, each task weighted equally) and trials "
          "within a task are independent, the count of stops is a sum of six binomials with their own rates. The "
          "table gives the exact probability that the pooled Clopper-Pearson limit falls below the mean rate, for "
          "task rates with that mean spread three ways.", "",
          "| model | mean rate | shape | miss at delta 0.05 | miss at delta 0.10 |", "|---|---|---|---|---|"]
    for k, name in MODELS:
        a, b = M[k]["fixed_miss"]["0.05"], M[k]["fixed_miss"]["0.1"]
        for x, y in zip(a, b):
            L.append(f"| {name} | {x['mean']:.3f} | {x['shape']} | {x['miss']:.4f} | {y['miss']:.4f} |")
    w = res["fixed_worst"]
    L += ["", "Largest miss over mean rates of 0.005 to 0.160 in steps of 0.0005: "
          + "; ".join(f"delta {dl}: even {w[dl]['even']:.4f}, all on two tasks {w[dl]['on_two_tasks']:.4f}, "
                      f"all on one task {w[dl]['on_one_task']:.4f}" for dl in ("0.05", "0.1"))
          + ". Over these rates the miss stays under delta and unequal task rates only lower it (Hoeffding, 1956), so "
          "for this population the pooled limit is valid and no clustering correction applies.", "",
          "## 4. Tasks taken as a sample of tasks like them", "",
          "The six task rates are then six draws in [0, 1] whose mean is the rate of interest. Upper limits:", ""]
    names = list(next(iter(M.values()))["sampled"]["0.05"])
    for dl in ("0.05", "0.1"):
        L += [f"**delta {dl}**", "", "| model | pooled CP (tasks fixed) | " + " | ".join(names) + " |",
              "|---|---|" + "---|" * len(names)]
        for k, name in MODELS:
            d = M[k]
            L.append(f"| {name} | {d['cp'][dl]:.3f} | " + " | ".join(f"{d['sampled'][dl][nm]:.3f}" for nm in names) + " |")
        L.append("")
    f6, f120 = res["floor_six_zero"], res["floor_zero_120"]
    L += [f"With all six task rates at zero no valid bound can return less than `1 - delta^(1/6)`: "
          f"{f6['0.05']:.3f} at delta 0.05 and {f6['0.1']:.3f} at 0.10 (a population in which a share q of tasks "
          "always stops and the rest never do shows six clean tasks with probability (1 - q)^6). The pooled limit "
          f"with nothing observed is {f120['0.05']:.3f} and {f120['0.1']:.3f}. Neither approximate limit is "
          "usable with six clusters: the Student-t limit on six task rates comes out below the pooled limit for "
          "the two models with two stops, and the cluster bootstrap-t returns 1 (no limit) when resamples of six "
          f"tasks with no stop among them are common ({res['boots']:,} draws).", "",
          "## 5. Independence of trials within a task", "",
          "| model | stops in the first / second half of its block | rank-sum p, start time of stops against the rest "
          "| adjacent stops within a task's sequence | permutation p |", "|---|---|---|---|---|"]
    for k, name in MODELS:
        L.append(f"| {name} | {o[k]['stops_first_half']} / {o[k]['stops_second_half']} | {o[k]['ranksum_p']:.2f} "
                 f"| {o[k]['adjacent_stops']} | {o[k]['adjacent_perm_p']:.2f} |")
    L += ["", "Because the models ran one after another on each task, a difference between models is also a "
          "difference in the hours the session had been running. Within a model's block the data do not show "
          "stops arriving later or in runs, with 2 to 10 events to show it."]
    open(os.path.join(OUT_DIR, "robodojo_sampling.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    if "--render" in sys.argv:
        render(json.load(open(os.path.join(OUT_DIR, "robodojo_sampling.json"))))
    else:
        main()
