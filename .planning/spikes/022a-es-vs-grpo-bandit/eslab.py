"""Spike 022a: evolution strategies against GRPO on the synthetic bandit.

Every arm runs the real ``SeldonianLLMPolicy`` (predicted tests, ``LagrangianReward``, one
safety test) on ``SyntheticEnv``; only the backend's ``train`` differs. Two policies: the
bandit's own softmax-linear one (36 parameters) and a tanh net 8 -> 300 -> 300 -> 4 (about
94k weights, last layer zero so it starts at the uniform reference). See README.md for the
arms, the budget rule and the pre-registered expectations.

    ../../../.venv/bin/python eslab.py --stage check
    ../../../.venv/bin/python eslab.py --stage tune
    ../../../.venv/bin/python eslab.py --stage eval
    ../../../.venv/bin/python eslab.py --stage report
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))

from seldonian.llm.data import split_prompts  # noqa: E402
from seldonian.llm.policy import (Constraint, PolicyBackend, SeldonianLLMPolicy,  # noqa: E402
                                  predicted_width)
from seldonian.llm.rewards import LagrangianReward  # noqa: E402
from seldonian.llm.synthetic import (SyntheticEnv, SyntheticJudge, SyntheticReward,  # noqa: E402
                                     format_responses, parse_contexts, softmax)

CFG = dict(population=20000, n=1000, d=8, pressure=1.0, margin=0.03, delta=0.1,
           bound="clopper_pearson", steps=200, group_size=8, prompts_per_step=8, beta=0.01,
           predict_every=25, predict_n=512, lam0=5.0, eta=100.0, lam_max=20.0)
GENS_PER_STEP = CFG["group_size"] * CFG["prompts_per_step"]
TUNE_SEEDS = range(1000, 1020)
POP_GRID = ((4, 8), (8, 8), (8, 16), (16, 8), (30, 16), (8, 32))
# round 2 (2026-11-10): round 1's choices sat on grid edges, so every grid was widened
GRID = {
    "linear": dict(lr=(0.05, 0.1, 0.2, 0.4), sigma=(0.1, 0.3, 0.6, 1.0),
                   sigma0=(0.3, 1.0, 2.0, 4.0)),
    "mlp": dict(lr=(0.0003, 0.001, 0.003, 0.01, 0.03, 0.1), sigma=(0.01, 0.03, 0.1, 0.3)),
}
ARMS = {"linear": ("grpo", "es", "es_x8", "cma", "es_direct"),
        "mlp": ("grpo", "es", "es_x8", "es_direct")}


# ------------------------------------------------------------------ policies

class LinearPolicy:
    """``logits = x W^T + c``, the bandit's own policy; starts at zero (uniform)."""

    def __init__(self, d, n_actions, seed=0):
        self.d, self.A = d, n_actions
        self.n = n_actions * d + n_actions

    def init(self):
        return np.zeros(self.n)

    def logits(self, theta, X):
        W = theta[:self.A * self.d].reshape(self.A, self.d)
        return X @ W.T + theta[self.A * self.d:]

    def backward(self, theta, X, g_logits):
        return np.concatenate([(g_logits.T @ X).ravel(), g_logits.sum(axis=0)])


class MLPPolicy:
    """Two tanh layers of ``h`` units; the output layer starts at zero (uniform policy)."""

    def __init__(self, d, n_actions, h=300, seed=0):
        self.d, self.A, self.h, self.seed = d, n_actions, h, seed
        shapes = [(h, d), (h,), (h, h), (h,), (n_actions, h), (n_actions,)]
        sizes = [int(np.prod(s)) for s in shapes]
        self.shapes, self.cuts = shapes, np.cumsum([0] + sizes)
        self.n = int(self.cuts[-1])

    def unpack(self, theta):
        return [theta[a:b].reshape(s) for a, b, s in zip(self.cuts[:-1], self.cuts[1:],
                                                         self.shapes)]

    def init(self):
        rng = np.random.default_rng([self.seed, 22])
        theta = np.zeros(self.n)
        W1, _, W2, _, _, _ = self.unpack(theta)        # views into theta
        W1[:] = rng.standard_normal(W1.shape) / np.sqrt(self.d)
        W2[:] = rng.standard_normal(W2.shape) / np.sqrt(self.h)
        return theta

    def _forward(self, theta, X):
        W1, b1, W2, b2, W3, b3 = self.unpack(theta)
        h1 = np.tanh(X @ W1.T + b1)
        h2 = np.tanh(h1 @ W2.T + b2)
        return h1, h2, h2 @ W3.T + b3

    def logits(self, theta, X):
        return self._forward(theta, X)[2]

    def backward(self, theta, X, g_logits):
        _, _, W2, _, W3, _ = self.unpack(theta)
        h1, h2, _ = self._forward(theta, X)
        d2 = (g_logits @ W3) * (1 - h2 ** 2)
        d1 = (d2 @ W2) * (1 - h1 ** 2)
        return np.concatenate([(d1.T @ X).ravel(), d1.sum(0), (d2.T @ h1).ravel(), d2.sum(0),
                               (g_logits.T @ h2).ravel(), g_logits.sum(0)])


POLICIES = {"linear": LinearPolicy, "mlp": MLPPolicy}


class Adam:
    def __init__(self, n, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m, self.v, self.t = np.zeros(n), np.zeros(n), 0

    def step(self, g):
        """The descent step for gradient ``g`` (subtract it)."""
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        return self.lr * (self.m / (1 - self.b1 ** self.t)) / (
            np.sqrt(self.v / (1 - self.b2 ** self.t)) + self.eps)


# ------------------------------------------------------------------ backends

class LabBackend(PolicyBackend):
    """Shared sampling, ground truth and bookkeeping; subclasses implement ``train``."""

    def __init__(self, env, pol, max_steps, beta, seed, budget_mult=1):
        self.env, self.pol, self.max_steps, self.beta = env, pol, max_steps, beta
        self.budget = budget_mult * max_steps * GENS_PER_STEP
        self.rng = np.random.default_rng(seed)
        self.theta = pol.init()
        self.ref = self.theta.copy()
        self.trace = []           # (equivalent step, true rate) at every predicted test
        self.gens = 0

    def pi(self, theta, idx):
        return softmax(self.pol.logits(theta, self.env.X[idx]))

    def draw(self, theta, idx, u_a, u_v):
        """Actions and violation labels from fixed uniforms (common random numbers)."""
        cdf = np.cumsum(self.pi(theta, idx), axis=1)
        a = np.minimum((u_a[:, None] > cdf).sum(axis=1), self.env.n_actions - 1)
        return a, (u_v < self.env.p_v[idx, a]).astype(np.int64)

    def kl(self, theta, idx):
        p = self.pi(theta, idx)
        return float(np.mean(np.sum(p * (np.log(p) - np.log(self.pi(self.ref, idx))), axis=1)))

    def true_rate(self, theta=None):
        idx = np.arange(self.env.n_contexts)
        p = self.pi(self.theta if theta is None else theta, idx)
        return float(np.mean(np.sum(p * self.env.p_v, axis=1)))

    def true_reward(self, theta=None):
        idx = np.arange(self.env.n_contexts)
        p = self.pi(self.theta if theta is None else theta, idx)
        return float(np.mean(np.sum(p * self.env.mean_reward, axis=1)))

    def generate(self, prompts, max_new_tokens=256, temperature=1.0):
        idx = parse_contexts(prompts)
        return format_responses(*self.draw(self.theta, idx, self.rng.random(len(idx)),
                                           self.rng.random(len(idx))))

    def save_checkpoint(self, tag):
        return self.theta.copy()

    def load_checkpoint(self, handle):
        self.theta = handle.copy()

    def tick(self, equiv, on_step):
        if equiv % CFG["predict_every"] == 0:
            self.trace.append((equiv, self.true_rate()))
        on_step(equiv)

    def setup(self, records):
        self.idx_all = parse_contexts([r["prompt"] for r in records])
        self.group_of = {int(i): r.get("group") for i, r in zip(self.idx_all, records)}

    def score(self, reward, idx, actions, v):
        prompts = [f"ctx:{int(i)}" for i in idx]
        return np.asarray(reward(prompts, format_responses(actions, v),
                                 groups=[self.group_of[int(i)] for i in idx],
                                 references=[None] * len(idx)), dtype=float)


class GRPOBackend(LabBackend):
    """``SyntheticBackend.train`` for any policy: group-normalised advantages, Adam, KL."""

    def __init__(self, *a, lr=0.05, **kw):
        super().__init__(*a, **kw)
        self.opt = Adam(self.pol.n, lr)

    def loss_grad(self, theta, idx, actions, adv):
        """Loss ``-mean(A log pi_a) + beta mean KL(pi || ref)`` and its gradient."""
        X = self.env.X[idx]
        p = softmax(self.pol.logits(theta, X))
        rows = np.arange(len(idx))
        onehot = np.zeros_like(p)
        onehot[rows, actions] = 1.0
        diff = np.log(p) - np.log(self.pi(self.ref, idx))
        kl = np.sum(p * diff, axis=1, keepdims=True)
        g_logits = (-(adv[:, None] * (onehot - p)) + self.beta * p * (diff - kl)) / len(idx)
        loss = -np.mean(adv * np.log(p[rows, actions])) + self.beta * float(kl.mean())
        return loss, self.pol.backward(theta, X, g_logits)

    def train(self, records, reward, on_step):
        self.setup(records)
        k, G = CFG["prompts_per_step"], CFG["group_size"]
        for step in range(1, self.max_steps + 1):
            idx = self.idx_all[np.repeat(self.rng.choice(len(records), k, replace=False), G)]
            a, v = self.draw(self.theta, idx, self.rng.random(len(idx)),
                             self.rng.random(len(idx)))
            r = self.score(reward, idx, a, v).reshape(k, G)
            adv = (r - r.mean(1, keepdims=True)) / (r.std(1, ddof=1, keepdims=True) + 1e-8)
            self.theta = self.theta - self.opt.step(self.loss_grad(self.theta, idx, a,
                                                                   adv.ravel())[1])
            self.gens += k * G
            self.tick(step, on_step)


class ESBackend(LabBackend):
    """
    Antithetic ES with centred-rank fitness and Adam. One generation draws ``m`` prompts and
    one set of uniforms, shared by all ``pop`` members. ``direct`` (set by the runner) swaps
    the shaped reward for the predicted-upper-bound barrier.
    """

    def __init__(self, *a, lr=0.05, sigma=0.1, pop=16, m=8, **kw):
        super().__init__(*a, **kw)
        self.sigma, self.pop, self.m = sigma, pop - pop % 2, m
        self.opt = Adam(self.pol.n, lr)
        self.direct = None

    def fitness(self, thetas, reward, records):
        """Fitness of each parameter vector on one shared draw of prompts and uniforms."""
        idx = self.idx_all[self.rng.choice(len(records), self.m, replace=False)]
        u_a, u_v = self.rng.random(self.m), self.rng.random(self.m)
        draws = [self.draw(t, idx, u_a, u_v) for t in thetas]
        a = np.concatenate([d[0] for d in draws])
        v = np.concatenate([d[1] for d in draws])
        r = self.score(reward, np.tile(idx, len(thetas)), a, v).reshape(len(thetas), self.m)
        f = r.mean(axis=1)
        if self.beta:
            f = f - self.beta * np.array([self.kl(t, idx) for t in thetas])
        if self.direct is not None:
            k = v.reshape(len(thetas), self.m).sum(axis=1)
            g = k / self.m + self.direct["width"][k] - self.direct["threshold"]
            f = np.where(g <= 0, f, -1e6 - g)
        self.gens += len(thetas) * self.m
        return f

    def advance(self, on_step):
        while self.equiv < self.budget // GENS_PER_STEP and \
                self.gens >= GENS_PER_STEP * (self.equiv + 1):
            self.equiv += 1
            self.tick(self.equiv, on_step)

    def train(self, records, reward, on_step):
        self.setup(records)
        self.equiv, half = 0, self.pop // 2
        while self.gens < self.budget:
            eps = self.rng.standard_normal((half, self.pol.n))
            thetas = [self.theta + s * self.sigma * e for e in eps for s in (1.0, -1.0)]
            f = self.fitness(thetas, reward, records)
            ranks = np.empty(self.pop)
            ranks[np.argsort(f, kind="stable")] = np.arange(self.pop)
            w = (ranks / (self.pop - 1) - 0.5).reshape(half, 2)
            grad = (w[:, 0] - w[:, 1]) @ eps / (self.pop * self.sigma)
            self.theta = self.theta + self.opt.step(grad)       # ascent
            self.advance(on_step)


class CMABackend(ESBackend):
    """pycma on the same fitness (full covariance; the 36-parameter policy only)."""

    def __init__(self, *a, sigma0=0.3, pop=16, m=8, seed=0, **kw):
        super().__init__(*a, pop=pop, m=m, seed=seed, **kw)
        self.sigma0, self.cma_seed = sigma0, seed

    def train(self, records, reward, on_step):
        import cma
        self.setup(records)
        self.equiv = 0
        es = cma.CMAEvolutionStrategy(self.theta, self.sigma0, dict(
            popsize=self.pop, seed=self.cma_seed + 1, verbose=-9, tolfun=0, tolfunhist=0,
            tolx=0, tolstagnation=10 ** 9, tolflatfitness=10 ** 9, maxiter=10 ** 9))
        while self.gens < self.budget:
            xs = es.ask()
            f = self.fitness([np.asarray(x) for x in xs], reward, records)
            es.tell(xs, list(-f))
            self.theta = np.asarray(es.mean).copy()
            self.advance(on_step)


# ------------------------------------------------------------------ one run

def make_backend(arm, env, pol, seed, hp):
    kw = dict(max_steps=CFG["steps"], beta=CFG["beta"], seed=seed + 2)
    if arm == "grpo":
        return GRPOBackend(env, pol, lr=hp["lr"], **kw)
    if arm == "cma":
        return CMABackend(env, pol, sigma0=hp["sigma0"], pop=hp["pop"], m=hp["m"], **kw)
    return ESBackend(env, pol, lr=hp["lr"], sigma=hp["sigma"], pop=hp["pop"], m=hp["m"],
                     budget_mult=8 if arm == "es_x8" else 1, **kw)


def run(seed, policy, arm, hp):
    t0 = time.time()
    env = SyntheticEnv(n_contexts=CFG["population"], d=CFG["d"], pressure=CFG["pressure"],
                       seed=seed)
    d_c, d_s = split_prompts(env.records(CFG["n"], seed=seed), test_size=0.4, seed=seed)
    pol = POLICIES[policy](env.d, env.n_actions, seed=seed)
    backend = make_backend(arm, env, pol, seed, hp)
    judge = SyntheticJudge(1.0, 1.0, seed=seed)
    base = SyntheticReward(env, seed=seed + 1)
    ref_rate, ref_reward = backend.true_rate(), backend.true_reward()
    threshold = ref_rate + CFG["margin"]
    c = Constraint("harm", judge, threshold=threshold, group=None, bound=CFG["bound"])
    if arm == "es_direct":
        reward = base
        ks = np.arange(backend.m + 1)
        backend.direct = dict(threshold=threshold, width=np.array([
            predicted_width(k / backend.m, len(d_s), CFG["delta"], CFG["bound"],
                            m=CFG["predict_n"]) for k in ks]))
    else:
        reward = LagrangianReward(base, [(judge, None)], names=["harm"], lam0=CFG["lam0"],
                                  eta=CFG["eta"], lam_max=CFG["lam_max"])
    policy_obj = SeldonianLLMPolicy(backend, d_c, d_s, reward=reward, constraints=[c],
                                    delta=CFG["delta"], predict_every=CFG["predict_every"],
                                    predict_n=CFG["predict_n"], seed=seed)
    solution = policy_obj.fit(seldonian=True) is not None
    rate, rew = backend.true_rate(), backend.true_reward()
    trace = np.array([t[1] for t in backend.trace])
    lam = getattr(reward, "lambdas", {}).get("harm")
    return dict(
        seed=seed, policy=policy, arm=arm, hp=hp, solution=bool(solution),
        threshold=float(threshold), ref_rate=ref_rate, ref_reward=ref_reward,
        rate=rate, reward=rew, violation=bool(solution and rate > threshold),
        utility=float(rew - ref_reward) if solution else 0.0,
        selected=str(policy_obj.selected["step"]),
        train_over=float(np.mean(trace > threshold)) if len(trace) else None,
        trace=[round(float(x), 4) for x in trace], lam_final=None if lam is None else float(lam),
        kl=backend.kl(backend.theta, np.arange(env.n_contexts)),
        dist=float(np.linalg.norm(backend.theta - backend.ref)), gens=int(backend.gens),
        seconds=round(time.time() - t0, 2))


def _run(job):
    return run(*job)


def sweep(jobs, workers):
    t0, out = time.time(), []
    with ProcessPoolExecutor(workers) as ex:
        for i, row in enumerate(ex.map(_run, jobs, chunksize=4), 1):
            out.append(row)
            if i % 200 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} runs, {time.time() - t0:.0f} s", flush=True)
    return out


# ------------------------------------------------------------------ stages

def grid(policy, arm):
    g = GRID[policy]
    if arm == "grpo":
        return [dict(lr=lr) for lr in g["lr"]]
    if arm == "cma":
        return [dict(sigma0=s, pop=p, m=m) for s in g["sigma0"] for p, m in POP_GRID]
    return [dict(lr=lr, sigma=s, pop=p, m=m)
            for lr in g["lr"] for s in g["sigma"] for p, m in POP_GRID]


def stage_check():
    """Finite differences for both policies' gradients; linear GRPO against the project's."""
    env = SyntheticEnv(n_contexts=2000, seed=3)
    rng = np.random.default_rng(0)
    for name in POLICIES:
        pol = POLICIES[name](env.d, env.n_actions, seed=3)
        b = GRPOBackend(env, pol, max_steps=1, beta=0.3, seed=0)
        theta = b.theta + 0.1 * rng.standard_normal(pol.n)
        idx = np.repeat(rng.choice(2000, 8, replace=False), 8)
        a, _ = b.draw(theta, idx, rng.random(64), rng.random(64))
        adv = rng.standard_normal(64)
        _, g = b.loss_grad(theta, idx, a, adv)
        worst = 0.0
        for j in rng.choice(pol.n, min(pol.n, 40), replace=False):
            e = np.zeros(pol.n)
            e[j] = 1e-5
            fd = (b.loss_grad(theta + e, idx, a, adv)[0]
                  - b.loss_grad(theta - e, idx, a, adv)[0]) / 2e-5
            worst = max(worst, abs(fd - g[j]))
        print(f"{name}: n = {pol.n}, max |finite difference - gradient| = {worst:.2e}")
        assert worst < 1e-6
    from seldonian.llm.synthetic import SyntheticBackend
    pol = LinearPolicy(env.d, env.n_actions)
    b = GRPOBackend(env, pol, max_steps=1, beta=0.3, seed=0)
    sb = SyntheticBackend(env, beta=0.3)
    sb.W, sb.c = rng.standard_normal((4, 8)) * 0.1, rng.standard_normal(4) * 0.1
    theta = np.concatenate([sb.W.ravel(), sb.c])
    idx = np.repeat(rng.choice(2000, 8, replace=False), 8)
    a, _ = b.draw(theta, idx, rng.random(64), rng.random(64))
    adv = rng.standard_normal(64)
    gW, gc = sb.gradient(idx, a, adv)
    # the project backend's reference is its own start (zeros), as here
    gap = np.abs(np.concatenate([gW.ravel(), gc]) - b.loss_grad(theta, idx, a, adv)[1]).max()
    print(f"linear GRPO gradient against SyntheticBackend.gradient: max gap {gap:.2e}")
    assert gap < 1e-12
    print("CHECK OK")


def utility_table(rows):
    best = {}
    for r in rows:
        best.setdefault((r["policy"], r["arm"], json.dumps(r["hp"], sort_keys=True)),
                        []).append(r["utility"])
    return {k: float(np.mean(v)) for k, v in best.items()}


def stage_tune(workers):
    jobs = [(s, policy, arm, hp) for policy in ARMS for arm in ARMS[policy]
            if arm != "es_x8" for hp in grid(policy, arm) for s in TUNE_SEEDS]
    print(f"tuning: {len(jobs)} runs")
    rows = sweep(jobs, workers)
    table = utility_table(rows)
    chosen = {}
    for (policy, arm, hp), u in sorted(table.items(), key=lambda kv: -kv[1]):
        chosen.setdefault(f"{policy}/{arm}", dict(hp=json.loads(hp), utility=u))
    for policy in ARMS:
        chosen[f"{policy}/es_x8"] = chosen[f"{policy}/es"]
    json.dump(dict(chosen=chosen, table={"|".join(k): v for k, v in table.items()}),
              open(os.path.join(HERE, "tune.json"), "w"), indent=1)
    for k, v in sorted(chosen.items()):
        print(f"{k:18s} {v['hp']}  tuning utility {v['utility']:.4f}")


def stage_eval(workers, seeds):
    chosen = json.load(open(os.path.join(HERE, "tune.json")))["chosen"]
    jobs = [(s, policy, arm, chosen[f"{policy}/{arm}"]["hp"])
            for policy in ARMS for arm in ARMS[policy] for s in range(seeds)]
    print(f"evaluation: {len(jobs)} runs")
    rows = sweep(jobs, workers)
    json.dump(rows, open(os.path.join(HERE, "results.json"), "w"))


def stage_report():
    rows = json.load(open(os.path.join(HERE, "results.json")))
    chosen = json.load(open(os.path.join(HERE, "tune.json")))["chosen"]
    out = ["# Spike 022a results", "",
           "Seeds 0-59, paired across arms. Utility = true reward of the returned policy minus",
           "the reference policy's, No Solution Found = 0. `se` is the standard error over seeds;",
           "the last two columns are the paired difference from GRPO.", ""]
    for policy in ARMS:
        n_par = POLICIES[policy](CFG["d"], 4).n
        out += [f"## {policy} policy ({n_par:,} parameters)", "",
                "| arm | solutions | over threshold | utility (se) | checkpoints over | "
                "final multiplier | KL to ref | utility - GRPO (se) | % of GRPO |",
                "|---|---|---|---|---|---|---|---|---|"]
        by = {a: sorted((r for r in rows if r["policy"] == policy and r["arm"] == a),
                        key=lambda r: r["seed"]) for a in ARMS[policy]}
        u0 = np.array([r["utility"] for r in by["grpo"]])
        for a in ARMS[policy]:
            R = by[a]
            n = len(R)
            u = np.array([r["utility"] for r in R])
            sol = sum(r["solution"] for r in R)
            viol = sum(r["violation"] for r in R)
            lam = [r["lam_final"] for r in R if r["lam_final"] is not None]
            d = u - u0
            out.append(
                f"| {a} | {sol}/{n} | {viol}/{sol} | {u.mean():.4f} ({u.std(ddof=1) / n ** .5:.4f}) "
                f"| {np.mean([r['train_over'] for r in R]):.3f} "
                f"| {f'{np.mean(lam):.1f}' if lam else '-'} |"
                + f" {np.mean([r['kl'] for r in R]):.4f} "
                + (f"| {d.mean():+.4f} ({d.std(ddof=1) / n ** .5:.4f}) "
                   f"| {100 * u.mean() / u0.mean():.0f}% |" if a != "grpo" else "| | |"))
        out += ["", "Tuned settings: " + "; ".join(
            f"`{a}` {chosen[f'{policy}/{a}']['hp']}" for a in ARMS[policy]), ""]
    open(os.path.join(HERE, "results.md"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("check", "tune", "eval", "report"))
    ap.add_argument("--workers", type=int, default=os.cpu_count() // 2)
    ap.add_argument("--seeds", type=int, default=60)
    args = ap.parse_args()
    if args.stage == "check":
        stage_check()
    elif args.stage == "tune":
        stage_tune(args.workers)
    elif args.stage == "eval":
        stage_eval(args.workers, args.seeds)
    else:
        stage_report()
