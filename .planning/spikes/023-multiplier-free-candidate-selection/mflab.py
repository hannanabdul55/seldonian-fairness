"""Spike 023: candidate selection without LagrangianReward's dual-ascent multiplier.

Every arm runs the real ``SeldonianLLMPolicy`` (predicted tests every 25 steps, the
best-feasible-checkpoint rule, one safety test) on 022a's bandit and policies; only the way
training is steered differs. See README.md for the arms, the per-step bound surrogate, the
pre-registered expectations and the decision rule (registered in commit ff15b31 before any run).

    ../../../.venv/bin/python mflab.py --stage check
    ../../../.venv/bin/python mflab.py --stage tune
    ../../../.venv/bin/python mflab.py --stage eval
    ../../../.venv/bin/python mflab.py --stage report
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
from scipy.special import betaincinv  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LAB022 = os.path.join(HERE, "..", "022a-es-vs-grpo-bandit")
sys.path.insert(0, LAB022)

import eslab  # noqa: E402
from eslab import CFG, POLICIES, GRPOBackend, softmax  # noqa: E402
from seldonian.bounds import clopper_pearson_bounds  # noqa: E402
from seldonian.llm.data import split_prompts  # noqa: E402
from seldonian.llm.policy import Constraint, SeldonianLLMPolicy, effective_n  # noqa: E402
from seldonian.llm.rewards import LagrangianReward  # noqa: E402
from seldonian.llm.synthetic import SyntheticEnv, SyntheticJudge, SyntheticReward  # noqa: E402

EMA = 0.2
TUNE_SEEDS = range(1000, 1020)
LR = {"linear": (0.05, 0.1, 0.2, 0.4, 0.8), "mlp": (0.0001, 0.0003, 0.001, 0.003)}
ARMS = ("lagrangian", "select", "feas_first", "projected", "smooth_pen", "es_direct")


def grid(policy, arm):
    lrs = LR[policy]
    if arm == "lagrangian":
        return [dict(lr=lr, eta=e, lam0=l0) for lr in lrs for e in (30.0, 100.0, 300.0)
                for l0 in (1.0, 5.0)]
    if arm in ("select", "feas_first"):
        return [dict(lr=lr) for lr in lrs]
    if arm == "projected":
        return [dict(lr=lr, eta_p=e) for lr in lrs for e in (1.0, 10.0, 100.0)]
    if arm == "smooth_pen":
        return [dict(lr=lr, c0=c0, T=T) for lr in lrs for c0 in (5.0, 20.0, 50.0)
                for T in (0.005, 0.02)]
    if arm == "lagrangian_step":       # spike 023b
        return [dict(lr=lr, eta_s=e, lam0=l0) for lr in lrs for e in (1.0, 10.0, 100.0)
                for l0 in (1.0, 5.0)]
    if arm == "es_direct":
        chosen = json.load(open(os.path.join(LAB022, "tune.json")))["chosen"]
        return [chosen[f"{policy}/es_direct"]["hp"]]
    raise ValueError(arm)


# ------------------------------------------------------------------ the bound surrogate

class BoundSurrogate:
    """
    The predicted test's Clopper-Pearson upper limit as a smooth function of a running rate.
    ``U(r) = betaincinv(k + 1, n - k, 1 - delta)`` with real ``k = r n`` is exactly what
    ``clopper_pearson_bounds`` computes for a sample with mean ``r`` evaluated at size ``n``,
    and it is differentiable in ``r`` on ``[0, 1)``.
    """

    def __init__(self, threshold, n, delta, a=EMA):
        self.threshold, self.n, self.delta, self.a = threshold, n, delta, a
        self.rate = None

    def U(self, r):
        k = np.clip(r, 0.0, 1.0 - 1e-9) * self.n
        return float(betaincinv(k + 1.0, self.n - k, 1.0 - self.delta))

    def dU(self, r, h=1e-6):
        lo, hi = max(r - h, 0.0), min(r + h, 1.0 - 2e-9)
        return (self.U(hi) - self.U(lo)) / (hi - lo)

    def update(self, v):
        m = float(np.mean(v))
        self.rate = m if self.rate is None else (1 - self.a) * self.rate + self.a * m
        return self.c

    @property
    def c(self):
        return self.U(self.rate) - self.threshold


def projected_direction(g_r, g_c, c, eta_p):
    """Ascent direction ``g_r - k g_c`` with ``k = max(0, (<g_r, g_c> + eta_p c) / |g_c|^2)``."""
    gg = float(g_c @ g_c)
    if gg < 1e-18:
        return g_r, 0.0
    k = max(0.0, (float(g_r @ g_c) + eta_p * c) / gg)
    return g_r - k * g_c, k


# ------------------------------------------------------------------ the backend

class MFBackend(GRPOBackend):
    """022a's GRPO backend steered by the bound surrogate instead of a shaped reward."""

    def __init__(self, *a, mode, hp, surrogate, **kw):
        super().__init__(*a, lr=hp["lr"], **kw)
        self.mode, self.hp, self.sur = mode, hp, surrogate
        self.infeasible_steps = 0
        self.kappas = []
        self.lam = hp.get("lam0", 0.0)
        self.lam_dual = 0.0

    def ascent_grad(self, theta, idx, actions, adv):
        """Gradient of ``mean(adv * log pi_a)`` (no KL)."""
        X = self.env.X[idx]
        p = softmax(self.pol.logits(theta, X))
        onehot = np.zeros_like(p)
        onehot[np.arange(len(idx)), actions] = 1.0
        return self.pol.backward(theta, X, adv[:, None] * (onehot - p) / len(idx))

    def kl_grad(self, theta, idx):
        X = self.env.X[idx]
        p = softmax(self.pol.logits(theta, X))
        diff = np.log(p) - np.log(self.pi(self.ref, idx))
        kl = np.sum(p * diff, axis=1, keepdims=True)
        return self.pol.backward(theta, X, self.beta * p * (diff - kl) / len(idx))

    def train(self, records, reward, on_step):
        self.setup(records)
        k, G = CFG["prompts_per_step"], CFG["group_size"]
        for step in range(1, self.max_steps + 1):
            idx = self.idx_all[np.repeat(self.rng.choice(len(records), k, replace=False), G)]
            a, v = self.draw(self.theta, idx, self.rng.random(len(idx)),
                             self.rng.random(len(idx)))
            r = self.score(reward, idx, a, v).reshape(k, G)
            vb = v.reshape(k, G).astype(float)
            if self.sur.rate is None:          # first step: start the running rate
                self.sur.update(v)
            c = self.sur.c                     # predictable: from steps before this one
            if c > 0:
                self.infeasible_steps += 1
            if self.mode == "projected":
                adv_r = (r - r.mean(1, keepdims=True)) / (r.std(1, ddof=1, keepdims=True) + 1e-8)
                adv_c = vb - vb.mean(1, keepdims=True)
                g_r = self.ascent_grad(self.theta, idx, a, adv_r.ravel())
                g_c = self.ascent_grad(self.theta, idx, a, adv_c.ravel())
                d, kap = projected_direction(g_r, g_c, c, self.hp["eta_p"])
                self.kappas.append(kap)
                grad = -d + self.kl_grad(self.theta, idx)
            else:
                if self.mode == "select":
                    shaped = r
                elif self.mode == "feas_first":
                    shaped = r if c <= 0 else -vb
                elif self.mode == "lagrangian_step":
                    # spike 023b: dual ascent at every step on the same surrogate
                    self.lam = float(np.clip(self.lam + self.hp["eta_s"] * c, 0.0,
                                             CFG["lam_max"]))
                    self.kappas.append(self.lam)
                    shaped = r - self.lam * vb
                elif self.mode == "pi":
                    # spike 023c: smooth penalty plus a slow per-step dual (integral) term
                    self.lam_dual = float(np.clip(self.lam_dual + self.hp["eta_i"] * c, 0.0, 200.0))
                    lam = self.lam_dual + self.hp["c0"] * self.sur.dU(self.sur.rate) / (
                        1.0 + np.exp(-c / self.hp["T"]))
                    self.kappas.append(lam)
                    shaped = r - lam * vb
                elif self.mode == "smooth_pen":
                    lam = self.hp["c0"] * self.sur.dU(self.sur.rate) / (
                        1.0 + np.exp(-c / self.hp["T"]))
                    self.kappas.append(lam)
                    shaped = r - lam * vb
                else:
                    raise ValueError(self.mode)
                adv = (shaped - shaped.mean(1, keepdims=True)) / (
                    shaped.std(1, ddof=1, keepdims=True) + 1e-8)
                grad = self.loss_grad(self.theta, idx, a, adv.ravel())[1]
            self.theta = self.theta - self.opt.step(grad)
            self.sur.update(v)
            self.gens += k * G
            self.tick(step, on_step)


# ------------------------------------------------------------------ one run

class ScaledReward:
    """Base reward times a constant (spike 023c); ``base`` lets reporting unwrap it."""

    def __init__(self, base, scale):
        self.base, self.scale, self.name = base, float(scale), f"scaled{scale}:" + base.name

    def __call__(self, *a, **kw):
        return self.scale * np.asarray(self.base(*a, **kw), dtype=float)


def run(seed, policy, arm, hp):
    if arm == "es_direct":
        row = eslab.run(seed, policy, "es_direct", hp)
        row.update(pred_pass_safety_fail=bool(row["selected"] != "final" and not row["solution"]),
                   infeasible_share=None, mean_kappa=None)
        return row
    t0 = time.time()
    env = SyntheticEnv(n_contexts=CFG["population"], d=CFG["d"], pressure=CFG["pressure"],
                       seed=seed)
    d_c, d_s = split_prompts(env.records(CFG["n"], seed=seed), test_size=0.4, seed=seed)
    pol = POLICIES[policy](env.d, env.n_actions, seed=seed)
    kw = dict(max_steps=CFG["steps"], beta=CFG["beta"], seed=seed + 2)
    judge = SyntheticJudge(1.0, 1.0, seed=seed)
    base = SyntheticReward(env, seed=seed + 1)
    if hp.get("scale", 1.0) != 1.0:
        base = ScaledReward(base, hp["scale"])
    if arm == "lagrangian":
        backend = GRPOBackend(env, pol, lr=hp["lr"], **kw)
    else:
        backend = MFBackend(env, pol, mode=arm, hp=hp, surrogate=None, **kw)
    ref_rate, ref_reward = backend.true_rate(), backend.true_reward()
    threshold = ref_rate + CFG["margin"]
    if arm != "lagrangian":
        backend.sur = BoundSurrogate(threshold, effective_n(CFG["predict_n"], len(d_s)),
                                     CFG["delta"])
        reward = base
    else:
        reward = LagrangianReward(base, [(judge, None)], names=["harm"], lam0=hp["lam0"],
                                  eta=hp["eta"], lam_max=CFG["lam_max"])
    c = Constraint("harm", judge, threshold=threshold, group=None, bound=CFG["bound"])
    policy_obj = SeldonianLLMPolicy(backend, d_c, d_s, reward=reward, constraints=[c],
                                    delta=CFG["delta"], predict_every=CFG["predict_every"],
                                    predict_n=CFG["predict_n"], seed=seed)
    solution = policy_obj.fit(seldonian=True) is not None
    rate, rew = backend.true_rate(), backend.true_reward()
    trace = np.array([t[1] for t in backend.trace])
    selected = str(policy_obj.selected["step"])
    lam = getattr(reward, "lambdas", {}).get("harm")
    kap = getattr(backend, "kappas", [])
    return dict(
        seed=seed, policy=policy, arm=arm, hp=hp, solution=bool(solution),
        threshold=float(threshold), ref_rate=ref_rate, ref_reward=ref_reward,
        rate=rate, reward=rew, violation=bool(solution and rate > threshold),
        utility=float(rew - ref_reward) if solution else 0.0, selected=selected,
        pred_pass_safety_fail=bool(selected != "final" and not solution),
        train_over=float(np.mean(trace > threshold)) if len(trace) else None,
        trace=[round(float(x), 4) for x in trace],
        lam_final=None if lam is None else float(lam),
        infeasible_share=(getattr(backend, "infeasible_steps", 0) / CFG["steps"]
                          if arm != "lagrangian" else None),
        mean_kappa=float(np.mean(kap)) if len(kap) else None,
        kl=backend.kl(backend.theta, np.arange(env.n_contexts)),
        gens=int(backend.gens), seconds=round(time.time() - t0, 2))


def _run(job):
    return run(*job)


def sweep(jobs, workers):
    t0, out = time.time(), []
    with ProcessPoolExecutor(workers) as ex:
        for i, row in enumerate(ex.map(_run, jobs, chunksize=2), 1):
            out.append(row)
            if i % 100 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} runs, {time.time() - t0:.0f} s", flush=True)
    return out


# ------------------------------------------------------------------ stages

def stage_check():
    rng = np.random.default_rng(0)
    # 1. the smooth bound equals the library's Clopper-Pearson upper limit, at real k too
    worst = 0.0
    for _ in range(300):
        m0 = int(rng.integers(20, 600))
        k1 = int(rng.integers(0, m0))
        n = int(rng.integers(50, 600))
        x = np.r_[np.ones(k1), np.zeros(m0 - k1)]
        lib = float(clopper_pearson_bounds(x, 0.1, n=n).upper)
        sur = BoundSurrogate(0.0, n, 0.1).U(k1 / m0)
        worst = max(worst, abs(lib - sur))
    print(f"smooth bound vs clopper_pearson_bounds: max |diff| {worst:.2e} over 300 cases")
    # 2. its derivative against a coarser finite difference
    s = BoundSurrogate(0.0, 205, 0.1)
    errs = [abs(s.dU(r) - (s.U(r + 1e-4) - s.U(r - 1e-4)) / 2e-4) / s.dU(r)
            for r in np.linspace(0.01, 0.6, 60)]
    print(f"dU: max relative diff vs h=1e-4 finite difference {max(errs):.2e}; "
          f"dU in [{min(s.dU(r) for r in np.linspace(0.01, .6, 60)):.3f}, "
          f"{max(s.dU(r) for r in np.linspace(0.01, .6, 60)):.3f}] on r in [0.01, 0.6]")
    # 3. the projection identity <d, g_c> = -eta_p c when the constraint is active
    worst = 0.0
    for _ in range(1000):
        g_r, g_c = rng.standard_normal(36), rng.standard_normal(36) * 0.1
        c, eta = float(rng.normal(0, 0.05)), float(10 ** rng.uniform(0, 2))
        d, k = projected_direction(g_r, g_c, c, eta)
        if k > 0:
            worst = max(worst, abs(float(d @ g_c) + eta * c))
        else:
            assert float(g_r @ g_c) + eta * c <= 1e-12
    print(f"projection identity: max |<d,g_c> + eta c| {worst:.2e} over active cases")
    # 4. one run per arm per policy, end to end
    for policy in ("linear", "mlp"):
        for arm in ARMS:
            hp = grid(policy, arm)[0]
            row = run(0, policy, arm, hp)
            print(f"  {policy:6s} {arm:11s} solution={row['solution']} "
                  f"utility={row['utility']:.3f} over={row['train_over']} "
                  f"infeasible={row['infeasible_share']} {row['seconds']} s")


def stage_tune(workers):
    jobs = [(s, p, a, hp) for p in ("linear", "mlp") for a in ARMS if a != "es_direct"
            for hp in grid(p, a) for s in TUNE_SEEDS]
    print(f"tuning: {len(jobs)} runs")
    rows = sweep(jobs, workers)
    chosen, sens = {}, {}
    for p in ("linear", "mlp"):
        for a in ARMS:
            if a == "es_direct":
                chosen[f"{p}/{a}"] = dict(hp=grid(p, a)[0], utility=None)
                continue
            by = {}
            for r in rows:
                if r["policy"] == p and r["arm"] == a:
                    by.setdefault(json.dumps(r["hp"], sort_keys=True), []).append(r["utility"])
            means = {k: float(np.mean(v)) for k, v in by.items()}
            best = max(means, key=means.get)
            chosen[f"{p}/{a}"] = dict(hp=json.loads(best), utility=means[best])
            vals = np.array(list(means.values()))
            sens[f"{p}/{a}"] = dict(n_configs=len(vals), median=float(np.median(vals)),
                                    worst=float(vals.min()), best=float(vals.max()))
    json.dump(dict(chosen=chosen, sensitivity=sens, rows=rows),
              open(os.path.join(HERE, "tune.json"), "w"))
    print(json.dumps(dict(chosen=chosen, sensitivity=sens), indent=1))


def stage_eval(workers, seeds):
    chosen = json.load(open(os.path.join(HERE, "tune.json")))["chosen"]
    jobs = [(s, p, a, chosen[f"{p}/{a}"]["hp"]) for p in ("linear", "mlp") for a in ARMS
            for s in seeds]
    print(f"evaluation: {len(jobs)} runs")
    rows = sweep(jobs, workers)
    json.dump(rows, open(os.path.join(HERE, "results.json"), "w"))


def stage_tune_b(workers):
    jobs = [(s, p, "lagrangian_step", hp) for p in ("linear", "mlp")
            for hp in grid(p, "lagrangian_step") for s in TUNE_SEEDS]
    print(f"023b tuning: {len(jobs)} runs")
    rows = sweep(jobs, workers)
    chosen, sens = {}, {}
    for p in ("linear", "mlp"):
        by = {}
        for r in rows:
            if r["policy"] == p:
                by.setdefault(json.dumps(r["hp"], sort_keys=True), []).append(r["utility"])
        means = {k: float(np.mean(v)) for k, v in by.items()}
        best = max(means, key=means.get)
        chosen[p] = dict(hp=json.loads(best), utility=means[best])
        vals = np.array(list(means.values()))
        sens[p] = dict(n_configs=len(vals), median=float(np.median(vals)),
                       worst=float(vals.min()), best=float(vals.max()))
    json.dump(dict(chosen=chosen, sensitivity=sens, rows=rows),
              open(os.path.join(HERE, "tune_b.json"), "w"))
    print(json.dumps(dict(chosen=chosen, sensitivity=sens), indent=1))


def stage_eval_b(workers, seeds):
    chosen = json.load(open(os.path.join(HERE, "tune_b.json")))["chosen"]
    jobs = [(s, p, "lagrangian_step", chosen[p]["hp"]) for p in ("linear", "mlp") for s in seeds]
    rows = sweep(jobs, workers)
    json.dump(rows, open(os.path.join(HERE, "results_b.json"), "w"))


def stage_report_b():
    tb = json.load(open(os.path.join(HERE, "tune_b.json")))
    rows = json.load(open(os.path.join(HERE, "results.json"))) + \
        json.load(open(os.path.join(HERE, "results_b.json")))
    out = ["# Spike 023b results", "", "Paired on seeds 0-59 with spike 023's rows.", "",
           "| policy | arm | solutions | over threshold | utility (se) | ckpts over | "
           "smooth_pen minus this (se) | grid median / worst |", "|---|---|---|---|---|---|---|---|"]
    for p in ("linear", "mlp"):
        sp = {r["seed"]: r["utility"] for r in rows if r["policy"] == p and r["arm"] == "smooth_pen"}
        for a in ("lagrangian", "lagrangian_step", "smooth_pen"):
            rs = [r for r in rows if r["policy"] == p and r["arm"] == a]
            u = np.array([r["utility"] for r in rs])
            d = np.array([sp[r["seed"]] - r["utility"] for r in rs])
            over = np.mean([r["train_over"] for r in rs])
            sens = tb["sensitivity"][p] if a == "lagrangian_step" else None
            out.append(f"| {p} | `{a}` | {sum(r['solution'] for r in rs)}/{len(rs)} | "
                       f"{sum(r['violation'] for r in rs)} | {u.mean():.3f} "
                       f"({u.std(ddof=1) / np.sqrt(len(u)):.3f}) | {100 * over:.1f}% | "
                       f"{d.mean():+.3f} ({d.std(ddof=1) / np.sqrt(len(d)):.3f}) | "
                       + (f"{sens['median']:.3f} / {sens['worst']:.3f}" if sens else "see 023")
                       + " |")
    out += ["", f"Chosen: `{json.dumps({p: tb['chosen'][p]['hp'] for p in tb['chosen']})}`"]
    open(os.path.join(HERE, "results_b.md"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))


C_SCALES = (1.0, 3.0, 10.0)


def stage_c(workers):
    """Spike 023c: tune eta_i at s = 1, then every arm at s in C_SCALES on seeds 0-59."""
    ch = json.load(open(os.path.join(HERE, "tune.json")))["chosen"]
    sp = ch["linear/smooth_pen"]["hp"]
    tune_jobs = [(s, "linear", "pi", dict(sp, eta_i=e)) for e in (0.3, 1.0, 3.0, 10.0)
                 for s in TUNE_SEEDS]
    trows = sweep(tune_jobs, workers)
    by = {}
    for r in trows:
        by.setdefault(r["hp"]["eta_i"], []).append(r["utility"])
    eta = max(by, key=lambda e: np.mean(by[e]))
    hps = {"lagrangian": ch["linear/lagrangian"]["hp"], "smooth_pen": sp,
           "pi": dict(sp, eta_i=eta)}
    jobs = [(s, "linear", a, dict(hp, scale=sc)) for sc in C_SCALES for a, hp in hps.items()
            for s in range(60)]
    rows = sweep(jobs, workers)
    json.dump(dict(tune={str(k): float(np.mean(v)) for k, v in by.items()}, eta_i=eta,
                   rows=rows), open(os.path.join(HERE, "results_c.json"), "w"))
    out = ["# Spike 023c results", "",
           f"Linear policy, seeds 0-59. pi's eta_i = {eta} (tuning means "
           f"{ {k: round(float(np.mean(v)), 3) for k, v in by.items()} }).", "",
           "| reward scale | arm | solutions | over threshold | utility (se) | ckpts over | "
           "mean penalty weight | final multiplier |", "|---|---|---|---|---|---|---|---|"]
    for sc in C_SCALES:
        for a in hps:
            rs = [r for r in rows if r["arm"] == a and r["hp"]["scale"] == sc]
            u = np.array([r["utility"] for r in rs])
            mk = [r["mean_kappa"] for r in rs if r["mean_kappa"] is not None]
            lf = [r["lam_final"] for r in rs if r["lam_final"] is not None]
            out.append(f"| {sc:g} | `{a}` | {sum(r['solution'] for r in rs)}/60 | "
                       f"{sum(r['violation'] for r in rs)} | {u.mean():.3f} "
                       f"({u.std(ddof=1) / np.sqrt(60):.3f}) | "
                       f"{100 * np.mean([r['train_over'] for r in rs]):.1f}% | "
                       f"{np.mean(mk) if mk else float('nan'):.2f} | "
                       f"{np.mean(lf) if lf else float('nan'):.2f} |")
    open(os.path.join(HERE, "results_c.md"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))


def stage_report():
    tune = json.load(open(os.path.join(HERE, "tune.json")))
    rows = json.load(open(os.path.join(HERE, "results.json")))
    out = ["# Spike 023 results", "",
           "60 paired evaluation seeds (0-59) per arm and policy; utility = true reward of the "
           "returned policy minus the reference's, No Solution Found = 0. Paired differences "
           "are against `lagrangian` on the same seeds.", ""]
    for p in ("linear", "mlp"):
        base = {r["seed"]: r for r in rows if r["policy"] == p and r["arm"] == "lagrangian"}
        u_base = np.mean([r["utility"] for r in base.values()])
        out += [f"## {p}", "",
                "| arm | hp | solutions | over threshold | utility (se) | % of lagrangian | "
                "paired diff (se) | train ckpts over | pred pass, safety fail | "
                "grid median / worst (tuning) |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        for a in ARMS:
            rs = sorted([r for r in rows if r["policy"] == p and r["arm"] == a],
                        key=lambda r: r["seed"])
            u = np.array([r["utility"] for r in rs])
            diff = np.array([r["utility"] - base[r["seed"]]["utility"] for r in rs])
            sol = sum(r["solution"] for r in rs)
            viol = sum(r["violation"] for r in rs)
            over = np.mean([r["train_over"] for r in rs if r["train_over"] is not None])
            ppsf = sum(r["pred_pass_safety_fail"] for r in rs)
            sens = tune["sensitivity"].get(f"{p}/{a}")
            sens_s = f"{sens['median']:.3f} / {sens['worst']:.3f} ({sens['n_configs']})" \
                if sens else "not tuned"
            out.append(
                f"| `{a}` | `{json.dumps(rs[0]['hp'])}` | {sol}/{len(rs)} | {viol} | "
                f"{u.mean():.3f} ({u.std(ddof=1) / np.sqrt(len(u)):.3f}) | "
                f"{100 * u.mean() / u_base:.0f}% | "
                f"{diff.mean():+.3f} ({diff.std(ddof=1) / np.sqrt(len(diff)):.3f}) | "
                f"{100 * over:.1f}% | {ppsf} | {sens_s} |")
        out.append("")
    open(os.path.join(HERE, "results.md"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("check", "tune", "eval", "report", "tune_b", "eval_b", "report_b", "c"))
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    args = ap.parse_args()
    if args.stage == "check":
        stage_check()
    elif args.stage == "tune":
        stage_tune(args.workers)
    elif args.stage == "eval":
        stage_eval(args.workers, range(60))
    elif args.stage == "tune_b":
        stage_tune_b(args.workers)
    elif args.stage == "eval_b":
        stage_eval_b(args.workers, range(60))
    elif args.stage == "c":
        stage_c(args.workers)
    elif args.stage == "report_b":
        stage_report_b()
    else:
        stage_report()
