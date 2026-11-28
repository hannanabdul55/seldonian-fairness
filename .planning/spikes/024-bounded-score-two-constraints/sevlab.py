"""Spike 024: steering candidate selection with a bounded severity score and two constraints.

Every arm runs the real ``SeldonianLLMPolicy`` (predicted tests every 25 steps, the
best-feasible-checkpoint rule, one safety test) on 022a's bandit with a severity score added;
only the way training is steered differs. See README.md for the environment, the arms, the
expectations and the decision rule (registered in commit adfb4fd before any run).

    ../../../.venv/bin/python sevlab.py --stage check
    ../../../.venv/bin/python sevlab.py --stage tune
    ../../../.venv/bin/python sevlab.py --stage eval
    ../../../.venv/bin/python sevlab.py --stage report
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import deque  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
LAB022 = os.path.join(HERE, "..", "022a-es-vs-grpo-bandit")
sys.path.insert(0, LAB022)

from eslab import CFG, POLICIES, GRPOBackend, softmax  # noqa: E402
from seldonian.llm.constraints import CallableFeature, ExpressionConstraint, Measure  # noqa: E402
from seldonian.llm.data import split_prompts  # noqa: E402
from seldonian.llm.policy import Constraint, SeldonianLLMPolicy, effective_n  # noqa: E402
from seldonian.llm.rewards import LagrangianReward  # noqa: E402
from seldonian.llm.smooth_bounds import (bentkus_upper, betting_influence,  # noqa: E402
                                         betting_upper, cp_upper)
from seldonian.llm.synthetic import (SyntheticEnv, SyntheticJudge, SyntheticReward,  # noqa: E402
                                     parse_contexts, sigmoid)

SEV_BIAS = (-0.8, -2.5, -0.8, -2.5)
KAPPA = (1.5, 20.0, 20.0, 1.5)
V_SCALE = 0.5
PRESSURE_S = 2.0
PRESSURE_V = 2.0          # amendment 1 (README): both pressures 2.0, registered at 1.0
MARGIN = dict(sev=0.03, harm=0.03)
EMA, BUF_STEPS, T = 0.2, 5, 0.02
TUNE_SEEDS = range(1000, 1020)
LR = {"linear": (0.05, 0.1, 0.2, 0.4, 0.8), "mlp": (0.0001, 0.0003, 0.001, 0.003)}
ARMS = ("lagrangian", "pi_mean", "pi_full", "pi_mean_sep", "pi_full_sep")
_H_RE = re.compile(r";h=([0-9.eE+-]+)")


def grid(policy, arm):
    lrs = LR[policy]
    if arm == "lagrangian":
        return [dict(lr=lr, eta=e, lam0=l0) for lr in lrs for e in (30.0, 100.0, 300.0)
                for l0 in (1.0, 5.0)]
    if arm.startswith("pi_"):
        return [dict(lr=lr, c0=c0, eta_i=e) for lr in lrs for c0 in (10.0, 30.0)
                for e in (3.0, 10.0, 30.0)]
    raise ValueError(arm)


# ------------------------------------------------------------------ environment

class SevEnv(SyntheticEnv):
    """022a's bandit plus a severity score ``h ~ Beta(kappa_a S, kappa_a (1 - S))``."""

    def __init__(self, seed, **kw):
        super().__init__(n_contexts=CFG["population"], d=CFG["d"], pressure=PRESSURE_V,
                         seed=seed, **kw)
        rng = np.random.default_rng([seed, 24])
        self.V = rng.standard_normal((self.n_actions, self.d)) * V_SCALE
        self.S = sigmoid(self.X @ self.V.T + np.asarray(SEV_BIAS))
        self.kappa = np.asarray(KAPPA)
        self.mean_reward = self.mean_reward + PRESSURE_S * self.S
        self.adv = self.group_idx == self.groups.index("adversarial")


def sev_values(prompts, responses, references=None):
    out = []
    for r in responses:
        m = _H_RE.search(r)
        if m is None:
            raise ValueError(f"no severity in {r!r}")
        out.append(float(m.group(1)))
    return np.asarray(out)


def fmt(actions, v, h):
    return [f"a={a};v={vv};h={hh:.6f}" for a, vv, hh in zip(actions.tolist(), v.tolist(),
                                                           h.tolist())]


# ------------------------------------------------------------------ backends

class SevGRPO(GRPOBackend):
    """022a's GRPO backend whose responses carry a severity draw."""

    def draw(self, theta, idx, u_a, u_v):
        a, v = super().draw(theta, idx, u_a, u_v)
        s = self.env.S[idx, a]
        k = self.env.kappa[a]
        self.last_h = np.clip(self.rng.beta(k * s, k * (1 - s)), 0.0, 1.0)
        return a, v

    def generate(self, prompts, max_new_tokens=256, temperature=1.0):
        idx = parse_contexts(prompts)
        a, v = self.draw(self.theta, idx, self.rng.random(len(idx)), self.rng.random(len(idx)))
        return fmt(a, v, self.last_h)

    def score(self, reward, idx, actions, v):
        prompts = [f"ctx:{int(i)}" for i in idx]
        return np.asarray(reward(prompts, fmt(actions, v, self.last_h),
                                 groups=[self.group_of[int(i)] for i in idx],
                                 references=[None] * len(idx)), dtype=float)

    def true_sev(self, theta=None):
        idx = np.nonzero(self.env.adv)[0]
        p = self.pi(self.theta if theta is None else theta, idx)
        return float(np.mean(np.sum(p * self.env.S[idx], axis=1)))

    def spread_share(self, theta=None):
        """Share of the severe actions 0 and 2 on adversarial prompts that is action 0."""
        idx = np.nonzero(self.env.adv)[0]
        p = self.pi(self.theta if theta is None else theta, idx).mean(axis=0)
        return float(p[0] / max(p[0] + p[2], 1e-12)), float(p[0] + p[2])

    def tick(self, equiv, on_step):
        if equiv % CFG["predict_every"] == 0:
            self.trace.append((equiv, self.true_rate(), self.true_sev()))
        on_step(equiv)


class PIBackend(SevGRPO):
    """PI steering, one term per constraint, with a mean-only or influence `sev` surrogate."""

    def __init__(self, *a, hp, sev_mode, separate, thresholds, n_harm, n_sev, delta_each, **kw):
        super().__init__(*a, lr=hp["lr"], **kw)
        self.hp, self.sev_mode, self.separate = hp, sev_mode, separate
        self.tau, self.n_harm, self.n_sev, self.d1 = thresholds, n_harm, n_sev, delta_each
        self.rate, self.buf = None, deque(maxlen=BUF_STEPS)
        self.lam_dual = dict(sev=0.0, harm=0.0)
        self.s0 = None
        self.infeasible_steps = dict(sev=0, harm=0)
        self.lams = dict(sev=[], harm=[])
        self.eb_neg = []

    def harm_state(self):
        r = torch.tensor(self.rate, dtype=torch.float64, requires_grad=True)
        u = cp_upper(r, self.n_harm, self.d1)
        u.backward()
        return float(u.detach()) - self.tau["harm"], float(r.grad)

    def sev_state(self, h_new):
        """``(c, weights at h_new)`` for the `sev` surrogate on the buffer before this step."""
        buf = np.concatenate(self.buf)
        if self.sev_mode == "full":
            u, psi = betting_influence(buf, self.n_sev, self.d1, at=h_new)
            mu, sd = float(buf.mean()), float(buf.std(ddof=1))
            if sd > 0:
                self.eb_neg.append(
                    1.0 - np.sqrt(2 * np.log(2 / self.d1) / self.n_sev) * mu / sd < 0)
            return u - self.tau["sev"], psi
        mu = torch.tensor(float(buf.mean()), dtype=torch.float64, requires_grad=True)
        u = bentkus_upper(mu, self.n_sev, self.d1)
        u.backward()
        return float(u.detach()) - self.tau["sev"], float(mu.grad) * (h_new - float(mu.detach()))

    def lam(self, name, c):
        self.lam_dual[name] = float(np.clip(self.lam_dual[name] + self.hp["eta_i"] * c, 0, 200))
        out = self.hp["c0"] / (1.0 + np.exp(-c / T)) + self.lam_dual[name]
        self.lams[name].append(out)
        if c > 0:
            self.infeasible_steps[name] += 1
        return out

    def train(self, records, reward, on_step):
        self.setup(records)
        k, G = CFG["prompts_per_step"], CFG["group_size"]
        for step in range(1, self.max_steps + 1):
            idx = self.idx_all[np.repeat(self.rng.choice(len(records), k, replace=False), G)]
            a, v = self.draw(self.theta, idx, self.rng.random(len(idx)),
                             self.rng.random(len(idx)))
            h = self.last_h
            adv_mask = self.env.adv[idx]
            r = self.score(reward, idx, a, v).reshape(k, G)
            if self.rate is None:          # first step: start the running statistics
                self.rate = float(v.mean())
                self.buf.append(h[adv_mask])
                self.s0 = float(np.sqrt(np.mean(r.var(axis=1, ddof=1)))) + 1e-8
            c_h, du = self.harm_state()
            c_s, w_s = self.sev_state(h[adv_mask])
            pen = self.lam("harm", c_h) * du * (v - self.rate)
            ws = np.zeros(len(idx))
            ws[adv_mask] = w_s
            pen = (pen + self.lam("sev", c_s) * ws).reshape(k, G)
            if self.separate:
                adv = (r - r.mean(1, keepdims=True)) / (r.std(1, ddof=1, keepdims=True) + 1e-8)
                adv = adv - (pen - pen.mean(1, keepdims=True)) / self.s0
            else:
                shaped = r - pen
                adv = (shaped - shaped.mean(1, keepdims=True)) / (
                    shaped.std(1, ddof=1, keepdims=True) + 1e-8)
            self.theta = self.theta - self.opt.step(self.loss_grad(self.theta, idx, a,
                                                                   adv.ravel())[1])
            self.rate = (1 - EMA) * self.rate + EMA * float(v.mean())
            if adv_mask.any():
                self.buf.append(h[adv_mask])
            self.gens += k * G
            self.tick(step, on_step)


# ------------------------------------------------------------------ one run

def run(seed, policy, arm, hp, unconstrained=False):
    t0 = time.time()
    env = SevEnv(seed)
    d_c, d_s = split_prompts(env.records(CFG["n"], seed=seed), test_size=0.4, seed=seed)
    pol = POLICIES[policy](env.d, env.n_actions, seed=seed)
    kw = dict(max_steps=CFG["steps"], beta=CFG["beta"], seed=seed + 2)
    judge = SyntheticJudge(1.0, 1.0, seed=seed)
    sev_f = CallableFeature(sev_values, 0.0, 1.0, "severity")
    base = SyntheticReward(env, seed=seed + 1)
    delta_each = CFG["delta"] / 2
    n_adv_c = sum(r["group"] == "adversarial" for r in d_c)
    n_adv_s = sum(r["group"] == "adversarial" for r in d_s)
    n_pred_adv = max(2, int(round(CFG["predict_n"] * n_adv_c / len(d_c))))
    n_harm = effective_n(CFG["predict_n"], len(d_s))
    n_sev = effective_n(n_pred_adv, n_adv_s)
    probe = SevGRPO(env, pol, lr=0.1, **kw)
    ref = dict(harm=probe.true_rate(), sev=probe.true_sev(), reward=probe.true_reward())
    tau = dict(harm=ref["harm"] + MARGIN["harm"], sev=ref["sev"] + MARGIN["sev"])
    if arm == "lagrangian" or unconstrained:
        backend = SevGRPO(env, pol, lr=hp["lr"], **kw)
        reward = base if unconstrained else LagrangianReward(
            base, [(sev_f, "adversarial"), (judge, None)], names=["sev", "harm"],
            lam0=hp["lam0"], eta=hp["eta"], lam_max=CFG["lam_max"])
    else:
        backend = PIBackend(env, pol, hp=hp, sev_mode="full" if "full" in arm else "mean",
                            separate=arm.endswith("_sep"), thresholds=tau, n_harm=n_harm,
                            n_sev=n_sev, delta_each=delta_each, **kw)
        reward = base
    cons = [ExpressionConstraint("sev", {"s": Measure("s", sev_f, "adversarial")},
                                 lambda m: m["s"], tau["sev"], bound="betting_mixture",
                                 monotone=True),
            Constraint("harm", judge, threshold=tau["harm"], group=None,
                       bound="clopper_pearson")]
    pobj = SeldonianLLMPolicy(backend, d_c, d_s, reward=reward, constraints=cons,
                              delta=CFG["delta"], predict_every=CFG["predict_every"],
                              predict_n=CFG["predict_n"], seed=seed)
    solution = pobj.fit(seldonian=True) is not None
    rate, sev, rew = backend.true_rate(), backend.true_sev(), backend.true_reward()
    tr = np.array([(t[1], t[2]) for t in backend.trace])
    selected = str(pobj.selected["step"])
    spread, severe = backend.spread_share()
    lam = getattr(reward, "lambdas", None)
    row = dict(
        seed=seed, policy=policy, arm=arm, hp=hp, solution=bool(solution),
        tau=tau, ref=ref, rate=rate, sev=sev, reward=rew,
        violation=bool(solution and (rate > tau["harm"] or sev > tau["sev"])),
        utility=float(rew - ref["reward"]) if solution else 0.0, selected=selected,
        pred_pass_safety_fail=bool(selected != "final" and not solution),
        train_over=float(np.mean((tr[:, 0] > tau["harm"]) | (tr[:, 1] > tau["sev"])))
        if len(tr) else None,
        max_over=dict(harm=float(tr[:, 0].max() - tau["harm"]),
                      sev=float(tr[:, 1].max() - tau["sev"])) if len(tr) else None,
        spread_share=spread, severe_share=severe, n_sev=n_sev, n_harm=n_harm,
        lam_final=lam, gens=int(backend.gens), seconds=round(time.time() - t0, 2))
    if isinstance(backend, PIBackend):
        row.update(infeasible_share={k: v / CFG["steps"]
                                     for k, v in backend.infeasible_steps.items()},
                   mean_lam={k: float(np.mean(v)) for k, v in backend.lams.items()},
                   eb_neg_share=float(np.mean(backend.eb_neg)) if backend.eb_neg else None)
    return row


def _run(job):
    return run(*job)


def sweep(jobs, workers):
    t0, out = time.time(), []
    with ProcessPoolExecutor(workers) as ex:
        for i, row in enumerate(ex.map(_run, jobs, chunksize=2), 1):
            out.append(row)
            if i % 200 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} runs, {time.time() - t0:.0f} s", flush=True)
    return out


# ------------------------------------------------------------------ stages

def stage_check(workers):
    rng = np.random.default_rng(0)
    # 1. influence at new points equals the in-sample influence where they coincide
    x = rng.beta(2, 6, size=160)
    _, psi = betting_influence(x, 120, 0.05)
    _, psi_at = betting_influence(x, 120, 0.05, at=x[:30])
    print(f"influence at new points vs in-sample: max |diff| {np.abs(psi_at - psi[:30]).max():.1e}")
    # 2. weight signs: rising in h for both sev surrogates; harm weight rises with v
    grid_h = np.linspace(0, 1, 41)
    _, w_full = betting_influence(x, 120, 0.05, at=grid_h)
    mu = torch.tensor(float(x.mean()), dtype=torch.float64, requires_grad=True)
    bentkus_upper(mu, 120, 0.05).backward()
    r = torch.tensor(0.15, dtype=torch.float64, requires_grad=True)
    cp_upper(r, 224, 0.05).backward()
    print(f"full weights increasing in h: {bool(np.all(np.diff(w_full) > 0))}; "
          f"mean-only slope B' = {mu.grad.item():.3f} > 0; harm U' = {r.grad.item():.3f} > 0; "
          f"full weight at h=0, mean, 1: {w_full[0]:.3f}, "
          f"{np.interp(x.mean(), grid_h, w_full):.3f}, {w_full[-1]:.3f}; "
          f"betting U {float(betting_upper(x, 120, 0.05)):.4f}")
    # 3. the environment binds: unconstrained GRPO (lr 0.4) leaves both feasible regions
    rows = sweep([(s, "linear", "lagrangian", dict(lr=0.4), True) for s in range(5)], workers)
    binds = []
    for row in rows:
        mo = row["max_over"]
        binds.append(mo["harm"] > 0 and mo["sev"] > 0)
        print(f"  unconstrained seed {row['seed']}: max over threshold harm {mo['harm']:+.3f} "
              f"sev {mo['sev']:+.3f}; final rate {row['rate']:.3f} (tau {row['tau']['harm']:.3f}) "
              f"sev {row['sev']:.3f} (tau {row['tau']['sev']:.3f}); n_sev {row['n_sev']}")
    print(f"environment binds on both constraints in {sum(binds)} of 5 seeds")
    # 4. one run per arm per policy, end to end
    jobs = [(0, p, a, grid(p, a)[len(grid(p, a)) // 2]) for p in ("linear", "mlp") for a in ARMS]
    for row in sweep(jobs, workers):
        print(f"  {row['policy']:6s} {row['arm']:12s} solution={row['solution']} "
              f"utility={row['utility']:.3f} over={row['train_over']} "
              f"spread={row['spread_share']:.2f} infeasible={row.get('infeasible_share')} "
              f"eb_neg={row.get('eb_neg_share')} {row['seconds']} s")
    return all(binds)


def stage_tune(workers):
    jobs = [(s, p, a, hp) for p in ("linear", "mlp") for a in ARMS for hp in grid(p, a)
            for s in TUNE_SEEDS]
    print(f"tuning: {len(jobs)} runs")
    rows = sweep(jobs, workers)
    chosen, sens = {}, {}
    for p in ("linear", "mlp"):
        for a in ARMS:
            by = {}
            for r in rows:
                if r["policy"] == p and r["arm"] == a:
                    by.setdefault(json.dumps(r["hp"], sort_keys=True), []).append(r["utility"])
            means = {k: float(np.mean(v)) for k, v in by.items()}
            best = max(means, key=means.get)
            chosen[f"{p}/{a}"] = dict(hp=json.loads(best), utility=means[best])
            vals = sorted(means.values())
            sens[f"{p}/{a}"] = dict(median=float(np.median(vals)), worst=vals[0], best=vals[-1])
            print(f"{p:6s} {a:12s} best {means[best]:.3f} {best}  median {np.median(vals):.3f}")
    json.dump(dict(chosen=chosen, sensitivity=sens), open(os.path.join(HERE, "tune.json"), "w"),
              indent=1)
    with open(os.path.join(HERE, "tune_rows.json"), "w") as f:
        json.dump(rows, f)


def stage_eval(workers, seeds):
    chosen = json.load(open(os.path.join(HERE, "tune.json")))["chosen"]
    jobs = [(s, p, a, chosen[f"{p}/{a}"]["hp"]) for p in ("linear", "mlp") for a in ARMS
            for s in seeds]
    print(f"evaluation: {len(jobs)} runs")
    rows = sweep(jobs, workers)
    json.dump(rows, open(os.path.join(HERE, "results.json"), "w"))


def stage_report():
    tune = json.load(open(os.path.join(HERE, "tune.json")))
    rows = json.load(open(os.path.join(HERE, "results.json")))
    out = ["# Spike 024 results", "",
           "60 paired evaluation seeds (0-59) per arm and policy; utility = true reward of the "
           "returned policy minus the reference's (No Solution Found = 0). Paired differences "
           "are against `lagrangian` unless stated.", ""]
    summary = {}
    for p in ("linear", "mlp"):
        out += [f"## {p}", "",
                "| Arm | settings | solutions | over | utility (se) | paired vs lagrangian (se) "
                "| ckpts over | pred-pass safety-fail | grid median | spread share | "
                "severe share | EB C'(0)<0 |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        by = {a: sorted([r for r in rows if r["policy"] == p and r["arm"] == a],
                        key=lambda r: r["seed"]) for a in ARMS}
        base = np.array([r["utility"] for r in by["lagrangian"]])
        for a in ARMS:
            rs = by[a]
            u = np.array([r["utility"] for r in rs])
            d = u - base
            sol = [r for r in rs if r["solution"]]
            eb = [r["eb_neg_share"] for r in rs if r.get("eb_neg_share") is not None]
            s = dict(solutions=len(sol), over=sum(r["violation"] for r in rs),
                     utility=float(u.mean()), se=float(u.std(ddof=1) / np.sqrt(len(u))),
                     diff=float(d.mean()), diff_se=float(d.std(ddof=1) / np.sqrt(len(d))),
                     ckpt_over=float(np.mean([r["train_over"] for r in rs])),
                     ppsf=sum(r["pred_pass_safety_fail"] for r in rs),
                     median=tune["sensitivity"][f"{p}/{a}"]["median"],
                     spread=float(np.mean([r["spread_share"] for r in sol])) if sol else None,
                     severe=float(np.mean([r["severe_share"] for r in sol])) if sol else None,
                     eb_neg=float(np.mean(eb)) if eb else None)
            summary[f"{p}/{a}"] = s
            hp = tune["chosen"][f"{p}/{a}"]["hp"]
            fmt_ = (lambda v: "" if v is None else f"{v:.2f}")
            out.append(f"| `{a}` | {hp} | {s['solutions']}/60 | {s['over']} | {s['utility']:.3f} "
                       f"({s['se']:.3f}) | {'' if a == 'lagrangian' else f'{s['diff']:+.3f} ({s['diff_se']:.3f})'} "
                       f"| {100 * s['ckpt_over']:.1f}% | {s['ppsf']} | {s['median']:.3f} | "
                       f"{fmt_(s['spread'])} | {fmt_(s['severe'])} | "
                       f"{'' if s['eb_neg'] is None else f'{100 * s['eb_neg']:.1f}%'} |")
        out.append("")
        out.append("Pairs (paired difference, se):")
        out.append("")
        for x, y in (("pi_full", "pi_mean"), ("pi_full_sep", "pi_mean_sep"),
                     ("pi_mean_sep", "pi_mean"), ("pi_full_sep", "pi_full")):
            ux = np.array([r["utility"] for r in by[x]])
            uy = np.array([r["utility"] for r in by[y]])
            d = ux - uy
            summary[f"{p}/{x}-{y}"] = dict(diff=float(d.mean()),
                                           se=float(d.std(ddof=1) / np.sqrt(len(d))))
            out.append(f"- `{x}` minus `{y}`: {d.mean():+.3f} ({d.std(ddof=1) / np.sqrt(len(d)):.3f})")
        out.append("")
    open(os.path.join(HERE, "results.md"), "w").write("\n".join(out) + "\n")
    json.dump(summary, open(os.path.join(HERE, "summary.json"), "w"), indent=1)
    print("\n".join(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("check", "tune", "eval", "report"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.stage == "check":
        ok = stage_check(args.workers)
        print("CHECK PASSED" if ok else "CHECK FAILED: the environment does not bind")
    elif args.stage == "tune":
        stage_tune(args.workers)
    elif args.stage == "eval":
        stage_eval(args.workers, range(60))
    else:
        stage_report()
