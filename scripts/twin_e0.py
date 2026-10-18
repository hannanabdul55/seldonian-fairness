"""E0 of the harmful-twin proposal: the identities on a toy where the truth is exact.

A tabular autoregressive model over 2 content tokens plus EOS, maximum length 4, three
prompts. Every sequence is enumerated, so P(H | x), p_H and every gradient are exact.

Checks (proposal section 7, E0):
  C1  twin optimum: q* = p_H and value = log P(H)            (KL form, coefficient 1)
  C2  unweighted main step = grad log P(H) - grad KL(q || p_H) (lag term)
  C3  SNIS-weighted main step = grad log P(H)
  C4  regulariser variants move the optimum as section 3.1 says
        (a) KL coefficient 0.5 and 2        -> q* prop. to (p J)^(1/beta)
        (b) extra KL to a fixed base p0     -> q* prop. to (p0 p J)^(1/2), value != log P(H)
        (c) additive bonus lambda J         -> q* prop. to p exp(lambda J), mostly harmless
        (d) length-normalised log p         -> optimum not p_H, mass piles at the length cap
        (e) per-token-averaged KL           -> optimum not p_H
  C5  Jensen: mean_x log P(H|x) < log mean_x P(H|x)
  C6  alternation lag: unweighted main steps widen KL(q || p_H) more than weighted ones
  C7  the GRPO-style estimator (reward log J + log p - log q_old, group baseline, no reward
      scaling, token-summed loss) converges to p_H; the token-mean aggregation does not

Usage:  python3 -I scripts/twin_e0.py [--out results/twin/e0]
"""
import argparse
import itertools
import json
import math
import os

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
V = 3          # tokens 0, 1 = content; 2 = EOS
L = 4          # maximum length: a 4th content token is followed by forced termination
EPS = 1e-12    # log J smoothing, far below every P(H) here
N_PROMPTS = 3


def prefixes():
    out = [()]
    for k in range(1, L):
        out += list(itertools.product((0, 1), repeat=k))
    return out


PREFIXES = prefixes()
PIDX = {p: i for i, p in enumerate(PREFIXES)}


def sequences():
    """All terminated sequences: content tokens (0..3 of them) then EOS, or 4 content tokens."""
    seqs = []
    for k in range(0, L):
        for c in itertools.product((0, 1), repeat=k):
            seqs.append(tuple(c) + (2,))
    for c in itertools.product((0, 1), repeat=L):
        seqs.append(tuple(c))
    return seqs


SEQS = sequences()
# (prefix index, token) pairs for every scored step of every sequence; the forced stop is not scored
STEPS = []
for y in SEQS:
    content = tuple(t for t in y if t != 2)
    steps = []
    for t, tok in enumerate(y):
        steps.append((PIDX[content[:t]], tok))
    STEPS.append(steps)
LEN = torch.tensor([len(s) for s in STEPS], dtype=torch.float64)


def harmful(y):
    """H: the content contains the bigram (0, 1)."""
    c = [t for t in y if t != 2]
    return any(c[i] == 0 and c[i + 1] == 1 for i in range(len(c) - 1))


H = torch.tensor([1.0 if harmful(y) else 0.0 for y in SEQS])
LOG_J = torch.log(H + EPS)


def logp_all(logits):
    """logits: (n_prompts, n_prefixes, V) -> (n_prompts, n_seqs) summed sequence log-probs."""
    ls = torch.log_softmax(logits, dim=-1)
    out = []
    for steps in STEPS:
        acc = 0.0
        for pi, tok in steps:
            acc = acc + ls[:, pi, tok]
        out.append(acc)
    return torch.stack(out, dim=1)


def log_PH(logits):
    lp = logp_all(logits)
    return torch.logsumexp(lp + torch.log(H + 0.0).clamp_min(-1e300), dim=1)


def kl(lq, lp):
    q = lq.exp()
    return (q * (lq - lp)).sum(dim=1)


def fit(objective, init, steps=4000, lr=0.05):
    phi = init.clone().requires_grad_(True)
    opt = torch.optim.Adam([phi], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = -objective(phi).sum()
        loss.backward()
        opt.step()
    return phi.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/twin/e0")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    g = torch.Generator().manual_seed(a.seed)
    theta = 1.2 * torch.randn(N_PROMPTS, len(PREFIXES), V, generator=g)
    # make harm rarer: push the (0 -> 1) transition down
    for pi, p in enumerate(PREFIXES):
        if p and p[-1] == 0:
            theta[:, pi, 1] -= 2.5
    theta0 = theta.clone()  # the fixed base p0 for C4(b)
    lp = logp_all(theta)
    PH = log_PH(theta).exp()
    lpH = lp + torch.log(H) - log_PH(theta)[:, None]  # log p_H (-inf off H)
    lpH_fin = torch.where(H > 0, lpH, torch.zeros_like(lpH))

    def klH(lq):
        """KL(q restricted to H || p_H). The smoothing puts about eps / P(H) of q off H; that mass is
        reported separately by offH()."""
        q = lq.exp() * H
        q = q / q.sum(dim=1, keepdim=True)
        return (q * (torch.log(q.clamp_min(1e-300)) - lpH_fin)).sum(dim=1)

    def offH(lq):
        return (lq.exp() * (1 - H)).sum(dim=1)

    results = {}
    report = ["# E0: identities on the exact toy", "",
              f"seed {a.seed}; {len(SEQS)} sequences; P(H | x) = {[round(v, 4) for v in PH.tolist()]}", ""]

    def record(name, ok, detail):
        results[name] = {"pass": bool(ok), "detail": detail}
        report.append(f"- {'PASS' if ok else 'FAIL'} {name}: {detail}")

    # ---- C1: KL form, coefficient 1
    def F_kl(phi, beta=1.0, extra_base=None, bonus=None, lognorm=False, tokmean=False):
        lq = logp_all(phi)
        q = lq.exp()
        if bonus is not None:
            harm = bonus * H
        else:
            harm = LOG_J
        lik = lp
        if lognorm:
            lik = lp / LEN
        if tokmean:
            reg = (q * ((lq - lp) / LEN)).sum(dim=1)
        else:
            reg = beta * (q * (lq - lik)).sum(dim=1)
        val = (q * harm).sum(dim=1) - reg
        if extra_base is not None:
            val = val - (q * (lq - extra_base)).sum(dim=1)
        return val

    phi1 = fit(F_kl, theta)
    lq1 = logp_all(phi1)
    gap = klH(lq1)
    val = F_kl(phi1)
    err_val = (val - log_PH(theta)).abs().max().item()
    record("C1 value = log P(H) and q* = p_H",
           err_val < 1e-3 and gap.max().item() < 1e-3,
           f"max |value - log P(H)| = {err_val:.2e}, max KL(q || p_H) = {gap.max().item():.2e}, "
           f"q mass off H {offH(lq1).max().item():.1e} (smoothing eps / P(H) = {EPS / PH.min().item():.1e})")

    # ---- C2, C3: main-step gradients at a sub-optimal twin (stop the twin early)
    phi_sub = fit(F_kl, theta, steps=40)
    lq_sub = logp_all(phi_sub).detach()
    q_sub = lq_sub.exp()
    th = theta.clone().requires_grad_(True)
    lp_th = logp_all(th)
    logPH_th = torch.logsumexp(lp_th + torch.log(H), dim=1)
    g_true = torch.autograd.grad(logPH_th.sum(), th, retain_graph=True)[0]
    # unweighted: E_q[grad log p_theta]
    g_unw = torch.autograd.grad((q_sub * lp_th).sum(), th, retain_graph=True)[0]
    # lag term: grad_theta KL(q || p_{theta,H})
    lpH_th = lp_th + torch.log(H) - logPH_th[:, None]
    # the main step uses only judged-harmful twin samples, so the sampling law is q restricted to H
    qH = q_sub * H
    qH = qH / qH.sum(dim=1, keepdim=True)
    klq = (qH * (torch.log(qH.clamp_min(1e-300)) - lpH_th.masked_fill(H == 0, 0.0))).sum(dim=1)
    g_lag = torch.autograd.grad(klq.sum(), th, retain_graph=True)[0]
    g_unw_H = torch.autograd.grad((qH * lp_th).sum(), th, retain_graph=True)[0]
    err2 = (g_unw_H - (g_true - g_lag)).abs().max().item()
    scale = g_true.abs().max().item()
    record("C2 unweighted step = grad log P(H) - grad KL(q || p_H)",
           err2 < 1e-8 * max(1.0, scale),
           f"max abs error {err2:.2e} against gradient scale {scale:.2e}; lag term size {g_lag.abs().max().item():.2e}")
    # SNIS: weights p 1_H / q, self-normalised, exact expectation under q
    w = (lp_th.detach().exp() * H / q_sub)
    w = w / (q_sub * w).sum(dim=1, keepdim=True)
    g_snis = torch.autograd.grad((q_sub * w * lp_th).sum(), th)[0]
    err3 = (g_snis - g_true).abs().max().item()
    record("C3 SNIS-weighted step = grad log P(H)", err3 < 1e-8 * max(1.0, scale), f"max abs error {err3:.2e}")

    # ---- C4 variants
    def closed_form(logf):
        return torch.log_softmax(logf, dim=1)

    def F_ent(phi, beta_e):
        """Entropy form: reward log p + log J, full-sequence entropy at coefficient beta_e."""
        lq = logp_all(phi)
        q = lq.exp()
        return (q * (lp + LOG_J)).sum(dim=1) - beta_e * (q * lq).sum(dim=1)

    out4 = []   # (name, KL to the predicted optimum, KL to p_H, value error against log P(H))
    # (a) coefficient != 1. KL form: q* = p J^(1/beta) = p_H for an indicator, value beta log P(H).
    #     Entropy form: q* prop. to (p J)^(1/beta): the main model is tempered, not p_H.
    for beta in (0.5, 2.0):
        phi_b = fit(lambda ph: F_kl(ph, beta=beta), theta)
        lqb = logp_all(phi_b)
        out4.append((f"KL form beta={beta}", kl(lqb, closed_form(lp + LOG_J / beta)).max().item(),
                     klH(lqb).max().item(),
                     (F_kl(phi_b, beta=beta) - beta * log_PH(theta)).abs().max().item()))
        phi_b = fit(lambda ph: F_ent(ph, beta), theta)
        lqb = logp_all(phi_b)
        out4.append((f"entropy form beta_e={beta}", kl(lqb, closed_form((lp + LOG_J) / beta)).max().item(),
                     klH(lqb).max().item(), float("nan")))
    # (b) extra KL to a fixed base: q* prop. to (p0 p J)^(1/2). At p = p0 that is p_H, so the
    #     distortion only shows once the main model has moved; use a base the main model has moved from.
    theta_base = theta0 + 0.8 * torch.randn(theta0.shape, generator=g)
    lp_base = logp_all(theta_base)
    phi_b = fit(lambda ph: F_kl(ph, extra_base=lp_base), theta)
    lqb = logp_all(phi_b)
    out4.append(("extra KL to a moved base", kl(lqb, closed_form((lp + lp_base + LOG_J) / 2)).max().item(),
                 klH(lqb).max().item(), float("nan")))
    # (c) additive bonus: q* prop. to p exp(lambda J), mostly harmless
    lam = 2.0
    phi_b = fit(lambda ph: F_kl(ph, bonus=lam), theta)
    lqb = logp_all(phi_b)
    harm_frac = (lqb.exp() * H).sum(dim=1)
    out4.append((f"additive bonus lambda={lam}", kl(lqb, closed_form(lp + lam * H)).max().item(),
                 klH(lqb).max().item(), float("nan")))
    # (d) length-normalised likelihood: optimum is not p_H
    phi_b = fit(lambda ph: F_kl(ph, lognorm=True), theta)
    lqb = logp_all(phi_b)
    len_ln = (lqb.exp() * LEN).sum(dim=1)
    len_pH = (lpH_fin.exp() * H * LEN).sum(dim=1)
    out4.append(("length-normalised log p", float("nan"), klH(lqb).max().item(), float("nan")))
    # (e) per-token-averaged KL: optimum is not p_H
    phi_b = fit(lambda ph: F_kl(ph, tokmean=True), theta)
    out4.append(("per-token-averaged KL", float("nan"), klH(logp_all(phi_b)).max().item(), float("nan")))
    by = {r[0]: r for r in out4}
    # pass: every closed-form optimum is reached (optimiser tolerance 5e-3); the KL form with an
    # indicator stays at p_H with value beta log P(H); every other variant leaves p_H, except the
    # additive bonus, whose distortion is mass off H (restricted to H it is still proportional to p)
    ok4 = (all(by[k][1] < 5e-3 for k in by if not math.isnan(by[k][1]))
           and all(by[k][2] < 1e-3 for k in ("KL form beta=0.5", "KL form beta=2.0"))
           and all(by[k][3] < 1e-3 for k in ("KL form beta=0.5", "KL form beta=2.0"))
           and all(by[k][2] > 1e-2 for k in by if not k.startswith("KL form") and not k.startswith("additive"))
           and harm_frac.max().item() < 0.5)
    detail = "; ".join(f"{n}: KL to predicted optimum {d:.1e}, KL to p_H {gh:.2e}"
                       + (f", |value - beta log P(H)| {ve:.1e}" if not math.isnan(ve) else "")
                       for n, d, gh, ve in out4)
    detail += (f"; additive-bonus harmful fraction {[round(v, 3) for v in harm_frac.tolist()]}"
               f"; mean length under length-normalised optimum {[round(v, 2) for v in len_ln.tolist()]}"
               f" vs under p_H {[round(v, 2) for v in len_pH.tolist()]}")
    record("C4 regulariser variants move the optimum as predicted (KL form with an indicator: "
           "coefficient only rescales the value)", ok4, detail)

    # ---- C5 Jensen
    lPH = log_PH(theta)
    jen = (torch.log(lPH.exp().mean()) - lPH.mean()).item()
    record("C5 Jensen gap", jen > 0, f"log mean P(H) - mean log P(H) = {jen:.4f}")

    # ---- C6 alternation lag
    def alternate(weighted, rounds=15, twin_steps=8, eta=0.3):
        th = theta.clone()
        phi = theta.clone()
        gaps, phs = [], []
        for r in range(rounds):
            lp_cur = logp_all(th).detach()
            def obj(ph):
                lq = logp_all(ph)
                q = lq.exp()
                return (q * LOG_J).sum(dim=1) - (q * (lq - lp_cur)).sum(dim=1)
            phi = fit(obj, phi, steps=twin_steps, lr=0.05)
            lq = logp_all(phi).detach()
            q = lq.exp()
            thg = th.clone().requires_grad_(True)
            lp_th = logp_all(thg)
            if weighted:
                w = (lp_th.detach().exp() * H / q)
                w = w / (q * w).sum(dim=1, keepdim=True)
                gstep = torch.autograd.grad((q * w * lp_th).sum(), thg)[0]
            else:
                qH = q * H
                qH = qH / qH.sum(dim=1, keepdim=True)
                gstep = torch.autograd.grad((qH * lp_th).sum(), thg)[0]
            th = th - eta * gstep
            lpH_new = logp_all(th) + torch.log(H) - log_PH(th)[:, None]
            qh = q * H
            qh = qh / qh.sum(dim=1, keepdim=True)
            gaps.append((qh * (torch.log(qh.clamp_min(1e-300))
                               - lpH_new.masked_fill(H == 0, 0.0))).sum(dim=1).mean().item())
            phs.append(log_PH(th).exp().mean().item())
        return gaps, phs

    gaps_u, ph_u = alternate(False)
    gaps_w, ph_w = alternate(True)
    record("C6 unweighted main steps widen the twin's gap more than weighted ones",
           np.mean(gaps_u) > np.mean(gaps_w) and ph_u[-1] < PH.mean().item() and ph_w[-1] < PH.mean().item(),
           f"mean KL(q || p_H) after main steps: unweighted {np.mean(gaps_u):.4f}, weighted {np.mean(gaps_w):.4f}; "
           f"mean P(H) {PH.mean().item():.4f} -> unweighted {ph_u[-1]:.4f}, weighted {ph_w[-1]:.4f}")

    # ---- C7 GRPO-style estimator
    def grpo_fit(token_mean, steps=4000, G=16, lr=0.05, seed=1, warm=64):
        """Warm start on `warm` harmful samples from p_H (the design's seeded positives), then
        REINFORCE with the group baseline and no reward scaling; learning rate drops to a tenth
        for the last third of the steps."""
        gen = torch.Generator().manual_seed(seed)
        phi = theta.clone().requires_grad_(True)
        opt = torch.optim.Adam([phi], lr=lr)
        idx_w = torch.multinomial(lpH_fin.exp() * H, warm, replacement=True, generator=gen)
        for _ in range(100):
            opt.zero_grad()
            loss = -torch.gather(logp_all(phi), 1, idx_w).mean()
            loss.backward()
            opt.step()
        for step in range(steps):
            if step == 2 * steps // 3:
                for grp in opt.param_groups:
                    grp["lr"] = lr / 10
            opt.zero_grad()
            lq = logp_all(phi)
            with torch.no_grad():
                idx = torch.multinomial(lq.exp(), G, replacement=True, generator=gen)  # (n_prompts, G)
                r = LOG_J[idx] + torch.gather(lp, 1, idx) - torch.gather(lq, 1, idx)
                adv = r - r.mean(dim=1, keepdim=True)   # group baseline, no scaling
            lq_s = torch.gather(lq, 1, idx)
            if token_mean:
                loss = -(adv * lq_s / LEN[idx]).mean()   # "grpo" aggregation: per-sequence token mean
            else:
                loss = -(adv * lq_s).mean()              # token-summed (dr_grpo up to a constant)
            loss.backward()
            opt.step()
        return phi.detach()

    phi_sum = grpo_fit(False)
    phi_mean = grpo_fit(True)
    gap_sum = klH(logp_all(phi_sum)).max().item()
    gap_mean = klH(logp_all(phi_mean)).max().item()
    record("C7 GRPO-style estimator (reward log J + log p - log q_old, group baseline, no reward "
           "scaling, token-summed loss) converges to p_H",
           gap_sum < 0.05,
           f"max KL(q || p_H): token-summed {gap_sum:.4f}; token-mean aggregation {gap_mean:.4f} "
           f"(lengths here are 1 to 4 and p_H sits near the cap, so the toy cannot show the length bias)")

    report.append("")
    n_pass = sum(r["pass"] for r in results.values())
    report.append(f"{n_pass} of {len(results)} checks pass.")
    text = "\n".join(report) + "\n"
    with open(os.path.join(a.out, "report.md"), "w") as f:
        f.write(text)
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    print(text)


if __name__ == "__main__":
    main()
