"""Spike 017 library: upper confidence bounds on a true rate measured through a judge.

The constraint is ``E[Y] <= tau`` with ``Y`` the gold 0/1 label (a human's, or an exact
count). The judge gives ``f`` for every scored response: either its 0/1 label or its
probability ``p`` in [0, 1]. ``N`` responses are judged; ``n`` of them carry ``Y``.

Every function returns a one-sided ``1 - delta`` *upper* bound ``U`` on ``E[Y]`` (the
safety test passes when ``U <= tau``) and is vectorised over a leading ``reps`` axis, so a
plasmode sweep is one call. Conventions follow the project: each limit is one-sided at the
``delta`` it is given, and a union over k limits passes ``delta / k``.

Routes
------
``classical``      Clopper-Pearson on the n labels. Exact; ignores the judge.
``naive``          Clopper-Pearson on the N judge labels. Valid only for a perfect judge.
``youden``         ``(q_hi - FA_lo) / (sens_lo - FA_lo)`` with sens, FA from the labels
                   (the correction of ``seldonian/llm/calibration.py``, estimated directly).
``answer_aware``   spike 006: false alarms only on answers, ``(q_hi - FA_lo w_lo) /
                   (sens_lo - FA_lo)`` with ``w`` the answer rate.
``ppi_clt``        ``mean_u(f) + mean_n(Y - f)``, normal limit (PPI, arXiv:2301.09633).
``ppipp_clt``      ``mean_n(Y) + lam (mean_u(f) - mean_n(f))`` with the variance-minimising
                   ``lam`` (PPI++, arXiv:2311.01453). ``lam = 0`` is the labels alone.
``ppipp_wilson``   PPI++ with the label variance evaluated at the hypothesised rate, as the
                   Wilson interval does; equals the Wilson bound at ``lam = 0``.
``ppipp_boot``     PPI++ with a studentised (bootstrap-t) limit: the labelled pairs are
                   resampled, ``lam`` is re-estimated in every resample, and the bound is
                   ``est - q_delta(t*) se``. Second-order correct for a one-sided limit, so it
                   absorbs the skew of a rarely-firing judge and the bias of an estimated
                   ``lam`` that the normal limit ignores. A resample with zero variance counts
                   as ``t* = -inf``, so too few positives give a vacuous bound, not a wrong one.
``ppi_block``      finite-sample PPI for a *fixed* ``lam``: each label is paired with a
                   disjoint block of unlabelled scores, ``Z_i = Y_i - lam f_i + lam
                   mean(block_i)``, i.i.d. in [-lam, 1 + lam] with mean ``E[Y]`` and the PPI
                   variance, so one betting bound at the full ``delta`` applies.
``ppi_exact3``     0/1 judge, finite-sample: ``E[Y] = E[f] + P(miss) - P(false alarm)``,
                   three Clopper-Pearson limits at ``delta / 3``.
``ppi_bounded``    any ``f`` in [0, 1], finite-sample: the project's betting bound on
                   ``mean_N(f)`` and on ``mean_n(Y - f)`` in [-1, 1], ``delta / 2`` each.
                   This is "E[p] as a bounded feature" plus its rectifier.
``strat_exact``    0/1 judge, post-stratified: ``pi a + (1 - pi) b`` with ``a``, ``b`` the
                   gold rates among flagged and cleared; Clopper-Pearson at ``delta / 3``.
                   Exact whether labels were drawn at random or by stratum of ``f``.

    ../../../.venv/bin/python check_cert.py      # identities and coverage on known truth
"""
import os
import sys

import numpy as np
from scipy.special import betaincinv
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


# ------------------------------------------------------------------ Clopper-Pearson, vectorised

def cp_upper(k, n, delta):
    """Exact upper limit for a binomial proportion; 1 where ``n == 0`` or ``k == n``."""
    k, n = np.broadcast_arrays(np.asarray(k, dtype=float), np.asarray(n, dtype=float))
    ok = (n > 0) & (k < n)
    out = np.ones(k.shape)
    out[ok] = betaincinv(k[ok] + 1, n[ok] - k[ok], 1 - delta)
    return out


def cp_lower(k, n, delta):
    """Exact lower limit; 0 where ``n == 0`` or ``k == 0``."""
    k, n = np.broadcast_arrays(np.asarray(k, dtype=float), np.asarray(n, dtype=float))
    ok = (n > 0) & (k > 0)
    out = np.zeros(k.shape)
    out[ok] = betaincinv(k[ok], n[ok] - k[ok] + 1, delta)
    return out


def _2d(*xs):
    return [np.atleast_2d(np.asarray(x, dtype=float)) for x in xs]


# ------------------------------------------------------------------ routes

def classical(y_lab, delta):
    (y,) = _2d(y_lab)
    return cp_upper(y.sum(1), y.shape[1], delta)


def naive(f_all, delta):
    (f,) = _2d(f_all)
    return cp_upper(f.sum(1), f.shape[1], delta)


def rates(y_lab, f_lab):
    """Counts behind sens and FA: (tp, n_pos, fp, n_neg) per rep."""
    y, f = _2d(y_lab, f_lab)
    pos = y.sum(1)
    return (y * f).sum(1), pos, ((1 - y) * f).sum(1), y.shape[1] - pos


def youden(y_lab, f_lab, f_all, delta, cal=None, refuse_negative=False):
    """
    Rogan-Gladen with one-sided limits. ``cal`` = (tp, n_pos, fp, n_neg) overrides the
    counts from ``y_lab``/``f_lab`` (calibration carried from another population).
    Returns 1 (vacuous) where ``sens_lo <= FA_lo``. A negative value (the flag rate's upper
    limit is under the false-alarm rate's lower limit, which the model ``q = s p + a (1 - p)``
    rules out) is clipped to 0, the usual estimator and what the stored studies measure;
    ``refuse_negative`` returns 1 there instead, which is what a certificate should do.
    """
    (fa_,) = _2d(f_all)
    tp, pos, fp, neg = cal if cal is not None else rates(y_lab, f_lab)
    s_lo = cp_lower(tp, pos, delta / 3)
    a_lo = cp_lower(fp, neg, delta / 3)
    q_hi = cp_upper(fa_.sum(1), fa_.shape[1], delta / 3)
    j = s_lo - a_lo
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(j > 0, (q_hi - a_lo) / j, 1.0)
    if refuse_negative:
        u = np.where(u < 0, 1.0, u)
    return np.clip(u, 0.0, 1.0)


def answer_aware(f_all, ans_all, delta, cal, refuse_negative=False):
    """
    Spike 006's correction. ``cal`` = (tp, n_pos, fp_answered, n_neg_answered): recall, and
    the false-alarm rate among *answered* non-violations. Model: a refusal is never flagged
    and never a violation, so ``obs = sens r + FA (w - r)``.
    """
    fa_, an = _2d(f_all, ans_all)
    tp, pos, fp, neg = cal
    s_lo = cp_lower(tp, pos, delta / 4)
    a_lo = cp_lower(fp, neg, delta / 4)
    q_hi = cp_upper(fa_.sum(1), fa_.shape[1], delta / 4)
    w_lo = cp_lower(an.sum(1), an.shape[1], delta / 4)
    j = s_lo - a_lo
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(j > 0, (q_hi - a_lo * w_lo) / j, 1.0)
    if refuse_negative:
        u = np.where(u < 0, 1.0, u)
    return np.clip(u, 0.0, 1.0)


def ppi_point(y_lab, f_lab, f_unl, lam=None):
    """PPI (``lam = 1``) or PPI++ (``lam = None``): estimate, variance, lam per rep."""
    y, fl, fu = _2d(y_lab, f_lab, f_unl)
    n, nu = y.shape[1], fu.shape[1]
    if lam is None:
        var_f = np.concatenate([fl, fu], axis=1).var(axis=1, ddof=1)
        cov = ((y - y.mean(1, keepdims=True)) * (fl - fl.mean(1, keepdims=True))).sum(1) / (n - 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            lam = np.where(var_f > 0, cov / ((1 + n / nu) * var_f), 0.0)
    lam = np.broadcast_to(np.asarray(lam, dtype=float), (y.shape[0],))
    est = y.mean(1) + lam * (fu.mean(1) - fl.mean(1))
    var = (y - lam[:, None] * fl).var(axis=1, ddof=1) / n + lam ** 2 * fu.var(axis=1, ddof=1) / nu
    return est, var, lam


def ppi_clt(y_lab, f_lab, f_unl, delta, lam=1.0):
    est, var, _ = ppi_point(y_lab, f_lab, f_unl, lam)
    return est + norm.ppf(1 - delta) * np.sqrt(var)


def ppipp_clt(y_lab, f_lab, f_unl, delta):
    return ppi_clt(y_lab, f_lab, f_unl, delta, lam=None)


def ppipp_wilson(y_lab, f_lab, f_unl, delta, lam=None):
    """
    Score-type PPI++: the smallest ``m >= est`` with ``(m - est)^2 >= z^2 V(m)``, where
    ``V(m) = (m (1 - m) - 2 lam cov + lam^2 var_f) / n + lam^2 var_f / Nu`` puts the label
    variance at the hypothesis instead of the estimate. A quadratic in ``m``.
    """
    y, fl, fu = _2d(y_lab, f_lab, f_unl)
    n, nu = y.shape[1], fu.shape[1]
    est, _, lam = ppi_point(y, fl, fu, lam)
    var_f = np.concatenate([fl, fu], axis=1).var(axis=1, ddof=1)
    cov = ((y - y.mean(1, keepdims=True)) * (fl - fl.mean(1, keepdims=True))).sum(1) / (n - 1)
    k = -2 * lam * cov + lam ** 2 * var_f * (1 + n / nu)
    z2 = norm.ppf(1 - delta) ** 2 / n
    e = np.clip(est, 0.0, 1.0)
    # (1 + z2) m^2 - (2 e + z2) m + e^2 - z2 k = 0, upper root; floor the variance at 0
    a, b, c_ = 1 + z2, -(2 * e + z2), e ** 2 - z2 * k
    disc = np.maximum(b ** 2 - 4 * a * c_, 0.0)
    return np.clip((-b + np.sqrt(disc)) / (2 * a), e, 1.0)


def ppipp_boot(y_lab, f_lab, f_unl, delta, lam=None, boots=300, seed=0, degenerate="sign"):
    """Bootstrap-t upper bound for PPI++ (``lam=None``) or PPI (``lam=1.0``); loops over reps.

    A resample with zero variance has no studentised statistic. ``degenerate="sign"`` sends it
    to -inf when its estimate is under the sample's and to +inf otherwise, which is the
    conservative side for a rate (the resample is all zeros). ``"low"`` sends every one to
    -inf; use it when the labels are differences, where an all-zero resample can sit above a
    negative estimate and the sign rule would drop it from the lower tail."""
    y, fl, fu = _2d(y_lab, f_lab, f_unl)
    reps, n = y.shape
    nu = fu.shape[1]
    rng = np.random.default_rng(seed)
    est, var, _ = ppi_point(y, fl, fu, lam)
    se = np.sqrt(var)
    fbar_u = fu.mean(1)
    var_fu = fu.var(axis=1, ddof=1)
    out = np.empty(reps)
    for r in range(reps):
        idx = rng.integers(0, n, size=(boots, n))
        ys, fs = y[r][idx], fl[r][idx]
        ym, fm = ys.mean(1), fs.mean(1)
        fub = fbar_u[r] + np.sqrt(var_fu[r] / nu) * rng.standard_normal(boots)
        if lam is None:
            cov = ((ys - ym[:, None]) * (fs - fm[:, None])).sum(1) / (n - 1)
            lb = cov / ((1 + n / nu) * var_fu[r]) if var_fu[r] > 0 else np.zeros(boots)
        else:
            lb = np.full(boots, float(lam))
        eb = ym + lb * (fub - fm)
        vb = (ys - lb[:, None] * fs).var(axis=1, ddof=1) / n + lb ** 2 * var_fu[r] / nu
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(vb > 0, (eb - est[r]) / np.sqrt(vb),
                         np.where((eb < est[r]) | (degenerate == "low"), -np.inf, np.inf))
        q = np.quantile(t, delta, method="lower")
        out[r] = est[r] - q * se[r] if np.isfinite(q) else 1.0
    return np.clip(out, 0.0, 1.0)


def ppi_block(y_lab, f_lab, f_unl, delta, lam, bound="betting"):
    """Finite-sample PPI for fixed ``lam`` in [0, 3] (scalar or one per rep); loops over reps."""
    from seldonian.bounds import get_bound
    fn = get_bound(bound)
    y, fl, fu = _2d(y_lab, f_lab, f_unl)
    n = y.shape[1]
    m = fu.shape[1] // n
    lam = np.clip(np.broadcast_to(np.asarray(lam, dtype=float), (y.shape[0],)), 0.0, 3.0)
    out = np.empty(y.shape[0])
    for r in range(y.shape[0]):
        if m < 1 or lam[r] == 0.0:
            z, lo, hi = y[r], 0.0, 1.0
        else:
            g = fu[r, :m * n].reshape(n, m).mean(1)
            z, lo, hi = y[r] - lam[r] * fl[r] + lam[r] * g, -lam[r], 1.0 + lam[r]
        out[r] = fn(z, delta, a=lo, b=hi).upper
    return np.clip(out, 0.0, 1.0)


def lam_hat(y_lab, f_lab, n_unl_ratio=None):
    """PPI++'s lam from a labelled sample (``cov / var_f``, shrunk by ``1 + n / Nu`` if given)."""
    y, f = _2d(y_lab, f_lab)
    var_f = f.var(axis=1, ddof=1)
    cov = ((y - y.mean(1, keepdims=True)) * (f - f.mean(1, keepdims=True))).sum(1) / (y.shape[1] - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = np.where(var_f > 0, cov / var_f, 0.0)
    return lam / (1 + n_unl_ratio) if n_unl_ratio else lam


def ppi_exact3(y_lab, f_lab, f_all, delta):
    y, fl, fa_ = _2d(y_lab, f_lab, f_all)
    n = y.shape[1]
    q_hi = cp_upper(fa_.sum(1), fa_.shape[1], delta / 3)
    miss_hi = cp_upper((y * (1 - fl)).sum(1), n, delta / 3)
    fa_lo = cp_lower(((1 - y) * fl).sum(1), n, delta / 3)
    return np.clip(q_hi + miss_hi - fa_lo, 0.0, 1.0)


def strat_exact(y_lab, f_lab, f_all, delta):
    """
    Post-stratified on the 0/1 judge. ``a`` and ``b`` are conditionally binomial given the
    judge labels, so the limits are exact under random *or* judge-stratified labelling.
    """
    y, fl, fa_ = _2d(y_lab, f_lab, f_all)
    n1 = fl.sum(1)
    n0 = fl.shape[1] - n1
    a_hi = cp_upper((y * fl).sum(1), n1, delta / 3)
    b_hi = cp_upper((y * (1 - fl)).sum(1), n0, delta / 3)
    k, big_n = fa_.sum(1), fa_.shape[1]
    pi = np.where(a_hi >= b_hi, cp_upper(k, big_n, delta / 3), cp_lower(k, big_n, delta / 3))
    return np.clip(pi * a_hi + (1 - pi) * b_hi, 0.0, 1.0)


def ppi_bounded(y_lab, f_lab, f_all, delta, bound="betting"):
    """Finite-sample PPI for any f in [0, 1]; loops over reps (the project bound is scalar)."""
    from seldonian.bounds import get_bound
    fn = get_bound(bound)
    y, fl, fa_ = _2d(y_lab, f_lab, f_all)
    out = np.empty(y.shape[0])
    for r in range(y.shape[0]):
        out[r] = fn(fa_[r], delta / 2).upper + fn(y[r] - fl[r], delta / 2, a=-1.0, b=1.0).upper
    return np.clip(out, 0.0, 1.0)


# ------------------------------------------------------------------ theory

def rho2(y, f):
    y, f = np.asarray(y, dtype=float), np.asarray(f, dtype=float)
    if y.std() == 0 or f.std() == 0:
        return 0.0
    return float(np.corrcoef(y, f)[0, 1] ** 2)


def gain_ppipp(rho_sq, n, n_unl):
    """Effective-label gain of PPI++ over the labels alone: ``1 / (1 - rho^2 Nu / (Nu + n))``."""
    return 1.0 / (1.0 - rho_sq * n_unl / (n_unl + n))


def gain_ppi(y, f, n, n_unl):
    """The same for plain PPI (lam = 1): ``var(Y) / (var(Y - f) + var(f) n / Nu)``."""
    y, f = np.asarray(y, dtype=float), np.asarray(f, dtype=float)
    return float(y.var() / ((y - f).var() + f.var() * n / n_unl))


def rho2_binary(r, sens, fa_rate):
    """rho^2 between a rate-r label and a 0/1 judge with the given sens and false-alarm rate."""
    q = sens * r + fa_rate * (1 - r)
    return r * (1 - r) * (sens - fa_rate) ** 2 / (q * (1 - q)) if 0 < q < 1 else 0.0


# ------------------------------------------------------------------ the certificate

K0 = 10          # PPI++ needs at least this many positives and negatives among the labels
K_CARRY = 30     # a carried calibration needs at least this many human positives


def certify(tau, delta, *, exact_feature=None, y_lab=None, f_lab=None, f_unl=None, seed=0,
            wording=None, judge=None):
    """
    One constraint, one certificate. The route is fixed by the data's *shape*, never by which
    bound came out smaller (the smaller of two 95% bounds missed up to 0.078 in ``route.md``):

    - ``exact_feature`` given (a verifiable property, computed by code on every response):
      Clopper-Pearson on it. No labels, no judge.
    - labels with fewer than ``K0`` positives or negatives: Clopper-Pearson on the labels;
      the judge is not used.
    - otherwise PPI++ on the judge feature with the bootstrap-t limit.

    Returns the card: route, estimate, upper bound, whether ``tau`` is certified, the smallest
    threshold this data certifies, and for PPI++ the judged-scale threshold (the largest mean
    of ``f`` on the unlabelled responses that would still pass).
    """
    import hashlib
    card = dict(tau=tau, delta=delta, judge=judge,
                wording_sha=hashlib.sha1(wording.encode()).hexdigest()[:10] if wording else None)
    if exact_feature is not None:
        x = np.asarray(exact_feature, dtype=float)
        u = float(cp_upper(x.sum(), len(x), delta))
        card.update(route="deterministic feature, Clopper-Pearson", estimate=float(x.mean()),
                    upper=u, n_scored=len(x), n_labels=0, note="exact; no labels and no judge")
    else:
        y, fl, fu = (np.asarray(a, dtype=float) for a in (y_lab, f_lab, f_unl))
        n, k = len(y), int(y.sum())
        cl = float(cp_upper(k, n, delta))
        card.update(n_labels=n, positives=k, n_scored=n + len(fu), labels_only_upper=cl,
                    rho2=rho2(y, fl), judged_mean=float(fu.mean()))
        if min(k, n - k) < K0:
            card.update(route="labels only, Clopper-Pearson", estimate=k / n, upper=cl,
                        note=f"judge not used: {min(k, n - k)} < {K0} labels in the rarer class")
        else:
            est, var, lam = (float(a[0]) for a in ppi_point(y, fl, fu, None))
            u = float(ppipp_boot(y, fl, fu, delta, seed=seed)[0])
            thr = (tau - (u - est) - (y.mean() - lam * fl.mean())) / lam if lam > 0 else None
            card.update(route="PPI++, bootstrap-t", estimate=est, upper=u, lam=lam,
                        gain=gain_ppipp(card["rho2"], n, len(fu)),
                        effective_labels=((cl - k / n) / max(u - est, 1e-12)) ** 2 * n,
                        judged_threshold=thr, note="approximate (second-order); labels must be "
                        "an i.i.d. sample of the scored responses")
    card["certified"] = bool(card["upper"] <= tau)
    card["smallest_tau"] = card["upper"]
    return card


def certify_carried(tau, delta, f_all, cal, answered=None):
    """
    A certificate from a calibration measured elsewhere: ``cal`` = (caught, positives,
    false alarms, negatives). Refuses below ``K_CARRY`` positives. Valid only if the judge's
    recall and false-alarm rate are the same on the certified population (stage E).
    """
    tp, pos, fp, neg = cal
    card = dict(tau=tau, delta=delta, route="carried calibration", positives=int(pos),
                negatives=int(neg), n_scored=len(f_all), judged_mean=float(np.mean(f_all)),
                note="assumes the judge's recall and false-alarm rate carry over unchanged")
    if pos < K_CARRY:
        card.update(upper=1.0, certified=False, smallest_tau=1.0,
                    refused=f"{int(pos)} human positives; {K_CARRY} needed to bound recall")
        return card
    # a negative estimate contradicts the carried rates, so it refuses (audit of 2026-10-06, item 8)
    if answered is None:
        u = float(youden(None, None, f_all, delta, cal=cal, refuse_negative=True)[0])
    else:
        u = float(answer_aware(f_all, answered, delta, cal, refuse_negative=True)[0])
    card.update(upper=u, certified=bool(u <= tau), smallest_tau=u)
    return card
