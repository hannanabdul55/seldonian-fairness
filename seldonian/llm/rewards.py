"""
Reward models and reward shapers.

Every reward exposes ``__call__(prompts, responses, groups=None, references=None)``
returning a float array with one score per (prompt, response) pair. The same object
scores rollouts inside GRPO (via :func:`seldonian.llm.backend.HFGRPOBackend`) and
held-out samples during evaluation, so the two never disagree.
"""
import numpy as np


class Reward:
    name = "reward"

    def __call__(self, prompts, responses, groups=None, references=None):
        raise NotImplementedError


class SequenceClassifierReward(Reward):
    """
    Reward model with a single-logit sequence-classification head, scored on the
    chat-templated (prompt, response) conversation. Default is
    ``Skywork/Skywork-Reward-V2-Qwen3-0.6B``, small enough to stay resident next to a
    sub-2B policy on a 12 GB card.
    """

    def __init__(self, model_name="Skywork/Skywork-Reward-V2-Qwen3-0.6B", device=None,
                 batch_size=16, max_length=1024, dtype=None):
        self.name = f"rm:{model_name}"
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.dtype = dtype
        from seldonian.llm.judges import _device
        self.device = _device(device)
        self._model = None
        self._tok = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            dtype = self.dtype or (torch.bfloat16 if self.device == "cuda" else torch.float32)
            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            if self._tok.pad_token is None:
                self._tok.pad_token = self._tok.eos_token
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, dtype=dtype, num_labels=1).to(self.device).eval()
            self._model.config.pad_token_id = self._tok.pad_token_id

    def __call__(self, prompts, responses, groups=None, references=None):
        import torch
        self._load()
        scores = []
        with torch.no_grad():
            for i in range(0, len(prompts), self.batch_size):
                convs = [[{"role": "user", "content": p}, {"role": "assistant", "content": r}]
                         for p, r in zip(prompts[i:i + self.batch_size],
                                         responses[i:i + self.batch_size])]
                texts = self._tok.apply_chat_template(convs, tokenize=False)
                enc = self._tok(texts, padding=True, truncation=True, max_length=self.max_length,
                                return_tensors="pt").to(self.device)
                scores.extend(self._model(**enc).logits[:, 0].float().cpu().tolist())
        return np.asarray(scores, dtype=float)


class ExactMatchReward(Reward):
    """Verifiable reward: 1 if the final number matches the reference, else 0."""

    name = "exact_match"

    def __call__(self, prompts, responses, groups=None, references=None):
        from seldonian.llm.judges import extract_final_number
        if references is None:
            raise ValueError("ExactMatchReward needs references")
        out = []
        for resp, ref in zip(responses, references):
            got, want = extract_final_number(resp), extract_final_number(str(ref))
            out.append(float(got is not None and want is not None and got == want))
        return np.asarray(out)


class BonusReward(Reward):
    """
    Reward-pressure knob: ``base + alpha * (1 - judge)`` on prompts in ``group``
    (``None`` = everywhere). With a refusal judge and the adversarial group this
    pays the policy for *complying* with harmful requests, so the harm constraint
    binds by construction and ``alpha`` scales how hard the objective pushes
    against it. ``on=1`` pays for the judge's violation event itself instead (the
    direct analogue of the synthetic environment's ``rho * p_v``). Exposes
    ``base`` so reporting unwraps to the underlying reward.
    """

    def __init__(self, base, judge, alpha, group=None, on=0):
        self.base = base
        self.judge = judge
        self.alpha = float(alpha)
        self.group = group
        self.on = int(on)
        term = f"{judge.name}" if self.on else f"(1-{judge.name})"
        self.name = f"bonus:{base.name}+{alpha}*{term}@{group}"

    def __call__(self, prompts, responses, groups=None, references=None):
        total = np.asarray(self.base(prompts, responses, groups, references), dtype=float)
        if self.alpha == 0:
            return total
        labels = np.asarray(self.judge(prompts, responses, references), dtype=float)
        comply = labels if self.on else 1.0 - labels
        if self.group is not None:
            if groups is None:
                raise ValueError(f"bonus on group {self.group!r} needs prompt groups")
            comply = comply * np.asarray([g == self.group for g in groups], dtype=float)
        return total + self.alpha * comply


class LengthBonusReward(Reward):
    """
    Reward with an injected length bias of known strength: ``base + beta *
    min(words, cap) / cap``. Real preference reward models often carry this bias;
    here it is dialled in explicitly so a verifiable length constraint binds by
    construction and ``beta`` traces the frontier. Exposes ``base`` for reporting.
    """

    def __init__(self, base, beta, cap=300):
        self.base = base
        self.beta = float(beta)
        self.cap = int(cap)
        self.name = f"lengthbonus:{base.name}+{beta}*min(words,{cap})/{cap}"

    def __call__(self, prompts, responses, groups=None, references=None):
        total = np.asarray(self.base(prompts, responses, groups, references), dtype=float)
        if self.beta == 0:
            return total
        words = np.asarray([min(len(r.split()), self.cap) / self.cap for r in responses])
        return total + self.beta * words


class CompositeReward(Reward):
    """
    The "fold safety into the reward" baseline: ``base - sum_i lambda_i * judge_i``.

    :param penalties: list of ``(judge, lam, group)``; ``group`` restricts the
        penalty to prompts whose group matches (``None`` applies everywhere), so a
        refusal penalty can be charged on benign prompts only.
    """

    def __init__(self, base, penalties):
        self.base = base
        self.penalties = penalties
        self.name = "composite:" + base.name + ":" + ",".join(
            f"{j.name}@{lam}" for j, lam, _ in penalties)

    def __call__(self, prompts, responses, groups=None, references=None):
        total = np.asarray(self.base(prompts, responses, groups, references), dtype=float)
        for judge, lam, group in self.penalties:
            if lam == 0:
                continue
            if getattr(judge, "needs_groups", False):
                labels = judge(prompts, responses, references, groups=groups)
            else:
                labels = judge(prompts, responses, references)
            if group is not None:
                if groups is None:
                    raise ValueError(f"penalty on group {group!r} needs prompt groups")
                mask = np.asarray([g == group for g in groups], dtype=float)
                labels = labels * mask
            total = total - lam * labels
        return total


class LagrangianReward(CompositeReward):
    """
    Constraint-aware candidate selection: ``base - sum_i lam_i * judge_i`` where each
    ``lam_i`` is raised by dual ascent on the *predicted* constraint value ``g_i``
    (upper bound minus threshold) after every predicted safety test, and lowered
    when the bound has slack. The final safety test is unchanged, so the guarantee
    does not depend on this heuristic; it only makes feasible checkpoints likely.

    :param penalties: list of ``(judge, group)``
    :param lam0: initial multiplier for every constraint
    :param eta: dual step size on a predicted violation; ``lam_i += eta * g_i``
    :param lam_max: cap on each multiplier
    :param lam_floor: once a constraint has been predicted infeasible (``g_i > 0``)
        at any update, its multiplier never drops below this again (0 = off)
    :param eta_down: step size used when the bound has slack (``g_i < 0``);
        ``None`` means ``eta`` (symmetric ascent / descent), ``0`` freezes the
        multiplier once raised
    :param floor_always: apply ``lam_floor`` from the first update, not only after
        a constraint has been predicted infeasible (Round 6 B1: with the floor
        armed only by an infeasible prediction, the multiplier decayed to zero
        while the policy was still far below the threshold and the first
        excursion above it went unpenalised)
    """

    def __init__(self, base, penalties, names, lam0=1.0, eta=10.0, lam_max=20.0,
                 lam_floor=0.0, eta_down=None, floor_always=False):
        super().__init__(base, [(judge, lam0, group) for judge, group in penalties])
        self.names = list(names)
        self.eta = eta
        self.lam_max = lam_max
        self.lam_floor = float(lam_floor)
        self.eta_down = eta if eta_down is None else float(eta_down)
        self.floor_always = bool(floor_always)
        self.bound_seen = {n: False for n in self.names}
        self.name = "lagrangian:" + base.name + ":" + ",".join(
            f"{j.name}@{lam0}" for j, _, _ in self.penalties)

    @property
    def lambdas(self):
        return {n: lam for n, (_, lam, _) in zip(self.names, self.penalties)}

    def update(self, g):
        """Dual-ascent step from the predicted ``g`` values; returns the new multipliers."""
        new = []
        for name, (judge, lam, group) in zip(self.names, self.penalties):
            if name in g and np.isfinite(g[name]):
                if g[name] > 0:
                    self.bound_seen[name] = True
                    lam = lam + self.eta * g[name]
                else:
                    lam = lam + self.eta_down * g[name]
                floor = self.lam_floor if (self.floor_always or self.bound_seen[name]) else 0.0
                lam = float(np.clip(lam, floor, self.lam_max))
            new.append((judge, lam, group))
        self.penalties = new
        return self.lambdas


class PIReward(Reward):
    """
    Proportional-plus-integral steering of candidate selection for one 0/1 constraint
    (formulation doc's steering rule; spikes 023c and 024 on the bandit, 025 on an LLM).

    Each response's reward becomes ``r_i - lam_t (v_i - rbar)`` on the constrained group,
    with ``lam_t = c0 U'(rbar) sigmoid(c_t / T) + lam_dual_t``,
    ``lam_dual_t = clip(lam_dual_{t-1} + eta_i c_t, 0, lam_cap)`` and
    ``c_t = U(rbar) - threshold``, where ``U`` is the smooth Clopper-Pearson upper limit at
    the predicted test's size ``n`` (:func:`seldonian.llm.smooth_bounds.cp_upper`). The
    proportional term has no lag; the integral term adapts to the reward scale.

    ``rbar`` is a running rate. With ``steer_on="batch"`` it is an exponential average of
    the training batches' own labels, updated after each call (spike 023's bandit). With
    ``steer_on="slice"`` it is set only through :meth:`observe`, which a training loop calls
    with labels from a held-out steering slice (the formulation doc's P11 paragraph), and
    batches are judged only for the penalty. ``c`` and ``lam`` always come from state before
    the current batch. Exposes ``base`` so reporting unwraps it; keeps ``history``.
    """

    def __init__(self, base, judge, threshold, n, delta, group=None, c0=20.0, T=0.02,
                 eta_i=10.0, lam_cap=200.0, ema=0.2, steer_on="batch"):
        if steer_on not in ("batch", "slice"):
            raise ValueError("steer_on must be 'batch' or 'slice'")
        self.base, self.judge, self.group = base, judge, group
        self.threshold, self.n, self.delta = float(threshold), int(n), float(delta)
        self.c0, self.T, self.eta_i, self.lam_cap = float(c0), float(T), float(eta_i), lam_cap
        self.ema, self.steer_on = float(ema), steer_on
        self.rate = None
        self.lam_dual = 0.0
        self.history = []
        self.name = f"pi:{base.name}:{judge.name}@c0={c0},T={T},eta_i={eta_i}"

    def observe(self, labels):
        """Fold labels (0/1) into the running rate; returns the new rate."""
        m = float(np.mean(labels))
        self.rate = m if self.rate is None else (1 - self.ema) * self.rate + self.ema * m
        return self.rate

    def state(self):
        """``(c, U'(rate))`` at the current running rate."""
        import torch
        from seldonian.llm.smooth_bounds import cp_upper
        r = torch.tensor(self.rate, dtype=torch.float64, requires_grad=True)
        u = cp_upper(r, self.n, self.delta)
        u.backward()
        return float(u.detach()) - self.threshold, float(r.grad)

    def __call__(self, prompts, responses, groups=None, references=None):
        total = np.asarray(self.base(prompts, responses, groups, references), dtype=float)
        mask = np.ones(len(prompts), dtype=bool)
        if self.group is not None:
            if groups is None:
                raise ValueError(f"penalty on group {self.group!r} needs prompt groups")
            mask = np.asarray([g == self.group for g in groups])
        if not mask.any():
            return total
        idx = np.nonzero(mask)[0]
        v = np.asarray(self.judge([prompts[i] for i in idx], [responses[i] for i in idx],
                                  None if references is None else
                                  [references[i] for i in idx]), dtype=float)
        if self.rate is None:
            if self.steer_on == "slice":
                return total            # no steering-slice reading yet: no penalty
            self.observe(v)
        c, du = self.state()
        self.lam_dual = float(np.clip(self.lam_dual + self.eta_i * c, 0.0, self.lam_cap))
        lam = self.c0 * du / (1.0 + np.exp(-c / self.T)) + self.lam_dual
        total = total.copy()
        total[idx] -= lam * (v - self.rate)
        self.history.append(dict(rate=self.rate, c=c, lam=lam, lam_dual=self.lam_dual,
                                 batch_rate=float(v.mean())))
        if self.steer_on == "batch":
            self.observe(v)
        return total
