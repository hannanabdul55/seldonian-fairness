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
    :param eta: dual step size; ``lam_i += eta * g_i``
    :param lam_max: cap on each multiplier
    """

    def __init__(self, base, penalties, names, lam0=1.0, eta=10.0, lam_max=20.0):
        super().__init__(base, [(judge, lam0, group) for judge, group in penalties])
        self.names = list(names)
        self.eta = eta
        self.lam_max = lam_max
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
                lam = float(np.clip(lam + self.eta * g[name], 0.0, self.lam_max))
            new.append((judge, lam, group))
        self.penalties = new
        return self.lambdas
