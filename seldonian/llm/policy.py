"""
Seldonian candidate selection and safety test for an LLM policy.

The policy is queryable, so the safety test is a direct Monte Carlo estimate: sample
one response per held-out safety prompt, judge it, and bound the mean with
:func:`seldonian.bounds.ttest_bounds` or :func:`seldonian.bounds.hoeffdings_bounds`.
No importance weights are needed. The same estimator on a candidate-set subset with
the doubled interval (``predict=True``) is the *predicted* safety test used to
discard infeasible checkpoints during training.

The training loop itself lives behind :class:`PolicyBackend` so this module has no
model dependencies and the safety logic is unit-testable with a mock backend.
"""
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from seldonian.algorithm import SeldonianAlgorithm
from seldonian.bounds import hoeffdings_bounds, ttest_bounds

BOUNDS = {"ttest": ttest_bounds, "hoeffding": hoeffdings_bounds}


def effective_n(m, n_s):
    """
    Sample size at which to bound (predicted rate - final safety-test rate): the
    prediction uses ``m`` samples and the final test ``n_s``, so the difference has
    variance ``sd^2 (1/m + 1/n_s)``.
    """
    return max(int(1.0 / (1.0 / m + 1.0 / n_s)), 2)


def predicted_width(rate, n_s, delta, bound="ttest", inflation=1.0, m=None):
    """
    Expected one-sided width the *predicted* safety test adds to a 0/1 rate: the
    Student-t (or Hoeffding) interval at :func:`effective_n` (``m`` prediction
    samples, ``n_s`` safety prompts; ``m=None`` means the safety-set size alone),
    times ``inflation``. A relative-threshold margin smaller than this cannot be
    satisfied by a policy at the reference rate.
    """
    from scipy.stats import t as tdist
    n = n_s if m is None else effective_n(m, n_s)
    if bound == "hoeffding":
        return inflation * float(np.sqrt(np.log(1 / delta) / (2 * n)))
    sd = float(np.sqrt(max(rate * (1 - rate), 1e-12)))
    return inflation * sd / np.sqrt(n) * float(tdist.ppf(1 - delta, n - 1))


@dataclass
class Constraint:
    """
    ``g(theta) = P(judge == 1 | prompts in group) - threshold``.

    :param judge: a :class:`seldonian.llm.judges.Judge` (1 = violation event)
    :param threshold: tau; set it after measuring the reference policy for a
        relative constraint (see :meth:`SeldonianLLMPolicy.evaluate`)
    :param group: restrict to prompt records with this ``group``; ``None`` = all
    :param bound: ``"ttest"`` (default) or ``"hoeffding"``
    """
    name: str
    judge: object
    threshold: float
    group: str = None
    bound: str = "ttest"

    def select(self, records):
        if self.group is None:
            return list(range(len(records)))
        return [i for i, r in enumerate(records) if r.get("group") == self.group]


class PolicyBackend(ABC):
    """What :class:`SeldonianLLMPolicy` needs from a trainable LLM."""

    @abstractmethod
    def generate(self, prompts, max_new_tokens=256, temperature=1.0):
        """One sampled response string per prompt string."""

    @abstractmethod
    def train(self, records, reward, on_step):
        """
        Run candidate selection on prompt ``records`` with ``reward`` (a
        :class:`seldonian.llm.rewards.Reward`), calling ``on_step(step)`` after every
        optimizer step.
        """

    @abstractmethod
    def save_checkpoint(self, tag):
        """Persist the current policy parameters; return a handle for :meth:`load_checkpoint`."""

    @abstractmethod
    def load_checkpoint(self, handle):
        """Restore parameters saved by :meth:`save_checkpoint`."""


@dataclass
class PredictedTest:
    step: int
    g: dict
    rates: dict
    upper: dict
    reward: float
    feasible: bool
    checkpoint: object = None
    seconds: float = 0.0
    n_samples: int = 0
    lambdas: dict = None


@dataclass
class SafetyReport:
    g: dict
    rates: dict
    upper: dict
    n: dict
    passed: bool
    reward: float
    mean_length: float
    seconds: float
    extra: dict = field(default_factory=dict)


class SeldonianLLMPolicy(SeldonianAlgorithm):
    """
    :param backend: a :class:`PolicyBackend`
    :param prompts_c: candidate prompt records (training + predicted tests)
    :param prompts_s: safety prompt records. Sealed during :meth:`fit`; read exactly
        once, by the final safety test.
    :param reward: a :class:`seldonian.llm.rewards.Reward`, or ``None``
    :param constraints: list of :class:`Constraint`; ``delta`` is split evenly across them
    :param delta: overall failure probability
    :param predict_every: run the predicted safety test every this many optimizer steps
    :param predict_n: number of candidate prompts sampled for each predicted test
    :param predict_inflation: multiplier on the predicted-test interval, which is
        already computed at the effective size of prediction + safety samples
        (1.0 = no extra inflation; the classification models' convention is 2.0 at
        the safety-set size alone)
    """

    def __init__(self, backend, prompts_c, prompts_s, reward=None, constraints=(),
                 delta=0.05, predict_every=25, predict_n=256, max_new_tokens=256,
                 temperature=1.0, seed=0, verbose=False, predict_inflation=1.0):
        self.backend = backend
        self.prompts_c = list(prompts_c)
        self.prompts_s = list(prompts_s)
        self.reward = reward
        self.constraints = list(constraints)
        self.delta = delta
        self.predict_every = predict_every
        self.predict_n = predict_n
        self.predict_inflation = predict_inflation
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.seed = seed
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)
        self._sealed = False
        self.history = []
        self.selected = None
        self.safety_report = None
        self.safety_tests_run = 0
        overlap = {r["prompt_id"] for r in self.prompts_c} & {r["prompt_id"] for r in self.prompts_s}
        if overlap:
            raise ValueError(f"{len(overlap)} prompt ids appear in both D_c and D_s")

    # -------------------------------------------------------------- helpers

    def _log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    @property
    def delta_each(self):
        return self.delta / max(len(self.constraints), 1)

    def n_safety(self, constraint):
        """Safety-set size the final test of this constraint will use."""
        return len(constraint.select(self.prompts_s))

    def sample(self, records):
        prompts = [r["prompt"] for r in records]
        responses = self.backend.generate(prompts, max_new_tokens=self.max_new_tokens,
                                          temperature=self.temperature)
        if len(responses) != len(prompts):
            raise RuntimeError("backend returned a different number of responses than prompts")
        return list(responses)

    def score_reward(self, records, responses):
        """
        Mean *base* reward: shaped rewards (composite / Lagrangian) unwrap to the
        underlying reward model so checkpoint ranking and reporting are comparable
        across methods.
        """
        if self.reward is None:
            return np.full(len(records), np.nan)
        reward = getattr(self.reward, "base", self.reward)
        return np.asarray(reward([r["prompt"] for r in records], responses,
                                      groups=[r.get("group") for r in records],
                                      references=[r.get("reference") for r in records]),
                          dtype=float)

    def constraint_values(self, records, responses, predict, ub=True):
        """
        Per-constraint ``g`` (positive = violated), rates, upper bounds and the
        ``n`` each bound was computed for. With ``predict=True`` the interval is
        doubled and ``n`` is the safety-set size, as in the classification models.
        """
        g, rates, upper, ns = {}, {}, {}, {}
        for c in self.constraints:
            idx = c.select(records)
            n_s = self.n_safety(c)
            ns[c.name] = n_s
            if len(idx) < 2 or n_s < 2:
                g[c.name], rates[c.name], upper[c.name] = np.inf, np.nan, np.inf
                continue
            labels = c.judge([records[i]["prompt"] for i in idx], [responses[i] for i in idx],
                             [records[i].get("reference") for i in idx])
            labels = np.asarray(labels, dtype=float)
            rates[c.name] = float(labels.mean())
            if ub:
                if predict:
                    # bound (predicted - final) at the effective size of both samples,
                    # then apply the explicit inflation (predict=False here so the
                    # library's own x2 is not stacked on top)
                    rv = BOUNDS[c.bound](labels, self.delta_each, n=effective_n(len(idx), n_s))
                    upper[c.name] = rates[c.name] + self.predict_inflation * (
                        float(rv.upper) - rates[c.name])
                else:
                    rv = BOUNDS[c.bound](labels, self.delta_each, n=n_s)
                    upper[c.name] = float(rv.upper)
            else:
                upper[c.name] = rates[c.name]
            g[c.name] = upper[c.name] - c.threshold
        return g, rates, upper, ns

    # -------------------------------------------------------------- evaluation

    def evaluate(self, records, responses=None):
        """
        Plain rates and mean reward on ``records`` (no bounds, no guarantee). Used to
        set relative thresholds from the reference policy and to report baselines.
        """
        if responses is None:
            responses = self.sample(records)
        _, rates, _, _ = self.constraint_values(records, responses, predict=False, ub=False)
        rewards = self.score_reward(records, responses)
        return {"rates": rates, "reward": float(np.nanmean(rewards)) if len(rewards) else np.nan,
                "mean_length": float(np.mean([len(r) for r in responses])) if responses else 0.0,
                "responses": responses, "rewards": rewards}

    def set_relative_thresholds(self, margins, n=None):
        """
        Measure the *reference* (untrained) policy on a candidate-set subset and set
        each constraint's threshold to ``reference rate + margins[name]``. Must be
        called before :meth:`fit`. Returns the measured rates.
        """
        if self.history:
            raise RuntimeError("set thresholds before training starts")
        records = self._candidate_subset(n or self.predict_n)
        ev = self.evaluate(records)
        for c in self.constraints:
            if c.name in margins:
                c.threshold = ev["rates"][c.name] + margins[c.name]
        return ev["rates"]

    def _candidate_subset(self, n):
        """Group-stratified subset of D_c, so every constraint gets samples."""
        groups = sorted({r.get("group") for r in self.prompts_c}, key=str)
        out = []
        for g in groups:
            members = [r for r in self.prompts_c if r.get("group") == g]
            k = max(2, int(round(n * len(members) / len(self.prompts_c))))
            k = min(k, len(members))
            pick = self.rng.choice(len(members), size=k, replace=False)
            out.extend(members[int(i)] for i in pick)
        return out

    # -------------------------------------------------------------- Seldonian core

    def _safetyTest(self, predict=False, ub=True):
        """
        Max over constraints of the upper-bounded ``g``. ``predict=True`` runs the
        predicted test on a candidate subset; otherwise this *is* the safety test on
        D_s, which is refused while ``fit`` is running and counted so a run cannot
        spend it twice.
        """
        t0 = time.time()
        if predict:
            records = self._candidate_subset(self.predict_n)
        else:
            if self._sealed:
                raise RuntimeError("D_s is sealed during candidate selection")
            if self.safety_tests_run >= 1:
                raise RuntimeError("the safety test has already been run once for this policy")
            records = self.prompts_s
        responses = self.sample(records)
        g, rates, upper, ns = self.constraint_values(records, responses, predict=predict, ub=ub)
        rewards = self.score_reward(records, responses)
        worst = max(g.values()) if g else 0.0
        if predict:
            self._last_prediction = PredictedTest(
                step=-1, g=g, rates=rates, upper=upper, reward=float(np.nanmean(rewards)) if len(rewards) else np.nan,
                feasible=bool(worst <= 0), seconds=time.time() - t0, n_samples=len(records))
        else:
            self.safety_tests_run += 1
            self.safety_report = SafetyReport(
                g=g, rates=rates, upper=upper, n=ns, passed=bool(worst <= 0),
                reward=float(np.nanmean(rewards)) if len(rewards) else np.nan,
                mean_length=float(np.mean([len(r) for r in responses])) if responses else 0.0,
                seconds=time.time() - t0)
            self._safety_episodes = (records, responses, rewards)
        return worst

    def fit(self, seldonian=True, **kwargs):
        """
        Candidate selection, then one safety test.

        With ``seldonian=True`` (default) checkpoints that fail the predicted test are
        never selected, and ``None`` is returned (No Solution Found) when the selected
        checkpoint fails the safety test. With ``seldonian=False`` the final checkpoint
        is kept and no safety test is run; call :meth:`evaluate` for diagnostics.
        """
        if self.safety_tests_run:
            raise RuntimeError("this policy has already been fit and tested")
        best = None
        self._sealed = True
        t0 = time.time()

        def on_step(step):
            nonlocal best
            if not seldonian or not self.constraints or step % self.predict_every != 0:
                return
            self._safetyTest(predict=True)
            pred = self._last_prediction
            pred.step = step
            if pred.feasible and (best is None or pred.reward > best.reward):
                pred.checkpoint = self.backend.save_checkpoint(f"feasible-step{step}")
                best = pred
            if hasattr(self.reward, "update"):
                # constraint-aware candidate selection: dual ascent on the predicted g
                pred.lambdas = self.reward.update(pred.g)
            self.history.append(pred)
            self._log(f"step {step}: predicted g={pred.g} reward={pred.reward:.3f} "
                      f"feasible={pred.feasible} lambdas={pred.lambdas}")

        try:
            self.backend.train(self.prompts_c, self.reward, on_step)
        finally:
            self._sealed = False
        self.train_seconds = time.time() - t0

        if not seldonian:
            self.selected = {"step": "final", "reason": "unconstrained"}
            return self
        if best is not None:
            self.backend.load_checkpoint(best.checkpoint)
            self.selected = {"step": best.step, "reason": "best feasible predicted test",
                             "predicted_reward": best.reward, "predicted_g": best.g}
        else:
            self.selected = {"step": "final", "reason": "no checkpoint passed the predicted test"}
        g = self._safetyTest()
        self._log(f"safety test: g={self.safety_report.g} passed={self.safety_report.passed}")
        return self if g <= 0 else None

    def predict(self, X):
        """Sample responses for prompt strings (or records) with the current policy."""
        records = [x if isinstance(x, dict) else {"prompt": x} for x in X]
        return self.sample(records)

    def data(self):
        return self.prompts_c, self.prompts_s
