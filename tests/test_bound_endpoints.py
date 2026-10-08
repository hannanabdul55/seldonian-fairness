"""End-point checks of every upper bound the certification paper uses (audit of 2026-10-06).

A coverage study cannot find a bound that is wrong only when the sample holds no positive, or
only positives: the stratified Wilson-type bound of spike 013 returned its estimate there and the
error stood as a finding in the paper. Each bound is checked here at zero positives and at all
positives against a closed form, or against the value a certificate must fall back to (1).
The bounds the paper lists as failing are pinned at their degenerate value, so a change to one of
them is noticed."""
import os
import sys

import numpy as np
import pytest
from scipy.stats import norm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for d in ("scripts", ".planning/spikes/013-stratified-safety-set",
          ".planning/spikes/017-calibration-carrying-certificate",
          ".planning/spikes/020-agentdojo-injection-certificate"):
    sys.path.insert(0, os.path.join(ROOT, d))

import agentdojo_recheck as AR     # noqa: E402
import cert017 as c                # noqa: E402
import p9_certificate as P9        # noqa: E402
import stratbounds as SB           # noqa: E402
import stratppi_baseline as BL     # noqa: E402
import wilson_exact as WE          # noqa: E402
from seldonian.bounds import BOUNDS  # noqa: E402

DELTAS = (0.05, 0.1)
GRID = 2.6e-4                      # b1w searches a grid of 1 / 4000 of the distance to 1


def wilson(k, n, delta):
    return float(WE.wilson_upper(k, n, delta))


class TestStratifiedWilson:
    @pytest.mark.parametrize("delta", DELTAS)
    @pytest.mark.parametrize("n", [25, 100, 200, 400])
    def test_one_stratum_is_the_wilson_limit_at_every_count(self, n, delta):
        for k in range(n + 1):
            got = SB.b1w(k, n, 1.0, delta)
            assert wilson(k, n, delta) - 1e-9 <= got <= wilson(k, n, delta) + GRID, (k, got)

    @pytest.mark.parametrize("delta", DELTAS)
    def test_zero_positives_is_the_closed_form(self, delta):
        z2 = norm.ppf(1 - delta) ** 2
        assert SB.b1w(0, 100, 1.0, delta) == pytest.approx(z2 / (100 + z2), abs=GRID)

    @pytest.mark.parametrize("delta", DELTAS)
    def test_pure_strata_do_not_collapse_onto_the_estimate(self, delta):
        n, W = [25] * 4, [0.25] * 4
        assert SB.b1w([0] * 4, n, W, delta) > 0.01                    # no positive anywhere
        assert SB.b1w([0, 0, 25, 25], n, W, delta) > 0.5 + 0.005      # every stratum all-zero or all-one
        assert SB.b1w([25] * 4, n, W, delta) == 1.0

    def test_other_stratified_bounds_are_positive_at_zero(self):
        n, W = [25] * 4, [0.25] * 4
        assert SB.b1([0] * 4, n, W, 0.05) > 0.01
        assert SB.b2([0] * 4, n, W, 0.05) > 0.01


class TestLabelsAlone:
    @pytest.mark.parametrize("delta", DELTAS)
    @pytest.mark.parametrize("n", [59, 100, 299])
    def test_clopper_pearson_end_points(self, n, delta):
        assert float(c.cp_upper(0, n, delta)) == pytest.approx(1 - delta ** (1 / n))
        assert float(c.cp_upper(n, n, delta)) == 1.0
        assert float(c.cp_lower(0, n, delta)) == 0.0
        assert float(c.cp_lower(n, n, delta)) == pytest.approx(delta ** (1 / n))

    @pytest.mark.parametrize("name", ["clopper_pearson", "bentkus", "chernoff_kl", "convex_order", "betting",
                                      "betting_mixture", "anderson", "hoeffdings", "empirical_bernstein"])
    def test_library_bounds_cover_the_rule_of_three_at_zero(self, name):
        # no valid upper limit at zero positives can sit under the exact one, 1 - delta^(1/n)
        n, delta = 100, 0.05
        assert float(BOUNDS[name](np.zeros(n), delta).upper) >= 1 - delta ** (1 / n) - 1e-6

    def test_exact_enumeration_of_clopper_pearson_never_exceeds_delta(self):
        for delta in DELTAS:
            for n in (59, 100, 299):
                assert WE.peaks(WE.cp_upper, n, delta)[1].max() <= delta + 1e-9


class TestJudgeAssisted:
    def setup_method(self):
        rng = np.random.default_rng(0)
        self.f, self.fu = rng.random((1, 100)), rng.random((1, 500))

    def test_bootstrap_t_is_vacuous_when_the_labels_are_constant(self):
        assert float(c.ppipp_boot(np.zeros((1, 100)), self.f, self.fu, 0.05)[0]) == 1.0
        assert float(c.ppipp_boot(np.ones((1, 100)), self.f, self.fu, 0.05)[0]) == 1.0

    def test_wilson_form_at_zero_positives(self):
        z2 = norm.ppf(0.95) ** 2
        assert float(c.ppipp_wilson(np.zeros((1, 100)), self.f, self.fu, 0.05)[0]) == pytest.approx(z2 / (100 + z2))

    def test_stratified_bootstrap_t_is_vacuous_at_zero(self):
        st, pool, W = np.repeat(np.arange(4), 25), [(100, 0.5, 0.1)] * 4, np.full(4, 0.25)
        ub, _ = BL.stratppi_boot(np.zeros(100), self.f[0], st, pool, W, (0.05,), np.random.default_rng(1))
        assert ub == [1.0]

    def test_the_normal_limits_the_paper_rejects_return_zero(self):
        # pinned, not endorsed: PPI++ and StratPPI with a normal quantile have zero width at zero positives
        st, pool, W = np.repeat(np.arange(4), 25), [(100, 0.5, 0.1)] * 4, np.full(4, 0.25)
        assert float(c.ppipp_clt(np.zeros((1, 100)), self.f, self.fu, 0.05)[0]) == 0.0
        assert float(BL.stratppi(np.zeros(100), self.f[0], st, pool, W, (0.05,))[0][0]) == 0.0
        assert float(BOUNDS["ttest"](np.zeros(100), 0.05).upper) == 0.0


class TestClusteredAndPaired:
    def test_cluster_bootstrap_t_is_vacuous_at_zero_and_at_all(self):
        rng = np.random.default_rng(0)
        sizes = np.full(20, 10)
        assert AR.cluster_t_arr(sizes, np.zeros(20), 0.05, rng, 500) == 1.0
        assert AR.cluster_t_arr(sizes, np.full(20, 10.0), 0.05, rng, 500) == 1.0

    def test_two_way_basic_bootstrap_returns_zero_at_zero_successes(self):
        # pinned, not endorsed: the paper lists this bound as failing
        Y, V = np.zeros((6, 5)), np.ones((6, 5), bool)
        assert AR.twoway_arr(Y, V, 0.05, np.random.default_rng(0), 500) == 0.0

    def test_paired_bootstrap_t_option_is_vacuous_when_resamples_are_mostly_degenerate(self):
        # one -1 in 50 differences: 36% of resamples are all zero and sit above the estimate, so the
        # sign rule sends them to +inf and leaves a tight limit (audit item 7). ``degenerate="low"``
        # sends them to the lower tail.
        d = np.zeros(50)
        d[0] = -1
        x = (d + 1) / 2
        assert 2 * float(c.ppipp_boot(x, np.zeros(50), np.zeros(40), 0.05, boots=2000, degenerate="low")[0]) - 1 == 1.0
        assert P9.betting_upper(d, 0.05) > 0.05                       # the exact limit, for scale

    def test_paired_certificate_limits_are_vacuous_in_the_same_case(self):
        # scripts/p9_certificate.py, amendment 2.2 of the paper plan (2026-10-07): before it both
        # limits returned 0.004 here, where the exact betting limit is 0.076
        d = np.zeros(50)
        d[0] = -1
        assert P9.pool_upper(d, np.zeros(50), 0.0, 0.0, 0.05, boots=2000)[0] == 1.0
        assert P9.new_prompts_upper(d, np.zeros(50), np.zeros(40), 0.05, boots=2000) == 1.0


class TestCarriedCalibration:
    def test_a_negative_estimate_refuses(self):
        # recall 200 / 225, false alarms 100 / 225, and the target flags only 5%: the carried
        # false-alarm rate is above the flag rate, which the model rules out (audit item 8)
        flags = np.zeros(2000)
        flags[:100] = 1
        cal = (200, 225, 100, 225)
        arr = tuple(np.array([x]) for x in cal)
        assert float(c.youden(None, None, flags[None], 0.05, cal=arr)[0]) == 0.0           # the study's estimator
        assert float(c.youden(None, None, flags[None], 0.05, cal=arr, refuse_negative=True)[0]) == 1.0
        card = c.certify_carried(0.05, 0.05, flags, cal)
        assert card["certified"] is False and card["upper"] == 1.0


class TestDefaults:
    def test_language_model_path_no_longer_defaults_to_the_t_bound(self):
        import inspect

        from seldonian.llm.constraints import ExpressionConstraint, PairedDifferenceConstraint, rate_constraint
        from seldonian.llm.policy import BOUNDS as LLM_BOUNDS, DEFAULT_BOUND, DEFAULT_BOUNDED, Constraint
        assert (DEFAULT_BOUND, DEFAULT_BOUNDED) == ("clopper_pearson", "bentkus")
        assert Constraint("c", None, 0.1).bound == DEFAULT_BOUND
        default = lambda fn: inspect.signature(fn).parameters["bound"].default      # noqa: E731
        assert default(rate_constraint) == DEFAULT_BOUND
        assert default(ExpressionConstraint.__init__) == default(PairedDifferenceConstraint.__init__) == DEFAULT_BOUNDED
        for name in (DEFAULT_BOUND, DEFAULT_BOUNDED):                 # neither collapses at zero positives
            assert float(LLM_BOUNDS[name](np.zeros(100), 0.05).upper) >= 1 - 0.05 ** 0.01 - 1e-6

