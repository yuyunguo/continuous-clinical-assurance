"""Tests for `bootstrap_pvalue` and `holm_correction` (manuscript Methods Sec. 2.6): the Holm-Bonferroni multiplicity
correction applied to the Tier 2 real-data replicate comparisons before an arm's reduction is described as
statistically distinguishable from no effect.
"""

import numpy as np
import pytest

from src.eval_module.tafd import arm_reductions_with_holm_correction, bootstrap_difference, bootstrap_pvalue, holm_correction


def test_bootstrap_pvalue_is_small_for_a_clear_true_difference():
    rng = np.random.default_rng(0)
    reference = np.random.default_rng(1).normal(10.0, 1.0, size=200)   # clearly larger TAFD
    candidate = np.random.default_rng(2).normal(2.0, 1.0, size=200)    # clearly smaller TAFD
    p = bootstrap_pvalue(reference, candidate, rng)
    assert p < 0.01


def test_bootstrap_pvalue_is_not_significant_when_there_is_no_real_difference():
    """Independent draws from the SAME distribution (not the same seed -- an identical-seed draw would make
    reference and candidate numerically equal, degenerating the bootstrap difference to exactly 0 everywhere and
    giving a meaningless p=0 at that boundary, not a genuine null test). A large sample keeps the bootstrap p-value
    estimate stable; the assertion checks non-significance (p > 0.05), not an arbitrarily large p."""
    rng = np.random.default_rng(0)
    reference = np.random.default_rng(1).normal(5.0, 2.0, size=2000)
    candidate = np.random.default_rng(6).normal(5.0, 2.0, size=2000)   # same distribution, independent draw
    p = bootstrap_pvalue(reference, candidate, rng)
    assert p > 0.05


def test_bootstrap_pvalue_is_bounded_in_0_1():
    rng = np.random.default_rng(3)
    reference = np.random.default_rng(4).normal(0.0, 1.0, size=50)
    candidate = np.random.default_rng(5).normal(0.0, 1.0, size=50)
    p = bootstrap_pvalue(reference, candidate, rng)
    assert 0.0 <= p <= 1.0


def test_bootstrap_pvalue_equals_twice_the_one_sided_tail_probability():
    """Pins the exact two-sided formula p = 2*min(q, 1-q) (not a one-sided q alone): replicates the same bootstrap
    draws independently (same rng seed) to compute q directly, then checks the relationship holds exactly. Uses a
    MODERATE effect size deliberately (q far from the 0/1 boundary): too large an effect makes q degenerate to
    exactly 0 or 1, where p = q and p = 2*min(q,1-q) coincide trivially and the factor-of-2 property goes untested.
    """
    reference = np.random.default_rng(1).normal(5.3, 2.0, size=60)
    candidate = np.random.default_rng(2).normal(5.0, 2.0, size=60)
    p = bootstrap_pvalue(reference, candidate, np.random.default_rng(42))
    rng2 = np.random.default_rng(42)
    draws = rng2.integers(0, len(reference), size=(2000, len(reference)))
    diff = reference[draws].mean(axis=1) - candidate[draws].mean(axis=1)
    q = float((diff <= 0).mean())
    assert 0.1 < q < 0.9   # sanity: confirms this scenario actually avoids the degenerate boundary
    assert p == pytest.approx(2.0 * min(q, 1.0 - q))


def test_bootstrap_pvalue_symmetric_regardless_of_which_arm_is_reference():
    """p should be the same (up to bootstrap noise) whether reference-candidate or candidate-reference is tested,
    since the two-sided test doesn't care about the sign of the true difference -- checked here with a fixed,
    shared rng seed so the same draws are used both ways, making the two p-values exactly comparable."""
    reference = np.random.default_rng(1).normal(8.0, 1.0, size=300)
    candidate = np.random.default_rng(2).normal(3.0, 1.0, size=300)
    p_forward = bootstrap_pvalue(reference, candidate, np.random.default_rng(9))
    p_backward = bootstrap_pvalue(candidate, reference, np.random.default_rng(9))
    assert abs(p_forward - p_backward) < 0.05


def test_holm_correction_rejects_all_when_every_p_value_is_tiny():
    assert holm_correction([0.0001, 0.0002, 0.0003], alpha=0.05) == [True, True, True]


def test_holm_correction_rejects_none_when_every_p_value_is_large():
    assert holm_correction([0.9, 0.8, 0.95], alpha=0.05) == [False, False, False]


def test_holm_correction_step_down_stops_at_the_first_failure():
    """A classic Holm example: p-values 0.01, 0.02, 0.20, 0.50 at alpha=0.05, m=4.
    Sorted thresholds: 0.05/4=0.0125, 0.05/3=0.0167, 0.05/2=0.025, 0.05/1=0.05.
    p=0.01 <= 0.0125 (reject); p=0.02 > 0.0167 (fail) -- stop: 0.20 and 0.50 are also not rejected."""
    p_values = [0.01, 0.02, 0.20, 0.50]
    assert holm_correction(p_values, alpha=0.05) == [True, False, False, False]


def test_holm_correction_is_less_conservative_than_flat_bonferroni():
    """A p-value that a flat Bonferroni correction (alpha/m) would reject, but that only survives Holm's step-down
    threshold for the smallest rank, must be handled correctly: p=0.02 at m=4 fails flat Bonferroni (0.05/4=0.0125)
    but passes Holm's rank-0 threshold (also 0.0125) only if it truly is the smallest p-value in the family."""
    p_values = [0.02, 0.03, 0.04, 0.045]   # smallest, 0.02, vs rank-0 threshold 0.0125: still fails here
    assert holm_correction(p_values, alpha=0.05) == [False, False, False, False]
    p_values_smaller = [0.01, 0.03, 0.04, 0.045]   # smallest, 0.01, now clears the rank-0 threshold 0.0125
    assert holm_correction(p_values_smaller, alpha=0.05) == [True, False, False, False]


def test_arm_reductions_with_holm_correction_ci_matches_a_standalone_bootstrap_difference_call():
    """The CI in the combined helper's output must be bit-identical to calling `bootstrap_difference` alone with
    the same seed -- i.e. the p-value computation (a separate rng) must not perturb the already-established CI."""
    tafd = {"conventional": np.random.default_rng(1).normal(10.0, 2.0, size=80),
           "arm_a": np.random.default_rng(2).normal(8.0, 2.0, size=80),
           "arm_b": np.random.default_rng(3).normal(9.5, 2.0, size=80)}
    expected_lo, expected_hi = bootstrap_difference(tafd["conventional"], tafd["arm_a"], np.random.default_rng(7))
    result = arm_reductions_with_holm_correction(tafd, ["conventional", "arm_a", "arm_b"], "conventional",
                                                 ci_rng=np.random.default_rng(7), p_rng=np.random.default_rng(99))
    assert result["arm_a"]["lo"] == pytest.approx(expected_lo)
    assert result["arm_a"]["hi"] == pytest.approx(expected_hi)


def test_arm_reductions_with_holm_correction_marks_a_clear_effect_significant_and_a_null_one_not():
    tafd = {"conventional": np.random.default_rng(1).normal(10.0, 1.0, size=200),
           "big_effect": np.random.default_rng(2).normal(2.0, 1.0, size=200),   # huge, unmistakable reduction
           "no_effect": np.random.default_rng(3).normal(10.0, 1.0, size=200)}   # same distribution as conventional
    result = arm_reductions_with_holm_correction(tafd, ["conventional", "big_effect", "no_effect"], "conventional",
                                                 ci_rng=np.random.default_rng(7), p_rng=np.random.default_rng(99))
    assert result["big_effect"]["significant"] == 1.0
    assert result["no_effect"]["significant"] == 0.0
    assert set(result.keys()) == {"big_effect", "no_effect"}   # reference arm itself must not appear in the output


def test_holm_correction_preserves_input_order_in_its_output():
    """The output list must align with the INPUT order of p_values, not the sorted order used internally."""
    p_values = [0.5, 0.0001, 0.9]   # the tiny p-value is in the middle position
    result = holm_correction(p_values, alpha=0.05)
    assert result[1] is True   # the tiny p-value's position must be the one marked rejected
    assert result[0] is False and result[2] is False
