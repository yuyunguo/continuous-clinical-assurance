"""Residual harm exposure: false positives count, and the severity cutoff belongs to EHOR only."""

from dataclasses import replace

import numpy as np

from src.sim_module import Batch, expected_rhe


def manual_batch():
    """Four cases: a class-1 false positive, a class-3 false negative, and two correct decisions."""
    return Batch(x=np.zeros((4, 2)), y=np.array([0, 1, 0, 1]), g=np.zeros(4, dtype=int), severity=np.array([1, 3, 2, 1]),
                 p_hat=np.full(4, 0.5), reported_p=np.full(4, 0.5), decision=np.array([1, 0, 0, 1]))


def no_review(cfg):
    return replace(cfg, reviewer=replace(cfg.reviewer, coverage=0.0))


def test_false_positives_and_false_negatives_both_contribute(cfg):
    weights, fn = cfg.harm.weights, cfg.harm.fn_factor
    expected = 1000.0 * (weights[0] + fn * weights[2]) / 4  # the FP of class 1, and the FN of class 3 with its extra factor
    assert abs(expected_rhe(manual_batch(), no_review(cfg)) - expected) < 1e-9


def test_false_negative_factor_scales_only_missed_deterioration(cfg):
    base = no_review(cfg)
    one, three = replace(base, harm=replace(cfg.harm, fn_factor=1.0)), replace(base, harm=replace(cfg.harm, fn_factor=3.0))
    gain = expected_rhe(manual_batch(), three) - expected_rhe(manual_batch(), one)
    assert abs(gain - 1000.0 * 2.0 * cfg.harm.weights[2] / 4) < 1e-9       # only the class-3 false negative gains, by (3 - 1) x its weight
    only_fp = replace(manual_batch(), y=np.array([0, 0, 0, 0]), decision=np.array([1, 1, 0, 0]), severity=np.array([2, 2, 1, 1]))
    assert abs(expected_rhe(only_fp, three) - expected_rhe(only_fp, one)) < 1e-12


def test_class_1_errors_are_not_ignored(cfg):
    only_fp = replace(manual_batch(), y=np.array([0, 0, 0, 0]), decision=np.array([1, 0, 0, 0]), severity=np.array([1, 1, 1, 1]))
    assert expected_rhe(only_fp, no_review(cfg)) > 0.0


def test_severity_cutoff_does_not_change_rhe(cfg):
    other = replace(no_review(cfg), harm=replace(cfg.harm, severity_cutoff=3))
    assert expected_rhe(manual_batch(), other) == expected_rhe(manual_batch(), no_review(cfg))


def test_interception_reduces_rhe(cfg):
    assert expected_rhe(manual_batch(), cfg) < expected_rhe(manual_batch(), no_review(cfg))
