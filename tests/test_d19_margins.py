"""D19: absolute TAFD differences in windows, and lead time."""

import numpy as np

from src.eval_module.tafd import bootstrap_difference, classify_replicate, rmst_difference


def test_rmst_difference_is_in_windows_and_positive_when_the_candidate_is_faster():
    ref, cand = np.array([4.0, 6.0, 8.0]), np.array([2.0, 3.0, 4.0])
    assert rmst_difference(ref, cand) == 3.0


def test_bootstrap_difference_is_paired_and_covers_the_estimate():
    rng = np.random.default_rng(0)
    ref = rng.integers(2, 12, 300).astype(float)
    cand = np.maximum(ref - rng.integers(0, 4, 300), 0)   # a varying gain of 1.5 windows on average
    lo, hi = bootstrap_difference(ref, cand, np.random.default_rng(1), n_boot=500)
    assert lo < rmst_difference(ref, cand) < hi and lo > 1.0


def test_lead_time_is_reported_for_anticipatory_detections_only():
    events = np.zeros(40, dtype=bool)
    events[11] = True
    early = classify_replicate(events, s0=10, t0=14, horizon=20, j=2)
    assert early["anticipatory"] and early["lead_time"] == 3
    events = np.zeros(40, dtype=bool)
    events[18] = True
    late = classify_replicate(events, s0=10, t0=14, horizon=20, j=2)
    assert not late["anticipatory"] and late["lead_time"] == 0 and late["tafd"] == 4
