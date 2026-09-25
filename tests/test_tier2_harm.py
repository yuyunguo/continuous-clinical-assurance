"""Real-data harm evaluation on a Tier 2 StreamData: expected RHE and onset ratio (decision D18), reusing
`src.sim_module.harm` unchanged via a `Batch` built by flattening StreamData windows.

Unlike Tier 1's exact, near-noiseless synthetic baseline (HARM_POPULATION = 1,000,000 draws), a real baseline is
limited to however many real cases fall in the baseline windows -- a real, flagged noise concern (`Tier2-Readiness.md`).
"""

import numpy as np
import pytest

from src.monitor_module.streams import StreamData
from src.sim_module.config import load_config
from src.sim_module.harm import expected_rhe
from src.tier2_module.harm import batch_from_windows, tier2_onset_index, tier2_onset_index_rolling, tier2_rhe_baselines


@pytest.fixture
def phase2_cfg():
    return load_config()


def _stream(y, g, severity, decision, reported_p, n_baseline):
    """A hand-built StreamData for controlled onset tests. All arrays shape (W, m)."""
    y, g, severity, decision, reported_p = (np.asarray(a) for a in (y, g, severity, decision, reported_p))
    w, m = y.shape
    return StreamData(y=y, g=g, severity=severity, decision=decision, reported_p=reported_p,
                      x_model=np.zeros((w, m, 1)), n_baseline=n_baseline, s0=None, progress=np.zeros(w - n_baseline))


def test_batch_from_windows_flattens_the_selected_windows_in_order():
    y = np.array([[0, 1], [1, 0], [0, 0]])
    stream = _stream(y=y, g=np.zeros_like(y), severity=np.ones_like(y), decision=y, reported_p=np.full(y.shape, 0.5), n_baseline=1)
    batch = batch_from_windows(stream, 1, 3)
    assert list(batch.y) == [1, 0, 0, 0]   # windows 1 and 2, flattened row-major


def test_batch_from_windows_carries_decision_and_reported_p_for_the_reviewer_model():
    y = np.array([[1, 1]])
    decision = np.array([[0, 1]])
    reported_p = np.array([[0.9, 0.1]])
    stream = _stream(y=y, g=np.zeros_like(y), severity=np.full(y.shape, 3), decision=decision, reported_p=reported_p, n_baseline=0)
    batch = batch_from_windows(stream, 0, 1)
    assert list(batch.decision) == [0, 1]
    assert list(batch.reported_p) == [0.9, 0.1]


def test_tier2_rhe_baselines_matches_expected_rhe_computed_directly_on_the_same_batch(phase2_cfg):
    rng = np.random.default_rng(0)
    w, m = 5, 200
    y = (rng.random((w, m)) < 0.2).astype(int)
    decision = (rng.random((w, m)) < 0.2).astype(int)
    severity = rng.integers(1, 4, size=(w, m))
    g = (rng.random((w, m)) < 0.5).astype(int)
    reported_p = rng.random((w, m))
    stream = _stream(y=y, g=g, severity=severity, decision=decision, reported_p=reported_p, n_baseline=5)
    baselines = tier2_rhe_baselines(stream, phase2_cfg)
    direct_batch = batch_from_windows(stream, 0, 5)
    assert baselines["pooled"] == pytest.approx(expected_rhe(direct_batch, phase2_cfg))
    subgroup_batch = direct_batch.select(direct_batch.g == 1)
    assert baselines["subgroup"] == pytest.approx(expected_rhe(subgroup_batch, phase2_cfg))


def test_tier2_onset_index_finds_no_onset_when_harm_never_rises(phase2_cfg):
    rng = np.random.default_rng(1)
    w, m = 10, 300
    y = (rng.random((w, m)) < 0.1).astype(int)
    decision = y.copy()   # perfect decisions throughout: no errors, so harm never rises
    severity = np.ones((w, m), dtype=int)
    g = np.zeros((w, m), dtype=int)
    reported_p = np.where(decision == 1, 0.9, 0.1).astype(float)
    stream = _stream(y=y, g=g, severity=severity, decision=decision, reported_p=reported_p, n_baseline=4)
    baselines = tier2_rhe_baselines(stream, phase2_cfg)
    assert tier2_onset_index(stream, phase2_cfg, baselines) is None


def test_tier2_onset_index_handles_a_zero_real_baseline_explicitly(phase2_cfg):
    """A real, finite baseline can show literally zero expected harm (unlike Tier 1's near-noiseless 1,000,000-case
    synthetic baseline). Any harm at all against a zero baseline is an unbounded relative increase and must count
    as onset, without raising a division-by-zero error."""
    w, m = 6, 100
    n_baseline = 3
    y = np.zeros((w, m), dtype=int)          # no true events anywhere: baseline expected harm is exactly zero
    decision = np.zeros((w, m), dtype=int)
    decision[n_baseline + 1:] = 1            # from monitored window 1 onward, every case is a false alarm
    severity = np.full((w, m), 2)
    g = np.zeros((w, m), dtype=int)
    reported_p = np.full((w, m), 0.5)
    stream = _stream(y=y, g=g, severity=severity, decision=decision, reported_p=reported_p, n_baseline=n_baseline)
    baselines = tier2_rhe_baselines(stream, phase2_cfg)
    assert baselines["pooled"] == 0.0
    onset = tier2_onset_index(stream, phase2_cfg, baselines)
    assert onset == 1


def test_tier2_onset_index_detects_a_step_change_to_severe_errors(phase2_cfg):
    w, m = 10, 300
    n_baseline = 4
    y = np.ones((w, m), dtype=int)         # every case is a true deterioration
    decision = np.ones((w, m), dtype=int)  # correctly caught at baseline
    decision[n_baseline + 2:] = 0          # from monitored window 2 onward, every case is missed (false negative)
    severity = np.full((w, m), 3)          # class 3: the most severe, so the harm rise is unambiguous
    g = np.zeros((w, m), dtype=int)
    reported_p = np.where(decision == 1, 0.9, 0.1).astype(float)
    stream = _stream(y=y, g=g, severity=severity, decision=decision, reported_p=reported_p, n_baseline=n_baseline)
    baselines = tier2_rhe_baselines(stream, phase2_cfg)
    onset = tier2_onset_index(stream, phase2_cfg, baselines)
    assert onset is not None
    assert onset == 2   # the step happens exactly at monitored window 2


def _nonzero_baseline_stream(w: int, n_baseline: int, m: int = 400, n_events: int = 100, baseline_miss: int = 5) -> StreamData:
    """A deterministic (noise-free) stream: `n_events` of `m` cases are true events in every window, with a fixed
    `baseline_miss` of them missed throughout, giving a nonzero real baseline with no cross-window sampling noise."""
    y = np.zeros((w, m), dtype=int)
    y[:, :n_events] = 1
    decision = y.copy()
    decision[:, :baseline_miss] = 0
    severity = np.where(y == 1, 2, 1)
    g = np.zeros((w, m), dtype=int)
    reported_p = np.where(decision == 1, 0.9, 0.1).astype(float)
    return _stream(y=y, g=g, severity=severity, decision=decision, reported_p=reported_p, n_baseline=n_baseline)


def test_tier2_onset_index_uses_the_ratio_threshold_against_a_nonzero_real_baseline(phase2_cfg):
    """The one scenario the tests above never exercise: a baseline with some real, nonzero harm, and a monitored
    rise that must actually clear the 1 + kappa ratio, not just the zero-baseline special case."""
    n_baseline = 4
    stream = _nonzero_baseline_stream(w=12, n_baseline=n_baseline)
    # from monitored window 5 onward (absolute window 9), every true event is missed outright: a large, unambiguous rise
    stream.decision[n_baseline + 5:, :100] = 0
    baselines = tier2_rhe_baselines(stream, phase2_cfg)
    assert baselines["pooled"] > 0.0   # confirms this test actually exercises the ratio branch, not the zero-baseline guard
    onset = tier2_onset_index(stream, phase2_cfg, baselines)
    assert onset == 5


def test_tier2_onset_index_finds_no_onset_when_the_rise_never_clears_kappa(phase2_cfg):
    """The same fixed miss rate throughout, baseline and monitored alike, so the ratio never rises: no onset."""
    stream = _nonzero_baseline_stream(w=8, n_baseline=4)
    baselines = tier2_rhe_baselines(stream, phase2_cfg)
    assert baselines["pooled"] > 0.0
    assert tier2_onset_index(stream, phase2_cfg, baselines) is None


def _smoothly_drifting_stream(w: int, n_baseline: int, m: int = 400, n_events: int = 100,
                              start_miss: int = 20, rate: float = 0.3) -> StreamData:
    """A deterministic stream where the miss count rises smoothly, window by window, across the ENTIRE stream
    (baseline included) -- real calendar drift, with no injected perturbation. Window k's miss count is
    ``round(start_miss + rate * k)`` of the ``n_events`` true events: gentle enough that the relative change over
    any short run of nearby windows stays well under kappa, while the change over many windows does not.
    """
    y = np.zeros((w, m), dtype=int)
    y[:, :n_events] = 1
    decision = y.copy()
    for k in range(w):
        miss = min(n_events, round(start_miss + rate * k))
        decision[k, :miss] = 0
    severity = np.where(y == 1, 2, 1)
    g = np.zeros((w, m), dtype=int)
    reported_p = np.where(decision == 1, 0.9, 0.1).astype(float)
    return _stream(y=y, g=g, severity=severity, decision=decision, reported_p=reported_p, n_baseline=n_baseline)


def test_fixed_baseline_falsely_triggers_onset_on_smooth_real_drift_alone(phase2_cfg):
    """Demonstrates the confound directly: with no injected perturbation, smooth real drift alone is enough to
    cross the fixed early-baseline ratio threshold well before the stream ends."""
    n_baseline = 4
    stream = _smoothly_drifting_stream(w=30, n_baseline=n_baseline)
    baselines = tier2_rhe_baselines(stream, phase2_cfg)
    assert tier2_onset_index(stream, phase2_cfg, baselines) is not None


def test_rolling_baseline_does_not_false_trigger_on_the_same_smooth_real_drift(phase2_cfg):
    """The fix: a short rolling lookback tracks the current level, so the same smooth drift that fools the fixed
    baseline does not cross 1 + kappa relative to its own recent past."""
    n_baseline = 4
    stream = _smoothly_drifting_stream(w=30, n_baseline=n_baseline)
    onset = tier2_onset_index_rolling(stream, phase2_cfg, n_lookback=5)
    assert onset is None


def test_rolling_baseline_still_detects_an_abrupt_persistent_step(phase2_cfg):
    """The trade-off's boundary case: an abrupt, persistent step (not smooth drift) must still be detected, since
    the windows just before the step are still flat and give the rolling baseline an accurate recent reference."""
    n_baseline = 4
    stream = _nonzero_baseline_stream(w=12, n_baseline=n_baseline)
    stream.decision[n_baseline + 5:, :100] = 0   # same abrupt step as the fixed-baseline step-change test
    onset = tier2_onset_index_rolling(stream, phase2_cfg, n_lookback=3)
    assert onset == 5


def test_rolling_baseline_skips_windows_without_enough_lookback_history(phase2_cfg):
    """A lookback longer than the available real history before a window must not crash -- it is skipped until
    enough real windows have accumulated."""
    n_baseline = 2
    stream = _nonzero_baseline_stream(w=10, n_baseline=n_baseline)
    stream.decision[n_baseline + 5:, :100] = 0
    onset = tier2_onset_index_rolling(stream, phase2_cfg, n_lookback=6)   # only 2 baseline windows exist at first
    assert onset == 5


def test_rolling_baseline_handles_a_zero_real_lookback_baseline_explicitly(phase2_cfg):
    """The same zero-baseline edge case as the fixed variant, now for a rolling lookback window."""
    w, m = 8, 100
    n_baseline = 3
    y = np.zeros((w, m), dtype=int)
    decision = np.zeros((w, m), dtype=int)
    decision[n_baseline + 1:] = 1   # from monitored window 1 onward, every case is a false alarm
    severity = np.full((w, m), 2)
    g = np.zeros((w, m), dtype=int)
    reported_p = np.full((w, m), 0.5)
    stream = _stream(y=y, g=g, severity=severity, decision=decision, reported_p=reported_p, n_baseline=n_baseline)
    onset = tier2_onset_index_rolling(stream, phase2_cfg, n_lookback=2)
    assert onset == 1


def test_rolling_baseline_uses_exactly_n_lookback_windows_not_one_more(phase2_cfg):
    """A one-window-too-many lookback would dilute the baseline with an unrelated high-miss window and miss the
    boundary crossing entirely; this pins both the lookback window count and the >= boundary at once."""
    w, m, n_events = 5, 200, 100
    y = np.zeros((w, m), dtype=int)
    y[:, :n_events] = 1
    decision = y.copy()
    miss_by_window = [50, 4, 4, 4, 5]   # window 0's miss=50 must NOT be included in a 3-window lookback for window 4
    for k, miss in enumerate(miss_by_window):
        decision[k, :miss] = 0
    severity = np.where(y == 1, 2, 1)
    g = np.zeros((w, m), dtype=int)
    reported_p = np.where(decision == 1, 0.9, 0.1).astype(float)
    stream = _stream(y=y, g=g, severity=severity, decision=decision, reported_p=reported_p, n_baseline=0)
    # windows 1-3 (miss=4 each) give a baseline harm exactly 4/5 of window 4's (miss=5): ratio 1.25 = 1 + kappa exactly
    onset = tier2_onset_index_rolling(stream, phase2_cfg, n_lookback=3)
    assert onset == 4


def test_rolling_baseline_with_no_fixed_baseline_at_all_skips_only_the_very_first_window(phase2_cfg):
    """With n_baseline=0, monitored window 0 has no real history to look back on and must be skipped; the real
    step later in the stream must still be found correctly."""
    stream = _nonzero_baseline_stream(w=8, n_baseline=0, baseline_miss=5)
    stream.decision[4:, :100] = 0   # a step at monitored window 4
    onset = tier2_onset_index_rolling(stream, phase2_cfg, n_lookback=2)
    assert onset == 4
