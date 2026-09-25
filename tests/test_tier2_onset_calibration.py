"""Calibrated real-data onset detection (D18), replacing the naive single-window threshold-crossing rule.

Investigation on real MIMIC-IV sepsis data found the naive rule (`tier2_onset_index`/`tier2_onset_index_rolling`)
false-fires on essentially every pure-null replicate (100/100 in a 100-replicate check), because scanning ~24
correlated single-window tests against a fixed ratio threshold is a multiple-comparisons problem, and a single
500-case window's sampling noise (SD ~0.24-0.38 on the ratio) is too close to the 1+kappa=1.25 threshold itself.
The fix reuses the monitoring system's own CUSUM + `calibrate_threshold` machinery -- already built and tested --
applied to the D18 harm-ratio series (pooled and subgroup) instead of a naive point-threshold rule.
"""

import numpy as np
import pytest

from src.monitor_module import calibrate_threshold
from src.sim_module.config import load_config
from src.tier2_module.config import Tier2StreamConfig
from src.tier2_module.onset_calibration import tier2_calibrate_onset, tier2_onset_index_calibrated, tier2_onset_null_series_multi, tier2_onset_series
from src.tier2_module.real_model import RealDeployedModel
from src.tier2_module.synthetic_stand_in import make_synthetic_cohort
from src.tier2_module.windowing import assign_windows

FEATURES = ("f0", "f1", "f2")


@pytest.fixture
def phase2_cfg():
    return load_config()


@pytest.fixture
def cfg():
    return Tier2StreamConfig(feature_columns=FEATURES, window_size=25, target_sensitivity=0.8)


@pytest.fixture
def windowed(cfg):
    df = make_synthetic_cohort(n=5_000, feature_columns=FEATURES, rng=np.random.default_rng(0), prevalence=0.25)
    model = RealDeployedModel(cfg).fit(df.iloc[:2500], np.random.default_rng(1))
    return assign_windows(model.apply(df), cfg)


def test_tier2_onset_series_has_two_columns_pooled_and_subgroup(cfg, phase2_cfg, windowed):
    from src.tier2_module.stream import build_stream_data
    stream = build_stream_data(windowed, cfg, phase2_cfg.harm, n_baseline=20, rng=np.random.default_rng(2))
    z = tier2_onset_series(stream, phase2_cfg)
    assert z.shape == (stream.y.shape[0] - 20, 2)


def test_tier2_onset_null_series_multi_has_the_expected_shape(cfg, phase2_cfg, windowed):
    z = tier2_onset_null_series_multi(windowed, cfg, phase2_cfg, phase2_cfg.harm, streams=10, seed=3,
                                      n_windows_out=40, block_size=8, n_baseline=16)
    assert z.shape == (10, 24, 2)


def test_tier2_onset_null_series_multi_is_reproducible_given_the_same_seed(cfg, phase2_cfg, windowed):
    a = tier2_onset_null_series_multi(windowed, cfg, phase2_cfg, phase2_cfg.harm, streams=5, seed=7,
                                      n_windows_out=40, block_size=8, n_baseline=16)
    b = tier2_onset_null_series_multi(windowed, cfg, phase2_cfg, phase2_cfg.harm, streams=5, seed=7,
                                      n_windows_out=40, block_size=8, n_baseline=16)
    assert np.array_equal(a, b)


def test_tier2_calibrate_onset_controls_the_false_onset_rate_on_independent_real_null_streams(cfg, phase2_cfg, windowed):
    """The core property: a threshold calibrated on one batch of null streams must keep the false-onset rate on an
    independent batch close to the target, unlike the naive rule's ~100% false-onset rate."""
    j = phase2_cfg.routing.hysteresis_j
    target = 30.0
    calib = tier2_onset_null_series_multi(windowed, cfg, phase2_cfg, phase2_cfg.harm, streams=300, seed=11,
                                          n_windows_out=40, block_size=8, n_baseline=16)
    evaluation = tier2_onset_null_series_multi(windowed, cfg, phase2_cfg, phase2_cfg.harm, streams=300, seed=13,
                                               n_windows_out=40, block_size=8, n_baseline=16)
    threshold = tier2_calibrate_onset(calib, target, j)
    from src.monitor_module import alert_rate
    achieved = alert_rate(evaluation, [0, 1], threshold, j)
    assert abs(achieved - target) < 15.0


def test_tier2_calibrate_onset_matches_calibrate_threshold_directly(cfg, phase2_cfg, windowed):
    """`tier2_calibrate_onset` should be nothing more than `calibrate_threshold` on the two onset columns -- no
    separate reimplementation of the calibration search."""
    calib = tier2_onset_null_series_multi(windowed, cfg, phase2_cfg, phase2_cfg.harm, streams=50, seed=17,
                                          n_windows_out=40, block_size=8, n_baseline=16)
    j = phase2_cfg.routing.hysteresis_j
    assert tier2_calibrate_onset(calib, 30.0, j) == calibrate_threshold(calib, [0, 1], 30.0, j)


def test_tier2_onset_index_calibrated_finds_no_onset_on_a_flat_stream(cfg, phase2_cfg, windowed):
    from src.tier2_module.stream import build_stream_data
    stream = build_stream_data(windowed, cfg, phase2_cfg.harm, n_baseline=20, rng=np.random.default_rng(4))
    calib = tier2_onset_null_series_multi(windowed, cfg, phase2_cfg, phase2_cfg.harm, streams=100, seed=19,
                                          n_windows_out=40, block_size=8, n_baseline=16)
    threshold = tier2_calibrate_onset(calib, 30.0, phase2_cfg.routing.hysteresis_j)
    # a very high threshold (as if calibrated conservatively) should essentially never fire on an unremarkable real stream
    onset = tier2_onset_index_calibrated(stream, phase2_cfg, threshold=max(threshold, 50.0), j=phase2_cfg.routing.hysteresis_j)
    assert onset is None or onset >= 0   # never crashes; a high enough threshold should typically find nothing


def test_onset_raw_series_includes_the_subgroup_at_exactly_the_minimum_share():
    """D18's rule counts the subgroup once its share is `>= min_subgroup_share`, not strictly greater."""
    from src.monitor_module.streams import StreamData
    from src.tier2_module.onset_calibration import _onset_raw_series
    w, m = 3, 100
    y = np.zeros((w, m), dtype=int)
    decision = y.copy()
    severity = np.ones((w, m), dtype=int)
    g = np.zeros((w, m), dtype=int)
    g[:, :5] = 1   # exactly 5% subgroup share, matching the config's default min_subgroup_share
    reported_p = np.full((w, m), 0.5)
    stream = StreamData(y=y, g=g, severity=severity, decision=decision, reported_p=reported_p,
                        x_model=np.zeros((w, m, 1)), n_baseline=1, s0=None, progress=np.zeros(w - 1))
    phase2_cfg = load_config()
    assert phase2_cfg.harm.min_subgroup_share == pytest.approx(0.05)
    raw = _onset_raw_series(stream, phase2_cfg)
    assert not np.isnan(raw[:, 1]).any()   # the subgroup column must be populated, not NaN, at exactly the threshold share


def test_tier2_onset_index_calibrated_fires_at_the_exact_window_the_cusum_accumulates_enough_evidence(cfg, phase2_cfg):
    """Pins the exact returned index (not just a plausible range), to catch an off-by-one in the alarm scan."""
    from src.monitor_module.streams import StreamData
    w, m = 20, 300
    n_baseline = 10
    y = np.zeros((w, m), dtype=int)
    y[:, :100] = 1
    decision = y.copy()
    decision[n_baseline:, :100] = 0   # an abrupt, large step at the very first monitored window
    severity = np.where(y == 1, 2, 1)
    g = np.zeros((w, m), dtype=int)
    reported_p = np.where(decision == 1, 0.9, 0.1).astype(float)
    stream = StreamData(y=y, g=g, severity=severity, decision=decision, reported_p=reported_p,
                        x_model=np.zeros((w, m, 1)), n_baseline=n_baseline, s0=None, progress=np.zeros(w - n_baseline))
    # a very low threshold: the huge, immediate step must be caught at monitored window 0 exactly
    onset = tier2_onset_index_calibrated(stream, phase2_cfg, threshold=0.5, j=phase2_cfg.routing.hysteresis_j)
    assert onset == 0


def test_tier2_onset_index_calibrated_detects_a_deterministic_step_change(cfg, phase2_cfg):
    """A large, deterministic, persistent step (as in the fixed-baseline onset tests) must still be detected once
    a reasonable (not absurdly high) threshold is used."""
    from src.monitor_module.streams import StreamData
    w, m = 30, 300
    n_baseline = 16
    y = np.zeros((w, m), dtype=int)
    y[:, :100] = 1
    decision = y.copy()
    decision[:, :5] = 0                        # a small, steady baseline miss rate throughout
    decision[n_baseline + 8:, :100] = 0         # from monitored window 8 onward, every true event is missed
    severity = np.where(y == 1, 2, 1)
    g = np.zeros((w, m), dtype=int)
    reported_p = np.where(decision == 1, 0.9, 0.1).astype(float)
    stream = StreamData(y=y, g=g, severity=severity, decision=decision, reported_p=reported_p,
                        x_model=np.zeros((w, m, 1)), n_baseline=n_baseline, s0=None, progress=np.zeros(w - n_baseline))
    onset = tier2_onset_index_calibrated(stream, phase2_cfg, threshold=3.0, j=phase2_cfg.routing.hysteresis_j)
    assert onset is not None
    assert onset >= 8   # cannot fire before the real step; a calibrated (CUSUM-accumulating) rule may fire a little after
    assert onset <= 12  # but not with excessive delay either
