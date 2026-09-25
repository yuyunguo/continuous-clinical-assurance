"""M3: windowed streams, per-window statistics, detectors, and alert episodes."""

from dataclasses import replace

import numpy as np
import pytest

from src.eval_module.metrics import auroc, brier, calibration_slope, ece, ppv, sensitivity, specificity
from src.monitor_module import (LABEL_DEPENDENT, apply_label_lag, build_stream, cusum_alarms, episode_starts,
                                progress_schedule, standardize, window_statistics)
from src.sim_module import PerturbationConfig, PerturbationFactory


def small_cfg(cfg):
    """A small stream geometry so the tests stay fast."""
    return replace(cfg, stream=replace(cfg.stream, cases_per_window=300, baseline_windows=12, monitored_windows=30))


def test_progress_schedules():
    assert list(progress_schedule("abrupt", 3, 8, 4)) == [0, 0, 0, 1, 1, 1, 1, 1]
    ramp = progress_schedule("gradual", 2, 10, 4)
    assert list(ramp[:2]) == [0, 0] and ramp[2] == 0.25 and ramp[5] == 1.0 and ramp[-1] == 1.0 and np.all(np.diff(ramp) >= 0)
    steps = progress_schedule("incremental", 2, 10, 6)
    assert set(np.round(np.unique(steps), 3)) == {0.0, round(1 / 3, 3), round(2 / 3, 3), 1.0}


def test_stream_shapes_and_perturbation_starts_at_s0(world, model, cfg):
    c = small_cfg(cfg)
    pert = PerturbationFactory("class_prior_shift")(PerturbationConfig(level_value=3.0))
    stream = build_stream(world, model, pert, "abrupt", 10, c, np.random.default_rng(1))
    total = c.stream.baseline_windows + c.stream.monitored_windows
    assert stream.y.shape == (total, 300) and stream.x_model.shape == (total, 300, c.generator.n_features)
    prevalence = stream.y.mean(axis=1)
    assert prevalence[: c.stream.baseline_windows + 10].mean() < 0.2 < prevalence[c.stream.baseline_windows + 10:].mean()


def test_vectorized_statistics_equal_the_scalar_metrics(world, model, cfg):
    c = small_cfg(cfg)
    stream = build_stream(world, model, None, "abrupt", 5, c, np.random.default_rng(2))
    stats = window_statistics(stream, c)
    for w in (0, 7, 25):
        y, d, p = stream.y[w], stream.decision[w], stream.reported_p[w]
        assert stats["P1"][w] == pytest.approx(sensitivity(y, d)) and stats["P2"][w] == pytest.approx(specificity(y, d))
        assert stats["P3"][w] == pytest.approx(ppv(y, d)) and stats["P4"][w] == pytest.approx(auroc(y, p))
        assert stats["U1"][w] == pytest.approx(ece(y, p)) and stats["U2"][w] == pytest.approx(brier(y, p))
        assert stats["U3"][w] == pytest.approx(abs(calibration_slope(y, p) - 1.0), abs=1e-3)


def test_label_free_indicators_are_not_lagged():
    assert "U5" not in LABEL_DEPENDENT and "C2" not in LABEL_DEPENDENT and {"P1", "U1"} <= set(LABEL_DEPENDENT)


def test_label_lag_delays_label_dependent_series():
    z = np.arange(10, dtype=float)[None, :, None]
    lagged = apply_label_lag(z, baseline=4, lag=2)
    assert lagged.shape == (1, 6, 1) and list(lagged[0, :, 0]) == [2, 3, 4, 5, 6, 7]
    assert list(apply_label_lag(z, baseline=4, lag=0)[0, :, 0]) == [4, 5, 6, 7, 8, 9]


def test_cusum_is_quiet_on_noise_and_fires_on_a_step():
    rng = np.random.default_rng(3)
    quiet = rng.standard_normal((200, 60, 1))
    stepped = quiet.copy()
    stepped[:, 30:, :] += 2.0
    assert cusum_alarms(quiet, h=8.0, k_ref=0.5).mean() < 0.01
    hits = cusum_alarms(stepped, h=8.0, k_ref=0.5)
    assert hits[:, 30:, 0].any(axis=1).mean() > 0.95 and hits[:, :30, 0].mean() < 0.01


def test_episode_starts_respect_hysteresis():
    events = np.array([[0, 1, 1, 0, 0, 0, 1, 0, 1, 0]], dtype=bool)
    assert list(np.flatnonzero(episode_starts(events, j=2)[0])) == [1, 6]      # the event at 8 continues the episode begun at 6
    assert list(np.flatnonzero(episode_starts(events, j=1)[0])) == [1, 6, 8]


def test_standardize_uses_the_baseline_windows_only():
    series = np.concatenate([np.full(10, 5.0) + np.tile([-1.0, 1.0], 5), np.full(6, 9.0)])[None, :, None]
    z = standardize(series, baseline=10, direction=np.array([1.0]))
    assert z.shape == (1, 6, 1) and z[0, :, 0].min() > 3.0


def test_cusum_restarts_after_an_alarm():
    z = np.full((1, 16, 1), 2.0)
    alarms = cusum_alarms(z, h=5.0, k_ref=0.5)
    assert list(np.flatnonzero(alarms[0, :, 0])) == [3, 7, 11, 15]


def test_remaining_statistics_match_independent_formulas(world, model, cfg):
    c = small_cfg(cfg)
    stream = build_stream(world, model, None, "abrupt", 5, c, np.random.default_rng(4))
    stats = window_statistics(stream, c)
    weights = np.asarray(c.harm.weights)
    for w in (1, 14, 30):
        y, d, p, sev, g = stream.y[w], stream.decision[w], stream.reported_p[w], stream.severity[w], stream.g[w]
        err = d != y
        factor = np.where((y == 1) & (d == 0), c.harm.fn_factor, 1.0)
        assert stats["P5"][w] == pytest.approx(1000 * np.mean(weights[sev - 1] * factor * err))
        assert stats["P6"][w] == pytest.approx(abs(err[g == 1].mean() - err[g == 0].mean()))
        conf = np.where(d == 1, p, 1 - p)
        assert stats["U4"][w] == pytest.approx(err[conf >= 0.9].mean())
        base_mean = stream.x_model[: stream.n_baseline].mean(axis=(0, 1))
        assert stats["C2"][w] == pytest.approx(len(y) * np.sum((stream.x_model[w].mean(axis=0) - base_mean) ** 2))


def test_ks_and_input_statistics_react_to_distribution_change(world, model, cfg):
    c = small_cfg(cfg)
    pert = PerturbationFactory("covariate_mean_shift")(PerturbationConfig(level_value=1.0))
    stream = build_stream(world, model, pert, "abrupt", 10, c, np.random.default_rng(5))
    stats = window_statistics(stream, c)
    base = c.stream.baseline_windows
    assert stats["U5"][base + 12:].mean() > 3 * stats["U5"][:base].mean()
    assert stats["C2"][base + 12:].mean() > 3 * stats["C2"][:base].mean()


@pytest.mark.parametrize("scenario_type, level, columns", [
    ("outcome_model_change", 0.40, ("P1", "P2", "P4", "U1", "U3")),
    ("covariate_mean_shift", 1.0, ("C2", "U5")),
])
def test_bad_changes_give_positive_scores_after_the_deviation(world, model, cfg, scenario_type, level, columns):
    """The direction of each indicator is right: a real deviation makes its standardized score clearly positive."""
    from src.monitor_module import INDICATOR_ORDER, stream_scores
    c = small_cfg(cfg)
    pert = PerturbationFactory(scenario_type)(PerturbationConfig(level_value=level))
    scores = np.mean([stream_scores(build_stream(world, model, pert, "abrupt", 6, c, np.random.default_rng(10 + i)), c)[12:] for i in range(6)], axis=0)
    for name in columns:
        assert scores[:, INDICATOR_ORDER.index(name)].mean() > 1.0, name


def test_ks_statistic_equals_a_brute_force_computation(world, model, cfg):
    """Detection standardizes scores, so scale errors are invisible to behavior. Check the value directly."""
    c = small_cfg(cfg)
    stream = build_stream(world, model, None, "abrupt", 5, c, np.random.default_rng(8))
    stats = window_statistics(stream, c)
    reference = stream.reported_p[: stream.n_baseline].ravel()
    for w in (2, 15):
        v = np.sort(stream.reported_p[w])
        f_ref = np.array([(reference <= x).mean() for x in v])
        m = len(v)
        brute = max(np.abs(np.arange(1, m + 1) / m - f_ref).max(), np.abs(np.arange(0, m) / m - f_ref).max())
        assert stats["U5"][w] == pytest.approx(brute)
