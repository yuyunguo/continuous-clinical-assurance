"""D21: the rolling four-window calibration statistic (a pre-specified sensitivity for RH2)."""

import numpy as np
import pytest

from src.monitor_module.rolling import ROLLING_ORDER, rolling_scores_multi, rolling_statistics
from src.monitor_module.statistics import _ece, _slope_deviation, window_statistics
from src.monitor_module.streams import build_stream, progress_schedule
from src.sim_module import PerturbationConfig, PerturbationFactory


@pytest.fixture(scope="module")
def null_stream(cfg, world, model):
    return build_stream(world, model, None, "abrupt", 20, cfg, np.random.default_rng(3))


def test_rolling_statistic_equals_the_metric_on_the_pooled_trailing_windows(null_stream):
    stats = rolling_statistics(null_stream, span=4)
    assert set(stats) == set(ROLLING_ORDER)
    w = 70
    y = null_stream.y[w - 3:w + 1].reshape(1, -1)
    p = null_stream.reported_p[w - 3:w + 1].reshape(1, -1)
    assert stats["U1r"][w] == pytest.approx(_ece(y, p)[0])
    assert stats["U2r"][w] == pytest.approx(((p - y) ** 2).mean())
    assert stats["U3r"][w] == pytest.approx(_slope_deviation(y, p)[0])
    assert np.isnan(stats["U1r"][:3]).all() and not np.isnan(stats["U1r"][3:]).any()


def test_pooling_four_windows_reduces_baseline_noise(cfg, null_stream):
    stats = rolling_statistics(null_stream, span=4)
    plain = window_statistics(null_stream, cfg)
    b = null_stream.n_baseline
    assert np.nanstd(stats["U2r"][b - 30:b]) < 0.7 * np.nanstd(plain["U2"][b - 30:b])


def test_span_one_reproduces_the_per_window_statistics(cfg, null_stream):
    one = rolling_statistics(null_stream, span=1)
    plain = window_statistics(null_stream, cfg)
    assert np.allclose(one["U1r"], plain["U1"]) and np.allclose(one["U2r"], plain["U2"])


def test_scores_are_all_label_lagged_and_respond_to_calibration_drift(cfg, world, model):
    pert = PerturbationFactory("outcome_model_change")(PerturbationConfig(level_value=0.41))
    stream = build_stream(world, model, pert, "abrupt", 20, cfg, np.random.default_rng(4))
    z = rolling_scores_multi(stream, (0, 8), span=4)
    monitored = cfg.stream.monitored_windows
    assert z[0].shape == (monitored, len(ROLLING_ORDER))
    assert np.allclose(z[8][8:], z[0][:-8])                 # every rolling statistic needs outcomes, so all of them arrive with the lag
    assert z[0][30:60, ROLLING_ORDER.index("U2r")].mean() > 3.0
    assert progress_schedule("abrupt", 20, monitored, cfg.stream.ramp_windows)[20] == 1.0
