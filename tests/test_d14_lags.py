"""D14: label lags 0, 2, and 8 from one pass over the streams."""

from dataclasses import replace

import numpy as np

from src.monitor_module import INDICATOR_ORDER, LABEL_DEPENDENT, build_stream, null_scores, null_scores_multi, stream_scores, stream_scores_multi


def small_cfg(cfg, lag=2):
    return replace(cfg, stream=replace(cfg.stream, cases_per_window=250, baseline_windows=16, monitored_windows=30, label_lag=lag))


def test_multi_lag_scores_equal_single_lag_scores(world, model, cfg):
    c = small_cfg(cfg)
    stream = build_stream(world, model, None, "abrupt", 5, c, np.random.default_rng(3))
    multi = stream_scores_multi(stream, c, (0, 2, 8))
    for lag in (0, 2, 8):
        single = stream_scores(stream, small_cfg(cfg, lag))
        assert np.allclose(multi[lag], single)


def test_only_label_dependent_columns_move_with_the_lag(world, model, cfg):
    c = small_cfg(cfg)
    stream = build_stream(world, model, None, "abrupt", 5, c, np.random.default_rng(4))
    m = stream_scores_multi(stream, c, (0, 2))
    for k, name in enumerate(INDICATOR_ORDER):
        if name in LABEL_DEPENDENT:
            assert np.allclose(m[2][2:, k], m[0][:-2, k])          # arrives two windows later
        else:
            assert np.array_equal(m[2][:, k], m[0][:, k])           # label-free indicators do not wait


def test_null_scores_multi_matches_null_scores(world, model, cfg):
    c = small_cfg(cfg)
    multi = null_scores_multi(world, model, c, streams=6, seed=11, lags=(0, 2))
    assert multi[2].shape == (6, 30, len(INDICATOR_ORDER))
    assert np.allclose(multi[2], null_scores(world, model, c, streams=6, seed=11))
