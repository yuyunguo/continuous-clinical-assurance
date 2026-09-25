"""Matched false-alarm calibration on Tier 2 (real or synthetic stand-in) data, via the block-bootstrap null
streams from `resample.py`. Reuses `src.monitor_module`'s indicator/CUSUM/calibration machinery unchanged — it
operates on `StreamData` and `Phase2Config` alone, with no dependence on the Tier 1 synthetic generator.
"""

import numpy as np
import pytest

from src.monitor_module import INDICATOR_ORDER, alert_rate, calibrate_threshold, indicator_indices
from src.sim_module.config import load_config
from src.tier2_module.calibrate import tier2_null_scores_multi
from src.tier2_module.config import Tier2StreamConfig
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
    return assign_windows(model.apply(df), cfg)   # 200 windows of 25 cases


def test_tier2_null_scores_multi_has_the_expected_shape(cfg, phase2_cfg, windowed):
    scores = tier2_null_scores_multi(windowed, cfg, phase2_cfg, phase2_cfg.harm, streams=20, seed=3, lags=(0, 2),
                                     n_windows_out=60, block_size=10, n_baseline=20)
    assert set(scores) == {0, 2}
    assert scores[0].shape == (20, 40, len(INDICATOR_ORDER))   # 20 streams, 60-20=40 monitored windows, K indicators


def test_tier2_null_scores_multi_is_reproducible_given_the_same_seed(cfg, phase2_cfg, windowed):
    a = tier2_null_scores_multi(windowed, cfg, phase2_cfg, phase2_cfg.harm, streams=5, seed=7, lags=(0,),
                                n_windows_out=40, block_size=8, n_baseline=15)
    b = tier2_null_scores_multi(windowed, cfg, phase2_cfg, phase2_cfg.harm, streams=5, seed=7, lags=(0,),
                                n_windows_out=40, block_size=8, n_baseline=15)
    assert np.array_equal(a[0], b[0])


def test_tier2_null_scores_multi_different_seeds_give_different_streams(cfg, phase2_cfg, windowed):
    a = tier2_null_scores_multi(windowed, cfg, phase2_cfg, phase2_cfg.harm, streams=5, seed=7, lags=(0,),
                                n_windows_out=40, block_size=8, n_baseline=15)
    b = tier2_null_scores_multi(windowed, cfg, phase2_cfg, phase2_cfg.harm, streams=5, seed=8, lags=(0,),
                                n_windows_out=40, block_size=8, n_baseline=15)
    assert not np.array_equal(a[0], b[0])


def test_calibrated_threshold_matches_the_target_alert_rate_on_independent_evaluation_streams(cfg, phase2_cfg, windowed):
    """The exact matched-calibration recipe `m4_pilot.py` uses: calibrate on one null-stream batch, check the
    resulting rate on an independent batch is close to the budget."""
    target = 30.0   # decision D2 primary budget, per 1,000 windows
    j = phase2_cfg.routing.hysteresis_j
    calib = tier2_null_scores_multi(windowed, cfg, phase2_cfg, phase2_cfg.harm, streams=300, seed=11, lags=(0,),
                                    n_windows_out=60, block_size=10, n_baseline=20)[0]
    evaluation = tier2_null_scores_multi(windowed, cfg, phase2_cfg, phase2_cfg.harm, streams=300, seed=13, lags=(0,),
                                         n_windows_out=60, block_size=10, n_baseline=20)[0]
    cols = indicator_indices(list(INDICATOR_ORDER))
    threshold = calibrate_threshold(calib, cols, target, j)
    achieved = alert_rate(evaluation, cols, threshold, j)
    assert abs(achieved - target) < 10.0   # matched calibration: within a generous tolerance for a modest replicate count
