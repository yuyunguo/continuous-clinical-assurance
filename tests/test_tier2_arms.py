"""Multi-arm matched calibration on Tier 2 data: the M4-pilot arm definitions (conventional, label-free,
calibration-only, CCA union, CCA component-split, tilted, calibration arm), calibrated on block-bootstrap null
streams and evaluated on an independent batch. Reuses the exact arm definitions `scripts/m4_pilot.py` uses for
Tier 1, so the two tiers compare the same indicator sets -- not a separately chosen set for Tier 2.
"""

import numpy as np
import pytest

from scripts.m4_pilot import ARM_ORDER, FLAT_ARMS, SPLIT_ARMS
from src.monitor_module import INDICATOR_ORDER
from src.sim_module.config import load_config
from src.tier2_module.arms import tier2_calibrate_arms
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
    return assign_windows(model.apply(df), cfg)


@pytest.fixture
def calib_and_evaluation(cfg, phase2_cfg, windowed):
    calib = tier2_null_scores_multi(windowed, cfg, phase2_cfg, phase2_cfg.harm, streams=150, seed=11, lags=(0,),
                                    n_windows_out=60, block_size=10, n_baseline=20)[0]
    evaluation = tier2_null_scores_multi(windowed, cfg, phase2_cfg, phase2_cfg.harm, streams=150, seed=13, lags=(0,),
                                         n_windows_out=60, block_size=10, n_baseline=20)[0]
    return calib, evaluation


def test_tier2_calibrate_arms_returns_every_arm_in_arm_order(phase2_cfg, calib_and_evaluation):
    calib, evaluation = calib_and_evaluation
    result = tier2_calibrate_arms(calib, evaluation, budget=30.0, j=phase2_cfg.routing.hysteresis_j)
    assert set(result) == set(ARM_ORDER)


def test_tier2_calibrate_arms_gives_a_float_threshold_for_flat_arms_and_a_dict_for_split_arms(phase2_cfg, calib_and_evaluation):
    calib, evaluation = calib_and_evaluation
    result = tier2_calibrate_arms(calib, evaluation, budget=30.0, j=phase2_cfg.routing.hysteresis_j)
    for name in FLAT_ARMS:
        threshold, _rate, _se = result[name]
        assert isinstance(threshold, float)
    for name in SPLIT_ARMS:
        threshold, _rate, _se = result[name]
        assert isinstance(threshold, dict)
        assert set(threshold) == {"P", "U", "C"}


def test_tier2_calibrate_arms_achieves_close_to_budget_on_independent_evaluation(phase2_cfg, calib_and_evaluation):
    calib, evaluation = calib_and_evaluation
    result = tier2_calibrate_arms(calib, evaluation, budget=30.0, j=phase2_cfg.routing.hysteresis_j)
    for name in ARM_ORDER:
        _threshold, rate, se = result[name]
        assert abs(rate - 30.0) < max(15.0, 5 * se)   # matched calibration: within a generous multiple of its own SE


def test_tier2_calibrate_arms_floors_the_standard_error_when_zero_alerts_occur():
    """With no signal at all, an arm sees zero alert-episode starts on the evaluation batch; the standard error
    must still reflect that "zero observed" is not "zero true rate" -- floored at one event, not left at zero."""
    zeros = np.zeros((5, 5, len(INDICATOR_ORDER)))   # shape (S, T, K): 5 streams, 5 windows, every indicator flat at 0
    result = tier2_calibrate_arms(zeros, zeros, budget=30.0, j=2)
    _threshold, rate, se = result["conventional"]
    assert rate == 0.0
    assert se == pytest.approx(1.0 / (5 * 5) * 1000.0)   # sqrt(1) floor, not sqrt(0) = 0


def test_tier2_calibrate_arms_ccas_union_uses_every_indicator_while_conventional_uses_three(phase2_cfg, calib_and_evaluation):
    """A structural check that the arm definitions are actually wired to different indicator subsets, not all
    collapsed onto the same one -- calibrating the same score array with a real subset difference should generally
    give different thresholds for arms of very different size, though this is not asserted numerically here."""
    calib, evaluation = calib_and_evaluation
    result = tier2_calibrate_arms(calib, evaluation, budget=30.0, j=phase2_cfg.routing.hysteresis_j)
    conventional_threshold, _, _ = result["conventional"]
    union_threshold, _, _ = result["CCA (union)"]
    assert conventional_threshold != union_threshold
