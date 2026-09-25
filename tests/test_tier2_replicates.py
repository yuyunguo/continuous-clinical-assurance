"""Multi-replicate TAFD comparison on Tier 2: many resampled-and-perturbed replicate streams, each classified by
`src.eval_module.tafd.classify_replicate` (reused unchanged — it operates on plain arrays, with no dependence on
the Tier 1 synthetic generator), then compared across arms with `rmst_difference`/`bootstrap_difference`.
"""

import functools

import numpy as np
import pytest

from scripts.m4_pilot import ARM_ORDER
from src.eval_module.tafd import bootstrap_difference, rmst_difference
from src.sim_module.config import load_config
from src.tier2_module.arms import tier2_calibrate_arms
from src.tier2_module.calibrate import tier2_null_scores_multi
from src.tier2_module.config import Tier2StreamConfig
from src.tier2_module.onset_calibration import tier2_calibrate_onset, tier2_onset_null_series_multi
from src.tier2_module.perturb import rescale_and_rescore
from src.tier2_module.real_model import RealDeployedModel
from src.tier2_module.replicates import tier2_replicate_tafd
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
def fitted_windowed(cfg):
    df = make_synthetic_cohort(n=8_000, feature_columns=FEATURES, rng=np.random.default_rng(0), prevalence=0.25)
    model = RealDeployedModel(cfg).fit(df.iloc[:4000], np.random.default_rng(1))
    windowed = assign_windows(model.apply(df), cfg)
    return windowed, model


@pytest.fixture
def arm_thresholds(cfg, phase2_cfg, fitted_windowed):
    windowed, _model = fitted_windowed
    calib = tier2_null_scores_multi(windowed, cfg, phase2_cfg, phase2_cfg.harm, streams=150, seed=11, lags=(0,),
                                    n_windows_out=40, block_size=8, n_baseline=15)[0]
    evaluation = tier2_null_scores_multi(windowed, cfg, phase2_cfg, phase2_cfg.harm, streams=150, seed=13, lags=(0,),
                                         n_windows_out=40, block_size=8, n_baseline=15)[0]
    return tier2_calibrate_arms(calib, evaluation, budget=30.0, j=phase2_cfg.routing.hysteresis_j)


@pytest.fixture
def onset_threshold(cfg, phase2_cfg, fitted_windowed):
    """A calibrated onset threshold (see `onset_calibration.py`): a low target here (not the 30/1,000 monitoring
    budget) so the naive-rule confound this replaced (near-100% false-onset rate) does not resurface in tests."""
    windowed, _model = fitted_windowed
    calib = tier2_onset_null_series_multi(windowed, cfg, phase2_cfg, phase2_cfg.harm, streams=150, seed=41,
                                          n_windows_out=40, block_size=8, n_baseline=15)
    return tier2_calibrate_onset(calib, target_per_1000=5.0, j=phase2_cfg.routing.hysteresis_j)


def _rescale_perturb_fn(cfg, model):
    """A perturbation function of exactly (resampled_windowed, s0), bound to a fixed rescale spec."""
    return functools.partial(rescale_and_rescore, cfg=cfg, model=model, column="f0", level_value=4.0, shape="abrupt",
                             n_baseline=10, ramp_windows=1)


def test_tier2_replicate_tafd_returns_an_array_per_arm(cfg, phase2_cfg, fitted_windowed, arm_thresholds, onset_threshold):
    windowed, model = fitted_windowed
    perturb_fn = _rescale_perturb_fn(cfg, model)
    result = tier2_replicate_tafd(windowed, cfg, phase2_cfg, phase2_cfg.harm, arm_thresholds, lag=0,
                                  perturb_fn=perturb_fn, n_replicates=15, n_windows_out=40, n_baseline=10,
                                  block_size=8, horizon=15, onset_threshold=onset_threshold, rng=np.random.default_rng(20))
    assert set(result) == set(ARM_ORDER)
    for name in ARM_ORDER:
        assert result[name].dtype == float
        assert len(result[name]) <= 15   # replicates with no detectable onset (rare here) are dropped


def test_tier2_replicate_tafd_every_value_is_within_the_horizon(cfg, phase2_cfg, fitted_windowed, arm_thresholds, onset_threshold):
    windowed, model = fitted_windowed
    perturb_fn = _rescale_perturb_fn(cfg, model)
    horizon = 15
    result = tier2_replicate_tafd(windowed, cfg, phase2_cfg, phase2_cfg.harm, arm_thresholds, lag=0,
                                  perturb_fn=perturb_fn, n_replicates=15, n_windows_out=40, n_baseline=10,
                                  block_size=8, horizon=horizon, onset_threshold=onset_threshold, rng=np.random.default_rng(21))
    for name in ARM_ORDER:
        assert (result[name] >= 0).all()
        assert (result[name] <= horizon).all()


def test_tier2_replicate_tafd_label_free_arm_detects_input_only_perturbation_no_slower_than_conventional(
    cfg, phase2_cfg, fitted_windowed, arm_thresholds, onset_threshold
):
    """The qualitative finding the demo already showed for one replicate, now checked in aggregate: a label-free
    arm should not be slower, on average, than the purely performance-based conventional arm at detecting a
    rescale (an input-only perturbation with no label signal until decisions shift). The calibrated onset rule
    (unlike the naive rule it replaced) is appropriately strict, so few of these small synthetic replicates give a
    confidently-detected onset -- more replicates are drawn here than elsewhere in this file to get a stable
    enough surviving sample, not to cherry-pick a favorable one."""
    windowed, model = fitted_windowed
    perturb_fn = _rescale_perturb_fn(cfg, model)
    result = tier2_replicate_tafd(windowed, cfg, phase2_cfg, phase2_cfg.harm, arm_thresholds, lag=0,
                                  perturb_fn=perturb_fn, n_replicates=200, n_windows_out=40, n_baseline=10,
                                  block_size=8, horizon=15, onset_threshold=onset_threshold, rng=np.random.default_rng(22))
    assert len(result["conventional"]) > 5 and len(result["conv + label-free"]) > 5   # enough replicates survived to compare
    reduction = rmst_difference(result["conventional"], result["conv + label-free"])
    assert reduction >= 0.0


def test_tier2_replicate_tafd_bootstrap_difference_is_reusable_unchanged(cfg, phase2_cfg, fitted_windowed, arm_thresholds, onset_threshold):
    """`bootstrap_difference` from src.eval_module.tafd needs no Tier 2-specific version at all."""
    windowed, model = fitted_windowed
    perturb_fn = _rescale_perturb_fn(cfg, model)
    result = tier2_replicate_tafd(windowed, cfg, phase2_cfg, phase2_cfg.harm, arm_thresholds, lag=0,
                                  perturb_fn=perturb_fn, n_replicates=30, n_windows_out=40, n_baseline=10,
                                  block_size=8, horizon=15, onset_threshold=onset_threshold, rng=np.random.default_rng(23))
    lo, hi = bootstrap_difference(result["conventional"], result["conv + label-free"], np.random.default_rng(1))
    assert lo <= hi


def test_tier2_replicate_tafd_drops_a_replicate_whose_onset_leaves_no_room_for_the_horizon(cfg, phase2_cfg, fitted_windowed, arm_thresholds, onset_threshold, monkeypatch):
    """When the real onset window is found but is too close to the resampled stream's end for the full horizon to
    fit, the replicate must be dropped, not truncated or padded."""
    import src.tier2_module.replicates as replicates_module
    windowed, model = fitted_windowed
    perturb_fn = _rescale_perturb_fn(cfg, model)
    n_windows_out, n_baseline, horizon = 40, 10, 15
    n_monitored = n_windows_out - n_baseline
    monkeypatch.setattr(replicates_module, "tier2_onset_index_calibrated", lambda *a, **k: n_monitored - 1)   # always right at the end
    result = tier2_replicate_tafd(windowed, cfg, phase2_cfg, phase2_cfg.harm, arm_thresholds, lag=0,
                                  perturb_fn=perturb_fn, n_replicates=5, n_windows_out=n_windows_out,
                                  n_baseline=n_baseline, block_size=8, horizon=horizon, onset_threshold=onset_threshold,
                                  rng=np.random.default_rng(24))
    for name in arm_thresholds:
        assert len(result[name]) == 0


def test_tier2_replicate_tafd_draws_a_different_onset_window_each_replicate(cfg, phase2_cfg, fitted_windowed, arm_thresholds, onset_threshold):
    """s0 must actually be drawn per replicate, not fixed -- recorded via a spy wrapper around perturb_fn."""
    windowed, model = fitted_windowed
    base_perturb_fn = _rescale_perturb_fn(cfg, model)
    seen_s0 = []

    def spy_perturb_fn(resampled, s0):
        seen_s0.append(s0)
        return base_perturb_fn(resampled, s0=s0)

    tier2_replicate_tafd(windowed, cfg, phase2_cfg, phase2_cfg.harm, arm_thresholds, lag=0, perturb_fn=spy_perturb_fn,
                         n_replicates=20, n_windows_out=40, n_baseline=10, block_size=8, horizon=15, onset_threshold=onset_threshold,
                         rng=np.random.default_rng(25))
    assert len(set(seen_s0)) > 1


def test_tier2_replicate_tafd_appends_classify_replicates_own_tafd_value(cfg, phase2_cfg, fitted_windowed, arm_thresholds, onset_threshold, monkeypatch):
    """The appended value must be `classify_replicate`'s own `tafd`, not a hardcoded placeholder -- checked with a
    stub that returns a distinctive sentinel value."""
    import src.tier2_module.replicates as replicates_module
    windowed, model = fitted_windowed
    perturb_fn = _rescale_perturb_fn(cfg, model)
    monkeypatch.setattr(replicates_module, "classify_replicate",
                        lambda *a, **k: {"tafd": 7, "censored": False, "false_alarms": 0, "anticipatory": False, "carry_in": False, "lead_time": 0})
    result = tier2_replicate_tafd(windowed, cfg, phase2_cfg, phase2_cfg.harm, arm_thresholds, lag=0, perturb_fn=perturb_fn,
                                  n_replicates=5, n_windows_out=40, n_baseline=10, block_size=8, horizon=15,
                                  onset_threshold=onset_threshold, rng=np.random.default_rng(26))
    for name in arm_thresholds:
        if len(result[name]):
            assert (result[name] == 7).all()


def test_tier2_replicate_tafd_is_reproducible_given_the_same_seed(cfg, phase2_cfg, fitted_windowed, arm_thresholds, onset_threshold):
    windowed, model = fitted_windowed
    perturb_fn = _rescale_perturb_fn(cfg, model)
    a = tier2_replicate_tafd(windowed, cfg, phase2_cfg, phase2_cfg.harm, arm_thresholds, lag=0, perturb_fn=perturb_fn,
                             n_replicates=10, n_windows_out=40, n_baseline=10, block_size=8, horizon=15,
                             onset_threshold=onset_threshold, rng=np.random.default_rng(30))
    b = tier2_replicate_tafd(windowed, cfg, phase2_cfg, phase2_cfg.harm, arm_thresholds, lag=0, perturb_fn=perturb_fn,
                             n_replicates=10, n_windows_out=40, n_baseline=10, block_size=8, horizon=15,
                             onset_threshold=onset_threshold, rng=np.random.default_rng(30))
    for name in ARM_ORDER:
        assert list(a[name]) == list(b[name])
