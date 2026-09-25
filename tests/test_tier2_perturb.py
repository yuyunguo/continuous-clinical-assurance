"""Input-only Tier 2 perturbations: rescale, missingness, and model-version regression.

These act only on model inputs or on the model itself, never on a real case's true label, so injecting them onto
real (fixed, already-observed) cohorts never overrides a real outcome. Outcome-affecting scenarios (SC08, SC19) are
out of scope here (`Tier2-Readiness.md` §3, decided 2026-09-22).
"""

import numpy as np
import pytest

from src.tier2_module.config import Tier2StreamConfig
from src.tier2_module.perturb import (group_missingness_and_rescore, missingness_and_rescore, rescale_and_rescore,
                                      scope_creep_and_rescore, version_regression_and_rescore, window_progress)
from src.tier2_module.real_model import RealDeployedModel
from src.tier2_module.synthetic_stand_in import make_synthetic_cohort
from src.tier2_module.windowing import assign_windows

FEATURES = ("f0", "f1", "f2")


@pytest.fixture
def cfg():
    return Tier2StreamConfig(feature_columns=FEATURES, window_size=50, target_sensitivity=0.8)


@pytest.fixture
def fitted_windowed(cfg):
    df = make_synthetic_cohort(n=1_000, feature_columns=FEATURES, rng=np.random.default_rng(0), prevalence=0.25)
    model = RealDeployedModel(cfg).fit(df.iloc[:500], np.random.default_rng(1))
    windowed = assign_windows(model.apply(df), cfg)
    return windowed, model


def test_window_progress_is_zero_for_every_baseline_window():
    progress = window_progress(n_windows=10, n_baseline=4, shape="abrupt", s0=0, ramp_windows=1)
    assert (progress[:4] == 0.0).all()


def test_window_progress_matches_progress_schedule_after_the_baseline():
    progress = window_progress(n_windows=10, n_baseline=4, shape="abrupt", s0=2, ramp_windows=1)
    # monitored windows are absolute windows 4..9 (6 of them); abrupt at monitored index 2 means absolute window 6 onward is 1
    assert list(progress[4:]) == [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]


def test_window_progress_rejects_s0_beyond_the_monitored_window_count():
    with pytest.raises(ValueError, match="s0"):
        window_progress(n_windows=10, n_baseline=4, shape="abrupt", s0=6, ramp_windows=1)


def test_rescale_leaves_baseline_windows_unchanged(cfg, fitted_windowed):
    windowed, model = fitted_windowed
    n_windows = windowed["window"].nunique()
    original = windowed.iloc[:cfg.window_size]["f0"].to_numpy()   # window 0, a baseline window
    scored = rescale_and_rescore(windowed, cfg, model, column="f0", level_value=3.0, shape="abrupt", s0=0,
                                 n_baseline=n_windows - 1, ramp_windows=1)   # one monitored window, at the far end
    assert np.allclose(scored.iloc[:cfg.window_size]["f0"].to_numpy(), original)


def test_rescale_multiplies_the_column_by_the_full_level_after_onset(cfg, fitted_windowed):
    """`rescale_and_rescore` multiplies by the full level; `model.apply()` (called at the end of `rescale_and_rescore`)
    then winsorizes the result to the training cohort's own fit-period bounds -- expected here too, since a real
    deployed model's data-quality safeguards apply to every cohort it scores, perturbed or not. The rescaled value
    is compared against the model's own winsorized expectation, not the raw, unclipped multiplication."""
    windowed, model = fitted_windowed
    n_windows = windowed["window"].nunique()
    n_baseline = n_windows - 2   # leave exactly 2 monitored windows
    original_last_window = windowed[windowed["window"] == n_windows - 1]["f0"].to_numpy()
    scored = rescale_and_rescore(windowed, cfg, model, column="f0", level_value=3.0, shape="abrupt", s0=0,
                                 n_baseline=n_baseline, ramp_windows=1)
    rescaled_last_window = scored[scored["window"] == n_windows - 1]["f0"].to_numpy()
    lo, hi = model.winsorize_bounds["f0"]
    expected = np.clip(original_last_window * 3.0, lo, hi)
    assert np.allclose(rescaled_last_window, expected)


def test_rescale_rescores_with_the_model_not_just_the_column(cfg, fitted_windowed):
    windowed, model = fitted_windowed
    n_windows = windowed["window"].nunique()
    scored = rescale_and_rescore(windowed, cfg, model, column="f0", level_value=5.0, shape="abrupt", s0=0,
                                 n_baseline=n_windows - 1, ramp_windows=1)
    last_window = scored[scored["window"] == n_windows - 1]
    scaled = model._scaler.transform(last_window[list(FEATURES)].to_numpy())
    recomputed = model._clf.predict_proba(scaled)[:, 1]
    assert np.allclose(last_window["p_hat"].to_numpy(), recomputed)


def test_missingness_masks_roughly_the_target_share_of_cases_in_a_full_effect_window(cfg, fitted_windowed):
    windowed, model = fitted_windowed
    n_windows = windowed["window"].nunique()
    scored = missingness_and_rescore(windowed, cfg, model, column="f0", level_value=0.6, shape="abrupt", s0=0,
                                     n_baseline=n_windows - 1, ramp_windows=1, rng=np.random.default_rng(2))
    last_window = scored[scored["window"] == n_windows - 1]
    masked_share = np.isclose(last_window["f0"].to_numpy(), model.impute_means["f0"]).mean()
    assert abs(masked_share - 0.6) < 0.15


def test_missingness_never_masks_a_baseline_window(cfg, fitted_windowed):
    windowed, model = fitted_windowed
    n_windows = windowed["window"].nunique()
    original = windowed.iloc[:cfg.window_size]["f0"].to_numpy()   # window 0, a baseline window
    scored = missingness_and_rescore(windowed, cfg, model, column="f0", level_value=1.0, shape="abrupt", s0=0,
                                     n_baseline=n_windows - 1, ramp_windows=1, rng=np.random.default_rng(3))
    assert np.allclose(scored.iloc[:cfg.window_size]["f0"].to_numpy(), original)


def test_group_missingness_masks_the_same_share_of_cases_as_single_column_missingness(cfg, fitted_windowed):
    windowed, model = fitted_windowed
    n_windows = windowed["window"].nunique()
    scored = group_missingness_and_rescore(windowed, cfg, model, columns=("f0", "f1"), level_value=0.6, shape="abrupt",
                                           s0=0, n_baseline=n_windows - 1, ramp_windows=1, rng=np.random.default_rng(2))
    last_window = scored[scored["window"] == n_windows - 1]
    masked_share = np.isclose(last_window["f0"].to_numpy(), model.impute_means["f0"]).mean()
    assert abs(masked_share - 0.6) < 0.15


def test_group_missingness_masks_every_column_in_the_group_for_the_same_affected_case(cfg, fitted_windowed):
    """The whole group must go missing together per case (one shared mask), not each column independently --
    otherwise this would offer no advantage over calling `missingness_and_rescore` once per column. Uses a partial
    probability (0.5): at prob=1.0 every case is masked regardless of whether the mask is shared or drawn
    independently per column, so that level trivially passes even a broken (per-column) implementation."""
    windowed, model = fitted_windowed
    n_windows = windowed["window"].nunique()
    scored = group_missingness_and_rescore(windowed, cfg, model, columns=("f0", "f1"), level_value=0.5, shape="abrupt",
                                           s0=0, n_baseline=n_windows - 1, ramp_windows=1, rng=np.random.default_rng(2))
    last_window = scored[scored["window"] == n_windows - 1]
    f0_masked = np.isclose(last_window["f0"].to_numpy(), model.impute_means["f0"])
    f1_masked = np.isclose(last_window["f1"].to_numpy(), model.impute_means["f1"])
    assert 0 < f0_masked.mean() < 1   # the probability is genuinely partial, not degenerate at 0 or 1
    assert np.array_equal(f0_masked, f1_masked)   # same cases masked in both columns


def test_group_missingness_never_masks_a_baseline_window(cfg, fitted_windowed):
    windowed, model = fitted_windowed
    n_windows = windowed["window"].nunique()
    original = windowed.iloc[:cfg.window_size][["f0", "f1"]].to_numpy()   # window 0, a baseline window
    scored = group_missingness_and_rescore(windowed, cfg, model, columns=("f0", "f1"), level_value=1.0, shape="abrupt",
                                           s0=0, n_baseline=n_windows - 1, ramp_windows=1, rng=np.random.default_rng(3))
    assert np.allclose(scored.iloc[:cfg.window_size][["f0", "f1"]].to_numpy(), original)


def test_group_missingness_rescores_with_the_model_not_just_the_columns(cfg, fitted_windowed):
    windowed, model = fitted_windowed
    n_windows = windowed["window"].nunique()
    scored = group_missingness_and_rescore(windowed, cfg, model, columns=("f0", "f1"), level_value=1.0, shape="abrupt",
                                           s0=0, n_baseline=n_windows - 1, ramp_windows=1, rng=np.random.default_rng(4))
    last_window = scored[scored["window"] == n_windows - 1]
    scaled = model._scaler.transform(last_window[list(FEATURES)].to_numpy())
    recomputed = model._clf.predict_proba(scaled)[:, 1]
    assert np.allclose(last_window["p_hat"].to_numpy(), recomputed)


@pytest.fixture
def out_of_scope_pool():
    """A distinct synthetic population (a different prevalence) standing in for real, genuinely out-of-scope cases
    (e.g. eICU) that a model fit only on `fitted_windowed`'s population has never seen."""
    return make_synthetic_cohort(n=300, feature_columns=FEATURES, rng=np.random.default_rng(50), prevalence=0.6)


def test_scope_creep_replaces_roughly_the_target_share_of_cases_in_a_full_effect_window(cfg, fitted_windowed, out_of_scope_pool):
    windowed, model = fitted_windowed
    n_windows = windowed["window"].nunique()
    scored = scope_creep_and_rescore(windowed, cfg, model, out_of_scope_pool, level_value=0.6, shape="abrupt", s0=0,
                                     n_baseline=n_windows - 1, ramp_windows=1, rng=np.random.default_rng(51))
    last_window = scored[scored["window"] == n_windows - 1]
    assert abs(last_window["out_of_scope"].mean() - 0.6) < 0.15


def test_scope_creep_never_touches_a_baseline_window(cfg, fitted_windowed, out_of_scope_pool):
    windowed, model = fitted_windowed
    n_windows = windowed["window"].nunique()
    original = windowed.iloc[:cfg.window_size].copy()
    scored = scope_creep_and_rescore(windowed, cfg, model, out_of_scope_pool, level_value=1.0, shape="abrupt", s0=0,
                                     n_baseline=n_windows - 1, ramp_windows=1, rng=np.random.default_rng(52))
    baseline = scored.iloc[:cfg.window_size]
    assert not baseline["out_of_scope"].any()
    assert np.allclose(baseline[list(FEATURES)].to_numpy(), original[list(FEATURES)].to_numpy())


def test_scope_creep_replaced_cases_carry_the_pools_own_label_and_features(cfg, fitted_windowed, out_of_scope_pool):
    """A replaced case's label, subgroup, and features must be the drawn pool row's OWN real values -- not the
    original MIMIC case's values with only a flag flipped, and not any fabricated/synthetic override."""
    windowed, model = fitted_windowed
    n_windows = windowed["window"].nunique()
    scored = scope_creep_and_rescore(windowed, cfg, model, out_of_scope_pool, level_value=1.0, shape="abrupt", s0=0,
                                     n_baseline=n_windows - 1, ramp_windows=1, rng=np.random.default_rng(53))
    last_window = scored[scored["window"] == n_windows - 1]
    replaced = last_window[last_window["out_of_scope"]]
    assert len(replaced) == len(last_window)   # level_value=1.0: the whole window is replaced
    pool_f0 = set(np.round(out_of_scope_pool["f0"].to_numpy(), 6))
    assert set(np.round(replaced["f0"].to_numpy(), 6)).issubset(pool_f0)


def test_scope_creep_replaced_cases_are_scored_with_the_model_not_copied_unscored(cfg, fitted_windowed, out_of_scope_pool):
    windowed, model = fitted_windowed
    n_windows = windowed["window"].nunique()
    scored = scope_creep_and_rescore(windowed, cfg, model, out_of_scope_pool, level_value=1.0, shape="abrupt", s0=0,
                                     n_baseline=n_windows - 1, ramp_windows=1, rng=np.random.default_rng(54))
    last_window = scored[scored["window"] == n_windows - 1]
    scaled = model._scaler.transform(last_window[list(FEATURES)].to_numpy())
    recomputed = model._clf.predict_proba(scaled)[:, 1]
    assert np.allclose(last_window["p_hat"].to_numpy(), recomputed)


def test_scope_creep_at_zero_level_leaves_every_case_in_scope(cfg, fitted_windowed, out_of_scope_pool):
    windowed, model = fitted_windowed
    n_windows = windowed["window"].nunique()
    scored = scope_creep_and_rescore(windowed, cfg, model, out_of_scope_pool, level_value=0.0, shape="abrupt", s0=0,
                                     n_baseline=n_windows - 1, ramp_windows=1, rng=np.random.default_rng(55))
    assert not scored["out_of_scope"].any()
    assert np.allclose(scored[list(FEATURES)].to_numpy(), windowed[list(FEATURES)].to_numpy())


def test_version_regression_changes_scores_only_from_onset_onward(cfg, fitted_windowed):
    windowed, model = fitted_windowed
    n_windows = windowed["window"].nunique()
    n_baseline = n_windows - 1
    scored = version_regression_and_rescore(windowed, cfg, model, level_value=1.0, shape="abrupt", s0=0,
                                            n_baseline=n_baseline, ramp_windows=1, seed=5)
    baseline_p = scored[scored["window"] < n_baseline]["p_hat"].to_numpy()
    original_baseline_p = windowed[windowed["window"] < n_baseline].pipe(model.apply)["p_hat"].to_numpy() \
        if "p_hat" not in windowed.columns else windowed[windowed["window"] < n_baseline]["p_hat"].to_numpy()
    assert np.allclose(baseline_p, original_baseline_p)
    last_window_p = scored[scored["window"] == n_windows - 1]["p_hat"].to_numpy()
    original_last_p = windowed[windowed["window"] == n_windows - 1]["p_hat"].to_numpy()
    assert not np.allclose(last_window_p, original_last_p)   # a full-strength (level 1.0) regression must change scores


def test_version_regression_at_zero_level_is_the_identity(cfg, fitted_windowed):
    windowed, model = fitted_windowed
    scored = version_regression_and_rescore(windowed, cfg, model, level_value=0.0, shape="abrupt", s0=0,
                                            n_baseline=0, ramp_windows=1, seed=5)
    assert np.allclose(scored["p_hat"].to_numpy(), windowed["p_hat"].to_numpy())
