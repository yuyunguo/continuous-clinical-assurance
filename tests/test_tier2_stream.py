"""Tier 2 real-data stream builder: synthetic stand-in, model, windowing, and StreamData assembly.

Every test here runs against the synthetic stand-in, never a real database (see `Tier2-Readiness.md`: row-level
MIMIC-IV and eICU data stays on the study author's machine)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.sim_module.config import HarmConfig
from src.tier2_module.config import Tier2StreamConfig
from src.tier2_module.real_model import RealDeployedModel
from src.tier2_module.stream import build_stream_data
from src.tier2_module.synthetic_stand_in import make_synthetic_cohort
from src.tier2_module.windowing import assign_windows

FEATURES = ("f0", "f1", "f2")


@pytest.fixture
def harm():
    return HarmConfig(severity_cutoff=2, weights=(1.0, 7.1, 100.0), severity_probs_positive=(0.5, 0.35, 0.15),
                      severity_probs_negative=(0.85, 0.12, 0.03), kappa=0.25, fn_factor=2.4)


@pytest.fixture
def cfg():
    return Tier2StreamConfig(feature_columns=FEATURES, window_size=50, subgroup_column="subgroup",
                             order_column="order_key", label_column="label", target_sensitivity=0.8)


def test_synthetic_cohort_has_the_expected_schema_and_size():
    df = make_synthetic_cohort(n=500, feature_columns=FEATURES, rng=np.random.default_rng(0), prevalence=0.2)
    assert len(df) == 500
    assert set(FEATURES) | {"order_key", "label", "subgroup"} <= set(df.columns)
    assert df["label"].isin([0, 1]).all()
    assert df["subgroup"].isin([0, 1]).all()
    assert (df["order_key"].values == np.sort(df["order_key"].values)).all()


def test_synthetic_cohort_prevalence_is_close_to_target():
    df = make_synthetic_cohort(n=20_000, feature_columns=FEATURES, rng=np.random.default_rng(1), prevalence=0.3)
    assert abs(df["label"].mean() - 0.3) < 0.02


def test_real_model_fit_apply_adds_scored_columns(cfg):
    df = make_synthetic_cohort(n=4_000, feature_columns=FEATURES, rng=np.random.default_rng(2), prevalence=0.25)
    train, rest = df.iloc[:2000], df.iloc[2000:]
    fitted = RealDeployedModel(cfg).fit(train, np.random.default_rng(3))
    scored = fitted.apply(rest)
    assert "p_hat" in scored.columns and "decision" in scored.columns
    assert scored["p_hat"].between(0, 1).all()
    assert scored["decision"].isin([0, 1]).all()


def test_real_model_threshold_hits_target_sensitivity_on_its_own_training_labels(cfg):
    df = make_synthetic_cohort(n=6_000, feature_columns=FEATURES, rng=np.random.default_rng(4), prevalence=0.3)
    fitted = RealDeployedModel(cfg).fit(df, np.random.default_rng(5))
    scored = fitted.apply(df)
    positives = scored[scored["label"] == 1]
    sensitivity = (positives["decision"] == 1).mean()
    assert abs(sensitivity - cfg.target_sensitivity) < 0.03


def test_real_model_fit_handles_missing_values_in_training_features(cfg):
    """Real data has genuine missingness (unlike the synthetic stand-in): sklearn's LogisticRegression rejects NaN
    outright, so fit must impute before fitting, not just compute impute_means for later use."""
    df = make_synthetic_cohort(n=2_000, feature_columns=FEATURES, rng=np.random.default_rng(28), prevalence=0.3)
    rng = np.random.default_rng(29)
    mask = rng.random((len(df), len(FEATURES))) < 0.2
    df.loc[:, FEATURES] = df[list(FEATURES)].mask(pd.DataFrame(mask, columns=list(FEATURES), index=df.index))
    fitted = RealDeployedModel(cfg).fit(df, rng)   # must not raise
    scored = fitted.apply(df)
    assert scored["p_hat"].notna().all()


def test_real_model_apply_writes_the_imputed_features_back_not_just_p_hat(cfg):
    """`build_stream_data`'s `x_model` reads feature columns straight from the scored DataFrame, so a real,
    genuinely missing value must be imputed in the returned DataFrame itself -- not just internally, used once for
    scoring and then dropped. Otherwise a downstream StreamData still carries real NaN (this was found by running
    the real MIMIC-IV extraction: raw NaN in x_model broke the C2 drift indicator's np.nanmean with no test catching it)."""
    df = make_synthetic_cohort(n=1_000, feature_columns=FEATURES, rng=np.random.default_rng(34), prevalence=0.3)
    fitted = RealDeployedModel(cfg).fit(df, np.random.default_rng(35))
    with_missing = df.copy()
    with_missing.loc[with_missing.index[:10], "f0"] = float("nan")
    scored = fitted.apply(with_missing)
    assert scored["f0"].notna().all()
    assert (scored["f0"].iloc[:10] == fitted.impute_means["f0"]).all()


def test_real_model_apply_handles_missing_values_using_the_training_impute_means(cfg):
    df = make_synthetic_cohort(n=1_000, feature_columns=FEATURES, rng=np.random.default_rng(30), prevalence=0.3)
    fitted = RealDeployedModel(cfg).fit(df, np.random.default_rng(31))
    with_missing = df.copy()
    with_missing.loc[with_missing.index[:10], "f0"] = float("nan")
    scored = fitted.apply(with_missing)
    assert scored["p_hat"].iloc[:10].notna().all()
    # imputing the missing values manually and scoring must give the identical result apply() already produced
    # (winsorize first, matching apply()'s own order, since fit/apply clip extreme values before imputing NaN)
    manually_imputed = with_missing.copy()
    manually_imputed.loc[manually_imputed.index[:10], "f0"] = fitted.impute_means["f0"]
    winsorized = fitted._winsorize(manually_imputed[list(FEATURES)].to_numpy())
    scaled = fitted._scaler.transform(winsorized)
    expected = fitted._clf.predict_proba(scaled)[:, 1]
    assert np.allclose(scored["p_hat"].to_numpy(), expected)


def test_real_model_fit_uses_cross_validated_log_loss_regularization(cfg):
    """Motivated by the readmission anomaly (Tier2-Readiness.md SS8): an unregularized fit on real, correlated
    features produced a probability as extreme as 1e-135, which broke a downstream indicator's Newton's-method
    fit. `fit` must wire up cross-validated L2 selection by log loss (a model-fit criterion, unrelated to any
    monitoring arm's detection speed -- not accuracy or another scorer, and not a single, hand-picked C)."""
    from src.tier2_module.real_model import REGULARIZATION_GRID
    df = make_synthetic_cohort(n=1_000, feature_columns=FEATURES, rng=np.random.default_rng(41), prevalence=0.3)
    fitted = RealDeployedModel(cfg).fit(df, np.random.default_rng(41))
    assert fitted._clf.scoring == "neg_log_loss"
    assert tuple(fitted._clf.Cs_) == REGULARIZATION_GRID
    assert fitted._clf.C_[0] in REGULARIZATION_GRID   # the selected strength actually came from the grid


def test_real_model_standardizes_features_before_fitting(cfg):
    """Without standardization, L2 regularization penalizes a large-magnitude real feature (e.g. heart rate,
    ~60-150) far less per unit than a small-magnitude one (e.g. a normalized lab ratio), making the chosen
    regularization strength depend on arbitrary real-world units rather than genuine predictive structure."""
    df = make_synthetic_cohort(n=1_000, feature_columns=FEATURES, rng=np.random.default_rng(42), prevalence=0.3)
    fitted = RealDeployedModel(cfg).fit(df, np.random.default_rng(43))
    # compared against the WINSORIZED mean, not the raw one: fit() clips each feature's extreme tail (its own
    # training-fit percentiles) before ever computing a mean or fitting the scaler -- see winsorize_bounds' docstring.
    winsorized = fitted._winsorize(df[list(FEATURES)].to_numpy())
    assert np.allclose(fitted._scaler.mean_, winsorized.mean(axis=0), atol=1e-6)
    assert (fitted._scaler.scale_ > 0).all()


def test_real_model_winsorizes_extreme_values_to_the_training_fit_percentiles(cfg):
    """A rare, implausible extreme value (e.g. a MIMIC-IV chart-error artifact, Tier2-Readiness.md SS9) must be
    clipped to the training cohort's own fit-period percentile bounds before it ever reaches the model or (via the
    returned feature columns) the C2 monitoring indicator, which is especially sensitive to it (it sums squared
    per-feature deviations, so squaring amplifies an unclipped outlier enormously)."""
    df = make_synthetic_cohort(n=2_000, feature_columns=FEATURES, rng=np.random.default_rng(70), prevalence=0.3)
    fitted = RealDeployedModel(cfg).fit(df, np.random.default_rng(71))
    lo, hi = fitted.winsorize_bounds["f0"]
    extreme = df.copy()
    extreme.loc[extreme.index[0], "f0"] = hi + 1_000_000.0   # far beyond any plausible bound
    scored = fitted.apply(extreme)
    assert scored["f0"].iloc[0] == pytest.approx(hi)
    assert lo < hi   # the bounds must be genuinely distinct, not a degenerate (-inf, inf) default that never clips


def test_real_model_winsorize_bounds_fall_back_to_a_no_op_for_an_all_missing_training_column(cfg):
    """An all-missing training column has no real percentile to bound with (`np.nanpercentile` gives NaN). Falling
    back to (nan, nan) would be a real bug: `np.clip` with NaN bounds turns EVERY value NaN, so a later cohort
    where that column has genuine, valid values would have all of them destroyed by `_winsorize`, not just the
    all-missing training case. The bounds must fall back to a genuine no-op, (-inf, inf), instead."""
    df = make_synthetic_cohort(n=500, feature_columns=FEATURES, rng=np.random.default_rng(74), prevalence=0.3)
    df["f0"] = float("nan")
    fitted = RealDeployedModel(cfg).fit(df, np.random.default_rng(75))
    assert fitted.winsorize_bounds["f0"] == (-np.inf, np.inf)
    evaluation = make_synthetic_cohort(n=50, feature_columns=FEATURES, rng=np.random.default_rng(76), prevalence=0.3)
    scored = fitted.apply(evaluation)
    assert np.allclose(scored["f0"].to_numpy(), evaluation["f0"].to_numpy())   # genuine values must survive, not become NaN/0


def test_real_model_winsorize_bounds_come_from_the_training_cohort_only(cfg):
    """`winsorize_bounds` must be fit on `train`, not on whatever cohort `apply` is later called with -- an
    extreme value in an evaluation cohort must not widen the bounds used to judge it (no leakage, same discipline
    already applied to `impute_means`/`StandardScaler`)."""
    df = make_synthetic_cohort(n=1_000, feature_columns=FEATURES, rng=np.random.default_rng(72), prevalence=0.3)
    fitted = RealDeployedModel(cfg).fit(df, np.random.default_rng(73))
    lo, hi = fitted.winsorize_bounds["f0"]
    evaluation = df.copy()
    evaluation.loc[evaluation.index[0], "f0"] = hi * 1000.0   # a much larger extreme value, only in the evaluation cohort
    fitted.apply(evaluation)   # must not mutate winsorize_bounds as a side effect
    assert fitted.winsorize_bounds["f0"] == (lo, hi)


def test_real_model_falls_back_to_zero_when_a_training_column_is_entirely_missing(cfg):
    """An all-missing training column has no real mean to impute with (pandas gives NaN); imputing with NaN would
    put NaN right back into the model, so it must fall back to 0.0 instead."""
    df = make_synthetic_cohort(n=500, feature_columns=FEATURES, rng=np.random.default_rng(32), prevalence=0.3)
    df["f0"] = float("nan")
    fitted = RealDeployedModel(cfg).fit(df, np.random.default_rng(33))
    assert fitted.impute_means["f0"] == 0.0
    scored = fitted.apply(df)
    assert scored["p_hat"].notna().all()


def test_real_model_impute_means_match_the_actual_training_column_means(cfg):
    """`impute_means` must match the training column's mean AFTER winsorization, not the raw mean -- computed post-
    clip specifically so a handful of extreme outlier rows can't skew the value later used to fill in missing data."""
    df = make_synthetic_cohort(n=500, feature_columns=FEATURES, rng=np.random.default_rng(24), prevalence=0.3)
    fitted = RealDeployedModel(cfg).fit(df, np.random.default_rng(25))
    winsorized = fitted._winsorize(df[list(FEATURES)].to_numpy())
    for i, col in enumerate(FEATURES):
        assert abs(fitted.impute_means[col] - winsorized[:, i].mean()) < 1e-9


def test_real_model_with_regression_direction_is_orthogonal_to_the_original_weights(cfg):
    df = make_synthetic_cohort(n=2_000, feature_columns=FEATURES, rng=np.random.default_rng(26), prevalence=0.3)
    fitted = RealDeployedModel(cfg).fit(df, np.random.default_rng(27))
    original_w = fitted._clf.coef_[0].copy()
    regressed = fitted.with_regression(1.0, seed=1)   # full replacement: the new weight vector must be orthogonal to the original
    new_w = regressed._clf.coef_[0]
    cosine = (new_w @ original_w) / (np.linalg.norm(new_w) * np.linalg.norm(original_w))
    assert abs(cosine) < 0.05


def test_real_model_decision_boundary_is_inclusive_of_the_threshold(cfg):
    """`apply`'s comparison must be `>=`, not `>`: a case scored exactly at the threshold is flagged."""
    df = make_synthetic_cohort(n=200, feature_columns=FEATURES, rng=np.random.default_rng(16), prevalence=0.3)
    fitted = RealDeployedModel(cfg).fit(df, np.random.default_rng(17))
    one_row = df.iloc[:1]
    scaled = fitted._scaler.transform(one_row[list(FEATURES)].to_numpy())
    exact_p = float(fitted._clf.predict_proba(scaled)[:, 1][0])
    fitted.threshold = exact_p   # the row's own score is now exactly the threshold
    assert fitted.apply(one_row)["decision"].iloc[0] == 1


def test_real_model_actually_learns_the_label_and_is_not_a_tautology(cfg):
    """The threshold-by-quantile construction hits the target sensitivity even for an uninformative model,
    so the fit must be checked separately for whether the features actually predict the label."""
    df = make_synthetic_cohort(n=6_000, feature_columns=FEATURES, rng=np.random.default_rng(18), prevalence=0.3)
    fitted = RealDeployedModel(cfg).fit(df.iloc[:3000], np.random.default_rng(19))
    scored = fitted.apply(df.iloc[3000:])
    auroc = roc_auc_score(scored["label"], scored["p_hat"])
    assert auroc > 0.65


def test_assign_windows_is_monotonic_in_order_key_and_truncates_the_remainder(cfg):
    df = make_synthetic_cohort(n=133, feature_columns=FEATURES, rng=np.random.default_rng(6), prevalence=0.2)
    windowed = assign_windows(df, cfg)
    assert len(windowed) == 100                        # 133 // 50 * 50, remainder dropped
    assert windowed["window"].nunique() == 2
    assert windowed.groupby("window")["order_key"].count().eq(cfg.window_size).all()
    # every window's order keys precede the next window's
    bounds = windowed.groupby("window")["order_key"].agg(["min", "max"])
    assert bounds.loc[0, "max"] < bounds.loc[1, "min"]


def test_assign_windows_breaks_order_key_ties_by_original_row_order(cfg):
    """A stable sort preserves relative row order among cases with the same order key (e.g. the same calendar day)."""
    df = pd.DataFrame({"order_key": [0] * cfg.window_size, "label": range(cfg.window_size),
                       "subgroup": [0] * cfg.window_size, **{c: [0.0] * cfg.window_size for c in FEATURES}})
    windowed = assign_windows(df, cfg)
    assert list(windowed["label"]) == list(range(cfg.window_size))


def test_assign_windows_raises_if_fewer_rows_than_one_window(cfg):
    df = make_synthetic_cohort(n=10, feature_columns=FEATURES, rng=np.random.default_rng(7), prevalence=0.2)
    with pytest.raises(ValueError, match="at least one window"):
        assign_windows(df, cfg)


def test_build_stream_data_shapes_and_baseline_split(cfg, harm):
    df = make_synthetic_cohort(n=1_000, feature_columns=FEATURES, rng=np.random.default_rng(8), prevalence=0.25)
    fitted = RealDeployedModel(cfg).fit(df.iloc[:500], np.random.default_rng(9))
    scored = fitted.apply(df)
    windowed = assign_windows(scored, cfg)
    n_windows = windowed["window"].nunique()
    stream = build_stream_data(windowed, cfg, harm, n_baseline=3, rng=np.random.default_rng(10))
    assert stream.y.shape == (n_windows, cfg.window_size)
    assert stream.g.shape == (n_windows, cfg.window_size)
    assert stream.severity.shape == (n_windows, cfg.window_size)
    assert stream.decision.shape == (n_windows, cfg.window_size)
    assert stream.reported_p.shape == (n_windows, cfg.window_size)
    assert stream.x_model.shape == (n_windows, cfg.window_size, len(FEATURES))
    assert stream.n_baseline == 3
    assert stream.progress.shape == (n_windows - 3,)
    assert stream.severity.min() >= 1 and stream.severity.max() <= 3


def test_build_stream_data_rejects_n_baseline_larger_than_the_window_count(cfg, harm):
    df = make_synthetic_cohort(n=200, feature_columns=FEATURES, rng=np.random.default_rng(20), prevalence=0.2)
    fitted = RealDeployedModel(cfg).fit(df, np.random.default_rng(21))
    windowed = assign_windows(fitted.apply(df), cfg)
    n_windows = windowed["window"].nunique()
    with pytest.raises(ValueError, match="n_baseline"):
        build_stream_data(windowed, cfg, harm, n_baseline=n_windows + 1, rng=np.random.default_rng(22))


def test_synthetic_cohort_features_actually_separate_by_label(cfg):
    """The stand-in's features must depend on the sign of the label, or a model fit on them cannot be informative."""
    df = make_synthetic_cohort(n=4_000, feature_columns=FEATURES, rng=np.random.default_rng(23), prevalence=0.3)
    pos_mean = df.loc[df["label"] == 1, list(FEATURES)].mean().mean()
    neg_mean = df.loc[df["label"] == 0, list(FEATURES)].mean().mean()
    assert pos_mean - neg_mean > 0.3


def test_build_stream_data_severity_matches_the_elicited_distribution(cfg, harm):
    df = make_synthetic_cohort(n=60_000, feature_columns=FEATURES, rng=np.random.default_rng(11), prevalence=0.3)
    fitted = RealDeployedModel(cfg).fit(df.iloc[:2000], np.random.default_rng(12))
    scored = fitted.apply(df)
    windowed = assign_windows(scored, cfg)
    stream = build_stream_data(windowed, cfg, harm, n_baseline=1, rng=np.random.default_rng(13))
    is_event = stream.y.reshape(-1) == 1
    sev = stream.severity.reshape(-1)
    observed = np.array([(sev[is_event] == c).mean() for c in (1, 2, 3)])
    assert np.abs(observed - np.asarray(harm.severity_probs_positive)).max() < 0.03


def test_build_stream_data_rejects_a_window_size_that_disagrees_with_the_config(cfg, harm):
    df = make_synthetic_cohort(n=200, feature_columns=FEATURES, rng=np.random.default_rng(14), prevalence=0.2)
    windowed = assign_windows(df, cfg)
    windowed = windowed.iloc[:-1]   # break the even-window invariant
    with pytest.raises(ValueError, match="window"):
        build_stream_data(windowed, cfg, harm, n_baseline=1, rng=np.random.default_rng(15))


def test_build_stream_data_defaults_out_of_scope_to_none_when_the_column_is_absent(cfg, harm):
    """Every prior Tier 2 stream (no SC19 scope-creep applied) must keep `out_of_scope=None`, so `window_statistics`'
    C3 falls back to its always-in-scope placeholder exactly as before -- this wiring must not change existing
    behavior for any cohort that never went through `scope_creep_and_rescore`."""
    df = make_synthetic_cohort(n=1_000, feature_columns=FEATURES, rng=np.random.default_rng(80), prevalence=0.25)
    fitted = RealDeployedModel(cfg).fit(df.iloc[:500], np.random.default_rng(81))
    windowed = assign_windows(fitted.apply(df), cfg)
    stream = build_stream_data(windowed, cfg, harm, n_baseline=3, rng=np.random.default_rng(82))
    assert stream.out_of_scope is None


def test_build_stream_data_wires_a_real_out_of_scope_column_when_present(cfg, harm):
    """When `cfg.out_of_scope_column` is present (set by `scope_creep_and_rescore`), `build_stream_data` must read
    it into `StreamData.out_of_scope` with the correct (W, m) shape and values, not silently drop it."""
    df = make_synthetic_cohort(n=1_000, feature_columns=FEATURES, rng=np.random.default_rng(83), prevalence=0.25)
    fitted = RealDeployedModel(cfg).fit(df.iloc[:500], np.random.default_rng(84))
    windowed = assign_windows(fitted.apply(df), cfg)
    windowed["out_of_scope"] = windowed.index % 3 == 0   # an arbitrary, deterministic, non-degenerate pattern
    stream = build_stream_data(windowed, cfg, harm, n_baseline=3, rng=np.random.default_rng(85))
    assert stream.out_of_scope is not None
    assert stream.out_of_scope.shape == (windowed["window"].nunique(), cfg.window_size)
    assert 0 < stream.out_of_scope.mean() < 1   # genuinely mixed, not degenerate all-True/all-False


def test_out_of_scope_reaches_the_real_c3_indicator(cfg, harm):
    """End-to-end: a real `out_of_scope` column, once wired through `build_stream_data`, must make C3 respond --
    proof that C3 is no longer Tier 2's always-1.0 placeholder once scope creep is applied."""
    from src.monitor_module.statistics import window_statistics
    from src.sim_module.config import load_config
    phase2_cfg = load_config()
    df = make_synthetic_cohort(n=2_000, feature_columns=FEATURES, rng=np.random.default_rng(86), prevalence=0.25)
    fitted = RealDeployedModel(cfg).fit(df.iloc[:1000], np.random.default_rng(87))
    windowed = assign_windows(fitted.apply(df), cfg)
    windowed["out_of_scope"] = windowed.index % 4 == 0   # 25% out of scope, deterministic
    stream = build_stream_data(windowed, cfg, harm, n_baseline=1, rng=np.random.default_rng(88))
    c3 = window_statistics(stream, phase2_cfg)["C3"]
    # 1.0 - the out-of-scope share: the deterministic pattern doesn't divide evenly into 50-row windows (12 or 13 of
    # 50 per window, depending on the window's starting parity), so it lands at 0.74/0.76, not exactly 0.75
    assert np.allclose(c3, 0.75, atol=0.02)
