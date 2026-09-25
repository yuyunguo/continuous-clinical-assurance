"""The Tier 2 deployed model: a logistic regression on real (or synthetic stand-in) features, frozen after fitting.

Mirrors `src.sim_module.model.DeployedModel`: fit once, then a fixed threshold set to hit a target sensitivity on
the training labels.

Real data has genuine missingness (unlike the synthetic stand-in, which has none by default): `LogisticRegression`
rejects NaN outright, so `fit` and `apply` impute with the training column means (`impute_means`, the same field
`missingness_and_rescore` uses for the SC05/SC06 perturbation) before ever calling into sklearn.

Real, correlated clinical features on their raw scale can also make an unregularized fit produce degenerate
predictions (a probability as extreme as 1e-135 was observed on real MIMIC-IV readmission data), which broke a
downstream indicator's numerically unstable Newton's-method fit (`Tier2-Readiness.md` §8, the readmission anomaly).
Standardizing the features and choosing the L2 strength by cross-validated log loss fixes this: log loss is a
pure model-fit criterion, unrelated to any monitoring arm's detection speed, so this is not "tuning to favor a
monitoring arm" in the sense the project's coding rules warn against — it changes the deployed model's own
predictions on their own fit-quality terms, before any monitor ever sees them.

Real vital/lab extraction also carries a small number of implausible extreme values — well under 0.1% of rows per
affected column, e.g. a blood-pressure feature with a clinically plausible 99.9th percentile around 227 but a raw
maximum of 8,999,090 (`Tier2-Readiness.md` §9) — almost certainly MIMIC-IV chart-error artifacts, not genuine
physiology. These went unnoticed by the model itself (a handful of rows in 80,000+) but drove the C2 monitoring
indicator's calibration search to its ceiling, since C2 sums *squared* per-feature deviations and squaring
amplifies exactly this kind of rare extreme outlier. Fixed by winsorizing each feature at percentiles fit on the
training cohort only (no leakage, same discipline as `impute_means`/`StandardScaler`): a data-driven cutoff, not a
hand-picked clinical bound per feature, chosen because the diagnostic showed genuine extreme-but-real values sit
comfortably inside the 99.9th percentile while the erroneous ones sit far beyond it."""

import copy
import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

from .config import Tier2StreamConfig

REGULARIZATION_GRID = tuple(np.logspace(-4, 1, 10))   # candidate L2 strengths (C); chosen by 5-fold cross-validated log loss, not by hand
CV_FOLDS = 5
WINSORIZE_LOWER_PERCENTILE = 0.1
WINSORIZE_UPPER_PERCENTILE = 99.9


class RealDeployedModel:
    """Logistic model fit once on a training cohort and then frozen."""

    def __init__(self, cfg: Tier2StreamConfig) -> None:
        self.cfg = cfg
        self._clf = LogisticRegressionCV(Cs=REGULARIZATION_GRID, cv=CV_FOLDS, scoring="neg_log_loss",
                                         penalty="l2", max_iter=2000)
        self._scaler = StandardScaler()
        self.threshold: float = 0.5
        self.impute_means: Dict[str, float] = {c: 0.0 for c in cfg.feature_columns}
        self.winsorize_bounds: Dict[str, Tuple[float, float]] = {c: (-np.inf, np.inf) for c in cfg.feature_columns}

    def _winsorize(self, x: np.ndarray) -> np.ndarray:
        """Clip each feature to its `winsorize_bounds`. NaN passes through unchanged (`np.clip` propagates NaN),
        so this composes safely with `_impute` in either order."""
        lo = np.array([self.winsorize_bounds[c][0] for c in self.cfg.feature_columns])
        hi = np.array([self.winsorize_bounds[c][1] for c in self.cfg.feature_columns])
        return np.clip(x, lo, hi)

    def _impute(self, x: np.ndarray) -> np.ndarray:
        """Replace NaN in each feature column with that column's `impute_means` value."""
        x = x.copy()
        means = np.array([self.impute_means[c] for c in self.cfg.feature_columns])
        rows, cols = np.where(np.isnan(x))
        x[rows, cols] = means[cols]
        return x

    def fit(self, train: pd.DataFrame, rng: np.random.Generator) -> "RealDeployedModel":
        """Fit on `train` and set the threshold that gives the target sensitivity on its own positive cases."""
        del rng   # accepted for interface symmetry with `DeployedModel.fit`; the logistic fit itself is deterministic given `train`
        x_raw = train[list(self.cfg.feature_columns)].to_numpy()
        # an all-missing training column makes both `nanpercentile` and `nanmean` below return NaN for it, with a
        # RuntimeWarning ("All-NaN slice"/"Mean of empty slice") -- expected and handled explicitly right after
        # each call (falling back to a no-op bound / to 0.0), so the warning itself is suppressed here rather than
        # left to alarm on an already-handled, already-tested edge case.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            lo_p = np.nanpercentile(x_raw, WINSORIZE_LOWER_PERCENTILE, axis=0)
            hi_p = np.nanpercentile(x_raw, WINSORIZE_UPPER_PERCENTILE, axis=0)
        # an all-missing training column has no real percentile to bound with; falling back to (nan, nan) would make
        # `_winsorize` (via `np.clip`) turn EVERY value NaN for that column on every future `apply()` call, including
        # a later cohort where the column has genuine, valid values -- so it falls back to the no-op (-inf, inf)
        # bound instead, matching `impute_means`' own 0.0 fallback for the same edge case.
        self.winsorize_bounds = {c: ((float(lo_p[i]) if np.isfinite(lo_p[i]) else -np.inf),
                                     (float(hi_p[i]) if np.isfinite(hi_p[i]) else np.inf))
                                 for i, c in enumerate(self.cfg.feature_columns)}
        x_winsorized = self._winsorize(x_raw)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            means = np.nanmean(x_winsorized, axis=0)   # computed post-winsorization, so a handful of extreme rows don't skew it
        self.impute_means = {c: (float(means[i]) if np.isfinite(means[i]) else 0.0) for i, c in enumerate(self.cfg.feature_columns)}
        x_imputed = self._impute(x_winsorized)
        x = self._scaler.fit_transform(x_imputed)
        y = train[self.cfg.label_column].to_numpy()
        self._clf.fit(x, y)
        p = self._clf.predict_proba(x)[:, 1]
        self.threshold = float(np.quantile(p[y == 1], 1 - self.cfg.target_sensitivity))
        return self

    def with_regression(self, fraction: float, seed: int = 0) -> "RealDeployedModel":
        """A copy whose weight vector has a fraction replaced by an orthogonal direction of equal length.

        Models a version update that regresses (SC17), mirroring `src.sim_module.model.DeployedModel.with_regression`.
        The threshold and imputation means are unchanged, and the original model is not modified.
        """
        clone = copy.deepcopy(self)
        if fraction > 0:
            w = clone._clf.coef_[0]
            r = np.random.default_rng(seed).standard_normal(len(w))
            r = r - (r @ w) / (w @ w) * w
            clone._clf.coef_ = ((1.0 - fraction) * w + fraction * np.linalg.norm(w) * r / np.linalg.norm(r))[None, :]
        return clone

    def apply(self, cohort: pd.DataFrame) -> pd.DataFrame:
        """Copy of `cohort` with `p_hat` and `decision` columns added, and its feature columns winsorized+imputed.

        The winsorized, imputed values are written back into the returned feature columns, not used only internally
        for scoring: `stream.py`'s `build_stream_data` reads `x_model` straight from these columns, and any
        indicator that touches it (e.g. C2, the data-drift statistic) would otherwise still see real NaN or the raw
        extreme outliers `winsorize_bounds` exists to contain. The returned features stay in their original
        (winsorized, imputed) units — standardization is internal to scoring only, so a perturbation like
        `rescale_and_rescore` still operates on real, interpretable feature values.
        """
        x_winsorized = self._winsorize(cohort[list(self.cfg.feature_columns)].to_numpy())
        x_imputed = self._impute(x_winsorized)
        p_hat = self._clf.predict_proba(self._scaler.transform(x_imputed))[:, 1]
        scored = cohort.copy()
        scored[list(self.cfg.feature_columns)] = x_imputed
        scored["p_hat"] = p_hat
        scored["decision"] = (p_hat >= self.threshold).astype(int)
        return scored
