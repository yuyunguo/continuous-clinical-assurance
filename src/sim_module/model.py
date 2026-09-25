"""The deployed model: a logistic regression on the features (subgroup-blind) with a fixed decision threshold."""

import copy
from dataclasses import replace
from typing import Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

from .config import GeneratorConfig
from .generator import Batch, SyntheticWorld


class DeployedModel:
    """Logistic model fit once on a training sample and then frozen."""

    def __init__(self, cfg: GeneratorConfig) -> None:
        self.cfg = cfg
        self._clf = LogisticRegression(C=np.inf, max_iter=500)
        self.threshold: float = 0.5
        self.impute_means: np.ndarray = np.zeros(cfg.n_features)

    def fit(self, world: SyntheticWorld, rng: np.random.Generator) -> "DeployedModel":
        """Fit on a fresh training sample and set the threshold that gives the target sensitivity."""
        train = world.sample(self.cfg.train_size, rng)
        self._clf.fit(train.x, train.y)
        self.impute_means = train.x.mean(axis=0)
        p = self._clf.predict_proba(train.x)[:, 1]
        self.threshold = float(np.quantile(p[train.y == 1], 1 - self.cfg.target_sensitivity))
        return self

    def with_regression(self, fraction: float, seed: int = 0) -> "DeployedModel":
        """A copy whose weight vector has a fraction replaced by an orthogonal direction of equal length.

        Models a version update that regresses. The threshold and imputation means are unchanged, and the original
        model is not modified.
        """
        clone = copy.deepcopy(self)
        if fraction > 0:
            w = clone._clf.coef_[0]
            r = np.random.default_rng(seed).standard_normal(len(w))
            r = r - (r @ w) / (w @ w) * w
            clone._clf.coef_ = ((1.0 - fraction) * w + fraction * np.linalg.norm(w) * r / np.linalg.norm(r))[None, :]
        return clone

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predicted probability of the outcome."""
        return self._clf.predict_proba(x)[:, 1]

    def apply(self, batch: Batch, missing_mask: Optional[np.ndarray] = None,
              rescale: Optional[Tuple[int, float]] = None) -> Batch:
        """Run the frozen model on a batch, applying input faults (missingness, rescaling) before scoring."""
        x_model = batch.x.copy()
        if rescale is not None:
            column, factor = rescale
            x_model[:, column] = x_model[:, column] * factor
        if missing_mask is not None:
            x_model = np.where(missing_mask, self.impute_means[None, :], x_model)
        p_hat = self.predict_proba(x_model)
        mask = np.zeros_like(batch.x, dtype=bool) if missing_mask is None else missing_mask
        return replace(batch, x_model=x_model, missing_mask=mask, p_hat=p_hat, reported_p=p_hat,
                       decision=(p_hat >= self.threshold).astype(int))
