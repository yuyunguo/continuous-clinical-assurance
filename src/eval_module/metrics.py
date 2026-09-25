"""Population and sample metrics used by the indicator monitors and the property tests."""

import numpy as np
from scipy.special import expit
from scipy.stats import rankdata

EPS = 1e-9


def logit(p: np.ndarray) -> np.ndarray:
    """Log-odds with clipping so that probabilities of 0 or 1 stay finite."""
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Logistic function."""
    return expit(z)


def sensitivity(y: np.ndarray, decision: np.ndarray) -> float:
    """Se = TP / (TP + FN)."""
    positives = y == 1
    return float(np.mean(decision[positives] == 1)) if positives.any() else float("nan")


def specificity(y: np.ndarray, decision: np.ndarray) -> float:
    """Sp = TN / (TN + FP)."""
    negatives = y == 0
    return float(np.mean(decision[negatives] == 0)) if negatives.any() else float("nan")


def ppv(y: np.ndarray, decision: np.ndarray) -> float:
    """PPV = TP / (TP + FP)."""
    flagged = decision == 1
    return float(np.mean(y[flagged] == 1)) if flagged.any() else float("nan")


def npv(y: np.ndarray, decision: np.ndarray) -> float:
    """NPV = TN / (TN + FN)."""
    cleared = decision == 0
    return float(np.mean(y[cleared] == 0)) if cleared.any() else float("nan")


def auroc(y: np.ndarray, score: np.ndarray) -> float:
    """Area under the ROC curve from ranks (Mann-Whitney)."""
    ranks = rankdata(score)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error with equal-width bins: sum_b (n_b / N) * |mean(y) - mean(p)|."""
    index = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        in_bin = index == b
        if in_bin.any():
            total += in_bin.mean() * abs(y[in_bin].mean() - p[in_bin].mean())
    return float(total)


def calibration_slope_intercept(y: np.ndarray, p: np.ndarray, iterations: int = 30) -> "tuple[float, float]":
    """Fit logit P(y=1) = a + b * logit(p) by Newton's method and return (b, a)."""
    z = logit(p)
    design = np.column_stack([np.ones_like(z), z])
    beta = np.array([0.0, 1.0])
    for _ in range(iterations):
        mu = sigmoid(design @ beta)
        weights = mu * (1 - mu) + EPS
        step = np.linalg.solve(design.T @ (design * weights[:, None]), design.T @ (y - mu))
        beta = beta + step
        if np.abs(step).max() < 1e-10:
            break
    return float(beta[1]), float(beta[0])


def calibration_slope(y: np.ndarray, p: np.ndarray) -> float:
    """Calibration slope b of the logistic recalibration."""
    return calibration_slope_intercept(y, p)[0]


def brier(y: np.ndarray, p: np.ndarray) -> float:
    """Brier score."""
    return float(np.mean((p - y) ** 2))
