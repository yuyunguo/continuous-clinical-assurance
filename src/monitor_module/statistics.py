"""Vectorized per-window statistics for the indicators testable with the data and model layer."""

from typing import Dict

import numpy as np
from scipy.stats import rankdata

from src.eval_module.metrics import EPS, logit, sigmoid
from src.sim_module.config import Phase2Config
from src.sim_module.harm import error_weights

from .streams import StreamData

INDICATOR_ORDER = ("P1", "P2", "P3", "P4", "P5", "P6", "U1", "U2", "U3", "U4", "U5", "C2", "C3")
# +1 means an increase is bad, -1 a decrease is bad. Two-sided statistics are already absolute deviations (+1).
DIRECTION = {"P1": -1, "P2": -1, "P3": -1, "P4": -1, "P5": 1, "P6": 1, "U1": 1, "U2": 1, "U3": 1, "U4": 1, "U5": 1, "C2": 1, "C3": -1}
INDICATOR_COMPONENT = {"P1": "P", "P2": "P", "P3": "P", "P4": "P", "P5": "P", "P6": "P", "U1": "U", "U2": "U", "U3": "U",
                       "U4": "U", "U5": "U", "C2": "C", "C3": "C"}   # state components of the instrument records
LABEL_DEPENDENT = ("P1", "P2", "P3", "P4", "P5", "P6", "U1", "U2", "U3", "U4")
HIGH_CONFIDENCE = 0.9
ECE_BINS = 15


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a / b with NaN where b is zero."""
    return np.where(b > 0, a / np.maximum(b, 1), np.nan)


def _auroc(y: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Per-window AUROC from ranks."""
    ranks = rankdata(score, axis=1)
    n1 = y.sum(axis=1)
    n0 = y.shape[1] - n1
    value = ((ranks * y).sum(axis=1) - n1 * (n1 + 1) / 2) / np.maximum(n1 * n0, 1)
    return np.where((n1 > 0) & (n0 > 0), value, np.nan)


def _ece(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Per-window expected calibration error: sum over bins of |sum(y) - sum(p)| / m."""
    index = np.minimum((p * ECE_BINS).astype(int), ECE_BINS - 1)
    total = np.zeros(y.shape[0])
    for b in range(ECE_BINS):
        in_bin = index == b
        total += np.abs((y * in_bin).sum(axis=1) - (p * in_bin).sum(axis=1))
    return total / y.shape[1]


def _slope_deviation(y: np.ndarray, p: np.ndarray, iterations: int = 30) -> np.ndarray:
    """Per-window |calibration slope - 1|, by Newton's method run on all windows at once."""
    z = logit(p)
    b0 = np.zeros(y.shape[0])
    b1 = np.ones(y.shape[0])
    for _ in range(iterations):
        mu = sigmoid(b0[:, None] + b1[:, None] * z)
        w = mu * (1 - mu) + EPS
        g0, g1 = (y - mu).sum(axis=1), (z * (y - mu)).sum(axis=1)
        h00, h01, h11 = w.sum(axis=1), (w * z).sum(axis=1), (w * z * z).sum(axis=1)
        det = h00 * h11 - h01 ** 2
        det = np.where(np.abs(det) < 1e-12, 1e-12, det)
        b0, b1 = b0 + (h11 * g0 - h01 * g1) / det, b1 + (h00 * g1 - h01 * g0) / det
    return np.abs(b1 - 1.0)


def _ks_to_reference(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Per-window Kolmogorov-Smirnov distance between a window's values and a sorted reference sample."""
    m = values.shape[1]
    sorted_values = np.sort(values, axis=1)
    f_ref = np.searchsorted(reference, sorted_values, side="right") / len(reference)
    upper = (np.arange(1, m + 1) / m)[None, :]
    lower = (np.arange(0, m) / m)[None, :]
    return np.maximum(np.abs(upper - f_ref).max(axis=1), np.abs(lower - f_ref).max(axis=1))


def window_statistics(stream: StreamData, cfg: Phase2Config) -> Dict[str, np.ndarray]:
    """Per-window statistic for each indicator, over baseline and monitored windows (length W)."""
    y, d, p = stream.y.astype(float), stream.decision.astype(float), stream.reported_p
    error = (stream.decision != stream.y).astype(float)
    weights = error_weights(stream.severity, stream.y, stream.decision, cfg)
    base = stream.n_baseline
    confidence = np.where(stream.decision == 1, p, 1 - p)
    high = confidence >= HIGH_CONFIDENCE
    subgroup = stream.g == 1
    reference = np.sort(p[:base].ravel())
    mean_ref = stream.x_model[:base].mean(axis=(0, 1))
    return {
        "P1": _safe_div((y * d).sum(axis=1), y.sum(axis=1)),
        "P2": _safe_div(((1 - y) * (1 - d)).sum(axis=1), (1 - y).sum(axis=1)),
        "P3": _safe_div((y * d).sum(axis=1), d.sum(axis=1)),
        "P4": _auroc(stream.y, p),
        "P5": 1000.0 * (weights * error).mean(axis=1),
        "P6": np.abs(_safe_div((error * subgroup).sum(axis=1), subgroup.sum(axis=1))
                     - _safe_div((error * ~subgroup).sum(axis=1), (~subgroup).sum(axis=1))),
        "U1": _ece(y, p),
        "U2": ((p - y) ** 2).mean(axis=1),
        "U3": _slope_deviation(y, p),
        "U4": _safe_div((error * high).sum(axis=1), high.sum(axis=1)),
        "U5": _ks_to_reference(p, reference),
        "C2": stream.x_model.shape[1] * ((stream.x_model.mean(axis=1) - mean_ref) ** 2).sum(axis=1),
        "C3": np.ones(stream.y.shape[0]) if stream.out_of_scope is None else 1.0 - stream.out_of_scope.mean(axis=1),
    }
