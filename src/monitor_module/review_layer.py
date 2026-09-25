"""Review outcomes for every case of a stream, and the per-window oversight indicators H1 to H6 (instrument domain C)."""

import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

from src.sim_module.config import Phase2Config
from src.sim_module.reviewer import recognition_probability
from src.sim_module.reviewer_schedule import reviewer_schedule

from .detectors import apply_label_lag, standardize
from .streams import StreamData

REVIEW_ORDER = ("H1", "H2", "H3", "H4", "H5", "H6", "H2u")   # H2u is the upward arm of H2, whose record charts the override rate in both directions
# +1: an increase is bad, -1: a decrease is bad. H3, H4, and H5 need adjudicated errors, so their values arrive with the label lag.
REVIEW_DIRECTION = {"H1": -1, "H2": -1, "H3": -1, "H4": 1, "H5": -1, "H6": 1, "H2u": 1}
REVIEW_LABEL_DEPENDENT = ("H3", "H4", "H5")


@dataclass(frozen=True)
class ReviewStream:
    """Boolean and latency arrays of shape (W, m). `examined` is documented review, `in_time` is examined before the point of no return."""

    examined: np.ndarray
    in_time: np.ndarray
    recognized: np.ndarray    # the reviewer recognized an AI error (whenever it happened)
    wrong: np.ndarray         # a correct output was examined and wrongly rejected
    error: np.ndarray
    required: np.ndarray      # an error at or above the severity cutoff (R)
    latency: np.ndarray


def build_review(stream: StreamData, cfg: Phase2Config, kind: Optional[str], level: float, progress: np.ndarray,
                 rng: np.random.Generator) -> ReviewStream:
    """Sample review outcomes for all windows. `progress` covers the monitored windows; baseline windows use the base reviewer."""
    full = np.concatenate([np.zeros(stream.n_baseline), progress])
    rv = cfg.reviewer
    par = reviewer_schedule(kind, level, full, rv) if kind else reviewer_schedule("review_coverage_drop", rv.coverage, full, rv)
    return build_review_params(stream, cfg, par, rng)


def build_review_params(stream: StreamData, cfg: Phase2Config, par: Dict[str, np.ndarray], rng: np.random.Generator) -> ReviewStream:
    """Sample review outcomes from explicit per-window reviewer parameters (each an array of length W)."""
    w, m = stream.y.shape
    rv = cfg.reviewer
    col = lambda name: par[name][:, None]  # noqa: E731
    error = stream.decision != stream.y
    examined = rng.random((w, m)) < col("coverage")
    latency = col("latency_median") * np.exp(rv.latency_sigma * rng.standard_normal((w, m)))
    recognized = examined & error & (rng.random((w, m)) < recognition_probability(stream.decision, stream.reported_p, stream.severity, rv.beta0,
                                                                              rv.beta_confidence, rv.beta_severity, col("automation_bias")))
    wrong = examined & ~error & (rng.random((w, m)) < col("false_reject"))
    return ReviewStream(examined=examined, in_time=examined & (latency <= rv.deadline), recognized=recognized, wrong=wrong, error=error,
                        required=error & (stream.severity >= cfg.harm.severity_cutoff), latency=latency)


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.where(den > 0, num / np.maximum(den, 1), np.nan)


def review_statistics(rev: ReviewStream, audit_fraction: float, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Per-window H1 to H6 (length W). H5 is estimated from a Poisson audit of `audit_fraction` of the cases."""
    m = rev.examined.shape[1]
    overrides = rev.recognized | rev.wrong
    audited = rng.random(rev.examined.shape) < audit_fraction
    intercepted = rev.required & rev.in_time & rev.recognized
    lat = np.where(rev.examined, rev.latency, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median_latency = np.nanmedian(lat, axis=1)
    override_rate = _ratio(overrides.sum(axis=1), rev.examined.sum(axis=1))
    return {"H1": rev.examined.sum(axis=1) / m,
            "H2": override_rate, "H2u": override_rate,
            "H3": _ratio(rev.recognized.sum(axis=1), overrides.sum(axis=1)),
            "H4": 1.0 - _ratio(rev.recognized.sum(axis=1), (rev.examined & rev.error).sum(axis=1)),
            "H5": _ratio((audited & intercepted).sum(axis=1), (audited & rev.required).sum(axis=1)),
            "H6": median_latency}


def review_scores_multi(stream: StreamData, rev: ReviewStream, audit_fraction: float, lags: Sequence[int],
                        rng: np.random.Generator) -> Dict[int, np.ndarray]:
    """Standardized H1 to H6 scores per label lag, shape (T, 6) each. Only the adjudicated indicators (H3, H4, H5) are lagged."""
    stats = review_statistics(rev, audit_fraction, rng)
    series = np.stack([stats[i] for i in REVIEW_ORDER], axis=1)[None]
    direction = np.array([REVIEW_DIRECTION[i] for i in REVIEW_ORDER], dtype=float)
    z_full = standardize(series, stream.n_baseline, direction, full=True)
    dependent = np.array([i in REVIEW_LABEL_DEPENDENT for i in REVIEW_ORDER])
    free = apply_label_lag(z_full, stream.n_baseline, 0)
    return {lag: np.where(dependent[None, None, :], apply_label_lag(z_full, stream.n_baseline, lag), free)[0] for lag in lags}
