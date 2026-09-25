"""Reviewer model (an assumption, not a measurement). Expected interception probability for AI errors."""

import numpy as np
from scipy.stats import norm

from src.eval_module.metrics import sigmoid

from .config import ReviewerConfig
from .generator import Batch


def recognition_probability(decision: np.ndarray, reported_p: np.ndarray, severity: np.ndarray, beta0: float, beta_confidence: float,
                            beta_severity: float, automation_bias: np.ndarray) -> np.ndarray:
    """Probability that a reviewer who examines an error recognizes it, for arrays of any shape."""
    confidence = np.where(decision == 1, reported_p, 1.0 - reported_p)
    detect = sigmoid(beta0 + beta_confidence * (1.0 - confidence) + beta_severity * (severity - 1))
    return (1.0 - automation_bias) * detect


def detection_probability(batch: Batch, cfg: ReviewerConfig) -> np.ndarray:
    """Probability that a reviewer who examines an AI error recognizes it.

    Reviewers catch errors more often when the displayed confidence in the decision is low and when the error is
    more severe. `reported_p` is the confidence the reviewer sees, so a distorted confidence changes behavior.
    Automation bias is the probability of accepting the output whatever its quality.
    """
    return recognition_probability(batch.decision, batch.reported_p, batch.severity, cfg.beta0, cfg.beta_confidence,
                                   cfg.beta_severity, np.asarray(cfg.automation_bias))


def in_time_probability(cfg: ReviewerConfig) -> float:
    """Probability that a review finishes before the point of no return (lognormal latency against the deadline)."""
    if not np.isfinite(cfg.deadline):
        return 1.0
    return float(norm.cdf(np.log(cfg.deadline / cfg.latency_median) / cfg.latency_sigma))


def interception_probability(batch: Batch, cfg: ReviewerConfig) -> np.ndarray:
    """Probability that a reviewer intercepts an error before the point of no return (coverage, in time, and detection)."""
    return cfg.coverage * in_time_probability(cfg) * detection_probability(batch, cfg)
