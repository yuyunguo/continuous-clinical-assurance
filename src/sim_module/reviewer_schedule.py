"""Time-varying reviewer parameters for the oversight scenarios SC10 to SC13 (design section 5)."""

from typing import Dict

import numpy as np

from .config import ReviewerConfig

PARAM_OF = {"reviewer_acceptance_of_ai_errors": "automation_bias", "review_coverage_drop": "coverage",
            "review_latency_increase": "latency_median", "blanket_rejection": "false_reject"}
KINDS = ("reviewer_acceptance_of_ai_errors", "review_coverage_drop", "review_latency_increase", "blanket_rejection")


def reviewer_schedule(kind: str, level: float, progress: np.ndarray, base: ReviewerConfig) -> Dict[str, np.ndarray]:
    """Per-window reviewer parameters for a perturbation with the given progress in [0, 1].

    Levels: added acceptance probability (SC10), the coverage at full effect (SC11), the latency multiplier at full
    effect (SC13), and added probability of rejecting a correct output (SC12).

    Raises:
        ValueError: When `kind` is not a reviewer perturbation.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown reviewer perturbation {kind!r}")
    n = len(progress)
    out = {"coverage": np.full(n, base.coverage), "automation_bias": np.full(n, base.automation_bias),
           "false_reject": np.full(n, base.false_reject), "latency_median": np.full(n, base.latency_median)}
    if kind == "reviewer_acceptance_of_ai_errors":
        out["automation_bias"] = base.automation_bias + level * progress
    elif kind == "review_coverage_drop":
        out["coverage"] = base.coverage + (level - base.coverage) * progress
    elif kind == "review_latency_increase":
        out["latency_median"] = base.latency_median * (1.0 + (level - 1.0) * progress)
    else:
        out["false_reject"] = base.false_reject + level * progress
    return out
