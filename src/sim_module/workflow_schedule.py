"""Time-varying workflow parameters for SC14, SC15, SC16, and SC22 (design section 5)."""

from typing import Dict

import numpy as np

from .config import WorkflowConfig

KINDS = ("escalation_delivery_failure", "alert_duplication", "turnaround_time_increase", "error_propagation_unchecked")
PARAM_OF = {"escalation_delivery_failure": "p_deliver", "alert_duplication": "alert_dup", "turnaround_time_increase": "tat_mult",
            "error_propagation_unchecked": "verify"}


def workflow_schedule(kind: str, level: float, progress: np.ndarray, base: WorkflowConfig) -> Dict[str, np.ndarray]:
    """Per-window workflow parameters for a perturbation with the given progress in [0, 1].

    Levels: the share of escalations that additionally fail (SC14), the alert multiplier at full effect (SC15), the turnaround multiplier at
    full effect (SC16), and the drop in the probability that downstream steps catch an error (SC22).

    Raises:
        ValueError: When `kind` is not a workflow perturbation.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown workflow perturbation {kind!r}")
    n = len(progress)
    out = {"p_deliver": np.full(n, base.p_deliver), "tat_mult": np.ones(n), "alert_dup": np.ones(n), "verify": np.full(n, base.verify)}
    if kind == "escalation_delivery_failure":
        out["p_deliver"] = base.p_deliver * (1.0 - level * progress)
    elif kind == "turnaround_time_increase":
        out["tat_mult"] = 1.0 + (level - 1.0) * progress
    elif kind == "alert_duplication":
        out["alert_dup"] = 1.0 + (level - 1.0) * progress
    else:
        out["verify"] = np.maximum(0.0, base.verify - level * progress)
    return out
