"""One schedule over both layers, so a scenario from either layer can be applied to the same stream (design sections 5 and 11)."""

from typing import Dict

import numpy as np

from .config import Phase2Config
from .reviewer_schedule import KINDS as REVIEWER_KINDS
from .reviewer_schedule import reviewer_schedule
from .workflow_schedule import KINDS as WORKFLOW_KINDS
from .workflow_schedule import workflow_schedule


def joint_schedule(kind: str, level: float, progress: np.ndarray, cfg: Phase2Config) -> Dict[str, np.ndarray]:
    """Per-window reviewer and workflow parameters for one perturbation, with the other layer at baseline.

    A slower review lengthens the turnaround it is part of, so the reviewer's latency multiplier also scales `tat_mult` (SC13).

    Raises:
        ValueError: When `kind` belongs to neither layer.
    """
    if kind not in REVIEWER_KINDS and kind not in WORKFLOW_KINDS:
        raise ValueError(f"unknown perturbation {kind!r}")
    rv = reviewer_schedule(kind, level, progress, cfg.reviewer) if kind in REVIEWER_KINDS else reviewer_schedule("review_coverage_drop", cfg.reviewer.coverage, progress, cfg.reviewer)
    wf = workflow_schedule(kind, level, progress, cfg.workflow) if kind in WORKFLOW_KINDS else workflow_schedule("alert_duplication", 1.0, progress, cfg.workflow)
    wf["tat_mult"] = wf["tat_mult"] * rv["latency_median"] / cfg.reviewer.latency_median
    # Alert fatigue (the raters' comment): duplicated alerts make reviewers accept more, by a size that is off unless `fatigue_bias` is set.
    rv["automation_bias"] = rv["automation_bias"] + cfg.workflow.fatigue_bias * (1.0 - 1.0 / wf["alert_dup"])
    return {**rv, **wf}
