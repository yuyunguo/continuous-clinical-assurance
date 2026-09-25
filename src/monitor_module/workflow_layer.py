"""Workflow outcomes for every case of a stream, and the per-window indicators H7 and O1 to O6 (instrument domains C and D)."""

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np

from src.sim_module.config import Phase2Config

from .detectors import apply_label_lag, standardize
from .review_layer import ReviewStream
from .streams import StreamData

WORKFLOW_ORDER = ("H7", "O1", "O2", "O3", "O4", "O5", "O6", "O3a")   # O3a is the action-rate half of O3 (burden is O3)
# +1: an increase is bad, -1: a decrease is bad. O5 and O6 need incident records linked to AI outputs, so they arrive with the label lag.
WORKFLOW_DIRECTION = {"H7": -1, "O1": -1, "O2": 1, "O3": 1, "O4": 1, "O5": 1, "O6": 1, "O3a": -1}
WORKFLOW_LABEL_DEPENDENT = ("O5", "O6")


@dataclass(frozen=True)
class WorkflowStream:
    """Arrays of shape (W, m). `required` marks cases that meet the escalation criteria, `alerts` counts alert messages, and `tat` is turnaround."""

    initiated: np.ndarray
    required: np.ndarray
    delivered: np.ndarray
    late: np.ndarray
    completed: np.ndarray
    alerts: np.ndarray
    tat: np.ndarray
    downstream: np.ndarray    # an AI error that neither the reviewer (in time) nor downstream verification caught
    sentinel: np.ndarray
    actioned: np.ndarray      # alert messages acted on (the action-rate half of O3)


def build_workflow(stream: StreamData, review: ReviewStream, cfg: Phase2Config, par: Dict[str, np.ndarray], rng: np.random.Generator) -> WorkflowStream:
    """Sample workflow outcomes from per-window parameters `p_deliver`, `tat_mult`, `alert_dup`, and `verify` (each of length W)."""
    w, m = stream.y.shape
    wf = cfg.workflow
    col = lambda name: par[name][:, None]  # noqa: E731
    initiated = stream.decision == 1
    required = initiated & (stream.severity >= cfg.harm.severity_cutoff)
    tat = wf.tat_median * col("tat_mult") * np.exp(wf.tat_sigma * rng.standard_normal((w, m)))
    delivered = required & (rng.random((w, m)) < col("p_deliver"))
    late = delivered & (tat > wf.tat_limit)
    p_complete = np.where(required & ~delivered, wf.completion * (1.0 - wf.abandon_after_failed_escalation), wf.completion)
    completed = initiated & (rng.random((w, m)) < p_complete)
    alerts = initiated * (1 + rng.poisson(np.maximum(col("alert_dup") - 1.0, 0.0), size=(w, m)))
    caught = review.in_time & review.recognized
    downstream = review.error & ~caught & (rng.random((w, m)) >= col("verify"))
    p_action = wf.action_base / (1.0 + wf.fatigue * np.maximum(col("alert_dup") - 1.0, 0.0))     # fatigue: more duplicated alerts, fewer acted on
    actioned = rng.binomial(alerts, np.broadcast_to(p_action, alerts.shape))
    return WorkflowStream(initiated=initiated, required=required, delivered=delivered, late=late, completed=completed, alerts=alerts, tat=tat,
                          downstream=downstream, sentinel=downstream & (stream.severity == wf.sentinel_severity), actioned=actioned)


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.where(den > 0, num / np.maximum(den, 1), np.nan)


def workflow_statistics(wf: WorkflowStream) -> Dict[str, np.ndarray]:
    """Per-window H7 and O1 to O6 (length W)."""
    m = wf.tat.shape[1]
    return {"H7": _ratio(wf.delivered.sum(axis=1), wf.required.sum(axis=1)),
            "O1": _ratio(wf.completed.sum(axis=1), wf.initiated.sum(axis=1)),
            "O2": _ratio((wf.required & (~wf.delivered | wf.late)).sum(axis=1), wf.required.sum(axis=1)),
            "O3": wf.alerts.sum(axis=1) / m,
            "O3a": _ratio(wf.actioned.sum(axis=1), wf.alerts.sum(axis=1)),
            "O4": np.percentile(wf.tat, 95, axis=1),
            "O5": wf.downstream.sum(axis=1) / m,
            "O6": wf.sentinel.sum(axis=1) / m}


def workflow_scores_multi(stream: StreamData, wf: WorkflowStream, lags: Sequence[int]) -> Dict[int, np.ndarray]:
    """Standardized H7 and O1 to O6 scores per label lag, shape (T, 7) each. Only O5 and O6 are lagged."""
    stats = workflow_statistics(wf)
    series = np.stack([stats[i] for i in WORKFLOW_ORDER], axis=1)[None]
    direction = np.array([WORKFLOW_DIRECTION[i] for i in WORKFLOW_ORDER], dtype=float)
    z_full = standardize(series, stream.n_baseline, direction, full=True)
    dependent = np.array([i in WORKFLOW_LABEL_DEPENDENT for i in WORKFLOW_ORDER])
    free = apply_label_lag(z_full, stream.n_baseline, 0)
    return {lag: np.where(dependent[None, None, :], apply_label_lag(z_full, stream.n_baseline, lag), free)[0] for lag in lags}
