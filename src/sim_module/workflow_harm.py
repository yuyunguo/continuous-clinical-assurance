"""Extended expected harm with workflow terms, used to define onset for the workflow scenarios (design sections 4.3 and 11)."""

from dataclasses import replace
from typing import Dict, Tuple

import numpy as np
from scipy.stats import norm

from .config import Phase2Config
from .generator import Batch
from .harm import expected_rhe

REVIEWER_KEYS = ("coverage", "automation_bias", "false_reject", "latency_median")


class WorkflowRatio:
    """Extended expected harm of a window's reviewer and workflow state relative to the baseline state, on a fixed population batch.

    Harm = the reviewer-layer RHE scaled by the share of unintercepted errors that downstream steps also miss, plus two workflow terms for
    true alerts: an escalation that never arrives, and (on the time-critical share) a delivery later than the service limit. Alert duplication
    has no direct harm (design SC15). The terms are assumptions.
    """

    def __init__(self, cfg: Phase2Config, population: Batch) -> None:
        self._cfg, self._batch = cfg, population
        w = np.asarray(cfg.harm.weights)[population.severity - 1] * cfg.harm.fn_factor   # a true alert that never arrives is a missed deterioration
        alert = population.decision == 1
        self._true_alert = w * (alert & (population.y == 1))
        self._true_required = w * (alert & (population.y == 1) & (population.severity >= cfg.harm.severity_cutoff))
        self._cache: Dict[Tuple[float, ...], float] = {}
        base = {**{k: getattr(cfg.reviewer, k) for k in REVIEWER_KEYS}, "p_deliver": cfg.workflow.p_deliver, "tat_mult": 1.0, "alert_dup": 1.0,
                "verify": cfg.workflow.verify}
        self._baseline = self._total(base)

    def _total(self, params: Dict[str, float]) -> float:
        cfg, wf = self._cfg, self._cfg.workflow
        reviewer = replace(cfg.reviewer, **{k: float(params[k]) for k in REVIEWER_KEYS})
        p_late = 1.0 - float(norm.cdf(np.log(wf.tat_limit / (wf.tat_median * params["tat_mult"])) / wf.tat_sigma))
        model_term = expected_rhe(self._batch, replace(cfg, reviewer=reviewer)) * (1.0 - params["verify"])
        escalation = 1000.0 * float(np.mean(self._true_required)) * (1.0 - params["p_deliver"])
        delay = 1000.0 * float(np.mean(self._true_alert)) * wf.time_critical_share * p_late
        return model_term + escalation + delay

    def __call__(self, params: Dict[str, float]) -> float:
        key = tuple(round(float(params[k]), 6) for k in sorted(params))
        if key not in self._cache:
            self._cache[key] = self._total(params) / self._baseline
        return self._cache[key]
