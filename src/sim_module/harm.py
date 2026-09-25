"""Residual harm exposure (design section 4.3): expected, from the simulation's hidden truth."""

from typing import Dict

import numpy as np

from .config import Phase2Config
from .generator import Batch
from .reviewer import interception_probability


def error_weights(severity: np.ndarray, y: np.ndarray, decision: np.ndarray, cfg: Phase2Config) -> np.ndarray:
    """Harm weight of each case if its decision is an error: the class weight, times `fn_factor` for a missed deterioration (y = 1, decision = 0)."""
    factor = np.where((y == 1) & (decision == 0), cfg.harm.fn_factor, 1.0)
    return np.asarray(cfg.harm.weights)[severity - 1] * factor


def expected_rhe(batch: Batch, cfg: Phase2Config) -> float:
    """Expected residual harm exposure per 1,000 cases.

    RHE = 1000 * mean( w(sev) * E * (1 - P(intercepted)) ), where E marks any AI error, false positive or false
    negative, and w is the severity weight. The severity cutoff is not used here: it defines which errors "require
    human detection" for EHOR. Delay-attributable harm needs the workflow layer and is not included here.
    """
    weights = error_weights(batch.severity, batch.y, batch.decision, cfg)
    error = batch.decision != batch.y
    missed = 1.0 - interception_probability(batch, cfg.reviewer)
    return float(1000.0 * np.mean(weights * error * missed))


# Cases per evaluation of expected harm. The elicited weights (a class-3 missed deterioration weighs 150) make expected harm heavy-tailed, and the onset ratio
# takes the larger of the pooled and subgroup ratios, which biases a noisy neutral reading upward. At 200,000 cases that bias was 7 percent, at 1,000,000 about 1.
HARM_POPULATION = 1_000_000

SUBGROUP = 1   # the predefined subgroup (decision D18). The VOE fixes a short list of such subgroups.


def rhe_baselines(baseline_batch: Batch, cfg: Phase2Config) -> Dict[str, float]:
    """Baseline expected RHE for the pooled cohort and for the predefined subgroup."""
    return {"pooled": expected_rhe(baseline_batch, cfg),
            "subgroup": expected_rhe(baseline_batch.select(baseline_batch.g == SUBGROUP), cfg)}


def onset_ratio(batch: Batch, cfg: Phase2Config, baselines: Dict[str, float]) -> float:
    """The ratio that defines onset (decision D18): the larger of the pooled ratio and the predefined subgroup's ratio.

    Each ratio is relative to its own baseline. The subgroup counts only if it is at least `min_subgroup_share` of the
    cases, so that onset does not depend on a noisy small stratum.
    """
    ratio = expected_rhe(batch, cfg) / baselines["pooled"]
    in_group = batch.g == SUBGROUP
    if in_group.mean() >= cfg.harm.min_subgroup_share and in_group.any():
        ratio = max(ratio, expected_rhe(batch.select(in_group), cfg) / baselines["subgroup"])
    return float(ratio)

