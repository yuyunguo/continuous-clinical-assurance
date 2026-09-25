"""Multi-arm matched calibration on Tier 2 score arrays, reusing the M4-pilot arm definitions unchanged.

`scripts/m4_pilot.py`'s `FLAT_ARMS`/`SPLIT_ARMS`/`ARM_ORDER` are imported directly rather than redefined here, so
Tier 1 and Tier 2 are calibrated and compared on the exact same indicator sets — not a separately chosen set for
Tier 2, which would risk quietly favoring one tier's result over the other.
"""

from typing import Dict, Tuple, Union

import numpy as np

from scripts.m4_pilot import ARM_ORDER, FLAT_ARMS, SPLIT_ARMS
from src.monitor_module import (INDICATOR_ORDER, calibrate_hierarchical, calibrate_threshold, component_groups,
                                episode_starts, indicator_indices, system_events, system_events_hierarchical)

Thresholds = Union[float, Dict[str, float]]
ArmResult = Tuple[Thresholds, float, float]   # threshold, achieved rate per 1,000 windows, its standard error


def tier2_calibrate_arms(calib: np.ndarray, evaluation: np.ndarray, budget: float, j: int) -> Dict[str, ArmResult]:
    """Calibrate every arm in `ARM_ORDER` on `calib`, report each arm's achieved alert-episode rate on `evaluation`.

    `calib`/`evaluation` are indicator score arrays for one label lag, shape (S, T, K) in `INDICATOR_ORDER` — for
    example `tier2_null_scores_multi(...)[lag]`. Mirrors `scripts.m4_pilot.calibrate_all`'s per-arm logic, applied
    to Tier 2 null streams instead of Tier 1's.
    """
    cols = {name: indicator_indices(ids) for name, ids in FLAT_ARMS.items()}
    groups = component_groups(INDICATOR_ORDER)
    out: Dict[str, ArmResult] = {}
    for name in ARM_ORDER:
        if name in SPLIT_ARMS:
            threshold: Thresholds = calibrate_hierarchical(calib, groups, budget, j, shares=SPLIT_ARMS[name])
            events = system_events_hierarchical(evaluation, groups, threshold)  # type: ignore[arg-type]
        else:
            threshold = calibrate_threshold(calib, cols[name], budget, j)
            events = system_events(evaluation, cols[name], threshold)  # type: ignore[arg-type]
        starts = episode_starts(events, j)
        rate = float(starts.sum() / starts.size * 1000.0)
        se = float(np.sqrt(max(starts.sum(), 1)) / starts.size * 1000.0)
        out[name] = (threshold, rate, se)
    return out
