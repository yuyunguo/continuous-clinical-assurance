"""TAFD classification and the paired restricted-mean comparison (design sections 4.2 and 9)."""

from typing import Dict, Tuple, Union

import numpy as np


def _starts(events: np.ndarray, j: int) -> np.ndarray:
    """Alert-episode starts in one series."""
    previous = np.zeros_like(events)
    for lag in range(1, j + 1):
        previous[lag:] |= events[:-lag]
    return events & ~previous


def classify_replicate(events: np.ndarray, s0: int, t0: int, horizon: int, j: int) -> Dict[str, Union[int, bool]]:
    """Classify one replicate.

    Episode starts before s0 are false alarms. The first event at or after s0 is the detection. It is anticipatory
    (TAFD 0) if it precedes onset t0, and otherwise TAFD = event - t0. A detection later than the horizon, or none,
    is censored at the horizon.
    """
    starts = _starts(events, j)
    false_alarms = int(starts[:s0].sum())
    carry_in = bool(events[max(0, s0 - j):s0].any())
    after = np.flatnonzero(events[s0:]) + s0
    first = int(after[0]) if len(after) else None
    if first is None or first >= t0 + horizon:
        return {"tafd": horizon, "censored": True, "false_alarms": false_alarms, "anticipatory": False, "carry_in": carry_in, "lead_time": 0}
    return {"tafd": max(0, first - t0), "censored": False, "false_alarms": false_alarms, "anticipatory": first < t0, "carry_in": carry_in,
            "lead_time": max(0, t0 - first)}


def relative_reduction(reference: np.ndarray, candidate: np.ndarray) -> float:
    """rho = (RMST_reference - RMST_candidate) / RMST_reference for per-replicate TAFD capped at the horizon."""
    return float((reference.mean() - candidate.mean()) / reference.mean())


def bootstrap_rho(reference: np.ndarray, candidate: np.ndarray, rng: np.random.Generator, n_boot: int = 2000) -> Tuple[float, float]:
    """95 percent percentile interval for rho, resampling replicates so that both systems stay paired."""
    n = len(reference)
    draws = rng.integers(0, n, size=(n_boot, n))
    ref_mean, cand_mean = reference[draws].mean(axis=1), candidate[draws].mean(axis=1)
    rho = (ref_mean - cand_mean) / ref_mean
    return float(np.percentile(rho, 2.5)), float(np.percentile(rho, 97.5))


def rmst_difference(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Reduction in mean TAFD, in windows: positive when the candidate detects earlier (decision D19)."""
    return float(reference.mean() - candidate.mean())


def bootstrap_difference(reference: np.ndarray, candidate: np.ndarray, rng: np.random.Generator, n_boot: int = 2000) -> Tuple[float, float]:
    """95 percent percentile interval for the reduction in windows, resampling replicates so that both systems stay paired."""
    draws = rng.integers(0, len(reference), size=(n_boot, len(reference)))
    diff = reference[draws].mean(axis=1) - candidate[draws].mean(axis=1)
    return float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))



def exposure_before_detection(excess: np.ndarray, t0: int, tafd: int) -> float:
    """Excess expected harm accrued in the `tafd` windows from onset t0 until detection (capped at the stream end).

    `excess` is the per-window excess expected RHE over baseline. Undetected replicates carry TAFD equal to the horizon.
    """
    return float(excess[t0:t0 + int(tafd)].sum())


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, n_boot: int = 2000) -> Tuple[float, float]:
    """95 percent percentile interval for a mean over replicates."""
    means = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
