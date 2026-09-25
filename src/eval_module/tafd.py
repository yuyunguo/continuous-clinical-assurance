"""TAFD classification and the paired restricted-mean comparison (design sections 4.2 and 9)."""

from typing import Dict, List, Sequence, Tuple, Union

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


def bootstrap_pvalue(reference: np.ndarray, candidate: np.ndarray, rng: np.random.Generator, n_boot: int = 2000) -> float:
    """Two-sided bootstrap p-value against a null of no difference in mean TAFD, from the same paired resampling
    scheme `bootstrap_difference` uses (a fresh `rng` draw, not the same draws as any prior `bootstrap_difference`
    call -- statistically equivalent at this `n_boot`, not bit-identical to an already-reported CI).

    p = 2 * min(q, 1 - q), where q is the share of bootstrap draws with reference - candidate <= 0: the standard
    two-sided percentile-bootstrap p-value (Manuscript Methods Sec. 2.6, the Holm-corrected significance test).
    """
    draws = rng.integers(0, len(reference), size=(n_boot, len(reference)))
    diff = reference[draws].mean(axis=1) - candidate[draws].mean(axis=1)
    q = float((diff <= 0).mean())
    return float(min(2.0 * min(q, 1.0 - q), 1.0))


def arm_reductions_with_holm_correction(tafd: Dict[str, np.ndarray], arm_order: Sequence[str], reference_name: str,
                                        ci_rng: np.random.Generator, p_rng: np.random.Generator,
                                        alpha: float = 0.05) -> Dict[str, Dict[str, float]]:
    """For every arm in `arm_order` except `reference_name`: the RMST reduction vs. the reference arm, its 95%
    percentile-bootstrap CI, a two-sided bootstrap p-value, and whether it survives a Holm-Bonferroni correction
    (manuscript Methods Sec. 2.6) across this family of comparisons.

    `ci_rng` and `p_rng` are separate generators: `ci_rng` reproduces the same CI already reported by a prior run
    using only `bootstrap_difference` with that seed (bit-identical, not just statistically equivalent); `p_rng` is
    an independent bootstrap estimate of the same underlying quantity, used only for the correction decision.
    """
    reference = tafd[reference_name]
    names = [n for n in arm_order if n != reference_name]
    reductions = [rmst_difference(reference, tafd[name]) for name in names]
    cis = [bootstrap_difference(reference, tafd[name], ci_rng) for name in names]
    p_values = [bootstrap_pvalue(reference, tafd[name], p_rng) for name in names]
    significant = holm_correction(p_values, alpha)
    return {name: {"reduction": r, "lo": lo, "hi": hi, "p": p, "significant": float(sig)}
           for name, r, (lo, hi), p, sig in zip(names, reductions, cis, p_values, significant)}


def holm_correction(p_values: Sequence[float], alpha: float = 0.05) -> List[bool]:
    """Holm-Bonferroni step-down test \\citep{holm1979}: which hypotheses are rejected (significant) at family-wise
    error rate `alpha`, given the family's `p_values`.

    Sorts p-values ascending (rank 0 = smallest) and rejects p_(rank) while p_(rank) <= alpha / (m - rank), stopping
    at the first failure to reject -- every later (larger) p-value in the sorted order is also not rejected, the
    step-down property that makes this less conservative than a flat Bonferroni correction while still controlling
    the family-wise error rate.
    """
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    reject = [False] * m
    for rank, i in enumerate(order):
        if p_values[i] <= alpha / (m - rank):
            reject[i] = True
        else:
            break
    return reject
