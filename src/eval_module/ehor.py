"""EHOR and IIR: ground truth, the design-weighted audit estimator, and the naive comparator (design section 10)."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from src.sim_module.generator import Batch
from src.sim_module.review import ReviewOutcome

STRATUM_MULTIPLIERS = (3.0, 1.0, 0.5)   # relative audit intensity for low, middle, and high displayed confidence


@dataclass(frozen=True)
class EhorTruth:
    """Known-truth EHOR with its decomposition Cov_R x EHOR_c."""

    ehor: float
    coverage: float
    conditional: float


def _ratio(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def ehor_truth(out: ReviewOutcome) -> EhorTruth:
    """EHOR over all cases, with Cov_R = P(reviewed | R) and EHOR_c = P(intercepted | R, reviewed)."""
    r_reviewed = out.R & out.reviewed
    return EhorTruth(ehor=_ratio(out.I.sum(), out.R.sum()), coverage=_ratio(r_reviewed.sum(), out.R.sum()),
                     conditional=_ratio(out.I.sum(), r_reviewed.sum()))


def ehor_ht(out: ReviewOutcome, w: np.ndarray) -> float:
    """Horvitz-Thompson EHOR: sum(w I) / sum(w R). Unaudited cases carry weight zero."""
    return _ratio(float((w * out.I).sum()), float((w * out.R).sum()))


def iir_ht(out: ReviewOutcome, w: np.ndarray) -> float:
    """Inappropriate-intervention rate: sum(w Z) / sum(w V), V = a correct output that was reviewed."""
    return _ratio(float((w * out.Z).sum()), float((w * out.correct_reviewed).sum()))


def naive_ehor(out: ReviewOutcome, discovery: float) -> float:
    """Comparator a health system computes without an audit.

    The denominator is the errors that reviewers flagged plus the missed errors that later surface through
    outcome reports, each with probability `discovery`. With discovery below 1, the missed errors are undercounted
    and the estimate is inflated.
    """
    caught = float(out.I.sum())
    missed = float((out.R & ~out.I).sum())
    return _ratio(caught, caught + discovery * missed)


def audit_sample(out: ReviewOutcome, batch: Batch, fraction: float, stratified: bool, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Poisson audit sample with expected size fraction x N. Returns the sampled mask and the weights 1/pi (zero if unsampled).

    Stratification is by displayed confidence, using only what the audit designer can see (never the outcome or the
    reviewer's action), and oversamples low-confidence outputs, which are more often errors.
    """
    n = batch.n
    if stratified:
        confidence = np.where(batch.decision == 1, batch.reported_p, 1.0 - batch.reported_p)
        cuts = np.quantile(confidence, [1 / 3, 2 / 3])
        mult = np.asarray(STRATUM_MULTIPLIERS)[np.digitize(confidence, cuts)]
    else:
        mult = np.ones(n)
    pi = np.minimum(1.0, fraction * n * mult / mult.sum())
    sampled = rng.random(n) < pi
    return sampled, np.where(sampled, 1.0 / pi, 0.0)


def ehor_ht_se(out: ReviewOutcome, w: np.ndarray) -> float:
    """Standard error of the Horvitz-Thompson EHOR under Poisson sampling, by linearization.

    With theta = A/B and z_i = I_i - theta R_i, Var(theta) is about sum over sampled cases of w_i (w_i - 1) z_i^2 / B^2,
    where B = sum(w R). It is zero when every case is audited (w = 1).
    """
    denom = float((w * out.R).sum())
    if denom <= 0:
        return 0.0
    theta = float((w * out.I).sum()) / denom
    z = out.I.astype(float) - theta * out.R
    return float(np.sqrt((w * (w - 1.0) * z ** 2).sum()) / denom)
