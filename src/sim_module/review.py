"""Per-case review outcomes sampled from the reviewer model (design sections 10 and 11)."""

from dataclasses import dataclass

import numpy as np

from .config import Phase2Config
from .generator import Batch
from .reviewer import detection_probability


@dataclass(frozen=True)
class ReviewOutcome:
    """Boolean arrays over the cases of a batch.

    R requires human detection (an AI error at or above the severity cutoff), reviewed means examined before the
    point of no return, I is an intercepted R, Z is a correct output that was reviewed and wrongly rejected, and
    correct_reviewed is V, the denominator of the inappropriate-intervention rate. `error` is any AI error.
    """

    R: np.ndarray
    reviewed: np.ndarray
    I: np.ndarray  # noqa: E741  (notation of the instrument specification)
    Z: np.ndarray
    correct_reviewed: np.ndarray
    error: np.ndarray
    intercepted_any: np.ndarray   # any AI error caught in time, whatever its severity (drives RHE)


def sample_review(batch: Batch, cfg: Phase2Config, rng: np.random.Generator) -> ReviewOutcome:
    """Sample who is reviewed in time, who detects an error, and who wrongly rejects a correct output."""
    error = batch.decision != batch.y
    requires = error & (batch.severity >= cfg.harm.severity_cutoff)
    reviewed = rng.random(batch.n) < cfg.reviewer.coverage
    caught = reviewed & (rng.random(batch.n) < detection_probability(batch, cfg.reviewer))
    wrong = reviewed & ~error & (rng.random(batch.n) < cfg.reviewer.false_reject)
    if np.isfinite(cfg.reviewer.deadline):
        # Drawn last, and only with a finite deadline, so the default draws are unchanged. A late review is not "in time".
        latency = cfg.reviewer.latency_median * np.exp(cfg.reviewer.latency_sigma * rng.standard_normal(batch.n))
        on_time = latency <= cfg.reviewer.deadline
        reviewed, caught, wrong = reviewed & on_time, caught & on_time, wrong & on_time
    return ReviewOutcome(R=requires, reviewed=reviewed, I=requires & caught, Z=wrong, correct_reviewed=reviewed & ~error, error=error,
                         intercepted_any=error & caught)
