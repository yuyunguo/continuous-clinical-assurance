"""Assigns cases to fixed-size, calendar-ordered windows (decision D6/D8: 500 cases per window)."""

import pandas as pd

from .config import Tier2StreamConfig


def assign_windows(cohort: pd.DataFrame, cfg: Tier2StreamConfig) -> pd.DataFrame:
    """Sort `cohort` by `cfg.order_column`, drop the remainder that does not fill a whole window, and add a `window` column.

    Real cohorts rarely divide evenly by `cfg.window_size`; the remainder is dropped rather than padded, so every
    window is a real, equally sized block of consecutive cases. Raises if there are fewer rows than one window.
    """
    ordered = cohort.sort_values(cfg.order_column, kind="stable").reset_index(drop=True)
    n_windows = len(ordered) // cfg.window_size
    if n_windows < 1:
        raise ValueError(f"cohort has {len(ordered)} rows, fewer than one window of {cfg.window_size}; need at least one window")
    kept = ordered.iloc[: n_windows * cfg.window_size].copy()
    kept["window"] = kept.index // cfg.window_size
    return kept
