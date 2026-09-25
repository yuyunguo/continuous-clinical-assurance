"""Block bootstrap over real windows: null streams for false-alarm calibration on a single real dataset.

Real windows are not exchangeable case by case (calendar order carries real serial structure), and there is only
one real dataset, so Tier 1's approach — draw as many independent null streams as needed from an infinite generator
— has no real-data analogue. A block bootstrap resamples contiguous runs of real windows with replacement, which
preserves real within-block structure while still producing many replicate streams from the one dataset
(`Tier2-Readiness.md` §3).
"""

import numpy as np
import pandas as pd

from src.monitor_module.streams import StreamData
from src.sim_module.config import HarmConfig

from .config import Tier2StreamConfig
from .stream import build_stream_data


def block_bootstrap_windows(windowed: pd.DataFrame, cfg: Tier2StreamConfig, n_windows_out: int, block_size: int,
                            rng: np.random.Generator) -> pd.DataFrame:
    """Resample `n_windows_out` windows from `windowed`, in contiguous blocks of `block_size` real windows.

    Each block's starting real window is drawn independently, with replacement, uniformly over the real windows
    that leave room for a full block; the `block_size` real windows from there are kept in their original order.
    The last block is truncated if `n_windows_out` is not a multiple of `block_size`.
    """
    n_real = int(windowed["window"].nunique())
    if not (1 <= block_size <= n_real):
        raise ValueError(f"block_size ({block_size}) must be between 1 and the number of real windows ({n_real})")
    n_blocks = -(-n_windows_out // block_size)   # ceil division
    starts = rng.integers(0, n_real - block_size + 1, size=n_blocks)
    parts = []
    out_window = 0
    for start in starts:
        for offset in range(block_size):
            if out_window >= n_windows_out:
                break
            block = windowed[windowed["window"] == start + offset].copy()
            block["window"] = out_window
            parts.append(block)
            out_window += 1
        if out_window >= n_windows_out:
            break
    return pd.concat(parts, ignore_index=True)


def block_bootstrap_stream(windowed: pd.DataFrame, cfg: Tier2StreamConfig, harm: HarmConfig, n_windows_out: int,
                           block_size: int, rng: np.random.Generator, n_baseline: int = 0) -> StreamData:
    """A null `StreamData` resampled from real windows, for calibration or for a mean-shift statistic reference.

    `n_baseline` windows (0 by default) serve as the standardization baseline (e.g. `window_statistics`'s reference
    mean/SD); the rest are "monitored" with no injected perturbation, for a matched false-alarm evaluation.
    """
    resampled = block_bootstrap_windows(windowed, cfg, n_windows_out, block_size, rng)
    return build_stream_data(resampled, cfg, harm, n_baseline=n_baseline, rng=rng)
