"""Matched false-alarm calibration on Tier 2 data: the one adapter needed to reuse `src.monitor_module`'s
indicator, CUSUM, and calibration machinery (`stream_scores_multi`, `calibrate_threshold`, `calibrate_hierarchical`,
...) unchanged, since that machinery operates on a `StreamData` and a `Phase2Config` alone — the only Tier-1-specific
piece was how null streams are drawn (`null_scores_multi`, which calls the infinite synthetic generator). Here they
come from the block bootstrap instead (`resample.py`).
"""

from typing import Dict, Sequence

import numpy as np
import pandas as pd

from src.monitor_module.detectors import stream_scores_multi
from src.sim_module.config import HarmConfig, Phase2Config

from .config import Tier2StreamConfig
from .resample import block_bootstrap_stream


def tier2_null_scores_multi(windowed: pd.DataFrame, cfg: Tier2StreamConfig, phase2_cfg: Phase2Config, harm: HarmConfig,
                            streams: int, seed: int, lags: Sequence[int], n_windows_out: int, block_size: int,
                            n_baseline: int) -> Dict[int, np.ndarray]:
    """Scores of `streams` independent block-bootstrap null streams, for each lag, from one pass over each.

    Mirrors `src.monitor_module.detectors.null_scores_multi`'s output shape (S, T, K) and seeding convention, so
    `calibrate_threshold`/`calibrate_hierarchical`/`alert_rate` apply unchanged.
    """
    per = [stream_scores_multi(
        block_bootstrap_stream(windowed, cfg, harm, n_windows_out, block_size, np.random.default_rng(seed * 100_003 + i), n_baseline=n_baseline),
        phase2_cfg, lags)
        for i in range(streams)]
    return {lag: np.stack([p[lag] for p in per]) for lag in lags}
