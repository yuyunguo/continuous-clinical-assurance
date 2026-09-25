"""Calibrated real-data onset detection (D18): a CUSUM-accumulated, calibrated-threshold replacement for the naive
single-window ratio-threshold rule in `harm.py`.

**Why:** investigation on real MIMIC-IV sepsis data found the naive rule false-fires on essentially every
pure-null replicate (100/100 in a 100-replicate check). Scanning ~24 correlated single-window ratio tests against
a fixed 1 + kappa threshold is a multiple-comparisons problem, and a single 500-case window's sampling noise (SD
roughly 0.24-0.38 on the ratio, for both the pooled and the subgroup ratio) is too close to the elicited kappa
(0.25) threshold itself to support a naive point-in-time test.

**The fix** reuses the monitoring system's own CUSUM + `calibrate_threshold` machinery — already built and tested
for the seven arms — applied to the D18 harm-ratio series (pooled and subgroup columns) instead of a naive point
threshold. This is not a new statistical framework: it is the same standardize -> CUSUM -> calibrate-against-null
pipeline `src.monitor_module` already applies to the 13 monitoring indicators, applied to a 2-column "indicator"
that IS the D18 ratio series, so the calibrated onset threshold controls a genuine false-onset rate by
construction, the same way the monitoring arms' calibrated thresholds control their false-alarm rate.
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.monitor_module.detectors import calibrate_threshold, cusum_alarms, episode_starts, standardize
from src.monitor_module.streams import StreamData
from src.sim_module.config import HarmConfig, Phase2Config
from src.sim_module.harm import SUBGROUP, expected_rhe

from .config import Tier2StreamConfig
from .harm import batch_from_windows
from .resample import block_bootstrap_windows
from .stream import build_stream_data

ONSET_DIRECTION = np.array([1.0, 1.0])   # a rise in either the pooled or the subgroup ratio is what D18 watches for


def _onset_raw_series(stream: StreamData, phase2_cfg: Phase2Config) -> np.ndarray:
    """Raw (pooled, subgroup) expected-RHE per window, shape (W, 2). Subgroup is NaN where its share of the window
    is below `min_subgroup_share` (D18's rule for when the subgroup counts at all)."""
    w = stream.y.shape[0]
    pooled = np.empty(w)
    subgroup = np.full(w, np.nan)
    for t in range(w):
        batch = batch_from_windows(stream, t, t + 1)
        pooled[t] = expected_rhe(batch, phase2_cfg)
        in_group = batch.g == SUBGROUP
        if in_group.mean() >= phase2_cfg.harm.min_subgroup_share and in_group.any():
            subgroup[t] = expected_rhe(batch.select(in_group), phase2_cfg)
    return np.stack([pooled, subgroup], axis=1)


def tier2_onset_series(stream: StreamData, phase2_cfg: Phase2Config) -> np.ndarray:
    """Standardized (pooled, subgroup) onset series for the monitored windows of one stream, shape (T, 2).

    Standardized against this stream's own baseline windows, exactly like `src.monitor_module.window_statistics`'s
    13 indicators are for the monitoring arms.
    """
    raw = _onset_raw_series(stream, phase2_cfg)[None]   # (1, W, 2)
    return standardize(raw, stream.n_baseline, ONSET_DIRECTION, full=False)[0]


def tier2_onset_null_series_multi(windowed: pd.DataFrame, cfg: Tier2StreamConfig, phase2_cfg: Phase2Config,
                                  harm: HarmConfig, streams: int, seed: int, n_windows_out: int, block_size: int,
                                  n_baseline: int) -> np.ndarray:
    """Standardized (pooled, subgroup) onset series of `streams` independent block-bootstrap null streams, shape
    (S, T, 2). Mirrors `src.tier2_module.calibrate.tier2_null_scores_multi`'s seeding and output shape."""
    series = []
    for i in range(streams):
        resampled = block_bootstrap_windows(windowed, cfg, n_windows_out, block_size, np.random.default_rng(seed * 100_003 + i))
        stream = build_stream_data(resampled, cfg, harm, n_baseline=n_baseline, rng=np.random.default_rng(seed * 100_003 + i))
        series.append(tier2_onset_series(stream, phase2_cfg))
    return np.stack(series)


def tier2_calibrate_onset(calib: np.ndarray, target_per_1000: float, j: int) -> float:
    """The CUSUM threshold that gives at most `target_per_1000` false onsets per 1,000 monitored windows on
    `calib` (e.g. from `tier2_onset_null_series_multi`). A thin, named wrapper: this is exactly
    `calibrate_threshold` over the two onset columns, not a separate calibration procedure."""
    return calibrate_threshold(calib, [0, 1], target_per_1000, j)


def tier2_onset_index_calibrated(stream: StreamData, phase2_cfg: Phase2Config, threshold: float, j: int) -> Optional[int]:
    """First monitored window at which the calibrated CUSUM rule fires on either the pooled or the subgroup onset
    series (the D18 "max of pooled and subgroup" rule, now CUSUM-accumulated and calibrated instead of a naive
    single-window threshold test)."""
    z = tier2_onset_series(stream, phase2_cfg)
    events = cusum_alarms(z[None], threshold).any(axis=2)
    starts = episode_starts(events, j)[0]
    nonzero = np.flatnonzero(starts)
    return int(nonzero[0]) if len(nonzero) else None
