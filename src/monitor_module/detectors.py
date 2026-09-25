"""Standardization, label lag, CUSUM detection, alert episodes, and matched false-alarm calibration."""

from typing import Dict, List, Optional, Sequence

import numpy as np

from src.sim_module.config import Phase2Config
from src.sim_module.generator import SyntheticWorld
from src.sim_module.model import DeployedModel
from src.sim_module.perturbations import Perturbation  # noqa: F401  (type context)

from .statistics import DIRECTION, INDICATOR_COMPONENT, INDICATOR_ORDER, LABEL_DEPENDENT, window_statistics
from .streams import StreamData, build_stream

K_REF = 0.5
SD_FLOOR = 1e-6


def standardize(series: np.ndarray, baseline: int, direction: np.ndarray, full: bool = False) -> np.ndarray:
    """Standardize by the baseline windows and orient so that larger is worse. Shape (S, W, K)."""
    base = series[:, :baseline, :]
    mean = np.nanmean(base, axis=1, keepdims=True)
    sd = np.maximum(np.nanstd(base, axis=1, ddof=1, keepdims=True), SD_FLOOR)
    z = np.nan_to_num(direction[None, None, :] * (series - mean) / sd)
    return z if full else z[:, baseline:, :]


def apply_label_lag(z_full: np.ndarray, baseline: int, lag: int) -> np.ndarray:
    """Monitored-window series in which each value arrives `lag` windows after its window."""
    start = baseline - lag
    return z_full[:, start:z_full.shape[1] - lag, :]


def cusum_alarms(z: np.ndarray, h: float, k_ref: float = K_REF) -> np.ndarray:
    """One-sided CUSUM with restart after an alarm, run for all streams and indicators at once."""
    state = np.zeros((z.shape[0], z.shape[2]))
    alarms = np.zeros(z.shape, dtype=bool)
    for t in range(z.shape[1]):
        state = np.maximum(0.0, state + z[:, t, :] - k_ref)
        hit = state > h
        alarms[:, t, :] = hit
        state = np.where(hit, 0.0, state)
    return alarms


def system_events(z: np.ndarray, indicators: Sequence[int], h: float, k_ref: float = K_REF) -> np.ndarray:
    """A system event occurs in a window when any selected indicator's detector alarms. Shape (S, T)."""
    return cusum_alarms(z[:, :, list(indicators)], h, k_ref).any(axis=2)


def episode_starts(events: np.ndarray, j: int) -> np.ndarray:
    """An alert episode starts at an event with no event in the previous j windows (hysteresis)."""
    previous = np.zeros_like(events)
    for lag in range(1, j + 1):
        previous[:, lag:] |= events[:, :-lag]
    return events & ~previous


def stream_scores_multi(stream: StreamData, cfg: Phase2Config, lags: Sequence[int]) -> Dict[int, np.ndarray]:
    """Standardized scores for several label lags from one pass over a stream. Each value has shape (T, K)."""
    stats = window_statistics(stream, cfg)
    series = np.stack([stats[i] for i in INDICATOR_ORDER], axis=1)[None]
    direction = np.array([DIRECTION[i] for i in INDICATOR_ORDER], dtype=float)
    z_full = standardize(series, stream.n_baseline, direction, full=True)
    dependent = np.array([i in LABEL_DEPENDENT for i in INDICATOR_ORDER])
    free = apply_label_lag(z_full, stream.n_baseline, 0)
    return {lag: np.where(dependent[None, None, :], apply_label_lag(z_full, stream.n_baseline, lag), free)[0] for lag in lags}


def stream_scores(stream: StreamData, cfg: Phase2Config, lag: Optional[int] = None) -> np.ndarray:
    """Standardized, label-lagged scores for one stream. Shape (T, K) in `INDICATOR_ORDER`."""
    use = cfg.stream.label_lag if lag is None else lag
    return stream_scores_multi(stream, cfg, (use,))[use]


def null_scores_multi(world: SyntheticWorld, model: DeployedModel, cfg: Phase2Config, streams: int, seed: int,
                      lags: Sequence[int]) -> Dict[int, np.ndarray]:
    """Scores of `streams` independent null streams for each lag, from one pass. Each value has shape (S, T, K)."""
    per = [stream_scores_multi(build_stream(world, model, None, "abrupt", 0, cfg, np.random.default_rng(seed * 100_003 + i)), cfg, lags)
           for i in range(streams)]
    return {lag: np.stack([p[lag] for p in per]) for lag in lags}


def null_scores(world: SyntheticWorld, model: DeployedModel, cfg: Phase2Config, streams: int, seed: int) -> np.ndarray:
    """Scores for `streams` independent null streams at the configured lag. Shape (S, T, K)."""
    return null_scores_multi(world, model, cfg, streams, seed, (cfg.stream.label_lag,))[cfg.stream.label_lag]


def alert_rate(z: np.ndarray, indicators: Sequence[int], h: float, j: int) -> float:
    """Alert-episode starts per 1,000 monitored windows."""
    starts = episode_starts(system_events(z, indicators, h), j)
    return float(starts.sum() / starts.size * 1000.0)


def calibrate_threshold(z: np.ndarray, indicators: Sequence[int], target_per_1000: float, j: int) -> float:
    """Smallest common CUSUM threshold h that gives at most the target alert-episode rate on null streams."""
    lo, hi = 0.2, 200.0
    for _ in range(40):
        mid = float(np.sqrt(lo * hi))
        if alert_rate(z, indicators, mid, j) > target_per_1000:
            lo = mid
        else:
            hi = mid
    return hi


def indicator_indices(names: Sequence[str]) -> List[int]:
    """Column indices in `INDICATOR_ORDER` for indicator names."""
    return [INDICATOR_ORDER.index(n) for n in names]


def component_groups(names: Sequence[str]) -> Dict[str, List[int]]:
    """Column indices grouped by the instrument's state component (C, P, U, ...)."""
    groups: Dict[str, List[int]] = {}
    for column, name in enumerate(names):
        groups.setdefault(INDICATOR_COMPONENT[name], []).append(column)
    return groups


def system_events_hierarchical(z: np.ndarray, groups: Dict[str, List[int]], thresholds: Dict[str, float],
                               k_ref: float = K_REF) -> np.ndarray:
    """A system event occurs when any component's own detector alarms, each with its own threshold. Shape (S, T)."""
    events = np.zeros(z.shape[:2], dtype=bool)
    for component, columns in groups.items():
        events |= system_events(z, columns, thresholds[component], k_ref)
    return events


def calibrate_hierarchical(z: np.ndarray, groups: Dict[str, List[int]], target_per_1000: float, j: int,
                           shares: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Component thresholds under a component-level budget split (decision D13).

    Each component gets a share of the false-alarm budget (equal by default) and is calibrated on its own. One common
    multiplier on the shares is then searched so that the union of components meets the system budget.
    """
    weight = shares or {c: 1.0 / len(groups) for c in groups}

    def thresholds_for(m: float) -> Dict[str, float]:
        return {c: calibrate_threshold(z, cols, m * target_per_1000 * weight[c], j) for c, cols in groups.items()}

    def system_rate(th: Dict[str, float]) -> float:
        starts = episode_starts(system_events_hierarchical(z, groups, th), j)
        return float(starts.sum() / starts.size * 1000.0)

    lo, hi = 0.3, 3.0
    best = thresholds_for(lo)
    for _ in range(12):
        mid = float(np.sqrt(lo * hi))
        candidate = thresholds_for(mid)
        if system_rate(candidate) <= target_per_1000:
            lo, best = mid, candidate
        else:
            hi = mid
    return best
