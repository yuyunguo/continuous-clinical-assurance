"""RH3 mixture streams: random reviewer failures, harm-defined window labels, and the AUROC gain (design section 9.1)."""

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.eval_module.metrics import auroc
from src.sim_module.config import Phase2Config, ReviewerConfig
from src.sim_module.generator import Batch, SyntheticWorld
from src.sim_module.model import DeployedModel
from src.sim_module.harm import expected_rhe
from src.sim_module.reviewer_schedule import PARAM_OF, reviewer_schedule

from .review_layer import build_review_params, review_scores_multi
from .streams import build_stream


@dataclass(frozen=True)
class Event:
    """One reviewer failure in a mixture stream: abrupt, starting at monitored window `start`."""

    kind: str
    level: float
    start: int


def mixture_schedule(rng: np.random.Generator, n_baseline: int, n_monitored: int, base: ReviewerConfig, levels: Dict[str, Sequence[float]],
                     start_low: int, start_high: int, weights: Optional[Sequence[float]] = None) -> Tuple[Dict[str, np.ndarray], List[Event]]:
    """Per-window reviewer parameters with one or two failures of distinct kinds at random levels and start windows.

    `weights` sets how often each kind is drawn (in the order of `levels`); None draws them equally.
    """
    w = n_baseline + n_monitored
    par = {"coverage": np.full(w, base.coverage), "automation_bias": np.full(w, base.automation_bias),
           "false_reject": np.full(w, base.false_reject), "latency_median": np.full(w, base.latency_median)}
    kinds = list(levels)
    prob = None if weights is None else np.asarray(weights, dtype=float) / float(np.sum(weights))
    size = int(rng.integers(1, 3))
    if prob is not None:
        size = min(size, int(np.count_nonzero(prob)))   # a kind with zero weight is never drawn, so it cannot fill a second slot
    picked = rng.choice(len(kinds), size=size, replace=False, p=prob)
    events = []
    for k in picked:
        kind = kinds[int(k)]
        event = Event(kind=kind, level=float(levels[kind][int(rng.integers(0, len(levels[kind])))]), start=int(rng.integers(start_low, start_high + 1)))
        value = reviewer_schedule(kind, event.level, np.array([1.0]), base)[PARAM_OF[kind]][0]
        par[PARAM_OF[kind]][n_baseline + event.start:] = value
        events.append(event)
    return par, events


class WindowRatio:
    """Expected RHE of a window's reviewer state relative to the baseline reviewer, on a fixed population batch."""

    def __init__(self, cfg: Phase2Config, population: Batch) -> None:
        self._cfg, self._batch = cfg, population
        self._baseline = expected_rhe(population, cfg)
        self._cache: Dict[Tuple[float, ...], float] = {}

    def __call__(self, params: Dict[str, float]) -> float:
        key = tuple(round(float(params[k]), 6) for k in sorted(params))
        if key not in self._cache:
            reviewer = replace(self._cfg.reviewer, **{k: float(v) for k, v in params.items()})
            self._cache[key] = expected_rhe(self._batch, replace(self._cfg, reviewer=reviewer)) / self._baseline
        return self._cache[key]


@dataclass(frozen=True)
class GainResult:
    """AUROC of the base and the extended feature set on evaluation streams, the gain, and its stream-cluster bootstrap interval."""

    auroc_base: float
    auroc_full: float
    gain: float
    lo: float
    hi: float
    p: float          # one-sided: share of bootstrap gains at or below zero


def _scores(y_fit: np.ndarray, x_fit: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    """Logistic regression fit on flattened fit windows, scored on evaluation windows. Shapes (S, W, K) in, (S, W) out."""
    k = x_fit.shape[2]
    model = LogisticRegression(max_iter=1000).fit(x_fit.reshape(-1, k), y_fit.reshape(-1))
    return model.decision_function(x_eval.reshape(-1, k)).reshape(x_eval.shape[:2])


def auroc_gain(y_fit: np.ndarray, base_fit: np.ndarray, full_fit: np.ndarray, y_eval: np.ndarray, base_eval: np.ndarray, full_eval: np.ndarray,
               rng: np.random.Generator, n_boot: int = 500) -> GainResult:
    """Gain in AUROC of the full feature set over the base set, with a paired bootstrap that resamples whole streams."""
    s_base, s_full = _scores(y_fit, base_fit, base_eval), _scores(y_fit, full_fit, full_eval)
    n = y_eval.shape[0]
    gains = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y_eval[idx].reshape(-1)
        gains[b] = auroc(yy, s_full[idx].reshape(-1)) - auroc(yy, s_base[idx].reshape(-1))
    a_base, a_full = auroc(y_eval.reshape(-1), s_base.reshape(-1)), auroc(y_eval.reshape(-1), s_full.reshape(-1))
    return GainResult(a_base, a_full, a_full - a_base, float(np.nanpercentile(gains, 2.5)), float(np.nanpercentile(gains, 97.5)), float(np.nanmean(gains <= 0)))


def mixture_features(n: int, seed: int, cfg: Phase2Config, world: SyntheticWorld, model: DeployedModel, levels: Dict[str, Sequence[float]],
                     ratio: WindowRatio, lags: Sequence[int], audit_fraction: float,
                     weights: Optional[Sequence[float]] = None) -> Tuple[Dict[int, np.ndarray], np.ndarray, List[int]]:
    """Oversight scores per lag (S, T, 6), harm-defined window labels (S, T), and the number of failures in each stream."""
    b, t = cfg.stream.baseline_windows, cfg.stream.monitored_windows
    feats: Dict[int, List[np.ndarray]] = {lag: [] for lag in lags}
    labels, counts = [], []
    for i in range(n):
        rng = np.random.default_rng(seed * 100_003 + i)
        par, events = mixture_schedule(rng, b, t, cfg.reviewer, levels, cfg.stream.s0_low, cfg.stream.s0_high_abrupt, weights)
        stream = build_stream(world, model, None, "abrupt", 0, cfg, rng)
        z = review_scores_multi(stream, build_review_params(stream, cfg, par, rng), audit_fraction, lags, rng)
        for lag in lags:
            feats[lag].append(z[lag])
        labels.append([ratio({k: v[b + w] for k, v in par.items()}) >= 1 + cfg.harm.kappa for w in range(t)])
        counts.append(len(events))
    return {lag: np.stack(v) for lag, v in feats.items()}, np.array(labels, dtype=int), counts
