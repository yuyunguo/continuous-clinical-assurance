"""Real-data harm evaluation on a Tier 2 `StreamData`: expected RHE and onset detection (decision D18).

Reuses `src.sim_module.harm`'s `expected_rhe`/`rhe_baselines`/`onset_ratio` completely unchanged, via a `Batch`
built by flattening one or more `StreamData` windows — those functions only read `severity`, `y`, `decision`,
`reported_p`, and `g`, all of which a `StreamData` already carries.

**A real limitation, not yet resolved:** Tier 1 computes its RHE baseline from `HARM_POPULATION` = 1,000,000
synthetic draws, which is near-noiseless. A real baseline is limited to however many real cases actually fall in
the baseline windows — typically far fewer — so `tier2_rhe_baselines` and `tier2_onset_index` are noisier than
their Tier 1 counterparts. How much noisier, and whether pooling several windows per onset check would help, is
open; see `Tier2-Readiness.md`.
"""

from typing import Dict, Optional

import numpy as np

from src.monitor_module.streams import StreamData
from src.sim_module.config import Phase2Config
from src.sim_module.generator import Batch
from src.sim_module.harm import expected_rhe, onset_ratio, rhe_baselines


def batch_from_windows(stream: StreamData, window_start: int, window_stop: int) -> Batch:
    """A `Batch` flattening `StreamData` windows `[window_start, window_stop)` into one population of cases.

    `x` and `x_model` carry zero-width placeholders: `expected_rhe`/`onset_ratio` never read them.
    """
    y = stream.y[window_start:window_stop].reshape(-1)
    g = stream.g[window_start:window_stop].reshape(-1)
    severity = stream.severity[window_start:window_stop].reshape(-1)
    decision = stream.decision[window_start:window_stop].reshape(-1)
    reported_p = stream.reported_p[window_start:window_stop].reshape(-1)
    return Batch(x=np.empty((y.shape[0], 0)), y=y, g=g, severity=severity, decision=decision, reported_p=reported_p)


def tier2_rhe_baselines(stream: StreamData, phase2_cfg: Phase2Config) -> Dict[str, float]:
    """Baseline expected RHE (pooled and the predefined subgroup) from the stream's real baseline windows."""
    baseline_batch = batch_from_windows(stream, 0, stream.n_baseline)
    return rhe_baselines(baseline_batch, phase2_cfg)


def _crosses_onset(batch: Batch, phase2_cfg: Phase2Config, baselines: Dict[str, float]) -> bool:
    """Whether one window's batch counts as onset (D18) against `baselines`, handling the zero-baseline edge case.

    A real, finite baseline can show zero expected harm outright (unlike Tier 1's near-noiseless 1,000,000-case
    synthetic baseline, which essentially never does), which would otherwise divide by zero in `onset_ratio`. Any
    expected harm at all against a zero baseline is an unbounded relative increase, so it counts as onset; a zero
    baseline with zero window harm counts as no change.
    """
    if baselines["pooled"] <= 0.0:
        return expected_rhe(batch, phase2_cfg) > 0.0
    return bool(onset_ratio(batch, phase2_cfg, baselines) >= 1 + phase2_cfg.harm.kappa)


def tier2_onset_index(stream: StreamData, phase2_cfg: Phase2Config, baselines: Dict[str, float]) -> Optional[int]:
    """First monitored window (an index into the monitored segment) at which the onset ratio (D18) reaches 1 + kappa,
    against one fixed baseline (typically the stream's real baseline windows; see `tier2_rhe_baselines`).

    Scans real, already-materialized windows one at a time — unlike Tier 1's `onset_index`, which searches over a
    continuous progress value because the synthetic generator can redraw a batch at any progress. A real stream has
    no such freedom: each window's cases are whatever real cases fell in it.

    **A confound, not resolved by this function:** if the real data itself drifts over calendar time (as Sepsis-3
    prevalence does), a fixed early-period baseline will eventually look stale, and this function cannot tell that
    apart from an injected perturbation's onset — see `tier2_onset_index_rolling` for the alternative that avoids
    this, and `Tier2-Readiness.md` §7 for the trade-off between them.
    """
    n_monitored = stream.y.shape[0] - stream.n_baseline
    for t in range(n_monitored):
        w = stream.n_baseline + t
        if _crosses_onset(batch_from_windows(stream, w, w + 1), phase2_cfg, baselines):
            return t
    return None


def tier2_onset_index_rolling(stream: StreamData, phase2_cfg: Phase2Config, n_lookback: int) -> Optional[int]:
    """First monitored window at which harm exceeds 1 + kappa times a rolling baseline from the `n_lookback` real
    windows immediately preceding it, rather than one fixed early-period baseline.

    This is the fix for the confound `tier2_onset_index` cannot resolve: a rolling reference tracks real calendar
    drift (e.g. Sepsis-3's falling prevalence) automatically, so smooth real drift alone does not cross the ratio
    threshold. **Trade-off**, the same kind Tier 1's rolling-calibration sensitivity study found for the detection
    threshold (`M5f-Rolling-Calibration.md`): a persistent (not abrupt) rise can be partly absorbed into the
    rolling baseline itself over time, delaying or weakening detection of a genuine, sustained failure. An abrupt
    step is still detected promptly, since the windows just before it are unaffected and give an accurate reference.
    """
    n_monitored = stream.y.shape[0] - stream.n_baseline
    for t in range(n_monitored):
        w = stream.n_baseline + t
        lookback_start = max(0, w - n_lookback)
        if lookback_start >= w:
            continue   # not enough real history yet to form a rolling baseline
        baselines = rhe_baselines(batch_from_windows(stream, lookback_start, w), phase2_cfg)
        if _crosses_onset(batch_from_windows(stream, w, w + 1), phase2_cfg, baselines):
            return t
    return None
