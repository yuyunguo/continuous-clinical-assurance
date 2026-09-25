"""Multi-replicate TAFD comparison on Tier 2: repeated resampled-and-perturbed streams, classified per arm.

Reuses `src.eval_module.tafd.classify_replicate`/`rmst_difference`/`bootstrap_difference` completely unchanged —
they operate on plain arrays (`events`, `s0`, `t0`, `horizon`, `j`), with no dependence on the Tier 1 synthetic
generator. The only Tier 2-specific piece is producing each replicate: a block-bootstrap resample of the real
windows (`resample.py`), a perturbation applied from a randomly chosen onset window (`perturb.py`), and the real
onset window itself.

**Onset detection uses `tier2_onset_index_calibrated`** (`onset_calibration.py`), not the naive
`tier2_onset_index`/`tier2_onset_index_rolling`: investigation found the naive rule false-fires on essentially
every pure-null real replicate (100/100 in a 100-replicate check on real MIMIC-IV data), because a single
500-case window's sampling noise is too close to the elicited kappa threshold for a single-window point test to
be meaningful. The caller calibrates `onset_threshold` once (via `tier2_calibrate_onset`, exactly the same
block-bootstrap-null + `calibrate_threshold` pattern already used for the monitoring arms) and passes it in,
mirroring how `arm_thresholds` are already pre-calibrated outside the loop.
"""

from typing import Callable, Dict

import numpy as np
import pandas as pd

from scripts.m4_pilot import SPLIT_ARMS
from src.eval_module.tafd import classify_replicate
from src.monitor_module import (INDICATOR_ORDER, component_groups, indicator_indices, stream_scores_multi,
                                system_events, system_events_hierarchical)
from src.sim_module.config import HarmConfig, Phase2Config

from .arms import FLAT_ARMS, ArmResult
from .config import Tier2StreamConfig
from .onset_calibration import tier2_onset_index_calibrated
from .resample import block_bootstrap_windows
from .stream import build_stream_data

PerturbFn = Callable[..., pd.DataFrame]   # called as perturb_fn(resampled_windowed_cohort, s0=onset_window) -> scored cohort


def tier2_replicate_tafd(windowed: pd.DataFrame, cfg: Tier2StreamConfig, phase2_cfg: Phase2Config, harm: HarmConfig,
                         arm_thresholds: Dict[str, ArmResult], lag: int, perturb_fn: PerturbFn, n_replicates: int,
                         n_windows_out: int, n_baseline: int, block_size: int, horizon: int, onset_threshold: float,
                         rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Per-arm TAFD across `n_replicates` replicates, each a fresh block-bootstrap resample with `perturb_fn`
    applied from a randomly chosen monitored onset window. `perturb_fn` is called as `perturb_fn(resampled, s0=s0)`
    — bind everything else (e.g. via `functools.partial`) before passing it in. `onset_threshold` is a
    pre-calibrated CUSUM threshold from `tier2_calibrate_onset`.

    A replicate is dropped (not padded or imputed) if `tier2_onset_index_calibrated` finds no real onset, or if
    detection would run past `horizon` windows from the resampled stream's end — mirroring Tier 1's
    `m4_pilot.py`, which drops replicates the same way rather than inventing a value for them.
    """
    cols = {name: indicator_indices(ids) for name, ids in FLAT_ARMS.items()}
    groups = component_groups(INDICATOR_ORDER)
    n_monitored = n_windows_out - n_baseline
    j = phase2_cfg.routing.hysteresis_j
    tafd: Dict[str, list] = {name: [] for name in arm_thresholds}

    for _ in range(n_replicates):
        resampled = block_bootstrap_windows(windowed, cfg, n_windows_out, block_size, rng)
        s0 = int(rng.integers(0, n_monitored))
        perturbed = perturb_fn(resampled, s0=s0)
        stream = build_stream_data(perturbed, cfg, harm, n_baseline=n_baseline, rng=rng)
        t0 = tier2_onset_index_calibrated(stream, phase2_cfg, onset_threshold, j)
        if t0 is None or t0 + horizon > n_monitored:
            continue
        z = stream_scores_multi(stream, phase2_cfg, (lag,))[lag]
        for name, (threshold, _rate, _se) in arm_thresholds.items():
            if name in SPLIT_ARMS:
                events = system_events_hierarchical(z[None], groups, threshold)[0]  # type: ignore[arg-type]
            else:
                events = system_events(z[None], cols[name], threshold)[0]  # type: ignore[arg-type]
            out = classify_replicate(events, s0, t0, horizon, j)
            tafd[name].append(out["tafd"])

    return {name: np.array(values, dtype=float) for name, values in tafd.items()}
