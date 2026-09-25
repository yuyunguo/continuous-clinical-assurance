"""Windowed streams: baseline windows followed by monitored windows, with a perturbation that starts at s0."""

from dataclasses import dataclass
from itertools import groupby
from typing import Any, Dict, List, Optional

import numpy as np

from src.sim_module.config import Phase2Config
from src.sim_module.generator import Batch, SyntheticWorld
from src.sim_module.harm import HARM_POPULATION, expected_rhe, onset_ratio
from src.sim_module.model import DeployedModel
from src.sim_module.perturbations import Perturbation


@dataclass(frozen=True)
class StreamData:
    """Per-window arrays for one stream. The first `n_baseline` windows are the baseline."""

    y: np.ndarray            # (W, m)
    g: np.ndarray
    severity: np.ndarray
    decision: np.ndarray
    reported_p: np.ndarray
    x_model: np.ndarray      # (W, m, d)
    n_baseline: int
    s0: Optional[int]        # deviation start, as an index into the monitored windows
    progress: np.ndarray     # (T,) perturbation progress per monitored window
    out_of_scope: Optional[np.ndarray] = None   # (W, m) invocation outside the VOE indication (None: all in scope)


def progress_schedule(shape: str, s0: int, monitored: int, ramp: int) -> np.ndarray:
    """Perturbation progress in [0, 1] per monitored window for an abrupt, gradual, or incremental shape."""
    t = np.arange(monitored)
    steps = np.clip((t - s0 + 1) / ramp, 0.0, 1.0)
    if shape == "abrupt":
        return (t >= s0).astype(float)
    if shape == "gradual":
        return steps
    if shape == "incremental":
        return np.ceil(3 * steps - 1e-9) / 3.0
    raise ValueError(f"unknown shape {shape!r}")


def _stack(batches: List[Batch]) -> Batch:
    """Concatenate batches along the case axis."""
    return Batch.concat(batches)


def build_stream(world: SyntheticWorld, model: DeployedModel, perturbation: Optional[Perturbation], shape: str,
                 s0: int, cfg: Phase2Config, rng: np.random.Generator) -> StreamData:
    """Draw a stream. With no perturbation every window comes from the baseline world."""
    m, b, t = cfg.stream.cases_per_window, cfg.stream.baseline_windows, cfg.stream.monitored_windows
    progress = progress_schedule(shape, s0, t, cfg.stream.ramp_windows) if perturbation is not None else np.zeros(t)
    batches = [model.apply(world.sample(b * m, rng))]
    for value, run in groupby(progress):
        length = len(list(run))
        if perturbation is None or value == 0.0:
            batches.append(model.apply(world.sample(length * m, rng)))
        else:
            batches.append(perturbation.generate(world, model, length * m, rng, float(value)))
    batch = _stack(batches)
    w = b + t
    flag_rng = np.random.default_rng(int(rng.integers(0, 2 ** 32 - 1)))    # drawn after every case, so no earlier draw moves
    out_of_scope = batch.scope_flags() | (flag_rng.random(batch.n) < cfg.context.baseline_out_of_scope)
    return StreamData(y=batch.y.reshape(w, m), g=batch.g.reshape(w, m), severity=batch.severity.reshape(w, m),
                      decision=batch.decision.reshape(w, m), reported_p=batch.reported_p.reshape(w, m),
                      x_model=batch.x_model.reshape(w, m, -1), n_baseline=b,
                      s0=s0 if perturbation is not None else None, progress=progress, out_of_scope=out_of_scope.reshape(w, m))


def pooled_excess(pert: Any, world: SyntheticWorld, model: DeployedModel, cfg: Any, progress: float, baselines: Dict[str, float]) -> float:
    """Excess expected RHE per window over the pooled baseline, at a given perturbation progress (per 1,000 cases)."""
    batch = pert.generate(world, model, HARM_POPULATION, np.random.default_rng(31), progress)
    return max(0.0, expected_rhe(batch, cfg) - baselines["pooled"])


def excess_ratio(pert: Any, world: SyntheticWorld, model: DeployedModel, cfg: Any, progress: float, baselines: Dict[str, float]) -> float:
    """Excess harm per window in baseline-windows of the stratum that defines onset: the D18 onset ratio minus one.

    Where the pooled cohort defines onset this equals `pooled_excess` divided by the pooled baseline. Where a predefined subgroup defines it (SC09), the
    subgroup's own baseline is the unit, so a subgroup-concentrated failure counts on the same scale as a pooled one. Summed over the windows of delay,
    it is exposure in baseline-windows of harm, the unit of the D23 anchor.
    """
    batch = pert.generate(world, model, HARM_POPULATION, np.random.default_rng(31), progress)
    return max(0.0, onset_ratio(batch, cfg, baselines) - 1.0)

