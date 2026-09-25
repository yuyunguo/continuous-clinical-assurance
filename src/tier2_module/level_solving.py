"""Harm-scaled perturbation levels for Tier 2, mirroring `src.sim_module.harm_levels`: solve a perturbation's
parameter for a target D18 ratio (the larger of the pooled and subgroup expected-RHE ratios), rather than reusing
a magnitude chosen for the synthetic stand-in's much lower-dimensional world unchanged.

**Why this exists:** the first Tier 2 real-data replicate run reused Tier 1's SC07 rescale magnitude (3.0x, close
to Tier 1's own solved "medium" level of 3.51x) and found it produced no detectable real harm change at all — a
single rescaled feature's influence is diluted across many correlated real features after regularization, unlike
the synthetic world's few, purpose-built features. This solves the level on the real data itself, the same way
Tier 1 already solves scenario levels against explicit `harm_targets` (`phase2/scenarios.yaml`).

Uses the largest available real population (not a single 500-case window) for a low-noise ratio estimate — the
real-data analogue of Tier 1's `HARM_POPULATION = 1,000,000`-case evaluation, at whatever scale real data allows.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from src.sim_module.config import Phase2Config
from src.sim_module.generator import Batch, draw_severity_from_probs
from src.sim_module.harm import SUBGROUP, expected_rhe, rhe_baselines

from .config import Tier2StreamConfig
from .real_model import RealDeployedModel

GRID_POINTS = 9
RatioFn = Callable[[float], float]   # level_value -> the D18 ratio at that level, e.g. from `tier2_ratio_for`


@dataclass(frozen=True)
class LevelSolution:
    """Result of solving for a target ratio, mirroring `src.sim_module.harm_levels.LevelSolution`."""

    target: float
    value: Optional[float]
    ratio: Optional[float]
    attainable: bool
    max_ratio: float


def tier2_ratio_for(population: pd.DataFrame, cfg: Tier2StreamConfig, phase2_cfg: Phase2Config,
                    perturb: Callable[[pd.DataFrame], pd.DataFrame]) -> float:
    """The D18 ratio (max of pooled and subgroup expected-RHE ratios) between `population` (the baseline reference)
    and `perturb(population)` (the same cases, scored under some perturbation at some level).

    `perturb` takes the unperturbed population and returns it scored (with `decision`/`p_hat`) under one
    perturbation at one level — e.g. `lambda pop: model.apply(pop.assign(**{column: pop[column] * level_value}))`
    for a rescale, or `lambda pop: model.with_regression(fraction).apply(pop)` for a version regression. This
    generality is what lets `tier2_solve_level` calibrate levels for any perturbation type, not just rescale.
    """
    baseline_batch = _population_batch(population, cfg, phase2_cfg)
    perturbed_batch = _population_batch(perturb(population), cfg, phase2_cfg)

    baselines = rhe_baselines(baseline_batch, phase2_cfg)
    pooled_ratio = expected_rhe(perturbed_batch, phase2_cfg) / baselines["pooled"] if baselines["pooled"] > 0 else float("inf")
    in_group = perturbed_batch.g == SUBGROUP
    ratio = pooled_ratio
    if in_group.mean() >= phase2_cfg.harm.min_subgroup_share and in_group.any() and baselines["subgroup"] > 0:
        subgroup_ratio = expected_rhe(perturbed_batch.select(in_group), phase2_cfg) / baselines["subgroup"]
        ratio = max(ratio, subgroup_ratio)
    return float(ratio)


def _ratio_at(population: pd.DataFrame, cfg: Tier2StreamConfig, phase2_cfg: Phase2Config, model: RealDeployedModel,
             column: str, level_value: float) -> float:
    """The D18 ratio at full rescale effect on `population`. Kept as a thin, rescale-specific wrapper around
    `tier2_ratio_for` for `tier2_solve_rescale_level`'s existing call sites and tests."""
    def perturb(pop: pd.DataFrame) -> pd.DataFrame:
        rescaled = pop.copy()
        rescaled[column] = pop[column].to_numpy() * level_value
        return model.apply(rescaled)
    return tier2_ratio_for(population, cfg, phase2_cfg, perturb)


def _population_batch(scored: pd.DataFrame, cfg: Tier2StreamConfig, phase2_cfg: Phase2Config) -> Batch:
    """A single flat `Batch` from the ENTIRE `scored` population — not split into `cfg.window_size` monitoring
    windows, since level-solving is a one-shot population-scale evaluation, not a windowed stream. Severity is
    drawn from the same elicited distribution `build_stream_data` uses (`draw_severity_from_probs`), with a fixed
    seed so the same population gives the same severity draw every call (needed for `brentq`'s root search to
    converge on a stable function)."""
    y = scored[cfg.label_column].to_numpy()
    g = scored[cfg.subgroup_column].to_numpy()
    decision = scored["decision"].to_numpy()
    reported_p = scored["p_hat"].to_numpy()
    severity = draw_severity_from_probs(y, phase2_cfg.harm.severity_probs_positive, phase2_cfg.harm.severity_probs_negative,
                                        np.random.default_rng(0))
    return Batch(x=np.empty((len(y), 0)), y=y, g=g, severity=severity, decision=decision, reported_p=reported_p)


def tier2_solve_level(ratio_at: RatioFn, target_ratio: float, neutral: float, cap: float) -> LevelSolution:
    """Find the level (in `[neutral, cap]`) at which `ratio_at(level)` equals `target_ratio`.

    Mirrors `src.sim_module.harm_levels.solve_level`: grids the effect axis from `neutral` (no effect) to `cap`,
    reports the target as unattainable (with the best ratio reached) if even `cap` falls short, and root-finds
    with `brentq` inside the bracket that first reaches the target otherwise. Generic over any perturbation type
    that has a single scalar "how strong" parameter — `tier2_ratio_for` supplies `ratio_at` for a given `perturb`.
    """
    grid = np.linspace(neutral, cap, GRID_POINTS)
    ratios = np.array([ratio_at(float(v)) for v in grid])
    if ratios.max() < target_ratio:
        return LevelSolution(target_ratio, None, None, False, float(ratios.max()))
    hit = int(np.argmax(ratios >= target_ratio))
    if hit == 0:
        return LevelSolution(target_ratio, float(grid[0]), float(ratios[0]), True, float(ratios.max()))
    value = float(brentq(lambda v: ratio_at(v) - target_ratio, grid[hit - 1], grid[hit], xtol=1e-3))
    return LevelSolution(target_ratio, value, ratio_at(value), True, float(ratios.max()))


def tier2_solve_rescale_level(population: pd.DataFrame, cfg: Tier2StreamConfig, phase2_cfg: Phase2Config,
                              model: RealDeployedModel, column: str, target_ratio: float, cap: float) -> LevelSolution:
    """Find the rescale factor for `column` at which the D18 ratio on `population` equals `target_ratio`.

    A thin binding of `tier2_solve_level` to `_ratio_at`'s rescale mechanics; `neutral = 1.0` (no rescale).
    """
    return tier2_solve_level(lambda v: _ratio_at(population, cfg, phase2_cfg, model, column, v), target_ratio, 1.0, cap)
