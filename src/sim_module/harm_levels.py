"""Harm-scaled levels: solve a perturbation's parameter for a target ratio of expected RHE to its baseline.

Levels defined on the harm scale are comparable across mechanisms and re-solve when the harm weights change. Each
perturbation has an effect axis from a neutral value to a plausibility cap. A target that the cap cannot reach is
reported as unattainable, together with the best ratio reached.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.optimize import brentq

from .config import Phase2Config
from .generator import SyntheticWorld
from .harm import HARM_POPULATION, expected_rhe, onset_ratio, rhe_baselines
from .model import DeployedModel
from .perturbations import PerturbationConfig
from .registry import PerturbationFactory

DEFAULT_CASES = HARM_POPULATION
SEED = 31
GRID_POINTS = 9


@dataclass(frozen=True)
class Axis:
    """Effect axis of a perturbation: the neutral value and the largest plausible value."""

    neutral: float
    cap: float
    unit: str


AXES: Dict[str, Axis] = {
    "covariate_mean_shift": Axis(0.0, 1.5, "SD along the discriminative direction"),
    "class_prior_shift": Axis(1.0, 3.0, "x prevalence"),
    "slope_and_intercept_drift": Axis(1.0, 0.5, "calibration slope"),
    "shrink_scores_to_base_rate_recalibrated": Axis(0.0, 0.6, "lambda"),
    "mcar_missingness": Axis(0.0, 0.5, "fraction missing"),
    "variable_rescale": Axis(1.0, 10.0, "x scale factor"),
    "outcome_model_change": Axis(0.0, 0.45, "relative sensitivity loss"),   # the maximum reachable loss is about 0.497
    "subgroup_outcome_model_change": Axis(0.0, 0.45, "relative sensitivity loss within the subgroup"),
    "version_swap_with_regression": Axis(0.0, 0.9, "fraction of the weight vector replaced"),
    "covariate_and_concept_drift": Axis(0.0, 0.45, "relative sensitivity loss on top of a 0.5 SD covariate shift"),
    "out_of_scope_use_share": Axis(0.0, 0.8, "share of invocations outside the indication"),
}


@dataclass(frozen=True)
class LevelSolution:
    """Result of solving for a target ratio."""

    target: float
    value: Optional[float]
    ratio: Optional[float]
    attainable: bool
    max_ratio: float
    monotone: bool


def baseline_rhe(world: SyntheticWorld, model: DeployedModel, cfg: Phase2Config, n: int = DEFAULT_CASES) -> float:
    """Expected RHE of the baseline world, at the solver's sample size and seed."""
    return expected_rhe(model.apply(world.sample(n, np.random.default_rng(SEED))), cfg)


def baselines_for(world: SyntheticWorld, model: DeployedModel, cfg: Phase2Config, n: int = DEFAULT_CASES) -> Dict[str, float]:
    """Pooled and subgroup baseline RHE at the solver's sample size and seed."""
    return rhe_baselines(model.apply(world.sample(n, np.random.default_rng(SEED))), cfg)


def ratio_at(perturbation_type: str, value: float, world: SyntheticWorld, model: DeployedModel, cfg: Phase2Config,
             n: int = DEFAULT_CASES, mode: Optional[str] = None, baseline: Optional[Dict[str, float]] = None) -> float:
    """Onset ratio under a perturbation at full effect (decision D18: the larger of the pooled and subgroup ratios)."""
    pert = PerturbationFactory(perturbation_type)(PerturbationConfig(level_value=value, mode=mode))
    batch = pert.generate(world, model, n, np.random.default_rng(SEED), 1.0)
    return onset_ratio(batch, cfg, baseline if baseline is not None else baselines_for(world, model, cfg, n))


def solve_levels(perturbation_type: str, targets: Sequence[float], world: SyntheticWorld, model: DeployedModel,
                 cfg: Phase2Config, n: int = DEFAULT_CASES, mode: Optional[str] = None) -> List[LevelSolution]:
    """Solve for several target ratios, sharing one baseline and one effect grid.

    Raises:
        KeyError: When the perturbation type has no effect axis.
    """
    axis = AXES[perturbation_type]
    base = baselines_for(world, model, cfg, n)

    def value_at(s: float) -> float:
        return axis.neutral + s * (axis.cap - axis.neutral)

    def ratio(s: float) -> float:
        return ratio_at(perturbation_type, value_at(s), world, model, cfg, n, mode, base)

    grid = np.linspace(0.0, 1.0, GRID_POINTS)
    ratios = np.array([ratio(s) for s in grid])
    monotone = bool(np.all(np.diff(ratios) >= -0.02))
    out: List[LevelSolution] = []
    for target in targets:
        if ratios.max() < target:
            out.append(LevelSolution(target, None, None, False, float(ratios.max()), monotone))
            continue
        hit = int(np.argmax(ratios >= target))
        if hit == 0:
            out.append(LevelSolution(target, value_at(0.0), float(ratios[0]), True, float(ratios.max()), monotone))
            continue
        s = float(brentq(lambda x: ratio(x) - target, grid[hit - 1], grid[hit], xtol=1e-4))
        out.append(LevelSolution(target, value_at(s), float(ratio(s)), True, float(ratios.max()), monotone))
    return out


def solve_level(perturbation_type: str, target: float, world: SyntheticWorld, model: DeployedModel, cfg: Phase2Config,
                n: int = DEFAULT_CASES, mode: Optional[str] = None) -> LevelSolution:
    """Find the parameter value at which the expected-RHE ratio equals `target`, or report that the cap cannot reach it."""
    return solve_levels(perturbation_type, [target], world, model, cfg, n, mode)[0]
