"""Tier 1 simulation: synthetic world, deployed model, reviewer, harm, and perturbations."""

from . import perturbations  # noqa: F401  (registers perturbations)
from .config import Phase2Config, load_config
from .generator import Batch, SyntheticWorld
from .harm import HARM_POPULATION, expected_rhe, onset_ratio, rhe_baselines
from .harm_levels import AXES, LevelSolution, baseline_rhe, baselines_for, ratio_at, solve_level, solve_levels
from .levels import parse_level
from .model import DeployedModel
from .perturbations import PerturbationConfig
from .registry import PerturbationFactory, registered_perturbations
from .reviewer import interception_probability

__all__ = ["AXES", "HARM_POPULATION", "Batch", "LevelSolution", "baseline_rhe", "baselines_for", "DeployedModel", "Phase2Config", "PerturbationConfig", "PerturbationFactory", "SyntheticWorld",
           "expected_rhe", "interception_probability", "load_config", "onset_ratio", "parse_level", "ratio_at", "registered_perturbations", "rhe_baselines", "solve_level", "solve_levels"]
