"""Base class for perturbations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Optional

import numpy as np

from src.sim_module.generator import Batch, SyntheticWorld
from src.sim_module.model import DeployedModel


@dataclass(frozen=True)
class PerturbationConfig:
    """Level of a perturbation and an optional mechanism variant."""

    level_value: float
    mode: Optional[str] = None


class Perturbation(ABC):
    """A defined operation on the simulated world, scaled by `progress` in [0, 1].

    progress = 1 is the full effect (abrupt shapes and the end of a ramp). Ratio-type levels are interpolated
    from 1, and amount-type levels from 0.
    """

    name: ClassVar[str] = ""

    def __init__(self, cfg: PerturbationConfig) -> None:
        self.cfg = cfg

    def ramp_ratio(self, progress: float) -> float:
        """Interpolate a ratio-type level from 1 (no effect) to its full value."""
        return 1.0 + progress * (self.cfg.level_value - 1.0)

    def ramp_amount(self, progress: float) -> float:
        """Scale an amount-type level from 0 (no effect) to its full value."""
        return progress * self.cfg.level_value

    @abstractmethod
    def generate(self, world: SyntheticWorld, model: DeployedModel, n: int, rng: np.random.Generator,
                 progress: float = 1.0) -> Batch:
        """Draw `n` cases under the perturbation and run the deployed model on them."""
