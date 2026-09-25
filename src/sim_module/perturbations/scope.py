"""Scope creep (SC19): a growing share of invocations falls outside the VOE indication."""

from dataclasses import replace

import numpy as np

from src.sim_module.generator import Batch, SyntheticWorld
from src.sim_module.model import DeployedModel
from src.sim_module.registry import register_perturbation

from .base import Perturbation, PerturbationConfig
from .data_layer import CovariateAndConceptDrift

OUT_OF_SCOPE_LOSS = 0.4   # relative sensitivity loss outside the indication, on top of the compound scenario's fixed 0.5 SD covariate shift


@register_perturbation("out_of_scope_use_share")
class OutOfScopeUse(Perturbation):
    """The level is the share of invocations outside the indication at full effect. Those cases come from a shifted population on which
    the model performs worse (an assumption), and they carry the out-of-scope flag that C3 counts."""

    def generate(self, world: SyntheticWorld, model: DeployedModel, n: int, rng: np.random.Generator,
                 progress: float = 1.0) -> Batch:
        share = min(self.ramp_amount(progress), 1.0)
        n_out = int(rng.binomial(n, share))
        parts = [model.apply(world.sample(n - n_out, rng))]
        if n_out:
            outside = CovariateAndConceptDrift(PerturbationConfig(level_value=OUT_OF_SCOPE_LOSS)).generate(world, model, n_out, rng, 1.0)
            parts.append(replace(outside, out_of_scope=np.ones(n_out, dtype=bool)))
        joined = Batch.concat(parts)
        return joined.select(rng.permutation(joined.n))     # every window of a run carries its share
