"""Perturbations that act on the data and model layer (catalog scenarios SC00 to SC05, SC07, SC08)."""

from dataclasses import replace
from typing import Optional

import numpy as np
from scipy.optimize import brentq

from src.eval_module.metrics import logit, sigmoid
from src.sim_module.generator import Batch, SyntheticWorld
from src.sim_module.model import DeployedModel
from src.sim_module.registry import register_perturbation

from .base import Perturbation

CRITICAL_VARIABLE = 0  # the VOE-critical input used by the missingness and rescaling scenarios


def solve_slope_for_relative_se_loss(world: SyntheticWorld, model: DeployedModel, loss: float,
                                     group: Optional[int] = None) -> float:
    """Outcome-link slope at which sensitivity at the fixed threshold falls by `loss` (relative), prevalence held.

    With `group` set, sensitivity and prevalence are those of that subgroup. Sensitivity is computed from expected
    outcomes on a fixed baseline pool, so the solution is deterministic.
    """
    if loss <= 0:
        return 1.0
    z, g = world.pool()
    flagged = (model.predict_proba(world.pool_inputs()) >= model.threshold).astype(float)
    if group is not None:
        z, flagged = z[g == group], flagged[g == group]

    def sensitivity_at(slope: float) -> float:
        mu = sigmoid(world.intercept_for_prevalence(slope, group) + slope * z)
        return float((mu * flagged).sum() / mu.sum())

    baseline_se = sensitivity_at(1.0)
    maximum_loss = 1.0 - sensitivity_at(0.0) / baseline_se   # a pure-noise outcome link still flags the same share of cases
    if loss >= maximum_loss:
        raise ValueError(f"relative sensitivity loss {loss:.3f} is not reachable; the maximum is {maximum_loss:.3f}")
    target = baseline_se * (1.0 - loss)
    return float(brentq(lambda s: sensitivity_at(s) - target, 0.0, 1.0, xtol=1e-6))


@register_perturbation("none")
class NoPerturbation(Perturbation):
    """Control: the baseline world."""

    def generate(self, world: SyntheticWorld, model: DeployedModel, n: int, rng: np.random.Generator,
                 progress: float = 1.0) -> Batch:
        return model.apply(world.sample(n, rng))


@register_perturbation("covariate_mean_shift")
class CovariateMeanShift(Perturbation):
    """Shift P(x) along the discriminative direction while P(y | x) stays the true posterior (SC01)."""

    def generate(self, world: SyntheticWorld, model: DeployedModel, n: int, rng: np.random.Generator,
                 progress: float = 1.0) -> Batch:
        shift = self.ramp_amount(progress) * world.direction
        return model.apply(world.sample_x_first(n, rng, mean_shift=shift))


@register_perturbation("class_prior_shift")
class ClassPriorShift(Perturbation):
    """Multiply prevalence with class-conditional distributions fixed, so Se, Sp, and AUROC are invariant (SC02)."""

    def generate(self, world: SyntheticWorld, model: DeployedModel, n: int, rng: np.random.Generator,
                 progress: float = 1.0) -> Batch:
        prevalence = min(0.95, world.cfg.prevalence * self.ramp_ratio(progress))
        return model.apply(world.sample(n, rng, prevalence=prevalence))


@register_perturbation("slope_and_intercept_drift")
class CalibrationSlopeDrift(Perturbation):
    """Overconfidence (SC03). The level is the target calibration slope.

    mode "output_transform" (default): the reported confidence becomes sigmoid(logit(p) / slope). This is monotone, so
    the ranking and AUROC are exactly unchanged, and decisions still use the original model output. Only the
    confidence a reviewer sees drifts, so conventional performance metrics cannot see this variant by construction.

    mode "outcome_link": the outcome depends less on the score (slope below 1), with prevalence held. AUROC then falls,
    so the AUROC constraint of the catalog holds only for mild slopes.
    """

    def generate(self, world: SyntheticWorld, model: DeployedModel, n: int, rng: np.random.Generator,
                 progress: float = 1.0) -> Batch:
        slope = self.ramp_ratio(progress)
        mode = self.cfg.mode or "output_transform"
        if mode == "output_transform":
            batch = model.apply(world.sample(n, rng))
            return replace(batch, reported_p=sigmoid(logit(batch.p_hat) / slope))
        if mode == "outcome_link":
            return model.apply(world.sample_x_first(n, rng, slope=slope, intercept=world.intercept_for_prevalence(slope)))
        raise ValueError(f"unknown mode {mode!r} for slope_and_intercept_drift")


@register_perturbation("shrink_scores_to_base_rate_recalibrated")
class ShrunkButCalibrated(Perturbation):
    """Discrimination falls at maintained calibration (SC04): outcomes follow the degraded probabilities."""

    def generate(self, world: SyntheticWorld, model: DeployedModel, n: int, rng: np.random.Generator,
                 progress: float = 1.0) -> Batch:
        lam = self.ramp_amount(progress)
        batch = model.apply(world.sample(n, rng))
        p_new = (1.0 - lam) * batch.p_hat + lam * world.cfg.prevalence
        y = (rng.random(n) < p_new).astype(int)
        return replace(batch, y=y, severity=world.draw_severity(y, rng), reported_p=p_new,
                       decision=(p_new >= model.threshold).astype(int))


@register_perturbation("mcar_missingness")
class RandomMissingness(Perturbation):
    """Missing completely at random in the critical variable, imputed with the training mean (SC05)."""

    def generate(self, world: SyntheticWorld, model: DeployedModel, n: int, rng: np.random.Generator,
                 progress: float = 1.0) -> Batch:
        batch = world.sample(n, rng)
        mask = np.zeros_like(batch.x, dtype=bool)
        mask[:, CRITICAL_VARIABLE] = rng.random(n) < self.ramp_amount(progress)
        return model.apply(batch, missing_mask=mask)


@register_perturbation("variable_rescale")
class VariableRescale(Perturbation):
    """A unit or scale change in the critical input (SC07)."""

    def generate(self, world: SyntheticWorld, model: DeployedModel, n: int, rng: np.random.Generator,
                 progress: float = 1.0) -> Batch:
        return model.apply(world.sample(n, rng), rescale=(CRITICAL_VARIABLE, self.ramp_ratio(progress)))


@register_perturbation("outcome_model_change")
class ConceptDrift(Perturbation):
    """P(y | x) weakens while P(x) and prevalence stay fixed. The level is the relative sensitivity loss (SC08)."""

    def generate(self, world: SyntheticWorld, model: DeployedModel, n: int, rng: np.random.Generator,
                 progress: float = 1.0) -> Batch:
        slope = solve_slope_for_relative_se_loss(world, model, self.ramp_amount(progress))
        return model.apply(world.sample_x_first(n, rng, slope=slope, intercept=world.intercept_for_prevalence(slope)))


@register_perturbation("subgroup_outcome_model_change")
class SubgroupConceptDrift(Perturbation):
    """Concept drift confined to subgroup 1, with P(x) and both prevalences fixed. The level is the relative
    sensitivity loss within the subgroup (SC09)."""

    def generate(self, world: SyntheticWorld, model: DeployedModel, n: int, rng: np.random.Generator,
                 progress: float = 1.0) -> Batch:
        slope = solve_slope_for_relative_se_loss(world, model, self.ramp_amount(progress), group=1)
        baseline_intercept = float(np.log(world.cfg.prevalence / (1 - world.cfg.prevalence)))
        return model.apply(world.sample_x_first(
            n, rng, slope_by_group=(1.0, slope),
            intercept_by_group=(baseline_intercept, world.intercept_for_prevalence(slope, group=1))))


@register_perturbation("version_swap_with_regression")
class VersionRegression(Perturbation):
    """The deployed model is replaced by a regressed version. The world is unchanged. The level is the fraction of
    the weight vector replaced by an orthogonal direction (SC17)."""

    def generate(self, world: SyntheticWorld, model: DeployedModel, n: int, rng: np.random.Generator,
                 progress: float = 1.0) -> Batch:
        return model.with_regression(self.ramp_amount(progress)).apply(world.sample(n, rng))


COMPOUND_SHIFT_SD = 0.5   # fixed covariate shift along the discriminative direction in the compound scenario


@register_perturbation("covariate_and_concept_drift")
class CovariateAndConceptDrift(Perturbation):
    """Two changes at once: a fixed covariate shift and a weakening outcome link whose level is a relative sensitivity
    loss, as in SC08 (compound scenario)."""

    def generate(self, world: SyntheticWorld, model: DeployedModel, n: int, rng: np.random.Generator,
                 progress: float = 1.0) -> Batch:
        slope = solve_slope_for_relative_se_loss(world, model, self.ramp_amount(progress))
        shift = progress * COMPOUND_SHIFT_SD * world.direction
        return model.apply(world.sample_x_first(n, rng, mean_shift=shift, slope=slope,
                                                intercept=world.intercept_for_prevalence(slope)))

