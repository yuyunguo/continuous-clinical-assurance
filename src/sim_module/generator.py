"""Tier 1 synthetic world.

Baseline: y ~ Bernoulli(pi), subgroup g ~ Bernoulli(share), x | y, g ~ N(mu_{y,g}, I) with mu_{1,g} = v_g / 2 and
mu_{0,g} = -v_g / 2. The true posterior is then exactly logit P(y=1 | x, g) = log(pi / (1 - pi)) + v_g . x, which is
what lets the perturbations keep Se, Sp, or AUROC invariant by construction.
"""

from dataclasses import dataclass, field, replace
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import brentq

from src.eval_module.metrics import sigmoid

from .config import GeneratorConfig, HarmConfig

POOL_SIZE = 100_000
POOL_SEED = 987_654


@dataclass(frozen=True)
class Batch:
    """A block of cases. The model fields are empty arrays until `DeployedModel.apply` fills them in."""

    x: np.ndarray
    y: np.ndarray
    g: np.ndarray
    severity: np.ndarray
    x_model: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    missing_mask: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=bool))
    p_hat: np.ndarray = field(default_factory=lambda: np.empty(0))
    reported_p: np.ndarray = field(default_factory=lambda: np.empty(0))
    decision: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    out_of_scope: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))   # invoked outside the VOE indication (empty: all in scope)

    @property
    def n(self) -> int:
        """Number of cases."""
        return int(self.y.shape[0])

    def scope_flags(self) -> np.ndarray:
        """The out-of-scope flag of every case, false where the batch carries none."""
        return self.out_of_scope if self.out_of_scope.shape[0] == self.n else np.zeros(self.n, dtype=bool)

    @classmethod
    def concat(cls, batches: "List[Batch]") -> "Batch":
        """Concatenate batches along the case axis. Batches without a scope flag count as in scope."""
        cat = lambda name: np.concatenate([getattr(b, name) for b in batches])  # noqa: E731
        return cls(x=cat("x"), y=cat("y"), g=cat("g"), severity=cat("severity"), x_model=cat("x_model"), missing_mask=cat("missing_mask"),
                   p_hat=cat("p_hat"), reported_p=cat("reported_p"), decision=cat("decision"),
                   out_of_scope=np.concatenate([b.scope_flags() for b in batches]))

    def _take(self, index: Union[slice, np.ndarray]) -> "Batch":
        """The cases at `index`. Model fields that are still empty (a batch the model has not scored) stay empty."""
        pick = lambda a: a[index] if a.shape[0] == self.n else a  # noqa: E731
        return Batch(x=self.x[index], y=self.y[index], g=self.g[index], severity=self.severity[index], x_model=pick(self.x_model),
                     missing_mask=pick(self.missing_mask), p_hat=pick(self.p_hat), reported_p=pick(self.reported_p), decision=pick(self.decision),
                     out_of_scope=self.scope_flags()[index])

    def slice(self, start: int, stop: int) -> "Batch":
        """Cases start to stop (used to estimate Monte Carlo error from independent chunks)."""
        return self._take(slice(start, stop))

    def select(self, mask: np.ndarray) -> "Batch":
        """The cases where `mask` is true (or at the given indices), with every field kept aligned."""
        return self._take(mask)

    def with_outcomes(self, y: np.ndarray, severity: np.ndarray) -> "Batch":
        """Copy of the batch with new outcomes and severities."""
        return replace(self, y=y, severity=severity)


def draw_severity_from_probs(y: np.ndarray, sev_pos: Tuple[float, ...], sev_neg: Tuple[float, ...],
                              rng: np.random.Generator) -> np.ndarray:
    """Severity class (1 to 3) from class-conditional probabilities, given the outcome. Shared by `SyntheticWorld` and the Tier 2 real-data loader,
    so a real case's severity is drawn from the same elicited distribution as a synthetic one."""
    u = rng.random(y.shape[0])
    edges_pos = np.cumsum(sev_pos)
    edges_neg = np.cumsum(sev_neg)
    return np.where(y == 1, np.searchsorted(edges_pos, u), np.searchsorted(edges_neg, u)).astype(int) + 1


class SyntheticWorld:
    """Data-generating process for the Tier 1 simulation."""

    def __init__(self, cfg: GeneratorConfig, harm: Optional[HarmConfig] = None) -> None:
        self.cfg = cfg
        weights = np.asarray(cfg.feature_weights, dtype=float)
        self.v = cfg.separation * weights / np.linalg.norm(weights)
        self.v_group = self.v * cfg.subgroup_separation_scale
        self.direction = self.v / np.linalg.norm(self.v)
        self._sev_pos: Tuple[float, ...] = (0.50, 0.35, 0.15)
        self._sev_neg: Tuple[float, ...] = (0.85, 0.12, 0.03)
        if harm is not None:
            self._sev_pos, self._sev_neg = harm.severity_probs_positive, harm.severity_probs_negative
        self._pool: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._pool_x: Optional[np.ndarray] = None

    def _separation_for(self, g: np.ndarray) -> np.ndarray:
        """Per-case separation vector, shape (n, d)."""
        return np.where(g[:, None] == 1, self.v_group, self.v)

    def draw_severity(self, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Severity class (1 to 3) whose distribution depends on the outcome."""
        return draw_severity_from_probs(y, self._sev_pos, self._sev_neg, rng)

    def sample(self, n: int, rng: np.random.Generator, prevalence: Optional[float] = None) -> Batch:
        """Baseline (generative) sample, optionally with a different prevalence and fixed class-conditionals."""
        pi = self.cfg.prevalence if prevalence is None else prevalence
        y = (rng.random(n) < pi).astype(int)
        g = (rng.random(n) < self.cfg.subgroup_share).astype(int)
        sign = np.where(y == 1, 0.5, -0.5)[:, None]
        x = sign * self._separation_for(g) + rng.standard_normal((n, self.cfg.n_features))
        return Batch(x=x, y=y, g=g, severity=self.draw_severity(y, rng))

    def score_direction(self, x: np.ndarray, g: np.ndarray) -> np.ndarray:
        """The linear part v_g . x of the true log-odds (without the prior term)."""
        return np.einsum("ij,ij->i", x, self._separation_for(g))

    def true_logit(self, x: np.ndarray, g: np.ndarray) -> np.ndarray:
        """Exact baseline log-odds of the outcome given x and g."""
        pi = self.cfg.prevalence
        return np.log(pi / (1 - pi)) + self.score_direction(x, g)

    def sample_x_first(self, n: int, rng: np.random.Generator, mean_shift: Optional[np.ndarray] = None,
                       slope: float = 1.0, intercept: Optional[float] = None,
                       slope_by_group: Optional[Tuple[float, float]] = None,
                       intercept_by_group: Optional[Tuple[float, float]] = None) -> Batch:
        """Draw x from the baseline marginal (optionally shifted), then y from a linked risk model.

        y | x, g ~ Bernoulli(sigmoid(intercept + slope * v_g . x')). With slope 1 and the baseline intercept this is
        the true posterior, so a mean shift is a pure covariate shift. A slope below 1 weakens the outcome's
        dependence on x (concept drift) while leaving P(x) unchanged.
        """
        base = self.sample(n, rng)
        x = base.x if mean_shift is None else base.x + mean_shift
        if intercept is None:
            intercept = float(np.log(self.cfg.prevalence / (1 - self.cfg.prevalence)))
        slope_i: Union[float, np.ndarray]
        intercept_i: Union[float, np.ndarray, None]
        if slope_by_group is not None and intercept_by_group is not None:
            slope_i = np.where(base.g == 1, slope_by_group[1], slope_by_group[0])
            intercept_i = np.where(base.g == 1, intercept_by_group[1], intercept_by_group[0])
        else:
            slope_i, intercept_i = slope, intercept
        y = (rng.random(n) < sigmoid(intercept_i + slope_i * self.score_direction(x, base.g))).astype(int)
        return Batch(x=x, y=y, g=base.g, severity=self.draw_severity(y, rng))

    def pool(self) -> Tuple[np.ndarray, np.ndarray]:
        """A fixed baseline pool (score direction values and subgroup), for deterministic root finding."""
        if self._pool is None:
            batch = self.sample(POOL_SIZE, np.random.default_rng(POOL_SEED))
            self._pool = (self.score_direction(batch.x, batch.g), batch.g)
            self._pool_x = batch.x
        return self._pool

    def pool_inputs(self) -> np.ndarray:
        """Feature matrix of the fixed baseline pool."""
        self.pool()
        assert self._pool_x is not None
        return self._pool_x

    def intercept_for_prevalence(self, slope: float, group: Optional[int] = None) -> float:
        """Intercept a such that E[sigmoid(a + slope * z)] equals the baseline prevalence, optionally within one subgroup."""
        z, g = self.pool()
        if group is not None:
            z = z[g == group]
        return float(brentq(lambda a: sigmoid(a + slope * z).mean() - self.cfg.prevalence, -25.0, 25.0, xtol=1e-9))
