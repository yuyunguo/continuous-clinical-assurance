"""Frozen configuration dataclasses for the Phase II simulation, loaded from Hydra-style YAML."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple, Union, cast

from omegaconf import OmegaConf

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "run" / "conf" / "experiment" / "phase2_defaults.yaml"


@dataclass(frozen=True)
class GeneratorConfig:
    """Tier 1 data-generating process and deployed-model settings."""

    n_features: int
    prevalence: float
    separation: float
    feature_weights: Tuple[float, ...]
    subgroup_share: float
    subgroup_separation_scale: float
    train_size: int
    target_sensitivity: float


@dataclass(frozen=True)
class StreamConfig:
    """Stream geometry (decision D6)."""

    cases_per_window: int
    baseline_windows: int
    monitored_windows: int
    label_lag: int
    horizon: int
    s0_low: int
    s0_high_abrupt: int
    s0_high_ramped: int
    ramp_windows: int


@dataclass(frozen=True)
class HarmConfig:
    """Severity classes, weights, and the meaningful-deviation threshold (decisions D3 and D4)."""

    severity_cutoff: int
    weights: Tuple[float, ...]
    severity_probs_positive: Tuple[float, ...]
    severity_probs_negative: Tuple[float, ...]
    kappa: float
    min_subgroup_share: float = 0.05
    fn_factor: float = 1.0   # extra weight of a missed deterioration (false negative) relative to an unneeded alert of the same class


@dataclass(frozen=True)
class ReviewerConfig:
    """Reviewer model. An assumption, not a measurement (design section 11)."""

    coverage: float
    beta0: float
    beta_confidence: float
    beta_severity: float
    automation_bias: float = 0.0   # probability of accepting an output whatever its quality
    false_reject: float = 0.02     # probability of rejecting a correct output that was reviewed
    latency_median: float = 1.0    # median review latency, in arbitrary time units
    latency_sigma: float = 0.6     # log-scale spread of the lognormal review latency
    deadline: float = float("inf")  # point of no return, in the same units. Infinite: every review is in time


@dataclass(frozen=True)
class BudgetConfig:
    """False-alarm budget (decision D2)."""

    false_alarms_per_1000_windows: float


@dataclass(frozen=True)
class RoutingConfig:
    """CCA routing configuration (decision D7)."""

    persistence_k: int
    hysteresis_j: int
    corroboration: bool


@dataclass(frozen=True)
class BandsConfig:
    """Acceptance bands for milestone M1, fixed before M1 was first run."""

    seeds: int
    ece_max: float
    auroc_min: float
    auroc_max: float
    sensitivity_tolerance: float


@dataclass(frozen=True)
class TolerancesConfig:
    """Tolerances for population-level property tests (milestone M2)."""

    population_cases: int
    rate_abs: float
    auroc_abs: float
    input_mean_abs: float


@dataclass(frozen=True)
class WorkflowConfig:
    """Workflow model (design section 11). Placeholders, not measurements."""

    p_deliver: float = 0.98                    # probability an escalation is delivered to the responsible clinician
    tat_median: float = 1.0                    # median turnaround, in arbitrary time units
    tat_sigma: float = 0.5                     # log-scale spread of the lognormal turnaround
    tat_limit: float = 4.0                     # service limit (VOE), in the same units. Later deliveries count as late
    time_critical_share: float = 0.2           # share of true alerts on a pathway where delay causes harm
    completion: float = 0.98                   # probability that a workflow with an alert runs to completion
    abandon_after_failed_escalation: float = 0.5   # extra loss of completion when a required escalation fails
    verify: float = 0.5                        # probability that downstream steps catch an AI error the reviewer missed
    sentinel_severity: int = 3                 # severity class counted by the sentinel-event rate (O6)
    action_base: float = 0.7                   # share of alert messages that are acted on at baseline (O3 action rate)
    fatigue: float = 0.15                      # the action rate falls as base / (1 + fatigue x (duplication - 1))
    fatigue_bias: float = 0.0                  # extra automation bias from duplicated alerts, as bias x (1 - 1 / duplication). Off in the primary: no measured size


@dataclass(frozen=True)
class ContextConfig:
    """Context model (design section 11). Placeholders, not measurements."""

    baseline_out_of_scope: float = 0.02   # share of invocations that are off-indication at baseline, with no effect on the data or the model


@dataclass(frozen=True)
class Phase2Config:
    """Complete protocol configuration."""

    seed: int
    generator: GeneratorConfig
    stream: StreamConfig
    harm: HarmConfig
    reviewer: ReviewerConfig
    budget: BudgetConfig
    routing: RoutingConfig
    m1_bands: BandsConfig
    tolerances: TolerancesConfig
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    context: ContextConfig = field(default_factory=ContextConfig)


def _tuples(section: Dict[str, Any]) -> Dict[str, Any]:
    """Convert YAML lists to tuples so that frozen dataclasses stay hashable."""
    return {k: tuple(v) if isinstance(v, list) else v for k, v in section.items()}


def scenarios_path() -> Path:
    """The scenario catalog: `PHASE2_SCENARIOS` if set (a D3 sensitivity scheme), otherwise `phase2/scenarios.yaml`."""
    return Path(os.environ.get("PHASE2_SCENARIOS", DEFAULT_CONFIG.parents[3] / "phase2" / "scenarios.yaml"))


def results_dir() -> Path:
    """Where stored replicates and headline results go: `PHASE2_RESULTS` if set, otherwise `phase2/results`. Created if missing."""
    path = Path(os.environ.get("PHASE2_RESULTS", DEFAULT_CONFIG.parents[3] / "phase2" / "results"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def report_path(name: str) -> Path:
    """Where a report is written: the results folder when `PHASE2_RESULTS` is set (a sensitivity scheme), otherwise `phase2/`."""
    return results_dir() / name if "PHASE2_RESULTS" in os.environ else DEFAULT_CONFIG.parents[3] / "phase2" / name


def false_alarm_budget() -> float:
    """The false-alarm budget (decision D2), in alert episodes per 1,000 monitored windows: `M5_BUDGET` if set (the budget sweep), otherwise the config."""
    return float(os.environ["M5_BUDGET"]) if "M5_BUDGET" in os.environ else load_config().budget.false_alarms_per_1000_windows


def load_config(path: Union[str, Path, None] = None) -> Phase2Config:
    """Load the protocol configuration.

    Args:
        path: YAML file with the sections of `Phase2Config`. Defaults to `PHASE2_CONFIG` if set, otherwise the protocol defaults.

    Returns:
        Immutable configuration.

    Raises:
        FileNotFoundError: When the file does not exist.
        TypeError: When a section has missing or unexpected keys.
    """
    chosen = path if path is not None else os.environ.get("PHASE2_CONFIG", DEFAULT_CONFIG)
    raw = cast(Dict[str, Any], OmegaConf.to_container(OmegaConf.load(chosen), resolve=True))
    return Phase2Config(
        seed=int(raw["seed"]),
        generator=GeneratorConfig(**_tuples(raw["generator"])),
        stream=StreamConfig(**raw["stream"]),
        harm=HarmConfig(**_tuples(raw["harm"])),
        reviewer=ReviewerConfig(**raw["reviewer"]),
        budget=BudgetConfig(**raw["budget"]),
        routing=RoutingConfig(**raw["routing"]),
        m1_bands=BandsConfig(**raw["m1_bands"]),
        tolerances=TolerancesConfig(**raw["tolerances"]),
        workflow=WorkflowConfig(**raw.get("workflow", {})),
        context=ContextConfig(**raw.get("context", {})),
    )
