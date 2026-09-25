"""Input-only perturbations on real (or synthetic stand-in) Tier 2 windows.

Rescale (SC07), missingness (SC05/SC06), and version regression (SC17) act only on the model's inputs or on the
model itself, never on a real case's `label` — so injecting them onto a fixed, already-observed cohort never
overrides a real outcome. Outcome-affecting scenarios (SC08 concept drift, SC19 scope creep) are out of scope here:
injecting them would mean synthetically overriding real outcomes, a decision deferred by the study author
(`Tier2-Readiness.md` §3, 2026-09-22).

Progress follows the same `progress_schedule` as Tier 1 (`src.monitor_module.streams`), so an onset window `s0` and
a shape ("abrupt", "gradual", "incremental") mean the same thing in both tiers.
"""

from typing import Sequence, Union

import numpy as np
import pandas as pd

from src.monitor_module.streams import progress_schedule

from .config import Tier2StreamConfig
from .real_model import RealDeployedModel

FloatOrArray = Union[float, np.ndarray]


def ramp_ratio(level_value: float, progress: FloatOrArray) -> FloatOrArray:
    """Interpolate a ratio-type level from 1 (no effect) to its full value. Mirrors `Perturbation.ramp_ratio`."""
    return 1.0 + progress * (level_value - 1.0)


def ramp_amount(level_value: float, progress: FloatOrArray) -> FloatOrArray:
    """Scale an amount-type level from 0 (no effect) to its full value. Mirrors `Perturbation.ramp_amount`."""
    return progress * level_value


def window_progress(n_windows: int, n_baseline: int, shape: str, s0: int, ramp_windows: int) -> np.ndarray:
    """Perturbation progress per absolute window: 0 for every baseline window, `progress_schedule` after."""
    n_monitored = n_windows - n_baseline
    if not (0 <= s0 < max(n_monitored, 1)) or n_monitored <= 0:
        raise ValueError(f"s0 ({s0}) must be a valid monitored-window index; there are {n_monitored} monitored windows")
    monitored_progress = progress_schedule(shape, s0, n_monitored, ramp_windows)
    return np.concatenate([np.zeros(n_baseline), monitored_progress])


def _row_progress(windowed: pd.DataFrame, n_baseline: int, shape: str, s0: int, ramp_windows: int) -> np.ndarray:
    """Per-row progress, looked up from each row's `window`."""
    n_windows = int(windowed["window"].nunique())
    progress = window_progress(n_windows, n_baseline, shape, s0, ramp_windows)
    return progress[windowed["window"].to_numpy()]


def rescale_and_rescore(windowed: pd.DataFrame, cfg: Tier2StreamConfig, model: RealDeployedModel, column: str,
                        level_value: float, shape: str, s0: int, n_baseline: int, ramp_windows: int) -> pd.DataFrame:
    """A unit or scale change in `column` (SC07): multiply it by the ramped ratio, then rescore with `model`."""
    del cfg   # accepted for interface symmetry with the other perturbations here; not needed directly
    progress = _row_progress(windowed, n_baseline, shape, s0, ramp_windows)
    ratio = ramp_ratio(level_value, progress)
    modified = windowed.copy()
    modified[column] = windowed[column].to_numpy() * ratio
    return model.apply(modified)


def missingness_and_rescore(windowed: pd.DataFrame, cfg: Tier2StreamConfig, model: RealDeployedModel, column: str,
                            level_value: float, shape: str, s0: int, n_baseline: int, ramp_windows: int,
                            rng: np.random.Generator) -> pd.DataFrame:
    """Missing completely at random in `column` (SC05/SC06), imputed with the training mean, then rescored."""
    del cfg
    progress = _row_progress(windowed, n_baseline, shape, s0, ramp_windows)
    prob = ramp_amount(level_value, progress)
    mask = rng.random(len(windowed)) < prob
    modified = windowed.copy()
    modified.loc[mask, column] = model.impute_means[column]
    return model.apply(modified)


def group_missingness_and_rescore(windowed: pd.DataFrame, cfg: Tier2StreamConfig, model: RealDeployedModel,
                                  columns: Sequence[str], level_value: float, shape: str, s0: int, n_baseline: int,
                                  ramp_windows: int, rng: np.random.Generator) -> pd.DataFrame:
    """Missing completely at random, simultaneously, in every column of `columns` (SC05/SC06, generalized).

    A single-column real missingness diagnostic on the sepsis extraction reached only a 1.04 D18 ratio even at
    prob=1.0 (always missing) -- the same dilution a single rescaled column showed (`level_solving.py`'s docstring).
    A group of correlated columns going missing together -- e.g. one vital's mean/min/max/last/count, as one sensor
    or device dropping out would in practice, not five independent failures -- reached 1.25, closer to (though still
    short of) Tier 1's "medium" target. The SAME per-case mask applies to every column in `columns`, so an affected
    case loses the whole group at once; `missingness_and_rescore` is kept as the single-column special case (masking
    `[column]` here would behave identically, but the existing call sites and tests are left on the simpler name).
    """
    del cfg
    progress = _row_progress(windowed, n_baseline, shape, s0, ramp_windows)
    prob = ramp_amount(level_value, progress)
    mask = rng.random(len(windowed)) < prob
    modified = windowed.copy()
    for column in columns:
        modified.loc[mask, column] = model.impute_means[column]
    return model.apply(modified)


def scope_creep_and_rescore(windowed: pd.DataFrame, cfg: Tier2StreamConfig, model: RealDeployedModel,
                            out_of_scope_pool: pd.DataFrame, level_value: float, shape: str, s0: int, n_baseline: int,
                            ramp_windows: int, rng: np.random.Generator) -> pd.DataFrame:
    """A growing share of each monitored window's cases is replaced with real cases drawn from `out_of_scope_pool`
    (SC19, scope creep) -- e.g. real eICU cases, a genuinely different hospital system a MIMIC-trained `model` has
    never seen. Every replaced case keeps its OWN real label, subgroup, and features: nothing about any case's real
    outcome is invented, unlike SC08/SC19's usual injection mechanism (Tier2-Readiness.md SS3/SS10). Adds an
    `out_of_scope` column (True for a replaced case) so `build_stream_data` can wire the real C3 indicator, instead
    of Tier 2's previous always-in-scope placeholder.

    `out_of_scope_pool` must carry `cfg.feature_columns`, `cfg.label_column`, and `cfg.subgroup_column` in the same
    units/encoding as `windowed` (its own real values) -- it is rescored with `model` here, once, regardless of how
    many replicates draw from it.
    """
    n_windows = int(windowed["window"].nunique())
    progress = window_progress(n_windows, n_baseline, shape, s0, ramp_windows)
    scored_pool = model.apply(out_of_scope_pool)
    replace_columns = list(cfg.feature_columns) + [cfg.label_column, cfg.subgroup_column, "p_hat", "decision"]

    modified = windowed.copy()
    modified[cfg.out_of_scope_column] = False
    for w, group in modified.groupby("window", sort=False):
        share = float(ramp_amount(level_value, progress[w]))
        k = round(share * len(group))
        if k <= 0:
            continue
        replace_index = rng.choice(group.index.to_numpy(), size=k, replace=False)
        draw_positions = rng.integers(0, len(scored_pool), size=k)
        modified.loc[replace_index, replace_columns] = scored_pool.iloc[draw_positions][replace_columns].to_numpy()
        modified.loc[replace_index, cfg.out_of_scope_column] = True
    return modified


def version_regression_and_rescore(windowed: pd.DataFrame, cfg: Tier2StreamConfig, model: RealDeployedModel,
                                   level_value: float, shape: str, s0: int, n_baseline: int, ramp_windows: int,
                                   seed: int = 0) -> pd.DataFrame:
    """The deployed model is replaced by a regressed version from `s0` (SC17); the real inputs are unchanged.

    The fraction replaced ramps per window (all cases in a window share one fraction, since `s0` is a window index),
    so each distinct window is scored with its own regressed model, sharing one fixed regression direction (`seed`).
    """
    del cfg
    n_windows = int(windowed["window"].nunique())
    progress = window_progress(n_windows, n_baseline, shape, s0, ramp_windows)
    parts = []
    for w, group in windowed.groupby("window", sort=False):
        fraction = float(ramp_amount(level_value, progress[w]))
        parts.append(model.with_regression(fraction, seed=seed).apply(group))
    return pd.concat(parts).sort_index()
