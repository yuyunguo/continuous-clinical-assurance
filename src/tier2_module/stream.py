"""Turns a windowed, scored cohort table into a `StreamData`, the shape the Tier 1 monitors and evaluators consume.

Severity is drawn from the same elicited class-conditional distribution as Tier 1 (`HarmConfig`), not measured in
the real data (no clinician severity label exists per real case). This is a modeling choice, not a data fact: see
`Tier2-Readiness.md`.

`out_of_scope` (SC19, scope creep) is optional: if `cfg.out_of_scope_column` is present in `windowed` (set by
`scope_creep_and_rescore` in `perturb.py`, marking real cases drawn from a genuinely out-of-scope population, e.g.
eICU, as `True`), it is wired into `StreamData` so the C3 indicator gets real content instead of Tier 2's previous
always-in-scope placeholder (`src.monitor_module.statistics.window_statistics`'s `C3` falls back to a constant 1.0
when `stream.out_of_scope is None`). Absent, every case is treated as in scope, matching all prior Tier 2 behavior.
"""

import numpy as np
import pandas as pd

from src.monitor_module.streams import StreamData
from src.sim_module.config import HarmConfig
from src.sim_module.generator import draw_severity_from_probs

from .config import Tier2StreamConfig


def build_stream_data(windowed: pd.DataFrame, cfg: Tier2StreamConfig, harm: HarmConfig, n_baseline: int,
                      rng: np.random.Generator) -> StreamData:
    """Build a `StreamData` from a cohort table already passed through `assign_windows` and a fitted model's `apply`.

    Requires `windowed` to have exactly `n_windows * cfg.window_size` rows (as `assign_windows` produces) and the
    `p_hat`/`decision` columns a `RealDeployedModel.apply` call adds.
    """
    m = cfg.window_size
    if len(windowed) % m != 0:
        raise ValueError(f"windowed cohort has {len(windowed)} rows, not a multiple of the window size {m}; call assign_windows first")
    w = len(windowed) // m
    if n_baseline < 0 or n_baseline > w:
        raise ValueError(f"n_baseline ({n_baseline}) must be between 0 and the number of windows ({w})")
    ordered = windowed.sort_values(["window", cfg.order_column], kind="stable")

    y = ordered[cfg.label_column].to_numpy().reshape(w, m)
    g = ordered[cfg.subgroup_column].to_numpy().reshape(w, m)
    decision = ordered["decision"].to_numpy().reshape(w, m)
    reported_p = ordered["p_hat"].to_numpy().reshape(w, m)
    x_model = ordered[list(cfg.feature_columns)].to_numpy().reshape(w, m, len(cfg.feature_columns))
    severity = draw_severity_from_probs(y.reshape(-1), harm.severity_probs_positive, harm.severity_probs_negative, rng).reshape(w, m)
    out_of_scope = ordered[cfg.out_of_scope_column].to_numpy().astype(bool).reshape(w, m) if cfg.out_of_scope_column in ordered.columns else None

    return StreamData(y=y, g=g, severity=severity, decision=decision, reported_p=reported_p, x_model=x_model,
                      n_baseline=n_baseline, s0=None, progress=np.zeros(w - n_baseline), out_of_scope=out_of_scope)
