"""Configuration for the Tier 2 real-data stream builder (design section `Tier2-Readiness.md`)."""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Tier2StreamConfig:
    """Column names and geometry for turning a real (or synthetic stand-in) cohort table into a `StreamData`.

    The cohort table is one row per case, with a numeric or date-like `order_column` that fixes calendar order
    (for example `icu_intime`, cast to an integer), a binary `label_column`, a binary `subgroup_column`, and the
    feature columns the deployed model is trained on.
    """

    feature_columns: Tuple[str, ...]
    window_size: int = 500                # cases per window (decision D6/D8: 500 cases)
    subgroup_column: str = "subgroup"
    order_column: str = "order_key"
    label_column: str = "label"
    target_sensitivity: float = 0.8       # matches `GeneratorConfig.target_sensitivity` (Tier 1), for a comparable operating point
    out_of_scope_column: str = "out_of_scope"   # optional boolean column (SC19 scope creep); absent means all in scope
