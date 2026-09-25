"""Schema-matched synthetic stand-in for a real Tier 2 cohort table.

Used to build and test the Tier 2 pipeline without touching MIMIC-IV or eICU. It reproduces the shape the real
extraction script (`scripts/tier2_extract_features.py`, run on the study author's machine) is expected to
produce: one row per case, an order key, a binary label, a binary subgroup, and feature columns. It makes no claim
to reproduce real clinical feature distributions.
"""

from typing import Sequence

import numpy as np
import pandas as pd


def make_synthetic_cohort(n: int, feature_columns: Sequence[str], rng: np.random.Generator,
                          prevalence: float = 0.2, subgroup_share: float = 0.5) -> pd.DataFrame:
    """A synthetic cohort table with the real schema: `order_key`, `label`, `subgroup`, and the given feature columns.

    `x | label ~ N(sign(label) * mu, I)` with a fixed per-feature effect `mu`, so the label is learnable from the
    features but not degenerate. `order_key` is 0..n-1, standing in for calendar order.
    """
    label = (rng.random(n) < prevalence).astype(int)
    subgroup = (rng.random(n) < subgroup_share).astype(int)
    mu = np.linspace(0.3, 0.6, num=len(feature_columns))
    sign = np.where(label == 1, 1.0, -1.0)[:, None]
    x = sign * mu[None, :] + rng.standard_normal((n, len(feature_columns)))
    df = pd.DataFrame(x, columns=list(feature_columns))
    df["order_key"] = np.arange(n)
    df["label"] = label
    df["subgroup"] = subgroup
    return df
