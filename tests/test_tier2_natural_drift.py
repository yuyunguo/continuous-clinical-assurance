"""Unit tests for `scripts/m7_tier2_natural_drift.py`'s composite ordering logic.

`add_composite_order` is the fix for a real ordering bug: `order_key` (shifted `icu_intime`) is not comparable
across patients (MIMIC-IV's de-identification shifts it independently per patient). These tests never touch a
database -- `add_composite_order` is pure pandas logic over an in-memory DataFrame.
"""

import numpy as np
import pandas as pd
import pytest

from scripts.m7_tier2_natural_drift import BUCKET_ORDER, add_composite_order


def test_composite_order_never_mixes_two_calendar_buckets(rng=np.random.default_rng(0)):
    """The property that actually matters: sorting by `combined_order` must never place a later-bucket case before
    an earlier-bucket one, regardless of `order_key`'s (unreliable) within-bucket meaning."""
    n = 300
    df = pd.DataFrame({
        "anchor_year_group": rng.choice(BUCKET_ORDER, size=n),
        "order_key": pd.to_datetime(rng.integers(0, 10_000, size=n), unit="D", origin="2100-01-01"),   # arbitrary, unordered-across-patients shifted dates
    })
    out = add_composite_order(df).sort_values("combined_order", kind="stable").reset_index(drop=True)
    bucket_rank = out["anchor_year_group"].map({b: i for i, b in enumerate(BUCKET_ORDER)})
    assert bucket_rank.is_monotonic_increasing


def test_composite_order_orders_within_a_bucket_by_order_key():
    df = pd.DataFrame({"anchor_year_group": ["2011 - 2013", "2011 - 2013", "2011 - 2013"],
                       "order_key": pd.to_datetime(["2150-06-01", "2150-01-01", "2150-03-01"])})
    out = add_composite_order(df).sort_values("combined_order", kind="stable").reset_index(drop=True)
    assert list(out["order_key"]) == list(pd.to_datetime(["2150-01-01", "2150-03-01", "2150-06-01"]))


def test_composite_order_rejects_an_unknown_bucket_label():
    df = pd.DataFrame({"anchor_year_group": ["not a real bucket"], "order_key": pd.to_datetime(["2100-01-01"])})
    with pytest.raises(ValueError, match="BUCKET_ORDER"):
        add_composite_order(df)


def test_composite_order_values_fall_in_the_expected_bucket_rank_range():
    df = pd.DataFrame({"anchor_year_group": ["2008 - 2010", "2020 - 2022"],
                       "order_key": pd.to_datetime(["2100-01-01", "2100-01-01"])})
    out = add_composite_order(df)
    assert 0 <= out.loc[out["anchor_year_group"] == "2008 - 2010", "combined_order"].iloc[0] < 1
    last_rank = len(BUCKET_ORDER) - 1
    assert last_rank <= out.loc[out["anchor_year_group"] == "2020 - 2022", "combined_order"].iloc[0] < last_rank + 1
