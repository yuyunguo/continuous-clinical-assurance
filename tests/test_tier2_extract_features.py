"""Unit tests for `scripts/tier2_extract_features.py`'s pandas transformations, against stubbed database results.

No real connection is used: `pd.read_sql` is monkeypatched per test with a fixed DataFrame, so these tests never
touch a network or a real database (see `Tier2-Readiness.md`: row-level data stays on the study author's machine).
"""

import pandas as pd
import pytest

from scripts import tier2_extract_features as tef


class _NullConn:
    """Stands in for a psycopg2 connection; `pd.read_sql` is monkeypatched, so nothing on this object is called."""


def test_aggregate_pivots_stat_and_feature_into_the_expected_column_names(monkeypatch):
    raw = pd.DataFrame({
        "stay_id": [1, 1, 1, 2],
        "charttime": pd.to_datetime(["2020-01-01 01:00", "2020-01-01 02:00", "2020-01-01 03:00", "2020-01-01 01:00"]),
        "itemid": [220045, 220045, 220210, 220045],   # two heart-rate readings and one resp-rate reading for stay 1
        "valuenum": [80.0, 90.0, 18.0, 70.0],
    })
    monkeypatch.setattr(tef.pd, "read_sql", lambda *a, **k: raw)
    cohort = pd.DataFrame({"stay_id": [1, 2], "icu_intime": pd.to_datetime(["2020-01-01 00:00"] * 2),
                           "prediction_time": pd.to_datetime(["2020-01-02 00:00"] * 2)})
    wide = tef._aggregate(_NullConn(), cohort, "mimic_icu.chartevents", "stay_id", tef.VITAL_MAP, tef.ALL_VITAL_IDS)
    row1 = wide[wide["stay_id"] == 1].iloc[0]
    assert row1["vital_heart_rate_mean"] == 85.0
    assert row1["vital_heart_rate_min"] == 80.0
    assert row1["vital_heart_rate_max"] == 90.0
    assert row1["vital_heart_rate_last"] == 90.0
    assert row1["vital_heart_rate_count"] == 2
    assert row1["vital_resp_rate_mean"] == 18.0


def test_aggregate_excludes_readings_at_or_after_the_prediction_time(monkeypatch):
    raw = pd.DataFrame({
        "stay_id": [1, 1],
        "charttime": pd.to_datetime(["2020-01-01 12:00", "2020-01-02 00:00"]),   # second reading is exactly at the 24h cutoff
        "itemid": [220045, 220045],
        "valuenum": [80.0, 999.0],
    })
    monkeypatch.setattr(tef.pd, "read_sql", lambda *a, **k: raw)
    cohort = pd.DataFrame({"stay_id": [1], "icu_intime": pd.to_datetime(["2020-01-01 00:00"]),
                           "prediction_time": pd.to_datetime(["2020-01-02 00:00"])})
    wide = tef._aggregate(_NullConn(), cohort, "mimic_icu.chartevents", "stay_id", tef.VITAL_MAP, tef.ALL_VITAL_IDS)
    assert wide.iloc[0]["vital_heart_rate_mean"] == 80.0   # the 999.0 reading at the cutoff must be excluded


def test_aggregate_leaves_a_stay_with_no_readings_all_nan(monkeypatch):
    raw = pd.DataFrame({"stay_id": pd.Series([], dtype=int), "charttime": pd.Series([], dtype="datetime64[ns]"),
                        "itemid": pd.Series([], dtype=int), "valuenum": pd.Series([], dtype=float)})
    monkeypatch.setattr(tef.pd, "read_sql", lambda *a, **k: raw)
    cohort = pd.DataFrame({"stay_id": [1], "icu_intime": pd.to_datetime(["2020-01-01 00:00"]),
                           "prediction_time": pd.to_datetime(["2020-01-02 00:00"])})
    wide = tef._aggregate(_NullConn(), cohort, "mimic_icu.chartevents", "stay_id", tef.VITAL_MAP, tef.ALL_VITAL_IDS)
    assert list(wide["stay_id"]) == [1]


def test_label_mortality_reads_the_cohort_column_directly():
    cohort = pd.DataFrame({"hadm_id": [10, 11, 12], "mortality": [0, 1, 0]})
    labels = tef.label_mortality(cohort)
    assert labels.loc[11] == 1 and labels.loc[10] == 0


def test_label_sepsis3_flags_only_hadm_ids_whose_stay_has_sepsis3(monkeypatch):
    hits = pd.DataFrame({"stay_id": [201]})   # only stay 201 meets Sepsis-3 criteria
    monkeypatch.setattr(tef.pd, "read_sql", lambda *a, **k: hits)
    cohort = pd.DataFrame({"hadm_id": [10, 11, 12], "stay_id": [200, 201, 202]})
    labels = tef.label_sepsis3(_NullConn(), cohort)
    assert labels.loc[10] == 0 and labels.loc[11] == 1 and labels.loc[12] == 0


def test_label_readmission_flags_a_return_within_30_days_and_excludes_deaths(monkeypatch):
    adm = pd.DataFrame({
        "hadm_id": [10, 11, 12, 20],
        "subject_id": [1, 1, 3, 2],
        "hospital_expire_flag": [0, 0, 0, 1],
        "dischtime": pd.to_datetime(["2020-01-01", "2020-06-01", "2020-01-01", "2020-01-01"]),
        "next_admit": pd.to_datetime(["2020-01-15", pd.NaT, "2020-02-15", "2020-01-05"]),
        # subject 1 returns within 30 days; subject 3 returns at day 45 (past the 30-day cutoff); subject 2 died (excluded)
    })
    monkeypatch.setattr(tef.pd, "read_sql", lambda *a, **k: adm)
    cohort = pd.DataFrame({"hadm_id": [10, 11, 12, 20], "subject_id": [1, 1, 3, 2]})
    labels = tef.label_readmission(_NullConn(), cohort)
    assert labels.loc[10] == 1        # readmitted within 30 days
    assert labels.loc[11] == 0        # no next admission
    assert labels.loc[12] == 0        # readmitted, but after 30 days
    assert 20 not in labels.index     # excluded: died on the index admission


def test_task_choices_cover_the_d8_order():
    assert set(tef.LABEL_FNS) == {"sepsis", "readmission", "mortality"}


def test_feature_columns_have_115_entries_matching_the_papers_count():
    assert len(tef.FEATURE_COLUMNS) == 115


def test_non_feature_columns_includes_anchor_year_group():
    """`anchor_year_group` must be retained (not silently dropped, and not later mistaken for a model feature by
    any downstream `c not in NON_FEATURE_COLUMNS` filter) -- it is the de-identification-safe field a valid
    cross-patient natural-drift test needs (`scripts/m7_tier2_natural_drift.py`); `order_key` alone is not one."""
    assert "anchor_year_group" in tef.NON_FEATURE_COLUMNS
    assert set(tef.NON_FEATURE_COLUMNS).isdisjoint(tef.FEATURE_COLUMNS)


def test_extract_cohort_selects_anchor_year_group(monkeypatch):
    raw = pd.DataFrame({"stay_id": [1], "hadm_id": [10], "subject_id": [100],
                        "icu_intime": pd.to_datetime(["2020-01-01"]), "prediction_time": pd.to_datetime(["2020-01-02"]),
                        "mortality": [0], "gender": ["F"], "anchor_year_group": ["2020 - 2022"]})
    monkeypatch.setattr(tef.pd, "read_sql", lambda *a, **k: raw)
    cohort = tef.extract_cohort(_NullConn())
    assert cohort["anchor_year_group"].iloc[0] == "2020 - 2022"


@pytest.mark.parametrize("stat", ["mean", "min", "max", "last", "count"])
def test_every_vital_and_lab_has_all_five_aggregate_statistics(stat):
    expected = {f"{feat}_{stat}" for feat in list(tef.VITAL_MAP) + list(tef.LAB_MAP)}
    assert expected <= set(tef.FEATURE_COLUMNS)


def test_build_table_output_has_anchor_year_group_and_order_key_not_confused_with_features(monkeypatch):
    """`build_table`'s output columns must be exactly `NON_FEATURE_COLUMNS` plus `FEATURE_COLUMNS` -- neither
    `order_key` nor the newly-added `anchor_year_group` should ever end up counted as a model feature."""
    cohort_raw = pd.DataFrame({"stay_id": [1], "hadm_id": [10], "subject_id": [100],
                               "icu_intime": pd.to_datetime(["2020-01-01"]),
                               "prediction_time": pd.to_datetime(["2020-01-02"]), "mortality": [0], "gender": ["F"],
                               "anchor_year_group": ["2020 - 2022"]})
    empty_events = pd.DataFrame({"stay_id": pd.Series([], dtype=int), "charttime": pd.Series([], dtype="datetime64[ns]"),
                                 "itemid": pd.Series([], dtype=int), "valuenum": pd.Series([], dtype=float)})
    empty_labs = pd.DataFrame({"hadm_id": pd.Series([], dtype=int), "charttime": pd.Series([], dtype="datetime64[ns]"),
                               "itemid": pd.Series([], dtype=int), "valuenum": pd.Series([], dtype=float)})
    calls = iter([cohort_raw, empty_events, empty_labs])
    monkeypatch.setattr(tef.pd, "read_sql", lambda *a, **k: next(calls))
    table = tef.build_table(_NullConn(), "mortality")
    assert set(table.columns) == set(tef.NON_FEATURE_COLUMNS) | set(tef.FEATURE_COLUMNS)
    assert table["anchor_year_group"].iloc[0] == "2020 - 2022"
