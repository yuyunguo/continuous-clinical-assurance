"""Unit tests for `scripts/tier2_extract_eicu_pool.py`'s pandas transformations, against stubbed database results.

No real connection is used: `pd.read_sql` is monkeypatched per test with a fixed DataFrame, so these tests never
touch a network or a real database (see `Tier2-Readiness.md`: row-level data stays on the study author's machine).
"""

import pandas as pd
import pytest

from scripts import tier2_extract_eicu_pool as tep


class _NullConn:
    """Stands in for a psycopg2 connection; `pd.read_sql` is monkeypatched, so nothing on this object is called."""


def test_extract_cohort_labels_expired_as_the_positive_class(monkeypatch):
    raw = pd.DataFrame({"patientunitstayid": [1, 2, 3], "gender": ["Female", "Male", "Female"],
                        "hospitaldischargestatus": ["Expired", "Alive", "Alive"]})
    monkeypatch.setattr(tep.pd, "read_sql", lambda *a, **k: raw)
    cohort = tep.extract_cohort(_NullConn())
    assert list(cohort["label"]) == [1, 0, 0]
    assert list(cohort["subgroup"]) == [1, 0, 1]


def test_aggregate_periodic_pivots_stat_and_feature_into_the_expected_column_names(monkeypatch):
    raw = pd.DataFrame({
        "patientunitstayid": [1, 1, 1, 2],
        "observationoffset": [10, 70, 130, 10],
        "heartrate": [80, 90, 100, 70],
        "respiration": [18, None, 20, 16],
    })
    monkeypatch.setattr(tep.pd, "read_sql", lambda *a, **k: raw)
    cohort = pd.DataFrame({"patientunitstayid": [1, 2]})
    column_map = {"vital_heart_rate": "heartrate", "vital_resp_rate": "respiration"}
    wide = tep._aggregate_periodic(_NullConn(), cohort, "eicu_crd.vitalperiodic", column_map)
    row1 = wide[wide["patientunitstayid"] == 1].iloc[0]
    assert row1["vital_heart_rate_mean"] == 90.0
    assert row1["vital_heart_rate_min"] == 80.0
    assert row1["vital_heart_rate_max"] == 100.0
    assert row1["vital_heart_rate_last"] == 100.0   # last by observationoffset, 130
    assert row1["vital_heart_rate_count"] == 3
    assert row1["vital_resp_rate_count"] == 2   # the None reading is dropped, not counted


def test_aggregate_periodic_excludes_readings_after_the_24_hour_window(monkeypatch):
    raw = pd.DataFrame({"patientunitstayid": [1, 1], "observationoffset": [10, 1441], "heartrate": [80, 999]})
    monkeypatch.setattr(tep.pd, "read_sql", lambda *a, **k: raw)
    cohort = pd.DataFrame({"patientunitstayid": [1]})
    wide = tep._aggregate_periodic(_NullConn(), cohort, "eicu_crd.vitalperiodic", {"vital_heart_rate": "heartrate"})
    assert wide.iloc[0]["vital_heart_rate_mean"] == 80.0   # the 999 reading past 1440 minutes must be excluded


def test_aggregate_periodic_leaves_a_stay_with_no_readings_all_nan(monkeypatch):
    raw = pd.DataFrame({"patientunitstayid": pd.Series([], dtype=int), "observationoffset": pd.Series([], dtype=int),
                        "heartrate": pd.Series([], dtype=float)})
    monkeypatch.setattr(tep.pd, "read_sql", lambda *a, **k: raw)
    cohort = pd.DataFrame({"patientunitstayid": [1]})
    wide = tep._aggregate_periodic(_NullConn(), cohort, "eicu_crd.vitalperiodic", {"vital_heart_rate": "heartrate"})
    assert list(wide["patientunitstayid"]) == [1]
    assert wide.iloc[0]["vital_heart_rate_mean"] != wide.iloc[0]["vital_heart_rate_mean"]   # NaN


def test_aggregate_labs_pivots_labname_into_the_expected_column_names(monkeypatch):
    raw = pd.DataFrame({
        "patientunitstayid": [1, 1, 2],
        "labresultoffset": [10, 70, 10],
        "labname": ["creatinine", "creatinine", "sodium"],
        "labresult": [1.0, 1.4, 138.0],
    })
    monkeypatch.setattr(tep.pd, "read_sql", lambda *a, **k: raw)
    cohort = pd.DataFrame({"patientunitstayid": [1, 2]})
    wide = tep._aggregate_labs(_NullConn(), cohort)
    row1 = wide[wide["patientunitstayid"] == 1].iloc[0]
    assert row1["lab_creatinine_mean"] == pytest.approx(1.2)
    assert row1["lab_creatinine_last"] == pytest.approx(1.4)   # last by labresultoffset, 70
    row2 = wide[wide["patientunitstayid"] == 2].iloc[0]
    assert row2["lab_sodium_mean"] == pytest.approx(138.0)


def test_build_pool_has_all_115_feature_columns_plus_label_and_subgroup_no_order_key(monkeypatch):
    """The pool has no `order_key` (`scope_creep_and_rescore` only draws rows at random, never windows this pool by
    its own calendar order), unlike the MIMIC extraction's output table."""
    cohort_raw = pd.DataFrame({"patientunitstayid": [1, 2], "gender": ["Female", "Male"],
                               "hospitaldischargestatus": ["Alive", "Expired"]})
    empty_base = {"patientunitstayid": pd.Series([], dtype=int), "observationoffset": pd.Series([], dtype=int)}
    empty_periodic = pd.DataFrame({**empty_base, **{c: pd.Series([], dtype=float) for c in tep.VITALPERIODIC_MAP.values()}})
    empty_aperiodic = pd.DataFrame({**empty_base, **{c: pd.Series([], dtype=float) for c in tep.VITALAPERIODIC_MAP.values()}})
    empty_lab = pd.DataFrame({"patientunitstayid": pd.Series([], dtype=int), "labresultoffset": pd.Series([], dtype=int),
                              "labname": pd.Series([], dtype=str), "labresult": pd.Series([], dtype=float)})
    calls = iter([cohort_raw, empty_periodic, empty_aperiodic, empty_lab])
    monkeypatch.setattr(tep.pd, "read_sql", lambda *a, **k: next(calls))
    table = tep.build_pool(_NullConn())
    assert "order_key" not in table.columns
    assert set(table.columns) == {"label", "subgroup", *tep.FEATURE_COLUMNS}
    assert len(table) == 2


def test_lab_name_map_has_all_13_labs_matching_mimics_lab_map():
    from scripts.tier2_extract_features import LAB_MAP
    assert set(tep.LAB_NAME_MAP) == set(LAB_MAP)


def test_vital_maps_together_cover_every_vital_except_temp_f():
    """`vital_temp_c` is mapped directly (eICU's single `temperature` field, assumed Celsius, see module docstring);
    `vital_temp_f` alone has no eICU source and is left to `RealDeployedModel`'s NaN imputation. Every other MIMIC
    vital must have an eICU source column."""
    from scripts.tier2_extract_features import VITAL_MAP
    covered = set(tep.VITALPERIODIC_MAP) | set(tep.VITALAPERIODIC_MAP)
    expected = set(VITAL_MAP) - {"vital_temp_f"}
    assert covered == expected
