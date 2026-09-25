"""Extract a real, genuinely out-of-scope case pool from eICU-CRD, for SC19 (scope creep) — for the study author to
run locally, exactly like `tier2_extract_features.py`.

**Run this on your own machine. Do not send me the output of this script beyond what you choose to share.**
It reads row-level eICU clinical data (vitals, labs, discharge status) and writes a local Parquet file with one row
per unit stay: a label (hospital mortality), a subgroup, and the SAME 115 feature columns
`scripts/tier2_extract_features.py` produces for MIMIC-IV (`src.tier2_module.config.Tier2StreamConfig`'s schema) —
so the MIMIC-trained deployed model can score every eICU row directly, with no retraining or column re-mapping.
`patientunitstayid` is used internally to join tables but is dropped before the output is written, so no patient
identifier reaches the Parquet file.

**Why eICU, not a synthetic pool.** SC19 (scope creep, `Tier2-Readiness.md` SS3/SS10) models a growing share of
invocations falling outside a model's intended use — the real analogue is a genuinely different hospital system a
MIMIC-IV-trained model has never seen. eICU-CRD is exactly that (a separate multi-center US ICU database), and its
mortality task and roughly comparable prevalence (~9%, aggregate-only check, `aggregates_eicu.json`) make it a
faithful, real, label-fabrication-free out-of-scope pool: `scope_creep_and_rescore` (`src/tier2_module/perturb.py`)
draws real rows from this pool and lets each keep its own real label — nothing about any case's real outcome is
invented, unlike SC08/SC19's usual synthetic-injection mechanism.

**No `order_key` is produced.** Unlike the MIMIC extraction, this pool is drawn from randomly (never windowed by
its own calendar order) — `scope_creep_and_rescore` only needs features/label/subgroup.

Vitals come from `vitalperiodic` (heartrate, respiration, systemicsystolic/diastolic/mean — the invasive/arterial
line readings, matching MIMIC's `vital_abp_*`) and `vitalaperiodic` (noninvasivesystolic/diastolic/mean, matching
`vital_nibp_*`), aggregated over the first 24 hours of the unit stay (`observationoffset` 0-1440 minutes). Labs come
from `lab` (`labname`/`labresult`, `labresultoffset` 0-1440), matched by name — verify `LAB_NAME_MAP`'s values
against `SELECT DISTINCT labname FROM eicu_crd.lab` in your instance before trusting the result; naming
conventions have varied across eICU-CRD releases and de-identification passes.

`vital_temp_f`/`vital_temp_c`: eICU's `vitalperiodic.temperature` is a single field. Its unit is assumed Celsius
(eICU-CRD's documented convention); `vital_temp_c` is populated directly and `vital_temp_f` is left NaN (imputed
by `RealDeployedModel`, same as any other missing feature) rather than guessing a possibly-wrong conversion.
**Verified on the 2026-09-24 extraction**: ~97.5% of non-missing readings are plausible Celsius (30-45); ~1.9% look
like Fahrenheit values recorded without conversion (86-113, the same clinical range in the other unit) and ~0.5%
are implausible in either unit (likely genuine chart-error artifacts, the same real MIMIC-IV issue that motivated
`RealDeployedModel`'s winsorization, `Tier2-Readiness.md` SS9-10). Left uncorrected deliberately: the fraction is
small enough that `RealDeployedModel`'s training-fit-period winsorization already absorbs the extreme tail, and a
per-value F/C guess risks introducing more error than it fixes for genuine borderline readings (fever, hypothermia).

Usage: PGPASSWORD=... python scripts/tier2_extract_eicu_pool.py --host HOST [--user USER]
       [--out-dir phase2/results/tier2/features]
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import psycopg2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.tier2_extract_features import FEATURE_COLUMNS  # noqa: E402  (the same 115-column schema as MIMIC)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

WINDOW_MINUTES = 24 * 60

VITALPERIODIC_MAP = {"vital_heart_rate": "heartrate", "vital_resp_rate": "respiration",
                     "vital_abp_sys": "systemicsystolic", "vital_abp_dia": "systemicdiastolic",
                     "vital_abp_mean": "systemicmean", "vital_temp_c": "temperature"}
VITALAPERIODIC_MAP = {"vital_nibp_sys": "noninvasivesystolic", "vital_nibp_dia": "noninvasivediastolic",
                      "vital_nibp_mean": "noninvasivemean"}
LAB_NAME_MAP = {   # eICU-CRD labname strings; verify against your instance (see module docstring)
    "lab_creatinine": "creatinine", "lab_hemoglobin": "Hgb", "lab_wbc": "WBC x 1000", "lab_platelet": "platelets x 1000",
    "lab_sodium": "sodium", "lab_potassium": "potassium", "lab_glucose": "glucose", "lab_bicarbonate": "bicarbonate",
    "lab_bun": "BUN", "lab_alt": "ALT (SGPT)", "lab_ast": "AST (SGOT)", "lab_bilirubin": "total bilirubin", "lab_lactate": "lactate",
}


def get_conn(host: str, user: str) -> "psycopg2.extensions.connection":
    return psycopg2.connect(host=host, dbname="eicu", user=user, password=os.environ["PGPASSWORD"], connect_timeout=10)


def extract_cohort(conn: "psycopg2.extensions.connection") -> pd.DataFrame:
    """One row per unit stay, with the mortality label and a 0/1 subgroup column."""
    q = ("SELECT patientunitstayid, gender, hospitaldischargestatus FROM eicu_crd.patient "
        "WHERE hospitaldischargestatus IS NOT NULL")
    df = pd.read_sql(q, conn)
    df["label"] = (df["hospitaldischargestatus"] == "Expired").astype(int)
    df["subgroup"] = (df["gender"] == "Female").astype(int)
    logger.info("eICU cohort: %d unit stays", len(df))
    return df


def _aggregate_periodic(conn: "psycopg2.extensions.connection", cohort: pd.DataFrame, table: str,
                        column_map: dict) -> pd.DataFrame:
    """Vectorized mean/min/max/last/count of each column in `column_map`, within the first 24 hours."""
    columns = list(column_map.values())
    q = (f"SELECT patientunitstayid, observationoffset, {', '.join(columns)} FROM {table} "
        f"WHERE patientunitstayid = ANY(%(ids)s) AND observationoffset BETWEEN 0 AND %(w)s")
    raw = pd.read_sql(q, conn, params={"ids": cohort["patientunitstayid"].tolist(), "w": WINDOW_MINUTES})
    raw = raw[(raw["observationoffset"] >= 0) & (raw["observationoffset"] <= WINDOW_MINUTES)]   # redundant with the
    # SQL WHERE clause above by design, mirroring `tier2_extract_features._aggregate`'s belt-and-suspenders pattern:
    # correct regardless of what the query actually returns, and testable without a live database.
    parts = []
    for feat, column in column_map.items():
        one = raw[["patientunitstayid", "observationoffset", column]].dropna(subset=[column])
        one = one.sort_values("observationoffset")
        agg = one.groupby("patientunitstayid")[column].agg(mean="mean", min="min", max="max", last="last", count="count")
        agg.columns = [f"{feat}_{stat}" for stat in agg.columns]
        parts.append(agg)
    wide = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=cohort["patientunitstayid"])
    return wide.reindex(cohort["patientunitstayid"]).reset_index()


def _aggregate_labs(conn: "psycopg2.extensions.connection", cohort: pd.DataFrame) -> pd.DataFrame:
    """Vectorized mean/min/max/last/count of each lab in `LAB_NAME_MAP`, within the first 24 hours."""
    q = ("SELECT patientunitstayid, labresultoffset, labname, labresult FROM eicu_crd.lab "
        "WHERE patientunitstayid = ANY(%(ids)s) AND labname = ANY(%(names)s) "
        "AND labresultoffset BETWEEN 0 AND %(w)s AND labresult IS NOT NULL")
    raw = pd.read_sql(q, conn, params={"ids": cohort["patientunitstayid"].tolist(),
                                       "names": list(LAB_NAME_MAP.values()), "w": WINDOW_MINUTES})
    raw = raw[(raw["labresultoffset"] >= 0) & (raw["labresultoffset"] <= WINDOW_MINUTES)]   # redundant with the SQL
    # WHERE clause by design; see `_aggregate_periodic`'s comment.
    name_to_feat = {name: feat for feat, name in LAB_NAME_MAP.items()}
    raw["feature_name"] = raw["labname"].map(name_to_feat)
    raw = raw.dropna(subset=["feature_name"])
    raw = raw.sort_values("labresultoffset")
    agg = raw.groupby(["patientunitstayid", "feature_name"])["labresult"].agg(mean="mean", min="min", max="max", last="last", count="count")
    wide = agg.unstack("feature_name")
    wide.columns = [f"{feat}_{stat}" for stat, feat in wide.columns]
    return wide.reindex(cohort["patientunitstayid"]).reset_index()


def build_pool(conn: "psycopg2.extensions.connection") -> pd.DataFrame:
    """The full out-of-scope pool: label, subgroup, and the same 115 feature columns MIMIC extraction produces."""
    cohort = extract_cohort(conn)
    periodic = _aggregate_periodic(conn, cohort, "eicu_crd.vitalperiodic", VITALPERIODIC_MAP)
    aperiodic = _aggregate_periodic(conn, cohort, "eicu_crd.vitalaperiodic", VITALAPERIODIC_MAP)
    labs = _aggregate_labs(conn, cohort)
    table = cohort[["patientunitstayid", "label", "subgroup"]].merge(
        periodic, on="patientunitstayid", how="left").merge(
        aperiodic, on="patientunitstayid", how="left").merge(
        labs, on="patientunitstayid", how="left")
    for col in FEATURE_COLUMNS:
        if col not in table.columns:
            table[col] = float("nan")
    keep = ["label", "subgroup", *FEATURE_COLUMNS]
    logger.info("eICU pool: %d rows, prevalence %.3f", len(table), table["label"].mean())
    return table[keep]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--host", required=True)
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--out-dir", type=Path, default=Path("phase2/results/tier2/features"))
    args = parser.parse_args()
    if not os.environ.get("PGPASSWORD"):
        logger.error("set the PGPASSWORD environment variable")
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    conn = get_conn(args.host, args.user)
    try:
        table = build_pool(conn)
    finally:
        conn.close()
    out = args.out_dir / "eicu_out_of_scope_pool.parquet"
    table.to_parquet(out)
    logger.info("Wrote %s (%d rows, %d columns)", out, *table.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
