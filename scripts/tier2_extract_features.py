"""Extract Tier 2 real-data features and labels from MIMIC-IV, for the study author to run locally.

**Run this on your own machine. Do not send me the output of this script beyond what you choose to share.**
It reads row-level clinical data (chart events, lab events, diagnosis codes) and writes local Parquet files with
one row per case: an order key, a label, a subgroup, and feature columns — the schema
`src.tier2_module.config.Tier2StreamConfig` and `assign_windows`/`RealDeployedModel` expect. `hadm_id`, `subject_id`,
and `stay_id` are used internally to join tables but are dropped before the output is written (see `keep` in
`build_table`), so no patient identifier reaches the Parquet file. Still exercise your own judgment before sharing
any of it with anyone, including me — 500-plus rows of real vitals and labs are worth a second look first.

Usage: PGPASSWORD=... python scripts/tier2_extract_features.py --host HOST --task {sepsis,readmission,mortality}
       [--user USER] [--out-dir phase2/results/tier2/features]

Vitals and labs are aggregated over the first 24 hours of the ICU stay (mean, min, max, last, count), the same
window and aggregation the earlier paper [yu2026deferral] used and the MIMIC-III replication in its own repository
(`deferral_frontier/experiments/extract_mimic3_features.py`) mirrors. Item IDs are the MetaVision set from that
script, which MIMIC-IV also uses; if a feature comes back all-missing, check the item ID against `mimic_icu.d_items`
in your instance before trusting the result.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd
import psycopg2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

VITAL_MAP = {
    "vital_heart_rate": (220045,), "vital_resp_rate": (220210, 224690),
    "vital_abp_sys": (220050, 220179), "vital_abp_dia": (220051, 220180), "vital_abp_mean": (220052, 220181),
    "vital_nibp_sys": (220179,), "vital_nibp_dia": (220180,), "vital_nibp_mean": (220181,),
    "vital_temp_f": (223761,), "vital_temp_c": (223762,),
}
LAB_MAP = {
    "lab_creatinine": (50912,), "lab_hemoglobin": (50811,), "lab_wbc": (51301,), "lab_platelet": (51265,),
    "lab_sodium": (50983,), "lab_potassium": (50971,), "lab_glucose": (50931,), "lab_bicarbonate": (50882,),
    "lab_bun": (51006,), "lab_alt": (50861,), "lab_ast": (50878,), "lab_bilirubin": (50885,), "lab_lactate": (50813,),
}
ALL_VITAL_IDS = tuple(i for ids in VITAL_MAP.values() for i in ids)
ALL_LAB_IDS = tuple(i for ids in LAB_MAP.values() for i in ids)
FEATURE_COLUMNS = tuple(f"{feat}_{stat}" for feat in list(VITAL_MAP) + list(LAB_MAP) for stat in ("mean", "min", "max", "last", "count"))
# Every retained non-feature column: `order_key` (`icu_intime`, shifted per patient -- valid only for within-patient
# and, per MIMIC-IV's documented de-identification, approximately within-anchor_year_group ordering, NOT across the
# whole cohort; see `scripts/m7_tier2_natural_drift.py`), `label`, `subgroup`, and `anchor_year_group` (the properly
# de-identification-safe 3-year calendar bucket MIMIC-IV itself sanctions for cross-patient temporal analysis).
NON_FEATURE_COLUMNS = ("order_key", "label", "subgroup", "anchor_year_group")


def get_conn(host: str, user: str) -> "psycopg2.extensions.connection":
    return psycopg2.connect(host=host, dbname="mimiciv", user=user, password=os.environ["PGPASSWORD"], connect_timeout=10)


def extract_cohort(conn: "psycopg2.extensions.connection") -> pd.DataFrame:
    """The Tier 2 cohort (`public.paper2_stays`), with the 24-hour prediction point, a 0/1 subgroup column, and
    `anchor_year_group` (the de-identification-safe 3-year calendar bucket, for a valid cross-patient natural-drift
    ordering -- `order_key`/`icu_intime` alone is not one; see `NON_FEATURE_COLUMNS`)."""
    q = ("SELECT stay_id, hadm_id, subject_id, icu_intime, icu_intime + interval '24 hours' AS prediction_time, "
        "mortality, gender, anchor_year_group FROM public.paper2_stays")
    df = pd.read_sql(q, conn)
    df["subgroup"] = (df["gender"] == "F").astype(int)
    logger.info("Cohort: %d stays", len(df))
    return df


def _aggregate(conn: "psycopg2.extensions.connection", cohort: pd.DataFrame, table: str, id_col: str,
              item_map: dict, ids: Tuple[int, ...]) -> pd.DataFrame:
    """Vectorized mean/min/max/last/count of each feature in `item_map`, within [icu_intime, prediction_time)."""
    q = (f"SELECT {id_col}, charttime, itemid, valuenum FROM {table} WHERE {id_col} = ANY(%(ids)s) "
        f"AND itemid = ANY(%(items)s) AND valuenum IS NOT NULL")
    raw = pd.read_sql(q, conn, params={"ids": cohort[id_col].tolist(), "items": list(ids)})
    id_to_feat = {i: feat for feat, item_ids in item_map.items() for i in item_ids}
    raw["feature_name"] = raw["itemid"].map(id_to_feat)
    raw = raw.dropna(subset=["feature_name"])
    raw = raw.merge(cohort[[id_col, "icu_intime", "prediction_time"]], on=id_col)
    raw = raw[(raw["charttime"] >= raw["icu_intime"]) & (raw["charttime"] < raw["prediction_time"])]
    agg = raw.groupby([id_col, "feature_name"])["valuenum"].agg(mean="mean", min="min", max="max", last="last", count="count")
    wide = agg.unstack("feature_name")
    wide.columns = [f"{feat}_{stat}" for stat, feat in wide.columns]
    return wide.reindex(cohort[id_col]).reset_index()


def label_mortality(cohort: pd.DataFrame) -> pd.Series:
    return cohort.set_index("hadm_id")["mortality"].astype(int)


def label_sepsis3(conn: "psycopg2.extensions.connection", cohort: pd.DataFrame) -> pd.Series:
    """Sepsis-3 (`mimic_derived.sepsis3`), per D8 (revised 2026-09-22): clinical criteria, not the paper's ICD-9/10
    definition. Chosen over ICD because ICD sepsis is roughly flat across calendar periods (13.8-14.5%) while
    Sepsis-3 shows real, substantial drift (48.4% to 34.9%) -- see `Tier2-Readiness.md` for the full reasoning,
    including the resulting loss of direct comparability to the paper's deployed-model baseline (which was built
    against the ICD label) and the confound between that real drift and any injected perturbation's onset.
    """
    q = "SELECT stay_id FROM mimic_derived.sepsis3 WHERE stay_id = ANY(%(s)s) AND sepsis3"
    pos_stays = pd.read_sql(q, conn, params={"s": cohort["stay_id"].tolist()})["stay_id"]
    return cohort.set_index("hadm_id")["stay_id"].isin(pos_stays).astype(int)


def label_readmission(conn: "psycopg2.extensions.connection", cohort: pd.DataFrame) -> pd.Series:
    """30-day readmission to the same health system, survivors only (matches `tier2_aggregates.py`'s definition)."""
    q = ("SELECT hadm_id, subject_id, hospital_expire_flag, dischtime, "
        "lead(admittime) OVER (PARTITION BY subject_id ORDER BY admittime) AS next_admit FROM mimic_core.admissions "
        "WHERE subject_id = ANY(%(s)s)")
    adm = pd.read_sql(q, conn, params={"s": cohort["subject_id"].tolist()})
    adm = adm[adm["hadm_id"].isin(cohort["hadm_id"])]
    adm = adm[adm["hospital_expire_flag"] == 0]
    readmitted = (adm["next_admit"] <= adm["dischtime"] + pd.Timedelta(days=30)).astype(int)
    return readmitted.set_axis(adm["hadm_id"])


LABEL_FNS = {"mortality": lambda conn, cohort: label_mortality(cohort),
            "sepsis": label_sepsis3,
            "readmission": label_readmission}


def build_table(conn: "psycopg2.extensions.connection", task: str) -> pd.DataFrame:
    """The full feature/label/order-key/subgroup table for one task, ready for `assign_windows`."""
    cohort = extract_cohort(conn)
    vitals = _aggregate(conn, cohort, "mimic_icu.chartevents", "stay_id", VITAL_MAP, ALL_VITAL_IDS)
    labs = _aggregate(conn, cohort, "mimic_hosp.labevents", "hadm_id", LAB_MAP, ALL_LAB_IDS)
    features = cohort[["hadm_id", "stay_id", "icu_intime", "subgroup", "anchor_year_group"]].merge(
        vitals, on="stay_id", how="left").merge(labs, on="hadm_id", how="left")
    labels = LABEL_FNS[task](conn, cohort)
    table = features.merge(labels.rename("label"), left_on="hadm_id", right_index=True, how="inner")
    table = table.rename(columns={"icu_intime": "order_key"})
    for col in FEATURE_COLUMNS:
        if col not in table.columns:
            table[col] = float("nan")
    keep = [*NON_FEATURE_COLUMNS, *FEATURE_COLUMNS]
    logger.info("Task %s: %d rows, prevalence %.3f", task, len(table), table["label"].mean())
    return table[keep]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--host", required=True)
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--task", required=True, choices=sorted(LABEL_FNS))
    parser.add_argument("--out-dir", type=Path, default=Path("phase2/results/tier2/features"))
    args = parser.parse_args()
    if not os.environ.get("PGPASSWORD"):
        logger.error("set the PGPASSWORD environment variable")
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    conn = get_conn(args.host, args.user)
    try:
        table = build_table(conn, args.task)
    finally:
        conn.close()
    out = args.out_dir / f"{args.task}.parquet"
    table.to_parquet(out)
    logger.info("Wrote %s (%d rows, %d columns)", out, *table.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
