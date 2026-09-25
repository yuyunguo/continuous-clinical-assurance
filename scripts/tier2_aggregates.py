"""Aggregate sizing of the Tier 2 datasets: counts and event rates by period or hospital. No patient rows. Cells below 10 are withheld.

Usage: PGPASSWORD=... python scripts/tier2_aggregates.py --host HOST [--user USER] [--out-dir phase2/results/tier2]

Runs a fixed set of aggregate queries, read-only, on the MIMIC-IV database (the earlier paper's ICU cohort table, the derived sepsis table, and hospital
admissions for readmission) and on the eICU database (stays by discharge year, hospital sizes, unit types, and APACHE predicted mortality availability).
It writes one JSON per database. Every count below 10 is replaced by null before anything is written.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import psycopg2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.aggregates import MIN_CELL, run_aggregate  # noqa: E402

logger = logging.getLogger(__name__)
Queries = Dict[str, Tuple[str, Tuple[str, ...]]]   # name -> (sql, count columns); the first count column is the cell size

SEPSIS_ICD9 = ("99591", "99592", "78552")     # severe sepsis, severe sepsis w/ septic shock, septic shock (ICD-9-CM)
SEPSIS_ICD10 = ("R6520", "R6521")              # severe sepsis without/with septic shock (ICD-10-CM); standard equivalents of the ICD-9 set above
SEPSIS_ICD_CODES = SEPSIS_ICD9 + SEPSIS_ICD10  # not verified against the original paper's extraction script, which is not in the repository; flagged in Tier2-Readiness.md

MIMIC: Queries = {
    "cohort_by_period": ("SELECT anchor_year_group AS period, count(*) AS n, sum(mortality) AS deaths FROM public.paper2_stays GROUP BY 1 ORDER BY 1", ("n", "deaths")),
    "sepsis_icd_by_period": (f"SELECT ps.anchor_year_group AS period, count(*) AS n, "
                             f"sum(CASE WHEN d.hadm_id IS NOT NULL THEN 1 ELSE 0 END) AS sepsis_icd FROM public.paper2_stays ps "
                             f"LEFT JOIN (SELECT DISTINCT hadm_id FROM mimic_hosp.diagnoses_icd WHERE icd_code IN {SEPSIS_ICD_CODES}) d "
                             f"ON d.hadm_id = ps.hadm_id GROUP BY 1 ORDER BY 1", ("n", "sepsis_icd")),
    "cohort_row_number": ("SELECT rn, count(*) AS n FROM public.paper2_stays GROUP BY 1 ORDER BY 1 LIMIT 5", ("n",)),
    "sepsis_by_period": ("SELECT ps.anchor_year_group AS period, count(*) AS n, sum(CASE WHEN s.sepsis3 THEN 1 ELSE 0 END) AS sepsis FROM public.paper2_stays ps "
                         "LEFT JOIN mimic_derived.sepsis3 s ON s.stay_id = ps.stay_id GROUP BY 1 ORDER BY 1", ("n", "sepsis")),
    "admissions_by_period": ("SELECT p.anchor_year_group AS period, count(*) AS n, sum(a.hospital_expire_flag) AS deaths FROM mimic_core.admissions a "
                             "JOIN mimic_core.patients p USING (subject_id) GROUP BY 1 ORDER BY 1", ("n", "deaths")),
    "readmission_30d_by_period": ("WITH a AS (SELECT p.anchor_year_group AS period, ad.hospital_expire_flag AS died, ad.dischtime, "
                                  "lead(ad.admittime) OVER (PARTITION BY ad.subject_id ORDER BY ad.admittime) AS next_admit FROM mimic_core.admissions ad "
                                  "JOIN mimic_core.patients p USING (subject_id)) SELECT period, count(*) AS n, "
                                  "sum(CASE WHEN next_admit <= dischtime + interval '30 days' THEN 1 ELSE 0 END) AS readmitted FROM a WHERE died = 0 GROUP BY 1 ORDER BY 1",
                                  ("n", "readmitted")),
}
EICU: Queries = {
    "stays_by_discharge_year": ("SELECT hospitaldischargeyear AS year, count(*) AS n, sum(CASE WHEN hospitaldischargestatus = 'Expired' THEN 1 ELSE 0 END) AS deaths "
                                "FROM eicu_crd.patient GROUP BY 1 ORDER BY 1", ("n", "deaths")),
    "hospital_size_bins": ("WITH h AS (SELECT hospitalid, count(*) AS n FROM eicu_crd.patient GROUP BY 1) SELECT CASE WHEN n < 100 THEN 'under 100' WHEN n < 500 THEN '100 to 499' "
                           "WHEN n < 1000 THEN '500 to 999' WHEN n < 2000 THEN '1000 to 1999' ELSE '2000 or more' END AS size_bin, count(*) AS hospitals, sum(n) AS stays "
                           "FROM h GROUP BY 1 ORDER BY 3", ("hospitals", "stays")),
    "unit_types": ("SELECT unittype AS unit_type, count(*) AS n FROM eicu_crd.patient GROUP BY 1 ORDER BY 2 DESC", ("n",)),
    "apache_predicted_mortality_available": ("SELECT count(*) AS n, sum(CASE WHEN predictedhospitalmortality IS NOT NULL AND predictedhospitalmortality <> '' THEN 1 ELSE 0 END) "
                                             "AS with_prediction FROM eicu_crd.apachepatientresult", ("n", "with_prediction")),
}


def collect(host: str, dbname: str, user: str, queries: Queries) -> Dict[str, List[Dict[str, object]]]:
    """Run each query on one database and return the suppressed rows by name."""
    conn = psycopg2.connect(host=host, dbname=dbname, user=user, password=os.environ["PGPASSWORD"], connect_timeout=10)
    try:
        return {name: run_aggregate(conn, sql, keys) for name, (sql, keys) in queries.items()}
    finally:
        conn.close()


def main() -> int:
    """Run the aggregate queries on both databases and write the JSON."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--host", required=True)
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--out-dir", type=Path, default=Path("phase2/results/tier2"))
    args = parser.parse_args()
    if not os.environ.get("PGPASSWORD"):
        logger.error("set the PGPASSWORD environment variable")
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for dbname, queries in (("mimiciv", MIMIC), ("eicu", EICU)):
        result = collect(args.host, dbname, args.user, queries)
        (args.out_dir / f"aggregates_{dbname}.json").write_text(json.dumps({"min_cell": MIN_CELL, "results": result}, indent=1, default=str), encoding="utf-8")
        logger.info("Wrote aggregates_%s.json (%d aggregate tables)", dbname, len(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
