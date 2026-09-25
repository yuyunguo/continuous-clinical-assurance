"""List the tables and columns of a PostgreSQL database from the catalog: names, types, and estimated row counts. No table contents.

Usage: PGPASSWORD=... python scripts/tier2_pg_schema_dump.py --host HOST --dbname mimiciv --user USER [--port 5432] [--out schema_mimiciv.json]

Run it on your own machine, once for each database (for example mimiciv and eicu), and send back only the JSON it writes. The session is set read-only, only the
catalog is queried, and the password is read from the PGPASSWORD environment variable, never from the command line or a file.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import psycopg2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.pg_schema import dump_pg_schema  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> int:
    """Connect read-only, dump the catalog, and write the JSON."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--dbname", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    password = os.environ.get("PGPASSWORD")
    if not password:
        logger.error("set the PGPASSWORD environment variable")
        return 1
    conn = psycopg2.connect(host=args.host, port=args.port, dbname=args.dbname, user=args.user, password=password, connect_timeout=10)
    try:
        schema = dump_pg_schema(conn)
    finally:
        conn.close()
    out = args.out or Path(f"schema_{args.dbname}.json")
    out.write_text(json.dumps({args.dbname: schema}, indent=1), encoding="utf-8")
    logger.info("Wrote %s: %d tables. It holds names, types, and estimated row counts only.", out, len(schema))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
