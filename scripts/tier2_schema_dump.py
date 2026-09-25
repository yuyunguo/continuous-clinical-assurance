"""List the tables of a dataset folder: file names, column headers, and optionally row counts. No data values.

Usage: python scripts/tier2_schema_dump.py PATH [PATH ...] [--counts] [--out schema_dump.json]

Run it on your own machine, on the folder that holds a dataset (for example the MIMIC-IV folder or the eICU folder), and send back only the JSON it writes.
It reads the first line of each .csv or .csv.gz file for the headers. With --counts it also counts the rows of each table, a single integer per table.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.schema import dump_schema  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> int:
    """Dump the schema of each folder given."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("paths", nargs="+", type=Path, help="dataset folders")
    parser.add_argument("--counts", action="store_true", help="also count the rows of each table")
    parser.add_argument("--out", type=Path, default=Path("schema_dump.json"))
    args = parser.parse_args()
    result: Dict[str, Any] = {}
    for root in args.paths:
        if not root.is_dir():
            logger.error("not a folder: %s", root)
            return 1
        result[root.name] = dump_schema(root, count_rows=args.counts)
        logger.info("%s: %d tables", root.name, len(result[root.name]))
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    logger.info("Wrote %s. It holds table names, column headers, and row counts only.", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
