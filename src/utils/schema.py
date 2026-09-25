"""Schema of a folder of CSV tables: names, column headers, and optional row counts. Never a data value."""

import csv
import gzip
import io
import sys
from pathlib import Path
from typing import Any, Dict, IO


csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))   # free-text tables have very long fields


def _open(path: Path) -> IO[str]:
    """Text handle for a .csv or .csv.gz file."""
    if path.suffix == ".gz":
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def dump_schema(root: Path, count_rows: bool = False) -> Dict[str, Dict[str, Any]]:
    """Column headers of every `.csv` and `.csv.gz` under `root`, keyed by the path relative to `root`.

    Only the first line of each file is read for the headers. With `count_rows` the rest of the file is scanned to count the rows
    (a single integer per table), and no value is kept.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or not (path.name.endswith(".csv") or path.name.endswith(".csv.gz")):
            continue
        with _open(path) as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
            entry: Dict[str, Any] = {"columns": header}
            if count_rows:
                entry["rows"] = sum(1 for _ in reader) if header else 0
        out[path.relative_to(root).as_posix()] = entry
    return out
