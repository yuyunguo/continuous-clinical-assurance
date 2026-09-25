"""Aggregate-only queries for Tier 2 sizing, with small-cell suppression."""

from typing import Any, Dict, List, Sequence

MIN_CELL = 10   # a count below this is withheld, as are the events inside a withheld cell


def suppress_small(rows: List[Dict[str, Any]], count_keys: Sequence[str]) -> List[Dict[str, Any]]:
    """Replace every count below `MIN_CELL` by None. If the first count key (the cell size) is withheld, every count in the row is."""
    out = []
    for row in rows:
        new = dict(row)
        cell = row.get(count_keys[0])          # read before any count is replaced, so a withheld cell size still withholds the rest of the row
        for key in count_keys:
            value = new.get(key)
            if value is not None and (value < MIN_CELL or (cell is not None and cell < MIN_CELL)):
                new[key] = None
        out.append(new)
    return out


def run_aggregate(conn: Any, sql: str, count_keys: Sequence[str]) -> List[Dict[str, Any]]:
    """Run one aggregate query in a read-only session and return its rows, with small cells suppressed."""
    conn.rollback()                      # a previous query leaves a transaction open, and psycopg2 cannot change the session inside one
    conn.set_session(readonly=True)
    with conn.cursor() as cur:
        cur.execute(sql)
        names = [d[0] for d in cur.description]
        rows = [dict(zip(names, r)) for r in cur.fetchall()]
    return suppress_small(rows, count_keys)
