"""Schema of a PostgreSQL database from the catalog: tables, columns, types, and estimated row counts. Never a table's contents."""

from typing import Any, Dict

COLUMNS_SQL = (
    "SELECT table_schema, table_name, column_name, data_type FROM information_schema.columns "
    "WHERE table_schema NOT IN ('pg_catalog', 'information_schema') ORDER BY table_schema, table_name, ordinal_position")
ROWS_SQL = (
    "SELECT n.nspname, c.relname, c.reltuples FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
    "WHERE c.relkind IN ('r', 'p') AND n.nspname NOT IN ('pg_catalog', 'information_schema')")


def dump_pg_schema(conn: Any) -> Dict[str, Dict[str, Any]]:
    """Columns and estimated row counts of every user table, keyed `schema.table`.

    The session is set read-only, and only the two catalog queries above are run. Row counts come from the planner's statistics
    (`reltuples`, -1 for a table never analyzed, reported as None), so no table is scanned.
    """
    conn.set_session(readonly=True)
    out: Dict[str, Dict[str, Any]] = {}
    with conn.cursor() as cur:
        cur.execute(COLUMNS_SQL)
        for schema, table, column, dtype in cur.fetchall():
            out.setdefault(f"{schema}.{table}", {"columns": [], "estimated_rows": None})["columns"].append({"name": column, "type": dtype})
        cur.execute(ROWS_SQL)
        for schema, table, tuples in cur.fetchall():
            key = f"{schema}.{table}"
            if key in out and tuples >= 0:
                out[key]["estimated_rows"] = int(tuples)
    return out
