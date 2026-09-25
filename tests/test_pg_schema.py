"""The PostgreSQL schema dump reads catalog metadata only (names, types, estimated row counts), never a table's contents."""

from src.utils.pg_schema import COLUMNS_SQL, ROWS_SQL, dump_pg_schema


class FakeCursor:
    def __init__(self, columns, rows):
        self.columns, self.rows, self.executed = columns, rows, []

    def execute(self, sql):
        self.executed.append(sql)

    def fetchall(self):
        return self.rows_estimates if "reltuples" in self.executed[-1].lower() else self.rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeConn:
    def __init__(self):
        self.cur = FakeCursor(None, [("mimiciv_hosp", "patients", "subject_id", "integer"), ("mimiciv_hosp", "patients", "gender", "character varying"),
                                     ("mimiciv_icu", "icustays", "stay_id", "integer"), ("mimiciv_icu", "empty_table", "a", "text")])
        self.cur.rows_estimates = [("mimiciv_hosp", "patients", 364627.0), ("mimiciv_icu", "icustays", -1.0), ("mimiciv_icu", "empty_table", 0.0), ("other", "no_columns", 5.0)]
        self.read_only = None

    def cursor(self):
        return self.cur

    def set_session(self, readonly=None, **kw):
        self.read_only = readonly


def test_dump_groups_columns_by_table_and_attaches_estimated_rows():
    conn = FakeConn()
    out = dump_pg_schema(conn)
    assert out["mimiciv_hosp.patients"]["columns"] == [{"name": "subject_id", "type": "integer"}, {"name": "gender", "type": "character varying"}]
    assert out["mimiciv_hosp.patients"]["estimated_rows"] == 364627
    assert out["mimiciv_icu.icustays"]["estimated_rows"] is None            # a table never analyzed reports -1
    assert out["mimiciv_icu.empty_table"]["estimated_rows"] == 0             # an analyzed empty table is 0, not unknown
    assert "other.no_columns" not in out                                     # a relation without column metadata is ignored
    assert isinstance(out["mimiciv_hosp.patients"]["estimated_rows"], int)


def test_session_is_read_only_and_only_catalog_queries_run():
    conn = FakeConn()
    dump_pg_schema(conn)
    assert conn.read_only is True
    for sql in conn.cur.executed:
        text = sql.lower()
        assert "information_schema" in text or "pg_class" in text
        assert " from public." not in text and "select *" not in text
    assert "information_schema.columns" in COLUMNS_SQL.lower() and "reltuples" in ROWS_SQL.lower()


def test_system_schemas_are_excluded_by_the_queries():
    assert "pg_catalog" in COLUMNS_SQL and "information_schema" in COLUMNS_SQL and "pg_catalog" in ROWS_SQL
