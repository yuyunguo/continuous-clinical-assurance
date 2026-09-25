"""Aggregate queries for Tier 2 sizing: read-only, and small cells are suppressed before anything is reported."""

from src.utils.aggregates import MIN_CELL, run_aggregate, suppress_small


def test_cells_below_the_minimum_are_suppressed_including_the_events_inside_them():
    rows = [{"period": "2008 - 2010", "n": 5000, "events": 900}, {"period": "2011 - 2013", "n": 9, "events": 2}, {"period": "2014 - 2016", "n": 12, "events": 4}]
    out = suppress_small(rows, count_keys=("n", "events"))
    assert out[0] == {"period": "2008 - 2010", "n": 5000, "events": 900}
    assert out[1] == {"period": "2011 - 2013", "n": None, "events": None}         # n below 10: the whole cell is withheld
    assert out[2]["n"] == 12 and out[2]["events"] is None                          # events below 10 are withheld on their own


def test_zero_is_suppressed_only_when_it_hides_a_small_cell():
    assert suppress_small([{"g": "a", "n": 0}], count_keys=("n",))[0]["n"] is None
    assert MIN_CELL == 10 and suppress_small([{"g": "a", "n": 10}], count_keys=("n",))[0]["n"] == 10


def test_query_runs_in_a_read_only_session_and_returns_named_columns():
    class Cur:
        description = [("period",), ("n",)]
        executed = []

        def execute(self, sql):
            self.executed.append(sql)

        def fetchall(self):
            return [("a", 20), ("b", 3)]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class Conn:
        cur = Cur()
        read_only = None

        def cursor(self):
            return self.cur

        def rollback(self):
            pass

        def set_session(self, readonly=None, **kw):
            self.read_only = readonly

    conn = Conn()
    out = run_aggregate(conn, "select period, n from t", count_keys=("n",))
    assert conn.read_only is True and out == [{"period": "a", "n": 20}, {"period": "b", "n": None}]


def test_a_withheld_cell_withholds_every_count_in_the_row_and_none_passes_through():
    row = {"g": "a", "n": 9, "other": 500}
    assert suppress_small([row], count_keys=("n", "other"))[0] == {"g": "a", "n": None, "other": None}       # a count unrelated to n is withheld too
    assert suppress_small([{"g": "a", "n": 50, "events": None}], count_keys=("n", "events"))[0] == {"g": "a", "n": 50, "events": None}


def test_two_queries_on_one_connection_work_like_psycopg2_where_set_session_fails_inside_a_transaction():
    class Cur:
        description = [("n",)]

        def __init__(self, conn):
            self.conn = conn

        def execute(self, sql):
            self.conn.in_transaction = True            # the first query opens a transaction, as psycopg2 does

        def fetchall(self):
            return [(50,)]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class Conn:
        in_transaction = False
        read_only = None

        def cursor(self):
            return Cur(self)

        def rollback(self):
            self.in_transaction = False

        def set_session(self, readonly=None, **kw):
            if self.in_transaction:
                raise RuntimeError("set_session cannot be used inside a transaction")
            self.read_only = readonly

    conn = Conn()
    assert run_aggregate(conn, "select 1", ("n",)) == [{"n": 50}]
    assert run_aggregate(conn, "select 2", ("n",)) == [{"n": 50}] and conn.read_only is True
