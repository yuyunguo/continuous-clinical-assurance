"""The Tier 2 schema dump reports table names, column headers, and (optionally) row counts, and never a data value."""

import gzip
import json

from src.utils.schema import dump_schema


def write(path, text, gz=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if gz:
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write(text)
    else:
        path.write_text(text, encoding="utf-8")


def test_reports_columns_and_never_a_value(tmp_path):
    write(tmp_path / "hosp" / "patients.csv", "subject_id,gender,anchor_age\n1001,SECRET123,55\n1002,X,60\n")
    write(tmp_path / "icu" / "icustays.csv.gz", "stay_id,subject_id,los\n7,1001,2.5\n", gz=True)
    out = dump_schema(tmp_path)
    assert out["hosp/patients.csv"]["columns"] == ["subject_id", "gender", "anchor_age"]
    assert out["icu/icustays.csv.gz"]["columns"] == ["stay_id", "subject_id", "los"]
    text = json.dumps(out)
    assert "SECRET123" not in text and "1001" not in text and "2.5" not in text
    assert all("rows" not in v for v in out.values())              # no counts unless asked


def test_row_counts_are_integers_only_and_optional(tmp_path):
    write(tmp_path / "a.csv", "x,y\n1,2\n3,4\n5,6\n")
    write(tmp_path / "b.csv.gz", "x\n", gz=True)
    write(tmp_path / "empty.csv", "")
    write(tmp_path / "notes.txt", "ignore me")
    out = dump_schema(tmp_path, count_rows=True)
    assert out["a.csv"]["rows"] == 3 and out["b.csv.gz"]["rows"] == 0
    assert out["empty.csv"] == {"columns": [], "rows": 0}
    assert "notes.txt" not in out
    assert set(out) == {"a.csv", "b.csv.gz", "empty.csv"}


def test_quoted_headers_and_nested_folders(tmp_path):
    write(tmp_path / "deep" / "er" / "t.csv", '"a b",c\n1,2\n')
    assert dump_schema(tmp_path)["deep/er/t.csv"]["columns"] == ["a b", "c"]


def test_row_count_ignores_newlines_inside_quoted_fields(tmp_path):
    write(tmp_path / "notes.csv", 'id,text\n1,"line one\nline two\nline three"\n2,plain\n')
    assert dump_schema(tmp_path, count_rows=True)["notes.csv"]["rows"] == 2


def test_directories_named_like_tables_are_skipped(tmp_path):
    (tmp_path / "weird.csv").mkdir()
    write(tmp_path / "weird.csv" / "inner.csv", "a\n1\n")
    assert list(dump_schema(tmp_path)) == ["weird.csv/inner.csv"]
