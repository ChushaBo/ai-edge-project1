"""Tests for utils.io."""

import pandas as pd

from utils.io import column_summary, write_parquet_atomic


def test_column_summary_percent_missing():
    df = pd.DataFrame({"a": [1, None, None, 4], "b": ["x", "y", "z", "w"]})
    s = column_summary(df).set_index("column")

    assert s.loc["a", "non_null"] == 2 and s.loc["a", "pct_missing"] == 50.0
    assert s.loc["b", "non_null"] == 4 and s.loc["b", "pct_missing"] == 0.0
    assert list(s.index) == ["a", "b"]


def test_column_summary_empty_frame():
    s = column_summary(pd.DataFrame({"a": pd.Series([], dtype="float64")}))
    assert s.loc[0, "pct_missing"] == 0.0


def test_write_parquet_atomic_leaves_no_temp_file(tmp_path):
    path = tmp_path / "out.parquet"
    df = pd.DataFrame({"id": [1, 2]})
    write_parquet_atomic(df, path)
    write_parquet_atomic(df, path)  # overwrite works on rerun

    assert pd.read_parquet(path).equals(df)
    assert [p.name for p in tmp_path.iterdir()] == ["out.parquet"]
