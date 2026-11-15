"""File helpers: atomic parquet writes and column summaries."""

from pathlib import Path

import pandas as pd


def write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    """Write to <path>.tmp then rename, so an interrupted run never leaves a partial file
    that a later run would mistake for a finished download."""
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def column_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: name, dtype, non-null count, percent missing (0-100)."""
    n = len(df)
    missing = df.isna().sum()
    return pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(t) for t in df.dtypes],
            "non_null": (n - missing).to_numpy(),
            "pct_missing": (100 * missing / n if n else missing * 0.0).round(2).to_numpy(),
        }
    )
