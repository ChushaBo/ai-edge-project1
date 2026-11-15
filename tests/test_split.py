"""Time-based split checks (CLAUDE.md rule 4)."""

import importlib.util

import pandas as pd
import pytest

from utils.config import REPO_ROOT, get_path, load_config, require_date


def _script_04():
    spec = importlib.util.spec_from_file_location("clean_claims", REPO_ROOT / "src" / "04_clean_claims.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_assign_split_inclusive_boundaries():
    dates = pd.Series(pd.to_datetime(
        ["2019-12-31", "2020-01-01", "2021-12-31", "2022-01-01"], utc=True))
    split = _script_04().assign_split(
        dates, pd.Timestamp("2019-12-31", tz="UTC"), pd.Timestamp("2021-12-31", tz="UTC"))
    assert split.tolist() == ["train", "validation", "validation", "test"]


def test_saved_splits_do_not_overlap_in_time():
    cfg = load_config()
    path = get_path(cfg, "interim", create=False) / "claims_clean.parquet"
    if not path.exists():
        pytest.skip("claims_clean.parquet not built (run src/04_clean_claims.py)")
    date = cfg["fields"]["claims_date_of_loss"]
    df = pd.read_parquet(path, columns=[date, "split"])
    assert set(df["split"]) == {"train", "validation", "test"}

    rng = df.groupby("split")[date].agg(["min", "max"])
    assert rng.loc["train", "max"] < rng.loc["validation", "min"]
    assert rng.loc["validation", "max"] < rng.loc["test", "min"]
    assert rng.loc["train", "max"] <= require_date(cfg, "splits.train_end")
    assert rng.loc["validation", "max"] <= require_date(cfg, "splits.validation_end")
    assert rng.loc["validation", "min"] > require_date(cfg, "splits.train_end")
    assert rng.loc["test", "min"] > require_date(cfg, "splits.validation_end")
