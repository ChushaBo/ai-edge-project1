"""Offline tests for 02_download_policies (local filtering of the bulk file)."""

import importlib.util

import pandas as pd
import pytest

from utils.config import REPO_ROOT


def _load_script_02():
    spec = importlib.util.spec_from_file_location(
        "download_policies", REPO_ROOT / "src" / "02_download_policies.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bulk(tmp_path, dates):
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "propertyState": ["FL", "FL", "FL", "GA", "FL"],
        "policyEffectiveDate": dates,
        "totalInsurancePremiumOfThePolicy": [100, 200, 300, 400, 500],
    })
    path = tmp_path / "bulk.parquet"
    df.to_parquet(path, index=False)
    return path


RAW = ["2024-12-31", "2025-01-01", "2025-12-31", "2025-06-01", "2026-01-01"]


@pytest.mark.parametrize("dates", [
    pd.to_datetime(RAW, utc=True),   # timestamp with time zone
    pd.to_datetime(RAW),             # naive timestamp
    RAW,                             # ISO strings
], ids=["utc", "naive", "string"])
def test_filter_bulk_keeps_state_and_year_inclusive(tmp_path, dates):
    script = _load_script_02()
    path = _bulk(tmp_path, dates)
    out = script.filter_bulk(path, ["id", "totalInsurancePremiumOfThePolicy"], "FL", 2025)
    assert sorted(out["id"]) == [2, 3]          # Jan 1 and Dec 31 in; GA and 2024/2026 out
    assert list(out.columns) == ["id", "totalInsurancePremiumOfThePolicy"]


def test_filter_bulk_stops_on_missing_column(tmp_path):
    script = _load_script_02()
    path = _bulk(tmp_path, RAW)
    with pytest.raises(SystemExit, match="lacks requested columns"):
        script.filter_bulk(path, ["id", "notAColumn"], "FL", 2025)


def test_filter_bulk_stops_on_corrupt_file(tmp_path):
    script = _load_script_02()
    path = tmp_path / "bad.parquet"
    path.write_bytes(b"not parquet")
    with pytest.raises(SystemExit, match="not a readable parquet"):
        script.filter_bulk(path, ["id"], "FL", 2025)
