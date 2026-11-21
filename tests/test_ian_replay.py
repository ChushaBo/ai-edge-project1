"""Tests for src/11_ian_replay.py's pure helpers (RUNBOOK 6.3)."""

import importlib.util

import pandas as pd
import pytest

from utils.config import REPO_ROOT


def _script_11():
    spec = importlib.util.spec_from_file_location("ian_replay", REPO_ROOT / "src" / "11_ian_replay.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_identify_ian_counties_requires_narrative_and_ian_text():
    m = _script_11()
    claims = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "dateOfLoss": pd.to_datetime(["2022-09-28", "2022-09-29", "2021-09-28", "2022-10-01"], utc=True),
        "countyCode": ["12071", "12071", "12071", "12103"],
        "has_narrative": [True, False, True, True],
        "narrative": ["Hurricane Ian brought surge...", "no text here", "Hurricane Ian...", "Flooding from heavy rain"],
    })
    ian_claims, counties = m.identify_ian_counties(claims, "countyCode")
    assert ian_claims["id"].tolist() == [1]  # 2: no narrative; 3: wrong year; 4: no "Ian" mention
    assert counties == ["12071"]


def test_historical_ian_rate_only_high_risk_zones_in_ian_counties():
    m = _script_11()
    claims = pd.DataFrame({
        "countyCode": ["12071", "12071", "12071", "12103"],
        "flood_zone_group": ["A", "X", "A", "A"],
        "dateOfLoss": pd.to_datetime(["2022-09-28", "2022-09-28", "2022-01-01", "2022-09-28"], utc=True),
    })
    policies = pd.DataFrame({
        "county": ["12071", "12071", "12103"],
        "flood_zone_group": ["A", "X", "A"],
    })
    rates = m.historical_ian_rate(claims, policies, "countyCode", ["12071"])
    # only 12071/A survives: in Ian window, high-risk zone, Ian-affected county
    assert rates[["county", "flood_zone_group"]].to_records(index=False).tolist() == [("12071", "A")]
    row = rates.iloc[0]
    assert row["n_claims"] == 1  # the 2022-01-01 row is outside the Ian window
    assert row["n_policies"] == 1
    assert row["ian_rate"] == pytest.approx(1.0)


def test_county_ian_embeddings_averages_per_county():
    m = _script_11()
    ian_claims = pd.DataFrame({"id": [1, 2, 3], "countyCode": ["12071", "12071", "12103"]})
    text_feats = pd.DataFrame({"id": [1, 2, 3], "txt_pc01": [0.1, 0.3, 0.9], "txt_pc02": [1.0, 2.0, 5.0]})
    out = m.county_ian_embeddings(ian_claims, text_feats, "countyCode", ["txt_pc01", "txt_pc02"])
    row_71 = out[out["countyCode"] == "12071"].iloc[0]
    assert row_71["txt_pc01"] == pytest.approx(0.2)
    assert row_71["txt_pc02"] == pytest.approx(1.5)
