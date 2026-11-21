"""Tests for src/10_shortfall.py's summarize() (RUNBOOK 6.2)."""

import importlib.util

import pandas as pd
import pytest

from utils.config import REPO_ROOT


def _script_10():
    spec = importlib.util.spec_from_file_location("shortfall", REPO_ROOT / "src" / "10_shortfall.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summarize_county_rows_and_statewide_row():
    m = _script_10()
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "county": ["12001", "12001", "12003", "12003"],
        "severity": [0.1, 0.2, 0.05, 0.3],
        "totalBuildingInsuranceCoverage": [100_000.0, 200_000.0, 150_000.0, 150_000.0],
        "frequency": [0.01, 0.01, 0.02, 0.02],
        "totalInsurancePremiumOfThePolicy": [500.0, 800.0, 400.0, 600.0],
        "policyCost": [600.0, 900.0, 450.0, 650.0],
        "fullRiskPremium": [700.0, 1000.0, 500.0, 700.0],
    })
    out = m.summarize(df, "severity", "frequency", "test_model")

    assert set(out["county"]) == {"12001", "12003", "ALL"}
    assert (out["severity_model"] == "test_model").all()

    row_12001 = out[out["county"] == "12001"].iloc[0]
    expected_loss_12001 = 0.1 * 100_000 * 0.01 + 0.2 * 200_000 * 0.01
    assert row_12001["expected_loss_total"] == pytest.approx(expected_loss_12001)
    assert row_12001["premium_total"] == 1300.0
    assert row_12001["n_policies"] == 2

    statewide = out[out["county"] == "ALL"].iloc[0]
    assert statewide["n_policies"] == 4
    assert statewide["premium_total"] == df["totalInsurancePremiumOfThePolicy"].sum()
    assert statewide["expected_loss_total"] == pytest.approx(
        out[out["county"] != "ALL"]["expected_loss_total"].sum())
    assert statewide["loss_ratio"] == pytest.approx(
        statewide["expected_loss_total"] / statewide["premium_total"])
    assert statewide["gap"] == pytest.approx(statewide["expected_loss_total"] - statewide["premium_total"])
