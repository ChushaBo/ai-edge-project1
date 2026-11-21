"""Tests for src/08_explain.py's dollar-translation helper (RUNBOOK 5.3)."""

import importlib.util

import numpy as np
import pandas as pd

from utils.config import REPO_ROOT


def _script_08():
    spec = importlib.util.spec_from_file_location("explain", REPO_ROOT / "src" / "08_explain.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dollar_effects_table_categorical():
    m = _script_08()
    values = pd.Series(pd.Categorical(["A", "A", "X", "X", "V"], categories=["A", "V", "X"]))
    shap_col = np.array([0.1, 0.2, -0.1, 0.0, 0.5])
    out = m.dollar_effects_table("flood_zone_group", shap_col, values)
    assert set(out["level"]) == {"A", "V", "X"}
    a_row = out[out["level"] == "A"].iloc[0]
    assert a_row["n"] == 2
    assert a_row["mean_shap_share"] == 0.15
    assert a_row["dollar_effect_250k_home"] == 0.15 * 250_000


def test_dollar_effects_table_numeric_terciles_sum_to_all_rows():
    m = _script_08()
    values = pd.Series(np.arange(30, dtype="float64"))
    shap_col = values.to_numpy() * 0.01
    out = m.dollar_effects_table("totalBuildingInsuranceCoverage", shap_col, values)
    assert out["n"].sum() == 30
    assert len(out) == 3  # low/mid/high
    # dollar effect should increase with the tercile, since shap increases with value
    assert out["dollar_effect_250k_home"].max() > out["dollar_effect_250k_home"].min()
