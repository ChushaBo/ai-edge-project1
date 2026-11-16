"""Tests for utils.models (feature encoding, dollar metrics, event grouping) and the
model-level rules in src/07_train_models.py (CLAUDE.md rules 2-4)."""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from utils import models
from utils.config import REPO_ROOT, load_config


def _script_07():
    spec = importlib.util.spec_from_file_location("train_models", REPO_ROOT / "src" / "07_train_models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- feature encoding: categories fixed from the fitting rows only (rule 4) --------

def test_categories_fit_on_training_rows_only():
    train = pd.DataFrame({"flood_zone_group": ["A", "A", "X", "V"]})
    test = pd.DataFrame({"flood_zone_group": ["A", "X", "V", "D"]})  # "D" unseen in training
    cats = models.fit_categories(train, ["flood_zone_group"])
    assert cats["flood_zone_group"] == ["A", "V", "X"]

    encoded = models.encode_features(test, ["flood_zone_group"], cats)
    assert encoded["flood_zone_group"].dtype.name == "category"
    assert list(encoded["flood_zone_group"].cat.categories) == ["A", "V", "X"]
    # the unseen test-only level becomes missing, never a new category
    assert encoded["flood_zone_group"].isna().tolist() == [False, False, False, True]


def test_encode_features_keeps_nullable_na_and_coerces_numerics():
    df = pd.DataFrame({
        "elevationDifference": pd.array([1, None, 3], dtype="Int64"),
        "elevatedBuildingIndicator": pd.array([True, False, None], dtype="boolean"),
    })
    out = models.encode_features(df, ["elevationDifference", "elevatedBuildingIndicator"], categories={})
    np.testing.assert_array_equal(out["elevationDifference"].to_numpy(), [1.0, np.nan, 3.0])
    np.testing.assert_array_equal(out["elevatedBuildingIndicator"].to_numpy(), [1.0, 0.0, np.nan])


def test_m1_features_exclude_month_and_text():
    assert "month_of_loss" not in models.M1_FEATURES
    assert not any(c.startswith(("txt_pc", "txtm_pc")) for c in models.M1_FEATURES)
    assert models.MONTH_FEATURE == "month_of_loss"


# --- dollar metrics: hand-computed on a tiny frame ---------------------------------

def test_dollar_metrics_matches_hand_computation():
    y_true = np.array([0.2, 0.5, 0.0])
    y_pred = np.array([0.3, 0.4, 0.1])
    coverage = np.array([100_000.0, 200_000.0, 50_000.0])
    m = models.dollar_metrics(y_true, y_pred, coverage)

    diff_share = y_pred - y_true          # [0.1, -0.1, 0.1]
    diff_dollar = diff_share * coverage   # [10000, -20000, 5000]
    assert m["mae_share"] == pytest.approx(np.mean(np.abs(diff_share)))
    assert m["mae_dollar"] == pytest.approx(np.mean(np.abs(diff_dollar)))
    assert m["actual_total_dollar"] == pytest.approx((y_true * coverage).sum())
    assert m["predicted_total_dollar"] == pytest.approx((y_pred * coverage).sum())
    expected_pct = 100 * (m["predicted_total_dollar"] - m["actual_total_dollar"]) / m["actual_total_dollar"]
    assert m["pct_error_total"] == pytest.approx(expected_pct)


# --- quarantine scope: baselines/M1 keep Helene, M2/M2_masked do not ----------------

def test_quarantine_applies_only_to_text_models_in_config():
    cfg = load_config()
    q = cfg["quarantine"]
    assert q["enabled"] is True
    assert set(q["applies_to"]) == {"M2", "M2_masked"}
    assert "M1" not in q["applies_to"]


# --- baselines: B1 falls back to B0 for an unseen or missing zone ------------------

def test_b1_falls_back_to_b0_for_unseen_or_missing_zone():
    m = _script_07()
    train = pd.DataFrame({"target": [0.1, 0.3, 0.5, 0.9],
                          "flood_zone_group": ["A", "A", "X", "X"]})
    means, overall = m.fit_b1(train, "flood_zone_group")
    test = pd.DataFrame({"flood_zone_group": ["A", "X", "V", None]})  # V unseen, last missing
    pred = m.predict_b1(test, "flood_zone_group", means, overall)
    assert pred[0] == pytest.approx(0.2)   # mean of A
    assert pred[1] == pytest.approx(0.7)   # mean of X
    assert pred[2] == pytest.approx(overall)  # unseen zone -> B0
    assert pred[3] == pytest.approx(overall)  # missing zone -> B0


# --- objective is fixed to L2; L1 is a diagnostic only, never selected -------------

def test_objective_is_l2_only_not_part_of_search_grid():
    cfg = load_config()
    models_cfg = cfg["models"]
    assert models_cfg["objective"] == "regression"
    assert "objective" not in models_cfg["search_grid"]
    assert models_cfg["l1_diagnostic_objective"] == "regression_l1"


# --- event grouping: matched claims group by episode, unmatched by date cluster ----

def test_event_groups_splits_matched_and_unmatched():
    df = pd.DataFrame({
        "has_narrative": [True, True, False, False, False],
        "matched_episode_ids": ["5", "5", "", "", ""],
        "dateOfLoss": pd.to_datetime(
            ["2022-09-28", "2022-09-29", "2010-01-01", "2010-01-02", "2010-06-01"], utc=True),
    })
    groups = models.event_groups(df, "dateOfLoss", gap_days=3)
    assert groups.iloc[0] == groups.iloc[1] == "episode_5"
    assert groups.iloc[2] == groups.iloc[3]          # within 3 days of each other
    assert groups.iloc[4] != groups.iloc[2]          # far away -> a different cluster
    assert groups.iloc[2].startswith("cluster_")
