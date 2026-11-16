"""Shared helpers for 07_train_models (and later 08/10/11): feature encoding that
respects the leakage and train-only-fit rules, dollar metrics, and event grouping.

Feature encoding: nullable Int64/boolean/string columns keep NA as NA. Low-cardinality
code columns (CATEGORICAL_COLUMNS) become pandas `category` dtype with categories fixed
from the FITTING rows only (CLAUDE.md rule 4); a level never seen while fitting becomes
missing, never a new level. Everything else is coerced to float64. LightGBM reads
`category` columns natively (categorical_feature="auto" in the sklearn API).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Low-cardinality codes treated as categories, not numbers (an "8" foundation type is
# not greater than a "2"). crsClassCode and elevationDifference are ordinal-ish counts,
# kept numeric.
CATEGORICAL_COLUMNS = {
    "foundationType", "condominiumCoverageTypeCode", "buildingDeductibleCode",
    "flood_zone_group", "occupancy_group", "basementEnclosureCrawlspaceType",
    "month_of_loss",
}

# The "Core 20" tabular features (config.yaml allowed_features + derived_features,
# minus the raw columns the derived features replace), pre-event only -> M1.
M1_FEATURES = [
    "totalBuildingInsuranceCoverage", "totalContentsInsuranceCoverage",
    "numberOfFloorsInTheInsuredBuilding", "foundationType", "elevatedBuildingIndicator",
    "postFIRMConstructionIndicator", "primaryResidenceIndicator", "rentalPropertyIndicator",
    "condominiumCoverageTypeCode", "policyCount", "crsClassCode", "buildingDeductibleCode",
    "basementEnclosureCrawlspaceType", "elevationDifference", "latitude", "longitude",
    "building_age_at_loss", "policy_tenure_years", "flood_zone_group", "occupancy_group",
]
MONTH_FEATURE = "month_of_loss"  # post-event per month_of_loss_models; M2 and the M1+month diagnostic only


def text_feature_columns(n_components: int, prefix: str) -> list[str]:
    return [f"{prefix}_pc{i + 1:02d}" for i in range(n_components)]


def fit_categories(df: pd.DataFrame, feature_cols: list[str]) -> dict[str, list[str]]:
    """Category levels for each categorical feature, read from the FITTING rows only."""
    cats: dict[str, list[str]] = {}
    for col in feature_cols:
        if col in CATEGORICAL_COLUMNS:
            cats[col] = sorted(df[col].astype("string").dropna().unique().tolist())
    return cats


def encode_features(df: pd.DataFrame, feature_cols: list[str],
                    categories: dict[str, list[str]]) -> pd.DataFrame:
    """Build the model matrix: category dtype for categorical columns (levels outside
    `categories` become missing), float64 for everything else."""
    out: dict[str, object] = {}
    for col in feature_cols:
        if col in categories:
            # Build categories from this column's own values, then remap to the fitting
            # columns' category list; set_categories turns any level absent from that
            # list (an unseen test-only level) into missing, never a new category.
            s = df[col].astype("string")
            cat = pd.Categorical(s.where(s.notna(), None))
            out[col] = cat.set_categories(categories[col])
        else:
            out[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    return pd.DataFrame(out, index=df.index)


def dollar_metrics(y_true_share: np.ndarray, y_pred_share: np.ndarray,
                   coverage: np.ndarray) -> dict[str, float]:
    """MAE/RMSE in share-of-coverage units and in dollars (share x coverage), plus
    actual/predicted portfolio totals and the total's percent error."""
    diff_share = y_pred_share - y_true_share
    diff_dollar = diff_share * coverage
    actual_total = float((y_true_share * coverage).sum())
    predicted_total = float((y_pred_share * coverage).sum())
    pct_error = 100.0 * (predicted_total - actual_total) / actual_total if actual_total else np.nan
    return {
        "n": int(len(y_true_share)),
        "mae_share": float(np.abs(diff_share).mean()),
        "rmse_share": float(np.sqrt((diff_share ** 2).mean())),
        "mae_dollar": float(np.abs(diff_dollar).mean()),
        "rmse_dollar": float(np.sqrt((diff_dollar ** 2).mean())),
        "actual_total_dollar": actual_total,
        "predicted_total_dollar": predicted_total,
        "pct_error_total": pct_error,
    }


def date_clusters(loss_dates: pd.Series, gap_days: int) -> pd.Series:
    """Cluster ids for unmatched claims: a new cluster starts after a gap of more than
    gap_days between consecutive distinct loss dates (used where no matched episode
    groups the claim)."""
    uniq = pd.Series(sorted(pd.to_datetime(loss_dates.unique())))
    gaps = uniq.diff().dt.days.fillna(0.0)
    cluster_id = (gaps > gap_days).cumsum()
    mapping = dict(zip(uniq, cluster_id))
    return loss_dates.map(mapping)


def event_groups(df: pd.DataFrame, date_col: str, gap_days: int = 3) -> pd.Series:
    """Group label per claim for metrics_by_event: the primary matched episode id for
    matched claims (lowest id among matched_episode_ids -- a deterministic proxy for
    "closest to loss date" since NOAA assigns episode ids roughly chronologically),
    otherwise a loss-date cluster id. Labels are disjoint between the two kinds."""
    has_text = df.get("has_narrative")
    if has_text is None:
        has_text = pd.Series(False, index=df.index)
    primary_episode = df["matched_episode_ids"].where(has_text, "").map(
        lambda s: s.split(";")[0] if s else None)
    clusters = date_clusters(df[date_col], gap_days)
    return pd.Series(np.where(has_text, "episode_" + primary_episode.fillna(""),
                              "cluster_" + clusters.astype(str)), index=df.index)


def event_summary(df: pd.DataFrame, group_col: str, date_col: str, target_col: str,
                  pred_col: str, coverage_col: str,
                  type_col: str | None = None) -> pd.DataFrame:
    """One row per event group: date range, n claims, actual vs predicted total $,
    percent error (RUNBOOK 5.1 metrics_by_event)."""
    d = df.assign(
        actual_dollar=df[target_col] * df[coverage_col],
        predicted_dollar=df[pred_col] * df[coverage_col],
    )
    agg = {"n_claims": (date_col, "size"), "date_min": (date_col, "min"),
          "date_max": (date_col, "max"), "actual_total_dollar": ("actual_dollar", "sum"),
          "predicted_total_dollar": ("predicted_dollar", "sum")}
    if type_col is not None:
        agg["event_types"] = (type_col, lambda s: ";".join(sorted(set(";".join(s).split(";")) - {""})))
    out = d.groupby(group_col).agg(**agg).reset_index()
    out["pct_error"] = 100 * (out["predicted_total_dollar"] - out["actual_total_dollar"]) / out["actual_total_dollar"]
    return out.sort_values("actual_total_dollar", ascending=False)
