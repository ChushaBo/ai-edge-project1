"""Leakage guards (CLAUDE.md rule 3): only allowed columns may reach the model.

The allowed set is allowed_features + derived_features + the meta columns (id, date
of loss, county, split, target). Any other column in the feature table fails.
"""

import importlib.util

import pandas as pd
import pytest

from utils.config import REPO_ROOT, get_path, load_config

# Post-loss fields and event identifiers that must never be features (claims v3 names).
FORBIDDEN = {
    "amountPaidOnBuildingClaim", "amountPaidOnContentsClaim",
    "amountPaidOnIncreasedCostOfComplianceClaim", "netBuildingPaymentAmount",
    "netContentsPaymentAmount", "netIccPaymentAmount", "buildingDamageAmount",
    "contentsDamageAmount", "buildingPropertyValue", "contentsPropertyValue",
    "buildingReplacementCost", "contentsReplacementCost", "replacementCostBasis",
    "causeOfDamage", "waterDepth", "exteriorWaterDepth", "interiorWaterDepth",
    "floodWaterDuration", "floodCharacteristicsIndicator", "eventDesignationNumber",
    "floodEvent", "ficoNumber", "nonPaymentReasonBuilding", "nonPaymentReasonContents",
    "openDate", "mostRecentPaymentDate", "mostRecentRecoveryDate", "totalSalvageRecovery",
    "totalBldgClaimPmtRecovery", "totalContentsClaimPmtRecovery", "totalIccClaimPmtRecovery",
    "totalSubrogationRecovery", "rateMethod", "floodZoneCurrent", "asOfDate", "yearOfLoss",
}


def _script_04():
    spec = importlib.util.spec_from_file_location("clean_claims", REPO_ROOT / "src" / "04_clean_claims.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def allowed_columns(cfg) -> set[str]:
    return (set(cfg["allowed_features"]) | set(cfg["derived_features"])
            | set(_script_04().meta_columns(cfg)))


def test_config_allows_no_forbidden_field():
    cfg = load_config()
    assert not FORBIDDEN & set(cfg["allowed_features"])
    assert not FORBIDDEN & set(cfg["derived_features"])
    # derived features may only come from allowed fields (or the date of loss, for month)
    sources = set(cfg["derived_features"].values())
    assert sources <= set(cfg["allowed_features"]) | {cfg["fields"]["claims_date_of_loss"]}


def test_month_of_loss_is_m2_only():
    assert load_config()["month_of_loss_models"] == ["M2"]


def test_clean_drops_columns_outside_allowed_list():
    """Synthetic claims with extra post-loss columns: none may survive cleaning."""
    cfg = load_config()
    n = 4
    raw = pd.DataFrame({
        "id": range(n),
        "asOfDate": pd.Timestamp("2026-09-08", tz="UTC"),
        "dateOfLoss": pd.to_datetime(["2015-06-01", "2020-06-01", "2023-06-01", "2023-07-01"], utc=True),
        "countyCode": "12086",
        "netBuildingPaymentAmount": [1000.0, 2000.0, 3000.0, 500000.0],
        "buildingDamageAmount": [1, 2, 3, 4],
        "causeOfDamage": "4",
        "eventDesignationNumber": "FL0124",
        "originalConstructionDate": pd.Timestamp("1980-01-01", tz="UTC"),
        "originalNBDate": pd.Timestamp("2000-01-01", tz="UTC"),
        "ratedFloodZone": ["AE", "A07", "VE", "B"],
        "occupancyType": [1, 11, 15, 18],
    })
    for col in cfg["allowed_features"]:
        if col not in raw:
            raw[col] = 1
    raw["totalBuildingInsuranceCoverage"] = 250000

    out = _script_04().clean(raw, cfg, steps=[])

    assert set(out.columns) <= allowed_columns(cfg)
    assert not FORBIDDEN & set(out.columns)
    assert out["target"].max() <= cfg["target"]["cap_share_at"]  # 500k paid on 250k cover is capped
    assert out["flood_zone_group"].tolist() == ["A", "A", "V", "X"]
    assert out["occupancy_group"].tolist() == ["single_unit", "single_unit",
                                               "five_plus_or_condo", "non_residential"]


def test_saved_feature_table_has_only_allowed_columns():
    cfg = load_config()
    path = get_path(cfg, "interim", create=False) / "claims_clean.parquet"
    if not path.exists():
        pytest.skip("claims_clean.parquet not built (run src/04_clean_claims.py)")
    cols = set(pd.read_parquet(path).columns)
    extra = cols - allowed_columns(cfg)
    assert not extra, f"columns outside the allowed list reached the feature table: {sorted(extra)}"
