"""Segment breakdown: where does M1's portfolio-total miss actually come from?

PROVISIONAL, alongside 07-11's outputs (docs/decisions.md open item). Uses the
FROZEN m1_lgbm.joblib, baselines.joblib, and m1_deploy_lgbm.joblib -- no refit.
Data only: no figures.

On test_full (112,573 claims), for M1 (official), B1, and M1_deploy (the
train+validation+test refit from 10_shortfall.py, pending team approval --
included here for comparison only, not as an adopted model): actual vs. predicted
total payout, percent error, and each segment's share of test_full's total actual
dollars, broken down four ways:

  (a) coverage band: <$100k, $100-250k, $250k-1M, >$1M (totalBuildingInsuranceCoverage).
  (b) occupancy type, from the RAW occupancyType code (NFIP data dictionary,
      docs/data_dictionaries/nfip_claims_v3_fields.csv), not the coarser
      occupancy_group used as a model feature -- that group merges condo
      associations with non-condo apartment buildings, which this breakdown
      needs to tell apart:
        single_family      : 1, 11, 14, 16
                             (single-family dwelling; the RR2.0 single-family code;
                             mobile/manufactured home; a single unit within a
                             multi-unit building -- grouped with single-family to
                             match the model feature occupancy_group's existing
                             "single_unit" bucket, config.yaml occupancy_groups)
        two_to_four_family : 2, 12  (2-4 unit residential building)
        other_residential  : 3, 13  (5+-unit residential building, NON-condo --
                                     the landlord/apartment-building case)
        condo_or_RCBAP     : 15     (residential condo association coverage --
                                     "Residential Condominium Building Association
                                     Policy"; codes 3/13 and 15 were one combined
                                     feature-engineering bucket, "five_plus_or_condo"
                                     -- this breakdown splits them)
        commercial         : 4, 6, 17, 18, 19  (non-residential building/business,
                                                including non-residential units and
                                                mobile homes)
  (c) flood zone group: V, A, AH/AO, X/other (X, D, and any unmapped code combined).
  (d) year of loss: 2022, 2023, 2024 (test_full's only years, by construction of
      the time-based split).

Inputs:
  - data/models/m1_lgbm.joblib, baselines.joblib, m1_deploy_lgbm.joblib
  - data/interim/claims_with_text.parquet (test_full)
  - config.yaml: fields.*, target.cap_share_at, paths.*
Outputs:
  - outputs/tables/segment_breakdown.csv
"""

import joblib
import numpy as np
import pandas as pd

from utils.config import get_path, load_config, load_pipeline_script, require
from utils.log import get_logger, log_rows
from utils.models import M1_FEATURES

logger = get_logger("08c_segment_breakdown")

OCCUPANCY_LABELS = {
    "single_family": [1, 11, 14, 16],
    "two_to_four_family": [2, 12],
    "other_residential": [3, 13],
    "condo_or_RCBAP": [15],
    "commercial": [4, 6, 17, 18, 19],
}
COVERAGE_BAND_EDGES = [0, 100_000, 250_000, 1_000_000, np.inf]
COVERAGE_BAND_LABELS = ["<$100k", "$100-250k", "$250k-1M", ">$1M"]


def occupancy_label(codes: pd.Series) -> pd.Series:
    lookup = {code: label for label, codes_ in OCCUPANCY_LABELS.items() for code in codes_}
    as_int = codes.astype("Int64")
    out = as_int.map(lookup).astype("string")
    unmapped = as_int.notna() & out.isna()
    if unmapped.any():
        logger.warning("%d claims have an occupancyType code not in the dictionary mapping: %s",
                       int(unmapped.sum()), sorted(as_int[unmapped].unique().tolist())[:10])
        out = out.mask(unmapped, "unmapped")
    return out


def flood_zone_combined(zone_group: pd.Series) -> pd.Series:
    mapping = {"V": "V", "A": "A", "AH_AO": "AH/AO"}
    return zone_group.map(mapping).fillna("X/other")


def segment_rows(df: pd.DataFrame, breakdown: str, segment_col: str, target_col: str,
                 cov_col: str, preds: dict) -> pd.DataFrame:
    grand_total_actual = float((df[target_col] * df[cov_col]).sum())
    rows = []
    for segment, g in df.groupby(segment_col, observed=True):
        actual_total = float((g[target_col] * g[cov_col]).sum())
        for model_name, pred in preds.items():
            pred_seg = pred[g.index.to_numpy()] if not isinstance(pred, pd.Series) else pred.loc[g.index]
            predicted_total = float((pred_seg * g[cov_col].to_numpy()).sum())
            pct_error = 100 * (predicted_total - actual_total) / actual_total if actual_total else np.nan
            rows.append({
                "breakdown": breakdown, "segment": str(segment), "model": model_name, "n": len(g),
                "actual_total_dollar": actual_total, "predicted_total_dollar": predicted_total,
                "pct_error": pct_error,
                "segment_share_of_actual_pct": 100 * actual_total / grand_total_actual,
            })
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    cap = float(require(cfg, "target.cap_share_at"))
    cov_col = require(cfg, "fields.claims_coverage")
    date_col = require(cfg, "fields.claims_date_of_loss")
    target_col = "target"
    zone_col = "flood_zone_group"
    tables, models_dir = get_path(cfg, "tables"), get_path(cfg, "models")

    m7 = load_pipeline_script("07_train_models")
    bundle_m1 = joblib.load(models_dir / "m1_lgbm.joblib")
    baselines = joblib.load(models_dir / "baselines.joblib")
    bundle_m1_deploy = joblib.load(models_dir / "m1_deploy_lgbm.joblib")

    cols = sorted(set(["id", "split", target_col, cov_col, zone_col, date_col] + M1_FEATURES))
    claims = pd.read_parquet(get_path(cfg, "interim") / "claims_with_text.parquet", columns=cols)
    test_full = claims[claims["split"] == "test"].reset_index(drop=True)

    # occupancyType (raw) was replaced by the derived occupancy_group during cleaning
    # (04_clean_claims.py) and isn't in claims_with_text.parquet -- pull it from the
    # raw claims file for the finer occupancy breakdown this script needs.
    raw_occ = pd.read_parquet(get_path(cfg, "raw") / "claims_fl.parquet", columns=["id", "occupancyType"])
    test_full = test_full.merge(raw_occ, on="id", how="left")
    log_rows(logger, test_full, "test_full (frozen M1, B1, M1_deploy, no refit)")

    pred_m1 = m7.predict_clipped(bundle_m1["model"], test_full, bundle_m1["feature_cols"],
                                 bundle_m1["categories"], cap)
    pred_b1 = np.clip(m7.predict_b1(test_full, zone_col, baselines["b1_means"], baselines["b1_overall"]),
                      0.0, cap)
    pred_m1_deploy = m7.predict_clipped(bundle_m1_deploy["model"], test_full, bundle_m1_deploy["feature_cols"],
                                        bundle_m1_deploy["categories"], cap)
    preds = {"M1": pred_m1, "B1": pred_b1, "M1_deploy": pred_m1_deploy}

    test_full = test_full.assign(
        coverage_band=pd.cut(test_full[cov_col], bins=COVERAGE_BAND_EDGES, labels=COVERAGE_BAND_LABELS,
                             right=False, include_lowest=True),
        occupancy_type_fine=occupancy_label(test_full["occupancyType"]),
        flood_zone_combined=flood_zone_combined(test_full[zone_col]),
        year_of_loss=test_full[date_col].dt.year,
    )

    out = pd.concat([
        segment_rows(test_full, "coverage_band", "coverage_band", target_col, cov_col, preds),
        segment_rows(test_full, "occupancy_type", "occupancy_type_fine", target_col, cov_col, preds),
        segment_rows(test_full, "flood_zone_group", "flood_zone_combined", target_col, cov_col, preds),
        segment_rows(test_full, "year_of_loss", "year_of_loss", target_col, cov_col, preds),
    ], ignore_index=True)
    out.to_csv(tables / "segment_breakdown.csv", index=False)
    log_rows(logger, out, "segment_breakdown.csv")

    logger.info("occupancy_type_fine counts:\n%s", test_full["occupancy_type_fine"].value_counts().to_string())
    logger.info("M1 pct_error by segment (worst first):\n%s",
               out[out.model == "M1"].sort_values("pct_error").to_string(index=False))


if __name__ == "__main__":
    main()
