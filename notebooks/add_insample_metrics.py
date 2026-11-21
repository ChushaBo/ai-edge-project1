"""One-off: add in-sample (train/trainval) rows to outputs/tables/metrics_overall.csv
by SCORING the already-saved, frozen model bundles -- no refit. Not run by
run_all.sh; this exists because the equivalent logic
(07_train_models.in_sample_overall_rows) is now also called from
07_train_models.py's own main(), so a FUTURE full run of 07 regenerates the complete
table on its own. This script applies that same logic to TODAY's already-fit models
without spending the ~90s to refit everything in 07's main().

Loads data/models/{m1,m1_month,m2,m2_masked}_lgbm.joblib and baselines.joblib
(all produced by a prior run of 07_train_models.py), reconstructs train_df/
trainval_df exactly as 07's main() does, computes the 4 new eval_set rows
("train", "trainval", "train_matched", "trainval_matched"), and merges them into the
existing metrics_overall.csv (replacing any rows with the same eval_set from a prior
run of this script, so it is safe to rerun).

Usage: python notebooks/add_insample_metrics.py
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.config import get_path, load_config, load_pipeline_script, require
from utils.log import get_logger, log_rows
from utils.matching import exclude_quarantined
from utils.models import M1_FEATURES, MONTH_FEATURE, text_feature_columns

logger = get_logger("add_insample_metrics")

NEW_EVAL_SETS = ("train", "trainval", "train_matched", "trainval_matched")


def main() -> None:
    cfg = load_config()
    cap = float(require(cfg, "target.cap_share_at"))
    k = int(require(cfg, "pca.n_components"))
    cov_col = require(cfg, "fields.claims_coverage")
    zone_col = "flood_zone_group"

    m7 = load_pipeline_script("07_train_models")
    models_dir, tables = get_path(cfg, "models"), get_path(cfg, "tables")

    meta_cols = ["id", "split", "target", "has_narrative", "quarantined"]
    claims_cols = sorted(set(meta_cols + M1_FEATURES + [MONTH_FEATURE]))
    claims = pd.read_parquet(get_path(cfg, "interim") / "claims_with_text.parquet", columns=claims_cols)
    txt_cols, txtm_cols = text_feature_columns(k, "txt"), text_feature_columns(k, "txtm")
    text_feats = pd.read_parquet(get_path(cfg, "processed") / "text_features.parquet",
                                 columns=["id"] + txt_cols + txtm_cols)
    merged = claims.merge(text_feats, on="id", how="left")

    train_df = exclude_quarantined(merged[merged["split"] == "train"], cfg, model="M1").reset_index(drop=True)
    val_df = exclude_quarantined(merged[merged["split"] == "validation"], cfg, model="M1").reset_index(drop=True)
    trainval_df = pd.concat([train_df, val_df], ignore_index=True)
    log_rows(logger, train_df, "train"); log_rows(logger, trainval_df, "train+validation")

    m1_month_features = M1_FEATURES + [MONTH_FEATURE]
    m2_features = m1_month_features + txt_cols + ["has_narrative"]
    m2_masked_features = m1_month_features + txtm_cols + ["has_narrative"]

    baselines = joblib.load(models_dir / "baselines.joblib")
    bundle_m1 = joblib.load(models_dir / "m1_lgbm.joblib")
    bundle_m1_month = joblib.load(models_dir / "m1_month_lgbm.joblib")
    bundle_m2 = joblib.load(models_dir / "m2_lgbm.joblib")
    bundle_m2_masked = joblib.load(models_dir / "m2_masked_lgbm.joblib")
    logger.info("Loaded frozen bundles from %s (no refit)", models_dir)

    new_rows = m7.in_sample_overall_rows(
        train_df, trainval_df, zone_col, cov_col, cap,
        baselines["b0"], baselines["b1_means"], baselines["b1_overall"],
        bundle_m1["model"], bundle_m1["categories"],
        bundle_m1_month["model"], bundle_m1_month["categories"], m1_month_features,
        bundle_m2["model"], bundle_m2["categories"], m2_features,
        bundle_m2_masked["model"], bundle_m2_masked["categories"], m2_masked_features,
    )
    new_df = pd.DataFrame(new_rows)[
        ["eval_set", "model", "n", "mae_share", "rmse_share", "mae_dollar", "rmse_dollar",
         "actual_total_dollar", "predicted_total_dollar", "pct_error_total"]
    ]

    existing = pd.read_csv(tables / "metrics_overall.csv")
    existing = existing[~existing["eval_set"].isin(NEW_EVAL_SETS)]  # idempotent on rerun
    out = pd.concat([existing, new_df], ignore_index=True)
    out.to_csv(tables / "metrics_overall.csv", index=False)
    log_rows(logger, out, "metrics_overall.csv (updated)")
    logger.info("New in-sample rows:\n%s", new_df.round(4).to_string(index=False))

    print("\n=== In-sample vs. out-of-sample, M1 (pricing model) ===")
    m1_trainval = new_df[(new_df.eval_set == "trainval") & (new_df.model == "M1")].iloc[0]
    print(f"trainval (in-sample):  MAE ${m1_trainval.mae_dollar:,.0f}  "
         f"pct_error_total {m1_trainval.pct_error_total:+.1f}%")
    print("test_full (out-of-sample): see outputs/tables/metrics_overall.csv "
         "(MAE $78,067, pct_error_total -57.6%, from the original 07 run)")


if __name__ == "__main__":
    main()
