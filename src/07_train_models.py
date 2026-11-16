"""Train and evaluate B0, B1, M1, M2, M2_masked.

Models:
  B0 = train+validation mean target.
  B1 = train+validation mean by flood_zone_group (fallback to B0 for an unseen or
       missing zone).
  M1 = LightGBM on the 20 tabular features (utils.models.M1_FEATURES). No month of
       loss, no text: pre-event information only, so M1 can price a policy before a
       flood (used for the shortfall in step 6.2).
  M1_month = diagnostic only, not one of the five reported models. M1's tuned
       hyperparameters, features = M1_FEATURES + month_of_loss. Reported so a gain
       from M2 over M1 cannot be mistaken for M2 having learned the season instead of
       the storm.
  M2 = M1_FEATURES + month_of_loss + txt_pc01..NN (raw-text PCA) + has_narrative.
  M2_masked = same, with txtm_pc01..NN (damage-figure-masked PCA) instead of txt_pc*.

Evaluation sets (quarantine is model-scoped, docs/decisions.md "Helene quarantine
applies to the text models only"):
  test_full    = every test claim (112,573). Scores B0, B1, M1 -- the headline numbers
                 that step 6.2's shortfall depends on.
  test_matched = test claims with a narrative (64,620; the Helene-quarantined claims
                 have none and are excluded). Scores B0, B1, M1, M1_month, M2,
                 M2_masked -- the only set where M1 vs M2 is a fair comparison, since
                 both see the same information.
Training rows (train+validation, 46,335) are identical across every model; only the
feature columns differ. Tuning (small grid, L2/regression objective only -- see
docs/decisions.md "M1/M2 objective fixed to L2") uses validation alone; the chosen
configuration is refit on train+validation and the test set is touched exactly once,
in evaluate_all() below.

Inputs:
  - data/interim/claims_with_text.parquet
  - data/processed/text_features.parquet
  - config.yaml: allowed_features, derived_features, target.*, splits.*, fields.*,
    quarantine.*, models.*, pca.n_components, random_seed, paths.*
Outputs:
  - data/models/baselines.joblib, m1_lgbm.joblib, m1_month_lgbm.joblib, m2_lgbm.joblib,
    m2_masked_lgbm.joblib
  - outputs/tables/metrics_validation.csv (every grid point tried, plus the L1
    diagnostic row per model)
  - outputs/tables/metrics_overall.csv (MAE, RMSE in target units and dollars, and
    portfolio total / percent error, by eval_set x model)
  - outputs/tables/metrics_by_event.csv (per storm event or date cluster: predicted vs.
    actual total payout and percent error, by eval_set x model)
  - outputs/figures/mae_by_model.png
Tuning uses validation only; the test set is evaluated once.
"""

import itertools
import random
import time

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.config import get_path, load_config, require
from utils.log import get_logger, log_rows
from utils.matching import exclude_quarantined
from utils.models import (M1_FEATURES, MONTH_FEATURE, dollar_metrics, encode_features,
                          event_groups, event_summary, fit_categories, text_feature_columns)

logger = get_logger("07_train_models")

TARGET = "target"
# Palette: dataviz skill reference palette, categorical slots 1-6 in their validated
# adjacent order (light mode; this is a static report figure, no dark-mode toggle).
MODEL_COLORS = {
    "B0": "#2a78d6", "B1": "#eb6834", "M1": "#1baf7a",
    "M1_month": "#eda100", "M2": "#e87ba4", "M2_masked": "#008300",
}
INK, SECONDARY_INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def fit_b0(df: pd.DataFrame) -> float:
    return float(df[TARGET].mean())


def fit_b1(df: pd.DataFrame, zone_col: str) -> tuple[dict, float]:
    means = df.groupby(zone_col)[TARGET].mean().to_dict()
    return means, float(df[TARGET].mean())


def predict_b1(df: pd.DataFrame, zone_col: str, means: dict, overall: float) -> np.ndarray:
    return df[zone_col].map(means).fillna(overall).to_numpy(dtype="float64")


# ---------------------------------------------------------------------------
# LightGBM: validation-only grid search (L2 only), train-only categories, final refit
# ---------------------------------------------------------------------------

def _fit_one(X_train, y_train, X_val, y_val, feature_cols, categories, params, esr):
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="l1",
             callbacks=[lgb.early_stopping(esr, verbose=False)])
    pred = model.predict(X_val, num_iteration=model.best_iteration_)
    return model, float(np.mean(np.abs(pred - y_val)))


def tune_lgbm(train_df, val_df, feature_cols, seed, grid, fixed_params, objective, esr, max_n):
    """Grid search on validation MAE (L2 objective only). Returns the winning
    params dict (+ best_iteration), every grid result, and the train-only categories."""
    categories = fit_categories(train_df, feature_cols)
    X_train = encode_features(train_df, feature_cols, categories)
    X_val = encode_features(val_df, feature_cols, categories)
    y_train, y_val = train_df[TARGET].to_numpy(), val_df[TARGET].to_numpy()

    results, best = [], None
    combos = list(itertools.product(grid["num_leaves"], grid["learning_rate"], grid["min_child_samples"]))
    for num_leaves, lr, mcs in combos:
        params = dict(fixed_params, objective=objective, num_leaves=num_leaves, learning_rate=lr,
                      min_child_samples=mcs, n_estimators=max_n, random_state=seed, verbosity=-1)
        model, mae = _fit_one(X_train, y_train, X_val, y_val, feature_cols, categories, params, esr)
        row = {"num_leaves": num_leaves, "learning_rate": lr, "min_child_samples": mcs,
              "objective": objective, "best_iteration": int(model.best_iteration_),
              "val_mae_share": mae, "diagnostic_only": False}
        results.append(row)
        if best is None or mae < best["val_mae_share"]:
            best = row
    logger.info("  grid: %d combinations, best val MAE %.4f (num_leaves=%d, lr=%.3f, min_child_samples=%d, iter=%d)",
               len(combos), best["val_mae_share"], best["num_leaves"], best["learning_rate"],
               best["min_child_samples"], best["best_iteration"])
    return best, results, categories


def l1_diagnostic(train_df, val_df, feature_cols, categories, seed, fixed_params, best, l1_objective, esr, max_n):
    """One L1 (regression_l1) fit at the winning L2 hyperparameters, logged as a
    validation-only diagnostic (never selected; see docs/decisions.md)."""
    X_train = encode_features(train_df, feature_cols, categories)
    X_val = encode_features(val_df, feature_cols, categories)
    y_train, y_val = train_df[TARGET].to_numpy(), val_df[TARGET].to_numpy()
    params = dict(fixed_params, objective=l1_objective, num_leaves=best["num_leaves"],
                 learning_rate=best["learning_rate"], min_child_samples=best["min_child_samples"],
                 n_estimators=max_n, random_state=seed, verbosity=-1)
    model, mae = _fit_one(X_train, y_train, X_val, y_val, feature_cols, categories, params, esr)
    logger.info("  L1 diagnostic (same other params): val MAE %.4f (L2 chosen: %.4f)", mae, best["val_mae_share"])
    return {"num_leaves": best["num_leaves"], "learning_rate": best["learning_rate"],
           "min_child_samples": best["min_child_samples"], "objective": l1_objective,
           "best_iteration": int(model.best_iteration_), "val_mae_share": mae, "diagnostic_only": True}


def final_fit(trainval_df, feature_cols, categories, seed, best, fixed_params, objective, train_rows):
    """Refit on train+validation at the winning hyperparameters, n_estimators scaled
    from the validation-selected best_iteration by the growth in training rows."""
    scale = len(trainval_df) / train_rows
    n_estimators = max(10, round(best["best_iteration"] * scale))
    params = dict(fixed_params, objective=objective, num_leaves=best["num_leaves"],
                 learning_rate=best["learning_rate"], min_child_samples=best["min_child_samples"],
                 n_estimators=n_estimators, random_state=seed, verbosity=-1)
    X = encode_features(trainval_df, feature_cols, categories)
    model = lgb.LGBMRegressor(**params)
    model.fit(X, trainval_df[TARGET].to_numpy())
    logger.info("  final fit: %d rows, n_estimators=%d (scaled from best_iteration=%d x %.3f)",
               len(trainval_df), n_estimators, best["best_iteration"], scale)
    return model


def predict_clipped(model, df, feature_cols, categories, cap):
    X = encode_features(df, feature_cols, categories)
    return np.clip(model.predict(X), 0.0, cap)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.perf_counter()
    cfg = load_config()
    seed = int(require(cfg, "random_seed"))
    set_seeds(seed)

    models_cfg = require(cfg, "models")
    grid, fixed_params = models_cfg["search_grid"], models_cfg["fixed_params"]
    objective, l1_objective = models_cfg["objective"], models_cfg["l1_diagnostic_objective"]
    esr, max_n = int(models_cfg["early_stopping_rounds"]), int(models_cfg["max_n_estimators"])
    cap = float(require(cfg, "target.cap_share_at"))
    k = int(require(cfg, "pca.n_components"))
    date_col = require(cfg, "fields.claims_date_of_loss")
    cov_col = require(cfg, "fields.claims_coverage")
    zone_col = "flood_zone_group"

    interim, processed = get_path(cfg, "interim"), get_path(cfg, "processed")
    models_dir, tables, figures = get_path(cfg, "models"), get_path(cfg, "tables"), get_path(cfg, "figures")

    # --- Load and merge ---------------------------------------------------
    meta_cols = ["id", "split", TARGET, "has_narrative", "quarantined",
                "matched_episode_ids", "matched_event_types", date_col, cov_col]
    claims_cols = sorted(set(meta_cols + M1_FEATURES + [MONTH_FEATURE]))
    claims = pd.read_parquet(interim / "claims_with_text.parquet", columns=claims_cols)
    log_rows(logger, claims, "claims_with_text (selected columns)")

    txt_cols, txtm_cols = text_feature_columns(k, "txt"), text_feature_columns(k, "txtm")
    text_feats = pd.read_parquet(processed / "text_features.parquet", columns=["id"] + txt_cols + txtm_cols)
    merged = claims.merge(text_feats, on="id", how="left")
    log_rows(logger, merged, "claims merged with text PCA features")

    train_df = exclude_quarantined(merged[merged["split"] == "train"], cfg, model="M1").reset_index(drop=True)
    val_df = exclude_quarantined(merged[merged["split"] == "validation"], cfg, model="M1").reset_index(drop=True)
    trainval_df = pd.concat([train_df, val_df], ignore_index=True)
    test_full = merged[merged["split"] == "test"].reset_index(drop=True)
    test_matched = test_full[test_full["has_narrative"]].reset_index(drop=True)
    if test_matched["quarantined"].any():
        raise SystemExit("STOP: a quarantined claim has has_narrative=True; quarantine rule is inconsistent")

    log_rows(logger, train_df, "train"); log_rows(logger, val_df, "validation")
    log_rows(logger, trainval_df, "train+validation (final fit)")
    log_rows(logger, test_full, "test_full (eval set, B0/B1/M1)")
    log_rows(logger, test_matched, "test_matched (eval set, all models)")

    m1_month_features = M1_FEATURES + [MONTH_FEATURE]
    m2_features = m1_month_features + txt_cols + ["has_narrative"]
    m2_masked_features = m1_month_features + txtm_cols + ["has_narrative"]

    # --- Baselines (no tuning; train+validation) ---------------------------
    b0 = fit_b0(trainval_df)
    b1_means, b1_overall = fit_b1(trainval_df, zone_col)
    joblib.dump({"b0": b0, "b1_means": b1_means, "b1_overall": b1_overall,
                "zone_col": zone_col, "fit_rows": len(trainval_df)}, models_dir / "baselines.joblib")
    logger.info("B0 (train+val mean) = %.4f; B1 zones: %s", b0, {k: round(v, 3) for k, v in b1_means.items()})

    validation_rows = []
    for name, pred_val in (("B0", np.full(len(val_df), fit_b0(train_df))),
                           ("B1", predict_b1(val_df, zone_col, *fit_b1(train_df, zone_col)))):
        m = dollar_metrics(val_df[TARGET].to_numpy(), pred_val, val_df[cov_col].to_numpy())
        validation_rows.append({"model": name, "num_leaves": None, "learning_rate": None,
                               "min_child_samples": None, "objective": "n/a (no tuning)",
                               "best_iteration": None, "val_mae_share": m["mae_share"],
                               "diagnostic_only": False})

    # --- M1 ------------------------------------------------------------------
    logger.info("Tuning M1 (%d features)...", len(M1_FEATURES))
    best_m1, results_m1, cats_m1 = tune_lgbm(train_df, val_df, M1_FEATURES, seed, grid,
                                             fixed_params, objective, esr, max_n)
    for r in results_m1:
        r["model"] = "M1"
    validation_rows += results_m1
    diag = l1_diagnostic(train_df, val_df, M1_FEATURES, cats_m1, seed, fixed_params, best_m1, l1_objective, esr, max_n)
    diag["model"] = "M1"
    validation_rows.append(diag)
    m1_model = final_fit(trainval_df, M1_FEATURES, cats_m1, seed, best_m1, fixed_params, objective, len(train_df))
    joblib.dump({"model": m1_model, "feature_cols": M1_FEATURES, "categories": cats_m1,
                "best_params": best_m1}, models_dir / "m1_lgbm.joblib")

    # --- M1_month (diagnostic: M1's tuned params + month_of_loss) ----------
    logger.info("Fitting M1_month diagnostic (M1's tuned params + month_of_loss)...")
    cats_m1_month = fit_categories(train_df, m1_month_features)
    m1_month_model = final_fit(trainval_df, m1_month_features, cats_m1_month, seed, best_m1,
                               fixed_params, objective, len(train_df))
    joblib.dump({"model": m1_month_model, "feature_cols": m1_month_features,
                "categories": cats_m1_month, "best_params": best_m1}, models_dir / "m1_month_lgbm.joblib")

    # --- M2 --------------------------------------------------------------
    logger.info("Tuning M2 (%d features)...", len(m2_features))
    best_m2, results_m2, cats_m2 = tune_lgbm(train_df, val_df, m2_features, seed, grid,
                                             fixed_params, objective, esr, max_n)
    for r in results_m2:
        r["model"] = "M2"
    validation_rows += results_m2
    diag = l1_diagnostic(train_df, val_df, m2_features, cats_m2, seed, fixed_params, best_m2, l1_objective, esr, max_n)
    diag["model"] = "M2"
    validation_rows.append(diag)
    m2_model = final_fit(trainval_df, m2_features, cats_m2, seed, best_m2, fixed_params, objective, len(train_df))
    joblib.dump({"model": m2_model, "feature_cols": m2_features, "categories": cats_m2,
                "best_params": best_m2}, models_dir / "m2_lgbm.joblib")

    # --- M2_masked ---------------------------------------------------------
    logger.info("Tuning M2_masked (%d features)...", len(m2_masked_features))
    best_m2m, results_m2m, cats_m2m = tune_lgbm(train_df, val_df, m2_masked_features, seed, grid,
                                                fixed_params, objective, esr, max_n)
    for r in results_m2m:
        r["model"] = "M2_masked"
    validation_rows += results_m2m
    diag = l1_diagnostic(train_df, val_df, m2_masked_features, cats_m2m, seed, fixed_params, best_m2m, l1_objective, esr, max_n)
    diag["model"] = "M2_masked"
    validation_rows.append(diag)
    m2_masked_model = final_fit(trainval_df, m2_masked_features, cats_m2m, seed, best_m2m,
                                fixed_params, objective, len(train_df))
    joblib.dump({"model": m2_masked_model, "feature_cols": m2_masked_features,
                "categories": cats_m2m, "best_params": best_m2m}, models_dir / "m2_masked_lgbm.joblib")

    pd.DataFrame(validation_rows).to_csv(tables / "metrics_validation.csv", index=False)
    logger.info("Saved metrics_validation.csv (%d rows)", len(validation_rows))

    # --- Evaluate once on test ---------------------------------------------
    preds = {"test_full": {}, "test_matched": {}}
    preds["test_full"]["B0"] = np.full(len(test_full), b0)
    preds["test_full"]["B1"] = predict_b1(test_full, zone_col, b1_means, b1_overall)
    preds["test_full"]["M1"] = predict_clipped(m1_model, test_full, M1_FEATURES, cats_m1, cap)

    preds["test_matched"]["B0"] = np.full(len(test_matched), b0)
    preds["test_matched"]["B1"] = predict_b1(test_matched, zone_col, b1_means, b1_overall)
    preds["test_matched"]["M1"] = predict_clipped(m1_model, test_matched, M1_FEATURES, cats_m1, cap)
    preds["test_matched"]["M1_month"] = predict_clipped(m1_month_model, test_matched, m1_month_features, cats_m1_month, cap)
    preds["test_matched"]["M2"] = predict_clipped(m2_model, test_matched, m2_features, cats_m2, cap)
    preds["test_matched"]["M2_masked"] = predict_clipped(m2_masked_model, test_matched, m2_masked_features, cats_m2m, cap)

    eval_frames = {"test_full": test_full, "test_matched": test_matched}
    overall_rows = []
    for eval_name, frame in eval_frames.items():
        for model_name, pred in preds[eval_name].items():
            m = dollar_metrics(frame[TARGET].to_numpy(), pred, frame[cov_col].to_numpy())
            m.update({"eval_set": eval_name, "model": model_name})
            overall_rows.append(m)
    overall = pd.DataFrame(overall_rows)[
        ["eval_set", "model", "n", "mae_share", "rmse_share", "mae_dollar", "rmse_dollar",
         "actual_total_dollar", "predicted_total_dollar", "pct_error_total"]
    ]
    overall.to_csv(tables / "metrics_overall.csv", index=False)
    logger.info("Saved metrics_overall.csv:\n%s", overall.round(4).to_string(index=False))

    # --- Per-event table -----------------------------------------------------
    event_rows = []
    for eval_name, frame in eval_frames.items():
        group = event_groups(frame, date_col)
        for model_name, pred in preds[eval_name].items():
            d = frame.assign(group=group, _pred=pred)
            summary = event_summary(d, "group", date_col, TARGET, "_pred", cov_col, type_col="matched_event_types")
            summary["model"], summary["eval_set"] = model_name, eval_name
            event_rows.append(summary)
    by_event = pd.concat(event_rows, ignore_index=True)[
        ["eval_set", "model", "group", "event_types", "date_min", "date_max", "n_claims",
         "actual_total_dollar", "predicted_total_dollar", "pct_error"]
    ]
    by_event.to_csv(tables / "metrics_by_event.csv", index=False)
    logger.info("Saved metrics_by_event.csv (%d rows)", len(by_event))

    # --- Figure ---------------------------------------------------------
    plot_mae_by_model(overall, figures / "mae_by_model.png")

    # --- Plain-English summary ------------------------------------------
    print_summary(overall)
    logger.info("Total runtime %.1f s", time.perf_counter() - t0)


def plot_mae_by_model(overall: pd.DataFrame, out_path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor=SURFACE)
    panels = [("test_full", ["B0", "B1", "M1"], "All test claims (112,573)"),
             ("test_matched", ["B0", "B1", "M1", "M1_month", "M2", "M2_masked"],
              "Test claims with a narrative (64,620)")]
    for ax, (eval_set, order, subtitle) in zip(axes, panels):
        d = overall[overall["eval_set"] == eval_set].set_index("model").loc[order]
        colors = [MODEL_COLORS[m] for m in order]
        ax.set_facecolor(SURFACE)
        bars = ax.bar(order, d["mae_dollar"], color=colors, width=0.6)
        ax.bar_label(bars, labels=[f"${v:,.0f}" for v in d["mae_dollar"]], padding=3,
                    color=INK, fontsize=9)
        ax.set_title(subtitle, color=INK, fontsize=11)
        ax.set_ylabel("Mean absolute error ($ building payout per claim)", color=SECONDARY_INK, fontsize=9)
        ax.tick_params(colors=MUTED)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRID)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_ylim(0, d["mae_dollar"].max() * 1.2)
    fig.suptitle("Claim severity model error: MAE across models", color=INK, fontsize=13, y=1.02)
    fig.text(0.5, -0.02,
             "Source: outputs/tables/metrics_overall.csv (src/07_train_models.py). "
             "MAE = |predicted - actual| building payout per claim, test set, evaluated once.",
             ha="center", color=MUTED, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def print_summary(overall: pd.DataFrame) -> None:
    def row(eval_set, model):
        r = overall[(overall["eval_set"] == eval_set) & (overall["model"] == model)].iloc[0]
        return r

    full = {m: row("test_full", m) for m in ("B0", "B1", "M1")}
    matched = {m: row("test_matched", m) for m in ("B0", "B1", "M1", "M1_month", "M2", "M2_masked")}

    lines = ["", "=== Plain-English summary (RUNBOOK 5.1) ===",
            f"On the full test set (112,573 claims), M1 (tabular model) has MAE "
            f"${full['M1']['mae_dollar']:,.0f} per claim vs. B1 (flood-zone benchmark) "
            f"${full['B1']['mae_dollar']:,.0f} -- "
            f"{'a' if full['M1']['mae_dollar'] < full['B1']['mae_dollar'] else 'no'} improvement.",
            f"Portfolio totals on test_full: actual ${full['M1']['actual_total_dollar']/1e6:,.1f}M; "
            f"M1 predicts ${full['M1']['predicted_total_dollar']/1e6:,.1f}M "
            f"({full['M1']['pct_error_total']:+.1f}%); B1 predicts "
            f"${full['B1']['predicted_total_dollar']/1e6:,.1f}M ({full['B1']['pct_error_total']:+.1f}%).",
            f"On the matched subset (64,620 claims with a storm narrative), M2 (tabular + text) has "
            f"MAE ${matched['M2']['mae_dollar']:,.0f} vs. M1's ${matched['M1']['mae_dollar']:,.0f} "
            f"on the SAME claims -- "
            f"{'text improves on tabular alone' if matched['M2']['mae_dollar'] < matched['M1']['mae_dollar'] else 'text does not improve on tabular alone'}.",
            f"M2_masked (damage figures replaced with placeholders) MAE "
            f"${matched['M2_masked']['mae_dollar']:,.0f} -- "
            f"{'keeps most of' if matched['M2_masked']['mae_dollar'] < matched['M1']['mae_dollar'] else 'loses'} "
            f"M2's gain over M1, so the text signal is "
            f"{'not' if matched['M2_masked']['mae_dollar'] < matched['M1']['mae_dollar'] else 'mostly'} just leaked damage dollars.",
            f"M1_month (M1 + month of loss, no text) MAE ${matched['M1_month']['mae_dollar']:,.0f} -- "
            f"compare to M2 to see how much of any text gain is really a seasonal effect.",
            ]
    for line in lines:
        print(line)
        logger.info(line)


if __name__ == "__main__":
    main()
