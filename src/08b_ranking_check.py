"""Ranking check for M1 and B1: do their predictions put claims in the right order,
even where the point predictions themselves are biased?

PROVISIONAL, alongside the rest of 07-11's outputs (docs/decisions.md open item: M1
underpredicts the test portfolio total). This uses the FROZEN m1_lgbm.joblib and
baselines.joblib from 07_train_models.py -- no refit, nothing retrained here or
anywhere downstream.

Method, on test_full (112,573 claims, both models scored with no quarantine
restriction since neither M1 nor B1 is in quarantine.applies_to):
  - Decile table: assign each claim a decile 1-10 by RANK of its predicted share
    (ties broken by original row order, so every decile has exactly n/10 claims --
    B1 only takes 5 distinct values statewide, so a value-based qcut would not give
    10 usable groups). Report, per decile: n, mean predicted share, mean actual
    share, total actual payout ($).
  - Spearman rank correlation between predicted and actual share: unlike MAE/RMSE,
    this is invariant to a model being systematically biased (over- or
    under-predicting by a constant factor) and only measures whether the ORDERING
    of claims by predicted risk matches their actual ordering.
  - Gini coefficient for risk ranking (the standard actuarial definition, e.g.
    Goldburd, Khare & Tevet, "Generalized Linear Models for Insurance Rating"):
    sort claims ascending by predicted share, plot cumulative % of actual payout
    against cumulative % of claims (a Lorenz curve), Gini = 1 - 2 x (area under that
    curve). 0 = no separation between low- and high-risk claims; higher = better
    separation. Computed in the same $ units for both models so they are directly
    comparable.

Inputs:
  - data/models/m1_lgbm.joblib, baselines.joblib (both frozen, from 07_train_models.py)
  - data/interim/claims_with_text.parquet (test_full)
  - config.yaml: fields.*, target.cap_share_at, random_seed, paths.*
Outputs:
  - outputs/tables/ranking_check_deciles.csv
  - outputs/tables/ranking_check_summary.csv (Spearman rho, Gini, by model)
  - outputs/figures/ranking_check_lift_chart.png
"""

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from utils.config import get_path, load_config, load_pipeline_script, require
from utils.log import get_logger, log_rows
from utils.models import M1_FEATURES

logger = get_logger("08b_ranking_check")
N_DECILES = 10
INK, SECONDARY_INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
MODEL_COLORS = {"M1": "#1baf7a", "B1": "#eb6834"}


def assign_deciles(pred: np.ndarray) -> np.ndarray:
    """Decile 1 (lowest predicted risk) .. 10 (highest), by RANK so every decile has
    exactly n/10 claims regardless of how many distinct predicted values exist."""
    ranks = pd.Series(pred).rank(method="first")
    return pd.qcut(ranks, N_DECILES, labels=False) + 1


def decile_table(df: pd.DataFrame, pred: np.ndarray, target_col: str, cov_col: str, model_name: str) -> pd.DataFrame:
    d = df.assign(decile=assign_deciles(pred), pred_share=pred,
                 actual_dollar=df[target_col] * df[cov_col])
    out = d.groupby("decile").agg(
        n=("decile", "size"),
        mean_predicted_share=("pred_share", "mean"),
        mean_actual_share=(target_col, "mean"),
        total_actual_payout_dollar=("actual_dollar", "sum"),
    ).reset_index()
    out["model"] = model_name
    return out


def gini_risk_ranking(y_actual_dollar: np.ndarray, y_pred_share: np.ndarray) -> float:
    """Lorenz-curve Gini: sort ascending by predicted risk, cumulative % of actual
    payout vs. cumulative % of claims. Gini = 1 - 2 x area under that curve."""
    order = np.argsort(y_pred_share, kind="mergesort")
    cum_loss = np.cumsum(y_actual_dollar[order])
    cum_loss_pct = cum_loss / cum_loss[-1]
    n = len(y_actual_dollar)
    cum_pop_pct = np.arange(1, n + 1) / n
    x = np.concatenate([[0.0], cum_pop_pct])
    y = np.concatenate([[0.0], cum_loss_pct])
    area = np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x)
    return 1 - 2 * area


def plot_lift_chart(deciles: pd.DataFrame, path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for model_name, color in MODEL_COLORS.items():
        d = deciles[deciles["model"] == model_name].sort_values("decile")
        ax.plot(d["decile"], d["mean_actual_share"], marker="o", color=color,
               label=f"{model_name}: actual", linewidth=2)
        ax.plot(d["decile"], d["mean_predicted_share"], marker="o", color=color,
               label=f"{model_name}: predicted", linewidth=1.3, linestyle="--", alpha=0.7)
    ax.set_xlabel("Decile of predicted risk (1 = lowest, 10 = highest)", color=SECONDARY_INK)
    ax.set_ylabel("Mean share of building coverage", color=SECONDARY_INK)
    ax.set_title("Lift chart: M1 vs. B1 risk ranking on test_full (112,573 claims)", color=INK, fontsize=13)
    ax.set_xticks(range(1, N_DECILES + 1))
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    fig.text(0.5, -0.02,
             "Source: outputs/tables/ranking_check_deciles.csv (src/08b_ranking_check.py), "
             "frozen M1/B1 from 07_train_models.py, no refit. PROVISIONAL pending RUNBOOK 5.2.",
             ha="center", color=MUTED, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    logger.info("Saved %s", path)


def main() -> None:
    cfg = load_config()
    cap = float(require(cfg, "target.cap_share_at"))
    cov_col = require(cfg, "fields.claims_coverage")
    target_col = "target"
    zone_col = "flood_zone_group"
    tables, figures, models_dir = get_path(cfg, "tables"), get_path(cfg, "figures"), get_path(cfg, "models")

    m7 = load_pipeline_script("07_train_models")
    bundle_m1 = joblib.load(models_dir / "m1_lgbm.joblib")
    baselines = joblib.load(models_dir / "baselines.joblib")

    cols = sorted(set(["id", "split", target_col, cov_col, zone_col] + M1_FEATURES))
    claims = pd.read_parquet(get_path(cfg, "interim") / "claims_with_text.parquet", columns=cols)
    test_full = claims[claims["split"] == "test"].reset_index(drop=True)
    log_rows(logger, test_full, "test_full (frozen M1 + B1, no refit)")

    pred_m1 = m7.predict_clipped(bundle_m1["model"], test_full, bundle_m1["feature_cols"],
                                 bundle_m1["categories"], cap)
    pred_b1 = np.clip(m7.predict_b1(test_full, zone_col, baselines["b1_means"], baselines["b1_overall"]),
                      0.0, cap)
    n_distinct_b1 = pd.Series(pred_b1).nunique()
    logger.info("B1 has only %d distinct predicted values statewide (zone means); "
               "deciles below are RANK-based (ties broken arbitrarily), not value-based", n_distinct_b1)

    deciles = pd.concat([
        decile_table(test_full, pred_m1, target_col, cov_col, "M1"),
        decile_table(test_full, pred_b1, target_col, cov_col, "B1"),
    ], ignore_index=True)
    deciles = deciles[["model", "decile", "n", "mean_predicted_share", "mean_actual_share",
                      "total_actual_payout_dollar"]].sort_values(["model", "decile"])
    deciles.to_csv(tables / "ranking_check_deciles.csv", index=False)
    log_rows(logger, deciles, "ranking_check_deciles.csv")

    actual_dollar = (test_full[target_col] * test_full[cov_col]).to_numpy()
    summary_rows = []
    for name, pred in (("M1", pred_m1), ("B1", pred_b1)):
        rho, pval = spearmanr(pred, test_full[target_col].to_numpy())
        gini = gini_risk_ranking(actual_dollar, pred)
        summary_rows.append({"model": name, "spearman_rho": rho, "spearman_pvalue": pval, "gini": gini})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(tables / "ranking_check_summary.csv", index=False)
    logger.info("Ranking summary (test_full, frozen models, no refit):\n%s",
               summary.round(4).to_string(index=False))

    plot_lift_chart(deciles, figures / "ranking_check_lift_chart.png")

    m1_row, b1_row = summary[summary["model"] == "M1"].iloc[0], summary[summary["model"] == "B1"].iloc[0]
    print("\n=== Does M1 rank risk better than B1? ===")
    print(f"Spearman rho: M1 {m1_row['spearman_rho']:.3f} vs. B1 {b1_row['spearman_rho']:.3f} "
         f"({'M1 higher' if m1_row['spearman_rho'] > b1_row['spearman_rho'] else 'B1 higher or tied'}); "
         f"Gini: M1 {m1_row['gini']:.3f} vs. B1 {b1_row['gini']:.3f} "
         f"({'M1 higher' if m1_row['gini'] > b1_row['gini'] else 'B1 higher or tied'}).")
    print("See outputs/figures/ranking_check_lift_chart.png and "
         "outputs/tables/ranking_check_deciles.csv for the full picture.")


if __name__ == "__main__":
    main()
