"""Ranking check for M1 and B1: do their predictions put claims in the right order,
even where the point predictions themselves are biased?

PROVISIONAL, alongside the rest of 07-11's outputs (docs/decisions.md open item: M1
underpredicts the test portfolio total). This uses the FROZEN m1_lgbm.joblib and
baselines.joblib from 07_train_models.py -- no refit, nothing retrained here or
anywhere downstream.

Method, on test_full (112,573 claims, both models scored with no quarantine
restriction since neither M1 nor B1 is in quarantine.applies_to). Every metric below
is computed TWICE, once sorting/ranking by predicted SHARE and once by predicted
DOLLARS (predicted share x coverage) -- labeled by the `sort_by` column in both
output tables, so both are kept, not just the latest:
  - Decile table: assign each claim a decile 1-10 by RANK of the sort key (ties
    broken by original row order, so every decile has exactly n/10 claims -- B1
    only takes ~4-5 distinct predicted-share values statewide, so a value-based
    qcut would not give 10 usable groups). Report, per decile: n, mean predicted
    share, mean actual share, total actual payout ($), and the share of that
    decile's claims paying over $1M (the concentration check requested after the
    first run: which decile holds the biggest claims).
  - Spearman rank correlation between the sort key and the matching actual value
    (share vs. share, or dollars vs. dollars): unlike MAE/RMSE, this is invariant to
    a model being systematically biased (over- or under-predicting by a constant
    factor) and only measures whether the ORDERING matches.
  - Gini coefficient for risk ranking (the standard actuarial definition, e.g.
    Goldburd, Khare & Tevet, "Generalized Linear Models for Insurance Rating"):
    sort claims ascending by the sort key, plot cumulative % of actual payout
    against cumulative % of claims (a Lorenz curve), Gini = 1 - 2 x (area under that
    curve). 0 = no separation between low- and high-risk claims; higher = better
    separation. Always computed in $ units (actual payout), only the SORT ORDER
    changes between the two versions -- this is the metric that mattered most: the
    first run sorted by predicted SHARE only, which ignores coverage size entirely
    and is why it disagreed with the share-based Spearman result.

Also, separately (no model involved): a within-storm variance check. On the claims
matched to a NOAA narrative (quarantined claims excluded, same scope as M2), split the
variance of the payout-share target into between-narrative and within-narrative parts
per split. Text features are identical for every claim sharing a narrative, so the
between-narrative share is an upper bound on what storm text can explain.

Inputs:
  - data/models/m1_lgbm.joblib, baselines.joblib (both frozen, from 07_train_models.py)
  - data/interim/claims_with_text.parquet (test_full; all splits for the variance check)
  - config.yaml: fields.*, target.cap_share_at, quarantine.enabled, random_seed, paths.*
Outputs:
  - outputs/tables/within_storm_variance.csv (per split: n claims, n distinct
    narratives, mean target, between- and within-narrative % of variance)
  - outputs/tables/ranking_check_deciles.csv (sort_by column: predicted_share /
    predicted_dollar)
  - outputs/tables/ranking_check_summary.csv (Spearman rho, Gini, by model x sort_by)
  - outputs/figures/ranking_check_lift_chart.png (share-sort only, from the first run;
    not regenerated here -- data only, per instruction)
"""

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from utils.config import get_path, load_config, load_pipeline_script, require
from utils.log import get_logger, log_rows
from utils.models import M1_FEATURES

logger = get_logger("08b_ranking_check")
N_DECILES = 10


def assign_deciles(pred: np.ndarray) -> np.ndarray:
    """Decile 1 (lowest predicted risk) .. 10 (highest), by RANK so every decile has
    exactly n/10 claims regardless of how many distinct predicted values exist."""
    ranks = pd.Series(pred).rank(method="first")
    return pd.qcut(ranks, N_DECILES, labels=False) + 1


def decile_table(df: pd.DataFrame, pred_share: np.ndarray, sort_key: np.ndarray, target_col: str,
                 cov_col: str, model_name: str, sort_by: str) -> pd.DataFrame:
    d = df.assign(decile=assign_deciles(sort_key), pred_share=pred_share,
                 actual_dollar=df[target_col] * df[cov_col])
    out = d.groupby("decile").agg(
        n=("decile", "size"),
        mean_predicted_share=("pred_share", "mean"),
        mean_actual_share=(target_col, "mean"),
        total_actual_payout_dollar=("actual_dollar", "sum"),
        pct_claims_over_1M=("actual_dollar", lambda s: 100 * (s > 1_000_000).mean()),
    ).reset_index()
    out["model"], out["sort_by"] = model_name, sort_by
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


def between_group_variance_share(y: np.ndarray, groups: np.ndarray) -> float:
    """Share of the (population) variance of y explained by group means:
    Var(E[y | group]) / Var(y), in [0, 1]. The remainder (1 - this) is within-group.
    Returns NaN when y has zero variance."""
    y = pd.Series(np.asarray(y, dtype=float))
    total = y.var(ddof=0)
    if total == 0:
        return float("nan")
    group_mean = y.groupby(np.asarray(groups)).transform("mean")
    return float(group_mean.var(ddof=0) / total)


def within_storm_variance(claims: pd.DataFrame, target_col: str, quarantine_on: bool) -> pd.DataFrame:
    """How much of the payout-share variance could storm text explain, even in principle?

    Every claim matched to the same NOAA narrative gets the same text features, so a
    text feature can at most separate narratives from each other (between-narrative
    variance), never claims within one narrative. This is an upper bound for M2's
    text component, computed in-sample per split: a perfect narrative -> mean-share
    lookup. Uses matched claims only, minus quarantined claims when quarantine is
    enabled (same scope as M2 / M2_masked)."""
    d = claims[claims["has_narrative"] & claims[target_col].notna()]
    if quarantine_on:
        d = d[~d["quarantined"]]
    rows = []
    for split in ("train", "validation", "test"):
        s = d[d["split"] == split]
        between = between_group_variance_share(s[target_col].to_numpy(), s["narrative"].to_numpy())
        rows.append({"split": split, "n_claims": len(s), "n_distinct_narratives": s["narrative"].nunique(),
                     "mean_target_share": s[target_col].mean(),
                     "between_narrative_pct": 100 * between, "within_narrative_pct": 100 * (1 - between)})
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    cap = float(require(cfg, "target.cap_share_at"))
    cov_col = require(cfg, "fields.claims_coverage")
    target_col = "target"
    zone_col = "flood_zone_group"
    tables, models_dir = get_path(cfg, "tables"), get_path(cfg, "models")

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

    coverage = test_full[cov_col].to_numpy()
    actual_dollar = (test_full[target_col] * test_full[cov_col]).to_numpy()
    pred_dollar_m1, pred_dollar_b1 = pred_m1 * coverage, pred_b1 * coverage

    # sort_by: "predicted_share" (the first run's sort key) and "predicted_dollar"
    # (predicted share x coverage) -- BOTH kept, not just the latest.
    sort_keys = {
        "predicted_share": {"M1": pred_m1, "B1": pred_b1},
        "predicted_dollar": {"M1": pred_dollar_m1, "B1": pred_dollar_b1},
    }
    actual_for_spearman = {"predicted_share": test_full[target_col].to_numpy(),
                           "predicted_dollar": actual_dollar}

    deciles = pd.concat([
        decile_table(test_full, pred, sort_keys[sort_by][name], target_col, cov_col, name, sort_by)
        for sort_by in sort_keys for name, pred in (("M1", pred_m1), ("B1", pred_b1))
    ], ignore_index=True)
    deciles = deciles[["sort_by", "model", "decile", "n", "mean_predicted_share", "mean_actual_share",
                      "total_actual_payout_dollar", "pct_claims_over_1M"]].sort_values(
        ["sort_by", "model", "decile"])
    deciles.to_csv(tables / "ranking_check_deciles.csv", index=False)
    log_rows(logger, deciles, "ranking_check_deciles.csv (both sort_by versions)")

    summary_rows = []
    for sort_by, keys in sort_keys.items():
        for name, key in keys.items():
            rho, pval = spearmanr(key, actual_for_spearman[sort_by])
            gini = gini_risk_ranking(actual_dollar, key)
            summary_rows.append({"model": name, "sort_by": sort_by, "spearman_rho": rho,
                                "spearman_pvalue": pval, "gini": gini})
    summary = pd.DataFrame(summary_rows)[["model", "sort_by", "spearman_rho", "spearman_pvalue", "gini"]]
    summary.to_csv(tables / "ranking_check_summary.csv", index=False)
    logger.info("Ranking summary (test_full, frozen models, no refit; both sort_by versions kept):\n%s",
               summary.round(4).to_string(index=False))

    # Within-storm variance check (why storm text can't add much; see within_storm_variance)
    text_cols = ["split", target_col, "narrative", "has_narrative", "quarantined"]
    claims_text = pd.read_parquet(get_path(cfg, "interim") / "claims_with_text.parquet", columns=text_cols)
    log_rows(logger, claims_text, "claims_with_text (for within-storm variance check)")
    wsv = within_storm_variance(claims_text, target_col, bool(require(cfg, "quarantine.enabled")))
    wsv.to_csv(tables / "within_storm_variance.csv", index=False)
    log_rows(logger, wsv, "within_storm_variance.csv")
    logger.info("Within-storm variance (matched claims, share of payout-share variance):\n%s",
               wsv.round(2).to_string(index=False))

    def lowest_decile_1m_pct(sort_by, model):
        row = deciles[(deciles.sort_by == sort_by) & (deciles.model == model) & (deciles.decile == 1)]
        return float(row["pct_claims_over_1M"].iloc[0])

    print("\n=== Corrected: Gini sorted by predicted DOLLARS (share x coverage) ===")
    for name in ("M1", "B1"):
        s = summary[(summary.model == name) & (summary.sort_by == "predicted_share")].iloc[0]
        d = summary[(summary.model == name) & (summary.sort_by == "predicted_dollar")].iloc[0]
        print(f"{name}: Gini by predicted SHARE = {s.gini:+.3f}  |  Gini by predicted DOLLARS = {d.gini:+.3f}")
    print("\nShare of $1M+ claims in each model's lowest-risk decile:")
    for name in ("M1", "B1"):
        print(f"{name}: by predicted share = {lowest_decile_1m_pct('predicted_share', name):.1f}%  |  "
             f"by predicted dollars = {lowest_decile_1m_pct('predicted_dollar', name):.1f}%")


if __name__ == "__main__":
    main()
