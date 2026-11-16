"""Explain M1 with SHAP.

PROVISIONAL: M1 itself has an open issue (docs/decisions.md, "OPEN ITEM for
RUNBOOK 5.2") -- it badly underpredicts the test portfolio total, apparently because
latitude/longitude let it memorize pre-2020 location averages. These SHAP outputs are
explanatory evidence for that open item, not a finished, signed-off explanation; they
are superseded if M1's feature set changes in 5.2.

Method: TreeExplainer on M1's saved LightGBM model, on a random sample of test rows
(fast and exact for tree models; no need for the full 112,573 test claims). Top
features are ranked by mean |SHAP value| (share-of-coverage units). For categorical
features, "dependence" is shown as SHAP value by category level (a strip/box plot)
rather than shap's continuous dependence_plot, which assumes an ordered numeric axis.
Dollar effects for a typical $250,000-coverage home multiply the SHAP value (a
share-of-coverage unit) by $250,000, i.e. "how much this feature pushes the predicted
payout up or down for a $250k home, holding the home's other features fixed at the
sampled rows' values."

Inputs:
  - data/models/m1_lgbm.joblib (trained M1)
  - data/interim/claims_with_text.parquet (sample of test rows)
  - config.yaml: random_seed, paths.*
Outputs:
  - outputs/figures/shap_summary.png
  - outputs/figures/shap_dependence_<feature>.png (top 5 features by mean |SHAP|,
    plus elevationDifference if not already in the top 5)
  - outputs/tables/shap_top_effects_dollars.csv (SHAP effect in dollars for a typical
    $250,000-coverage home, by feature and level/bin)
  - outputs/tables/shap_findings.md (plain-language summary; explicitly reports
    whether latitude/longitude dominate, per the open item above)
"""

import random

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from utils.config import get_path, load_config, require
from utils.log import get_logger, log_rows
from utils.models import M1_FEATURES, encode_features

logger = get_logger("08_explain")

TYPICAL_COVERAGE = 250_000.0
N_SAMPLE = 3000
FORCE_INCLUDE = ["elevationDifference"]
INK, SECONDARY_INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
BAR_COLOR = "#2a78d6"


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)


def save_fig(fig, path, source_note: str) -> None:
    fig.text(0.5, -0.02, source_note, ha="center", color=MUTED, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_summary(shap_values: np.ndarray, X: pd.DataFrame, path) -> None:
    fig = plt.figure(figsize=(8, 7), facecolor=SURFACE)
    shap.summary_plot(shap_values, X, show=False, plot_size=None)
    fig = plt.gcf()
    fig.set_facecolor(SURFACE)
    ax = plt.gca()
    ax.set_facecolor(SURFACE)
    ax.set_title("M1: which features move the predicted payout (SHAP)", color=INK, fontsize=12)
    ax.set_xlabel("SHAP value (impact on predicted share of building coverage)", color=SECONDARY_INK)
    ax.tick_params(colors=MUTED)
    save_fig(fig, path, "Source: data/models/m1_lgbm.joblib, SHAP TreeExplainer on "
            f"{len(X):,} sampled test claims (src/08_explain.py). PROVISIONAL pending RUNBOOK 5.2.")


def plot_dependence(feature: str, shap_col: np.ndarray, values: pd.Series, rank: int, path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5), facecolor=SURFACE)
    style_axes(ax)
    if values.dtype.name == "category":
        levels = [lvl for lvl in values.cat.categories if (values == lvl).any()]
        data = [shap_col[(values == lvl).to_numpy()] for lvl in levels]
        ax.boxplot(data, tick_labels=[str(l) for l in levels], showfliers=False,
                  boxprops=dict(color=BAR_COLOR), medianprops=dict(color="#eb6834"),
                  whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
        ax.set_xlabel(feature, color=SECONDARY_INK)
    else:
        x = pd.to_numeric(values, errors="coerce").to_numpy()
        finite = np.isfinite(x)
        ax.scatter(x[finite], shap_col[finite], s=10, alpha=0.35, color=BAR_COLOR, linewidths=0)
        ax.set_xlabel(feature, color=SECONDARY_INK)
    ax.set_ylabel("SHAP value (share of coverage)", color=SECONDARY_INK)
    ax.set_title(f"#{rank} by mean |SHAP|: {feature}", color=INK, fontsize=11)
    save_fig(fig, path, f"Source: {N_SAMPLE:,}-row test sample, src/08_explain.py. "
            "PROVISIONAL pending RUNBOOK 5.2.")


def dollar_effects_table(feature: str, shap_col: np.ndarray, values: pd.Series) -> pd.DataFrame:
    """SHAP effect for a typical $250,000-coverage home, by category level (categorical
    features) or low/mid/high tercile of the sampled values (numeric features)."""
    rows = []
    if values.dtype.name == "category":
        for lvl in values.cat.categories:
            mask = (values == lvl).to_numpy()
            if mask.sum() == 0:
                continue
            rows.append({"feature": feature, "level": str(lvl), "n": int(mask.sum()),
                        "mean_shap_share": shap_col[mask].mean()})
    else:
        x = pd.to_numeric(values, errors="coerce")
        finite = x.notna().to_numpy()
        if finite.sum() == 0:
            return pd.DataFrame()
        xf = x[finite]
        bins = pd.qcut(xf, q=3, duplicates="drop")
        names = ["low", "mid", "high"][: len(bins.cat.categories)]
        for name, interval in zip(names, bins.cat.categories):
            in_bin = (bins == interval).to_numpy()
            mask = np.zeros(len(values), dtype=bool)
            mask[np.flatnonzero(finite)[in_bin]] = True
            rows.append({"feature": feature, "level": f"{name} (x in {interval})",
                        "n": int(mask.sum()), "mean_shap_share": shap_col[mask].mean()})
    out = pd.DataFrame(rows)
    out["dollar_effect_250k_home"] = (out["mean_shap_share"] * TYPICAL_COVERAGE).round(0)
    out["mean_shap_share"] = out["mean_shap_share"].round(5)
    return out


def main() -> None:
    cfg = load_config()
    seed = int(require(cfg, "random_seed"))
    set_seeds(seed)
    date_col = require(cfg, "fields.claims_date_of_loss")
    figures, tables, models_dir = get_path(cfg, "figures"), get_path(cfg, "tables"), get_path(cfg, "models")

    bundle = joblib.load(models_dir / "m1_lgbm.joblib")
    model, categories, feature_cols = bundle["model"], bundle["categories"], bundle["feature_cols"]
    assert feature_cols == M1_FEATURES, "m1_lgbm.joblib feature list does not match utils.models.M1_FEATURES"

    cols = sorted(set(["id", "split", date_col] + M1_FEATURES))
    claims = pd.read_parquet(get_path(cfg, "interim") / "claims_with_text.parquet", columns=cols)
    test = claims[claims["split"] == "test"]
    log_rows(logger, test, "test claims available")
    sample = test.sample(n=min(N_SAMPLE, len(test)), random_state=seed).reset_index(drop=True)
    log_rows(logger, sample, "sampled for SHAP")

    X = encode_features(sample, M1_FEATURES, categories)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    mean_abs = pd.Series(np.abs(shap_values).mean(axis=0), index=M1_FEATURES).sort_values(ascending=False)
    total_mean_abs = mean_abs.sum()
    lat_long_share = 100 * mean_abs[["latitude", "longitude"]].sum() / total_mean_abs
    lat_rank = int(mean_abs.index.get_loc("latitude")) + 1
    long_rank = int(mean_abs.index.get_loc("longitude")) + 1
    splits = pd.Series(model.feature_importances_, index=model.feature_name_).sort_values(ascending=False)
    split_share = 100 * splits[["latitude", "longitude"]].sum() / splits.sum()
    logger.info("Mean |SHAP| by feature (top 10):\n%s", mean_abs.head(10).round(5).to_string())
    logger.info("By SHAP impact: latitude + longitude = %.1f%% of total mean |SHAP| "
               "(latitude rank #%d, longitude rank #%d of %d)",
               lat_long_share, lat_rank, long_rank, len(M1_FEATURES))
    logger.info("By tree split count: latitude + longitude = %.1f%% of all splits "
               "(latitude #%d, longitude #%d: %d, %d splits)",
               split_share, int(splits.index.get_loc("latitude")) + 1,
               int(splits.index.get_loc("longitude")) + 1, splits["latitude"], splits["longitude"])

    plot_summary(shap_values, X, figures / "shap_summary.png")

    top5 = mean_abs.head(5).index.tolist()
    ranked_features = list(dict.fromkeys(top5 + [f for f in FORCE_INCLUDE if f in M1_FEATURES]))
    dollar_tables = []
    for feat in ranked_features:
        rank = int(mean_abs.index.get_loc(feat)) + 1
        col_idx = M1_FEATURES.index(feat)
        plot_dependence(feat, shap_values[:, col_idx], X[feat], rank,
                        figures / f"shap_dependence_{feat}.png")
        dollar_tables.append(dollar_effects_table(feat, shap_values[:, col_idx], X[feat]))
    dollar_df = pd.concat(dollar_tables, ignore_index=True)
    dollar_df.to_csv(tables / "shap_top_effects_dollars.csv", index=False)
    logger.info("Saved shap_top_effects_dollars.csv (%d rows)", len(dollar_df))

    lines = [
        "# SHAP findings for M1 (RUNBOOK 5.3)",
        "",
        "**PROVISIONAL -- pending RUNBOOK 5.2.** M1 has an open item in "
        "`docs/decisions.md` (it underpredicts the test portfolio total by 57.6%). "
        "These findings are evidence toward that open item, not a final explanation.",
        "",
        "## Does latitude/longitude dominate SHAP?",
        "",
        f"**Partially, and the two metrics disagree.** By SHAP impact magnitude on this "
        f"{N_SAMPLE:,}-row sample, latitude + longitude are **{lat_long_share:.1f}%** of "
        f"total mean |SHAP| (latitude individually ranks #{lat_rank} of {len(M1_FEATURES)}, "
        f"longitude #{long_rank}) -- real, but `totalBuildingInsuranceCoverage` ranks #1 "
        "individually, so lat/long are not an outright majority of SHAP impact. By tree "
        f"split count, though, they clearly do dominate: **{split_share:.1f}%** of every "
        "split M1 makes is on latitude or longitude, more than any other feature by a wide "
        "margin (see the model-training log). The two metrics measure different things: "
        "SHAP impact is averaged over this representative sample of mostly-typical claims, "
        "while split count reflects how finely the tree partitions space -- which is "
        "exactly the mechanism in the open-item hypothesis (fine spatial leaves that "
        "memorize a location's pre-2020 average and cannot extrapolate to a catastrophe "
        "there). The hypothesis is about structure, not average impact on typical claims, "
        "so the split-count evidence is the more direct support for it.",
        "",
        "## Top features by mean |SHAP| (share-of-coverage units)",
        "",
        "| rank | feature | mean |SHAP| |",
        "|---|---|---|",
    ]
    for i, (feat, val) in enumerate(mean_abs.head(10).items(), start=1):
        lines.append(f"| {i} | {feat} | {val:.5f} |")
    lines += [
        "",
        "## Dependence (top 5, plus elevationDifference)",
        "",
        "See `outputs/figures/shap_dependence_<feature>.png` and "
        "`outputs/tables/shap_top_effects_dollars.csv` for the dollar translation "
        "(SHAP value x $250,000) at low/mid/high levels of each feature.",
        "",
    ]
    (tables / "shap_findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Saved shap_findings.md")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
