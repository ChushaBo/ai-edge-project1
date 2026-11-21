"""Replay Hurricane Ian (2022) on current policies using M2.

PROVISIONAL: pending RUNBOOK 5.2 (M1/M2's open item: M1 badly underpredicts the test
portfolio total, likely from latitude/longitude memorizing pre-2020 severity --
docs/decisions.md). M2 inherits the same tabular features as M1, so its severity here
likely understates the true replay payout too, for the (a) current-M2 version.

Two severity versions, run and saved side by side (overnight task):
  (a) m2        -- the official M2 from 07_train_models.py (train+validation only).
  (b) m2_deploy -- M2's tuned hyperparameters refit on ALL claims 2010-2024
                   (train+validation+test, excluding Helene-quarantined claims per
                   M2's established scope). CIRCULAR for this specific use: Ian's own
                   2022 claims (with their real narrative embeddings) are inside
                   m2_deploy's training data, so replaying Ian with m2_deploy is
                   partly testing whether the model memorized Ian, not whether it
                   generalizes to a similar future storm. Reported for comparison
                   only, flagged as circular everywhere it appears.

Assumptions (all stated here and in outputs/tables/ian_replay_assumptions.md):
  - Ian-affected counties = the 32 Florida counties where a 2022 claim matched a storm
    narrative containing "Ian" (36,646 claims). Only these counties are replayed.
  - High-risk flood zones = V, A, AH_AO (the Special Flood Hazard Area groups in
    config.yaml flood_zone_groups; X and D are assumed not to file a claim).
  - Historical Ian claim rate per county+zone = (claims with date of loss in the Ian
    window, 2022-09-23 to 2022-10-07) / (2025 policies in force in that county+zone).
    2025 policy counts stand in for 2022 exposure (no historical policy-count file;
    same limitation as 09_frequency.py).
  - Every current (2025) policy in an Ian-affected county, in a high-risk zone, is
    assumed to file a claim at that cell's historical rate (an expected count, not a
    0/1 outcome); the per-county Ian text embedding (mean of that county's Ian-matched
    claim embeddings) and month_of_loss = 9 (September, Ian's landfall month) are
    assigned to every one of its policies for the M2 prediction.
  - "Annual premium collected in those counties" = every policy in an Ian-affected
    county (any zone), not just the high-risk-zone ones that are assumed to claim.

Inputs:
  - data/interim/claims_with_text.parquet (Ian-matched narratives, Sept-Oct 2022)
  - data/processed/text_features.parquet
  - data/raw/policies_fl.parquet
  - data/models/m2_lgbm.joblib (from 07_train_models.py)
  - config.yaml: fields.*, flood_zone_groups, deployment_fit.enabled, random_seed, paths.*
Outputs:
  - outputs/tables/ian_replay_by_county.csv (predicted payout vs. annual premium by
    county, both severity versions)
  - outputs/tables/ian_replay_headline_comparison.csv (replay totals vs. the actual
    2022 Florida total in our own data vs. the publicly reported FEMA total)
  - outputs/figures/ian_replay_headline.png
  - outputs/tables/ian_replay_assumptions.md
  - data/models/m2_deploy_lgbm.joblib (if deployment_fit.enabled)
"""

import random

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.config import get_path, load_config, load_pipeline_script, require
from utils.log import get_logger, log_rows
from utils.matching import exclude_quarantined
from utils.models import MONTH_FEATURE, county_from_geoid, fit_categories, text_feature_columns

logger = get_logger("11_ian_replay")

IAN_WINDOW = (pd.Timestamp("2022-09-23", tz="UTC"), pd.Timestamp("2022-10-07", tz="UTC"))
HIGH_RISK_ZONES = ["V", "A", "AH_AO"]
REPLAY_MONTH = 9
# FEMA's reported final tally for Hurricane Ian NFIP claims, nationwide (overwhelmingly
# Florida): https://www.fema.gov/press-release (Jan 2024 update; 48,776 claims, $4.9B).
FEMA_REPORTED_IAN_TOTAL = 4_900_000_000.0
FEMA_REPORTED_IAN_CLAIMS = 48_776
INK, SECONDARY_INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def identify_ian_counties(claims: pd.DataFrame, county_col: str) -> list[str]:
    y22 = claims[claims["dateOfLoss"].dt.year == 2022]
    mask = y22["has_narrative"] & y22["narrative"].fillna("").str.upper().str.contains("IAN")
    ian_claims = y22[mask]
    logger.info("Ian-narrative-matched 2022 claims: %s, in %d counties",
               f"{len(ian_claims):,}", ian_claims[county_col].nunique())
    return ian_claims, sorted(ian_claims[county_col].unique())


def county_ian_embeddings(ian_claims: pd.DataFrame, text_feats: pd.DataFrame, county_col: str,
                          txt_cols: list[str]) -> pd.DataFrame:
    """Mean txt_pc* embedding per Ian-affected county, from that county's Ian-matched claims."""
    merged = ian_claims[["id", county_col]].merge(text_feats[["id"] + txt_cols], on="id", how="left")
    return merged.groupby(county_col)[txt_cols].mean().reset_index()


def historical_ian_rate(claims: pd.DataFrame, policies: pd.DataFrame, county_col: str,
                        ian_counties: list[str]) -> pd.DataFrame:
    """claims in the Ian window / 2025 policies, by county x flood_zone_group, for the
    high-risk zones in Ian-affected counties only."""
    window = claims[claims["dateOfLoss"].between(*IAN_WINDOW)]
    claim_counts = window.groupby([county_col, "flood_zone_group"]).size().rename("n_claims")
    policy_counts = policies.groupby(["county", "flood_zone_group"]).size().rename("n_policies")
    cells = pd.concat([claim_counts.rename_axis(["county", "flood_zone_group"]), policy_counts],
                      axis=1).fillna(0).reset_index()
    cells = cells[cells["county"].isin(ian_counties) & cells["flood_zone_group"].isin(HIGH_RISK_ZONES)]
    cells["ian_rate"] = np.where(cells["n_policies"] > 0, cells["n_claims"] / cells["n_policies"], 0.0)
    return cells[["county", "flood_zone_group", "n_claims", "n_policies", "ian_rate"]]


def build_replay_frame(policies: pd.DataFrame, ian_counties: list[str],
                       embeddings: pd.DataFrame, txt_cols: list[str], county_col: str) -> pd.DataFrame:
    frame = policies[policies["county"].isin(ian_counties)].copy()
    frame[MONTH_FEATURE] = REPLAY_MONTH
    frame["has_narrative"] = True
    frame = frame.merge(embeddings.rename(columns={county_col: "county"}), on="county", how="left")
    missing = frame[txt_cols[0]].isna().sum()
    if missing:
        raise SystemExit(f"STOP: {missing} replay policies have no county embedding (county not Ian-affected?)")
    return frame


def fit_deployment_m2(cfg, m07, seed: int) -> tuple[object, dict, list[str]]:
    """M2's tuned hyperparameters, refit on ALL non-quarantined claims 2010-2024.
    CIRCULAR for the Ian replay: Ian's own claims are in this training set."""
    bundle = joblib.load(get_path(cfg, "models") / "m2_lgbm.joblib")
    best_m2, m2_features = bundle["best_params"], bundle["feature_cols"]
    fixed_params, objective = cfg["models"]["fixed_params"], cfg["models"]["objective"]
    date_col = require(cfg, "fields.claims_date_of_loss")
    k = int(require(cfg, "pca.n_components"))
    txt_cols = text_feature_columns(k, "txt")
    cols = sorted(set(["id", "split", "target", date_col, "has_narrative", "quarantined"] + m2_features))
    claims = pd.read_parquet(get_path(cfg, "interim") / "claims_with_text.parquet",
                             columns=[c for c in cols if c not in txt_cols])
    text_feats = pd.read_parquet(get_path(cfg, "processed") / "text_features.parquet", columns=["id"] + txt_cols)
    claims = claims.merge(text_feats, on="id", how="left")
    claims = exclude_quarantined(claims, cfg, model="M2").reset_index(drop=True)
    log_rows(logger, claims, "claims for m2_deploy (all splits, quarantine-excluded)")
    train_rows = int((claims["split"] == "train").sum())
    cats = fit_categories(claims, m2_features)
    model = m07.final_fit(claims, m2_features, cats, seed, best_m2, fixed_params, objective, train_rows)
    joblib.dump({"model": model, "feature_cols": m2_features, "categories": cats, "best_params": best_m2,
                "fit_rows": len(claims), "note": "deployment fit on ALL claims; CIRCULAR for Ian replay "
                "(Ian's own 2022 claims are in this training set)"},
               get_path(cfg, "models") / "m2_deploy_lgbm.joblib")
    logger.info("m2_deploy: fit on %s rows", f"{len(claims):,}")
    return model, cats, m2_features


def plot_headline(totals: dict, path) -> None:
    order = ["historical_actual_fl", "fema_reported", "replay_m2", "replay_m2_deploy"]
    labels = {"historical_actual_fl": "Actual Ian payout\n(FL, our data)",
             "fema_reported": "FEMA reported\n(nationwide)",
             "replay_m2": "Replay: M2\n(current)",
             "replay_m2_deploy": "Replay: M2 deploy\n(CIRCULAR)"}
    colors = {"historical_actual_fl": "#2a78d6", "fema_reported": "#52514e",
             "replay_m2": "#1baf7a", "replay_m2_deploy": "#eb6834"}
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    vals = [totals[k] / 1e9 for k in order]
    bars = ax.bar([labels[k] for k in order], vals, color=[colors[k] for k in order], width=0.6)
    ax.bar_label(bars, labels=[f"${v:.2f}B" for v in vals], padding=3, color=INK, fontsize=10)
    ax.set_ylabel("Payout ($ billions)", color=SECONDARY_INK)
    ax.set_title("Hurricane Ian: replay vs. actual/reported payout", color=INK, fontsize=13)
    ax.tick_params(colors=MUTED)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    fig.text(0.5, -0.03,
             "Source: outputs/tables/ian_replay_headline_comparison.csv (src/11_ian_replay.py); "
             "FEMA figure per outputs/tables/ian_replay_assumptions.md. PROVISIONAL pending RUNBOOK 5.2; "
             "'M2 deploy' is circular (trained on Ian's own 2022 claims) -- see assumptions doc.",
             ha="center", color=MUTED, fontsize=7.5, wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    logger.info("Saved %s", path)


def main() -> None:
    cfg = load_config()
    seed = int(require(cfg, "random_seed"))
    set_seeds(seed)
    cap = float(require(cfg, "target.cap_share_at"))
    county_col = require(cfg, "fields.claims_county")
    k = int(require(cfg, "pca.n_components"))
    txt_cols = text_feature_columns(k, "txt")
    models_dir, tables, figures = get_path(cfg, "models"), get_path(cfg, "tables"), get_path(cfg, "figures")

    m04 = load_pipeline_script("04_clean_claims")
    m07 = load_pipeline_script("07_train_models")
    m10 = load_pipeline_script("10_shortfall")

    cols = ["id", "dateOfLoss", county_col, "has_narrative", "narrative", "flood_zone_group",
           "target", "totalBuildingInsuranceCoverage"]
    claims = pd.read_parquet(get_path(cfg, "interim") / "claims_with_text.parquet", columns=cols)
    log_rows(logger, claims, "claims_with_text")
    ian_claims, ian_counties = identify_ian_counties(claims, county_col)

    text_feats = pd.read_parquet(get_path(cfg, "processed") / "text_features.parquet", columns=["id"] + txt_cols)
    embeddings = county_ian_embeddings(ian_claims, text_feats, county_col, txt_cols)

    policies = m10.build_policy_features(cfg, m04)
    log_rows(logger, policies, "policy feature table (M1-ready)")

    rates = historical_ian_rate(claims, policies, county_col, ian_counties)
    logger.info("Historical Ian claim rate, high-risk zones, Ian-affected counties: min %.3f, "
               "median %.3f, max %.3f", rates["ian_rate"].min(), rates["ian_rate"].median(), rates["ian_rate"].max())

    replay = build_replay_frame(policies, ian_counties, embeddings, txt_cols, county_col)
    replay = replay.merge(rates[["county", "flood_zone_group", "ian_rate"]], on=["county", "flood_zone_group"],
                          how="left")
    replay["ian_rate"] = replay["ian_rate"].fillna(0.0)  # not a high-risk zone or no match -> assumed no claim
    log_rows(logger, replay, "replay policy frame (Ian-affected counties)")

    bundle_m2 = joblib.load(models_dir / "m2_lgbm.joblib")
    m2_features = bundle_m2["feature_cols"]
    replay["severity_m2"] = m07.predict_clipped(bundle_m2["model"], replay, m2_features,
                                                 bundle_m2["categories"], cap)

    if require(cfg, "deployment_fit.enabled"):
        model_deploy, cats_deploy, m2_features_deploy = fit_deployment_m2(cfg, m07, seed)
        replay["severity_m2_deploy"] = m07.predict_clipped(model_deploy, replay, m2_features_deploy,
                                                            cats_deploy, cap)
    else:
        logger.warning("deployment_fit.enabled is false; skipping m2_deploy severity")
        replay["severity_m2_deploy"] = np.nan

    for col in ("severity_m2", "severity_m2_deploy"):
        replay[f"expected_payout_{col}"] = (replay[col] * replay["totalBuildingInsuranceCoverage"]
                                            * replay["ian_rate"])

    by_county = replay.groupby("county").agg(
        n_policies=("id", "size"),
        premium_total=("totalInsurancePremiumOfThePolicy", "sum"),
        predicted_payout_m2=("expected_payout_severity_m2", "sum"),
        predicted_payout_m2_deploy=("expected_payout_severity_m2_deploy", "sum"),
    ).reset_index()
    statewide = pd.DataFrame([{
        "county": "ALL (Ian-affected counties)", "n_policies": len(replay),
        "premium_total": replay["totalInsurancePremiumOfThePolicy"].sum(),
        "predicted_payout_m2": replay["expected_payout_severity_m2"].sum(),
        "predicted_payout_m2_deploy": replay["expected_payout_severity_m2_deploy"].sum(),
    }])
    out = pd.concat([by_county, statewide], ignore_index=True)
    out.to_csv(tables / "ian_replay_by_county.csv", index=False)
    log_rows(logger, out, "ian_replay_by_county.csv")

    by_county_premium_total = float(statewide["premium_total"].iloc[0])
    actual_fl = float((ian_claims["target"] * ian_claims["totalBuildingInsuranceCoverage"]).sum())
    # Also the wider historical total (every 2022 claim in the Ian date window, matched or not)
    window_claims = claims[claims["dateOfLoss"].between(*IAN_WINDOW)]
    actual_fl_window = float((window_claims["target"] * window_claims["totalBuildingInsuranceCoverage"]).sum())

    totals = {
        "historical_actual_fl": actual_fl_window,
        "fema_reported": FEMA_REPORTED_IAN_TOTAL,
        "replay_m2": float(replay["expected_payout_severity_m2"].sum()),
        "replay_m2_deploy": float(replay["expected_payout_severity_m2_deploy"].sum()),
    }
    headline = pd.DataFrame([
        {"source": "historical_actual_fl_narrative_matched", "total_dollar": actual_fl,
        "note": f"{len(ian_claims):,} claims whose narrative matched Ian"},
        {"source": "historical_actual_fl_date_window", "total_dollar": actual_fl_window,
        "note": f"{len(window_claims):,} FL claims, loss date {IAN_WINDOW[0].date()}..{IAN_WINDOW[1].date()}"},
        {"source": "fema_reported_nationwide", "total_dollar": FEMA_REPORTED_IAN_TOTAL,
        "note": f"{FEMA_REPORTED_IAN_CLAIMS:,} claims (see ian_replay_assumptions.md for citation)"},
        {"source": "replay_m2_current", "total_dollar": totals["replay_m2"],
        "note": "scenario: today's policies, M2 (current, train+validation)"},
        {"source": "replay_m2_deploy_CIRCULAR", "total_dollar": totals["replay_m2_deploy"],
        "note": "scenario: today's policies, M2 deploy-fit on ALL claims INCLUDING Ian -- circular"},
    ])
    headline.to_csv(tables / "ian_replay_headline_comparison.csv", index=False)
    logger.info("PROVISIONAL (pending RUNBOOK 5.2) Ian replay headline:\n%s",
               headline.assign(total_dollar=headline["total_dollar"].round(0)).to_string(index=False))

    plot_headline(totals, figures / "ian_replay_headline.png")

    assumptions = f"""# Hurricane Ian replay -- assumptions (RUNBOOK 6.3)

PROVISIONAL, pending RUNBOOK 5.2 -- see docs/decisions.md open item. M2 shares M1's
tabular features, so the (a) current-M2 replay total likely understates the true
figure for the same reason M1 underpredicts the test portfolio total.

## Ian-affected counties
{len(ian_counties)} Florida counties where a 2022 claim's matched storm narrative
contained "Ian" ({len(ian_claims):,} claims): {", ".join(ian_counties)}.
Only policies in these counties are replayed.

## High-risk flood zones
{", ".join(HIGH_RISK_ZONES)} (the Special Flood Hazard Area groups in config.yaml
flood_zone_groups). X and D zones are assumed not to file a claim in this scenario.

## Historical Ian claim rate
claims with date of loss in {IAN_WINDOW[0].date()}..{IAN_WINDOW[1].date()}, by county x
flood_zone_group, divided by 2025 policies in force in that cell. 2025 policy counts
stand in for 2022 exposure (same limitation as 09_frequency.py: no historical
policy-count file exists to do better).

## Scenario applied to current policies
Every 2025 policy in an Ian-affected county is assumed to file a claim with
probability = its cell's historical Ian rate, if its flood zone is high-risk (0 claim
probability otherwise). Its text input for M2 is its county's Ian embedding (the mean
embedding of that county's Ian-matched claims) and month_of_loss = {REPLAY_MONTH}
(September, Ian's landfall month) -- i.e. "if a storm like Ian hit this county's
current policies again."

## "Annual premium collected in those counties"
Sum of totalInsurancePremiumOfThePolicy for every 2025 policy in an Ian-affected
county, ANY flood zone (not just the high-risk-zone policies assumed to claim):
${by_county_premium_total:,.0f} (outputs/tables/ian_replay_by_county.csv's "ALL" row,
premium_total column) -- the base the predicted payout totals below are compared
against.

## Two severity versions (run and saved side by side)
- replay_m2: the official M2 (07_train_models.py, train+validation only).
- replay_m2_deploy_CIRCULAR: M2's tuned hyperparameters refit on ALL claims
  2010-2024 (train+validation+test, Helene-quarantined claims excluded per M2's
  established scope). This is CIRCULAR for the Ian replay specifically: Ian's own
  2022 claims, with their real (not county-averaged) embeddings, are inside this
  model's training data, so it has already seen the actual outcome it is being asked
  to "predict" for a similar future storm. Reported for comparison only.

## Comparison to publicly reported figures
- Our own data, Florida, claims whose narrative matched Ian: ${actual_fl:,.0f}
  ({len(ian_claims):,} claims).
- Our own data, Florida, all claims with loss date in the Ian window (matched or
  not): ${actual_fl_window:,.0f} ({len(window_claims):,} claims).
- FEMA's reported final tally (nationwide, overwhelmingly Florida): ${FEMA_REPORTED_IAN_TOTAL:,.0f}
  ({FEMA_REPORTED_IAN_CLAIMS:,} claims). Source: FEMA press materials reporting NFIP's
  final Hurricane Ian payout (as summarized by Artemis.bm and FEMA.gov reporting,
  retrieved 2026-11-16); building-only vs. building+contents+ICC is not always
  specified in press figures, so treat this as approximate.
- The replay is a SCENARIO on today's (larger, more numerous) policy book, not a
  backtest of 2022 -- a replay total above the historical actual is not necessarily
  wrong, since Florida's NFIP book has grown since 2022. It is not a forecast.
"""
    (tables / "ian_replay_assumptions.md").write_text(assumptions, encoding="utf-8")
    logger.info("Saved ian_replay_assumptions.md")


if __name__ == "__main__":
    main()
