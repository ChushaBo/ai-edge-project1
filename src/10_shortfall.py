"""Compare expected annual loss with premium by county, with pluggable severity.

PROVISIONAL: M1 has an open item in docs/decisions.md (underpredicts the test
portfolio total by 57.6%), so these shortfall totals likely UNDERSTATE the true
shortfall. Three severity sources are run and saved side by side so the team can see
the spread while 5.2 is still open:
  (a) m1        -- the official M1 from 07_train_models.py (train+validation only).
  (b) m1_deploy -- M1's tuned hyperparameters refit on ALL claims 2010-2024
                   (train+validation+test), an implementation setting
                   (config.yaml deployment_fit.enabled) pending team approval
                   (docs/decisions.md). Forward-looking use only; does not touch
                   07_train_models.py or any reported test metric.
  (c) b1        -- the flood-zone-mean baseline, for reference (not catastrophe-aware
                   either, but does not inherit M1's spatial-memorization failure
                   mode -- see docs/decisions.md and outputs/tables/shap_findings.md).

Method: predicted severity (share of building coverage) x building coverage x
claim frequency for the policy's county+flood_zone_group (outputs/tables/
frequency_by_county_zone.csv, RUNBOOK 6.1) = expected annual loss per policy. Summed by
county and compared against total premium (fields.policies_premium, plus the two
comparison premium fields from the earlier premium decision in docs/decisions.md).

Inputs:
  - data/raw/policies_fl.parquet (current policy file, years.policies_year)
  - data/models/m1_lgbm.joblib, baselines.joblib (from 07_train_models.py)
  - data/interim/claims_with_text.parquet (only if deployment_fit.enabled, to refit
    m1_deploy)
  - outputs/tables/frequency_by_county_zone.csv (from 09_frequency.py)
  - Census cartographic county boundary file (sources.census_county_url; downloaded
    once to data/raw/census/)
  - config.yaml: fields.policies_*, cleaning.*, flood_zone_groups, occupancy_groups,
    deployment_fit.enabled, random_seed, paths.*
Outputs:
  - outputs/tables/shortfall_by_county.csv (long format: severity_model x county,
    with an ALL/statewide row per severity_model; policies, 3 premium measures,
    expected loss, gap, loss ratio)
  - outputs/tables/shortfall_totals_comparison.csv (the 3 statewide totals side by
    side)
  - outputs/figures/shortfall_loss_ratio_map.png (choropleth, severity model (a);
    the official RUNBOOK 6.2 deliverable)
  - data/models/m1_deploy_lgbm.joblib (if deployment_fit.enabled)
All outputs are marked PROVISIONAL (pending RUNBOOK 5.2) in their header/log.
"""

import random

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.config import get_path, load_config, load_pipeline_script, require
from utils.download import download_file, make_session
from utils.log import get_logger, log_rows
from utils.models import M1_FEATURES, county_from_geoid, fit_categories

logger = get_logger("10_shortfall")

INK, SECONDARY_INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def build_policy_features(cfg, m04) -> pd.DataFrame:
    """Policies -> the M1 feature table, renamed/derived the same way 04_clean_claims
    does for claims. building_age_at_loss and policy_tenure_years use
    policyEffectiveDate as the reference date (claims use dateOfLoss; a policy has no
    loss date, so its own term's effective date is the closest pre-event analogue --
    an implementation choice, not a config decision)."""
    cols = ["id", "censusGeoid", "ratedFloodZone", "occupancyType", "numberOfFloorsInInsuredBuilding",
           "originalConstructionDate", "originalNBDate", "policyEffectiveDate",
           "cancellationDateOfFloodPolicy", "totalBuildingInsuranceCoverage",
           "totalContentsInsuranceCoverage", "totalInsurancePremiumOfThePolicy", "policyCost",
           "fullRiskPremium", "foundationType", "elevatedBuildingIndicator",
           "postFIRMConstructionIndicator", "primaryResidenceIndicator", "rentalPropertyIndicator",
           "condominiumCoverageTypeCode", "policyCount", "crsClassCode", "buildingDeductibleCode",
           "basementEnclosureCrawlspaceType", "elevationDifference", "latitude", "longitude"]
    p = pd.read_parquet(get_path(cfg, "raw") / "policies_fl.parquet", columns=cols)
    log_rows(logger, p, "policies_fl (raw)")
    if require(cfg, "cleaning.exclude_cancelled_policies"):
        p = p[p["cancellationDateOfFloodPolicy"].isna()].reset_index(drop=True)
        log_rows(logger, p, "after excluding cancelled terms")

    lo, hi = require(cfg, "cleaning.building_age_years")
    age = m04.years_between(p["policyEffectiveDate"], p["originalConstructionDate"])
    bad_age = age.notna() & ~age.between(lo, hi)
    tenure = m04.years_between(p["policyEffectiveDate"], p["originalNBDate"])
    bad_tenure = tenure < require(cfg, "cleaning.min_policy_tenure_years")

    out = p.assign(
        county=county_from_geoid(p["censusGeoid"]),
        numberOfFloorsInTheInsuredBuilding=p["numberOfFloorsInInsuredBuilding"],
        building_age_at_loss=age.mask(bad_age),
        policy_tenure_years=tenure.mask(bad_tenure),
        flood_zone_group=m04.apply_group(p["ratedFloodZone"], require(cfg, "flood_zone_groups"), "flood zone (policies)"),
        occupancy_group=m04.apply_group(p["occupancyType"], require(cfg, "occupancy_groups"), "occupancy (policies)"),
    )
    missing = [c for c in M1_FEATURES if c not in out.columns]
    if missing:
        raise SystemExit(f"STOP: M1 features missing from policy feature table: {missing}")
    return out


def fit_deployment_m1(cfg, m07, seed: int) -> tuple[object, dict]:
    """Refit M1 (same tuned hyperparameters) on ALL claims 2010-2024. Implementation
    setting (deployment_fit.enabled); never touches 07_train_models.py's metrics."""
    best_m1 = joblib.load(get_path(cfg, "models") / "m1_lgbm.joblib")["best_params"]
    fixed_params, objective = cfg["models"]["fixed_params"], cfg["models"]["objective"]
    date_col = require(cfg, "fields.claims_date_of_loss")
    cols = sorted(set(["id", "split", "target", date_col] + M1_FEATURES))
    claims = pd.read_parquet(get_path(cfg, "interim") / "claims_with_text.parquet", columns=cols)
    log_rows(logger, claims, "claims_with_text (all splits, for m1_deploy)")
    # Reference size for n_estimators scaling = the original tuning train split (38,678
    # rows); m07.final_fit() scales best_iteration by len(this df) / train_rows.
    train_rows = int((claims["split"] == "train").sum())
    cats = fit_categories(claims, M1_FEATURES)
    model = m07.final_fit(claims, M1_FEATURES, cats, seed, best_m1, fixed_params, objective, train_rows)
    joblib.dump({"model": model, "feature_cols": M1_FEATURES, "categories": cats,
                "best_params": best_m1, "fit_rows": len(claims),
                "note": "deployment fit on ALL claims 2010-2024; pending team approval"},
               get_path(cfg, "models") / "m1_deploy_lgbm.joblib")
    logger.info("m1_deploy: fit on %s rows (train+validation+test)", f"{len(claims):,}")
    return model, cats


def county_boundaries(cfg):
    import geopandas as gpd

    url = require(cfg, "sources.census_county_url")
    dest = get_path(cfg, "raw") / "census" / url.rsplit("/", 1)[-1]
    session = make_session()
    downloaded = download_file(session, url, dest)
    logger.info("Census county boundaries: %s (%s)", dest, "downloaded" if downloaded else "cached")
    g = gpd.read_file(dest)
    fl = g[g["STATEFP"] == require(cfg, "scope.state_fips")].copy()
    fl["county"] = fl["GEOID"]
    return fl[["county", "NAME", "geometry"]]


def summarize(df: pd.DataFrame, severity_col: str, freq_col: str, model_name: str) -> pd.DataFrame:
    """Per county (+ an ALL row): policies, 3 premium measures, expected loss, gap,
    loss ratio, for one severity_model."""
    d = df.assign(expected_loss=df[severity_col] * df["totalBuildingInsuranceCoverage"] * df[freq_col])
    by_county = d.groupby("county").agg(
        n_policies=("id", "size"),
        premium_total=("totalInsurancePremiumOfThePolicy", "sum"),
        premium_policyCost_total=("policyCost", "sum"),
        premium_fullRisk_total=("fullRiskPremium", "sum"),
        expected_loss_total=("expected_loss", "sum"),
    ).reset_index()
    statewide = pd.DataFrame([{
        "county": "ALL", "n_policies": len(d), "premium_total": d["totalInsurancePremiumOfThePolicy"].sum(),
        "premium_policyCost_total": d["policyCost"].sum(), "premium_fullRisk_total": d["fullRiskPremium"].sum(),
        "expected_loss_total": d["expected_loss"].sum(),
    }])
    out = pd.concat([by_county, statewide], ignore_index=True)
    out["gap"] = out["expected_loss_total"] - out["premium_total"]
    out["loss_ratio"] = out["expected_loss_total"] / out["premium_total"]
    out["severity_model"] = model_name
    return out


def plot_loss_ratio_map(shortfall_m1: pd.DataFrame, boundaries, path) -> None:
    import geopandas as gpd

    by_county = shortfall_m1[shortfall_m1["county"] != "ALL"]
    merged = boundaries.merge(by_county, on="county", how="left")
    fig, ax = plt.subplots(figsize=(7, 7), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    merged.plot(column="loss_ratio", cmap="Blues", linewidth=0.4, edgecolor=GRID, ax=ax,
               legend=True, missing_kwds={"color": "#f0efec", "edgecolor": GRID, "label": "no policies"},
               legend_kwds={"label": "Expected annual loss / premium (M1, current)", "shrink": 0.6})
    ax.set_axis_off()
    ax.set_title("Florida NFIP shortfall by county: expected loss / premium ratio",
                color=INK, fontsize=13)
    fig.text(0.5, 0.01,
             "Source: outputs/tables/shortfall_by_county.csv (src/10_shortfall.py), severity "
             "model M1 (current). PROVISIONAL pending RUNBOOK 5.2 -- M1 likely understates "
             "the true shortfall (docs/decisions.md).",
             ha="center", color=MUTED, fontsize=8, wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    logger.info("Saved %s", path)


def main() -> None:
    cfg = load_config()
    seed = int(require(cfg, "random_seed"))
    set_seeds(seed)
    cap = float(require(cfg, "target.cap_share_at"))
    models_dir, tables, figures = get_path(cfg, "models"), get_path(cfg, "tables"), get_path(cfg, "figures")

    m04 = load_pipeline_script("04_clean_claims")
    m07 = load_pipeline_script("07_train_models")

    policies = build_policy_features(cfg, m04)
    log_rows(logger, policies, "policy feature table (M1-ready)")

    freq = pd.read_csv(get_path(cfg, "tables") / "frequency_by_county_zone.csv",
                       dtype={"county": "string"})
    policies = policies.merge(freq[["county", "flood_zone_group", "frequency"]],
                              on=["county", "flood_zone_group"], how="left")
    missing_freq = policies["frequency"].isna().sum()
    if missing_freq:
        missing_county = policies["county"].isna().sum()
        logger.warning("%d policies have no frequency match (%d of those have no censusGeoid at "
                       "all, so no county); treated as 0 expected claims. They still count in "
                       "the statewide premium total but drop out of the per-county breakdown.",
                       missing_freq, missing_county)
        policies["frequency"] = policies["frequency"].fillna(0.0)

    # --- (a) current M1 --------------------------------------------------
    bundle_m1 = joblib.load(models_dir / "m1_lgbm.joblib")
    policies["severity_m1"] = m07.predict_clipped(bundle_m1["model"], policies, M1_FEATURES,
                                                  bundle_m1["categories"], cap)

    # --- (b) M1 deployment fit (implementation setting, pending approval) ----
    results = []
    if require(cfg, "deployment_fit.enabled"):
        model_deploy, cats_deploy = fit_deployment_m1(cfg, m07, seed)
        policies["severity_m1_deploy"] = m07.predict_clipped(model_deploy, policies, M1_FEATURES,
                                                             cats_deploy, cap)
    else:
        logger.warning("deployment_fit.enabled is false; skipping m1_deploy severity")
        policies["severity_m1_deploy"] = np.nan

    # --- (c) B1 baseline --------------------------------------------------
    baselines = joblib.load(models_dir / "baselines.joblib")
    policies["severity_b1"] = np.clip(
        policies["flood_zone_group"].map(baselines["b1_means"]).fillna(baselines["b1_overall"]), 0.0, cap)

    tables_out = []
    for col, name in [("severity_m1", "m1"), ("severity_m1_deploy", "m1_deploy"), ("severity_b1", "b1")]:
        if policies[col].isna().all():
            continue
        tables_out.append(summarize(policies, col, "frequency", name))
    shortfall = pd.concat(tables_out, ignore_index=True)[
        ["severity_model", "county", "n_policies", "premium_total", "premium_policyCost_total",
         "premium_fullRisk_total", "expected_loss_total", "gap", "loss_ratio"]
    ]
    shortfall.to_csv(tables / "shortfall_by_county.csv", index=False)
    log_rows(logger, shortfall, "shortfall_by_county.csv")

    comparison = shortfall[shortfall["county"] == "ALL"].reset_index(drop=True)
    comparison.to_csv(tables / "shortfall_totals_comparison.csv", index=False)
    logger.info("PROVISIONAL (pending RUNBOOK 5.2) statewide totals:\n%s",
               comparison[["severity_model", "n_policies", "premium_total", "expected_loss_total",
                          "gap", "loss_ratio"]].round(3).to_string(index=False))

    # --- Choropleth (M1, current -- the official RUNBOOK 6.2 deliverable) --------
    try:
        boundaries = county_boundaries(cfg)
        plot_loss_ratio_map(shortfall[shortfall["severity_model"] == "m1"], boundaries,
                            figures / "shortfall_loss_ratio_map.png")
    except Exception:
        logger.exception("STOP (logged, continuing): choropleth failed; "
                         "shortfall_by_county.csv and shortfall_totals_comparison.csv are unaffected")


if __name__ == "__main__":
    main()
