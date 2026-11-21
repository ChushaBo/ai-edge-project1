"""Estimate annual claim frequency per policy by county and flood zone.

Method: frequency = (paid claims in the county+zone cell, summed over the whole
years.claims_start..claims_end window) / (policies-in-force in that cell x number of
years in the window). This is an annual probability that a policy has a paid claim.

Limitation (documented here, not a decision needing sign-off): only one policy
snapshot exists (years.policies_year, the 2025 term count), so it stands in for every
year's exposure in 2010-2024. If the book of business has grown or shrunk by county or
zone, this under- or over-states frequency for that cell; there is no historical
policy-count file to do better with.

Smoothing: raw cell rates are noisy where a county+zone has few claims or few
policies, so each estimate is shrunk with Buhlmann credibility (frequency.
credibility_k_county / credibility_k_cell in config.yaml, implementation settings, not
decisions) toward a coarser rate, two levels deep:
  1. statewide_rate = total claims / total policy-years (no shrinkage; the top level).
  2. county_rate: Z = n_claims_county / (n_claims_county + k_county); smoothed =
     Z * raw_county_rate + (1 - Z) * statewide_rate.
  3. cell_rate (county x flood_zone_group): Z = n_claims_cell / (n_claims_cell +
     k_cell); smoothed = Z * raw_cell_rate + (1 - Z) * county_smoothed_rate.
A cell with zero policies has an undefined raw rate (Z forced to 0); its smoothed rate
is just its county's rate.

Inputs:
  - data/interim/claims_clean.parquet (paid claims, numerator)
  - data/raw/policies_fl.parquet (policies-in-force counts, denominator)
  - config.yaml: fields.*, years.*, cleaning.exclude_cancelled_policies,
    flood_zone_groups, frequency.*, paths.*
Outputs:
  - outputs/tables/frequency_by_county_zone.csv (raw and smoothed rates; credibility
    weight; claim and policy counts per cell, plus the county and statewide rows they
    shrink toward)
"""

import pandas as pd

from utils.config import get_path, load_config, require
from utils.log import get_logger, log_rows
from utils.models import county_from_geoid

logger = get_logger("09_frequency")


def credibility_blend(n: pd.Series, raw: pd.Series, prior: pd.Series, k: float) -> tuple[pd.Series, pd.Series]:
    """Buhlmann credibility: Z = n / (n + k); blended = Z*raw + (1-Z)*prior.
    raw may be NaN (zero exposure); treated as Z=0 (fall back fully to prior)."""
    z = n / (n + k)
    z = z.where(raw.notna(), 0.0)
    blended = z * raw.fillna(0.0) + (1 - z) * prior
    return blended, z


def main() -> None:
    cfg = load_config()
    county_col_claims = require(cfg, "fields.claims_county")
    county_col_policies_src = require(cfg, "fields.policies_county")  # censusGeoid
    k_county = float(require(cfg, "frequency.credibility_k_county"))
    k_cell = float(require(cfg, "frequency.credibility_k_cell"))
    n_years = int(require(cfg, "years.claims_end")) - int(require(cfg, "years.claims_start")) + 1
    zone_col = "flood_zone_group"

    claims = pd.read_parquet(get_path(cfg, "interim") / "claims_clean.parquet",
                             columns=["id", county_col_claims, zone_col])
    log_rows(logger, claims, "claims_clean (paid claims, numerator)")

    policies = pd.read_parquet(get_path(cfg, "raw") / "policies_fl.parquet",
                               columns=["id", county_col_policies_src, "ratedFloodZone",
                                       "cancellationDateOfFloodPolicy"])
    log_rows(logger, policies, "policies_fl (raw)")
    if require(cfg, "cleaning.exclude_cancelled_policies"):
        policies = policies[policies["cancellationDateOfFloodPolicy"].isna()]
        log_rows(logger, policies, "after excluding cancelled terms")
    policies = policies.assign(
        county=county_from_geoid(policies[county_col_policies_src]),
        flood_zone_group=apply_zone_group(policies["ratedFloodZone"], require(cfg, "flood_zone_groups")),
    )
    claims = claims.rename(columns={county_col_claims: "county"})

    claim_counts = claims.groupby(["county", zone_col]).size().rename("n_claims")
    policy_counts = policies.groupby(["county", zone_col]).size().rename("n_policies")
    cells = pd.concat([claim_counts, policy_counts], axis=1).fillna(0).reset_index()
    cells["n_claims"] = cells["n_claims"].astype(int)
    cells["n_policies"] = cells["n_policies"].astype(int)

    county_claims = claims.groupby("county").size().rename("n_claims_county")
    county_policies = policies.groupby("county").size().rename("n_policies_county")

    statewide_claims = int(claims.shape[0])
    statewide_policies = int(policies.shape[0])
    statewide_rate = statewide_claims / (statewide_policies * n_years)
    logger.info("Statewide: %s claims / (%s policies x %d years) = %.5f claims/policy-year",
               f"{statewide_claims:,}", f"{statewide_policies:,}", n_years, statewide_rate)

    county_tbl = pd.concat([county_claims, county_policies], axis=1).fillna(0).reset_index()
    county_tbl["raw_county_rate"] = county_tbl["n_policies_county"].mask(
        county_tbl["n_policies_county"] == 0).pipe(
        lambda denom: county_tbl["n_claims_county"] / (denom * n_years))
    county_tbl["county_smoothed_rate"], _ = credibility_blend(
        county_tbl["n_claims_county"], county_tbl["raw_county_rate"], statewide_rate, k_county)

    cells = cells.merge(county_tbl[["county", "county_smoothed_rate"]], on="county", how="left")
    cells["raw_cell_rate"] = cells["n_policies"].mask(cells["n_policies"] == 0).pipe(
        lambda denom: cells["n_claims"] / (denom * n_years))
    cells["smoothed_rate"], cells["credibility_weight"] = credibility_blend(
        cells["n_claims"], cells["raw_cell_rate"], cells["county_smoothed_rate"], k_cell)

    out = cells[["county", zone_col, "n_claims", "n_policies", "raw_cell_rate",
                "county_smoothed_rate", "credibility_weight", "smoothed_rate"]].rename(
        columns={"raw_cell_rate": "raw_rate", "smoothed_rate": "frequency"})
    out = out.sort_values(["county", zone_col]).reset_index(drop=True)

    tables = get_path(cfg, "tables")
    out.to_csv(tables / "frequency_by_county_zone.csv", index=False)
    log_rows(logger, out, "frequency_by_county_zone.csv")
    logger.info("Frequency range: min %.5f, median %.5f, max %.5f (statewide %.5f)",
               out["frequency"].min(), out["frequency"].median(), out["frequency"].max(), statewide_rate)


def apply_zone_group(series: pd.Series, groups: dict) -> pd.Series:
    lookup = {str(code): name for name, codes in groups.items() for code in codes}
    as_text = series.astype("string")
    out = as_text.map(lookup).astype("string")
    return out.mask(as_text.notna() & out.isna(), "other")


if __name__ == "__main__":
    main()
