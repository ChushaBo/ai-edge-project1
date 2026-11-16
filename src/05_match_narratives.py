"""Match each claim to NOAA storm events and attach combined narratives.

Method (helpers in utils.matching):
  1. Keep Florida events whose EVENT_TYPE is in config storm_event_types.
  2. Map events to county FIPS: CZ_TYPE = C directly; CZ_TYPE = Z via the NWS
     zone-to-county files (all counties of a zone kept), checked against the county
     named in CZ_NAME. Zone numbers missing from the current files (renumbered zones)
     fall back to the CZ_NAME county. Every zone file in data/raw/nws_zone_county/
     is used.
  3. A claim matches an event in the same county whose begin date (calendar day,
     NOAA local time) is within [date of loss - match_window.days_before,
     date of loss + match_window.days_after].
  4. Per claim, combine unique episode narratives (repeated episode text once) and
     unique event narratives.
  5. Flag claims covered by config quarantine.rules (quarantined, quarantine_reason).
     Rows are kept; steps 06+ skip them while quarantine.enabled is true.
Matching is a lookup, not a fitted transform, so it is applied to all splits alike.

Inputs:
  - data/interim/claims_clean.parquet (from 04)
  - data/raw/storms_fl.parquet and data/raw/nws_zone_county/*.dbx (from 03)
  - config.yaml: storm_event_types, match_window.*, quarantine.*, scope.state_fips,
    fields.claims_county, fields.claims_date_of_loss, random_seed, paths.*
Outputs:
  - data/interim/claims_with_text.parquet: claims_clean columns plus narrative,
    n_matched_events, matched_event_types, matched_episode_ids, matched_event_ids,
    has_narrative, quarantined, quarantine_reason.
  - outputs/tables/match_rates.csv: match rate overall, by split, by year, and overall
    excluding quarantined claims.
  - outputs/tables/event_county_mapping.csv: events by CZ_TYPE and mapping method.
  - outputs/tables/match_spot_check.csv: 20 random matched claims (date, county,
    event types, narrative excerpt) for human review (RUNBOOK 3.2).
"""

import pandas as pd

from utils import matching
from utils.config import get_path, load_config, require
from utils.io import write_parquet_atomic
from utils.log import get_logger, log_rows
from utils.storms import read_zone_county

logger = get_logger("05_match_narratives")

EXCERPT_CHARS = 400


def load_zone_county(raw_dir, state_abbr: str) -> pd.DataFrame:
    files = sorted((raw_dir / "nws_zone_county").glob("*.dbx"))
    if not files:
        raise SystemExit("STOP: no NWS zone-county files in data/raw/nws_zone_county/ (run 03)")
    zc = pd.concat([read_zone_county(f) for f in files], ignore_index=True)
    zc = zc[zc["STATE"] == state_abbr].drop_duplicates(["ZONE", "FIPS"])
    logger.info("Zone-county files %s: %d %s zones, %d counties",
                [f.name for f in files], zc["ZONE"].nunique(), state_abbr, zc["FIPS"].nunique())
    return zc


def rate_table(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    def block(level, key):
        g = df.groupby(key)["has_narrative"]
        return pd.DataFrame({"level": level, "group": g.mean().index.astype(str),
                             "claims": g.size().to_numpy(), "matched": g.sum().to_numpy(),
                             "match_rate": g.mean().round(4).to_numpy()})
    overall = pd.DataFrame({"level": ["overall"], "group": ["all"], "claims": [len(df)],
                            "matched": [int(df["has_narrative"].sum())],
                            "match_rate": [round(df["has_narrative"].mean(), 4)]})
    return pd.concat([overall, block("split", "split"),
                      block("year", df[date_col].dt.year.rename("year"))], ignore_index=True)


def main() -> None:
    cfg = load_config()
    raw, interim, tables = get_path(cfg, "raw"), get_path(cfg, "interim"), get_path(cfg, "tables")
    county_col = require(cfg, "fields.claims_county")
    date_col = require(cfg, "fields.claims_date_of_loss")
    before = int(require(cfg, "match_window.days_before"))
    after = int(require(cfg, "match_window.days_after"))
    types = list(require(cfg, "storm_event_types"))

    claims = pd.read_parquet(interim / "claims_clean.parquet")
    log_rows(logger, claims, "claims_clean")
    storms = pd.read_parquet(raw / "storms_fl.parquet")
    log_rows(logger, storms, "storm events (all types)")
    storms = matching.filter_event_types(storms, types)
    log_rows(logger, storms, "storm events of configured types")

    zc = load_zone_county(raw, require(cfg, "scope.state_abbr"))
    events = matching.event_counties(storms, zc, require(cfg, "scope.state_fips"))
    mapping = (events.drop_duplicates("EVENT_ID").groupby(["CZ_TYPE", "map_method"])
               .size().rename("events").reset_index())
    for row in mapping.itertuples():
        logger.info("  CZ_TYPE %s mapped by %-15s %5d events", row.CZ_TYPE, row.map_method, row.events)
    unresolved = events["county_fips"].isna()
    if unresolved.any():
        logger.warning("%d zone events could not be mapped to a county (dropped)",
                       events.loc[unresolved, "EVENT_ID"].nunique())
    events = events[~unresolved]
    log_rows(logger, events, "event-county rows")

    pairs = matching.match_claims(claims, events, county_col, date_col, before, after)
    log_rows(logger, pairs, f"claim-event pairs (window -{before}/+{after} days)")
    text = matching.combine_narratives(pairs)

    out = claims.merge(text, on="id", how="left")
    if len(out) != len(claims):
        raise SystemExit(f"STOP: merge changed row count {len(claims)} -> {len(out)}")
    out["n_matched_events"] = out["n_matched_events"].fillna(0).astype("int64")
    for col in ("narrative", "matched_event_types", "matched_episode_ids", "matched_event_ids"):
        out[col] = out[col].fillna("").astype("string")
    out["has_narrative"] = out["narrative"].str.len() > 0

    out = matching.flag_quarantine(out, require(cfg, "quarantine.rules"), date_col)
    for reason, n in out["quarantine_reason"].value_counts().items():
        logger.info("Quarantined (%s): %s claims; kept in the file, flagged", reason, f"{n:,}")
    active = out[~out["quarantined"]]
    logger.info("Match rate excluding quarantined claims: %.1f%% (%s of %s)",
                100 * active["has_narrative"].mean(), f"{int(active['has_narrative'].sum()):,}",
                f"{len(active):,}")

    rates = rate_table(out, date_col)
    rates = pd.concat([rates, pd.DataFrame({
        "level": ["overall_excl_quarantine"], "group": ["all"], "claims": [len(active)],
        "matched": [int(active["has_narrative"].sum())],
        "match_rate": [round(active["has_narrative"].mean(), 4)]})], ignore_index=True)
    logger.info("Match rate overall: %.1f%% (%s of %s claims)", 100 * out["has_narrative"].mean(),
                f"{int(out['has_narrative'].sum()):,}", f"{len(out):,}")
    for r in rates[rates["level"] != "overall"].itertuples():
        logger.info("  %-5s %-10s %7s claims  match %.1f%%", r.level, r.group, f"{r.claims:,}",
                    100 * r.match_rate)
    train = out[(out["split"] == "train") & out["has_narrative"]]
    logger.info("Distinct narratives among matched training claims: %s", f"{train['narrative'].nunique():,}")

    write_parquet_atomic(out, interim / "claims_with_text.parquet")
    logger.info("Saved %s", interim / "claims_with_text.parquet")
    rates.to_csv(tables / "match_rates.csv", index=False)
    mapping.to_csv(tables / "event_county_mapping.csv", index=False)

    spot = out[out["has_narrative"]].sample(n=20, random_state=int(require(cfg, "random_seed")))
    spot = spot.assign(
        date_of_loss=spot[date_col].dt.strftime("%Y-%m-%d"),
        narrative_excerpt=spot["narrative"].str.slice(0, EXCERPT_CHARS).str.replace(r"\s+", " ", regex=True),
        plausible="",  # for the human reviewer (RUNBOOK 3.2): yes / no / unsure
    )[["id", "date_of_loss", county_col, "split", "n_matched_events", "matched_event_types",
       "narrative_excerpt", "plausible"]]
    spot.sort_values("date_of_loss").to_csv(tables / "match_spot_check.csv", index=False)
    logger.info("Saved match_rates.csv, event_county_mapping.csv, match_spot_check.csv to %s", tables)


if __name__ == "__main__":
    main()
