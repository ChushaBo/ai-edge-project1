"""Download NOAA Storm Events "details" files and the NWS zone-to-county file.

Inputs:
  - NOAA NCEI Storm Events bulk CSV directory (network), sources.storms_base_url.
    One details file per year in years.claims_start..years.claims_end. File names
    carry a creation date (StormEvents_details-ftp_v1.0_dYYYY_cYYYYMMDD.csv.gz), so
    they are discovered from the directory listing; the newest file per year wins.
  - NWS zone-to-county correlation page (network), sources.nws_zone_county_page,
    which links bpDDmmYY.dbx files (one per zone-change date).
  - config.yaml: sources.storms_*, sources.nws_zone_county_*, years.claims_start,
    years.claims_end, storm_event_types (only for reporting), paths.raw,
    paths.tables, paths.data_dictionaries
Outputs:
  - data/raw/storms_csv/StormEvents_details-*.csv.gz: raw yearly files (cache).
  - data/raw/storms_fl.parquet: STATE = FLORIDA rows, columns (NOAA names, unchanged):
    EPISODE_ID, EVENT_ID, STATE, STATE_FIPS, YEAR, EVENT_TYPE, BEGIN_DATE_TIME,
    END_DATE_TIME, CZ_TIMEZONE, CZ_TYPE, CZ_FIPS, CZ_NAME, DAMAGE_PROPERTY,
    EPISODE_NARRATIVE, EVENT_NARRATIVE. BEGIN/END_DATE_TIME are naive local times
    in CZ_TIMEZONE. DAMAGE_PROPERTY is kept as NOAA's raw text (e.g. "10.00K").
    CZ_FIPS is a county FIPS (3 digits) when CZ_TYPE = C and a forecast zone
    number when CZ_TYPE = Z.
  - data/raw/nws_zone_county/bpDDmmYY.dbx: every zone-county file linked from the
    NWS page, saved unchanged. Pipe-delimited, no header. Columns, in order
    (see utils.storms.ZONE_COUNTY_COLUMNS): STATE, ZONE, CWA, NAME, STATE_ZONE,
    COUNTY, FIPS (5-digit state+county), TIME_ZONE, FE_AREA, LAT, LON. A zone that
    covers several counties has one row per county.
  - docs/data_dictionaries/noaa_storm_events_bulk_csv_format.pdf (NOAA's format doc)
  - docs/data_dictionaries/nws_zone_county_fields.csv
  - outputs/tables/storms_columns.csv (column, dtype, non-null, percent missing)
Flags:
  - --force: re-download listings, yearly files and zone files, and rebuild outputs.
Idempotent: skips work whose outputs exist. Stops (saves nothing) if a year has no
file, a required column is missing, a year has zero Florida rows, or EVENT_IDs repeat.
"""

import argparse
import datetime as dt
import gzip

import pandas as pd

from utils import storms
from utils.config import get_path, load_config, require
from utils.download import download_file, get_text, make_session
from utils.io import column_summary, write_parquet_atomic
from utils.log import get_logger, log_rows

logger = get_logger("03_download_storms")

ZONE_FIELD_DESCRIPTIONS = {
    "STATE": "Two character state abbreviation",
    "ZONE": "Three character zone number",
    "CWA": "Three character CWA ID (of the zone, starting with 01 May 2018 file)",
    "NAME": "Zone name",
    "STATE_ZONE": "5 character state + three character zone number",
    "COUNTY": "County name",
    "FIPS": "5 character state-county FIPS code",
    "TIME_ZONE": "Time zone of polygon",
    "FE_AREA": "Feature Area (location in STATE)",
    "LAT": "Latitude of centroid of the zone",
    "LON": "Longitude of centroid of the zone",
}


def save_dictionaries(session, cfg, force: bool) -> None:
    dict_dir = get_path(cfg, "data_dictionaries")
    pdf_url = require(cfg, "sources.storms_base_url") + "Storm-Data-Bulk-csv-Format.pdf"
    if download_file(session, pdf_url, dict_dir / "noaa_storm_events_bulk_csv_format.pdf", force):
        logger.info("Saved NOAA bulk CSV format doc from %s", pdf_url)
    fields_path = dict_dir / "nws_zone_county_fields.csv"
    if force or not fields_path.exists():
        pd.DataFrame({
            "position": range(1, len(storms.ZONE_COUNTY_COLUMNS) + 1),
            "name": storms.ZONE_COUNTY_COLUMNS,
            "description": [ZONE_FIELD_DESCRIPTIONS[c] for c in storms.ZONE_COUNTY_COLUMNS],
        }).to_csv(fields_path, index=False)
        logger.info("Saved zone-county field list to %s (source: %s)",
                    fields_path, require(cfg, "sources.nws_zone_county_page"))


def download_zone_county(session, cfg, force: bool) -> list:
    zone_dir = get_path(cfg, "raw") / "nws_zone_county"
    existing = sorted(zone_dir.glob("bp*.dbx")) if zone_dir.exists() else []
    if existing and not force:
        logger.info("Zone-county files exist, not re-fetching: %s", [p.name for p in existing])
        return existing
    page = require(cfg, "sources.nws_zone_county_page")
    base = require(cfg, "sources.nws_zone_county_base")
    names = storms.parse_zone_county_links(get_text(session, page))
    logger.info("Zone-county files linked from %s: %s", page, names)
    for name in names:
        download_file(session, base + name, zone_dir / name, force=force)
    return sorted(zone_dir.glob("bp*.dbx"))


def build_storms(session, cfg, force: bool) -> pd.DataFrame:
    base = require(cfg, "sources.storms_base_url")
    state_name = require(cfg, "sources.storms_state_name")
    years = range(int(require(cfg, "years.claims_start")), int(require(cfg, "years.claims_end")) + 1)
    files = storms.parse_details_listing(get_text(session, base), years)
    cache = get_path(cfg, "raw") / "storms_csv"

    frames = []
    for year, name in files.items():
        path = cache / name
        if download_file(session, base + name, path, force=force):
            logger.info("Downloaded %s", name)
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            raw = pd.read_csv(f, dtype=str, keep_default_na=False, na_values=[""])
        storms.check_details_header(list(raw.columns))
        fl = storms.clean_details(raw, state_name)
        logger.info("%s: %s rows nationwide, %s rows with STATE = %s",
                    name, f"{len(raw):,}", f"{len(fl):,}", state_name)
        if fl.empty:
            raise SystemExit(f"STOP: {name} has no rows with STATE = {state_name}.")
        off_year = int((fl["YEAR"] != year).sum())
        if off_year:
            logger.warning("%s: %d rows have YEAR != %d", name, off_year, year)
        frames.append(fl)

    df = pd.concat(frames, ignore_index=True)
    log_rows(logger, df, "Florida storm events, all years")
    dupes = int(df["EVENT_ID"].duplicated().sum())
    if dupes:
        raise SystemExit(f"STOP: {dupes:,} duplicate EVENT_ID values. Nothing was saved.")
    return df


def report(df: pd.DataFrame, zone_files: list, cfg) -> None:
    for year, n in df["YEAR"].value_counts().sort_index().items():
        logger.info("  YEAR %s: %s rows", year, f"{n:,}")
    logger.info("Rows per EVENT_TYPE:")
    for event_type, n in df["EVENT_TYPE"].value_counts().items():
        logger.info("  %-28s %s", event_type, f"{n:,}")
    shares = df["CZ_TYPE"].value_counts(normalize=True).mul(100).round(1)
    logger.info("CZ_TYPE shares (%%): %s", shares.to_dict())

    configured = cfg.get("storm_event_types") or []
    present = set(df["EVENT_TYPE"].dropna())
    counts = {t: int((df["EVENT_TYPE"] == t).sum()) for t in configured}
    logger.info("Config storm_event_types in Florida data: %s", counts)
    absent = [t for t in configured if t not in present]
    if absent:
        logger.warning("Config storm_event_types with zero Florida rows (check spelling): %s",
                       absent)
    keywords = ("flood", "rain", "surge", "tide", "hurricane", "tropical", "typhoon")
    related = sorted(t for t in present - set(configured) if any(k in t.lower() for k in keywords))
    if related:
        logger.info("Flood-related EVENT_TYPEs present but not in config: %s", related)

    # Heads-up for step 3.1: do Florida zone events map with the zone-county files?
    current = storms.pick_current_zone_file([p.name for p in zone_files], dt.date.today())
    zone_sets = {p.name: set(storms.read_zone_county(p).query("STATE == 'FL'")["STATE_ZONE"])
                 for p in zone_files}
    z = df[df["CZ_TYPE"] == "Z"]
    keys = "FL" + z["CZ_FIPS"].astype(int).astype(str).str.zfill(3)
    logger.info("Zone-county file in effect today: %s (%d Florida zones)",
                current, len(zone_sets[current]))
    for name, zones in zone_sets.items():
        unmatched = ~keys.isin(zones)
        logger.info("  Z events whose zone is not in %s: %s of %s (%.1f%%)", name,
                    f"{unmatched.sum():,}", f"{len(z):,}", 100 * unmatched.mean())
    union = set().union(*zone_sets.values())
    by_year = (~keys.isin(union)).groupby(z["YEAR"]).mean().mul(100).round(1)
    if by_year.any():
        logger.warning("Share of Z events unmatched by any saved zone file, by year (%%): %s",
                       by_year[by_year > 0].to_dict())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download even if outputs exist")
    args = parser.parse_args()

    cfg = load_config()
    out_path = get_path(cfg, "raw") / "storms_fl.parquet"
    session = make_session()

    save_dictionaries(session, cfg, args.force)
    zone_files = download_zone_county(session, cfg, args.force)

    if out_path.exists() and not args.force:
        logger.info("Output exists, skipping download (use --force to refresh): %s", out_path)
        df = pd.read_parquet(out_path)
    else:
        df = build_storms(session, cfg, args.force)
        write_parquet_atomic(df, out_path)
        logger.info("Saved %s", out_path)

    log_rows(logger, df, "storms_fl.parquet total")
    report(df, zone_files, cfg)

    tables_path = get_path(cfg, "tables") / "storms_columns.csv"
    column_summary(df).to_csv(tables_path, index=False)
    logger.info("Saved column summary (%d columns) to %s", df.shape[1], tables_path)


if __name__ == "__main__":
    main()
