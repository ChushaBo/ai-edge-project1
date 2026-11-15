"""Helpers for NOAA Storm Events bulk CSVs and the NWS zone-to-county correlation file.

Used by 03_download_storms.py (download) and later by 05_match_narratives.py (matching).
"""

import datetime as dt
import re
from pathlib import Path

import pandas as pd

# --- NOAA Storm Events "details" files ---------------------------------------------

# e.g. StormEvents_details-ftp_v1.0_d2022_c20260625.csv.gz
DETAILS_RE = re.compile(r"StormEvents_details-ftp_v1\.0_d(\d{4})_c(\d{8})\.csv\.gz")

# Columns kept from each details file. Names are NOAA's, unchanged.
# The RUNBOOK 1.3 list, plus STATE, STATE_FIPS, YEAR (filter/QA) and CZ_TIMEZONE
# (BEGIN/END_DATE_TIME are local time in that zone).
DETAILS_COLUMNS = [
    "EPISODE_ID", "EVENT_ID", "STATE", "STATE_FIPS", "YEAR", "EVENT_TYPE",
    "BEGIN_DATE_TIME", "END_DATE_TIME", "CZ_TIMEZONE",
    "CZ_TYPE", "CZ_FIPS", "CZ_NAME", "DAMAGE_PROPERTY",
    "EPISODE_NARRATIVE", "EVENT_NARRATIVE",
]
_INT_COLUMNS = ["EPISODE_ID", "EVENT_ID", "STATE_FIPS", "YEAR", "CZ_FIPS"]
_DATETIME_COLUMNS = ["BEGIN_DATE_TIME", "END_DATE_TIME"]
NOAA_DATETIME_FORMAT = "%d-%b-%y %H:%M:%S"  # e.g. 20-FEB-22 21:18:00 (local time)


def parse_details_listing(html: str, years: range) -> dict[int, str]:
    """Map each requested year to its details file name in a directory listing.

    If a year has several files (different creation dates), the newest wins.
    Raises SystemExit if any requested year has no file.
    """
    latest: dict[int, tuple[str, str]] = {}
    for match in DETAILS_RE.finditer(html):
        year, created = int(match.group(1)), match.group(2)
        if year in years and (year not in latest or created > latest[year][0]):
            latest[year] = (created, match.group(0))
    missing = [y for y in years if y not in latest]
    if missing:
        raise SystemExit(f"STOP: no Storm Events details file listed for year(s) {missing}.")
    return {y: latest[y][1] for y in years}


def check_details_header(columns: list[str]) -> None:
    """Stop if a details file lacks any column we rely on (do not guess names)."""
    missing = [c for c in DETAILS_COLUMNS if c not in columns]
    if missing:
        raise SystemExit(f"STOP: details file lacks expected column(s) {missing}. "
                         f"Columns present: {columns}")


def clean_details(raw: pd.DataFrame, state_name: str) -> pd.DataFrame:
    """Keep DETAILS_COLUMNS for one state and cast types. Columns are not renamed."""
    df = raw.loc[raw["STATE"] == state_name, DETAILS_COLUMNS].copy()
    for col in _INT_COLUMNS:
        df[col] = pd.to_numeric(df[col]).astype("Int64")
    for col in _DATETIME_COLUMNS:
        df[col] = pd.to_datetime(df[col], format=NOAA_DATETIME_FORMAT)
    for col in df.columns.difference(_INT_COLUMNS + _DATETIME_COLUMNS):
        df[col] = df[col].astype("string")
    return df.reset_index(drop=True)


# --- NWS zone-to-county correlation file (bpDDmmYY.dbx) ----------------------------

# Pipe-delimited, no header row. Layout per https://www.weather.gov/gis/ZoneCounty.
ZONE_COUNTY_COLUMNS = [
    "STATE",       # two-character state abbreviation
    "ZONE",        # three-character zone number
    "CWA",         # three-character CWA (forecast office) ID of the zone
    "NAME",        # zone name
    "STATE_ZONE",  # state + three-character zone number, e.g. FL021
    "COUNTY",      # county name
    "FIPS",        # five-character state-county FIPS code, e.g. 12121
    "TIME_ZONE",   # time zone of the polygon
    "FE_AREA",     # feature area (location within the state)
    "LAT",         # latitude of the zone centroid
    "LON",         # longitude of the zone centroid
]

ZONE_FILE_RE = re.compile(r"bp(\d{2})([a-z]{2})(\d{2})\.dbx")
# NWS two-letter month codes used in the file names (e.g. bp18mr25, bp16ap26).
_NWS_MONTHS = {"ja": 1, "fe": 2, "mr": 3, "ap": 4, "my": 5, "jn": 6,
               "jl": 7, "au": 8, "se": 9, "oc": 10, "no": 11, "de": 12}


def zone_file_date(name: str) -> dt.date:
    """Effective date encoded in a zone-county file name, e.g. bp16ap26.dbx -> 2026-04-16."""
    match = ZONE_FILE_RE.fullmatch(Path(name).name)
    if not match or match.group(2) not in _NWS_MONTHS:
        raise ValueError(f"Not a recognised zone-county file name: {name}")
    day, month, year = int(match.group(1)), _NWS_MONTHS[match.group(2)], 2000 + int(match.group(3))
    return dt.date(year, month, day)


_HREF_RE = re.compile(r'href="([^"]*)"', re.IGNORECASE)


def parse_zone_county_links(html: str) -> list[str]:
    """Unique zone-county file names linked (href) from the NWS page, oldest first.

    Only hrefs count: the page text also mentions retired files (e.g. "bp01my18.dbx -
    now invalid") that are no longer downloadable.
    """
    names = set()
    for href in _HREF_RE.findall(html):
        match = ZONE_FILE_RE.search(href.rsplit("/", 1)[-1])
        if match:
            names.add(match.group(0))
    names = sorted(names, key=zone_file_date)
    if not names:
        raise SystemExit("STOP: no bpDDmmYY.dbx zone-county file linked from the NWS page.")
    return names


def pick_current_zone_file(names: list[str], today: dt.date) -> str:
    """The file in effect today: latest effective date on or before today."""
    in_effect = [n for n in names if zone_file_date(n) <= today]
    if not in_effect:
        raise SystemExit(f"STOP: no zone-county file in effect on {today}: {names}")
    return max(in_effect, key=zone_file_date)


def read_zone_county(path: Path) -> pd.DataFrame:
    """Read a zone-county .dbx file with named columns; codes kept as strings."""
    return pd.read_csv(path, sep="|", header=None, names=ZONE_COUNTY_COLUMNS,
                       dtype=str, keep_default_na=False, encoding="utf-8")
