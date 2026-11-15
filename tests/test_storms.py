"""Offline tests for utils.storms (NOAA listing/columns, NWS zone-county file names)."""

import datetime as dt

import pandas as pd
import pytest

from utils import storms

LISTING = """
<a href="StormEvents_details-ftp_v1.0_d2021_c20260323.csv.gz">...</a>
<a href="StormEvents_details-ftp_v1.0_d2022_c20250101.csv.gz">...</a>
<a href="StormEvents_details-ftp_v1.0_d2022_c20260625.csv.gz">...</a>
<a href="StormEvents_fatalities-ftp_v1.0_d2022_c20260625.csv.gz">...</a>
"""


def test_listing_picks_newest_details_file_per_year():
    files = storms.parse_details_listing(LISTING, range(2021, 2023))
    assert files == {
        2021: "StormEvents_details-ftp_v1.0_d2021_c20260323.csv.gz",
        2022: "StormEvents_details-ftp_v1.0_d2022_c20260625.csv.gz",
    }


def test_listing_stops_on_missing_year():
    with pytest.raises(SystemExit, match="2023"):
        storms.parse_details_listing(LISTING, range(2021, 2024))


def test_header_check_stops_on_missing_column():
    cols = [c for c in storms.DETAILS_COLUMNS if c != "CZ_FIPS"]
    with pytest.raises(SystemExit, match="CZ_FIPS"):
        storms.check_details_header(cols)


def test_clean_details_filters_state_and_casts():
    row = {c: "" for c in storms.DETAILS_COLUMNS}
    fl = {**row, "EPISODE_ID": "1", "EVENT_ID": "10", "STATE": "FLORIDA", "STATE_FIPS": "12",
          "YEAR": "2022", "EVENT_TYPE": "Storm Surge/Tide", "CZ_TYPE": "Z", "CZ_FIPS": "52",
          "BEGIN_DATE_TIME": "28-SEP-22 06:00:00", "END_DATE_TIME": "29-SEP-22 18:00:00",
          "DAMAGE_PROPERTY": "1.50M"}
    ga = {**fl, "EVENT_ID": "11", "STATE": "GEORGIA"}
    raw = pd.DataFrame([fl, ga]).assign(EXTRA="dropped")

    out = storms.clean_details(raw, "FLORIDA")

    assert list(out.columns) == storms.DETAILS_COLUMNS
    assert out["EVENT_ID"].tolist() == [10]
    assert out["CZ_FIPS"].dtype == "Int64" and out["CZ_FIPS"][0] == 52
    assert out["BEGIN_DATE_TIME"][0] == pd.Timestamp("2022-09-28 06:00:00")
    assert out["DAMAGE_PROPERTY"][0] == "1.50M"  # raw text kept


def test_zone_file_dates_and_current_pick():
    assert storms.zone_file_date("bp16ap26.dbx") == dt.date(2026, 4, 16)
    assert storms.zone_file_date("bp18mr25.dbx") == dt.date(2025, 3, 18)
    names = ["bp18mr25.dbx", "bp16ap26.dbx"]
    assert storms.pick_current_zone_file(names, dt.date(2026, 11, 15)) == "bp16ap26.dbx"
    assert storms.pick_current_zone_file(names, dt.date(2026, 1, 1)) == "bp18mr25.dbx"
    with pytest.raises(ValueError):
        storms.zone_file_date("bp32xx26.dbx")


def test_zone_links_ignore_text_mentions():
    html = ('<p>Starting with the 1 May 2018 file (bp01my18.dbx - now invalid)</p>'
            '<a href="/source/gis/Shapefiles/County/bp16ap26.dbx">bp16ap26.dbx</a>'
            '<a href="/source/gis/Shapefiles/County/bp18mr25.dbx">bp18mr25.dbx</a>')
    assert storms.parse_zone_county_links(html) == ["bp18mr25.dbx", "bp16ap26.dbx"]


def test_read_zone_county_keeps_codes_as_strings(tmp_path):
    path = tmp_path / "bp16ap26.dbx"
    path.write_text("FL|021|JAX|Suwannee|FL021|Suwannee|12121|E|nn|30.1956|-82.9915\n",
                    encoding="utf-8")
    zc = storms.read_zone_county(path)
    assert list(zc.columns) == storms.ZONE_COUNTY_COLUMNS
    assert zc.loc[0, "ZONE"] == "021" and zc.loc[0, "FIPS"] == "12121"
