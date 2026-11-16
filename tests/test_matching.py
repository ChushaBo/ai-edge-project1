"""Tests for claim-to-storm matching (utils.matching), on small synthetic data."""

import pandas as pd
import pytest

from utils import matching

ZONES = pd.DataFrame({  # NWS zone-county rows (codes as strings, like the .dbx files)
    "ZONE": ["050", "151", "200", "200"],
    "COUNTY": ["Pinellas", "Hillsborough", "Lee", "Charlotte"],
    "FIPS": ["12103", "12057", "12071", "12015"],
})


def _events(rows):
    cols = ["EVENT_ID", "EPISODE_ID", "EVENT_TYPE", "CZ_TYPE", "CZ_FIPS", "CZ_NAME",
            "BEGIN_DATE_TIME", "EPISODE_NARRATIVE", "EVENT_NARRATIVE"]
    return pd.DataFrame(rows, columns=cols)


def _claims(rows):
    df = pd.DataFrame(rows, columns=["id", "countyCode", "dateOfLoss"])
    df["dateOfLoss"] = pd.to_datetime(df["dateOfLoss"], utc=True)
    return df


def _match(claims, events, before=3, after=3):
    ev = matching.event_counties(events, ZONES, "12")
    return matching.match_claims(claims, ev, "countyCode", "dateOfLoss", before, after)


def test_county_coded_event_matches_same_county():
    events = _events([[1, 10, "Flood", "C", 103, "PINELLAS", "2022-09-28 14:00", "ep", "ev"]])
    pairs = _match(_claims([[100, "12103", "2022-09-28"], [101, "12057", "2022-09-28"]]), events)
    assert pairs["id"].tolist() == [100]  # other county does not match


def test_zone_coded_event_matches_via_zone_file():
    events = _events([[1, 10, "Storm Surge/Tide", "Z", 151, "COASTAL HILLSBOROUGH",
                       "2022-09-28 10:00", "ep", "ev"]])
    pairs = _match(_claims([[100, "12057", "2022-09-28"]]), events)
    assert pairs["id"].tolist() == [100]
    assert pairs["map_method"].tolist() == ["zone"]


def test_multi_county_zone_matches_every_county():
    events = _events([[1, 10, "Hurricane (Typhoon)", "Z", 200, "SOUTHWEST COAST",
                       "2022-09-28 10:00", "ep", "ev"]])
    claims = _claims([[100, "12071", "2022-09-28"], [101, "12015", "2022-09-28"],
                      [102, "12103", "2022-09-28"]])
    assert sorted(_match(claims, events)["id"]) == [100, 101]


def test_renumbered_zone_falls_back_to_county_name():
    # zone 033 is not in the zone file; CZ_NAME still names the county
    events = _events([[1, 10, "Flood", "C", 57, "HILLSBOROUGH", "2015-01-01 00:00", "", ""],
                      [2, 20, "Storm Surge/Tide", "Z", 33, "INALND HILLSBOROUGH",
                       "2015-06-01 09:00", "ep", "ev"]])
    pairs = _match(_claims([[100, "12057", "2015-06-02"]]), events)
    assert pairs["EVENT_ID"].tolist() == [2]
    assert pairs["map_method"].tolist() == ["name"]


@pytest.mark.parametrize("loss_day, matches", [
    ("2022-09-25", True),   # event begins 3 days after loss: inside
    ("2022-09-24", False),  # 4 days: outside
    ("2022-10-01", True),   # event began 3 days before loss: inside
    ("2022-10-02", False),  # 4 days before: outside
])
def test_window_edges_use_calendar_days(loss_day, matches):
    # late-evening local time must still count as 2022-09-28
    events = _events([[1, 10, "Flood", "C", 103, "PINELLAS", "2022-09-28 23:30", "ep", "ev"]])
    pairs = _match(_claims([[100, "12103", loss_day]]), events)
    assert (len(pairs) == 1) is matches


def test_wrong_event_type_is_filtered_and_missing_type_stops():
    events = _events([[1, 10, "Tornado", "C", 103, "PINELLAS", "2022-09-28 10:00", "ep", "ev"],
                      [2, 10, "Flood", "C", 103, "PINELLAS", "2022-09-28 10:00", "ep", "ev2"]])
    kept = matching.filter_event_types(events, ["Flood"])
    assert kept["EVENT_ID"].tolist() == [2]
    with pytest.raises(SystemExit, match="Hurricane"):
        matching.filter_event_types(events, ["Flood", "Hurricane"])


def test_narratives_deduplicate_repeated_episode_text():
    events = _events([
        [1, 10, "Flood", "C", 103, "PINELLAS", "2022-09-28 10:00", "Ian made landfall.", "Street flooding."],
        [2, 10, "Storm Surge/Tide", "C", 103, "PINELLAS", "2022-09-28 12:00", "Ian made landfall.", "Surge of 6 ft."],
        [3, 11, "Flood", "C", 103, "PINELLAS", "2022-09-29 08:00", "Ian made landfall.", ""],
    ])
    text = matching.combine_narratives(_match(_claims([[100, "12103", "2022-09-28"]]), events))
    row = text.iloc[0]
    assert row["narrative"].count("Ian made landfall.") == 1
    assert "Street flooding." in row["narrative"] and "Surge of 6 ft." in row["narrative"]
    assert row["n_matched_events"] == 3
    assert row["matched_event_types"] == "Flood;Storm Surge/Tide"
    assert row["matched_episode_ids"] == "10;11"


def test_county_key_normalizes_noaa_names():
    assert matching.county_key("COASTAL ST. JOHNS") == "STJOHNS"
    assert matching.county_key("INALND ST. JOHNS") == "STJOHNS"
    assert matching.county_key("Coastal Miami Dade County") == "MIAMIDADE"
    assert matching.county_key("MONROE/UPPER KEYS") == "MONROE"
    assert matching.county_key("Upper Keys in Monroe") == "MONROE"
    assert matching.county_key("EASTERN PUTNUM") == "PUTNAM"


def _quarantine_frame():
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "dateOfLoss": pd.to_datetime(["2024-09-26", "2024-09-26", "2024-10-09", "2024-09-30"], utc=True),
        "has_narrative": [False, True, False, False],
    })
    rules = [{"name": "helene_noaa_gap", "loss_date_from": "2024-09-24",
              "loss_date_to": "2024-09-30", "only_unmatched": True}]
    return matching.flag_quarantine(df, rules, "dateOfLoss")


def test_quarantine_flags_only_unmatched_claims_in_date_range():
    out = _quarantine_frame()
    assert out["quarantined"].tolist() == [True, False, False, True]  # 9/30 inclusive
    assert out["quarantine_reason"].fillna("").tolist() == ["helene_noaa_gap", "", "", "helene_noaa_gap"]
    assert len(out) == 4  # rows are kept, only flagged


def test_exclude_quarantined_respects_enabled_switch():
    out = _quarantine_frame()
    assert matching.exclude_quarantined(out, {"quarantine": {"enabled": True}})["id"].tolist() == [2, 3]
    assert len(matching.exclude_quarantined(out, {"quarantine": {"enabled": False}})) == 4


def test_exclude_quarantined_is_model_scoped():
    # Decided 2026-11-15: quarantine applies to the text models only, not baselines or M1.
    out = _quarantine_frame()
    cfg = {"quarantine": {"enabled": True, "applies_to": ["M2", "M2_masked"]}}
    assert matching.exclude_quarantined(out, cfg, model="M1")["id"].tolist() == [1, 2, 3, 4]
    assert matching.exclude_quarantined(out, cfg, model="M2")["id"].tolist() == [2, 3]
    assert matching.exclude_quarantined(out, cfg, model="M2_masked")["id"].tolist() == [2, 3]
    # model=None ignores applies_to and drops unconditionally on `enabled` (06's table).
    assert matching.exclude_quarantined(out, cfg, model=None)["id"].tolist() == [2, 3]


def test_spot_check_rerun_keeps_reviewer_answers(tmp_path):
    import importlib.util
    from utils.config import REPO_ROOT
    spec = importlib.util.spec_from_file_location("match05", REPO_ROOT / "src" / "05_match_narratives.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    path = tmp_path / "match_spot_check.csv"
    assert script.keep_reviews(path, pd.Series([1, 2])) == ["", ""]  # no file yet
    pd.DataFrame({"id": [1, 2], "plausible": ["yes", None]}).to_csv(path, index=False)
    assert script.keep_reviews(path, pd.Series([2, 1, 3])) == ["", "yes", ""]
