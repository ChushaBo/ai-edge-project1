"""Claim-to-storm-event matching helpers (used by 05_match_narratives).

Two parts:
  1. event_counties(): map each NOAA event to county FIPS codes.
     - CZ_TYPE = C: the county is STATE_FIPS + CZ_FIPS.
     - CZ_TYPE = Z: forecast zones, via the NWS zone-to-county file (a zone can cover
       several counties; all are kept) and via the county name inside CZ_NAME
       (e.g. "COASTAL ST. JOHNS"). Zones were renumbered over the years, so an old
       zone number can be missing from the current file or point to another county;
       the name is checked against the zone file (see resolve_zone_events).
  2. match_claims(): claims x events in the same county whose begin date falls
     within [date of loss - days_before, date of loss + days_after], compared as
     calendar dates (NOAA times are local; claim dates are UTC midnight).
"""

import re

import pandas as pd

# Words in NOAA/NWS zone names that qualify a county but are not part of its name.
_QUALIFIERS = {
    "COUNTY", "COASTAL", "INLAND", "INALND", "NORTH", "NORTHERN", "SOUTH", "SOUTHERN",
    "EAST", "EASTERN", "WEST", "WESTERN", "CENTRAL", "INTERIOR", "MAINLAND", "UPPER",
    "MIDDLE", "LOWER", "KEYS", "METRO", "METROPOLITAN", "FAR", "NORTHEAST", "NORTHWEST",
    "SOUTHEAST", "SOUTHWEST", "NORTHEASTERN", "NORTHWESTERN", "SOUTHEASTERN",
    "SOUTHWESTERN", "BEACHES", "COAST", "BARRIER", "ISLANDS", "WR",
}
# Old names and typos seen in NOAA CZ_NAME values.
_ALIASES = {"DADE": "MIAMIDADE", "SAINTJOHNS": "STJOHNS", "SAINTLUCIE": "STLUCIE",
            "PUTNUM": "PUTNAM"}


def filter_event_types(events: pd.DataFrame, types: list[str]) -> pd.DataFrame:
    """Keep events whose EVENT_TYPE is in `types`; stop if a configured type never occurs
    (usually a spelling mismatch, e.g. "Hurricane" vs NOAA's "Hurricane (Typhoon)")."""
    missing = sorted(set(types) - set(events["EVENT_TYPE"]))
    if missing:
        raise SystemExit(f"STOP: storm_event_types not found in NOAA data: {missing}")
    return events[events["EVENT_TYPE"].isin(types)]


def county_key(name: str | None) -> str | None:
    """Normalize a county or zone name to a lookup key: 'Coastal St. Johns' -> 'STJOHNS'."""
    if name is None or pd.isna(name):
        return None
    text = re.sub(r"[^A-Z ]", " ", str(name).upper().replace("SAINT ", "ST "))
    words = [w for w in text.split() if w not in _QUALIFIERS]
    if "IN" in words:  # NWS style: "Upper Keys in Monroe"
        words = words[words.index("IN") + 1:]
    key = "".join(words)
    return _ALIASES.get(key, key) or None


def county_name_lookup(county_events: pd.DataFrame, zone_county: pd.DataFrame,
                       state_fips: str) -> dict[str, str]:
    """county_key -> 5-digit FIPS, from county-coded NOAA events and the NWS zone file."""
    lookup: dict[str, str] = {}
    for name, fips in zip(zone_county["COUNTY"], zone_county["FIPS"]):
        if (key := county_key(name)):
            lookup[key] = fips
    for name, cz in zip(county_events["CZ_NAME"], county_events["CZ_FIPS"]):
        if (key := county_key(name)):
            lookup.setdefault(key, f"{state_fips}{int(cz):03d}")
    return lookup


def resolve_zone_events(zone_events: pd.DataFrame, zone_county: pd.DataFrame,
                        names: dict[str, str]) -> pd.DataFrame:
    """One row per (EVENT_ID, county_fips) for zone-coded events, with the method used.

    Rules, per event:
      - zone in the zone file and its counties include the CZ_NAME county -> zone counties
        ("zone").
      - zone in the file but CZ_NAME names a different county -> the CZ_NAME county
        ("name_over_zone"; the zone number was reused after renumbering).
      - zone not in the file -> the CZ_NAME county ("name").
      - zone in the file and CZ_NAME has no recognizable county -> zone counties
        ("zone_only").
      - neither -> dropped ("unresolved"), reported by the caller.
    """
    zmap = zone_county.groupby("ZONE")["FIPS"].agg(lambda s: sorted(set(s))).to_dict()
    rows = []
    for event_id, zone, cz_name in zip(zone_events["EVENT_ID"], zone_events["zone"],
                                       zone_events["CZ_NAME"]):
        zone_fips = zmap.get(zone, [])
        name_fips = names.get(county_key(cz_name))
        if zone_fips and name_fips:
            if name_fips in zone_fips:
                method, fips_list = "zone", zone_fips
            else:
                method, fips_list = "name_over_zone", [name_fips]
        elif name_fips:
            method, fips_list = "name", [name_fips]
        elif zone_fips:
            method, fips_list = "zone_only", zone_fips
        else:
            method, fips_list = "unresolved", []
        rows += [{"EVENT_ID": event_id, "county_fips": f, "method": method} for f in fips_list]
        if not fips_list:
            rows.append({"EVENT_ID": event_id, "county_fips": None, "method": method})
    return pd.DataFrame(rows, columns=["EVENT_ID", "county_fips", "method"])


def event_counties(events: pd.DataFrame, zone_county: pd.DataFrame,
                   state_fips: str) -> pd.DataFrame:
    """Events with one row per county they cover: adds county_fips, map_method, begin_date."""
    events = events.copy()
    county_ev = events[events["CZ_TYPE"] == "C"]
    zone_ev = events[events["CZ_TYPE"] == "Z"].assign(
        zone=lambda d: d["CZ_FIPS"].astype(int).map("{:03d}".format))

    names = county_name_lookup(county_ev, zone_county, state_fips)
    # astype(str) keeps an empty column as text, so the concatenation works with no C events
    by_county = county_ev.assign(
        county_fips=state_fips + county_ev["CZ_FIPS"].astype(int).map("{:03d}".format).astype(str),
        map_method="county")
    resolved = resolve_zone_events(zone_ev, zone_county, names)
    by_zone = zone_ev.drop(columns="zone").merge(
        resolved.rename(columns={"method": "map_method"}), on="EVENT_ID", how="left")

    out = pd.concat([by_county, by_zone], ignore_index=True)
    out["begin_date"] = pd.to_datetime(out["BEGIN_DATE_TIME"]).dt.normalize()
    return out


def match_claims(claims: pd.DataFrame, events: pd.DataFrame, county_col: str,
                 date_col: str, days_before: int, days_after: int) -> pd.DataFrame:
    """All (claim, event) pairs: same county, event begin date in
    [loss - days_before, loss + days_after] (calendar dates). Returns claim id + event columns."""
    loss_day = claims[date_col].dt.tz_localize(None).dt.normalize() \
        if claims[date_col].dt.tz is not None else claims[date_col].dt.normalize()
    base = pd.DataFrame({"id": claims["id"].to_numpy(), "county_fips": claims[county_col].to_numpy(),
                         "loss_day": loss_day.to_numpy()})
    candidates = pd.concat(
        [base.assign(begin_date=base["loss_day"] + pd.Timedelta(days=d))
         for d in range(-days_before, days_after + 1)], ignore_index=True)
    ev = events.dropna(subset=["county_fips"])
    return candidates.merge(ev, on=["county_fips", "begin_date"], how="inner")


def flag_quarantine(df: pd.DataFrame, rules: list[dict], date_col: str) -> pd.DataFrame:
    """Add quarantined (bool) and quarantine_reason (rule name) columns. Rows are kept.

    A rule matches claims whose loss date (calendar day) is within
    [loss_date_from, loss_date_to], restricted to claims without a narrative when
    only_unmatched is true. The first matching rule names the reason.
    """
    day = df[date_col].dt.tz_convert(None).dt.normalize() if df[date_col].dt.tz is not None \
        else df[date_col].dt.normalize()
    reason = pd.Series(pd.NA, index=df.index, dtype="string")
    for rule in rules or []:
        hit = day.between(pd.Timestamp(rule["loss_date_from"]), pd.Timestamp(rule["loss_date_to"]))
        if rule.get("only_unmatched"):
            hit &= ~df["has_narrative"]
        reason = reason.mask(hit & reason.isna(), rule["name"])
    return df.assign(quarantined=reason.notna(), quarantine_reason=reason)


def exclude_quarantined(df: pd.DataFrame, cfg: dict, model: str | None = None) -> pd.DataFrame:
    """Drop quarantined claims for `model` (for steps 06+).

    Quarantine is model-scoped (config quarantine.applies_to, decided 2026-11-15): rows
    are dropped only when quarantine.enabled is true AND model is in applies_to. Pass
    model=None to apply the drop unconditionally on enabled (ignores applies_to) -- used
    where the caller is not one specific model, e.g. a shared text-feature table that
    only text models ever read.
    """
    q = cfg.get("quarantine", {})
    scoped = model is None or model in q.get("applies_to", [])
    if q.get("enabled") and scoped and "quarantined" in df:
        return df[~df["quarantined"]]
    return df


def combine_narratives(pairs: pd.DataFrame) -> pd.DataFrame:
    """Per claim: unique episode narratives (each once, even if several matched events
    share an episode) followed by unique event narratives, plus match metadata."""
    pairs = pairs.sort_values(["id", "begin_date", "EPISODE_ID", "EVENT_ID"])

    def unique_text(values) -> list[str]:
        seen, out = set(), []
        for v in values:
            if isinstance(v, str) and v.strip() and v.strip() not in seen:
                seen.add(v.strip())
                out.append(v.strip())
        return out

    rows = []
    for claim_id, g in pairs.groupby("id", sort=False):
        episodes = unique_text(g.drop_duplicates("EPISODE_ID")["EPISODE_NARRATIVE"])
        events = unique_text(g["EVENT_NARRATIVE"])
        rows.append({
            "id": claim_id,
            "narrative": "\n\n".join(episodes + events),
            "n_matched_events": g["EVENT_ID"].nunique(),
            "matched_event_types": ";".join(sorted(g["EVENT_TYPE"].unique())),
            "matched_episode_ids": ";".join(str(x) for x in sorted(g["EPISODE_ID"].unique())),
            "matched_event_ids": ";".join(str(x) for x in sorted(g["EVENT_ID"].unique())),
        })
    return pd.DataFrame(rows)
