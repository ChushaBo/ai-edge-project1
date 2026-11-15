"""Clean claims, build the target and feature table, and assign time-based splits.

Applies the decisions in config.yaml (logged in docs/decisions.md, 2026-11-15):
  1. Drop exact copies: rows identical in every column except id and asOfDate
     (keep the first). cleaning.drop_exact_duplicate_claims
  2. Drop claims with $0 (or missing) building coverage. target.drop_zero_coverage
  3. Drop claims with payout <= 0 (zero, or negative after recoveries).
     target.drop_zero_paid
  4. Target = payout / coverage, capped at target.cap_share_at.
  5. Derived features (config derived_features): building age at loss and policy
     tenure in years (out-of-range values set to missing per cleaning.*), flood zone
     group and occupancy group (config groupings), month of loss (M2 only).
  6. Feature table = ONLY allowed_features + derived features; the raw columns that a
     derived feature replaces (dates, zone, occupancy codes) are dropped.
  7. Split by date of loss: train <= splits.train_end < validation <=
     splits.validation_end < test.
No CPI adjustment: target.definition = share_of_coverage needs none.

Inputs:
  - data/raw/claims_fl.parquet (from 01)
  - config.yaml: fields.claims_*, target.*, cleaning.*, allowed_features,
    derived_features, flood_zone_groups, occupancy_groups, splits.*, paths.*
Outputs:
  - data/interim/claims_clean.parquet: id, dateOfLoss, countyCode, split, target,
    allowed features, derived features.
  - outputs/tables/claims_summary_by_split.csv: claim count and target mean/median
    by split and flood zone group (with an ALL row per split).
  - outputs/tables/claims_cleaning_log.csv: rows remaining after each step.
"""

import pandas as pd

from utils.config import get_path, load_config, require, require_date
from utils.io import write_parquet_atomic
from utils.log import get_logger

logger = get_logger("04_clean_claims")

TARGET = "target"
SPLIT = "split"
ID = "id"
SPLIT_ORDER = ["train", "validation", "test"]


def meta_columns(cfg) -> list[str]:
    """Non-feature columns kept in the output (the leakage test allows these too)."""
    return [ID, require(cfg, "fields.claims_date_of_loss"),
            require(cfg, "fields.claims_county"), SPLIT, TARGET]


def group_lookup(groups: dict) -> dict:
    return {str(code): name for name, codes in groups.items() for code in codes}


def apply_group(series: pd.Series, groups: dict, label: str) -> pd.Series:
    lookup = group_lookup(groups)
    as_text = series.astype("string")
    out = as_text.map(lookup).astype("string")
    unknown = as_text.notna() & out.isna()
    if unknown.any():
        logger.warning("%s: %d rows with codes not in any group (set to 'other'): %s",
                       label, int(unknown.sum()), sorted(as_text[unknown].unique())[:20])
        out = out.mask(unknown, "other")
    return out


def years_between(later: pd.Series, earlier: pd.Series) -> pd.Series:
    return (later - earlier).dt.days / 365.25


def assign_split(dates: pd.Series, train_end: pd.Timestamp, validation_end: pd.Timestamp) -> pd.Series:
    """Inclusive boundaries: train <= train_end < validation <= validation_end < test."""
    split = pd.Series("test", index=dates.index, dtype="string")
    split = split.mask(dates <= validation_end, "validation")
    return split.mask(dates <= train_end, "train")


def clean(raw: pd.DataFrame, cfg, steps: list) -> pd.DataFrame:
    def step(name: str, frame: pd.DataFrame) -> pd.DataFrame:
        steps.append({"step": name, "rows": len(frame)})
        logger.info("%-55s %9s rows", name, f"{len(frame):,}")
        return frame

    paid = require(cfg, "fields.claims_paid")
    cov = require(cfg, "fields.claims_coverage")
    date = require(cfg, "fields.claims_date_of_loss")

    df = step("raw claims", raw)

    if require(cfg, "cleaning.drop_exact_duplicate_claims"):
        cols = [c for c in df.columns if c not in (ID, "asOfDate")]
        df = step("drop exact copies (all columns except id, asOfDate)",
                  df[~df.duplicated(cols, keep="first")])

    if require(cfg, "target.drop_zero_coverage"):
        df = step(f"drop $0 or missing {cov}", df[df[cov].fillna(0) > 0])

    if require(cfg, "target.drop_zero_paid"):
        n_zero = int((df[paid] == 0).sum())
        n_neg = int((df[paid] < 0).sum())
        logger.info("  payout == 0: %s rows; payout < 0: %s rows", f"{n_zero:,}", f"{n_neg:,}")
        df = step(f"drop {paid} <= 0", df[df[paid] > 0])

    if require(cfg, "target.definition") != "share_of_coverage":
        raise SystemExit("STOP: 04 only implements target.definition = share_of_coverage")
    cap = float(require(cfg, "target.cap_share_at"))
    share = df[paid].astype("float64") / df[cov].astype("float64")
    logger.info("  share of coverage > %.2f, capped: %s rows", cap, f"{int((share > cap).sum()):,}")
    df = df.assign(**{TARGET: share.clip(upper=cap)})

    # Derived features.
    lo, hi = require(cfg, "cleaning.building_age_years")
    age = years_between(df[date], df["originalConstructionDate"])
    bad_age = age.notna() & ~age.between(lo, hi)
    logger.info("  building age outside [%s, %s] years -> missing: %s rows", lo, hi, f"{int(bad_age.sum()):,}")
    tenure = years_between(df[date], df["originalNBDate"])
    bad_ten = tenure < require(cfg, "cleaning.min_policy_tenure_years")
    logger.info("  negative policy tenure -> missing: %s rows", f"{int(bad_ten.sum()):,}")
    df = df.assign(
        building_age_at_loss=age.mask(bad_age),
        policy_tenure_years=tenure.mask(bad_ten),
        flood_zone_group=apply_group(df["ratedFloodZone"], require(cfg, "flood_zone_groups"), "flood zone"),
        occupancy_group=apply_group(df["occupancyType"], require(cfg, "occupancy_groups"), "occupancy"),
        month_of_loss=df[date].dt.month.astype("Int64"),
    )

    # Feature table: only allowed features + derived, minus the raw sources they replace.
    derived = dict(require(cfg, "derived_features"))
    replaced = set(derived.values())
    allowed = [c for c in require(cfg, "allowed_features") if c not in replaced]
    missing = [c for c in allowed if c not in df.columns]
    if missing:
        raise SystemExit(f"STOP: allowed features missing from claims: {missing}")

    split = assign_split(df[date], require_date(cfg, "splits.train_end"),
                         require_date(cfg, "splits.validation_end"))
    df = df.assign(**{SPLIT: split})
    out = df[meta_columns(cfg) + allowed + list(derived)].reset_index(drop=True)
    step("final feature table", out)
    return out


def summary_by_split(df: pd.DataFrame) -> pd.DataFrame:
    """Per split: an ALL row, then one row per flood zone group."""
    rows = []
    for name in SPLIT_ORDER:
        part = df[df[SPLIT] == name]
        zones = part["flood_zone_group"].fillna("missing")
        groups = [("ALL", part)] + sorted(
            ((str(z), g) for z, g in part.groupby(zones)), key=lambda t: t[0])
        for zone, g in groups:
            rows.append({SPLIT: name, "flood_zone_group": zone, "claims": len(g),
                         "target_mean": round(g[TARGET].mean(), 4),
                         "target_median": round(g[TARGET].median(), 4)})
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    raw = pd.read_parquet(get_path(cfg, "raw") / "claims_fl.parquet")
    steps: list = []
    df = clean(raw, cfg, steps)

    date = require(cfg, "fields.claims_date_of_loss")
    for name in SPLIT_ORDER:
        part = df[df[SPLIT] == name]
        logger.info("split %-10s %8s rows  (%s to %s)  target mean %.3f", name, f"{len(part):,}",
                    f"{part[date].min():%Y-%m-%d}", f"{part[date].max():%Y-%m-%d}", part[TARGET].mean())

    out_path = get_path(cfg, "interim") / "claims_clean.parquet"
    write_parquet_atomic(df, out_path)
    logger.info("Saved %s (%d columns)", out_path, df.shape[1])

    tables = get_path(cfg, "tables")
    summary_by_split(df).to_csv(tables / "claims_summary_by_split.csv", index=False)
    pd.DataFrame(steps).to_csv(tables / "claims_cleaning_log.csv", index=False)
    logger.info("Saved claims_summary_by_split.csv and claims_cleaning_log.csv to %s", tables)


if __name__ == "__main__":
    main()
