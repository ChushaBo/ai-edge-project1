"""Download Florida NFIP policies from OpenFEMA (NFIP Redacted Policies v3).

Scope (decided 2026-11-15, see docs/decisions.md): policy terms with
policyEffectiveDate in years.policies_year. One row per annual term, so each policy is
counted once with one annual premium.

Method: the API can't page through this 75M-row dataset (503 timeouts after 60 s on
$skip paging; day-by-day queries ran about 20 s a day, 3+ hours a year). So the script
downloads FEMA's bulk parquet file once (3.7 GB, cached in data/raw/_bulk/) and filters
it locally, streaming row groups. The API's $count for the same filter is the check:
the script stops without saving if the two disagree.

Inputs:
  - OpenFEMA bulk parquet (network, once): sources.policies_bulk_url.
  - OpenFEMA API (network): $count for propertyState = scope.state_abbr and
    policyEffectiveDate in years.policies_year; DataSets and OpenFemaDataSetFields
    entries for the dictionary.
  - config.yaml: sources.*, scope.state_abbr, years.policies_year, paths.raw,
    paths.tables, paths.data_dictionaries
Outputs:
  - data/raw/_bulk/NfipPoliciesV3.parquet: the full national file (cache, gitignored).
  - data/raw/policies_fl.parquet: the selected columns, nothing renamed; types cast per
    the data dictionary.
  - docs/data_dictionaries/nfip_policies_v3_fields.csv and nfip_policies_v3_dataset.json
  - outputs/tables/policies_columns.csv: column, dtype, dictionary type, non-null
    count, percent missing.
  - outputs/tables/policies_premium_fields.csv: each premium and fee field with its
    dictionary description, non-null count, total, mean and median.
Flags:
  - --force: re-download data and dictionary even if they exist.
Idempotent: skips the download if policies_fl.parquet exists; the tables are rebuilt
from the saved file on every run.
"""

import argparse

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from utils import openfema
from utils.config import get_path, load_config, require
from utils.download import download_file
from utils.io import column_summary, write_parquet_atomic
from utils.log import get_logger, log_rows

logger = get_logger("02_download_policies")

PREMIUM_FIELDS = [
    "totalInsurancePremiumOfThePolicy", "policyCost", "fullRiskPremium",
    "federalPolicyFee", "hfiaaSurcharge", "reserveFundAssessment", "iccPremium",
    "communityProbationSurcharge",
]
# Components the dictionary lists for policyCost; checked against the data in main().
POLICY_COST_PARTS = [
    "totalInsurancePremiumOfThePolicy", "reserveFundAssessment", "federalPolicyFee",
    "hfiaaSurcharge",
]


def year_filter(state: str, year: int) -> str:
    return (f"propertyState eq '{state}' and policyEffectiveDate ge '{year}-01-01' "
            f"and policyEffectiveDate lt '{year + 1}-01-01'")


def _bound(field_type: pa.DataType, value: str) -> pa.Scalar:
    """A date bound typed to match the bulk file's column (timestamp, date or string)."""
    if pa.types.is_timestamp(field_type):
        ts = pd.Timestamp(value)
        ts = ts.tz_localize(field_type.tz) if field_type.tz else ts
        return pa.scalar(ts, type=field_type)
    if pa.types.is_date(field_type):
        return pa.scalar(pd.Timestamp(value).date(), type=field_type)
    return pa.scalar(value, type=field_type)


def filter_bulk(path, columns: list[str], state: str, year: int) -> pd.DataFrame:
    """Read only the wanted columns and Florida terms starting in `year`, streaming row
    groups so the 75M-row file never has to fit in memory."""
    try:
        dataset = ds.dataset(path, format="parquet")
    except Exception as exc:  # truncated or corrupt download
        raise SystemExit(f"STOP: {path} is not a readable parquet file ({exc}). "
                         "Delete it and rerun.") from exc
    schema = dataset.schema
    missing = sorted(set(columns) - set(schema.names))
    if missing:
        raise SystemExit(f"STOP: bulk file lacks requested columns {missing}")
    eff_type = schema.field("policyEffectiveDate").type
    logger.info("Bulk file: %s columns; policyEffectiveDate type %s",
                len(schema.names), eff_type)
    expr = ((ds.field("propertyState") == state)
            & (ds.field("policyEffectiveDate") >= _bound(eff_type, f"{year}-01-01"))
            & (ds.field("policyEffectiveDate") < _bound(eff_type, f"{year + 1}-01-01")))
    return dataset.to_table(columns=columns, filter=expr).to_pandas()


def download(session, cfg, fields: pd.DataFrame, bulk_path, force: bool) -> pd.DataFrame:
    base = require(cfg, "sources.openfema_base")
    dataset = require(cfg, "sources.policies_dataset")
    version = require(cfg, "sources.policies_version")
    columns = list(require(cfg, "sources.policies_columns"))
    state = require(cfg, "scope.state_abbr")
    year = int(require(cfg, "years.policies_year"))
    url = openfema.dataset_url(base, dataset, version)

    # The API's count is the reference the local filter must reproduce.
    total = openfema.fetch_count(session, url, year_filter(state, year))
    logger.info("%s: %s rows match [%s]", url, f"{total:,}", year_filter(state, year))

    bulk_url = require(cfg, "sources.policies_bulk_url")
    if download_file(session, bulk_url, bulk_path, force=force):
        logger.info("Downloaded %s (%.2f GB)", bulk_path, bulk_path.stat().st_size / 1e9)
    else:
        logger.info("Bulk file exists, not re-downloading: %s", bulk_path)

    df = filter_bulk(bulk_path, columns, state, year)
    log_rows(logger, df, f"filtered bulk file to {state}, effective {year}")
    df = openfema.cast_by_dictionary(df, fields)

    # Stop before saving: a saved file is trusted and skipped on every later run.
    if len(df) != total:
        raise SystemExit(f"STOP: bulk file gives {len(df):,} rows but the API reports {total:,}. "
                         "The bulk file and API may be on different refreshes. Nothing was saved.")
    dupes = int(df["id"].duplicated().sum())
    if dupes:
        raise SystemExit(f"STOP: {dupes:,} duplicate id values. Nothing was saved; "
                         "rerun with --force.")
    missing_cols = sorted(set(columns) - set(df.columns))
    if missing_cols:
        raise SystemExit(f"STOP: requested columns missing from the API response: {missing_cols}")
    return df[columns]


def premium_table(df: pd.DataFrame, fields: pd.DataFrame) -> pd.DataFrame:
    desc = dict(zip(fields["name"], fields["description"]))
    rows = []
    for f in PREMIUM_FIELDS:
        s = df[f].astype("float64")
        rows.append({"field": f, "non_null": int(s.notna().sum()), "total": s.sum(),
                     "mean": round(s.mean(), 2), "median": s.median(),
                     "description": " ".join(str(desc.get(f, "")).split())})
    return pd.DataFrame(rows)


def check_policy_cost(df: pd.DataFrame) -> None:
    parts = df[POLICY_COST_PARTS].astype("float64").fillna(0).sum(axis=1)
    cost = df["policyCost"].astype("float64")
    known = cost.notna()
    diff = cost[known] - parts[known]
    logger.info("policyCost == premium + reserve fund + federal fee + HFIAA surcharge in "
                "%.1f%% of rows; difference otherwise: median %s, mean %.2f",
                100 * (diff == 0).mean(), diff.median(), diff.mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download even if output exists")
    args = parser.parse_args()

    cfg = load_config()
    raw_dir = get_path(cfg, "raw")
    out_path = raw_dir / "policies_fl.parquet"
    session = openfema.make_session()

    fields = openfema.save_dictionary(
        session, require(cfg, "sources.openfema_base"), require(cfg, "sources.policies_dataset"),
        require(cfg, "sources.policies_version"), get_path(cfg, "data_dictionaries"),
        prefix="nfip_policies", force=args.force)
    wanted = list(require(cfg, "sources.policies_columns")) + PREMIUM_FIELDS
    unknown = sorted(set(wanted) - set(fields["name"]))
    if unknown:
        raise SystemExit(f"STOP: field(s) not in the policies data dictionary: {unknown}. "
                         "Check names before changing this script or config.")

    if out_path.exists() and not args.force:
        logger.info("Output exists, skipping download (use --force to refresh): %s", out_path)
        df = pd.read_parquet(out_path)
    else:
        bulk_path = raw_dir / "_bulk" / "NfipPoliciesV3.parquet"
        df = download(session, cfg, fields, bulk_path, args.force)
        write_parquet_atomic(df, out_path)
        logger.info("Saved %s", out_path)

    log_rows(logger, df, "policies_fl.parquet total")
    by_month = df["policyEffectiveDate"].dt.strftime("%Y-%m").value_counts().sort_index()
    for month, n in by_month.items():
        logger.info("  effective %s: %s rows", month, f"{n:,}")
    county = df["censusGeoid"].str[:5]
    logger.info("Counties (censusGeoid[:5]): %d distinct; %s rows with no censusGeoid",
                county.nunique(), f"{county.isna().sum():,}")
    logger.info("policyCount total (insured units): %s", f"{df['policyCount'].sum():,}")

    tables = get_path(cfg, "tables")
    summary = column_summary(df)
    summary.insert(2, "dictionary_type",
                   summary["column"].map(dict(zip(fields["name"], fields["type"]))))
    summary.to_csv(tables / "policies_columns.csv", index=False)
    premium_table(df, fields).to_csv(tables / "policies_premium_fields.csv", index=False)
    check_policy_cost(df)
    logger.info("Saved policies_columns.csv and policies_premium_fields.csv to %s", tables)


if __name__ == "__main__":
    main()
