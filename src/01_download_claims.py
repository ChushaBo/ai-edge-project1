"""Download Florida NFIP claims from OpenFEMA (NFIP Redacted Claims v3).

The task asked for "FIMA NFIP Redacted Claims v2". As of 2026-11-15 that endpoint
returns 404 and the OpenFEMA catalog lists the dataset as NfipClaims v3. Dataset name
and version come from config.yaml (sources.*).

Inputs:
  - OpenFEMA API (network): {sources.openfema_base}/v{claims_version}/{claims_dataset},
    filtered server-side to state = scope.state_abbr and yearOfLoss in
    years.claims_start..years.claims_end, paged with $top/$skip, ordered by id.
  - OpenFEMA metadata API (network): DataSets (catalog entry) and
    OpenFemaDataSetFields (field dictionary) for the same dataset version.
  - config.yaml: sources.*, scope.state_abbr, years.claims_start, years.claims_end,
    paths.raw, paths.tables, paths.data_dictionaries
Outputs:
  - data/raw/claims_fl.parquet: every column the API returns, nothing dropped or
    renamed; types cast per the data dictionary (text, boolean, Int64, float, UTC
    datetime).
  - docs/data_dictionaries/nfip_claims_v3_fields.csv: field dictionary.
  - docs/data_dictionaries/nfip_claims_v3_dataset.json: catalog entry (endpoint,
    refresh dates, dictionary page link).
  - outputs/tables/claims_columns.csv: column, dtype, dictionary type, non-null
    count, percent missing.
Flags:
  - --force: re-download the data and dictionary even if they exist.
Idempotent: skips the download if claims_fl.parquet exists; the column table is
rebuilt from the saved file on every run.
"""

import argparse
import json

import pandas as pd
from tqdm import tqdm

from utils import openfema
from utils.config import get_path, load_config, require
from utils.io import column_summary, write_parquet_atomic
from utils.log import get_logger, log_rows

logger = get_logger("01_download_claims")

# Fields this script relies on for filtering and paging. Stop if any are missing.
REQUIRED_FIELDS = ("id", "state", "yearOfLoss")


def load_or_fetch_dictionary(session, cfg, force: bool) -> pd.DataFrame:
    base = require(cfg, "sources.openfema_base")
    dataset = require(cfg, "sources.claims_dataset")
    version = require(cfg, "sources.claims_version")
    dict_dir = get_path(cfg, "data_dictionaries")
    fields_path = dict_dir / f"nfip_claims_v{version}_fields.csv"
    record_path = dict_dir / f"nfip_claims_v{version}_dataset.json"

    if fields_path.exists() and record_path.exists() and not force:
        logger.info("Data dictionary exists, not re-fetching: %s", fields_path)
        return pd.read_csv(fields_path)

    record = openfema.fetch_dataset_record(session, base, dataset, version)
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    fields = openfema.fetch_fields(session, base, dataset, version)
    keep = [c for c in ("sortOrder", "name", "title", "type", "isNullable", "isSearchable",
                        "primaryKey", "description", "lastRefresh") if c in fields]
    fields[keep].to_csv(fields_path, index=False)
    logger.info("Saved data dictionary (%d fields) to %s and %s",
                len(fields), fields_path, record_path)
    logger.info("Human-readable dictionary page: %s", record.get("dataDictionary"))
    return fields[keep]


def download(session, cfg, fields: pd.DataFrame) -> pd.DataFrame:
    base = require(cfg, "sources.openfema_base")
    dataset = require(cfg, "sources.claims_dataset")
    version = require(cfg, "sources.claims_version")
    state = require(cfg, "scope.state_abbr")
    start = int(require(cfg, "years.claims_start"))
    end = int(require(cfg, "years.claims_end"))

    url = openfema.dataset_url(base, dataset, version)
    filter_ = f"state eq '{state}' and yearOfLoss ge {start} and yearOfLoss le {end}"
    total = openfema.fetch_count(session, url, filter_)
    logger.info("%s: %s rows match [%s]", url, f"{total:,}", filter_)

    frames = []
    with tqdm(total=total, unit="rows", desc="claims") as bar:
        for records in openfema.iter_pages(session, url, dataset, filter_, total):
            frames.append(openfema.cast_by_dictionary(pd.DataFrame.from_records(records), fields))
            bar.update(len(records))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    log_rows(logger, df, "downloaded")

    if len(df) != total:
        logger.warning("Downloaded %d rows but the API reported %d. The dataset may have "
                       "refreshed mid-download; rerun with --force.", len(df), total)
    dupes = int(df["id"].duplicated().sum()) if "id" in df else 0
    if dupes:
        logger.warning("%d duplicate id values (paging overlap?). Not dropped.", dupes)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download even if output exists")
    args = parser.parse_args()

    cfg = load_config()
    out_path = get_path(cfg, "raw") / "claims_fl.parquet"
    session = openfema.make_session()

    fields = load_or_fetch_dictionary(session, cfg, args.force)
    missing = [f for f in REQUIRED_FIELDS if f not in set(fields["name"])]
    if missing:
        raise SystemExit(f"STOP: data dictionary lacks expected field(s) {missing}. "
                         "Check field names before changing this script.")

    if out_path.exists() and not args.force:
        logger.info("Output exists, skipping download (use --force to refresh): %s", out_path)
        df = pd.read_parquet(out_path)
    else:
        df = download(session, cfg, fields)
        write_parquet_atomic(df, out_path)
        logger.info("Saved %s", out_path)

    log_rows(logger, df, "claims_fl.parquet total")
    per_year = df["yearOfLoss"].value_counts().sort_index()
    for year, n in per_year.items():
        logger.info("  yearOfLoss %s: %s rows", year, f"{n:,}")
    start, end = int(cfg["years"]["claims_start"]), int(cfg["years"]["claims_end"])
    absent = sorted(set(range(start, end + 1)) - set(per_year.index.astype(int)))
    if absent:
        logger.warning("No rows for year(s) %s", absent)

    summary = column_summary(df)
    summary.insert(2, "dictionary_type", summary["column"].map(dict(zip(fields["name"], fields["type"]))))
    tables_path = get_path(cfg, "tables") / "claims_columns.csv"
    summary.to_csv(tables_path, index=False)
    logger.info("Saved column summary (%d columns) to %s", len(summary), tables_path)

    not_in_dict = sorted(set(df.columns) - set(fields["name"]))
    not_in_data = sorted(set(fields["name"]) - set(df.columns))
    if not_in_dict:
        logger.warning("Columns in data but not in dictionary: %s", not_in_dict)
    if not_in_data:
        logger.warning("Dictionary fields absent from data: %s", not_in_data)


if __name__ == "__main__":
    main()
