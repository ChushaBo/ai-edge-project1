"""Small OpenFEMA API client shared by the download scripts (01 claims, 02 policies).

OpenFEMA query conventions used here (https://www.fema.gov/about/openfema/api):
  - $filter  OData-style filter, e.g. "state eq 'FL' and yearOfLoss ge 2010"
  - $top     page size (max 10,000)
  - $skip    offset for paging
  - $orderby a stable sort key so pages don't overlap or skip rows
  - $count   true -> metadata.count holds the total matching rows
Field metadata (the machine-readable data dictionary) comes from the
OpenFemaDataSetFields dataset.
"""

import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from utils.log import get_logger

logger = get_logger("openfema")

MAX_PAGE_SIZE = 10_000
TIMEOUT_S = 120
USER_AGENT = "ai-edge-fl-flood-shortfall/0.1 (academic project; python-requests)"


def make_session(retries: int = 5, backoff: float = 2.0) -> requests.Session:
    """Session that retries connection errors, 429 and 5xx with exponential backoff."""
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers["User-Agent"] = USER_AGENT
    return session


_BODY_ERRORS = (
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    ValueError,  # truncated / invalid JSON
)


def _get_json(
    session: requests.Session, url: str, params: dict[str, Any], attempts: int = 6
) -> dict:
    """GET and parse JSON, retrying the whole request if the body read fails.

    The session's urllib3 Retry handles errors before the response starts (429/5xx,
    refused connections); this loop also covers resets while the body is streaming,
    which OpenFEMA does on large pages.
    """
    for attempt in range(1, attempts + 1):
        try:
            resp = session.get(url, params=params, timeout=TIMEOUT_S)
            resp.raise_for_status()
            if "json" not in resp.headers.get("Content-Type", ""):
                raise RuntimeError(
                    f"Expected JSON from {resp.url}, got {resp.headers.get('Content-Type')}"
                )
            return resp.json()
        except _BODY_ERRORS as exc:
            if attempt == attempts:
                raise
            wait = min(5 * 2 ** (attempt - 1), 120)
            logger.warning("Request failed (%s); retry %d/%d in %ds",
                           type(exc).__name__, attempt, attempts - 1, wait)
            time.sleep(wait)
    raise AssertionError("unreachable")


def dataset_url(base: str, dataset: str, version: int) -> str:
    return f"{base.rstrip('/')}/v{version}/{dataset}"


def fetch_count(session: requests.Session, url: str, filter_: str) -> int:
    """Total rows matching filter_."""
    data = _get_json(session, url, {"$filter": filter_, "$top": 1, "$count": "true"})
    return int(data["metadata"]["count"])


def iter_pages(
    session: requests.Session,
    url: str,
    dataset: str,
    filter_: str,
    total: int,
    order_by: str = "id",
    page_size: int = MAX_PAGE_SIZE,
    select: list[str] | None = None,
) -> Iterator[list[dict]]:
    """Yield lists of records, paging with $skip until `total` rows are covered."""
    for skip in range(0, total, page_size):
        params: dict[str, Any] = {
            "$filter": filter_,
            "$top": page_size,
            "$skip": skip,
            "$orderby": order_by,
        }
        if select:
            params["$select"] = ",".join(select)
        records = _get_json(session, url, params).get(dataset, [])
        if not records:
            break
        yield records


def iter_all(
    session: requests.Session,
    url: str,
    dataset: str,
    filter_: str,
    select: list[str] | None = None,
    page_size: int = MAX_PAGE_SIZE,
    order_by: str = "id",
) -> Iterator[list[dict]]:
    """Yield pages for a small filter without a count call: stop at the first short page.

    For slices (e.g. one day of policies) where a $count per slice would double the
    number of requests. Callers should still check the overall total.
    """
    skip = 0
    while True:
        params: dict[str, Any] = {
            "$filter": filter_, "$top": page_size, "$skip": skip, "$orderby": order_by,
        }
        if select:
            params["$select"] = ",".join(select)
        records = _get_json(session, url, params).get(dataset, [])
        if records:
            yield records
        if len(records) < page_size:
            return
        skip += page_size


def save_dictionary(
    session: requests.Session,
    base: str,
    dataset: str,
    version: int,
    out_dir: Path,
    prefix: str,
    force: bool = False,
) -> pd.DataFrame:
    """Save the field dictionary (<prefix>_v<version>_fields.csv) and catalog entry
    (<prefix>_v<version>_dataset.json) to out_dir, or load them if already saved."""
    fields_path = out_dir / f"{prefix}_v{version}_fields.csv"
    record_path = out_dir / f"{prefix}_v{version}_dataset.json"
    if fields_path.exists() and record_path.exists() and not force:
        logger.info("Data dictionary exists, not re-fetching: %s", fields_path)
        return pd.read_csv(fields_path)

    record = fetch_dataset_record(session, base, dataset, version)
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    fields = fetch_fields(session, base, dataset, version)
    keep = [c for c in ("sortOrder", "name", "title", "type", "isNullable", "isSearchable",
                        "primaryKey", "description", "lastRefresh") if c in fields]
    fields[keep].to_csv(fields_path, index=False)
    logger.info("Saved data dictionary (%d fields) to %s and %s", len(fields), fields_path, record_path)
    logger.info("Human-readable dictionary page: %s", record.get("dataDictionary"))
    return fields[keep]


def fetch_fields(session: requests.Session, base: str, dataset: str, version: int) -> pd.DataFrame:
    """Data dictionary for a dataset version: one row per field (name, type, description, ...)."""
    url = dataset_url(base, "OpenFemaDataSetFields", 1)
    filter_ = f"openFemaDataSet eq '{dataset}' and datasetVersion eq {version}"
    data = _get_json(session, url, {"$filter": filter_, "$top": 1000})
    fields = pd.DataFrame(data.get("OpenFemaDataSetFields", []))
    if fields.empty:
        raise RuntimeError(f"No field metadata found for {dataset} v{version}")
    return fields.sort_values("sortOrder" if "sortOrder" in fields else "name").reset_index(drop=True)


def fetch_dataset_record(session: requests.Session, base: str, dataset: str, version: int) -> dict:
    """Catalog entry (title, endpoint, refresh dates, dictionary link) for a dataset version."""
    url = dataset_url(base, "DataSets", 1)
    data = _get_json(session, url, {"$filter": f"name eq '{dataset}' and version eq {version}"})
    records = data.get("DataSets", [])
    if not records:
        raise RuntimeError(f"{dataset} v{version} not found in the OpenFEMA catalog")
    return records[0]


def cast_by_dictionary(df: pd.DataFrame, fields: pd.DataFrame) -> pd.DataFrame:
    """Cast columns to the types the data dictionary declares. Columns are not renamed or dropped.

    text -> string, boolean -> boolean, (small|big)int/integer -> Int64,
    decimal(...) -> float64, date/datetime -> UTC datetime. Unknown types are left as-is.
    """
    types = dict(zip(fields["name"], fields["type"].str.lower()))
    out = df.copy()
    for col in out.columns:
        t = types.get(col)
        if t is None:
            continue
        if t == "text":
            out[col] = out[col].astype("string")
        elif t == "boolean":
            out[col] = out[col].astype("boolean")
        elif t in ("smallint", "integer", "int", "bigint"):
            out[col] = pd.to_numeric(out[col]).astype("Int64")
        elif t.startswith("decimal") or t in ("float", "double", "numeric"):
            out[col] = pd.to_numeric(out[col]).astype("float64")
        elif t in ("date", "datetime"):
            out[col] = pd.to_datetime(out[col], utc=True)
    return out
