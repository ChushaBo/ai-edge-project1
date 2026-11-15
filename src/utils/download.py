"""Plain-file downloads (bulk CSVs, reference files) with retries and atomic writes.

API downloads (OpenFEMA) live in utils.openfema; this module is for fetching whole
files from a URL, e.g. NOAA Storm Events CSVs and the NWS zone-county file.
"""

import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from utils.log import get_logger

logger = get_logger("download")

TIMEOUT_S = 120
USER_AGENT = "ai-edge-fl-flood-shortfall/0.1 (academic project; python-requests)"

_BODY_ERRORS = (
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
)


def make_session(retries: int = 5, backoff: float = 2.0) -> requests.Session:
    """Session that retries connection errors, 429 and 5xx with exponential backoff."""
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
        respect_retry_after_header=True,
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers["User-Agent"] = USER_AGENT
    return session


def get_text(session: requests.Session, url: str, attempts: int = 5) -> str:
    """GET a small text resource (e.g. a directory listing), retrying body-read failures."""
    for attempt in range(1, attempts + 1):
        try:
            resp = session.get(url, timeout=TIMEOUT_S)
            resp.raise_for_status()
            return resp.text
        except _BODY_ERRORS as exc:
            if attempt == attempts:
                raise
            wait = min(5 * 2 ** (attempt - 1), 120)
            logger.warning("GET %s failed (%s); retry %d/%d in %ds",
                           url, type(exc).__name__, attempt, attempts - 1, wait)
            time.sleep(wait)
    raise AssertionError("unreachable")


def download_file(
    session: requests.Session, url: str, dest: Path, force: bool = False, attempts: int = 5
) -> bool:
    """Download url to dest. Returns True if downloaded, False if skipped (dest exists).

    Streams to <dest>.part and renames only after the full body arrives (and matches
    Content-Length when the server sends one), so a partial file is never left at dest.
    """
    if dest.exists() and not force:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, attempts + 1):
        try:
            with session.get(url, stream=True, timeout=TIMEOUT_S) as resp:
                resp.raise_for_status()
                expected = resp.headers.get("Content-Length")
                size = 0
                with open(part, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                        size += len(chunk)
            if expected is not None and "Content-Encoding" not in resp.headers \
                    and size != int(expected):
                raise requests.exceptions.ChunkedEncodingError(
                    f"got {size} bytes, expected {expected}")
            part.replace(dest)
            return True
        except _BODY_ERRORS as exc:
            if attempt == attempts:
                part.unlink(missing_ok=True)
                raise
            wait = min(5 * 2 ** (attempt - 1), 120)
            logger.warning("Download %s failed (%s); retry %d/%d in %ds",
                           url, type(exc).__name__, attempt, attempts - 1, wait)
            time.sleep(wait)
    raise AssertionError("unreachable")
