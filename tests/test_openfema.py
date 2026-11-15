"""Offline tests for utils.openfema (no network: the HTTP session is faked)."""

import pandas as pd
import pytest
import requests

from utils import openfema


class FakeResponse:
    def __init__(self, payload, content_type="application/json; charset=utf-8"):
        self._payload = payload
        self.headers = {"Content-Type": content_type}
        self.url = "https://example.test"

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class FakeSession:
    """Serves `records` honoring $top/$skip; can fail the first N calls."""

    def __init__(self, records=(), dataset="NfipClaims", fail_first=0, exc=None):
        self.records = list(records)
        self.dataset = dataset
        self.fail_first = fail_first
        self.exc = exc or requests.exceptions.ChunkedEncodingError("connection reset")
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(dict(params or {}))
        if len(self.calls) <= self.fail_first:
            raise self.exc
        skip, top = params.get("$skip", 0), params.get("$top", len(self.records))
        return FakeResponse({self.dataset: self.records[skip: skip + top]})


@pytest.fixture
def no_sleep(monkeypatch):
    waits = []
    monkeypatch.setattr(openfema.time, "sleep", waits.append)
    return waits


def test_iter_pages_covers_every_row_once():
    records = [{"id": i} for i in range(25)]
    session = FakeSession(records)
    pages = list(openfema.iter_pages(session, "u", "NfipClaims", "f", total=25, page_size=10))

    assert [len(p) for p in pages] == [10, 10, 5]
    assert [r["id"] for p in pages for r in p] == list(range(25))
    assert [c["$skip"] for c in session.calls] == [0, 10, 20]
    assert all(c["$orderby"] == "id" and c["$filter"] == "f" for c in session.calls)


def test_iter_pages_stops_early_on_empty_page():
    session = FakeSession([{"id": 1}])  # API reports more rows than it returns
    pages = list(openfema.iter_pages(session, "u", "NfipClaims", "f", total=30, page_size=10))
    assert pages == [[{"id": 1}]]
    assert len(session.calls) == 2


def test_get_json_retries_after_dropped_connection(no_sleep):
    session = FakeSession([{"id": 1}], fail_first=2)
    data = openfema._get_json(session, "u", {"$top": 1, "$skip": 0})
    assert data == {"NfipClaims": [{"id": 1}]}
    assert len(session.calls) == 3
    assert len(no_sleep) == 2


def test_get_json_gives_up_after_max_attempts(no_sleep):
    session = FakeSession([{"id": 1}], fail_first=99)
    with pytest.raises(requests.exceptions.ChunkedEncodingError):
        openfema._get_json(session, "u", {}, attempts=3)
    assert len(session.calls) == 3


def test_get_json_rejects_non_json(no_sleep):
    class HtmlSession(FakeSession):
        def get(self, url, params=None, timeout=None):
            return FakeResponse("<html>Page not found</html>", content_type="text/html")

    with pytest.raises(RuntimeError, match="Expected JSON"):
        openfema._get_json(HtmlSession(), "u", {})


def test_cast_by_dictionary_uses_declared_types():
    fields = pd.DataFrame({
        "name": ["state", "elevated", "floors", "paid", "dateOfLoss", "built"],
        "type": ["text", "boolean", "smallint", "decimal(12,2)", "datetime", "date"],
    })
    raw = pd.DataFrame({
        "state": ["FL", None],
        "elevated": [True, None],
        "floors": [2, None],
        "paid": ["1234.50", None],
        "dateOfLoss": ["2022-09-28T00:00:00.000Z", None],
        "built": ["1960-01-01", None],
        "notInDictionary": ["x", "y"],
    })
    out = openfema.cast_by_dictionary(raw, fields)

    assert list(out.columns) == list(raw.columns)  # nothing dropped or renamed
    assert out["state"].dtype == "string"
    assert out["elevated"].dtype == "boolean" and out["elevated"].isna()[1]
    assert out["floors"].dtype == "Int64" and out["floors"][0] == 2 and out["floors"].isna()[1]
    assert out["paid"].dtype == "float64" and out["paid"][0] == 1234.5
    assert str(out["dateOfLoss"].dtype).startswith("datetime64") and out["dateOfLoss"].dt.tz is not None
    assert out["dateOfLoss"][0] == pd.Timestamp("2022-09-28", tz="UTC")
    assert out["built"][0] == pd.Timestamp("1960-01-01", tz="UTC")
    assert out["notInDictionary"].tolist() == ["x", "y"]  # untouched


def _load_script_01():
    import importlib.util
    from utils.config import REPO_ROOT

    spec = importlib.util.spec_from_file_location(
        "download_claims", REPO_ROOT / "src" / "01_download_claims.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("pages, reported", [
    ([[{"id": 1}, {"id": 2}]], 3),            # short download
    ([[{"id": 1}], [{"id": 1}]], 2),          # duplicate ids
])
def test_download_stops_instead_of_saving_bad_data(monkeypatch, pages, reported):
    script = _load_script_01()
    monkeypatch.setattr(script.openfema, "fetch_count", lambda *a, **k: reported)
    monkeypatch.setattr(script.openfema, "iter_pages", lambda *a, **k: iter(pages))
    fields = pd.DataFrame({"name": ["id"], "type": ["bigint"]})
    cfg = {"sources": {"openfema_base": "b", "claims_dataset": "NfipClaims", "claims_version": 3},
           "scope": {"state_abbr": "FL"}, "years": {"claims_start": 2010, "claims_end": 2024}}

    with pytest.raises(SystemExit, match="STOP"):
        script.download(None, cfg, fields)
