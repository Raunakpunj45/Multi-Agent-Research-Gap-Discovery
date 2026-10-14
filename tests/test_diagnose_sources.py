import importlib.util
import json
from pathlib import Path

import pytest
import requests

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_sources.py"
spec = importlib.util.spec_from_file_location("diagnose_sources", SCRIPT)
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)

QUERY = "large language models scientific discovery"
SECRET = "supersecretvalue123"


def response(status, payload=None, text=None, url="https://example.org/x"):
    r = requests.Response()
    r.status_code = status
    r.reason = "mock"
    r.url = url
    r._content = (text if text is not None else json.dumps(payload)).encode()
    return r


def run(label_prefix):
    runs = diag.build_runs(QUERY, 5)
    label, search, counter = next(x for x in runs if x[0].startswith(label_prefix))
    return diag.diagnose(label, search, counter, QUERY)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)


def test_openalex_counts_and_pdf(monkeypatch):
    payload = {"results": [
        {"id": "W1", "display_name": "Large language models for scientific discovery",
         "publication_year": 2024, "doi": "https://doi.org/10.1/a", "authorships": [],
         "abstract_inverted_index": {"Large": [0], "models": [1]},
         "best_oa_location": {"pdf_url": "https://x/a.pdf"}},
        {"id": "W2", "display_name": "Unrelated zebra paper", "publication_year": 2020,
         "doi": None, "authorships": [], "abstract_inverted_index": None,
         "best_oa_location": None},
    ]}
    monkeypatch.setattr(requests, "get", lambda url, **kw: response(200, payload))
    r = run("OpenAlex (current")
    assert r.attempts[0].status == 200 and r.attempts[0].raw_count == 2
    assert (r.parsed, r.filtered, r.pdf_parsed, r.pdf_filtered) == (2, 1, 1, 1)
    assert r.error == ""


def test_http_error_keeps_status_body_and_cause(monkeypatch):
    monkeypatch.setattr(
        requests, "get",
        lambda url, **kw: response(400, text='{"error":"Invalid type value"}'),
    )
    r = run("OpenAlex (current")
    assert [a.status for a in r.attempts] == [400, 400, 400]
    assert "Invalid type value" in r.attempts[0].detail
    assert "OpenAlex request failed" in r.error and "400" in r.error
    assert r.parsed is None


def test_secrets_never_printed(monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", SECRET)
    leaky = f"https://api.openalex.org/works?api_key={SECRET}"
    monkeypatch.setattr(
        requests, "get",
        lambda url, **kw: response(403, text=f"bad key {SECRET}", url=leaky),
    )
    r = run("OpenAlex (current")
    assert SECRET not in diag.format_report(r)
    assert SECRET not in diag.summary([r])
    assert diag.redact_params([("api_key", SECRET), ("q", "x")]) == [("api_key", "***"), ("q", "x")]


def test_one_source_failing_does_not_stop_others(monkeypatch):
    def fake_get(url, **kw):
        if "openalex" in url:
            raise requests.ConnectionError("boom")
        if "crossref" in url:
            return response(200, {"message": {"items": [
                {"title": ["Large language models"], "DOI": "10.1/x", "author": []}]}})
        return response(500, text="down")

    monkeypatch.setattr(requests, "get", fake_get)
    openalex, crossref = run("OpenAlex (current"), run("Crossref")
    assert "ConnectionError" in openalex.error
    assert crossref.parsed == 1 and crossref.attempts[0].raw_count == 1
    # Crossref now sends one single-filter request per publication type.
    assert [[v for k, v in a.params if k == "filter"] for a in crossref.attempts] == [
        ["type:journal-article"], ["type:proceedings-article"]]
    assert "SUMMARY" in diag.summary([openalex, crossref])
