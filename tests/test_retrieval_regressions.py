"""Regression tests for the Stage 1 retrieval fixes. Mocked HTTP only, no live calls."""
import json

import requests

from src.retrieval.crossref import CrossrefRetriever
from src.retrieval.deduplicator import PaperDeduplicator
from src.retrieval.models import PaperRecord
from src.retrieval.openalex import OpenAlexRetriever


def response(status, payload=None, text=None):
    r = requests.Response()
    r.status_code = status
    r.reason = "mock"
    r.url = "https://example.org/mock"
    r._content = (text if text is not None else json.dumps(payload)).encode()
    return r


def filter_values(params):
    pairs = params.items() if isinstance(params, dict) else params
    return [v for k, v in pairs if k == "filter"]


def paper(**kw):
    base = dict(title="Some Title", authors=[], year=None, doi=None, abstract=None,
                source="x", source_id=None, landing_url=None, pdf_url=None)
    base.update(kw)
    return PaperRecord(**base)


# ---------------- OpenAlex: legacy type values matched nothing ----------------

def test_openalex_filter_uses_valid_openalex_types(monkeypatch):
    seen = []

    def fake_get(url, **kw):
        seen.append(kw["params"])
        return response(200, {"results": [
            {"id": "W1", "display_name": "A paper", "authorships": []}]})

    monkeypatch.setattr(requests, "get", fake_get)
    papers = OpenAlexRetriever().search("llm", 5, only_journal_conference=True)

    assert len(papers) == 1
    assert seen[0]["filter"] == "type:article|conference-paper"
    assert "journal-article" not in seen[0]["filter"]
    assert "proceedings-article" not in seen[0]["filter"]


def test_openalex_no_type_filter_when_not_peer_reviewed_only(monkeypatch):
    seen = []
    monkeypatch.setattr(
        requests, "get",
        lambda url, **kw: seen.append(kw["params"]) or response(200, {"results": []}))
    OpenAlexRetriever().search("llm", 5, only_journal_conference=False)
    assert "filter" not in seen[0]


# ---------------- Crossref: repeated filter params -> HTTP 400 ----------------

def crossref_api(journal_items, proceedings_items, calls):
    """Fake Crossref that rejects repeated 'filter' params like the real API."""
    def fake_get(url, **kw):
        params = kw["params"]
        calls.append(params)
        filters = filter_values(params)
        if len(filters) > 1:
            return response(400, text='{"message":{"type":"pair-list-form-invalid"}}')
        if filters == ["type:journal-article"]:
            return response(200, {"message": {"items": journal_items}})
        if filters == ["type:proceedings-article"]:
            return response(200, {"message": {"items": proceedings_items}})
        return response(200, {"message": {"items": journal_items + proceedings_items}})
    return fake_get


def item(title, doi):
    return {"title": [title], "DOI": doi, "author": []}


def test_crossref_sends_one_filter_per_request_and_covers_both_types(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = []
    monkeypatch.setattr(requests, "get", crossref_api(
        [item("J1", "10.1/j1")], [item("P1", "10.1/p1")], calls))

    papers = CrossrefRetriever().search("llm", 5, only_journal_conference=True)

    assert sorted(filter_values(c)[0] for c in calls) == [
        "type:journal-article", "type:proceedings-article"]
    assert all(len(filter_values(c)) == 1 for c in calls)
    assert {p.title for p in papers} == {"J1", "P1"}


def test_crossref_round_robin_cap_dedup_and_decision_letters(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    journal = [item("J1", "10.1/j1"), item("Decision Letter for J", "10.1/dl"),
               item("J2", "10.1/j2"), item("J3", "10.1/j3")]
    proceedings = [item("P1", "10.1/P1"), item("P1 dup", "10.1/p1"),
                   item("P2", "10.1/p2"), item("P3", "10.1/p3")]
    monkeypatch.setattr(requests, "get", crossref_api(journal, proceedings, []))

    papers = CrossrefRetriever().search("llm", 4, only_journal_conference=True)

    assert [p.title for p in papers] == ["J1", "P1", "J2", "P2"]


def test_crossref_without_type_filter_makes_single_unfiltered_request(monkeypatch):
    calls = []
    monkeypatch.setattr(requests, "get", crossref_api([item("J1", "10.1/j1")], [], calls))
    CrossrefRetriever().search("llm", 5, only_journal_conference=False)
    assert len(calls) == 1 and filter_values(calls[0]) == []


def test_crossref_failure_of_one_type_raises_instead_of_narrowing(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)

    def fake_get(url, **kw):
        if filter_values(kw["params"]) == ["type:proceedings-article"]:
            return response(500, text="down")
        return response(200, {"message": {"items": [item("J1", "10.1/j1")]}})

    monkeypatch.setattr(requests, "get", fake_get)
    try:
        CrossrefRetriever().search("llm", 5, only_journal_conference=True)
    except RuntimeError as error:
        assert "Crossref request failed" in str(error)
    else:
        raise AssertionError("expected RuntimeError, got silently narrowed results")


# ---------------- Deduplicator: metadata/PDF URL was dropped ----------------

def test_dedup_keeps_pdf_url_from_later_source_matched_by_doi():
    openalex = paper(source="openalex", doi="https://doi.org/10.1/ABC", pdf_url=None)
    scholar = paper(source="semantic_scholar", doi="10.1/abc",
                    pdf_url="https://x/a.pdf", abstract="An abstract")

    result = PaperDeduplicator().deduplicate([openalex, scholar])

    assert len(result) == 1
    assert result[0].source == "openalex"  # first record stays the primary one
    assert result[0].pdf_url == "https://x/a.pdf"
    assert result[0].abstract == "An abstract"
    assert openalex.pdf_url is None  # caller's record is not mutated


def test_dedup_title_fallback_merges_when_no_doi():
    crossref = paper(source="crossref", title="LLMs: for Science!", doi=None)
    scholar = paper(source="semantic_scholar", title="llms for science",
                    doi="10.1/x", pdf_url="https://x/b.pdf")

    result = PaperDeduplicator().deduplicate([crossref, scholar])

    assert len(result) == 1
    assert result[0].pdf_url == "https://x/b.pdf"
    assert result[0].doi == "10.1/x"


def test_dedup_gained_doi_still_matches_later_duplicates():
    a = paper(title="Shared Title", doi=None)
    b = paper(title="Shared Title", doi="10.1/z")
    c = paper(title="Different spelling entirely", doi="https://doi.org/10.1/Z",
              pdf_url="https://x/c.pdf")

    result = PaperDeduplicator().deduplicate([a, b, c])

    assert len(result) == 1 and result[0].pdf_url == "https://x/c.pdf"


def test_dedup_does_not_overwrite_and_keeps_distinct_papers():
    first = paper(title="One", doi="10.1/1", pdf_url="https://x/first.pdf")
    dup = paper(title="One", doi="10.1/1", pdf_url="https://x/other.pdf")
    other = paper(title="Two", doi="10.1/2")

    result = PaperDeduplicator().deduplicate([first, dup, other])

    assert [p.title for p in result] == ["One", "Two"]
    assert result[0].pdf_url == "https://x/first.pdf"
