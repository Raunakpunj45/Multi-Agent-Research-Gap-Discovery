"""Read-only diagnostic for OpenAlex / Crossref / Semantic Scholar retrieval.

Run from the project root:
    python scripts/diagnose_sources.py [--query "..."] [--max-results 5]

It calls the REAL retrievers (so request params and parsing match production)
with requests.get wrapped to record every HTTP attempt. It changes nothing,
needs no Gemini key, and never prints secrets.
"""
import argparse
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
from unittest import mock

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from src.retrieval.crossref import CrossrefRetriever
from src.retrieval.openalex import OpenAlexRetriever
from src.retrieval.relevance_filter import RelevanceFilter
from src.retrieval.semantic_scholar import SemanticScholarRetriever

DEFAULT_QUERY = "large language models scientific discovery"
WORK_TYPES_URL = "https://api.openalex.org/work-types"
RATE_LIMIT_URL = "https://api.openalex.org/rate-limit"
# The values OpenAlexRetriever currently sends in its type filter.
CURRENT_OPENALEX_TYPES = tuple(OpenAlexRetriever.PEER_REVIEWED_TYPES.split("|"))

_SECRET_NAME = re.compile(r"key|token|secret|password|authorization", re.I)
_URL_SECRET = re.compile(r"((?:api[_-]?key|key|token)=)[^&\s]+", re.I)


def scrub(text: object) -> str:
    """Mask any secret-looking env value and key=... URL parameter."""
    out = str(text)
    for name, value in os.environ.items():
        if _SECRET_NAME.search(name) and len(value) >= 8:
            out = out.replace(value, "***")
    return _URL_SECRET.sub(r"\1***", out)


def redact_params(params) -> list:
    # List of pairs, not a dict: Crossref repeats the "filter" key on purpose.
    pairs = params.items() if isinstance(params, dict) else (params or [])
    return [(k, "***" if _SECRET_NAME.search(str(k)) else v) for k, v in pairs]


def describe_error(exc: Optional[BaseException]) -> str:
    parts, seen = [], set()
    while exc is not None and id(exc) not in seen and len(parts) < 5:
        seen.add(id(exc))
        parts.append(scrub(f"{type(exc).__name__}: {exc}"))
        exc = exc.__cause__
    return " <- caused by ".join(parts)


@dataclass
class Attempt:
    url: str
    params: list
    status: Optional[int]
    seconds: float
    raw_count: Optional[int] = None
    detail: str = ""


@dataclass
class Report:
    label: str
    attempts: list = field(default_factory=list)
    parsed: Optional[int] = None
    filtered: Optional[int] = None
    pdf_parsed: Optional[int] = None
    pdf_filtered: Optional[int] = None
    error: str = ""


def _recording_get(real_get, attempts: list, raw_counter: Callable):
    def get(url, **kwargs):
        start = time.monotonic()
        params = redact_params(kwargs.get("params"))
        try:
            resp = real_get(url, **kwargs)
        except Exception as exc:
            attempts.append(
                Attempt(scrub(url), params, None, time.monotonic() - start,
                        None, scrub(f"{type(exc).__name__}: {exc}"))
            )
            raise
        raw, detail = None, ""
        if resp.status_code == 200:
            try:
                raw = raw_counter(resp.json())
            except Exception:
                detail = "HTTP 200 but body not in expected shape: " + resp.text[:200]
        else:
            detail = resp.text[:300]
        attempts.append(
            Attempt(scrub(resp.url or url), params, resp.status_code,
                    time.monotonic() - start, raw, scrub(detail))
        )
        return resp

    return get


def diagnose(label: str, search: Callable, raw_counter: Callable, query: str) -> Report:
    """Run one retriever search; never raises."""
    report = Report(label)
    wrapper = _recording_get(requests.get, report.attempts, raw_counter)
    try:
        with mock.patch.object(requests, "get", wrapper):
            papers = search()
        kept = RelevanceFilter().filter(papers, query)
    except Exception as exc:
        report.error = describe_error(exc)
        return report
    report.parsed = len(papers)
    report.filtered = len(kept)
    report.pdf_parsed = sum(1 for p in papers if p.pdf_url)
    report.pdf_filtered = sum(1 for p in kept if p.pdf_url)
    return report


def build_runs(query: str, max_results: int) -> list:
    openalex, crossref, semantic = (
        OpenAlexRetriever(), CrossrefRetriever(), SemanticScholarRetriever(),
    )
    oa_count = lambda j: len(j["results"])  # noqa: E731
    return [
        ("OpenAlex (current type filter)",
         lambda: openalex.search(query, max_results, only_journal_conference=True), oa_count),
        ("OpenAlex (no type filter)",
         lambda: openalex.search(query, max_results, only_journal_conference=False), oa_count),
        ("Crossref (current type filter)",
         lambda: crossref.search(query, max_results, only_journal_conference=True),
         lambda j: len(j["message"]["items"])),
        ("Semantic Scholar (current type filter)",
         lambda: semantic.search(query, max_results, only_journal_conference=True),
         lambda j: len(j["data"])),
    ]


def probe_openalex_extras() -> list:
    """Observations only: which type values OpenAlex lists, and key/budget status."""
    lines = ["=== OpenAlex extra probes ==="]
    try:
        r = requests.get(WORK_TYPES_URL, timeout=20)
        lines.append(f"GET {r.url} -> HTTP {r.status_code}")
        ids = []
        if r.status_code == 200:
            for item in r.json().get("results", []):
                ids.append(item.get("display_name") or str(item.get("id", "")).rsplit("/", 1)[-1])
        lines.append(f"  type values returned: {ids or '(none parsed)'}")
        lines += [f"  current filter value '{t}' listed: {t in ids}" for t in CURRENT_OPENALEX_TYPES]
        if r.status_code != 200 or not ids:
            lines.append(f"  body: {r.text[:300]}")
    except Exception as exc:
        lines.append(f"  work-types probe failed: {describe_error(exc)}")

    key = os.getenv("OPENALEX_API_KEY")
    if not key:
        lines.append("OPENALEX_API_KEY not set: rate-limit probe skipped.")
    else:
        try:
            r = requests.get(RATE_LIMIT_URL, params={"api_key": key}, timeout=20)
            lines.append(f"GET {RATE_LIMIT_URL}?api_key=*** -> HTTP {r.status_code}")
            lines.append(f"  body: {r.text[:300]}")
        except Exception as exc:
            lines.append(f"  rate-limit probe failed: {describe_error(exc)}")
    return [scrub(line) for line in lines]


def format_report(r: Report) -> str:
    lines = [f"=== {r.label} ==="]
    if r.attempts:
        lines.append(f"  url: {r.attempts[0].url}")
        lines.append(f"  params: {r.attempts[0].params}")
    for i, a in enumerate(r.attempts, 1):
        status = f"HTTP {a.status}" if a.status is not None else "no response"
        raw = "-" if a.raw_count is None else a.raw_count
        lines.append(f"  attempt {i}: {status} in {a.seconds:.1f}s, raw results: {raw}")
        if a.detail:
            lines.append(f"    detail: {a.detail}")
    if r.parsed is not None:
        lines.append(
            f"  parsed: {r.parsed} | after RelevanceFilter: {r.filtered} | "
            f"with pdf_url: {r.pdf_parsed} parsed, {r.pdf_filtered} after filter"
        )
    if r.error:
        lines.append(f"  ERROR: {r.error}")
    return scrub("\n".join(lines))


def summary(reports: list) -> str:
    rows = ["SUMMARY",
            f"{'source':40} {'last HTTP':>9} {'raw':>4} {'parsed':>6} {'filtered':>8} {'pdf':>4}  error"]
    for r in reports:
        last = r.attempts[-1] if r.attempts else None
        status = "-" if last is None or last.status is None else last.status
        raw = next((a.raw_count for a in reversed(r.attempts) if a.raw_count is not None), "-")
        cells = ["-" if v is None else v for v in (r.parsed, r.filtered, r.pdf_filtered)]
        rows.append(
            f"{r.label:40} {status!s:>9} {raw!s:>4} {cells[0]!s:>6} {cells[1]!s:>8} "
            f"{cells[2]!s:>4}  {'yes' if r.error else 'no'}"
        )
    return scrub("\n".join(rows))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--max-results", type=int, default=5)
    args = parser.parse_args(argv)

    print(f"Query: {args.query!r} | max_results per source: {args.max_results}")
    print(f"OPENALEX_API_KEY set: {bool(os.getenv('OPENALEX_API_KEY'))} | "
          f"SEMANTIC_SCHOLAR_API_KEY set: {bool(os.getenv('SEMANTIC_SCHOLAR_API_KEY'))}\n")

    reports = []
    for label, search, raw_counter in build_runs(args.query, args.max_results):
        reports.append(diagnose(label, search, raw_counter, args.query))
        print(format_report(reports[-1]) + "\n")

    print("\n".join(probe_openalex_extras()) + "\n")
    print(summary(reports))


if __name__ == "__main__":
    main()
