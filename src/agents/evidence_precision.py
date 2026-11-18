"""Deterministic precision rules for grounding analysis statements in quotes.

Derived from the 10-paper live run (ui-...-20261118T203959Z-667a10), where:

* GAP-003 was built only from "missing information" statements ("The paper does
  not specify a particular dataset or benchmark") grounded in an author/title
  header, a literature-table row and an unrelated sentence;
* GAP-001/002 rested on a compound statement ("agent coordination and
  evaluation remain challenging, automatic knowledge-graph construction can
  introduce ...") that was grounded in a sentence not containing that claim,
  while the phrase itself was a literature-table row about another work
  ("Singh et al. [14] ... Agent coordination and evaluation remain challenging").

These functions only classify text. They never change stored chunk text or
quotes; callers keep exact substrings and provenance.
"""
from __future__ import annotations

import re

from src.retrieval.relevance_filter import content_terms

# -- absence ("missing information") claims ----------------------------------

_ABSENCE_VERBS = (
    "specify|specified|report|reported|mention|mentioned|describe|described|name|named|"
    "provide|provided|include|included|introduce|introduced|use|used|employ|employed|"
    "detail|detailed|state|stated|identify|identified|present|presented|disclose|"
    "disclosed|discuss|discussed|release|released|define|defined|list|listed|"
    "reference|referenced|address|addressed|evaluate|evaluated|cover|covered"
)
_ABSENCE = re.compile(
    rf"\b(?:does|do|did|is|are|was|were|has|have)\s+not\s+(?:been\s+)?"
    rf"(?:explicitly\s+|clearly\s+|specifically\s+|directly\s+)?(?:{_ABSENCE_VERBS})\b"
    rf"|\b(?:doesn't|don't|didn't|isn't|aren't|wasn't|weren't)\s+"
    rf"(?:explicitly\s+|clearly\s+)?(?:{_ABSENCE_VERBS})\b"
    rf"|\bnot\s+(?:explicitly\s+|clearly\s+)?(?:specified|reported|mentioned|described|named|"
    rf"provided|included|introduced|detailed|stated|identified|disclosed|discussed|"
    rf"defined|given|listed)\b"
    rf"|\bno\s+(?:specific|explicit|particular|clear|named|defined|dedicated|separate|"
    rf"concrete|formal|external|public(?:ly\s+available)?|standard(?:ized)?)\b"
    rf"|^\s*no\s+[\w‐‑-]+(?:\s+[\w‐‑,-]+){{0,6}}\s+(?:is|are|was|were)\s+"
    rf"(?:explicitly\s+)?(?:specified|reported|mentioned|described|named|provided|given|"
    rf"introduced|used|employed|listed|discussed)\b"
    rf"|\bunspecified\b|\bnot\s+(?:been\s+)?(?:made\s+)?(?:publicly\s+)?available\b",
    re.IGNORECASE,
)
# An author explicitly stating an absence uses negation in the source text itself.
_SOURCE_NEGATION = re.compile(
    r"\b(?:no|not|none|never|without|lacks?|lacking|lack\s+of|absence|absent|unavailable|"
    r"missing|neither|nor)\b|n't\b",
    re.IGNORECASE,
)


def is_absence_claim(statement: str) -> bool:
    """True when the statement's content is that something is NOT mentioned/used."""
    return bool(_ABSENCE.search(statement or ""))


def quote_states_absence(quote: str) -> bool:
    """True when the source quote itself uses negation (an author-stated absence)."""
    return bool(_SOURCE_NEGATION.search(quote or ""))


# -- non-evidence text: author/title headers and bibliographic entries -------

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_HEADER_MARKER = re.compile(
    r"\b(?:ISSN|ISBN|Creative Commons|All Open Access|Open Access articles|"
    r"Conference abbreviation|Volume number|Corresponding author|Received:|Accepted:|"
    r"Published online)\b|©|<Conference",
    re.IGNORECASE,
)
# "Bagchi1," / "Abbineni∗," : a capitalized surname with an affiliation mark.
_MARKED_NAME = re.compile(r"\b[A-Z][a-z]+[∗†‡*¹²³]|\b[A-Z][a-z]+\d(?=\s*[,\s])")
_AFFILIATION = re.compile(r"\b(?:Department of|School of|Faculty of|Institute of|University of|"
                          r"Laboratory of)\b")
_BIBLIOGRAPHIC = re.compile(
    r"\barXiv(?:\s+preprint|:\s*\d)|\bProceedings of\b|\bIn Proc(?:eedings)?\b|"
    r"\bpp\.\s*\d|\bvol\.\s*\d|\bdoi:\s*10\.|https?://doi\.org/|^\s*References\b|"
    r"\bReferences\s+[A-Z][a-z]+\s",
)
_YEAR_DOT = re.compile(r"\b(?:19|20)\d{2}[a-z]?\.(?=\s|$)")


_KEYWORDS_LABEL = re.compile(r"\b(?:Keywords|Index Terms)\b\s*[:—-]", re.IGNORECASE)


def _is_keyword_list(text: str) -> bool:
    """A comma-separated list of capitalized terms (a paper's keyword line)."""
    items = [item.strip() for item in re.split(r"[,;]", text) if item.strip()]
    if len(items) < 4:
        return False
    capitalized = sum(1 for item in items if item[0].isupper() and len(item.split()) <= 5)
    return capitalized / len(items) >= 0.75


def non_evidence_reason(quote: str) -> str | None:
    """Why ``quote`` cannot be evidence (header, bibliography, keywords), else None."""
    text = quote or ""
    if _EMAIL.search(text) or _HEADER_MARKER.search(text):
        return "author/title header"
    marked = len(_MARKED_NAME.findall(text))
    if marked >= 3 or (marked >= 2 and _AFFILIATION.search(text)):
        return "author/title header"
    if _BIBLIOGRAPHIC.search(text) or len(_YEAR_DOT.findall(text)) >= 2:
        return "bibliographic entry"
    if _KEYWORDS_LABEL.search(text) or _is_keyword_list(text):
        return "keyword list"
    return None


# -- secondhand evidence ------------------------------------------------------

_CITATION_MARKER = re.compile(r"\[\s*\d+(?:\s*[,–—-]\s*\d+)*\s*\]")
_ET_AL = re.compile(r"\bet\s+al\b", re.IGNORECASE)
_AUTHOR_YEAR = re.compile(
    r"\(\s*[A-Z][A-Za-zÀ-ſ'-]+(?:\s+(?:et\s+al\.?|and|&)\s*[A-Za-zÀ-ſ'-]*)?"
    r",?\s+(?:19|20)\d{2}[a-z]?\s*\)")


def is_secondary(quote: str) -> bool:
    """True when the quote reports another work (citation marker, et al., author-year)."""
    text = quote or ""
    return bool(_CITATION_MARKER.search(text) or _ET_AL.search(text)
                or _AUTHOR_YEAR.search(text))


CONTEXT_LOOKBACK_CHARS = 250
_SENTENCE_END = re.compile(r"[.!?](?=\s)")


def is_secondary_in_context(quote: str, source_text: str | None) -> bool:
    """Secondary when the quote, or its own sentence / table row, cites another work.

    A quote can be the tail of a literature-table row ("Singh et al. [14] Agentic
    RAG ... Agent coordination and evaluation remain challenging") whose citation
    marker lies before the quote. The quote's context is the stored text from the
    previous sentence end (at most 250 characters back) to the end of the quote.
    """
    if is_secondary(quote):
        return True
    if not quote or not source_text:
        return False
    start = source_text.find(quote)
    if start < 0:
        return False
    window_start = max(0, start - CONTEXT_LOOKBACK_CHARS)
    before = source_text[window_start:start]
    ends = list(_SENTENCE_END.finditer(before))
    if ends:
        before = before[ends[-1].end():]
    return is_secondary(before + quote)


# -- claim containment ------------------------------------------------------

# Words that frame a statement but carry no claim of their own.
_FRAMING = {
    "paper", "study", "studie", "author", "work", "evidence", "indicate", "indicates",
    "note", "notes", "state", "states", "show", "shows", "report", "reports",
    "describe", "describes", "include", "includes", "also", "such", "well", "may",
    "can", "could", "would", "will", "various", "different", "current", "other",
    "one", "use", "used", "uses", "propose", "proposed", "present", "presented",
    "mention", "mentioned", "it", "they", "which", "while", "more", "most", "both",
    "each", "there", "those", "has", "have", "been", "being", "was", "were", "not",
    "no", "within", "through", "across", "than", "then", "how", "what", "further",
    "key", "main", "significant", "particular", "specific", "approach", "here",
}
_SOFT_BREAK = re.compile(r"(?<=[A-Za-z])-\s+(?=[a-z])")
STEM_LENGTH = 6
MIN_CLAIM_COVERAGE = 0.5
MIN_CLAIM_TERMS = 2


def _stems(text: str) -> set[str]:
    # Rejoin PDF line-break hyphenation ("evalu- ation") for matching only.
    terms = content_terms(_SOFT_BREAK.sub("", text or "")) - _FRAMING
    return {term[:STEM_LENGTH] for term in terms if not term.isdigit() or len(term) > 1}


def claim_coverage(statement: str, quote: str) -> float:
    """Share of the statement's substantive terms that occur in the quote."""
    claim = _stems(statement)
    if not claim:
        return 0.0
    return len(claim & _stems(quote)) / len(claim)


def contains_claim(statement: str, quote: str) -> bool:
    """The quote must contain the statement's substantive claim, not just its topic."""
    claim = _stems(statement)
    if not claim:
        return False
    matched = len(claim & _stems(quote))
    return matched >= min(MIN_CLAIM_TERMS, len(claim)) and matched / len(claim) >= MIN_CLAIM_COVERAGE


# -- compound statements ----------------------------------------------------

_WORDCHAR = r"[\w‐‑'-]"
_CLAUSE_START = re.compile(
    rf"^\s*(?:and\s+|but\s+|while\s+|whereas\s+)?{_WORDCHAR}+(?:\s+{_WORDCHAR}+){{0,4}}?\s+"
    r"(?:can|may|might|could|still|remains?|is|are|was|were|has|have|lacks?|limits?|"
    r"introduces?|requires?|suffers?|fails?|depends?|struggles?|cannot|needs?)\b",
    re.IGNORECASE,
)
_CONTRAST = re.compile(r",\s+(?=(?:but|while|whereas|although)\s)", re.IGNORECASE)
# A finite verb: the text before a split point must itself be a clause, so a
# noun list ("datasets, experimental data, benchmarks, or ...") is never split.
_HAS_VERB = re.compile(
    r"\b(?:can|may|might|could|still|remains?|is|are|was|were|has|have|had|lacks?|limits?|"
    r"introduces?|requires?|suffers?|fails?|depends?|struggles?|cannot|needs?|uses?|"
    r"relies|reports?|shows?|achieves?|indicates?|notes?|states?|focus(?:es)?|"
    r"\w+(?:ed|es))\b", re.IGNORECASE)
MIN_CLAUSE_WORDS = 4


def split_compound(statement: str) -> list[str]:
    """Split a statement into independently groundable clauses where practical.

    Splits at ", but/while/whereas/although" and at commas that begin a new
    clause with its own subject and verb ("..., automatic KG construction can
    introduce ..."). Pieces shorter than four words are kept with the previous
    piece, so lists of nouns ("a, b, and c") are never broken up.
    """
    text = " ".join((statement or "").split())
    pieces: list[str] = []
    for part in _CONTRAST.split(text):
        segments = part.split(", ")
        current = segments[0]
        for segment in segments[1:]:
            if (_CLAUSE_START.match(segment) and len(current.split()) >= MIN_CLAUSE_WORDS
                    and len(segment.split()) >= MIN_CLAUSE_WORDS
                    and _HAS_VERB.search(current)):
                pieces.append(current)
                current = segment
            else:
                current = f"{current}, {segment}"
        pieces.append(current)
    cleaned = []
    for piece in pieces:
        piece = re.sub(r"^(?:and|but|while|whereas|although)\s+", "", piece.strip(" ,;"),
                       flags=re.IGNORECASE)
        if piece and piece[0].islower():
            piece = piece[0].upper() + piece[1:]
        if piece:
            cleaned.append(piece)
    return cleaned or [text]
