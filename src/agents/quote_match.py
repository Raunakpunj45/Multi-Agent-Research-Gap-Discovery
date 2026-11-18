"""Locate a model-supplied quote in a stored chunk and return the chunk's own text.

Models often re-type quotes with typographic variants: non-breaking or en/em
dashes instead of ASCII hyphens, curly instead of straight quotes, or different
whitespace (including PDF line-break hyphenation such as "goal- setting").
Only those variants are treated as equivalent; every other character must match
exactly and in order. On a match the EXACT substring of the stored chunk is
returned, so a saved quote is always verbatim chunk text.
"""
from __future__ import annotations

import re

_DASHES = "-‐‑‒–—―−­"
_SINGLE_QUOTES = "'‘’‚‛′"
_DOUBLE_QUOTES = '"“”„‟″'

_DASH_CLASS = "[" + re.escape(_DASHES) + "]"
_SINGLE_CLASS = "[" + re.escape(_SINGLE_QUOTES) + "]"
_DOUBLE_CLASS = "[" + re.escape(_DOUBLE_QUOTES) + "]"


# A PDF line-break hyphenation inside a word in the STORED text, e.g. "chal- lenge"
# or a soft hyphen: allowed between two letters of the quote in the fallback pass.
_LINE_BREAK = r"(?:-\s+|­\s*)?"


def _pattern(quote: str, line_breaks: bool = False) -> str:
    parts: list[str] = []
    index = 0
    while index < len(quote):
        char = quote[index]
        if (line_breaks and char.isalpha() and index + 1 < len(quote)
                and quote[index + 1].isalpha()):
            parts.append(re.escape(char) + _LINE_BREAK)
            index += 1
            continue
        if char.isspace():
            while index < len(quote) and quote[index].isspace():
                index += 1
            # A whitespace run may also sit next to a dash (handled below).
            if parts and parts[-1].endswith(_DASH_CLASS + r"\s*"):
                continue
            parts.append(r"\s+")
            continue
        if char in _DASHES:
            if parts and parts[-1] == r"\s+":
                parts.pop()
            parts.append(r"\s*" + _DASH_CLASS + r"\s*")
        elif char in _SINGLE_QUOTES:
            parts.append(_SINGLE_CLASS)
        elif char in _DOUBLE_QUOTES:
            parts.append(_DOUBLE_CLASS)
        else:
            parts.append(re.escape(char))
        index += 1
    return "".join(parts)


def locate_quote(quote: str | None, text: str | None) -> str | None:
    """The exact substring of ``text`` that ``quote`` reproduces, or None."""
    if not quote or not text:
        return None
    stripped = quote.strip()
    if not stripped:
        return None
    if stripped in text:
        return stripped
    match = re.search(_pattern(stripped), text)
    if match is None:
        # Fallback: the stored PDF text may hyphenate words across line breaks
        # ("evalu- ation") where the model's quote does not. The returned quote
        # is still the stored text's own exact substring, hyphenation included.
        match = re.search(_pattern(stripped, line_breaks=True), text)
    if match is None:
        return None
    found = match.group(0).strip()
    return found or None
