import re

from .models import PaperRecord

# Lexical pre-filter for a single search query (used by standalone
# MultiSourceRetriever callers). The planner workflow does NOT use it: there,
# every deduplicated candidate is scored against the original topic with local
# embeddings and excluded only by the explicit relevance threshold.
#
# Planner queries are often boolean ('"LLM" AND "gap detection"'). Terms are
# content words only (no operators, quotes or stopwords), matched as whole
# words after acronym/phrase canonicalization, so "LLM" matches "large language
# model(s)" and "multi-agent" matches "multi agent" or "multiagent".
_TOKEN = re.compile(r"[a-z0-9]+")
_MARKUP = re.compile(r"<[^>]+>")
_BOOLEAN_OPERATORS = {"and", "or", "not"}
_STOPWORDS = {
    "a", "an", "the", "of", "for", "in", "on", "to", "with", "by", "from", "at",
    "as", "is", "are", "be", "its", "their", "this", "that", "these", "into",
    "via", "using", "based", "toward", "towards", "between", "over", "about",
}
# (pattern on lowercase text, canonical token). Applied to queries and papers
# alike, so each equivalence works in both directions.
_EQUIVALENCES = (
    (re.compile(r"\blarge[\s-]+language[\s-]+models?\b"), " llm "),
    (re.compile(r"\bllms\b"), " llm "),
    (re.compile(r"\bretrieval[\s-]+augmented[\s-]+generation\b"), " rag "),
    (re.compile(r"\bnatural[\s-]+language[\s-]+processing\b"), " nlp "),
    (re.compile(r"\bartificial[\s-]+intelligence\b"), " ai "),
    # "multi-agent" keeps "agent" too, so it still matches an "agent" query term.
    (re.compile(r"\bmulti[\s-]*agents?\b"), " multiagent agent "),
    (re.compile(r"\bmas\b"), " multiagent agent "),
)
MIN_TERM_MATCHES = 2
# A record without an abstract offers only its title: one matched content term
# is enough, so clearly relevant title-only records are not discarded.
MIN_TITLE_ONLY_MATCHES = 1


def _normalize(token: str) -> str:
    # Light plural folding so "agents" matches "agent".
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def canonical_text(text: str) -> str:
    """Lowercase text with markup removed and acronym/phrase equivalences applied."""
    value = _MARKUP.sub(" ", (text or "").lower())
    for pattern, token in _EQUIVALENCES:
        value = pattern.sub(token, value)
    return value


def content_terms(text: str) -> set[str]:
    """Canonical content words of a query or text, without operators or stopwords."""
    return {
        _normalize(token)
        for token in _TOKEN.findall(canonical_text(text))
        if token not in _BOOLEAN_OPERATORS and token not in _STOPWORDS
    }


class RelevanceFilter:
    def __init__(self, min_matches: int = MIN_TERM_MATCHES,
                 min_title_only_matches: int = MIN_TITLE_ONLY_MATCHES):
        if min_matches < 1 or min_title_only_matches < 1:
            raise ValueError("Minimum term matches must be at least 1.")
        self.min_matches = min_matches
        self.min_title_only_matches = min_title_only_matches

    def matched_terms(self, paper: PaperRecord, query: str) -> set[str]:
        text = " ".join([paper.title or "", paper.abstract or ""])
        return content_terms(query) & content_terms(text)

    def required_matches(self, paper: PaperRecord, query: str) -> int:
        terms = content_terms(query)
        minimum = (self.min_matches if (paper.abstract or "").strip()
                   else self.min_title_only_matches)
        return min(minimum, len(terms))

    def is_relevant(self, paper: PaperRecord, query: str) -> bool:
        if not content_terms(query):
            return True  # a query without content terms cannot filter
        return len(self.matched_terms(paper, query)) >= self.required_matches(paper, query)

    def filter(
        self,
        papers: list[PaperRecord],
        query: str,
    ) -> list[PaperRecord]:
        """Keep papers whose title/abstract contain enough distinct query terms.

        Papers with an abstract need ``min_matches`` distinct content terms
        (or all of them, for a shorter query); title-only papers need
        ``min_title_only_matches``.
        """
        return [paper for paper in papers if self.is_relevant(paper, query)]
