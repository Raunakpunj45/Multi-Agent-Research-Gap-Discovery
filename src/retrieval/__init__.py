from .models import PaperRecord
from .openalex import OpenAlexRetriever
from .crossref import CrossrefRetriever
from .semantic_scholar import SemanticScholarRetriever
from .arxiv import ArxivRetriever
from .deduplicator import PaperDeduplicator
from .multi_source import MultiSourceRetriever

__all__ = [
    "PaperRecord",
    "OpenAlexRetriever",
    "CrossrefRetriever",
    "SemanticScholarRetriever",
    "ArxivRetriever",
    "PaperDeduplicator",
    "MultiSourceRetriever",
]