"""Rank retrieved candidates by relevance to the ORIGINAL research topic.

Scores are cosine similarities between the topic (embedded as a query) and each
paper's title plus abstract (embedded as a document) with the local
sentence-transformers model, so no API is called. Ranking is deterministic:
score descending, then retrieval order.

Calibration (BAAI/bge-small-en-v1.5, topic "Multi-agent LLM systems for
scientific literature review and automated research gap discovery", 39
candidates of the phase-3 live run): papers on automated literature review or
gap discovery scored 0.78-0.89, LLM-agent papers from other domains 0.69-0.76,
and unrelated papers (chemistry, maintenance, blockchain, pedagogy) 0.52-0.69.
The default threshold 0.75 keeps every on-topic paper and the closest LLM-agent
papers and excludes every clearly unrelated one. It was calibrated on one topic
only; override it with PAPER_RELEVANCE_THRESHOLD.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass

from .models import PaperRecord

# v2: markup (e.g. JATS <jats:p> tags in Crossref abstracts) is stripped before scoring.
RANKING_METHOD = "topic_embedding_cosine_v2"
DEFAULT_RELEVANCE_THRESHOLD = 0.75
RELEVANCE_THRESHOLD_ENV = "PAPER_RELEVANCE_THRESHOLD"


def relevance_threshold() -> float:
    value = os.getenv(RELEVANCE_THRESHOLD_ENV)
    if value is None or not value.strip():
        return DEFAULT_RELEVANCE_THRESHOLD
    try:
        number = float(value)
    except ValueError as error:
        raise ValueError(f"{RELEVANCE_THRESHOLD_ENV} must be a number.") from error
    if not -1 <= number <= 1:
        raise ValueError(f"{RELEVANCE_THRESHOLD_ENV} must be in [-1, 1].")
    return number


@dataclass(frozen=True)
class RankedPaper:
    paper: PaperRecord
    score: float
    retrieval_index: int


_MARKUP = re.compile(r"<[^>]+>")


def paper_text(paper: PaperRecord) -> str:
    """Title and abstract, whitespace-normalized, without markup such as JATS tags."""
    title = " ".join(_MARKUP.sub(" ", paper.title or "").split())
    abstract = " ".join(_MARKUP.sub(" ", paper.abstract or "").split())
    return f"{title}. {abstract}" if title and abstract else (title or abstract)


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class PaperRelevanceRanker:
    def __init__(self, embedding_model=None):
        self._embedding_model = embedding_model  # lazy: no model load at construction

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            from src.embeddings import LocalEmbeddingModel

            self._embedding_model = LocalEmbeddingModel()
        return self._embedding_model

    def rank(self, topic: str, papers: list[PaperRecord]) -> list[RankedPaper]:
        if not topic or not topic.strip():
            raise ValueError("Research topic cannot be empty.")
        if not papers:
            return []
        texts = [paper_text(paper) for paper in papers]
        scorable = [index for index, text in enumerate(texts) if text]
        scores = [0.0] * len(papers)
        if scorable:
            query = self.embedding_model.embed_query(topic.strip())
            vectors = self.embedding_model.embed_documents([texts[i] for i in scorable])
            if len(vectors) != len(scorable):
                raise RuntimeError("Embedding model returned an unexpected number of vectors.")
            for index, vector in zip(scorable, vectors):
                scores[index] = round(_cosine(query, vector), 6)
        ranked = [RankedPaper(paper=paper, score=scores[index], retrieval_index=index)
                  for index, paper in enumerate(papers)]
        ranked.sort(key=lambda item: (-item.score, item.retrieval_index))
        return ranked
