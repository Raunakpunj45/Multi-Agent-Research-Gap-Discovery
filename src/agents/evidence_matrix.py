"""Deterministic cross-paper comparison: a paper x research-dimension evidence matrix.

Replaces the LLM cross-paper comparison on the default path (no model call,
no API tokens). From the saved Paper Analysis outputs it:

1. splits every analysis field (problem, methodology, dataset, results,
   limitations, future work) into statements;
2. grounds each statement in the paper's *own* retrieved chunks: the most
   similar chunk sentence (local embeddings) becomes the statement's quote,
   an exact substring of that chunk's stored text, with its exact chunk ID;
3. clusters related grounded statements of the same dimension across papers
   (average-linkage on local embeddings) and keeps only clusters supported by
   at least ``min_supporting_papers`` distinct papers.

The result is a standard ``MultiPaperComparison`` (so provenance, lineage
checks, critic integrity checks and checkpoint resume are unchanged) that also
carries the full ``EvidenceMatrix``. Clusters are similarity groupings, not
verified claims: two statements on the same subject may disagree. Downstream
gap detection, evidence validation and the critic judge their meaning.
"""
from __future__ import annotations

import math
import os
import re

from .artifact_provenance import ArtifactProvenance, fingerprint_json
from .evidence_precision import (
    contains_claim,
    is_absence_claim,
    is_secondary_in_context,
    non_evidence_reason,
    quote_states_absence,
    split_compound,
)
from .cross_paper_comparison_agent import (
    ComparativeFinding,
    EvidenceRef,
    MultiPaperComparison,
    UnderexploredDirection,
)
from .evidence_matrix_models import (
    CLUSTER_DIMENSIONS,
    DIMENSION_CODES,
    MATRIX_DIMENSIONS,
    ClusterMember,
    EvidenceCluster,
    EvidenceMatrix,
    MatrixCell,
)
from .paper_analysis_agent import PaperAnalysisResult

# Cross-paper comparison: "matrix" (default, deterministic, no LLM call) or
# "llm" (legacy CrossPaperComparisonAgent, kept for compatibility).
COMPARISON_MODE_ENV = "COMPARISON_MODE"
COMPARISON_MODES = ("matrix", "llm")


def comparison_mode() -> str:
    mode = (os.getenv(COMPARISON_MODE_ENV) or "matrix").strip().lower()
    if mode not in COMPARISON_MODES:
        raise ValueError(
            f"Unsupported {COMPARISON_MODE_ENV}={mode!r}; use one of {', '.join(COMPARISON_MODES)}."
        )
    return mode


MATRIX_STAGE = "evidence_matrix"
# v2: broader "evidence lacks this information" placeholder filtering.
# v3: compound statements split into clauses; a quote must contain the clause's
# claim; headers/bibliography never ground; absence claims and secondary
# (citing) evidence are kept out of clusters.
MATRIX_METHOD_VERSION = "evidence_matrix_v3"

# Calibrated offline on BAAI/bge-small-en-v1.5 (normalized embeddings):
# paraphrased limitations / future-work pairs score 0.76-0.84, unrelated
# limitations 0.47-0.66, and an analysis statement vs. its true source sentence
# ~0.64 (vs. ~0.49 for an unrelated sentence of the same paper). The cluster
# threshold favours precision: loosely related statements (e.g. "synthetic
# graphs only" vs "simulated records only", ~0.62) are not merged, because at
# that level unrelated statements merge too. Override per run with
# EVIDENCE_MATRIX_SIMILARITY_THRESHOLD / EVIDENCE_MATRIX_GROUNDING_THRESHOLD.
DEFAULT_SIMILARITY_THRESHOLD = 0.70
DEFAULT_GROUNDING_THRESHOLD = 0.55
SIMILARITY_THRESHOLD_ENV = "EVIDENCE_MATRIX_SIMILARITY_THRESHOLD"
GROUNDING_THRESHOLD_ENV = "EVIDENCE_MATRIX_GROUNDING_THRESHOLD"


def _threshold(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        number = float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a number.") from error
    if not 0 < number <= 1:
        raise ValueError(f"{name} must be in (0, 1].")
    return number

_NEGATION = re.compile(
    r"\b(?:not|no)\s+(?:explicitly\s+|clearly\s+|directly\s+)?"
    r"(?:available|provided|mentioned|specified|stated|described|discussed|"
    r"reported|identified|given|included|information|details?)\b",
    re.IGNORECASE,
)
_ABOUT_EVIDENCE = re.compile(
    r"\b(?:provided|retrieved|supplied|available)\s+(?:evidence|text|excerpts?|chunks?)\b|"
    r"\bin the (?:evidence|excerpts?|chunks?)\b|\binformation (?:is|was) not\b",
    re.IGNORECASE,
)
_PLACEHOLDER = re.compile(r"^(?:n/?a|none|not available|unknown|-+)\.?$", re.IGNORECASE)
# A statement that refers to the supplied evidence itself ("the evidence",
# "the provided evidence", "in the excerpts") AND negates ("does not provide",
# "No ... are mentioned", "lacks") describes missing information, not a finding
# of the paper, e.g. "The evidence does not provide explicit limitations" or
# "No future research directions are explicitly mentioned or supported by the
# provided evidence."
_EVIDENCE_REFERENCE = re.compile(
    r"\b(?:the|this|these|provided|retrieved|supplied|available|given)\s+"
    r"(?:evidence|excerpts?|chunks?|passages?)\b",
    re.IGNORECASE,
)
_ANY_NEGATION = re.compile(
    r"\b(?:not|no|none|neither|nor|without|lacks?|lacking|absent|missing)\b|n't\b",
    re.IGNORECASE,
)
_STATEMENT_SPLIT = re.compile(r"(?<=[.!?;])\s+|\s+[•*]\s+|\s+-\s+(?=[A-Z])")
_SENTENCE = re.compile(r"[^.!?]+(?:[.!?]+|$)")
_WORD = re.compile(r"\S+")

MIN_STATEMENT_WORDS = 2
MIN_QUOTE_WORDS = 5
QUOTE_WINDOW_WORDS = 40
QUOTE_WINDOW_STRIDE = 30


def _is_missing_information(statement: str) -> bool:
    """Analysis text saying the evidence lacks information is not a finding."""
    text = statement.strip()
    if _PLACEHOLDER.match(text):
        return True
    if _NEGATION.search(text) and _ABOUT_EVIDENCE.search(text):
        return True
    return bool(_EVIDENCE_REFERENCE.search(text) and _ANY_NEGATION.search(text))


def split_statements(text: str, limit: int) -> list[str]:
    """Clauses of one analysis field, without 'not available' placeholders.

    Sentences are split further into independently groundable clauses where a
    comma starts a new clause with its own subject and verb.
    """
    normalized = " ".join(str(text or "").split())
    statements: list[str] = []
    for sentence in _STATEMENT_SPLIT.split(normalized):
        for part in split_compound(sentence.strip(" -•*;")):
            part = part.strip(" -•*;")
            if len(part.split()) < MIN_STATEMENT_WORDS or _is_missing_information(part):
                continue
            if part not in statements:
                statements.append(part)
            if len(statements) >= limit:
                return statements
    return statements


def quote_candidates(text: str, max_chars: int) -> list[str]:
    """Candidate quotes of a chunk; every candidate is an exact substring of ``text``."""
    candidates: list[str] = []

    def add_span(start: int, end: int) -> None:
        words = list(_WORD.finditer(text, start, end))
        if not words:
            return
        if len(words) <= QUOTE_WINDOW_WORDS and words[-1].end() - words[0].start() <= max_chars:
            spans = [(words[0].start(), words[-1].end())]
        else:  # long or unpunctuated text: overlapping word windows
            spans = []
            for first in range(0, len(words), QUOTE_WINDOW_STRIDE):
                window = words[first:first + QUOTE_WINDOW_WORDS]
                spans.append((window[0].start(), window[-1].end()))
                if first + QUOTE_WINDOW_WORDS >= len(words):
                    break
        for begin, finish in spans:
            quote = text[begin:finish]
            if len(quote) > max_chars:  # cut at a word boundary, still a substring
                cut = text.rfind(" ", begin, begin + max_chars)
                quote = text[begin:cut if cut > begin else begin + max_chars]
            if len(quote.split()) >= MIN_QUOTE_WORDS and quote not in candidates:
                candidates.append(quote)

    for match in _SENTENCE.finditer(text):
        add_span(match.start(), match.end())
    if not candidates and text.strip():
        stripped = text.strip()
        start = text.index(stripped)
        candidates.append(text[start:start + min(len(stripped), max_chars)])
    return candidates


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def _average_linkage(similarity: list[list[float]], threshold: float) -> list[list[int]]:
    """Deterministic agglomerative clustering (average linkage, fixed tie order)."""
    clusters = [[index] for index in range(len(similarity))]
    while len(clusters) > 1:
        best, pair = threshold, None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                score = sum(similarity[i][j] for i in clusters[a] for j in clusters[b]) / (
                    len(clusters[a]) * len(clusters[b]))
                if score >= best and (pair is None or score > best):
                    best, pair = score, (a, b)
        if pair is None:
            break
        a, b = pair
        clusters[a] = sorted(clusters[a] + clusters[b])
        del clusters[b]
    return clusters


class EvidenceMatrixComparisonAgent:
    """Builds the evidence matrix and its comparison without any LLM call."""

    DIMENSIONS = MATRIX_DIMENSIONS

    def __init__(
        self,
        embedding_model=None,
        similarity_threshold: float | None = None,
        grounding_threshold: float | None = None,
        min_supporting_papers: int = 2,
        max_statements_per_cell: int = 6,
        max_clusters_per_dimension: int = 4,
        max_quote_chars: int = 300,
        max_papers: int = 60,
    ):
        if min_supporting_papers < 2:
            raise ValueError(
                "min_supporting_papers must be at least 2: a single-paper statement "
                "is not a cross-paper pattern."
            )
        if similarity_threshold is None:
            similarity_threshold = _threshold(SIMILARITY_THRESHOLD_ENV, DEFAULT_SIMILARITY_THRESHOLD)
        if grounding_threshold is None:
            grounding_threshold = _threshold(GROUNDING_THRESHOLD_ENV, DEFAULT_GROUNDING_THRESHOLD)
        if not 0 < similarity_threshold <= 1 or not 0 < grounding_threshold <= 1:
            raise ValueError("Similarity thresholds must be in (0, 1].")
        self._embedding_model = embedding_model  # lazy: no model load at construction
        self.similarity_threshold = similarity_threshold
        self.grounding_threshold = grounding_threshold
        self.min_supporting_papers = min_supporting_papers
        self.max_statements_per_cell = max_statements_per_cell
        self.max_clusters_per_dimension = max_clusters_per_dimension
        self.max_quote_chars = max_quote_chars
        self.max_papers = max_papers

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            from src.embeddings import LocalEmbeddingModel

            self._embedding_model = LocalEmbeddingModel()
        return self._embedding_model

    @property
    def embedding_model_name(self) -> str:
        model = self._embedding_model
        if model is None:
            from src.embeddings.local_embeddings import DEFAULT_MODEL

            return DEFAULT_MODEL
        return str(getattr(model, "model_name", type(model).__name__))

    # -- public API ----------------------------------------------------------

    def compare_many(
        self,
        results: list[PaperAnalysisResult],
        research_topic: str | None = None,
    ) -> MultiPaperComparison:
        paper_ids = self._validate_inputs(results)
        matrix = self.build_matrix(results, paper_ids)
        comparison = self._to_comparison(matrix)
        comparison.provenance = ArtifactProvenance(
            stage=MATRIX_STAGE,
            input_fingerprints={
                "paper_analyses": fingerprint_json(self._fingerprint_input(results, research_topic)),
            },
        )
        return comparison

    def build_matrix(self, results: list[PaperAnalysisResult],
                     paper_ids: list[str] | None = None) -> EvidenceMatrix:
        paper_ids = paper_ids or self._validate_inputs(results)

        # 1. statements (matrix rows x columns) and quote candidates per paper
        statements: list[tuple[str, str, str]] = []  # (paper_id, dimension, text)
        candidates: list[tuple[str, str, str]] = []  # (paper_id, chunk_id, quote)
        chunk_text: dict[str, str] = {}
        for paper_id, result in zip(paper_ids, results):
            for dimension in MATRIX_DIMENSIONS:
                for text in split_statements(getattr(result.analysis, dimension, ""),
                                             self.max_statements_per_cell):
                    statements.append((paper_id, dimension, text))
            for chunk in result.evidence:
                chunk_text[chunk.chunk_id] = chunk.text
                for quote in quote_candidates(chunk.text, self.max_quote_chars):
                    # Author/title headers and bibliographic entries are never evidence.
                    if non_evidence_reason(quote) is None:
                        candidates.append((paper_id, chunk.chunk_id, quote))

        texts = [text for _, _, text in statements] + [quote for _, _, quote in candidates]
        vectors = self.embedding_model.embed_documents(texts) if texts else []
        if len(vectors) != len(texts):
            raise RuntimeError("Embedding model returned an unexpected number of vectors.")
        statement_vectors = vectors[:len(statements)]
        candidate_vectors = vectors[len(statements):]

        # 2. ground each statement in the most similar sentence of its own
        #    paper's chunks that actually CONTAINS the statement's claim
        cells: list[MatrixCell] = []
        for (paper_id, dimension, text), vector in zip(statements, statement_vectors):
            best, best_score = None, -1.0
            for index, (owner, chunk_id, quote) in enumerate(candidates):
                if owner != paper_id:
                    continue  # never ground a statement in another paper's chunk
                if not contains_claim(text, quote):
                    continue  # topical similarity alone is not support
                score = _cosine(vector, candidate_vectors[index])
                if score > best_score:
                    best, best_score = (chunk_id, quote), score
            grounded = best is not None and best_score >= self.grounding_threshold
            quote = best[1] if grounded else None
            secondary = bool(quote and is_secondary_in_context(quote, chunk_text[best[0]]))
            exclusion = None
            if grounded and is_absence_claim(text) and not quote_states_absence(quote):
                exclusion = ("absence claim: the quote does not state the absence "
                             "(missing information is not evidence)")
            elif secondary:
                exclusion = ("secondary evidence: the quote reports another work, not "
                             "the paper's own finding")
            cells.append(MatrixCell(
                paper_id=paper_id, dimension=dimension, statement=text,
                chunk_id=best[0] if grounded else None, quote=quote,
                grounding_similarity=round(max(best_score, 0.0), 4),
                grounded=grounded,
                secondary=True if secondary else None,
                cluster_exclusion=exclusion,
            ))

        # 3. cluster grounded, primary, non-absence statements across papers
        clusters: list[EvidenceCluster] = []
        for dimension in CLUSTER_DIMENSIONS:
            members = [(cell, statement_vectors[index])
                       for index, cell in enumerate(cells)
                       if cell.dimension == dimension and cell.grounded
                       and cell.cluster_exclusion is None]
            clusters.extend(self._cluster_dimension(dimension, members, paper_ids))

        ungrounded = sum(1 for cell in cells if not cell.grounded)
        absence = sum(1 for cell in cells
                      if (cell.cluster_exclusion or "").startswith("absence"))
        secondary_cells = sum(1 for cell in cells if cell.secondary)
        caveats = [
            "Deterministic matrix: statements come from each paper's structured analysis "
            "and are grounded only in that paper's own retrieved chunks; no LLM was used.",
            "Clusters group statements by embedding similarity "
            f"(threshold {self.similarity_threshold}); similar statements may still "
            "disagree, so clusters are evidence leads, not verified findings.",
            f"Only clusters supported by at least {self.min_supporting_papers} distinct "
            "papers are reported; conflicts between papers are not detected here.",
        ]
        if ungrounded:
            caveats.append(
                f"{ungrounded} analysis statement(s) had no sentence in their paper's "
                f"retrieved chunks that contains the statement's claim and reaches the "
                f"grounding threshold {self.grounding_threshold}; they were excluded from "
                "clustering."
            )
        if absence:
            caveats.append(
                f"{absence} statement(s) only say that something is not mentioned and "
                "the quote does not state that absence; missing information in retrieved "
                "chunks is not evidence, so they were excluded from clustering."
            )
        if secondary_cells:
            caveats.append(
                f"{secondary_cells} statement(s) are grounded in secondary evidence "
                "(citation markers, et al., literature-table rows) reporting other works; "
                "they are not counted as the papers' own findings and were excluded from "
                "clustering."
            )
        return EvidenceMatrix(
            paper_ids=list(paper_ids), dimensions=list(MATRIX_DIMENSIONS),
            cells=cells, clusters=clusters,
            embedding_model=self.embedding_model_name,
            grounding_threshold=self.grounding_threshold,
            similarity_threshold=self.similarity_threshold,
            min_supporting_papers=self.min_supporting_papers,
            caveats=caveats,
        )

    # -- internals -------------------------------------------------------------

    def _validate_inputs(self, results: list[PaperAnalysisResult]) -> list[str]:
        if len(results) < 2:
            raise ValueError("At least two paper analyses are required.")
        if len(results) > self.max_papers:
            raise ValueError(
                f"Received {len(results)} paper analyses; the limit per comparison "
                f"is {self.max_papers}."
            )
        paper_ids: list[str] = []
        for number, result in enumerate(results, start=1):
            if not result.evidence:
                raise ValueError(f"Paper analysis #{number} has no evidence chunks.")
            paper_id = result.evidence[0].paper_id
            owners = {chunk.paper_id for chunk in result.evidence}
            if owners != {paper_id} or not paper_id or paper_id == "unknown":
                raise ValueError(
                    f"Paper analysis #{number} cites evidence from multiple or unknown "
                    f"papers: {sorted(owners)}."
                )
            if paper_id in paper_ids:
                raise ValueError(f"Duplicate paper ID in comparison input: {paper_id}")
            chunk_ids = [chunk.chunk_id for chunk in result.evidence]
            if len(chunk_ids) != len(set(chunk_ids)) or not all(chunk_ids):
                raise ValueError(f"Paper analysis #{number} has missing or duplicate chunk IDs.")
            paper_ids.append(paper_id)
        return paper_ids

    def _cluster_dimension(self, dimension, members, paper_ids) -> list[EvidenceCluster]:
        if len({cell.paper_id for cell, _ in members}) < self.min_supporting_papers:
            return []
        similarity = [[_cosine(a, b) for _, b in members] for _, a in members]
        groups = []
        for indices in _average_linkage(similarity, self.similarity_threshold):
            papers = list(dict.fromkeys(members[i][0].paper_id for i in indices))
            if len(papers) < self.min_supporting_papers:
                continue
            # One representative statement per paper: the most central one.
            chosen = []
            for paper in papers:
                own = [i for i in indices if members[i][0].paper_id == paper]
                others = [j for j in indices if members[j][0].paper_id != paper]
                chosen.append(max(own, key=lambda i: (
                    sum(similarity[i][j] for j in others) / len(others), -i)))
            pairs = [(a, b) for n, a in enumerate(chosen) for b in chosen[n + 1:]]
            mean = sum(similarity[a][b] for a, b in pairs) / len(pairs)
            central = max(chosen, key=lambda i: (
                sum(similarity[i][j] for j in chosen if j != i), -i))
            groups.append((len(papers), mean, min(indices), chosen, central))

        groups.sort(key=lambda item: (-item[0], -item[1], item[2]))
        order = {paper: position for position, paper in enumerate(paper_ids)}
        clusters = []
        for number, (_, mean, _, chosen, central) in enumerate(
                groups[:self.max_clusters_per_dimension], start=1):
            chosen = sorted(chosen, key=lambda i: order[members[i][0].paper_id])
            clusters.append(EvidenceCluster(
                cluster_id=f"{DIMENSION_CODES[dimension]}-{number}",
                dimension=dimension,
                paper_ids=[members[i][0].paper_id for i in chosen],
                members=[ClusterMember(
                    paper_id=members[i][0].paper_id, chunk_id=members[i][0].chunk_id,
                    statement=members[i][0].statement, quote=members[i][0].quote,
                ) for i in chosen],
                representative_statement=members[central][0].statement,
                mean_similarity=round(mean, 4),
            ))
        return clusters

    @staticmethod
    def _refs(cluster: EvidenceCluster) -> list[EvidenceRef]:
        refs: dict[str, list[str]] = {}
        for member in cluster.members:
            chunks = refs.setdefault(member.paper_id, [])
            if member.chunk_id not in chunks:
                chunks.append(member.chunk_id)
        return [EvidenceRef(paper_id=paper, chunk_ids=chunks) for paper, chunks in refs.items()]

    @staticmethod
    def _short(text: str, limit: int = 120) -> str:
        return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"

    def _to_comparison(self, matrix: EvidenceMatrix) -> MultiPaperComparison:
        recurring, limitations, directions = [], [], []
        for cluster in matrix.clusters:
            # Embedding similarity groups statements on the same subject; it
            # does not show that the papers agree or share a method.
            description = (
                f"{cluster.dimension}: similar statements from {len(cluster.paper_ids)} "
                f"papers (embedding similarity {cluster.mean_similarity}; topical "
                "similarity, not verified agreement): "
                + " | ".join(f"{m.paper_id}: {m.statement}" for m in cluster.members)
            )
            title = f"{cluster.cluster_id}: {self._short(cluster.representative_statement)}"
            if cluster.dimension == "future_work":
                directions.append(UnderexploredDirection(
                    direction=title, rationale=description, evidence=self._refs(cluster),
                ))
            elif cluster.dimension == "limitations":
                limitations.append(ComparativeFinding(
                    title=title, description=description, dimensions=[cluster.dimension],
                    evidence=self._refs(cluster),
                ))
            else:
                recurring.append(ComparativeFinding(
                    title=title, description=description, dimensions=[cluster.dimension],
                    evidence=self._refs(cluster),
                ))
        caveats = list(matrix.caveats) + [
            "Candidate directions come only from the supplied papers; their absence "
            "elsewhere is not established and novelty is limited to this corpus.",
        ]
        if not matrix.clusters:
            caveats.append(
                "No cross-paper pattern was supported by at least "
                f"{matrix.min_supporting_papers} distinct papers."
            )
        return MultiPaperComparison(
            paper_ids=list(matrix.paper_ids), paper_count=len(matrix.paper_ids),
            dimension_summaries=[], recurring_patterns=recurring,
            common_limitations=limitations, conflicting_findings=[],
            methodological_differences=[], underexplored_directions=directions,
            caveats=caveats, evidence_matrix=matrix,
        )

    def _fingerprint_input(self, results, research_topic) -> dict:
        return {
            "method": MATRIX_METHOD_VERSION,
            "research_topic": (research_topic or "").strip(),
            "embedding_model": self.embedding_model_name,
            "similarity_threshold": self.similarity_threshold,
            "grounding_threshold": self.grounding_threshold,
            "min_supporting_papers": self.min_supporting_papers,
            "max_statements_per_cell": self.max_statements_per_cell,
            "max_clusters_per_dimension": self.max_clusters_per_dimension,
            "max_quote_chars": self.max_quote_chars,
            "papers": [{
                "analysis": result.analysis.model_dump(mode="json"),
                "evidence": [{"paper_id": chunk.paper_id, "chunk_id": chunk.chunk_id,
                              "text": chunk.text} for chunk in result.evidence],
            } for result in results],
        }
