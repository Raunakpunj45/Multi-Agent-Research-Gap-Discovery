"""Data models of the deterministic paper x research-dimension evidence matrix.

Kept free of agent imports so the comparison artifact can embed the matrix
without an import cycle.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from .artifact_compat import OmitUnsetOptionalFields

# Matrix columns: the structured Paper Analysis fields.
MATRIX_DIMENSIONS = (
    "problem",
    "methodology",
    "dataset",
    "results",
    "limitations",
    "future_work",
)

# Dimensions whose statements are clustered across papers ("problem" is shared
# by construction: every paper was retrieved for the same topic).
CLUSTER_DIMENSIONS = (
    "limitations",
    "future_work",
    "methodology",
    "dataset",
    "results",
)

DIMENSION_CODES = {
    "problem": "PRB",
    "methodology": "MET",
    "dataset": "DAT",
    "results": "RES",
    "limitations": "LIM",
    "future_work": "FUT",
}


class MatrixCell(OmitUnsetOptionalFields):
    """One analysis statement of one paper, grounded in that paper's own chunk."""

    # Omitted while unset so matrices saved before these flags keep their fingerprints.
    _omit_when_none = ("secondary", "cluster_exclusion")

    paper_id: str
    dimension: str
    statement: str = Field(description="Sentence from the paper's structured analysis.")
    chunk_id: str | None = Field(
        default=None,
        description="Exact ID of the paper's own retrieved chunk that best matches the statement.",
    )
    quote: str | None = Field(
        default=None,
        description="Exact substring of that chunk's stored text (never paraphrased).",
    )
    grounding_similarity: float = 0.0
    grounded: bool = Field(
        default=False,
        description="True when the best-matching chunk sentence reaches the grounding threshold.",
    )
    secondary: bool | None = Field(
        default=None,
        description="True when the quote reports another work (citation marker, et al., "
                    "literature-table row): secondary evidence, not the paper's own finding.",
    )
    cluster_exclusion: str | None = Field(
        default=None,
        description="Why a grounded statement is kept out of cross-paper clusters.",
    )


class ClusterMember(BaseModel):
    """A paper's representative grounded statement inside a cluster."""

    paper_id: str
    chunk_id: str
    statement: str
    quote: str


class EvidenceCluster(BaseModel):
    """Related statements of one dimension, supported by >= 2 distinct papers."""

    cluster_id: str
    dimension: str
    paper_ids: list[str]
    members: list[ClusterMember]
    representative_statement: str
    mean_similarity: float


class EvidenceMatrix(BaseModel):
    paper_ids: list[str]
    dimensions: list[str]
    cells: list[MatrixCell]
    clusters: list[EvidenceCluster]
    embedding_model: str
    grounding_threshold: float
    similarity_threshold: float
    min_supporting_papers: int
    caveats: list[str] = Field(default_factory=list)

    def cell_for(self, paper_id: str, chunk_id: str) -> list[MatrixCell]:
        """Grounded cells of a paper that cite this exact chunk."""
        return [cell for cell in self.cells
                if cell.grounded and cell.paper_id == paper_id and cell.chunk_id == chunk_id]

    def quotes_for(self, paper_id: str, chunk_id: str) -> list[str]:
        """Exact quotes the matrix recorded for one (paper, chunk) pair."""
        quotes = [member.quote for cluster in self.clusters for member in cluster.members
                  if member.paper_id == paper_id and member.chunk_id == chunk_id]
        quotes += [cell.quote for cell in self.cell_for(paper_id, chunk_id) if cell.quote]
        return list(dict.fromkeys(quotes))
