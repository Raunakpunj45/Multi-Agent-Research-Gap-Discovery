"""Evaluation schemas independent of production agent response models."""
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class EvaluationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NoveltyIndication(str, Enum):
    INDICATED_WITHIN_SAMPLE = "INDICATED_WITHIN_SAMPLE"
    NOT_INDICATED_WITHIN_SAMPLE = "NOT_INDICATED_WITHIN_SAMPLE"
    NOT_ASSESSED = "NOT_ASSESSED"


class OutputCompletionStatus(str, Enum):
    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"


class Citation(EvaluationModel):
    paper_id: str
    chunk_id: str
    quote: str | None = None
    validation_status: str | None = None
    validation_limitations: list[str] = Field(default_factory=list)
    source_provenance: str | None = None


class Claim(EvaluationModel):
    text: str
    citations: list[Citation] = Field(default_factory=list)
    evidence_support: str | None = None
    validation_limitations: list[str] = Field(default_factory=list)


class ResearchQuestion(EvaluationModel):
    text: str
    rationale: str = ""
    citations: list[Citation] = Field(default_factory=list)


class Hypothesis(EvaluationModel):
    alternative: str
    null: str
    rationale: str = ""
    citations: list[Citation] = Field(default_factory=list)
    status: str = "PROPOSED"


class NoveltyAssessment(EvaluationModel):
    indication: NoveltyIndication = NoveltyIndication.NOT_ASSESSED
    rationale: str | None = None
    corpus_scope: str | None = None
    limitations: list[str] = Field(default_factory=list)


class CandidateGap(EvaluationModel):
    gap_id: str
    title: str
    description: str
    claims: list[Claim] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    research_questions: list[ResearchQuestion] = Field(default_factory=list)
    hypothesis: Hypothesis | None = None
    novelty: NoveltyAssessment = Field(default_factory=NoveltyAssessment)
    source_provenance: dict[str, str] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)


class EvaluationSystemOutput(EvaluationModel):
    """Normalized output that all evaluated systems must produce."""

    schema_version: str = "1.0"
    system_name: str
    dataset_version: str
    research_prompt_id: str | None = None
    paper_ids: list[str]
    completion_status: OutputCompletionStatus = OutputCompletionStatus.COMPLETE
    candidates: list[CandidateGap] = Field(default_factory=list)
    run_id: str | None = None
    source_provenance: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class HumanJudgment(EvaluationModel):
    annotation_id: str
    annotator_id: str
    candidate_gap_id: str
    reference_gap_id: str | None = None
    claim_id: str | None = None
    citation_id: str | None = None
    gap_relevance: int | None = Field(default=None, ge=1, le=5)
    evidence_support: str | None = None  # SUPPORTED / PARTIAL / UNSUPPORTED
    citation_support: str | None = None  # CORRECT / PARTIAL / INCORRECT
    novelty_indication: str | None = None  # SAMPLE_LOCAL / NOT_INDICATED / NOT_ASSESSED
    comments: str | None = None
    adjudicated: bool = False


class ReferenceGap(EvaluationModel):
    """Human-curated reference gap, scoped to the manifest prompt/corpus."""

    reference_gap_id: str
    research_prompt_id: str
    title: str
    description: str
    claims: list[Claim] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    corpus_scope: str
    curator_notes: str | None = None


class AnnotationRecord(EvaluationModel):
    """One annotation JSONL record; template rows are not gold judgments."""

    record_type: str = "annotation"
    annotation_status: str = "UNREVIEWED"
    dataset_version: str
    research_prompt_id: str
    system_run_id: str | None = None
    candidate_gap_id: str | None = None
    reference_gaps: list[ReferenceGap] = Field(default_factory=list)
    human_judgments: list[HumanJudgment] = Field(default_factory=list)


class MetricScore(EvaluationModel):
    name: str
    value: float | None
    numerator: float | None = None
    denominator: int = 0
    method: str  # AUTOMATIC or HUMAN
    missing_reason: str | None = None
    notes: list[str] = Field(default_factory=list)


class EvaluationRunManifest(EvaluationModel):
    run_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: str
    system_name: str
    dataset_version: str
    dataset_hash: str
    system_configuration: dict = Field(default_factory=dict)
    prompt_hash: str | None = None
    canonical_prompt_hash: str | None = None
    prompt_hash_status: str
    protocol_hash: str
    code_revision: str | None = None
    working_tree_dirty: bool | None = None
    research_prompt_ids: list[str] = Field(default_factory=list)
    evaluation_split: str | None = None
    paper_ids: list[str] = Field(default_factory=list)
    annotation_status: str = "MISSING"
    errors: list[str] = Field(default_factory=list)
    metrics: list[MetricScore] = Field(default_factory=list)
