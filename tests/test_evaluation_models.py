import pytest
from pydantic import ValidationError

from src.evaluation.models import (
    CandidateGap,
    Citation,
    Claim,
    EvaluationSystemOutput,
    NoveltyAssessment,
    NoveltyIndication,
    ReferenceGap,
    AnnotationRecord,
    ResearchQuestion,
)


def test_normalized_schema_keeps_claims_citations_questions_hypothesis_and_scope():
    output = EvaluationSystemOutput(
        system_name="rag_baseline", dataset_version="v1", paper_ids=["P1"],
        candidates=[CandidateGap(
            gap_id="G1", title="candidate", description="description",
            claims=[Claim(text="claim", citations=[Citation(
                paper_id="P1", chunk_id="P1-C1", validation_status="SUPPORTED",
                validation_limitations=["single chunk scope"],
            )])],
            research_questions=[ResearchQuestion(
                text="question", citations=[Citation(paper_id="P1", chunk_id="P1-C1")],
            )],
            novelty=NoveltyAssessment(
                indication=NoveltyIndication.NOT_ASSESSED,
                corpus_scope="Frozen test corpus only.",
            ),
        )],
    )
    restored = EvaluationSystemOutput.model_validate_json(output.model_dump_json())
    assert restored.candidates[0].claims[0].citations[0].chunk_id == "P1-C1"
    assert restored.candidates[0].research_questions[0].text == "question"
    assert restored.candidates[0].novelty.indication is NoveltyIndication.NOT_ASSESSED


def test_normalized_schema_rejects_unrecognized_global_novelty_value():
    with pytest.raises(ValidationError):
        NoveltyAssessment(indication="GLOBAL_NOVELTY_PROVEN")


def test_normalized_output_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        EvaluationSystemOutput(
            system_name="x", dataset_version="v1", paper_ids=[], candidates=[],
            unreviewed_field="should not disappear",
        )


def test_reference_gap_schema_requires_prompt_and_corpus_scope():
    reference = ReferenceGap(
        reference_gap_id="REF-001", research_prompt_id="Q1",
        title="human curated", description="bounded description",
        corpus_scope="Frozen evaluation sample only.",
    )
    assert reference.citations == [] and reference.claims == []
    record = AnnotationRecord(
        record_type="reference_set", dataset_version="v1",
        research_prompt_id="Q1", reference_gaps=[reference],
    )
    assert record.system_run_id is None and record.human_judgments == []
