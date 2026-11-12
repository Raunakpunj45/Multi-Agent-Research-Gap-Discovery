from types import SimpleNamespace

from src.agents import (
    ComparativeFinding,
    CriticAgent,
    CriticDecision,
    CriticReviewStatus,
    CriticismCategory,
    CriticismSeverity,
    EvidenceRef,
    EvidenceValidationResult,
    EvidenceValidationStatus,
    GapCategory,
    GapConfidence,
    GapOrigin,
    GapOriginAssessment,
    GapSupportStatus,
    GapValidation,
    GapValidationStatus,
    JointGapAssessment,
    MultiPaperComparison,
    ResearchGap,
    ResearchGapDetectionResult,
)
from src.agents.critic_agent import (
    _LLMCriticResult,
    _LLMCriticism,
    _LLMGapReview,
    _LLMRevisionSuggestion,
)

P1, P2 = "https://openalex.org/W1", "https://openalex.org/W2"
C1, C2 = f"{P1}_chunk_1", f"{P2}_chunk_1"


def source_refs():
    return [EvidenceRef(paper_id=P1, chunk_ids=[C1]), EvidenceRef(paper_id=P2, chunk_ids=[C2])]


def comparison():
    return MultiPaperComparison(
        paper_ids=[P1, P2], paper_count=2,
        dimension_summaries=[], recurring_patterns=[],
        common_limitations=[ComparativeFinding(
            title="Small evaluation", description="Both studies report small evaluations.",
            dimensions=["dataset"], evidence=source_refs(),
        )], conflicting_findings=[], methodological_differences=[],
        underexplored_directions=[], caveats=["Only two papers were compared."],
    )


def research_gap(origin=GapOrigin.INFERRED):
    return ResearchGap(
        gap_id="GAP-001", title="Evaluation gap",
        description="The evidence base lacks broad evaluation.",
        gap_category=GapCategory.EMPIRICAL,
        reasoning="Both papers describe small evaluations.",
        supporting_paper_ids=[P1, P2], supporting_chunk_ids=[C1, C2],
        supporting_evidence=source_refs(),
        evidence_summary="P1 and P2 report small evaluations.",
        gap_origin=origin, confidence=GapConfidence.MEDIUM,
        confidence_reason="Two papers contribute evidence.",
        research_opportunity="Evaluate across a broader sample.",
    )


def gap_result(gaps=None):
    return ResearchGapDetectionResult(
        research_topic="topic", paper_count=2, paper_ids=[P1, P2],
        gaps=[research_gap()] if gaps is None else gaps,
        gap_category_counts={"Empirical Gap": 1}, caveats=[],
    )


def validation(gap_id="GAP-001", status=GapValidationStatus.FULLY_SUPPORTED):
    joint = JointGapAssessment(
        status=GapSupportStatus.FULLY_SUPPORTED,
        complete_gap_assessment="The supplied chunks support the bounded gap.",
        cross_paper_inference="The papers can be synthesized.",
        evidence_summary_accuracy="The summary is aligned.",
        research_opportunity="A broader evaluation may be useful.",
        gap_origin="The gap is inferred across papers.",
        overgeneralization="No field-wide inference is warranted.",
        limitations="Only two studies were supplied.",
        counterevidence="None supplied.", source_evidence=source_refs(),
    )
    return EvidenceValidationResult(
        research_topic="topic", paper_ids=[P1, P2],
        gap_validations=[GapValidation(
            gap_id=gap_id, status=status, explanation="Evidence checked.",
            evidence_validations=[
                {"paper_id": P1, "chunk_id": C1, "status": "SUPPORTED", "findings": "P1 supports its contribution."},
                {"paper_id": P2, "chunk_id": C2, "status": "SUPPORTED", "findings": "P2 supports its contribution."},
            ], joint_assessment=joint,
        )],
        validation_counts={status.value: 1}, evidence_status_counts={"SUPPORTED": 2},
    )


def llm_review(gap_id="GAP-001", evidence_labels=None, origin=GapOriginAssessment.INFERRED):
    return _LLMCriticResult(gap_reviews=[_LLMGapReview(
        gap_id=gap_id, decision=CriticDecision.REVISE,
        origin_assessment=origin,
        origin_rationale="The exact combined gap is inferred by comparing papers.",
        reasoning="The evidence supports a bounded question, but not a field-wide claim.",
        criticisms=[_LLMCriticism(
            category=CriticismCategory.OVERGENERALIZATION,
            severity=CriticismSeverity.MEDIUM,
            affected_claim="The evidence base lacks broad evaluation.",
            issue="The supplied collection is too small to support a field-wide conclusion.",
            rationale="The comparison contains only two papers.",
            evidence_labels=["P1-E1"] if evidence_labels is None else evidence_labels,
        )],
        revision_suggestions=[_LLMRevisionSuggestion(
            field="description", current_issue="The claim exceeds the sample scope.",
            proposed_text="Within these two papers, evaluations remain limited in scope.",
            evidence_labels=["P1-E1", "P2-E1"],
        )],
    )])


class FakeClient:
    def __init__(self, parsed=None, error=None):
        self.parsed, self.error, self.calls = parsed, error, []
        self.models = SimpleNamespace(generate_content=self.generate)

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(parsed=self.parsed)


def test_complete_review_maps_evidence_labels_to_original_ids_and_bounds_novelty():
    client = FakeClient(llm_review())
    result = CriticAgent(client=client).critique(comparison(), gap_result(), validation())

    review = result.gap_reviews[0]
    assert result.review_status is CriticReviewStatus.COMPLETE
    assert review.decision is CriticDecision.REVISE
    assert review.origin_assessment is GapOriginAssessment.INFERRED
    assert review.criticisms[0].source_evidence == [EvidenceRef(paper_id=P1, chunk_ids=[C1])]
    assert review.revision_suggestions[0].source_evidence == source_refs()
    prompt = client.calls[0]["contents"]
    assert "FULLY_SUPPORTED does not establish novelty" in prompt
    assert P1 not in prompt and P2 not in prompt


def test_author_stated_origin_is_returned_as_a_distinct_classification():
    client = FakeClient(llm_review(origin=GapOriginAssessment.AUTHOR_STATED))
    result = CriticAgent(client=client).critique(
        comparison(), gap_result([research_gap(GapOrigin.EXPLICIT)]), validation()
    )
    assert result.gap_reviews[0].origin_assessment is GapOriginAssessment.AUTHOR_STATED


def test_missing_inputs_return_incomplete_without_calling_gemini():
    client = FakeClient(llm_review())
    result = CriticAgent(client=client).critique(None, gap_result(), validation())
    assert result.review_status is CriticReviewStatus.INCOMPLETE
    assert result.missing_inputs == ["cross-paper comparison output"]
    assert result.gap_reviews[0].decision is CriticDecision.REVISE
    assert not client.calls


def test_input_reference_mismatch_stops_before_gemini():
    bad_validation = validation()
    bad_validation.gap_validations[0].evidence_validations.pop()
    client = FakeClient(llm_review())
    result = CriticAgent(client=client).critique(comparison(), gap_result(), bad_validation)
    assert result.review_status is CriticReviewStatus.INCOMPLETE
    assert "references do not match" in result.errors[0]
    assert not client.calls


def test_model_failure_returns_safe_incomplete_reviews():
    class ProviderError(Exception):
        code = 429

    raw_details = "private retryAfter=48624s response_payload=SECRET_PROVIDER_BODY"
    client = FakeClient(error=ProviderError(raw_details))
    result = CriticAgent(client=client, max_retries=1).critique(
        comparison(), gap_result(), validation()
    )
    assert result.review_status is CriticReviewStatus.INCOMPLETE
    assert result.gap_reviews[0].review_status is CriticReviewStatus.INCOMPLETE
    assert result.gap_reviews[0].decision is CriticDecision.REVISE
    serialized = result.model_dump_json()
    assert result.errors == [
        "Critic assessment failed: API quota exhausted (HTTP 429)."
    ]
    assert "SECRET_PROVIDER_BODY" not in serialized
    assert "48624s" not in serialized
    assert "private retryAfter" not in serialized


def test_invalid_evidence_labels_are_omitted_and_mark_review_incomplete():
    client = FakeClient(llm_review(evidence_labels=["P9-E99"]))
    result = CriticAgent(client=client).critique(comparison(), gap_result(), validation())
    review = result.gap_reviews[0]
    assert review.review_status is CriticReviewStatus.INCOMPLETE
    assert review.criticisms[0].source_evidence == []
    assert any("unknown evidence label" in error for error in result.errors)


def test_incomplete_evidence_validation_cannot_be_accepted():
    accepted = _LLMGapReview(
        gap_id="GAP-001", decision=CriticDecision.ACCEPT,
        origin_assessment=GapOriginAssessment.INFERRED,
        origin_rationale="The gap is inferred.", reasoning="Looks supported.",
    )
    client = FakeClient(_LLMCriticResult(gap_reviews=[accepted]))
    result = CriticAgent(client=client).critique(
        comparison(), gap_result(), validation(status=GapValidationStatus.VALIDATION_INCOMPLETE)
    )
    review = result.gap_reviews[0]
    assert review.decision is CriticDecision.REVISE
    assert review.review_status is CriticReviewStatus.INCOMPLETE
    assert any("Evidence validation is incomplete" in caveat for caveat in review.caveats)


def test_omitted_gap_review_becomes_explicitly_incomplete():
    client = FakeClient(_LLMCriticResult(gap_reviews=[]))
    result = CriticAgent(client=client).critique(comparison(), gap_result(), validation())
    assert result.review_status is CriticReviewStatus.INCOMPLETE
    assert result.gap_reviews[0].gap_id == "GAP-001"
    assert "omitted" in result.errors[0]


def test_no_candidate_gaps_completes_without_gemini():
    client = FakeClient()
    result = CriticAgent(client=client).critique(
        comparison(), gap_result([]), evidence_validation=validation()
    )
    assert result.review_status is CriticReviewStatus.COMPLETE
    assert result.gap_reviews == []
    assert not client.calls
