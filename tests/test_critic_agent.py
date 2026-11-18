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
    fingerprint_inputs,
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
                {"paper_id": P1, "chunk_id": C1, "status": "SUPPORTED", "findings": "P1 supports its contribution.",
                 "supported_contribution_quote": "P1 reports a small evaluation"},
                {"paper_id": P2, "chunk_id": C2, "status": "SUPPORTED", "findings": "P2 supports its contribution.",
                 "supported_contribution_quote": "P2 reports a small evaluation"},
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


class SequencedClient:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []
        self.models = SimpleNamespace(generate_content=self.generate)

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(parsed=outcome)


def test_complete_review_maps_evidence_labels_to_original_ids_and_bounds_novelty():
    client = FakeClient(llm_review())
    result = CriticAgent(review_mode="auto", client=client).critique(comparison(), gap_result(), validation())

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
    result = CriticAgent(review_mode="auto", client=client).critique(
        comparison(), gap_result([research_gap(GapOrigin.EXPLICIT)]), validation()
    )
    assert result.gap_reviews[0].origin_assessment is GapOriginAssessment.AUTHOR_STATED


def test_missing_inputs_return_incomplete_without_calling_gemini():
    client = FakeClient(llm_review())
    result = CriticAgent(review_mode="auto", client=client).critique(None, gap_result(), validation())
    assert result.review_status is CriticReviewStatus.INCOMPLETE
    assert result.missing_inputs == ["cross-paper comparison output"]
    assert result.gap_reviews[0].decision is CriticDecision.REVISE
    assert not client.calls


def test_input_reference_mismatch_stops_before_gemini():
    bad_validation = validation()
    bad_validation.gap_validations[0].evidence_validations.pop()
    client = FakeClient(llm_review())
    result = CriticAgent(review_mode="auto", client=client).critique(comparison(), gap_result(), bad_validation)
    assert result.review_status is CriticReviewStatus.INCOMPLETE
    assert "references do not match" in result.errors[0]
    assert not client.calls


def test_model_failure_returns_safe_incomplete_reviews():
    class ProviderError(Exception):
        code = 429

    raw_details = "private retryAfter=48624s response_payload=SECRET_PROVIDER_BODY"
    client = FakeClient(error=ProviderError(raw_details))
    result = CriticAgent(review_mode="auto", client=client, max_retries=1).critique(
        comparison(), gap_result(), validation()
    )
    assert result.review_status is CriticReviewStatus.INCOMPLETE
    assert result.gap_reviews[0].review_status is CriticReviewStatus.INCOMPLETE
    assert result.gap_reviews[0].decision is CriticDecision.REVISE
    serialized = result.model_dump_json()
    assert result.errors == [
        "GAP-001: Critic assessment failed: API quota exhausted (HTTP 429)."
    ]
    assert "SECRET_PROVIDER_BODY" not in serialized
    assert "48624s" not in serialized
    assert "private retryAfter" not in serialized


def test_daily_quota_exhaustion_stops_retries_and_later_gap_calls():
    class ProviderError(Exception):
        code = 429

    daily_error = ProviderError(
        "RESOURCE_EXHAUSTED: quota exceeded for GenerateContentRequestsPerDay per day"
    )
    first = research_gap()
    second = research_gap().model_copy(update={"gap_id": "GAP-002"})
    gaps = gap_result([first, second])
    evidence = EvidenceValidationResult(
        research_topic="topic", paper_ids=[P1, P2],
        gap_validations=validation("GAP-001").gap_validations
        + validation("GAP-002").gap_validations,
        validation_counts={"FULLY_SUPPORTED": 2}, evidence_status_counts={"SUPPORTED": 4},
    )
    client = FakeClient(error=daily_error)

    result = CriticAgent(review_mode="auto", client=client, max_retries=5).critique(
        comparison(), gaps, evidence,
    )

    assert len(client.calls) == 1
    assert [review.review_status for review in result.gap_reviews] == [
        CriticReviewStatus.INCOMPLETE, CriticReviewStatus.INCOMPLETE,
    ]
    assert "daily API quota was exhausted" in result.gap_reviews[1].reasoning


def test_transient_rate_limit_retries_using_provider_retry_after(monkeypatch):
    class ProviderError(Exception):
        code = 429

        def __init__(self):
            self.response = SimpleNamespace(headers={"Retry-After": "3"})

    sleeps = []
    monkeypatch.setattr("src.agents.critic_agent.time.sleep", sleeps.append)
    client = SequencedClient([ProviderError(), llm_review()])
    agent = CriticAgent(review_mode="auto", client=client, max_retries=3, max_backoff_seconds=5)

    parsed = agent._generate("prompt")

    assert isinstance(parsed, _LLMCriticResult)
    assert len(client.calls) == 2
    assert sleeps == [3.0]


def test_transient_rate_limit_reads_google_retry_info_details(monkeypatch):
    class ProviderError(Exception):
        code = 429
        details = {"error": {"details": [{
            "@type": "type.googleapis.com/google.rpc.RetryInfo",
            "retryDelay": "4s",
        }]}}

    sleeps = []
    monkeypatch.setattr("src.agents.critic_agent.time.sleep", sleeps.append)
    client = SequencedClient([ProviderError("rate limit"), llm_review()])
    agent = CriticAgent(review_mode="auto", client=client, max_retries=3, max_backoff_seconds=5)

    agent._generate("prompt")

    assert len(client.calls) == 2
    assert sleeps == [4.0]


def test_transient_rate_limit_without_guidance_uses_bounded_backoff(monkeypatch):
    class ProviderError(Exception):
        code = 429

    sleeps = []
    monkeypatch.setattr("src.agents.critic_agent.time.sleep", sleeps.append)
    client = SequencedClient([ProviderError("rate limit"), ProviderError("rate limit"), llm_review()])
    agent = CriticAgent(review_mode="auto", client=client, max_retries=3,
                        base_backoff_seconds=2, max_backoff_seconds=3)

    agent._generate("prompt")

    assert len(client.calls) == 3
    assert sleeps == [2, 3]


def test_fingerprints_change_when_any_input_bytes_change():
    payloads = {
        "comparison": b"comparison-v1",
        "gap_detection": b"gaps-v1",
        "evidence_validation": b"validation-v1",
    }
    original = fingerprint_inputs(payloads)
    assert original == fingerprint_inputs(dict(payloads))
    for name in payloads:
        changed = dict(payloads)
        changed[name] += b" changed"
        assert fingerprint_inputs(changed) != original


def test_matching_successful_review_is_preserved_after_partial_failure():
    fingerprints = {
        "comparison": "a" * 64,
        "gap_detection": "b" * 64,
        "evidence_validation": "c" * 64,
    }
    previous = CriticAgent(review_mode="auto", client=FakeClient(llm_review())).critique(
        comparison(), gap_result(), validation(), input_fingerprints=fingerprints,
    )
    old_review = previous.gap_reviews[0].model_copy(update={"decision": CriticDecision.ACCEPT})
    previous = previous.model_copy(update={"gap_reviews": [old_review]})
    first = research_gap()
    second = research_gap().model_copy(update={"gap_id": "GAP-002"})
    current_gaps = gap_result([first, second])
    current_validation = EvidenceValidationResult(
        research_topic="topic", paper_ids=[P1, P2],
        gap_validations=validation("GAP-001").gap_validations
        + validation("GAP-002").gap_validations,
        validation_counts={"FULLY_SUPPORTED": 2}, evidence_status_counts={"SUPPORTED": 4},
    )
    client = SequencedClient([RuntimeError("temporary outage")])
    result = CriticAgent(review_mode="auto", client=client, max_retries=1).critique(
        comparison(), current_gaps, current_validation,
        input_fingerprints=fingerprints, previous_result=previous,
    )

    assert len(client.calls) == 1  # GAP-001 reused; only GAP-002 was assessed.
    assert result.gap_reviews[0].decision is CriticDecision.ACCEPT
    assert result.gap_reviews[0].review_status is CriticReviewStatus.COMPLETE
    assert any("Reused a previously successful review" in item
               for item in result.gap_reviews[0].caveats)
    assert result.gap_reviews[1].review_status is CriticReviewStatus.INCOMPLETE
    assert result.review_status is CriticReviewStatus.INCOMPLETE


def test_stale_successful_review_is_not_preserved():
    previous_fingerprints = {
        "comparison": "a" * 64,
        "gap_detection": "b" * 64,
        "evidence_validation": "c" * 64,
    }
    current_fingerprints = dict(previous_fingerprints, evidence_validation="d" * 64)
    previous = CriticAgent(review_mode="auto", client=FakeClient(llm_review())).critique(
        comparison(), gap_result(), validation(), input_fingerprints=previous_fingerprints,
    )
    previous = previous.model_copy(update={"gap_reviews": [previous.gap_reviews[0].model_copy(
        update={"decision": CriticDecision.ACCEPT})]})
    client = FakeClient(error=RuntimeError("temporary outage"))
    failed = CriticAgent(review_mode="auto", client=client, max_retries=1).critique(
        comparison(), gap_result(), validation(), input_fingerprints=current_fingerprints,
        previous_result=previous,
    )

    assert len(client.calls) == 1
    assert failed.gap_reviews[0].review_status is CriticReviewStatus.INCOMPLETE
    assert failed.gap_reviews[0].decision is CriticDecision.REVISE
    assert not any("Reused a previously successful review" in item
                   for item in failed.gap_reviews[0].caveats)


def test_invalid_evidence_labels_are_omitted_and_mark_review_incomplete():
    client = FakeClient(llm_review(evidence_labels=["P9-E99"]))
    result = CriticAgent(review_mode="auto", client=client).critique(comparison(), gap_result(), validation())
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
    result = CriticAgent(review_mode="auto", client=client).critique(
        comparison(), gap_result(), validation(status=GapValidationStatus.VALIDATION_INCOMPLETE)
    )
    review = result.gap_reviews[0]
    assert review.decision is CriticDecision.REVISE
    assert review.review_status is CriticReviewStatus.INCOMPLETE
    assert any("Evidence validation is incomplete" in caveat for caveat in review.caveats)


def test_omitted_gap_review_becomes_explicitly_incomplete():
    client = FakeClient(_LLMCriticResult(gap_reviews=[]))
    result = CriticAgent(review_mode="auto", client=client).critique(comparison(), gap_result(), validation())
    assert result.review_status is CriticReviewStatus.INCOMPLETE
    assert result.gap_reviews[0].gap_id == "GAP-001"
    assert "omitted" in result.errors[0]


def test_partial_success_across_multiple_gaps_keeps_success_and_marks_failed_gap():
    class ProviderError(Exception):
        code = 503

    first = research_gap()
    second = research_gap().model_copy(update={"gap_id": "GAP-002"})
    gaps = gap_result([first, second])
    evidence = EvidenceValidationResult(
        research_topic="topic", paper_ids=[P1, P2],
        gap_validations=validation("GAP-001").gap_validations
        + validation("GAP-002").gap_validations,
        validation_counts={"FULLY_SUPPORTED": 2}, evidence_status_counts={"SUPPORTED": 4},
    )
    client = SequencedClient([llm_review(), ProviderError("service unavailable")])

    result = CriticAgent(review_mode="auto", client=client, max_retries=1).critique(
        comparison(), gaps, evidence,
    )

    assert len(client.calls) == 2
    assert result.review_status is CriticReviewStatus.INCOMPLETE
    assert [(item.gap_id, item.review_status) for item in result.gap_reviews] == [
        ("GAP-001", CriticReviewStatus.COMPLETE),
        ("GAP-002", CriticReviewStatus.INCOMPLETE),
    ]


def test_no_candidate_gaps_completes_without_gemini():
    client = FakeClient()
    result = CriticAgent(review_mode="auto", client=client).critique(
        comparison(), gap_result([]), evidence_validation=validation()
    )
    assert result.review_status is CriticReviewStatus.COMPLETE
    assert result.gap_reviews == []
    assert not client.calls
