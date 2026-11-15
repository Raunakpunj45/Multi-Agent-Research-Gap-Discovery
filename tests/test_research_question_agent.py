import json
from types import SimpleNamespace

import pytest

from src.agents import (
    CriticDecision,
    CriticResult,
    CriticReviewStatus,
    EvidenceValidation,
    EvidenceRef,
    EvidenceValidationResult,
    EvidenceValidationStatus,
    GapCriticReview,
    GapSupportStatus,
    GapValidation,
    GapValidationStatus,
    GenerationStatus,
    HypothesisStatus,
    JointGapAssessment,
    ResearchGap,
    ResearchGapDetectionResult,
    ResearchQuestionAgent,
)
from src.agents.research_gap_detection_agent import (
    GapCategory,
    GapConfidence,
    GapOrigin,
)
from src.agents.research_question_agent import (
    _LLMHypothesis,
    _LLMQuestionSet,
    _LLMResearchQuestion,
)

P1, P2 = "https://openalex.org/W1", "https://openalex.org/W2"
C1, C2 = f"{P1}_chunk_1", f"{P2}_chunk_1"
P3, C3 = "https://openalex.org/W3", "https://openalex.org/W3_chunk_1"


def gap(gap_id="GAP-001"):
    refs = [EvidenceRef(paper_id=P1, chunk_ids=[C1]),
            EvidenceRef(paper_id=P2, chunk_ids=[C2])]
    return ResearchGap(
        gap_id=gap_id,
        title="Cross-paper evaluation gap",
        description="Two supplied studies identify related evaluation limits.",
        gap_category=GapCategory.EMPIRICAL,
        reasoning="The studies report complementary evaluation findings.",
        supporting_paper_ids=[P1, P2],
        supporting_chunk_ids=[C1, C2],
        supporting_evidence=refs,
        evidence_summary="P1 and P2 report evaluation findings.",
        gap_origin=GapOrigin.INFERRED,
        confidence=GapConfidence.MEDIUM,
        confidence_reason="Two papers contribute evidence.",
        research_opportunity="Test evaluation approaches across both contexts.",
    )


def gap_result(gaps=None):
    gaps = [gap()] if gaps is None else gaps
    return ResearchGapDetectionResult(
        research_topic="scientific discovery",
        paper_count=2,
        paper_ids=[P1, P2],
        gaps=gaps,
        gap_category_counts={"Empirical Gap": len(gaps)},
        caveats=[],
    )


def critic(review_status=CriticReviewStatus.COMPLETE,
           decision=CriticDecision.ACCEPT, gap_id="GAP-001"):
    return CriticResult(
        research_topic="scientific discovery",
        review_status=review_status,
        gap_reviews=[GapCriticReview(
            gap_id=gap_id,
            decision=decision,
            review_status=review_status,
            origin_assessment="INFERRED",
            origin_rationale="The gap is a cross-paper synthesis.",
            reasoning="The bounded gap is supported.",
            caveats=["Bounded to the supplied paper collection."],
        )],
    )


def validation(status=GapValidationStatus.FULLY_SUPPORTED,
               joint_status=GapSupportStatus.FULLY_SUPPORTED,
               item_statuses=None, gap_id="GAP-001"):
    item_statuses = item_statuses or [
        EvidenceValidationStatus.SUPPORTED,
        EvidenceValidationStatus.SUPPORTED,
    ]
    joint = JointGapAssessment(
        status=joint_status,
        complete_gap_assessment="The bounded gap is supported.",
        cross_paper_inference="The two findings motivate a shared question.",
        evidence_summary_accuracy="The summary matches the assessed contributions.",
        research_opportunity="Test across the two settings.",
        gap_origin="Inferred from the two studies.",
        overgeneralization="No claim beyond the supplied evidence.",
        limitations="Only two papers were assessed.",
        counterevidence="No counterevidence was supplied.",
        source_evidence=[EvidenceRef(paper_id=P1, chunk_ids=[C1]),
                         EvidenceRef(paper_id=P2, chunk_ids=[C2])],
    )
    evidence_items = []
    for paper, chunk_id, item_status in zip((P1, P2), (C1, C2), item_statuses):
        evidence_items.append({
            "paper_id": paper,
            "chunk_id": chunk_id,
            "status": item_status,
            "findings": json.dumps({
                "attributed_contribution": (
                    f"{'P1' if paper == P1 else 'P2'} reports an evaluation contribution."
                ),
                "supported_contribution": (
                    f"{'P1' if paper == P1 else 'P2'} reports an evaluation contribution."
                ),
                "limitations": (
                    "The contribution is only partially supported by this chunk."
                    if item_status is EvidenceValidationStatus.PARTIALLY_SUPPORTED
                    else "None."
                ),
                "claim_evidence_alignment": "The chunk supports the attributed contribution.",
            }),
            "claim_evidence_alignment": "The chunk supports the attributed contribution.",
            "evidence_summary_accuracy": "Accurate for this chunk.",
        })
    return EvidenceValidationResult(
        research_topic="scientific discovery",
        paper_ids=[P1, P2],
        gap_validations=[GapValidation(
            gap_id=gap_id,
            status=status,
            explanation="Validation assessment.",
            evidence_validations=evidence_items,
            joint_assessment=joint,
        )],
        validation_counts={status.value: 1},
        evidence_status_counts={"SUPPORTED": 2},
    )


def llm_response(question_labels=None, hypothesis_labels=None,
                 include_hypothesis=True, hypothesis_limitation=None):
    question_labels = ["P1-E1", "P2-E1"] if question_labels is None else question_labels
    hypothesis_labels = ["P1-E1", "P2-E1"] if hypothesis_labels is None else hypothesis_labels
    hypothesis = (_LLMHypothesis(
        alternative_hypothesis="A structured protocol improves reviewer agreement.",
        null_hypothesis="A structured protocol does not change reviewer agreement.",
        rationale="The two validated contributions motivate a measurable comparison.",
        evidence_labels=hypothesis_labels,
        independent_variable="Use of a structured protocol",
        dependent_variable="Reviewer agreement",
        comparator="Standard review without the protocol",
        measurable_outcome="Agreement score on a prespecified scale",
    ) if include_hypothesis else None)
    return _LLMQuestionSet(
        research_questions=[_LLMResearchQuestion(
            question="How does a structured protocol affect reviewer agreement across the two settings?",
            rationale="It tests the shared evaluation concern without claiming a result.",
            evidence_labels=question_labels,
        )],
        hypothesis=hypothesis,
        hypothesis_limitation=hypothesis_limitation,
    )


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


def run(client, critic_result=None, gap_detection=None, evidence_validation=None):
    return ResearchQuestionAgent(client=client, max_retries=1).generate_questions(
        critic_result or critic(),
        gap_detection or gap_result(),
        evidence_validation or validation(),
    )


def test_accepted_gap_generates_questions_hypothesis_and_validation_facts():
    client = FakeClient(llm_response())
    result = run(client)

    assert result.generation_status is GenerationStatus.COMPLETE
    output = result.gap_results[0]
    assert output.hypothesis_status is HypothesisStatus.GENERATED
    assert "A structured protocol" in output.hypothesis.alternative_hypothesis
    assert output.research_questions[0].source_evidence == [
        EvidenceRef(paper_id=P1, chunk_ids=[C1]),
        EvidenceRef(paper_id=P2, chunk_ids=[C2]),
    ]
    assert {fact.source_evidence.paper_id for fact in output.evidence_backed_facts} == {P1, P2}
    assert output.gap_origin == "inferred_cross_paper"
    assert output.critic_caveats == ["Bounded to the supplied paper collection."]
    prompt = client.calls[0]["contents"]
    assert "hypothesis_evidence_sufficient" in prompt
    assert "P1-E1" in prompt and "P2-E1" in prompt
    assert P1 not in prompt and P2 not in prompt


@pytest.mark.parametrize("decision,status", [
    (CriticDecision.REVISE, CriticReviewStatus.COMPLETE),
    (CriticDecision.REJECT, CriticReviewStatus.COMPLETE),
    (CriticDecision.ACCEPT, CriticReviewStatus.INCOMPLETE),
])
def test_nonaccepted_or_incomplete_critic_review_never_calls_model(decision, status):
    client = FakeClient(llm_response())
    result = run(client, critic_result=critic(review_status=status, decision=decision))

    assert result.gap_results == []
    assert result.skipped_gaps[0].gap_id == "GAP-001"
    assert not client.calls


@pytest.mark.parametrize("overall_status,errors", [
    (CriticReviewStatus.INCOMPLETE, []),
    (CriticReviewStatus.COMPLETE, ["Run-level critic error."]),
])
def test_incomplete_overall_critic_result_blocks_complete_accepted_gap(
    overall_status, errors,
):
    client = FakeClient(llm_response())
    critic_result = critic(
        review_status=CriticReviewStatus.COMPLETE,
        decision=CriticDecision.ACCEPT,
    ).model_copy(update={"review_status": overall_status, "errors": errors})

    result = run(client, critic_result=critic_result)

    assert result.generation_status is GenerationStatus.INCOMPLETE
    assert result.gap_results == []
    assert [item.gap_id for item in result.skipped_gaps] == ["GAP-001"]
    assert not client.calls


def test_only_accepted_gaps_are_sent_to_the_model():
    client = FakeClient(llm_response())
    first, second = gap("GAP-001"), gap("GAP-002")
    critic_result = CriticResult(
        research_topic="scientific discovery",
        review_status=CriticReviewStatus.COMPLETE,
        gap_reviews=[
            GapCriticReview(
                gap_id="GAP-001", decision=CriticDecision.ACCEPT,
                review_status=CriticReviewStatus.COMPLETE,
                origin_assessment="INFERRED", origin_rationale="Synthesis.",
                reasoning="Accepted.",
            ),
            GapCriticReview(
                gap_id="GAP-002", decision=CriticDecision.REVISE,
                review_status=CriticReviewStatus.COMPLETE,
                origin_assessment="INFERRED", origin_rationale="Synthesis.",
                reasoning="Needs revision.",
            ),
        ],
    )
    validation_result = EvidenceValidationResult(
        research_topic="scientific discovery", paper_ids=[P1, P2],
            gap_validations=(validation(gap_id="GAP-001").gap_validations
                             + validation(gap_id="GAP-002").gap_validations),
        validation_counts={"FULLY_SUPPORTED": 2},
        evidence_status_counts={"SUPPORTED": 4},
    )

    result = run(
        client,
        critic_result=critic_result,
        gap_detection=gap_result([first, second]),
        evidence_validation=validation_result,
    )

    assert [item.gap_id for item in result.gap_results] == ["GAP-001"]
    assert [item.gap_id for item in result.skipped_gaps] == ["GAP-002"]
    assert len(client.calls) == 1
    assert "GAP-002" not in client.calls[0]["contents"]


def test_successful_gap_results_survive_a_later_gap_api_failure():
    class QuotaError(Exception):
        code = 429

    first, second = gap("GAP-001"), gap("GAP-002")
    critic_result = CriticResult(
        research_topic="scientific discovery",
        review_status=CriticReviewStatus.COMPLETE,
        gap_reviews=[
            GapCriticReview(
                gap_id=gap_id, decision=CriticDecision.ACCEPT,
                review_status=CriticReviewStatus.COMPLETE,
                origin_assessment="INFERRED", origin_rationale="Synthesis.",
                reasoning="Accepted.",
            )
            for gap_id in ("GAP-001", "GAP-002")
        ],
    )
    validation_result = EvidenceValidationResult(
        research_topic="scientific discovery", paper_ids=[P1, P2],
        gap_validations=(validation(gap_id="GAP-001").gap_validations
                         + validation(gap_id="GAP-002").gap_validations),
        validation_counts={"FULLY_SUPPORTED": 2},
        evidence_status_counts={"SUPPORTED": 4},
    )
    client = SequencedClient([llm_response(), QuotaError("quota payload SECRET")])

    result = run(
        client,
        critic_result=critic_result,
        gap_detection=gap_result([first, second]),
        evidence_validation=validation_result,
    )

    assert result.generation_status is GenerationStatus.INCOMPLETE
    assert [item.gap_id for item in result.gap_results] == ["GAP-001"]
    assert [item.gap_id for item in result.skipped_gaps] == ["GAP-002"]
    assert "API quota exhausted (HTTP 429)" in result.errors[0]
    assert "SECRET" not in result.model_dump_json()
    assert any("INCOMPLETE PARTIAL GENERATION" in caveat for caveat in result.caveats)


def test_three_gap_run_withholds_quota_failed_gap_and_marks_output_nonfinal():
    class QuotaError(Exception):
        code = 429

    gaps = [gap(f"GAP-00{i}") for i in range(1, 4)]
    critic_result = CriticResult(
        research_topic="scientific discovery",
        review_status=CriticReviewStatus.COMPLETE,
        gap_reviews=[GapCriticReview(
            gap_id=item.gap_id, decision=CriticDecision.ACCEPT,
            review_status=CriticReviewStatus.COMPLETE,
            origin_assessment="INFERRED", origin_rationale="Synthesis.",
            reasoning="Accepted.",
        ) for item in gaps],
    )
    validation_result = EvidenceValidationResult(
        research_topic="scientific discovery", paper_ids=[P1, P2],
        gap_validations=sum(
            (validation(gap_id=item.gap_id).gap_validations for item in gaps), []
        ),
        validation_counts={"FULLY_SUPPORTED": 3},
        evidence_status_counts={"SUPPORTED": 6},
    )
    client = SequencedClient([llm_response(), llm_response(), QuotaError("quota")])
    result = run(
        client, critic_result=critic_result, gap_detection=gap_result(gaps),
        evidence_validation=validation_result,
    )

    assert result.generation_status is GenerationStatus.INCOMPLETE
    assert [item.gap_id for item in result.gap_results] == ["GAP-001", "GAP-002"]
    assert [item.gap_id for item in result.skipped_gaps] == ["GAP-003"]
    assert not any(item.gap_id == "GAP-003" for item in result.gap_results)
    assert "API quota exhausted (HTTP 429)" in result.errors[0]
    assert any("not a final report" in caveat for caveat in result.caveats)


def test_partial_validation_allows_questions_but_explicitly_withholds_hypothesis():
    client = FakeClient(llm_response(include_hypothesis=False))
    result = run(client, evidence_validation=validation(
        status=GapValidationStatus.PARTIALLY_SUPPORTED,
        joint_status=GapSupportStatus.PARTIALLY_SUPPORTED,
    ))

    output = result.gap_results[0]
    assert output.research_questions
    assert output.hypothesis is None
    assert output.hypothesis_status is HypothesisStatus.INSUFFICIENT_EVIDENCE
    assert "cannot be generated" in output.hypothesis_limitation


def test_multiple_chunks_keep_their_own_status_and_limitations():
    duplicate_chunk = f"{P1}_chunk_2"
    candidate_gap = gap()
    candidate_gap.supporting_evidence[0].chunk_ids.append(duplicate_chunk)
    candidate_gap.supporting_chunk_ids.append(duplicate_chunk)
    validated = validation(
        status=GapValidationStatus.PARTIALLY_SUPPORTED,
        joint_status=GapSupportStatus.PARTIALLY_SUPPORTED,
        item_statuses=[EvidenceValidationStatus.PARTIALLY_SUPPORTED,
                       EvidenceValidationStatus.SUPPORTED],
    )
    first_finding = {
        "attributed_contribution": (
            "The study uses a manually collected, richly annotated paper dataset and "
            "raises a question about transferability across disciplines."
        ),
        "supported_contribution": "The study reports a dataset.",
        "limitations": "This chunk does not verify the number of disciplines.",
    }
    second_finding = {
        "attributed_contribution": (
            "The study uses a manually collected richly annotated paper set and raises "
            "a question about transferability across disciplines."
        ),
        "supported_contribution": "The study reports a dataset.",
        "limitations": "None.",
    }
    validated.gap_validations[0].evidence_validations[0].findings = json.dumps(first_finding)
    validated.gap_validations[0].evidence_validations.append(EvidenceValidation(
        paper_id=P1,
        chunk_id=duplicate_chunk,
        status=EvidenceValidationStatus.SUPPORTED,
        findings=json.dumps(second_finding),
    ))
    validated.gap_validations[0].joint_assessment.source_evidence[0].chunk_ids.append(
        duplicate_chunk
    )
    response = llm_response(
        question_labels=["P1-E1", "P1-E2"], include_hypothesis=False
    )
    result = run(
        FakeClient(response),
        gap_detection=gap_result([candidate_gap]),
        evidence_validation=validated,
    )

    output = result.gap_results[0]
    paper_one_facts = [fact for fact in output.evidence_backed_facts
                       if fact.source_evidence.paper_id == P1]
    assert len(paper_one_facts) == 2
    by_chunk = {fact.source_evidence.chunk_ids[0]: fact for fact in paper_one_facts}
    assert by_chunk[C1].validation_status is EvidenceValidationStatus.PARTIALLY_SUPPORTED
    assert by_chunk[duplicate_chunk].validation_status is EvidenceValidationStatus.SUPPORTED
    assert len(by_chunk[C1].validation_limitations) == 1
    rationale = output.research_questions[0].rationale
    assert "PARTIALLY_SUPPORTED" in rationale
    assert "does not verify the number of disciplines" in rationale
    assert "tests the shared evaluation concern" not in rationale


def test_model_hypothesis_is_discarded_when_validation_is_insufficient():
    client = FakeClient(llm_response(include_hypothesis=True))
    result = run(client, evidence_validation=validation(
        status=GapValidationStatus.PARTIALLY_SUPPORTED,
        joint_status=GapSupportStatus.PARTIALLY_SUPPORTED,
    ))

    output = result.gap_results[0]
    assert output.hypothesis is None
    assert output.hypothesis_status is HypothesisStatus.INSUFFICIENT_EVIDENCE


def test_hypothesis_must_cite_only_supported_evidence():
    client = FakeClient(llm_response(hypothesis_labels=["P1-E1", "P3-E1"]))
    third_ref = EvidenceRef(paper_id=P3, chunk_ids=[C3])
    gap_with_third_paper = gap()
    gap_with_third_paper.supporting_paper_ids.append(P3)
    gap_with_third_paper.supporting_chunk_ids.append(C3)
    gap_with_third_paper.supporting_evidence.append(third_ref)
    validated = validation()
    validated.gap_validations[0].evidence_validations.append(EvidenceValidation(
        paper_id=P3,
        chunk_id=C3,
        status=EvidenceValidationStatus.PARTIALLY_SUPPORTED,
        findings=json.dumps({"attributed_contribution": "P3 provides partial support."}),
    ))
    validated.gap_validations[0].joint_assessment.source_evidence.append(third_ref)
    validated.paper_ids.append(P3)
    gap_detection = gap_result([gap_with_third_paper])
    gap_detection.paper_ids.append(P3)
    gap_detection.paper_count = 3
    result = run(
        client,
        gap_detection=gap_detection,
        evidence_validation=validated,
    )

    assert result.generation_status is GenerationStatus.COMPLETE
    assert result.gap_results[0].hypothesis is None
    assert result.gap_results[0].hypothesis_status is HypothesisStatus.INSUFFICIENT_EVIDENCE


def test_invalid_question_citation_fails_closed():
    client = FakeClient(llm_response(question_labels=["P9-E99"]))
    result = run(client)

    assert result.generation_status is GenerationStatus.INCOMPLETE
    assert result.gap_results == []
    assert "invalid or missing evidence label" in result.errors[0]


def test_invalid_hypothesis_citation_fails_closed():
    client = FakeClient(llm_response(hypothesis_labels=["P1-E1", "P9-E99"]))
    result = run(client)

    assert result.generation_status is GenerationStatus.INCOMPLETE
    assert result.gap_results == []
    assert "hypothesis cited invalid" in result.errors[0]


def test_validation_reference_mismatch_prevents_generation():
    client = FakeClient(llm_response())
    bad_validation = validation()
    bad_validation.gap_validations[0].evidence_validations.pop()
    result = run(client, evidence_validation=bad_validation)

    assert result.generation_status is GenerationStatus.INCOMPLETE
    assert any("references do not match" in error for error in result.errors)
    assert not client.calls


def test_malformed_validation_findings_fail_safely():
    client = FakeClient(llm_response())
    malformed = validation()
    malformed.gap_validations[0].evidence_validations[0].findings = "[]"
    result = run(client, evidence_validation=malformed)

    assert result.generation_status is GenerationStatus.COMPLETE
    assert len(result.gap_results[0].evidence_backed_facts) == 1
    assert result.gap_results[0].evidence_backed_facts[0].source_evidence.paper_id == P2
    assert len(client.calls) == 1


def test_missing_joint_assessment_prevents_generation():
    client = FakeClient(llm_response())
    missing_joint = validation()
    missing_joint.gap_validations[0].joint_assessment = None
    result = run(client, evidence_validation=missing_joint)

    assert result.generation_status is GenerationStatus.INCOMPLETE
    assert result.gap_results == []
    assert any("joint evidence assessment is missing" in error for error in result.errors)
    assert not client.calls


def test_stale_or_unfingerprinted_critic_review_is_rejected_before_generation():
    current = {
        "comparison": "a" * 64,
        "gap_detection": "b" * 64,
        "evidence_validation": "c" * 64,
    }
    stale = dict(current, evidence_validation="d" * 64)
    critic_result = critic().model_copy(update={"input_fingerprints": stale})
    client = FakeClient(llm_response())

    result = ResearchQuestionAgent(client=client).generate_questions(
        critic_result, gap_result(), validation(),
        critic_input_fingerprints=current,
    )

    assert result.generation_status is GenerationStatus.INCOMPLETE
    assert result.gap_results == []
    assert "fingerprints do not match" in result.errors[0]
    assert not client.calls


def test_unsupported_gap_is_skipped_without_calling_model():
    client = FakeClient(llm_response())
    result = run(client, evidence_validation=validation(
        status=GapValidationStatus.UNSUPPORTED,
        joint_status=GapSupportStatus.UNSUPPORTED,
        item_statuses=[EvidenceValidationStatus.UNSUPPORTED,
                       EvidenceValidationStatus.UNSUPPORTED],
    ))

    assert result.gap_results == []
    assert "does not support" in result.skipped_gaps[0].reason
    assert not client.calls


def test_missing_input_returns_incomplete_and_does_not_call_model():
    client = FakeClient(llm_response())
    result = ResearchQuestionAgent(client=client).generate_questions(None, gap_result(), validation())

    assert result.generation_status is GenerationStatus.INCOMPLETE
    assert result.missing_inputs == ["critic review output"]
    assert result.skipped_gaps[0].gap_id == "GAP-001"
    assert not client.calls


def test_provider_error_is_sanitized_and_keeps_gap_incomplete():
    client = FakeClient(error=RuntimeError(
        "429 RESOURCE_EXHAUSTED quota payload SECRET retryAfter=999s"
    ))
    result = run(client)

    assert result.generation_status is GenerationStatus.INCOMPLETE
    assert "API quota exhausted (HTTP 429)" in result.errors[0]
    serialized = result.model_dump_json()
    assert "SECRET" not in serialized
    assert "999s" not in serialized


def test_other_api_errors_are_distinguished_from_quota_failures():
    class ProviderError(Exception):
        code = 503

    client = FakeClient(error=ProviderError("provider detail SECRET payload"))
    result = run(client)

    assert result.generation_status is GenerationStatus.INCOMPLETE
    assert "provider API service error (HTTP 503)" in result.errors[0]
    assert "quota exhausted" not in result.errors[0]
    assert "SECRET" not in result.model_dump_json()


def test_no_claim_of_global_novelty_is_made():
    result = run(FakeClient(llm_response()))
    assert "do not establish global novelty" in result.caveats[0]


def test_hypothesis_without_test_dimensions_is_withheld():
    response = llm_response()
    response.hypothesis.independent_variable = None
    result = run(FakeClient(response))
    output = result.gap_results[0]
    assert output.hypothesis is None
    assert output.hypothesis_status is HypothesisStatus.INSUFFICIENT_EVIDENCE
    assert "measurable outcome" in output.hypothesis_limitation


def test_hypothesis_is_explicitly_a_proposal_not_an_evidence_finding():
    output = run(FakeClient(llm_response())).gap_results[0].hypothesis
    assert output.alternative_hypothesis.startswith("Proposed test:")
    assert output.null_hypothesis.startswith("Proposed null test:")
    assert "does not establish the predicted outcome" in output.rationale
    assert output.independent_variable and output.dependent_variable
    assert output.comparator and output.measurable_outcome
    assert output.narrower_subgap.startswith("Narrower sub-gap tested:")


def test_partial_chunk_does_not_inherit_other_chunks_limitation():
    duplicate_chunk = f"{P1}_chunk_2"
    candidate_gap = gap()
    candidate_gap.supporting_evidence[0].chunk_ids.append(duplicate_chunk)
    candidate_gap.supporting_chunk_ids.append(duplicate_chunk)
    validated = validation(item_statuses=[EvidenceValidationStatus.PARTIALLY_SUPPORTED,
                                          EvidenceValidationStatus.SUPPORTED])
    validated.gap_validations[0].evidence_validations[0].findings = json.dumps({
        "attributed_contribution": "P1 reports a limited observation.",
        "supported_contribution": "P1 reports a limited observation.",
        "limitations": "This chunk omits the sample size.",
    })
    validated.gap_validations[0].evidence_validations.append(EvidenceValidation(
        paper_id=P1, chunk_id=duplicate_chunk, status=EvidenceValidationStatus.SUPPORTED,
        findings=json.dumps({"attributed_contribution": "P1 reports an unrelated method."}),
    ))
    validated.gap_validations[0].joint_assessment.source_evidence[0].chunk_ids.append(duplicate_chunk)
    response = llm_response(question_labels=["P1-E1", "P1-E2"], include_hypothesis=False)
    result = run(FakeClient(response), gap_detection=gap_result([candidate_gap]),
                 evidence_validation=validated)
    rationale = result.gap_results[0].research_questions[0].rationale
    assert "This chunk omits the sample size" in rationale
    assert "the cited paper reports an unrelated method" in rationale
    assert rationale.count("PARTIALLY_SUPPORTED") == 1


def test_rationale_uses_only_exact_cited_chunk_contributions():
    result = run(FakeClient(llm_response(question_labels=["P1-E1"], include_hypothesis=False)))
    rationale = result.gap_results[0].research_questions[0].rationale
    assert "the cited paper reports an evaluation contribution" in rationale
    assert "P2-E1" not in rationale and "P1-E1" not in rationale


def test_hypothesis_generation_with_partial_evidence_is_withheld_even_if_model_returns_one():
    result = run(FakeClient(llm_response()), evidence_validation=validation(
        item_statuses=[EvidenceValidationStatus.PARTIALLY_SUPPORTED,
                       EvidenceValidationStatus.SUPPORTED]))
    assert result.gap_results[0].hypothesis is None
    assert result.gap_results[0].hypothesis_status is HypothesisStatus.INSUFFICIENT_EVIDENCE


def test_partial_chunk_that_explicitly_lacks_llm_evidence_is_not_used_for_llm_claim():
    partial = validation(item_statuses=[EvidenceValidationStatus.SUPPORTED,
                                        EvidenceValidationStatus.PARTIALLY_SUPPORTED])
    partial.gap_validations[0].evidence_validations[0].findings = json.dumps({
        "attributed_contribution": "P1 supports causal reasoning for LLM systems.",
        "supported_contribution": "P1 supports causal reasoning for LLM systems.",
        "mentions_llm": True,
        "mentions_faithfulness": False,
        "supports_causal_reasoning": True,
    })
    partial.gap_validations[0].evidence_validations[1].findings = json.dumps({
        "attributed_contribution": (
            "P2 discusses LLM faithfulness and general machine-learning causal reasoning."
        ),
        "limitations": "This specific chunk does not mention LLMs or faithfulness concerns.",
    })
    response = llm_response(question_labels=["P1-E1", "P2-E1"], include_hypothesis=False)
    response.research_questions[0].question = (
        "How should causal reasoning be embedded in LLM-assisted scientific workflows?"
    )
    result = run(FakeClient(response), evidence_validation=partial)
    question = result.gap_results[0].research_questions[0]
    assert question.source_evidence == [EvidenceRef(paper_id=P1, chunk_ids=[C1])]
    assert "P2" not in question.rationale and "P2-E1" not in question.rationale


def test_question_is_withheld_when_only_citation_conflicts_with_its_llm_claim():
    partial = validation(item_statuses=[EvidenceValidationStatus.SUPPORTED,
                                        EvidenceValidationStatus.PARTIALLY_SUPPORTED])
    partial.gap_validations[0].evidence_validations[0].findings = json.dumps({
        "attributed_contribution": "P1 discusses general machine-learning methods.",
        "limitations": "This chunk does not mention LLMs.",
    })
    response = llm_response(question_labels=["P1-E1"], include_hypothesis=False)
    response.research_questions[0].question = "How does this method affect LLM faithfulness?"
    result = run(FakeClient(response), evidence_validation=partial)
    assert result.gap_results == []
    assert any("No research questions remained" in error for error in result.errors)


@pytest.mark.parametrize("limitation", [
    "This chunk does not mention LLMs.",
    "LLMs are not discussed in this passage.",
    "There is no evidence about large language models here.",
    "The text does not address faithfulness of self-explanations.",
])
def test_limitation_variants_detect_explicit_subject_absence(limitation):
    subject = "mentions_faithfulness" if "faithfulness" in limitation else "mentions_llm"
    assert ResearchQuestionAgent._limitation_conflicts_with_subject(limitation, subject)


@pytest.mark.parametrize("flag_value", [None, False])
def test_llm_fact_is_withheld_when_subject_flag_is_missing_or_false(flag_value):
    item = validation().gap_validations[0].evidence_validations[0]
    item.findings = json.dumps({
        "attributed_contribution": "LLM self-explanations raise faithfulness concerns.",
        "supported_contribution": "LLM self-explanations raise faithfulness concerns.",
        "mentions_llm": flag_value,
        "mentions_faithfulness": True,
    })
    labels = {"P1-E1": (P1, C1, EvidenceValidationStatus.SUPPORTED)}
    assert ResearchQuestionAgent._validated_facts([item], labels) == []


def test_none_identified_is_not_published_as_a_limitation():
    for text in ("None identified.", "None identified within this chunk.",
                 "No limitations identified for this contribution."):
        assert ResearchQuestionAgent._clean_limitations(text) == []


def test_hypothesis_citing_only_one_paper_is_withheld():
    response = llm_response(hypothesis_labels=["P1-E1"])
    result = run(FakeClient(response))
    output = result.gap_results[0]
    assert output.hypothesis is None
    assert output.hypothesis_status is HypothesisStatus.INSUFFICIENT_EVIDENCE
    assert "two-paper minimum" in output.hypothesis_limitation


def test_lowercase_evidence_labels_are_normalized():
    response = llm_response(question_labels=["p1-e1", "p2-e1"],
                            hypothesis_labels=["p1-e1", "p2-e1"])
    result = run(FakeClient(response))
    output = result.gap_results[0]
    assert output.research_questions[0].source_evidence == [
        EvidenceRef(paper_id=P1, chunk_ids=[C1]),
        EvidenceRef(paper_id=P2, chunk_ids=[C2]),
    ]
    assert output.hypothesis_status is HypothesisStatus.GENERATED


def test_duplicate_and_near_duplicate_questions_are_removed():
    response = llm_response(include_hypothesis=False)
    response.research_questions.append(_LLMResearchQuestion(
        question="How does a structured protocol affect reviewer agreement across the two settings?",
        rationale="Duplicate phrasing.", evidence_labels=["P1-E1"],
    ))
    response.research_questions.append(_LLMResearchQuestion(
        question="How does a structured protocol affect reviewer agreement in both settings?",
        rationale="Near duplicate phrasing.", evidence_labels=["P2-E1"],
    ))
    output = run(FakeClient(response)).gap_results[0]
    assert len(output.research_questions) == 1


def test_same_chunk_semantic_duplicates_are_narrowed_then_deduplicated():
    assessed = validation()
    assessed.gap_validations[0].evidence_validations[0].findings = json.dumps({
        "attributed_contribution": (
            "P1 calls for advances in physics-infused ML and causal reasoning in AI for Science."
        ),
        "supported_contribution": (
            "P1 calls for advances in physics-infused ML and causal reasoning in AI for Science."
        ),
        "mentions_llm": True,
        "supports_physics_informed_ml": True,
        "supports_causal_reasoning": True,
    })
    response = llm_response(question_labels=["P1-E1"], include_hypothesis=False)
    response.research_questions = [
        _LLMResearchQuestion(
            question=(
                "How can physics-informed constraints and causal reasoning frameworks be "
                "integrated into LLM-assisted scientific discovery workflows?"
            ), rationale="Joint direction.", evidence_labels=["P1-E1"],
        ),
        _LLMResearchQuestion(
            question=(
                "What framework effectively combines physics-infused machine learning "
                "with causal reasoning for AI-driven scientific workflows?"
            ), rationale="Same direction.", evidence_labels=["P1-E1"],
        ),
    ]

    output = run(FakeClient(response), evidence_validation=assessed).gap_results[0]
    assert len(output.research_questions) == 1
    assert output.research_questions[0].question == (
        "How might physics-infused machine learning and causal reasoning be jointly "
        "investigated to advance AI for Science?"
    )
    assert output.hypothesis is None
    assert output.hypothesis_status is HypothesisStatus.INSUFFICIENT_EVIDENCE


def test_all_malformed_findings_withhold_questions_before_model_call():
    malformed = validation()
    for item in malformed.gap_validations[0].evidence_validations:
        item.findings = "not-json"
    client = FakeClient(llm_response())
    result = run(client, evidence_validation=malformed)
    assert result.gap_results == []
    assert not client.calls


def test_mocked_generation_is_byte_reproducible_with_stable_hypothesis_format():
    first = run(FakeClient(llm_response())).model_dump_json(indent=2)
    second = run(FakeClient(llm_response())).model_dump_json(indent=2)
    assert first == second
    data = json.loads(first)
    item = data["gap_results"][0]
    assert item["hypothesis"]["alternative_hypothesis"].startswith("Proposed test: ")
    assert item["hypothesis"]["null_hypothesis"].startswith("Proposed null test: ")
    assert item["hypothesis"]["rationale"].startswith(
        "The cited evidence motivates this proposed test; it does not establish the predicted outcome. "
    )
    assert item["hypothesis"]["narrower_subgap"].startswith("Narrower sub-gap tested:")
    assert item["gap_origin"] == "inferred_cross_paper"
    assert item["critic_caveats"] == ["Bounded to the supplied paper collection."]


def test_gap_001_survey_question_drops_irrelevant_dataset_citation():
    assessed = validation(item_statuses=[EvidenceValidationStatus.SUPPORTED,
                                         EvidenceValidationStatus.PARTIALLY_SUPPORTED])
    assessed.gap_validations[0].evidence_validations[0].findings = json.dumps({
        "attributed_contribution": "P1 leaves social-science LLM work outside its survey scope.",
        "supported_contribution": "P1 leaves social-science LLM work outside its survey scope.",
        "mentions_llm": True,
        "supports_social_science_survey_scope": True,
    })
    assessed.gap_validations[0].evidence_validations[1].findings = json.dumps({
        "attributed_contribution": "P2 presents an annotated social-science dataset.",
        "limitations": "This chunk does not verify the claimed cross-discipline scope.",
    })
    response = llm_response(question_labels=["P1-E1", "P2-E1"], include_hypothesis=False)
    response.research_questions[0].question = (
        "What social-science survey coverage is omitted from this LLM survey?"
    )
    output = run(FakeClient(response), evidence_validation=assessed).gap_results[0]
    assert output.research_questions[0].source_evidence == [
        EvidenceRef(paper_id=P1, chunk_ids=[C1])
    ]
    assert "P2" not in output.research_questions[0].rationale


def test_gap_001_social_science_performance_comparison_is_narrowed_to_supported_scope():
    assessed = validation(item_statuses=[EvidenceValidationStatus.SUPPORTED,
                                         EvidenceValidationStatus.PARTIALLY_SUPPORTED])
    assessed.gap_validations[0].evidence_validations[0].findings = json.dumps({
        "attributed_contribution": (
            "P1 leaves social-science LLM efforts outside its survey scope."
        ),
        "supported_contribution": (
            "P1 leaves social-science LLM efforts outside its survey scope."
        ),
        "mentions_llm": True,
        "supports_social_science_survey_scope": True,
    })
    response = llm_response(question_labels=["P1-E1"], include_hypothesis=False)
    response.research_questions[0].question = (
        "What evaluation frameworks compare LLM performance in social-science research "
        "with natural-science domains?"
    )

    question = run(FakeClient(response), evidence_validation=assessed).gap_results[0].research_questions[0]
    assert "performance" not in question.question.casefold()
    assert "compare" not in question.question.casefold()
    assert "leaves for future work" in question.question
    assert question.source_evidence == [EvidenceRef(paper_id=P1, chunk_ids=[C1])]


def test_gap_002_chunk_29_is_excluded_from_facts_questions_and_hypothesis():
    assessed = validation(item_statuses=[EvidenceValidationStatus.SUPPORTED,
                                         EvidenceValidationStatus.PARTIALLY_SUPPORTED])
    assessed.gap_validations[0].evidence_validations[0].findings = json.dumps({
        "attributed_contribution": "P1 calls for physics-informed ML and causal reasoning for AI science.",
        "supported_contribution": "P1 calls for physics-informed ML and causal reasoning for AI science.",
        "mentions_llm": True,
        "supports_physics_informed_ml": True,
        "supports_causal_reasoning": True,
    })
    assessed.gap_validations[0].evidence_validations[1].findings = json.dumps({
        "attributed_contribution": "P2 reports LLM faithfulness concerns and discusses causal reasoning.",
        "limitations": "This specific chunk does not mention LLMs or faithfulness concerns.",
        "mentions_llm": False,
        "mentions_faithfulness": False,
        "supports_causal_reasoning": True,
    })
    response = llm_response(
        question_labels=["P1-E1", "P2-E1"],
        hypothesis_labels=["P1-E1", "P2-E1"],
    )
    response.research_questions[0].question = (
        "How can causal reasoning be used in LLM-assisted scientific discovery?"
    )
    result = run(FakeClient(response), evidence_validation=assessed)
    output = result.gap_results[0]
    assert all(C2 not in fact.source_evidence.chunk_ids
               for fact in output.evidence_backed_facts)
    assert all(C2 not in ref.chunk_ids for question in output.research_questions
               for ref in question.source_evidence)
    assert output.hypothesis is None
    assert output.hypothesis_status is HypothesisStatus.INSUFFICIENT_EVIDENCE


def test_gap_002_integration_question_keeps_chunk_contributions_separate():
    assessed = validation()
    assessed.gap_validations[0].evidence_validations[0].findings = json.dumps({
        "attributed_contribution": "P1 calls for physics-informed ML and causal reasoning in AI for science.",
        "supported_contribution": "P1 calls for physics-informed ML and causal reasoning in AI for science.",
        "mentions_llm": True,
        "supports_physics_informed_ml": True,
        "supports_causal_reasoning": True,
    })
    assessed.gap_validations[0].evidence_validations[1].findings = json.dumps({
        "attributed_contribution": "P2 reports faithfulness concerns in LLM self-explanations and discusses domain knowledge injection.",
        "supported_contribution": "P2 reports faithfulness concerns in LLM self-explanations and discusses domain knowledge injection.",
        "mentions_llm": True,
        "mentions_faithfulness": True,
        "supports_domain_knowledge_injection": True,
    })
    response = llm_response(question_labels=["P1-E1", "P2-E1"], include_hypothesis=False)
    response.research_questions[0].question = (
        "What evaluation framework can study faithfulness of LLM explanations in "
        "workflows using physics-informed methods?"
    )
    question = run(FakeClient(response), evidence_validation=assessed).gap_results[0].research_questions[0]
    assert question.source_evidence == [
        EvidenceRef(paper_id=P1, chunk_ids=[C1]),
        EvidenceRef(paper_id=P2, chunk_ids=[C2]),
    ]
    assert "the cited paper calls for physics-informed ML" in question.rationale
    assert "the cited paper reports faithfulness concerns" in question.rationale
    assert "combined physics-informed and faithfulness direction is proposed" in question.rationale


def test_gap_003_intervention_is_explicitly_proposed_not_established():
    assessed = validation()
    assessed.gap_validations[0].evidence_validations[0].findings = json.dumps({
        "attributed_contribution": "P1 reports peer-review weaknesses and homogenized critiques from LLMs.",
        "supported_contribution": "P1 reports peer-review weaknesses and homogenized critiques from LLMs.",
        "mentions_llm": True,
        "supports_peer_review_quality_assessment": True,
        "supports_critique_homogenization": True,
    })
    response = llm_response(question_labels=["P1-E1"], include_hypothesis=False)
    response.research_questions[0].question = (
        "Could domain-specific rubrics be tested to reduce LLM peer-review critique homogenization?"
    )
    rationale = run(FakeClient(response), evidence_validation=assessed).gap_results[0].research_questions[0].rationale
    assert "proposed direction to test" in rationale
    assert "not a finding established" in rationale
