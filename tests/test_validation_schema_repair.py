"""Regression tests for the Phase 4 evidence-validation schema failure.

Live run: GAP-002's joint validation request was rejected by Groq with HTTP 400
``json_validate_failed`` ("missing properties: 'overgeneralization'"), so GAP-002
was recorded VALIDATION_INCOMPLETE and the run stopped; GAP-001 was validated
PARTIALLY_SUPPORTED. All data below is the saved run's (tests/phase4_validation_case.py).
Every test is offline: fake provider clients and an in-memory chunk store.
"""
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from src.agents.artifact_provenance import fingerprint_json
from src.agents.critic_agent import (
    CriticAgent,
    CriticDecision,
    CriticResult,
    CriticReviewStatus,
    GapCriticReview,
    GapOriginAssessment,
)
from src.agents.cross_paper_comparison_agent import EvidenceRef, MultiPaperComparison
from src.agents.evidence_validation_agent import (
    EvidenceValidationAgent,
    EvidenceValidationResult,
    EvidenceValidationStatus as EV,
    GapValidation,
    GapValidationStatus,
    _ChunkAssessment,
    _ContradictionAdjudication,
    _GapEvidenceAssessment,
    _JointAssessment,
)
from src.agents.gap_acceptance import acceptance_blockers
from src.agents.groq_client import GroqAPIError, _status_error
from src.agents.research_gap_detection_agent import ResearchGap, ResearchGapDetectionResult
from src.agents.structured_output import schema_contract, schema_problem
from tests import test_langgraph_workflow as wf
from tests.phase4_validation_case import (
    CHUNKS,
    GAPS,
    PAPER_IDS,
    SAVED_VALIDATIONS,
    SEMANTIC_FAILURE,
    TOPIC,
)

P1, P2 = PAPER_IDS
GROQ_MESSAGE = SEMANTIC_FAILURE.split("joint evidence assessment failed: ", 1)[1]


def gap_detection(*gap_ids):
    gaps = [ResearchGap.model_validate(g) for g in GAPS if not gap_ids or g["gap_id"] in gap_ids]
    return ResearchGapDetectionResult(
        research_topic=TOPIC, paper_count=2, paper_ids=list(PAPER_IDS), gaps=gaps,
        gap_category_counts={}, caveats=[])


class Store:
    def get_chunks_by_ids(self, chunk_ids):
        return SimpleNamespace(chunks=[
            {"chunk_id": cid, "text": CHUNKS[cid][1],
             "metadata": {"paper_id": CHUNKS[cid][0], "chunk_id": cid}}
            for cid in chunk_ids if cid in CHUNKS])


def schema_error():
    """The exact provider error of the live run, as the Groq adapter raises it."""
    return GroqAPIError(GROQ_MESSAGE, status_code=400, error_code="json_validate_failed")


def assessment(gap, statuses, quotes, joint_status, drop=(), **overrides):
    """A gap assessment whose quotes are verbatim chunk text."""
    labels = []
    for number, ref in enumerate(gap.supporting_evidence, start=1):
        labels += [(f"P{number}-E{index}", chunk) for index, chunk in enumerate(ref.chunk_ids, 1)]
    evidence = [{
        "evidence_label": label, "status": status,
        "attributed_contribution": f"Contribution attributed to {label}.",
        "claim_evidence_alignment": f"Alignment for {label}.",
        "supported_contribution": f"Supported part of {label}." if quote else None,
        "supported_contribution_quote": quote,
        "author_stated": False, "limitations": f"Limits of {label}.",
    } for (label, _), status, quote in zip(labels, statuses, quotes)]
    body = {
        "status": joint_status, "complete_gap_assessment": "Assessed against both chunks.",
        "cross_paper_inference": "Inferred across the two papers.",
        "evidence_summary_accuracy": "Partly accurate.",
        "research_opportunity": "Tentative.", "gap_origin": "Inferred across papers.",
        "overgeneralization": "The absence of a released dataset is not stated in the chunks.",
        "limitations": "Two retrieved chunks only.", "counterevidence": "None supplied.",
        "evidence": evidence, "contradictions": [],
    }
    body.update(overrides)
    for key in drop:
        body.pop(key)
    return body


# Assessment the model could legitimately return for GAP-002 (see test docstrings):
# P1 chunk_29 supports "titles and abstracts" only partially; P2 chunk_2 supports
# PDF processing; neither states that no annotated full-text dataset exists.
GAP2_QUOTES = ("mining full text rather than abstracts alone",
               "Research papers in PDF format are processed using PyMuPDF and EasyOCR")


def gap2_partial(**kw):
    return assessment(gap_detection("GAP-002").gaps[0],
                      ["PARTIALLY_SUPPORTED", "PARTIALLY_SUPPORTED"], GAP2_QUOTES,
                      "PARTIALLY_SUPPORTED", **kw)


class Client:
    """Fake provider: a scripted outcome per request (exception or parsed dict)."""

    def __init__(self, outcomes):
        self.outcomes, self.requests = list(outcomes), []
        self.models = SimpleNamespace(generate_content=self.generate)

    def generate(self, *, model, contents, config=None):
        self.requests.append({"contents": contents, "instruction": config.system_instruction})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, SimpleNamespace):
            return outcome
        return SimpleNamespace(parsed=outcome)


def validate(client, *gap_ids, **agent_kwargs):
    agent = EvidenceValidationAgent(vector_store=Store(), client=client, max_retries=3,
                                    base_backoff_seconds=0, max_backoff_seconds=0, **agent_kwargs)
    return agent.validate(gap_detection(*gap_ids))


# ---------------------------------------------------------------------------
# Prompt and schema agree on every required field
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model", [_GapEvidenceAssessment, _JointAssessment, _ChunkAssessment,
                                   _ContradictionAdjudication])
def test_output_contract_names_every_required_field_of_the_schema(model):
    contract = schema_contract(model)
    for name in model.model_json_schema()["required"]:
        assert f"- {name} (" in contract
    for name, definition in model.model_json_schema().get("$defs", {}).items():
        for field in definition.get("required", []):
            assert f"{field}:" in contract


def test_the_request_actually_sent_carries_the_contract_with_overgeneralization():
    client = Client([gap2_partial()])
    validate(client, "GAP-002")
    instruction = client.requests[0]["instruction"]
    assert instruction.startswith(EvidenceValidationAgent.GAP_SYSTEM_INSTRUCTION)
    assert "- overgeneralization (string; required)" in instruction
    assert "Never omit a required key" in instruction


def test_groq_adapter_keeps_the_provider_error_code():
    raw = SimpleNamespace(status_code=400, body={"error": {
        "message": "Generated JSON does not match the expected schema.",
        "code": "json_validate_failed"}})
    error = _status_error(raw)
    assert error.error_code == "json_validate_failed"
    assert error.status_code == 400
    assert schema_problem(error) is not None


# ---------------------------------------------------------------------------
# Exact live failure: missing overgeneralization
# ---------------------------------------------------------------------------

def test_persistent_missing_overgeneralization_stays_incomplete_with_precise_error():
    client = Client([schema_error(), schema_error()])
    result = validate(client, "GAP-002")

    item = result.gap_validations[0]
    assert len(client.requests) == 2                      # one bounded repair, no more
    assert item.status is GapValidationStatus.VALIDATION_INCOMPLETE
    assert item.joint_assessment is None
    for record in item.evidence_validations:
        assert record.status is None                      # never defaulted to SUPPORTED
        assert record.supported_contribution_quote is None
        assert "missing properties: 'overgeneralization'" in record.assessment_error
    failure = result.semantic_failures[0]
    assert failure.startswith("GAP-002: joint evidence assessment failed: _GapEvidenceAssessment "
                              "response failed schema validation after 2 attempt(s)")
    assert failure.count("missing properties: 'overgeneralization'") == 2
    assert acceptance_blockers(item, gap_detection("GAP-002").gaps[0])


def test_repair_request_names_the_missing_field_and_keeps_the_original_request():
    client = Client([schema_error(), gap2_partial()])
    validate(client, "GAP-002")
    first, second = (r["contents"] for r in client.requests)
    assert second.startswith(first)
    assert "rejected by schema validation" in second
    assert "missing properties: 'overgeneralization'" in second
    assert "overgeneralization" in second.split("---")[-1]


def test_repaired_response_is_accepted_only_on_its_own_merits():
    """After repair, GAP-002 is validated from the new response: here it is
    honestly PARTIALLY_SUPPORTED, and stays ineligible for acceptance."""
    client = Client([schema_error(), gap2_partial()])
    result = validate(client, "GAP-002")

    item = result.gap_validations[0]
    assert result.semantic_failures == []
    assert item.status is GapValidationStatus.PARTIALLY_SUPPORTED
    assert [r.status for r in item.evidence_validations] == [EV.PARTIALLY_SUPPORTED] * 2
    # Quotes stored as exact chunk text, with exact IDs.
    for record, quote in zip(item.evidence_validations, GAP2_QUOTES):
        assert record.supported_contribution_quote == quote
        assert quote in CHUNKS[record.chunk_id][1]
    assert [(r.paper_id, r.chunk_id) for r in item.evidence_validations] == [
        (P1, f"{P1}_chunk_29"), (P2, f"{P2}_chunk_2")]
    assert acceptance_blockers(item, gap_detection("GAP-002").gaps[0])


@pytest.mark.parametrize("drop, blank", [
    (("overgeneralization",), None),          # key absent (as in the live failure)
    ((), "overgeneralization"),               # key present but empty
    ((), "status"),
])
def test_locally_invalid_responses_are_repaired_never_filled(drop, blank):
    bad = gap2_partial(drop=drop, **({blank: "  "} if blank else {}))
    client = Client([bad, bad])
    result = validate(client, "GAP-002")
    item = result.gap_validations[0]
    assert len(client.requests) == 2
    assert item.status is GapValidationStatus.VALIDATION_INCOMPLETE
    assert all(r.status is None for r in item.evidence_validations)
    reason = result.semantic_failures[0]
    if drop:
        assert "missing required field 'overgeneralization'" in reason
    else:
        assert f"required text field(s) are blank: {blank}" in reason


def test_missing_evidence_item_field_is_not_defaulted():
    bad = gap2_partial()
    for item in bad["evidence"]:
        item.pop("status")
    result = validate(Client([bad, bad]), "GAP-002")
    assert "missing required field 'evidence.0.status'" in result.semantic_failures[0]
    assert all(r.status is None for r in result.gap_validations[0].evidence_validations)


def test_unparsed_text_response_is_diagnosed_precisely():
    text = json.dumps(gap2_partial(drop=("overgeneralization",)))
    unparsed = SimpleNamespace(parsed=None, text=text)
    result = validate(Client([unparsed, unparsed]), "GAP-002")
    assert "missing required field 'overgeneralization'" in result.semantic_failures[0]


def test_repairs_can_be_disabled():
    client = Client([schema_error()])
    result = validate(client, "GAP-002", max_schema_repairs=0)
    assert len(client.requests) == 1
    assert "after 1 attempt(s)" in result.semantic_failures[0]


def test_negative_repair_budget_is_rejected():
    with pytest.raises(ValueError, match="max_schema_repairs"):
        EvidenceValidationAgent(vector_store=Store(), client=Client([]), max_schema_repairs=-1)


def test_non_schema_client_errors_are_not_repaired():
    client = Client([GroqAPIError("Groq API error (HTTP 401): invalid key", status_code=401)])
    result = validate(client, "GAP-002")
    assert len(client.requests) == 1
    assert "HTTP 401" in result.semantic_failures[0]
    assert "schema validation" not in result.semantic_failures[0]


def test_workflow_stops_safely_when_validation_stays_incomplete():
    agents = wf.workflow_agents(paper_ids=list(PAPER_IDS))
    agents.gap_detection.detect_gaps.return_value = gap_detection()
    agents.evidence_validation = EvidenceValidationAgent(
        vector_store=Store(), max_retries=1,
        client=Client([gap1_saved_like_response(), schema_error(), schema_error()]))
    agents.retrieval.search.return_value = (
        wf.search_plan(), [wf.PaperRecord(title=p, authors=[], year=2026, doi=None, abstract=None,
                                          source="test", source_id=p, landing_url=None,
                                          pdf_url=None) for p in PAPER_IDS])
    result = wf.invoke(agents, pdf_paths={p: "x.pdf" for p in PAPER_IDS})

    assert result["workflow_status"] == "STOPPED"
    assert "Evidence validation was incomplete" in result["stop_reason"]
    statuses = {g.gap_id: g.status for g in result["evidence_validation"].gap_validations}
    assert statuses == {"GAP-001": GapValidationStatus.PARTIALLY_SUPPORTED,
                        "GAP-002": GapValidationStatus.VALIDATION_INCOMPLETE}
    agents.critic.critique.assert_not_called()
    agents.research_questions.generate_questions.assert_not_called()


# ---------------------------------------------------------------------------
# GAP-001: partially supported, must never be accepted
# ---------------------------------------------------------------------------

def gap1_saved_like_response():
    """Mirrors the saved GAP-001 assessment: P1 SUPPORTED, P2 PARTIALLY_SUPPORTED."""
    saved = SAVED_VALIDATIONS["GAP-001"]["evidence_validations"]
    return assessment(gap_detection("GAP-001").gaps[0],
                      [r["status"] for r in saved],
                      [r["supported_contribution_quote"] for r in saved],
                      "PARTIALLY_SUPPORTED")


def test_gap001_revalidates_partially_supported_with_exact_saved_quotes():
    result = validate(Client([gap1_saved_like_response()]), "GAP-001")
    item = result.gap_validations[0]
    saved = GapValidation.model_validate(SAVED_VALIDATIONS["GAP-001"])
    assert item.status is GapValidationStatus.PARTIALLY_SUPPORTED
    assert [(r.paper_id, r.chunk_id, r.status, r.supported_contribution_quote)
            for r in item.evidence_validations] == [
        (r.paper_id, r.chunk_id, r.status, r.supported_contribution_quote)
        for r in saved.evidence_validations]


def test_saved_gap001_record_fails_the_unchanged_acceptance_rule():
    saved = GapValidation.model_validate(SAVED_VALIDATIONS["GAP-001"])
    blockers = " ".join(acceptance_blockers(saved, gap_detection("GAP-001").gaps[0]))
    assert "gap-level validation is PARTIALLY_SUPPORTED" in blockers
    assert f"{P2}/{P2}_chunk_1 is PARTIALLY_SUPPORTED, not SUPPORTED" in blockers


def comparison():
    return MultiPaperComparison(
        paper_ids=list(PAPER_IDS), paper_count=2, dimension_summaries=[],
        recurring_patterns=[], common_limitations=[], conflicting_findings=[],
        methodological_differences=[], underexplored_directions=[], caveats=[])


def test_critic_rejects_saved_gap001_without_a_model_call():
    calls = []
    client = SimpleNamespace(models=SimpleNamespace(
        generate_content=lambda **kw: calls.append(kw)))
    validation = EvidenceValidationResult(
        research_topic=TOPIC, paper_ids=list(PAPER_IDS),
        gap_validations=[GapValidation.model_validate(SAVED_VALIDATIONS["GAP-001"])],
        validation_counts={}, evidence_status_counts={})
    from src.agents.cross_paper_comparison_agent import ComparativeFinding
    compare = comparison().model_copy(update={"recurring_patterns": [ComparativeFinding(
        title="MET-1", description="similar statements", dimensions=["methodology"],
        evidence=[EvidenceRef(paper_id=r["paper_id"], chunk_ids=r["chunk_ids"])
                  for r in GAPS[0]["supporting_evidence"]])]})
    result = CriticAgent(client=client, max_retries=1).critique(
        compare, gap_detection("GAP-001"), validation,
        input_fingerprints={"comparison": "a" * 64, "gap_detection": "b" * 64,
                            "evidence_validation": "c" * 64})
    assert calls == []
    assert result.gap_reviews[0].decision is CriticDecision.REJECT


def test_injected_accept_of_saved_gap001_cannot_reach_questions():
    agents = wf.workflow_agents(paper_ids=list(PAPER_IDS))
    agents.gap_detection.detect_gaps.return_value = gap_detection("GAP-001")
    agents.evidence_validation.validate.return_value = EvidenceValidationResult(
        research_topic=TOPIC, paper_ids=list(PAPER_IDS),
        gap_validations=[GapValidation.model_validate(SAVED_VALIDATIONS["GAP-001"])],
        validation_counts={}, evidence_status_counts={})
    agents.critic.critique.side_effect = lambda *a, **kw: CriticResult(
        research_topic=TOPIC, review_status=CriticReviewStatus.COMPLETE,
        gap_reviews=[GapCriticReview(
            gap_id="GAP-001", decision=CriticDecision.ACCEPT,
            review_status=CriticReviewStatus.COMPLETE,
            origin_assessment=GapOriginAssessment.INFERRED,
            origin_rationale="r", reasoning="Injected ACCEPT.")],
        input_fingerprints=kw["input_fingerprints"])
    agents.retrieval.search.return_value = (
        wf.search_plan(), [wf.PaperRecord(title=p, authors=[], year=2026, doi=None, abstract=None,
                                          source="test", source_id=p, landing_url=None,
                                          pdf_url=None) for p in PAPER_IDS])
    result = wf.invoke(agents, pdf_paths={p: "x.pdf" for p in PAPER_IDS})
    assert result["workflow_status"] == "STOPPED"
    assert "strict acceptance rule" in result["stop_reason"]
    agents.research_questions.generate_questions.assert_not_called()


# ---------------------------------------------------------------------------
# Backward compatibility of the saved records
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gap_id", ["GAP-001", "GAP-002"])
def test_saved_validation_records_round_trip_with_identical_fingerprints(gap_id):
    raw = SAVED_VALIDATIONS[gap_id]
    again = GapValidation.model_validate(raw).model_dump(mode="json")
    assert fingerprint_json(again) == fingerprint_json(raw)


def test_saved_incomplete_gap002_is_still_recognized_as_incomplete():
    saved = GapValidation.model_validate(SAVED_VALIDATIONS["GAP-002"])
    assert saved.status is GapValidationStatus.VALIDATION_INCOMPLETE
    assert saved.joint_assessment is None
    assert all(r.status is None for r in saved.evidence_validations)
    assert "missing properties: 'overgeneralization'" in saved.joint_assessment_error


def test_response_models_still_reject_unknown_keys():
    with pytest.raises(ValidationError):
        _GapEvidenceAssessment.model_validate({**gap2_partial(), "extra": "x"}, strict=True)
