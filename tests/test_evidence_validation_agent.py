import json
import hashlib
from types import SimpleNamespace
import sys

import pytest
from google.genai import types

from src.agents import (
    EvidenceRef, EvidenceValidationAgent, EvidenceValidationStatus,
    GapSupportStatus, GapValidationStatus, ResearchGapDetectionResult,
    MultiPaperComparison,
)
from src.agents.artifact_provenance import ArtifactProvenance
from src.agents.evidence_validation_agent import (
    EvidenceValidationResult, _ChunkAssessment, _ContradictionAdjudication, _JointAssessment,
    _ProposedContradiction, _gemini_compatible_schema,
)
from src.agents.research_gap_detection_agent import (
    GapCategory, GapConfidence, GapOrigin, ResearchGap,
)


def gap(refs=None, gap_id="GAP-001"):
    refs = refs or [EvidenceRef(paper_id="paper-A", chunk_ids=["a1"]),
                    EvidenceRef(paper_id="paper-B", chunk_ids=["b1"])]
    return ResearchGap(gap_id=gap_id, title="Evaluation gap",
        description="No shared evaluation protocol is established.",
        gap_category=GapCategory.METHODOLOGICAL, reasoning="Both papers show limits.",
        supporting_paper_ids=[r.paper_id for r in refs],
        supporting_chunk_ids=[c for r in refs for c in r.chunk_ids],
        supporting_evidence=refs, evidence_summary="P1 reports a small evaluation; P2 reports a larger evaluation.",
        gap_origin=GapOrigin.INFERRED, confidence=GapConfidence.MEDIUM,
        confidence_reason="Two papers", research_opportunity="Develop a protocol.")


def input_result(gaps=None):
    return ResearchGapDetectionResult(research_topic="topic", paper_count=2,
        paper_ids=["paper-A", "paper-B"], gaps=[gap()] if gaps is None else gaps,
        gap_category_counts={"Methodological Gap": 1}, caveats=[])


def lineaged_gap_input(detection, tmp_path):
    comparison_path = tmp_path / "comparison.json"
    comparison = MultiPaperComparison(
        paper_ids=["paper-A", "paper-B"], paper_count=2,
        dimension_summaries=[], recurring_patterns=[], common_limitations=[],
        conflicting_findings=[], methodological_differences=[],
        underexplored_directions=[], caveats=[],
        provenance=ArtifactProvenance(
            stage="cross_paper_comparison",
            input_fingerprints={"paper_analyses": "a" * 64},
        ),
    )
    comparison_bytes = comparison.model_dump_json().encode("utf-8")
    comparison_path.write_bytes(comparison_bytes)
    detection = detection.model_copy(update={"provenance": ArtifactProvenance(
        stage="research_gap_detection",
        input_fingerprints={"comparison": hashlib.sha256(comparison_bytes).hexdigest()},
    )})
    gap_bytes = detection.model_dump_json(indent=2).encode("utf-8")
    input_path = tmp_path / "gaps.json"
    input_path.write_bytes(gap_bytes)
    return input_path, comparison_path, hashlib.sha256(gap_bytes).hexdigest()


def schema_name(schema):
    return schema["title"]


class Store:
    def __init__(self, records=None, fail=False):
        self.records = records if records is not None else {
            "a1": {"chunk_id": "a1", "text": "The method improved accuracy.", "metadata": {"paper_id": "paper-A", "chunk_id": "a1"}},
            "b1": {"chunk_id": "b1", "text": "The method did not improve accuracy.", "metadata": {"paper_id": "paper-B", "chunk_id": "b1"}},
        }
        self.fail = fail
        self.calls = []

    def get_chunks_by_ids(self, ids):
        self.calls.extend(ids)
        if self.fail:
            raise RuntimeError("store unavailable")
        return SimpleNamespace(chunks=[self.records[i] for i in ids if i in self.records],
                               missing_ids=[i for i in ids if i not in self.records])


class Client:
    def __init__(self, chunk_statuses=None, joint_status=GapSupportStatus.PARTIALLY_SUPPORTED,
                 contradictions=None, fail_schema=None, adjudication=None,
                 fail_error="simulated model failure"):
        self.chunk_statuses = iter(chunk_statuses or ["SUPPORTED", "PARTIALLY_SUPPORTED"])
        self.joint_status = joint_status
        self.contradictions = contradictions or []
        self.fail_schema = fail_schema
        self.fail_error = fail_error
        self.adjudication = adjudication or _ContradictionAdjudication(
            same_subject=True, substantively_incompatible=True,
            confidence="HIGH", explanation="The quoted positions directly conflict.")
        self.prompts = []
        self.models = SimpleNamespace(generate_content=self.generate)

    def generate(self, **kwargs):
        schema = kwargs["config"].response_json_schema
        self.prompts.append((schema, kwargs["contents"]))
        if schema_name(schema) == getattr(self.fail_schema, "__name__", None):
            raise RuntimeError(self.fail_error)
        if schema_name(schema) == _ContradictionAdjudication.__name__:
            return SimpleNamespace(parsed=self.adjudication)
        if schema_name(schema) == _ChunkAssessment.__name__:
            return SimpleNamespace(parsed=_ChunkAssessment(
                status=next(self.chunk_statuses),
                attributed_contribution="This paper contributes a specific evaluation finding.",
                claim_evidence_alignment="Supports this paper's contribution.",
                evidence_summary_accuracy="Accurate for this paper.",
                limitations="Does not independently establish the cross-paper gap.",
                explanation="Assessment is limited to this paper's contribution."))
        return SimpleNamespace(parsed=_JointAssessment(
            status=self.joint_status.value,
            complete_gap_assessment="Joint assessment of all supplied papers.",
            cross_paper_inference="The inference is partially supported.",
            evidence_summary_accuracy="Summary mostly matches the cited sources.",
            research_opportunity="A protocol may follow from the evidence.",
            gap_origin="Inferred origin is appropriate.", overgeneralization="Field-wide claims need caution.",
            limitations="The sources cover a limited sample.", counterevidence="No opposing claim found.",
            contradictions=self.contradictions))


def contradiction(source_a="a1", quote_a="The method improved accuracy.",
                  source_b="b1", quote_b="The method did not improve accuracy."):
    return _ProposedContradiction(
        proposition_a="The method improved accuracy.",
        source_a=EvidenceRef(paper_id="paper-A", chunk_ids=[source_a]),
        supporting_quote_a=quote_a,
        proposition_b="The method did not improve accuracy.",
        source_b=EvidenceRef(paper_id="paper-B", chunk_ids=[source_b]),
        supporting_quote_b=quote_b,
        explanation="The reported outcomes directly oppose one another.", unresolved=True)


def run(client=None, store=None, gaps=None):
    client = client or Client()
    store = store or Store()
    result = EvidenceValidationAgent(vector_store=store, client=client).validate(input_result(gaps))
    return result, client, store


def test_gemini_developer_schemas_omit_additional_properties_and_local_models_stay_strict():
    schemas = [_gemini_compatible_schema(model) for model in (
        _ChunkAssessment, _JointAssessment, _ContradictionAdjudication,
    )]

    def has_additional_properties(value):
        if isinstance(value, dict):
            return "additionalProperties" in value or any(
                has_additional_properties(item) for item in value.values()
            )
        if isinstance(value, list):
            return any(has_additional_properties(item) for item in value)
        return False

    assert all(not has_additional_properties(schema) for schema in schemas)
    for schema in schemas:
        config = types.GenerateContentConfig(
            response_mime_type="application/json", response_json_schema=schema,
        )
        assert config.response_json_schema == schema
    quote_schema = schemas[0]["properties"]["subject_quotes"]
    assert quote_schema["type"] == "object"
    assert "supports_physics_informed_ml" in quote_schema["properties"]

    valid = {
        "status": "SUPPORTED", "attributed_contribution": "A contribution.",
        "claim_evidence_alignment": "Aligned.", "evidence_summary_accuracy": "Accurate.",
        "limitations": "Limited.", "explanation": "Supported.",
        "subject_quotes": {"supports_physics_informed_ml": "exact phrase"},
    }
    parsed = _ChunkAssessment.model_validate(valid, strict=True)
    assert parsed.subject_quotes["supports_physics_informed_ml"] == "exact phrase"
    with pytest.raises(Exception):
        _ChunkAssessment.model_validate({**valid, "unexpected": "field"}, strict=True)
    with pytest.raises(Exception):
        _ChunkAssessment.model_validate({**valid, "subject_quotes": {"unknown_flag": "quote"}}, strict=True)


def test_exact_gemini_additional_properties_api_error_fails_closed():
    exact_error = (
        "additionalProperties is only supported in Gemini Enterprise Agent Platform mode, "
        "not in Gemini Developer API mode."
    )
    out, _, _ = run(Client(fail_schema=_ChunkAssessment, fail_error=exact_error))
    gap_result = out.gap_validations[0]
    assert gap_result.status is GapValidationStatus.VALIDATION_INCOMPLETE
    assert all(item.status is None for item in gap_result.evidence_validations)
    assert all(exact_error in item.assessment_error for item in gap_result.evidence_validations)


def test_chunk_assesses_only_its_paper_contribution_and_joint_receives_all_verified_chunks():
    out, client, store = run()
    gap_result = out.gap_validations[0]
    assert store.calls == ["a1", "b1"]
    assert [e.status for e in gap_result.evidence_validations] == [
        EvidenceValidationStatus.SUPPORTED, EvidenceValidationStatus.PARTIALLY_SUPPORTED]
    chunk_prompts = [prompt for schema, prompt in client.prompts if schema_name(schema) == _ChunkAssessment.__name__]
    joint_prompts = [prompt for schema, prompt in client.prompts if schema_name(schema) == _JointAssessment.__name__]
    assert "do not require this chunk to establish the whole cross-paper gap" in EvidenceValidationAgent.CHUNK_SYSTEM_INSTRUCTION
    assert all(pid in prompt for pid, prompt in [("paper-A", chunk_prompts[0]), ("paper-B", chunk_prompts[1])])
    assert all(cid in joint_prompts[0] for cid in ("a1", "b1"))
    assert all(text in joint_prompts[0] for text in ("improved accuracy", "did not improve accuracy"))
    assert gap_result.joint_assessment is not None
    assert {ref.paper_id: ref.chunk_ids for ref in gap_result.joint_assessment.source_evidence} == {
        "paper-A": ["a1"], "paper-B": ["b1"]}
    assert gap_result.status is GapValidationStatus.PARTIALLY_SUPPORTED


def test_chunk_subject_mentions_are_structured_and_verified_against_exact_text():
    records = {
        "a1": {"chunk_id": "a1", "text": (
            "LLM self-explanations may fail faithfulness checks."
        ), "metadata": {"paper_id": "paper-A", "chunk_id": "a1"}},
        "b1": {"chunk_id": "b1", "text": (
            "Accurate model predictions do not establish causal relationships."
        ), "metadata": {"paper_id": "paper-B", "chunk_id": "b1"}},
    }
    output, _, _ = run(store=Store(records=records))
    first, second = output.gap_validations[0].evidence_validations
    assert first.mentions_llm is True
    assert first.mentions_faithfulness is True
    assert second.mentions_llm is False
    assert second.mentions_faithfulness is False
    assert json.loads(first.findings)["mentions_llm"] is True


def test_positive_subject_support_flag_requires_verbatim_chunk_quote():
    assessment = _ChunkAssessment(
        status="SUPPORTED",
        attributed_contribution="Physics-informed ML is called for.",
        supported_contribution="Physics-informed ML is called for.",
        supported_contribution_quote="physics-infused ML",
        subject_quotes={"supports_physics_informed_ml": "physics-infused ML"},
        supports_physics_informed_ml=True,
        claim_evidence_alignment="Exact source phrase supports this contribution.",
        evidence_summary_accuracy="Accurate.", limitations="None.", explanation="Supported.",
    )
    flags = EvidenceValidationAgent._verified_subject_flags(
        assessment, "The paper calls for physics-infused ML approaches."
    )
    assert flags["supports_physics_informed_ml"] is True
    assert flags["supported_contribution"] == "Physics-informed ML is called for."

    assessment.subject_quotes["supports_physics_informed_ml"] = "physics-informed methods"
    assessment.supported_contribution_quote = "physics-informed methods"
    flags = EvidenceValidationAgent._verified_subject_flags(
        assessment, "The paper calls for physics-infused ML approaches."
    )
    assert flags["supports_physics_informed_ml"] is False
    assert flags["supported_contribution"] is None


@pytest.mark.parametrize("joint,chunk_statuses,expected", [
    (GapSupportStatus.FULLY_SUPPORTED, ["SUPPORTED", "SUPPORTED"], GapValidationStatus.FULLY_SUPPORTED),
    (GapSupportStatus.PARTIALLY_SUPPORTED, ["SUPPORTED", "SUPPORTED"], GapValidationStatus.PARTIALLY_SUPPORTED),
    (GapSupportStatus.UNSUPPORTED, ["UNSUPPORTED", "UNSUPPORTED"], GapValidationStatus.UNSUPPORTED),
    (GapSupportStatus.UNSUPPORTED, ["PARTIALLY_SUPPORTED", "UNSUPPORTED"], GapValidationStatus.PARTIALLY_SUPPORTED),
])
def test_joint_assessment_determines_gap_status(joint, chunk_statuses, expected):
    out, _, _ = run(Client(chunk_statuses, joint_status=joint))
    assert out.gap_validations[0].status is expected


def test_only_verified_opposing_quotes_become_structured_contradictions():
    valid = contradiction()
    out, _, _ = run(Client(contradictions=[valid]))
    item = out.gap_validations[0]
    assert len(item.contradictions) == 1
    assert item.contradictions[0].source_a.chunk_ids == ["a1"]
    assert item.contradictions[0].source_b.chunk_ids == ["b1"]
    assert item.status is GapValidationStatus.PARTIALLY_SUPPORTED


def test_unrelated_propositions_with_valid_quotes_are_rejected():
    unrelated = contradiction(
        quote_b="The sample included 30 participants.")
    unrelated.proposition_a = "The method improved accuracy."
    unrelated.proposition_b = "The paper used a sample of 30 participants."
    adjudication = _ContradictionAdjudication(
        same_subject=False, substantively_incompatible=False,
        confidence="LOW", explanation="The statements concern different claims.")
    custom_store = Store({
        "a1": {"chunk_id": "a1", "text": "The method improved accuracy.", "metadata": {"paper_id": "paper-A"}},
        "b1": {"chunk_id": "b1", "text": "The sample included 30 participants.", "metadata": {"paper_id": "paper-B"}},
    })
    out, client, _ = run(Client(contradictions=[unrelated], adjudication=adjudication), store=custom_store)
    assert out.gap_validations[0].contradictions == []
    adjudication_prompts = [p for schema, p in client.prompts if schema_name(schema) == _ContradictionAdjudication.__name__]
    assert len(adjudication_prompts) == 1
    assert "Proposition A:" in adjudication_prompts[0]
    assert "Verified quote A:" in adjudication_prompts[0]
    assert "Gap title" not in adjudication_prompts[0]


@pytest.mark.parametrize("quote,expected", [
    ("The method improved\naccuracy.", True),  # PDF line-wrap normalization
    ("the method improved accuracy.", False),  # case must match
    ("The method improved reliability.", False),  # absent from source
])
def test_quote_verification_normalizes_whitespace_but_is_case_sensitive(quote, expected):
    candidate = contradiction(quote_a=quote)
    out, _, _ = run(Client(contradictions=[candidate]))
    assert bool(out.gap_validations[0].contradictions) is expected

    # A warning that only reflects missing mention has no quote found in the
    # source and must not be rendered as a contradiction.
    ungrounded = contradiction(quote_a="The study uses specialized neural models.")
    out, _, _ = run(Client(contradictions=[ungrounded], joint_status=GapSupportStatus.PARTIALLY_SUPPORTED))
    assert out.gap_validations[0].contradictions == []
    assert out.gap_validations[0].contradiction_warnings == []


def test_unresolved_contradiction_prevents_fully_supported():
    out, _, _ = run(Client(["SUPPORTED", "SUPPORTED"], GapSupportStatus.FULLY_SUPPORTED,
                           [contradiction()]))
    assert out.gap_validations[0].status is GapValidationStatus.PARTIALLY_SUPPORTED


def test_absence_of_mention_is_a_limitation_not_a_contradiction():
    out, _, _ = run(Client(joint_status=GapSupportStatus.PARTIALLY_SUPPORTED))
    joint = out.gap_validations[0].joint_assessment
    assert "No opposing claim" in joint.counterevidence
    assert out.gap_validations[0].contradictions == []


def test_missing_mismatched_and_retrieval_failures_make_validation_incomplete():
    refs = [EvidenceRef(paper_id="paper-A", chunk_ids=["absent"]),
            EvidenceRef(paper_id="paper-B", chunk_ids=["wrong-owner"])]
    store = Store({"wrong-owner": {"chunk_id": "wrong-owner", "text": "private text", "metadata": {"paper_id": "other", "chunk_id": "wrong-owner"}}})
    out, client, _ = run(store=store, gaps=[gap(refs)])
    assert [item.status for item in out.gap_validations[0].evidence_validations] == [
        EvidenceValidationStatus.NOT_FOUND, EvidenceValidationStatus.MISMATCHED]
    assert out.gap_validations[0].status is GapValidationStatus.VALIDATION_INCOMPLETE
    assert not client.prompts

    retrieval_out, _, _ = run(store=Store(fail=True))
    assert retrieval_out.gap_validations[0].status is GapValidationStatus.VALIDATION_INCOMPLETE
    assert retrieval_out.retrieval_failures


def test_chunk_assessment_failure_is_incomplete_and_located_on_evidence_item():
    out, _, _ = run(Client(fail_schema=_ChunkAssessment))
    assert out.gap_validations[0].status is GapValidationStatus.VALIDATION_INCOMPLETE
    assert all(item.status is None for item in out.gap_validations[0].evidence_validations)
    assert all(item.assessment_error for item in out.gap_validations[0].evidence_validations)
    assert all(item.retrieval_error is None for item in out.gap_validations[0].evidence_validations)
    assert any("GAP-001/paper-A/a1" in failure for failure in out.semantic_failures)
    assert any("GAP-001/paper-B/b1" in failure for failure in out.semantic_failures)


def test_joint_assessment_failure_is_incomplete_and_gap_scoped():
    out, _, _ = run(Client(["SUPPORTED", "SUPPORTED"], fail_schema=_JointAssessment))
    gap_result = out.gap_validations[0]
    assert gap_result.status is GapValidationStatus.VALIDATION_INCOMPLETE
    assert gap_result.joint_assessment is None
    assert "simulated model failure" in gap_result.joint_assessment_error
    # Chunk-level assessments remain intact; joint errors are not misreported
    # as errors on evidence assessments that succeeded.
    assert all(item.status is EvidenceValidationStatus.SUPPORTED for item in gap_result.evidence_validations)
    assert all(item.assessment_error is None and item.retrieval_error is None for item in gap_result.evidence_validations)
    assert any("GAP-001: joint Gemini assessment failed" in failure for failure in out.semantic_failures)


def test_empty_and_invalid_input():
    agent = EvidenceValidationAgent(vector_store=Store(), client=Client())
    assert agent.validate(input_result([])).gap_validations == []
    with pytest.raises(TypeError):
        agent.validate({})


def test_targeted_joint_retry_preserves_other_gap_and_skips_chunk_assessments(tmp_path, monkeypatch):
    import experiments.run_evidence_validation as runner

    target = gap(gap_id="GAP-003")
    other = gap(gap_id="GAP-001")
    detection = input_result([other, target])
    client = Client(
        chunk_statuses=["SUPPORTED"] * 4,
        joint_status=GapSupportStatus.PARTIALLY_SUPPORTED,
    )
    store = Store()
    agent = EvidenceValidationAgent(vector_store=store, client=client)
    original = agent.validate(detection)
    original_gaps = []
    for item in original.gap_validations:
        if item.gap_id == "GAP-003":
            item = item.model_copy(update={
                "status": GapValidationStatus.VALIDATION_INCOMPLETE,
                "joint_assessment": None,
                "cross_paper_inference": None,
                "contradictions": [],
                "contradiction_warnings": [],
                "joint_assessment_error": "simulated quota failure",
            })
        else:
            item = item.model_copy(update={"status": GapValidationStatus.FULLY_SUPPORTED})
        original_gaps.append(item)
    saved = original.model_copy(update={
        "gap_validations": original_gaps,
        "validation_counts": {
            "FULLY_SUPPORTED": 1, "PARTIALLY_SUPPORTED": 0,
            "UNSUPPORTED": 0, "VALIDATION_INCOMPLETE": 1,
        },
        "semantic_failures": ["GAP-003: joint Gemini assessment failed: simulated quota failure"],
    })

    input_path, comparison_path, gap_fingerprint = lineaged_gap_input(detection, tmp_path)
    saved.provenance = ArtifactProvenance(
        stage="evidence_validation",
        input_fingerprints={"gap_detection": gap_fingerprint},
    )
    output_path = tmp_path / "evidence.json"
    output_path.write_text(saved.model_dump_json(indent=2), encoding="utf-8")
    store.calls.clear()
    client.prompts.clear()
    monkeypatch.setattr(runner, "EvidenceValidationAgent", lambda: agent)
    monkeypatch.setattr(sys, "argv", [
        "run_evidence_validation.py", "--input", str(input_path),
        "--comparison", str(comparison_path), "--output", str(output_path),
        "--retry-joint", "GAP-003",
    ])

    runner.main()

    updated = EvidenceValidationResult.model_validate_json(output_path.read_text(encoding="utf-8"))
    by_id = {item.gap_id: item for item in updated.gap_validations}
    assert by_id["GAP-003"].status is GapValidationStatus.PARTIALLY_SUPPORTED
    assert by_id["GAP-003"].joint_assessment is not None
    assert by_id["GAP-001"].status is GapValidationStatus.FULLY_SUPPORTED
    assert all(schema_name(schema) == _JointAssessment.__name__ for schema, _ in client.prompts)
    assert store.calls == ["a1", "b1"]
    assert updated.semantic_failures == []


def test_targeted_joint_retry_failure_does_not_overwrite_saved_file(tmp_path, monkeypatch):
    import experiments.run_evidence_validation as runner

    target = gap(gap_id="GAP-003")
    detection = input_result([target])
    client = Client(chunk_statuses=["SUPPORTED"] * 2, fail_schema=_JointAssessment)
    store = Store()
    agent = EvidenceValidationAgent(vector_store=store, client=client)
    original = agent.validate(detection)
    incomplete = original.gap_validations[0].model_copy(update={
        "status": GapValidationStatus.VALIDATION_INCOMPLETE,
        "joint_assessment": None,
        "cross_paper_inference": None,
        "contradictions": [],
        "contradiction_warnings": [],
        "joint_assessment_error": "initial quota failure",
    })
    saved = original.model_copy(update={
        "gap_validations": [incomplete],
        "validation_counts": {
            "FULLY_SUPPORTED": 0, "PARTIALLY_SUPPORTED": 0,
            "UNSUPPORTED": 0, "VALIDATION_INCOMPLETE": 1,
        },
        "semantic_failures": ["GAP-003: joint Gemini assessment failed: initial quota failure"],
    })
    input_path, comparison_path, gap_fingerprint = lineaged_gap_input(detection, tmp_path)
    saved.provenance = ArtifactProvenance(
        stage="evidence_validation",
        input_fingerprints={"gap_detection": gap_fingerprint},
    )
    output_path = tmp_path / "evidence.json"
    original_bytes = saved.model_dump_json(indent=2).encode("utf-8")
    output_path.write_bytes(original_bytes)
    store.calls.clear()
    client.prompts.clear()
    monkeypatch.setattr(runner, "EvidenceValidationAgent", lambda: agent)
    monkeypatch.setattr(sys, "argv", [
        "run_evidence_validation.py", "--input", str(input_path),
        "--comparison", str(comparison_path), "--output", str(output_path),
        "--retry-joint", "GAP-003",
    ])

    with pytest.raises(RuntimeError, match="simulated model failure"):
        runner.main()

    assert output_path.read_bytes() == original_bytes
    assert all(schema_name(schema) == _JointAssessment.__name__ for schema, _ in client.prompts)
    assert store.calls == ["a1", "b1"]
