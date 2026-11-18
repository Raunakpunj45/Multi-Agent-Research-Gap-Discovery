import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from experiments import run_langgraph
from src.agents.artifact_provenance import (
    ArtifactProvenance, CANONICAL_JSON_SHA256, fingerprint_json,
)
from src.agents.critic_agent import (
    CriticDecision, CriticResult, CriticReviewStatus, GapCriticReview, GapOriginAssessment,
)
from src.agents.cross_paper_comparison_agent import EvidenceRef
from src.agents.evidence_validation_agent import (
    EvidenceValidation, EvidenceValidationResult, EvidenceValidationStatus,
    GapSupportStatus, GapValidation, GapValidationStatus, JointGapAssessment,
)
from src.agents.research_question_agent import GenerationStatus, ResearchQuestionResult
from src.workflow.langgraph_app import _apply_revision_suggestions
from tests.test_langgraph_workflow import (
    PAPER_IDS, comparison, critic_result, gap_result,
    validation, workflow_agents,
)


def test_latest_mixed_critic_artifact_retries_only_incomplete_reviews_and_ignores_placeholders(
    tmp_path, monkeypatch, langgraph_run_fixture,
):
    source, original_bytes = langgraph_run_fixture("critic_mixed_incomplete_reviews.json")
    initial = run_langgraph._prepare_critic_resume(source)
    prior = initial["critic_result"]
    assert [(review.gap_id, review.decision, review.review_status)
            for review in prior.gap_reviews] == [
        ("GAP-001", CriticDecision.REVISE, CriticReviewStatus.COMPLETE),
        ("GAP-002", CriticDecision.ACCEPT, CriticReviewStatus.COMPLETE),
        ("GAP-003", CriticDecision.REVISE, CriticReviewStatus.INCOMPLETE),
        ("GAP-004", CriticDecision.REVISE, CriticReviewStatus.INCOMPLETE),
    ]

    revised, changed = _apply_revision_suggestions(
        initial["gap_detection"], initial["comparison"], prior,
    )
    assert changed == ["GAP-001"]
    assert revised.gaps[1] == initial["gap_detection"].gaps[1]
    assert revised.gaps[2:] == initial["gap_detection"].gaps[2:]

    agents = workflow_agents()
    def accept_after_retry(_comparison, gaps, _evidence, **kwargs):
        return CriticResult(
            research_topic=gaps.research_topic,
            review_status=CriticReviewStatus.COMPLETE,
            gap_reviews=[GapCriticReview(
                gap_id=gap.gap_id, decision=CriticDecision.ACCEPT,
                review_status=CriticReviewStatus.COMPLETE,
                origin_assessment=GapOriginAssessment.INFERRED,
                origin_rationale="Reassessed.", reasoning="Accepted.",
            ) for gap in gaps.gaps],
            input_fingerprints=kwargs["input_fingerprints"],
        )
    agents.critic.critique.side_effect = accept_after_retry
    monkeypatch.setattr(run_langgraph, "default_workflow_agents", lambda **_kwargs: agents)
    output_base = tmp_path / "mixed-resume.json"
    monkeypatch.setattr(sys, "argv", [
        "run_langgraph.py", "--resume-from-critic", str(source),
        "--output", str(output_base),
    ])
    # The mocked critic ACCEPTs every gap, but fixture GAP-001 is
    # PARTIALLY_SUPPORTED and GAP-003/GAP-004 cite PARTIALLY_SUPPORTED chunks,
    # so the strict acceptance rule stops the run (exit 1) without questions.
    assert run_langgraph.main() == 1
    agents.research_questions.generate_questions.assert_not_called()
    agents.critic.critique.assert_called_once()
    kwargs = agents.critic.critique.call_args.kwargs
    assert kwargs["previous_result"].input_fingerprints == prior.input_fingerprints
    assert agents.retrieval.search.call_count == 0
    assert agents.paper_analysis.analyze_paper.call_count == 0
    assert agents.comparison.compare_many.call_count == 0
    assert agents.gap_detection.detect_gaps.call_count == 0
    assert agents.evidence_validation.validate.call_count == 0
    assert source.read_bytes() == original_bytes


def test_latest_completed_revisions_update_origin_text_and_replace_invalid_evidence_refs(
    langgraph_run_fixture,
):
    source, _ = langgraph_run_fixture("critic_completed_revisions.json")
    state = run_langgraph._prepare_critic_resume(source)
    prior = state["gap_detection"]
    prior_by_id = {gap.gap_id: gap for gap in prior.gaps}
    revised, changed = _apply_revision_suggestions(
        prior, state["comparison"], state["critic_result"],
    )
    revised_by_id = {gap.gap_id: gap for gap in revised.gaps}

    assert changed == ["GAP-001", "GAP-004"]
    assert revised_by_id["GAP-001"].gap_origin.value == "inferred_cross_paper"
    assert "chemical database" not in revised_by_id["GAP-001"].description.lower()
    assert revised_by_id["GAP-002"] == prior_by_id["GAP-002"]
    assert revised_by_id["GAP-003"] == prior_by_id["GAP-003"]

    gap4 = revised_by_id["GAP-004"]
    assert "P5-E1" not in gap4.evidence_summary
    assert "P5-E3" not in gap4.evidence_summary
    assert gap4.supporting_paper_ids == [
        "https://openalex.org/W4402683817",
        "https://openalex.org/W4410862096",
    ]
    assert set(gap4.supporting_chunk_ids) == {
        "https://openalex.org/W4402683817_chunk_40",
        "https://openalex.org/W4402683817_chunk_8",
        "https://openalex.org/W4410862096_chunk_9",
        "https://openalex.org/W4410862096_chunk_53",
    }
    assert {ref.paper_id for ref in gap4.supporting_evidence} == set(
        gap4.supporting_paper_ids
    )


def resume_artifacts(tmp_path):
    compare = comparison()
    gaps = gap_result()
    gaps.provenance = ArtifactProvenance(
        stage="research_gap_detection",
        input_fingerprints={"comparison": fingerprint_json(compare.model_dump(
            mode="json", exclude={"provenance"}))},
        fingerprint_schemes={"comparison": CANONICAL_JSON_SHA256},
    )
    evidence = validation(GapValidationStatus.PARTIALLY_SUPPORTED)
    prior_gap_validation = evidence.gap_validations[0]
    evidence.gap_validations = [prior_gap_validation.model_copy(update={
        "evidence_validations": [record.model_copy(update={
            "status": EvidenceValidationStatus.PARTIALLY_SUPPORTED,
        }) for record in prior_gap_validation.evidence_validations],
    })]
    evidence.provenance = ArtifactProvenance(
        stage="evidence_validation",
        input_fingerprints={"gap_detection": fingerprint_json(gaps.model_dump(
            mode="json", exclude={"provenance"}))},
        fingerprint_schemes={"gap_detection": CANONICAL_JSON_SHA256},
    )
    original_path = tmp_path / "original.json"
    evidence_path = tmp_path / "retry.json"
    original_path.write_text(json.dumps({
        "research_topic": "topic", "minimum_papers": 2, "max_revisions": 2,
        "revision_count": 0, "comparison": compare.model_dump(mode="json"),
        "gap_detection": gaps.model_dump(mode="json"),
    }), encoding="utf-8")
    evidence_path.write_text(evidence.model_dump_json(), encoding="utf-8")
    return original_path, evidence_path, compare, gaps, evidence


def test_resume_starts_at_critic_and_writes_unique_output(tmp_path, monkeypatch):
    original_path, evidence_path, _, _, evidence = resume_artifacts(tmp_path)
    original_bytes = original_path.read_bytes()
    evidence_bytes = evidence_path.read_bytes()
    agents = workflow_agents(evidence_status=GapValidationStatus.PARTIALLY_SUPPORTED)
    expected = run_langgraph._prepare_evidence_resume(original_path, evidence_path)
    fingerprints = {key: run_langgraph._model_fingerprint(expected[key]) for key in (
        "comparison", "gap_detection", "evidence_validation")}
    agents.critic.critique.side_effect = lambda *_args, **kwargs: critic_result(
        CriticDecision.ACCEPT, kwargs["input_fingerprints"],
    )
    monkeypatch.setattr(run_langgraph, "default_workflow_agents", lambda **_kwargs: agents)
    output_base = tmp_path / "resumed.json"
    monkeypatch.setattr(sys, "argv", [
        "run_langgraph.py", "--resume-from-evidence", str(original_path),
        "--resume-evidence-validation", str(evidence_path), "--output", str(output_base),
    ])

    # The resumed evidence is PARTIALLY_SUPPORTED, so the critic's ACCEPT is
    # never effective: the run stops (exit 1) and no questions are generated.
    assert run_langgraph.main() == 1
    outputs = list(tmp_path.glob("resumed-resume-*.json"))
    assert len(outputs) == 1
    saved = json.loads(outputs[0].read_text(encoding="utf-8"))
    assert saved["critic_input_fingerprints"] == fingerprints
    assert saved["evidence_validation"]["gap_validations"][0]["status"] == "PARTIALLY_SUPPORTED"
    assert saved["workflow_status"] == "STOPPED"
    assert "strict acceptance rule" in saved["stop_reason"]
    agents.research_questions.generate_questions.assert_not_called()
    agents.retrieval.search.assert_not_called()
    agents.paper_analysis.analyze_paper.assert_not_called()
    agents.comparison.compare_many.assert_not_called()
    agents.gap_detection.detect_gaps.assert_not_called()
    agents.evidence_validation.validate.assert_not_called()
    agents.critic.critique.assert_called_once()
    assert original_path.read_bytes() == original_bytes
    assert evidence_path.read_bytes() == evidence_bytes


@pytest.mark.parametrize("mutation, message", [
    ("fingerprint", "stale"),
    ("references", "references do not exactly match"),
])
def test_resume_rejects_mismatched_fingerprint_or_missing_reference(tmp_path, mutation, message):
    original_path, evidence_path, _, _, evidence = resume_artifacts(tmp_path)
    if mutation == "fingerprint":
        evidence.provenance.input_fingerprints["gap_detection"] = "0" * 64
    else:
        evidence.gap_validations[0].evidence_validations.pop()
    evidence_path.write_text(evidence.model_dump_json(), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        run_langgraph._prepare_evidence_resume(original_path, evidence_path)


def test_resume_rejects_invalid_evidence_artifact(tmp_path):
    original_path, evidence_path, *_ = resume_artifacts(tmp_path)
    evidence_path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="Could not load valid resume artifacts"):
        run_langgraph._prepare_evidence_resume(original_path, evidence_path)


def test_evidence_resume_revises_revalidates_and_obeys_saved_revision_limit(tmp_path, monkeypatch):
    original_path, evidence_path, *_ = resume_artifacts(tmp_path)
    agents = workflow_agents()
    agents.critic.critique.side_effect = lambda *_args, **kwargs: critic_result(
        CriticDecision.REVISE, kwargs["input_fingerprints"], with_revision=True,
    )
    monkeypatch.setattr(run_langgraph, "default_workflow_agents", lambda **_kwargs: agents)
    output_base = tmp_path / "revision.json"
    monkeypatch.setattr(sys, "argv", [
        "run_langgraph.py", "--resume-from-evidence", str(original_path),
        "--resume-evidence-validation", str(evidence_path), "--output", str(output_base),
    ])
    assert run_langgraph.main() == 1
    assert agents.evidence_validation.validate.call_count == 2
    assert agents.critic.critique.call_count == 3


def test_latest_saved_critic_revisions_revalidate_only_affected_gaps(
    tmp_path, monkeypatch, langgraph_run_fixture,
):
    source, source_bytes = langgraph_run_fixture("critic_two_revise_decisions.json")
    initial = run_langgraph._prepare_critic_resume(source)
    prior_gaps = {gap.gap_id: gap.model_dump(mode="json")
                  for gap in initial["gap_detection"].gaps}
    revised_ids = [review.gap_id for review in initial["critic_result"].gap_reviews
                   if review.decision is CriticDecision.REVISE]
    critic_agent = Mock()
    critic_agent.critique.side_effect = lambda _comparison, gap_result, _evidence, input_fingerprints, **_kwargs: CriticResult(
        research_topic=gap_result.research_topic,
        review_status=CriticReviewStatus.COMPLETE,
        gap_reviews=[GapCriticReview(
            gap_id=gap.gap_id, decision=CriticDecision.ACCEPT,
            review_status=CriticReviewStatus.COMPLETE,
            origin_assessment=GapOriginAssessment.INFERRED,
            origin_rationale="Rechecked after revision.", reasoning="Accepted.",
        ) for gap in gap_result.gaps],
        input_fingerprints=input_fingerprints,
    )
    evidence_agent = Mock()
    evidence_calls = []

    def validate_revised(gap_result):
        evidence_calls.append([gap.gap_id for gap in gap_result.gaps])
        validations = []
        for gap in gap_result.gaps:
            records = [EvidenceValidation(
                paper_id=ref.paper_id, chunk_id=chunk_id,
                status=EvidenceValidationStatus.SUPPORTED,
                findings="Revalidated after critic revision.",
            ) for ref in gap.supporting_evidence for chunk_id in ref.chunk_ids]
            validations.append(GapValidation(
                gap_id=gap.gap_id, status=GapValidationStatus.FULLY_SUPPORTED,
                explanation="Revalidated.", evidence_validations=records,
                joint_assessment=JointGapAssessment(
                    status=GapSupportStatus.FULLY_SUPPORTED,
                    complete_gap_assessment="Supported.", cross_paper_inference="Supported.",
                    evidence_summary_accuracy="Accurate.", research_opportunity="Valid.",
                    gap_origin="Inferred.", overgeneralization="Limited.",
                    limitations="Limited.", counterevidence="None.",
                    source_evidence=gap.supporting_evidence,
                ),
            ))
        return EvidenceValidationResult(
            research_topic=gap_result.research_topic, paper_ids=gap_result.paper_ids,
            gap_validations=validations, validation_counts={}, evidence_status_counts={},
        )

    evidence_agent.validate.side_effect = validate_revised
    question_agent = Mock()
    question_agent.generate_questions.return_value = ResearchQuestionResult(
        research_topic=initial["research_topic"], generation_status=GenerationStatus.COMPLETE,
    )
    agents = SimpleNamespace(
        retrieval=Mock(), paper_analysis=Mock(), comparison=Mock(), gap_detection=Mock(),
        evidence_validation=evidence_agent, critic=critic_agent, research_questions=question_agent,
    )
    monkeypatch.setattr(run_langgraph, "default_workflow_agents", lambda **_kwargs: agents)
    output_base = tmp_path / "continued.json"
    monkeypatch.setattr(sys, "argv", [
        "run_langgraph.py", "--resume-from-critic", str(source), "--output", str(output_base),
    ])

    # Only the revised gaps are revalidated, but the unrevised fixture GAP-003
    # (ACCEPTED earlier while citing PARTIALLY_SUPPORTED chunks) fails the
    # strict acceptance rule, so the run stops (exit 1) before questions.
    assert run_langgraph.main() == 1
    assert evidence_calls == [revised_ids]
    assert len(revised_ids) == 2
    assert critic_agent.critique.call_count == 1
    assert question_agent.generate_questions.call_count == 0
    outputs = list(tmp_path.glob("continued-PROMPT-001-*.json"))
    assert len(outputs) == 1
    saved = json.loads(outputs[0].read_text(encoding="utf-8"))
    assert saved["workflow_status"] == "STOPPED"
    assert "strict acceptance rule" in saved["stop_reason"]
    revised = {gap["gap_id"]: gap for gap in saved["gap_detection"]["gaps"]}
    for gap_id in revised_ids:
        assert revised[gap_id] != prior_gaps[gap_id]
    for gap_id in set(prior_gaps) - set(revised_ids):
        assert revised[gap_id] == prior_gaps[gap_id]
    assert saved["revision_count"] == 1
    assert all(review["decision"] == "ACCEPT"
               for review in saved["critic_result"]["gap_reviews"])
    assert source.read_bytes() == source_bytes
