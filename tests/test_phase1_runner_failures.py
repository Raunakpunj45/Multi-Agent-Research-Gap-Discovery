import hashlib
from pathlib import Path
import sys

import pytest

import experiments.run_evidence_validation as evidence_runner
import experiments.run_research_questions as questions_runner
from src.agents import (
    CriticDecision,
    CriticResult,
    CriticReviewStatus,
    EvidenceValidationResult,
    GapCriticReview,
    GapValidation,
    GapValidationStatus,
    GenerationStatus,
    ResearchGap,
    ResearchGapDetectionResult,
    ResearchQuestionResult,
)
from src.agents.artifact_provenance import ArtifactProvenance
from src.agents.research_gap_detection_agent import (
    GapCategory,
    GapConfidence,
    GapOrigin,
)


def gaps():
    return ResearchGapDetectionResult(
        research_topic="topic",
        paper_count=2,
        paper_ids=["P1", "P2"],
        gaps=[ResearchGap(
            gap_id="GAP-001", title="gap", description="description",
            gap_category=GapCategory.EMPIRICAL, reasoning="reasoning",
            supporting_paper_ids=["P1", "P2"], supporting_chunk_ids=["C1", "C2"],
            evidence_summary="summary", gap_origin=GapOrigin.INFERRED,
            confidence=GapConfidence.MEDIUM, confidence_reason="reason",
            research_opportunity="opportunity",
        )],
        gap_category_counts={},
        caveats=[],
    )


def evidence_result(status):
    return EvidenceValidationResult(
        research_topic="topic", paper_ids=["P1", "P2"],
        gap_validations=[GapValidation(
            gap_id="GAP-001", status=status, explanation="checked",
            evidence_validations=[],
        )],
        validation_counts={status.value: 1}, evidence_status_counts={},
    )


def comparison_artifact():
    from src.agents import MultiPaperComparison

    return MultiPaperComparison(
        paper_ids=["P1", "P2"], paper_count=2, dimension_summaries=[],
        recurring_patterns=[], common_limitations=[], conflicting_findings=[],
        methodological_differences=[], underexplored_directions=[], caveats=[],
        provenance=ArtifactProvenance(
            stage="cross_paper_comparison",
            input_fingerprints={"paper_analyses": "a" * 64},
        ),
    )


def test_evidence_runner_atomically_saves_then_exits_nonzero_for_incomplete_gap(
    tmp_path, monkeypatch,
):
    input_path = tmp_path / "gaps.json"
    comparison_path = tmp_path / "comparison.json"
    output_path = tmp_path / "validation.json"
    comparison_bytes = comparison_artifact().model_dump_json().encode("utf-8")
    comparison_path.write_bytes(comparison_bytes)
    gap_data = gaps().model_copy(update={"provenance": ArtifactProvenance(
        stage="research_gap_detection",
        input_fingerprints={"comparison": hashlib.sha256(comparison_bytes).hexdigest()},
    )})
    input_path.write_text(gap_data.model_dump_json(), encoding="utf-8")
    output_path.write_text("previous output", encoding="utf-8")
    incomplete = evidence_result(GapValidationStatus.VALIDATION_INCOMPLETE)

    class StubAgent:
        def validate(self, _gaps):
            return incomplete

    monkeypatch.setattr(evidence_runner, "EvidenceValidationAgent", StubAgent)
    monkeypatch.setattr(sys, "argv", [
        "run_evidence_validation.py", "--input", str(input_path),
        "--comparison", str(comparison_path), "--output", str(output_path),
    ])
    original_replace = evidence_runner.os.replace
    replacements = []

    def check_replace(source, destination):
        source, destination = Path(source), Path(destination)
        assert output_path.read_text(encoding="utf-8") == "previous output"
        assert EvidenceValidationResult.model_validate_json(
            source.read_text(encoding="utf-8")
        ).gap_validations[0].status is GapValidationStatus.VALIDATION_INCOMPLETE
        replacements.append((source, destination))
        original_replace(source, destination)

    monkeypatch.setattr(evidence_runner.os, "replace", check_replace)
    with pytest.raises(SystemExit) as exit_info:
        evidence_runner.main()

    assert exit_info.value.code == 1
    assert len(replacements) == 1
    assert EvidenceValidationResult.model_validate_json(
        output_path.read_text(encoding="utf-8")
    ).gap_validations[0].status is GapValidationStatus.VALIDATION_INCOMPLETE
    assert not list(tmp_path.glob(".validation.json.*.tmp"))


def test_evidence_runner_complete_result_saves_and_returns_success(tmp_path, monkeypatch):
    input_path = tmp_path / "gaps.json"
    comparison_path = tmp_path / "comparison.json"
    output_path = tmp_path / "validation.json"
    comparison_bytes = comparison_artifact().model_dump_json().encode("utf-8")
    comparison_path.write_bytes(comparison_bytes)
    gap_data = gaps().model_copy(update={"provenance": ArtifactProvenance(
        stage="research_gap_detection",
        input_fingerprints={"comparison": hashlib.sha256(comparison_bytes).hexdigest()},
    )})
    input_path.write_text(gap_data.model_dump_json(), encoding="utf-8")

    class StubAgent:
        def validate(self, _gaps):
            return evidence_result(GapValidationStatus.FULLY_SUPPORTED)

    monkeypatch.setattr(evidence_runner, "EvidenceValidationAgent", StubAgent)
    monkeypatch.setattr(sys, "argv", [
        "run_evidence_validation.py", "--input", str(input_path),
        "--comparison", str(comparison_path), "--output", str(output_path),
    ])
    evidence_runner.main()
    assert EvidenceValidationResult.model_validate_json(
        output_path.read_text(encoding="utf-8")
    ).gap_validations[0].status is GapValidationStatus.FULLY_SUPPORTED


@pytest.mark.parametrize("status,expected_exit", [
    (GenerationStatus.INCOMPLETE, 1),
    (GenerationStatus.COMPLETE, 0),
])
def test_research_question_runner_exit_matches_saved_overall_status(
    tmp_path, monkeypatch, status, expected_exit,
):
    comparison_path = tmp_path / "comparison.json"
    output_path = tmp_path / "questions.json"
    comparison_path.write_bytes(b"comparison bytes")
    comparison = comparison_artifact()
    comparison_payload = comparison.model_dump_json().encode("utf-8")
    comparison_path.write_bytes(comparison_payload)
    gap_payload = gaps().model_copy(update={"provenance": ArtifactProvenance(
        stage="research_gap_detection",
        input_fingerprints={"comparison": hashlib.sha256(comparison_payload).hexdigest()},
    )}).model_dump_json().encode("utf-8")
    (tmp_path / "critic.json").write_bytes(b"critic")
    (tmp_path / "gaps.json").write_bytes(gap_payload)
    validation = evidence_result(GapValidationStatus.FULLY_SUPPORTED).model_copy(update={
        "provenance": ArtifactProvenance(
            stage="evidence_validation",
            input_fingerprints={"gap_detection": hashlib.sha256(gap_payload).hexdigest()},
        )
    })
    (tmp_path / "validation.json").write_bytes(validation.model_dump_json().encode("utf-8"))
    output_path.write_text("previous output", encoding="utf-8")
    critic = CriticResult(
        review_status=CriticReviewStatus.COMPLETE,
        gap_reviews=[GapCriticReview(
            gap_id="GAP-001", decision=CriticDecision.ACCEPT,
            review_status=CriticReviewStatus.COMPLETE, origin_assessment="INFERRED",
            origin_rationale="origin", reasoning="reason",
        )],
    )
    monkeypatch.setattr(questions_runner, "load_optional", lambda path, _model: (
        {"critic": critic, "comparison": comparison, "gaps": gaps().model_copy(update={
            "provenance": ArtifactProvenance(stage="research_gap_detection",
                input_fingerprints={"comparison": hashlib.sha256(comparison_payload).hexdigest()}),
        }), "validation": validation}[path.stem],
        None,
        hashlib.sha256(path.read_bytes()).hexdigest(),
    ))

    class StubAgent:
        def generate_questions(self, *_args, **_kwargs):
            return ResearchQuestionResult(
                research_topic="topic", generation_status=status,
            )

    monkeypatch.setattr(questions_runner, "ResearchQuestionAgent", StubAgent)
    monkeypatch.setattr(sys, "argv", [
        "run_research_questions.py", "--comparison", str(comparison_path),
        "--critic", str(tmp_path / "critic.json"), "--gaps", str(tmp_path / "gaps.json"),
        "--validation", str(tmp_path / "validation.json"), "--output", str(output_path),
    ])

    if expected_exit:
        with pytest.raises(SystemExit) as exit_info:
            questions_runner.main()
        assert exit_info.value.code == expected_exit
    else:
        questions_runner.main()

    saved = ResearchQuestionResult.model_validate_json(
        output_path.read_text(encoding="utf-8")
    )
    assert saved.generation_status is status
    assert not list(tmp_path.glob(".questions.json.*.tmp"))
