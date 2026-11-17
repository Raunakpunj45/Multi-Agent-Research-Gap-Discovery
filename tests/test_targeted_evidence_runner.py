"""Offline integration coverage for LangGraph targeted evidence retry."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

import experiments.run_evidence_validation as runner
from src.agents.artifact_provenance import ArtifactProvenance, fingerprint_json
from src.agents.cross_paper_comparison_agent import MultiPaperComparison
from src.agents.evidence_validation_agent import (
    EvidenceValidation, EvidenceValidationResult, EvidenceValidationStatus,
    GapValidation, GapValidationStatus, GapSupportStatus, JointGapAssessment,
)
from src.agents.cross_paper_comparison_agent import EvidenceRef
from src.agents.research_gap_detection_agent import (
    GapCategory, GapConfidence, GapOrigin, ResearchGap, ResearchGapDetectionResult,
)


def _records(tmp_path):
    gap = ResearchGap(
        gap_id="GAP-004", title="A gap", description="Description",
        gap_category=GapCategory.METHODOLOGICAL, reasoning="Evidence",
        supporting_paper_ids=["p1", "p2"], supporting_chunk_ids=["c1", "c2"],
        supporting_evidence=[EvidenceRef(paper_id="p1", chunk_ids=["c1"]),
                             EvidenceRef(paper_id="p2", chunk_ids=["c2"])],
        evidence_summary="Summary", gap_origin=GapOrigin.INFERRED,
        confidence=GapConfidence.MEDIUM, confidence_reason="Reason",
        research_opportunity="Opportunity",
    )
    comparison = MultiPaperComparison(
        paper_ids=["p1", "p2"], paper_count=2, dimension_summaries=[],
        recurring_patterns=[], common_limitations=[], conflicting_findings=[],
        methodological_differences=[], underexplored_directions=[], caveats=[],
        provenance=ArtifactProvenance(stage="comparison", input_fingerprints={"paper_analyses": "a" * 64}),
    )
    detection = ResearchGapDetectionResult(
        research_topic="topic", paper_count=2, paper_ids=["p1", "p2"], gaps=[gap],
        gap_category_counts={}, caveats=[],
    )
    comparison_fp = fingerprint_json(comparison.model_dump(mode="json", exclude={"provenance"}))
    detection.provenance = ArtifactProvenance(
        stage="research_gap_detection", input_fingerprints={"comparison": comparison_fp},
    )
    gap_fp = fingerprint_json(detection.model_dump(mode="json", exclude={"provenance"}))
    saved = EvidenceValidationResult(
        research_topic="topic", paper_ids=["p1", "p2"],
        gap_validations=[GapValidation(
            gap_id="GAP-004", status=GapValidationStatus.VALIDATION_INCOMPLETE,
            explanation="Partial recovery needed.", evidence_validations=[
                EvidenceValidation(paper_id="p1", chunk_id="c1", status=EvidenceValidationStatus.SUPPORTED,
                                   findings="keep-this-byte-for-byte"),
                EvidenceValidation(paper_id="p2", chunk_id="c2", status=None,
                                   findings="failed", assessment_error="temporary"),
            ],
        )],
        validation_counts={"FULLY_SUPPORTED": 0, "PARTIALLY_SUPPORTED": 0,
                           "UNSUPPORTED": 0, "VALIDATION_INCOMPLETE": 1},
        evidence_status_counts={"SUPPORTED": 1},
        provenance=ArtifactProvenance(stage="evidence_validation", input_fingerprints={"gap_detection": gap_fp}),
    )
    return comparison, detection, saved


def _write_langgraph(path, records):
    comparison, detection, saved = records
    path.write_text(json.dumps({
        "comparison": comparison.model_dump(mode="json"),
        "gap_detection": detection.model_dump(mode="json"),
        "evidence_validation": saved.model_dump(mode="json"),
    }, indent=2), encoding="utf-8")


class FakeAgent:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def retry_failed_evidence(self, gap, saved):
        self.calls.append(gap.gap_id)
        if self.fail:
            raise RuntimeError("mock retry failure")
        # Simulates the already focused agent path and retains the successful record.
        return saved.model_copy(update={
            "status": GapValidationStatus.PARTIALLY_SUPPORTED,
            "evidence_validations": [saved.evidence_validations[0], saved.evidence_validations[1].model_copy(update={
                "status": EvidenceValidationStatus.UNSUPPORTED, "assessment_error": None,
            })],
            "joint_assessment": JointGapAssessment(
                status=GapSupportStatus.PARTIALLY_SUPPORTED, complete_gap_assessment="assessed",
                cross_paper_inference="inference", evidence_summary_accuracy="accurate",
                research_opportunity="opportunity", gap_origin="inferred", overgeneralization="limited",
                limitations="limited sample", counterevidence="none",
                source_evidence=[EvidenceRef(paper_id="p1", chunk_ids=["c1"]),
                                 EvidenceRef(paper_id="p2", chunk_ids=["c2"])],
            ),
        })


def _invoke(monkeypatch, artifact, output, agent, extra=()):
    monkeypatch.setattr(runner, "ROOT_DIR", artifact.parent)
    monkeypatch.setattr(runner, "EvidenceValidationAgent", lambda: agent)
    monkeypatch.setattr(sys, "argv", ["run_evidence_validation.py", "--input", str(artifact),
                                      "--output", str(output), "--retry-evidence", "GAP-004", *extra])
    runner.main()


def test_langgraph_extraction_and_retry_runner_end_to_end(tmp_path, monkeypatch):
    artifact = tmp_path / "langgraph.json"
    output = tmp_path / "retry.json"
    records = _records(tmp_path)
    _write_langgraph(artifact, records)
    original = artifact.read_bytes()
    fake = FakeAgent()
    _invoke(monkeypatch, artifact, output, fake)

    assert fake.calls == ["GAP-004"]
    assert artifact.read_bytes() == original
    extracted = list((tmp_path / "data" / "processed" / "targeted_retries").glob("*/"))
    assert len(extracted) == 1
    comparison = MultiPaperComparison.model_validate_json((extracted[0] / "comparison.json").read_text())
    detection = ResearchGapDetectionResult.model_validate_json((extracted[0] / "gap_detection.json").read_text())
    saved = EvidenceValidationResult.model_validate_json((extracted[0] / "evidence_validation.json").read_text())
    assert comparison.provenance.fingerprint_schemes["paper_analyses"] == "canonical_json_sha256"
    assert detection.provenance.fingerprint_schemes["comparison"] == "canonical_json_sha256"
    assert saved.provenance.input_fingerprints["gap_detection"] == fingerprint_json(
        detection.model_dump(mode="json", exclude={"provenance"}))
    updated = EvidenceValidationResult.model_validate_json(output.read_text())
    assert updated.gap_validations[0].status is GapValidationStatus.PARTIALLY_SUPPORTED
    assert updated.gap_validations[0].evidence_validations[0].findings == "keep-this-byte-for-byte"


def test_retry_exception_leaves_original_artifact_and_creates_no_output(tmp_path, monkeypatch):
    artifact, output = tmp_path / "langgraph.json", tmp_path / "retry.json"
    _write_langgraph(artifact, _records(tmp_path))
    before = artifact.read_bytes()
    with pytest.raises(RuntimeError, match="mock retry failure"):
        _invoke(monkeypatch, artifact, output, FakeAgent(fail=True))
    assert artifact.read_bytes() == before
    assert not output.exists()


@pytest.mark.parametrize("collision", ["input", "existing"])
def test_retry_rejects_colliding_or_existing_output(tmp_path, monkeypatch, collision):
    artifact = tmp_path / "langgraph.json"
    _write_langgraph(artifact, _records(tmp_path))
    output = artifact if collision == "input" else tmp_path / "already.json"
    if collision == "existing":
        output.write_text("do not overwrite")
    before = output.read_bytes() if output.exists() else None
    with pytest.raises(SystemExit):
        _invoke(monkeypatch, artifact, output, FakeAgent())
    if before is not None:
        assert output.read_bytes() == before


@pytest.mark.parametrize("tamper", ["comparison", "gap_fingerprint", "saved_fingerprint", "scheme"])
def test_langgraph_retry_rejects_tampered_or_mismatched_lineage(tmp_path, monkeypatch, tamper):
    artifact, output = tmp_path / "langgraph.json", tmp_path / "retry.json"
    _write_langgraph(artifact, _records(tmp_path))
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if tamper == "comparison":
        payload["comparison"]["caveats"].append("tampered")
    elif tamper == "gap_fingerprint":
        payload["gap_detection"]["provenance"]["input_fingerprints"]["comparison"] = "0" * 64
    elif tamper == "saved_fingerprint":
        payload["evidence_validation"]["provenance"]["input_fingerprints"]["gap_detection"] = "0" * 64
    else:
        payload["gap_detection"]["provenance"]["fingerprint_schemes"] = {"comparison": "ambiguous"}
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    fake = FakeAgent()
    with pytest.raises((SystemExit, ValueError)):
        _invoke(monkeypatch, artifact, output, fake)
    assert fake.calls == []
    assert not output.exists()
