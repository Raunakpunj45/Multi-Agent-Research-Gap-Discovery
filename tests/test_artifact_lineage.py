import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import experiments.run_evidence_validation as evidence_runner
import experiments.run_research_gap_detection as gap_runner
import experiments.run_multi_paper_comparison as comparison_runner
from src.agents import MultiPaperComparison, ResearchGapDetectionResult
from src.agents.artifact_provenance import (
    ArtifactProvenance,
    fingerprint_json,
    require_fingerprint,
)


def comparison(provenance=True):
    return MultiPaperComparison(
        paper_ids=["P1", "P2"], paper_count=2, dimension_summaries=[],
        recurring_patterns=[], common_limitations=[], conflicting_findings=[],
        methodological_differences=[], underexplored_directions=[], caveats=[],
        provenance=(ArtifactProvenance(
            stage="cross_paper_comparison",
            input_fingerprints={"paper_analyses": "a" * 64},
        ) if provenance else None),
    )


def test_canonical_input_fingerprint_is_deterministic_and_changes_with_input():
    first = {"paper_ids": ["P1", "P2"], "query": "analysis"}
    reordered = {"query": "analysis", "paper_ids": ["P1", "P2"]}
    changed = {"query": "different analysis", "paper_ids": ["P1", "P2"]}
    assert fingerprint_json(first) == fingerprint_json(reordered)
    assert fingerprint_json(first) != fingerprint_json(changed)


@pytest.mark.parametrize("provenance,expected", [
    (None, "missing provenance"),
    (ArtifactProvenance(stage="gap", input_fingerprints={}), "missing the 'comparison'"),
    (ArtifactProvenance(stage="gap", input_fingerprints={"comparison": "old"}), "stale"),
])
def test_lineage_rejects_missing_and_stale_fingerprints(provenance, expected):
    with pytest.raises(ValueError, match=expected):
        require_fingerprint(provenance, "comparison", "current", "Gap artifact")


def test_gap_runner_atomically_writes_comparison_fingerprint(tmp_path, monkeypatch):
    input_path = tmp_path / "comparison.json"
    output_path = tmp_path / "gaps.json"
    raw = comparison().model_dump_json(indent=2).encode("utf-8")
    input_path.write_bytes(raw)
    output_path.write_text("previous", encoding="utf-8")
    result = ResearchGapDetectionResult(
        research_topic="topic", paper_count=2, paper_ids=["P1", "P2"],
        gaps=[], gap_category_counts={}, caveats=[],
    )

    class StubAgent:
        def detect_gaps(self, *_args, **_kwargs):
            return result

    monkeypatch.setattr(gap_runner, "INPUT_PATH", input_path)
    monkeypatch.setattr(gap_runner, "OUTPUT_PATH", output_path)
    monkeypatch.setattr(gap_runner, "ResearchGapDetectionAgent", StubAgent)
    monkeypatch.setattr(sys, "argv", ["run_research_gap_detection.py"])
    original_replace = gap_runner.os.replace
    replacements = []

    def check_replace(source, destination):
        source, destination = Path(source), Path(destination)
        assert output_path.read_text(encoding="utf-8") == "previous"
        payload = json.loads(source.read_text(encoding="utf-8"))
        assert payload["provenance"]["input_fingerprints"]["comparison"] == hashlib.sha256(raw).hexdigest()
        replacements.append(True)
        original_replace(source, destination)

    monkeypatch.setattr(gap_runner.os, "replace", check_replace)
    gap_runner.main()
    saved = ResearchGapDetectionResult.model_validate_json(output_path.read_text(encoding="utf-8"))
    assert saved.provenance.input_fingerprints["comparison"] == hashlib.sha256(raw).hexdigest()
    assert replacements == [True]
    assert not list(tmp_path.glob(".gaps.json.*.tmp"))


def test_comparison_runner_atomically_writes_output(tmp_path, monkeypatch):
    output_path = tmp_path / "comparison.json"
    output_path.write_text("previous", encoding="utf-8")

    class StubPipeline:
        def analyze_and_compare_papers(self, **_kwargs):
            return comparison()

    monkeypatch.setattr(
        comparison_runner, "MultiSourceRetriever",
        lambda **_kwargs: SimpleNamespace(search=lambda **_search: []),
    )
    monkeypatch.setattr(comparison_runner, "PaperAnalysisPipeline", StubPipeline)
    monkeypatch.setattr(comparison_runner, "OUTPUT_PATH", output_path)
    monkeypatch.setattr(sys, "argv", ["run_multi_paper_comparison.py", "topic"])
    original_replace = comparison_runner.os.replace
    replacements = []

    def check_replace(source, destination):
        source, destination = Path(source), Path(destination)
        assert output_path.read_text(encoding="utf-8") == "previous"
        saved = MultiPaperComparison.model_validate_json(source.read_text(encoding="utf-8"))
        assert saved.provenance.input_fingerprints["paper_analyses"] == "a" * 64
        replacements.append(True)
        original_replace(source, destination)

    monkeypatch.setattr(comparison_runner.os, "replace", check_replace)
    comparison_runner.main()
    assert len(replacements) == 1
    assert MultiPaperComparison.model_validate_json(
        output_path.read_text(encoding="utf-8")
    ).provenance.stage == "cross_paper_comparison"
    assert not list(tmp_path.glob(".comparison.json.*.tmp"))


def test_comparison_runner_planner_route_is_explicit_opt_in(tmp_path, monkeypatch):
    output_path = tmp_path / "comparison.json"
    calls = []

    class StubPlannerRetrieval:
        def search(self, topic, max_results_per_source):
            calls.append((topic, max_results_per_source))
            return (object(), [])

    class StubPipeline:
        def analyze_and_compare_papers(self, **_kwargs):
            return comparison()

    monkeypatch.setattr(comparison_runner, "PlannerRetrievalAgent", StubPlannerRetrieval)
    monkeypatch.setattr(
        comparison_runner, "MultiSourceRetriever",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("default retrieval used")),
    )
    monkeypatch.setattr(comparison_runner, "PaperAnalysisPipeline", StubPipeline)
    monkeypatch.setattr(comparison_runner, "OUTPUT_PATH", output_path)
    monkeypatch.setattr(sys, "argv", [
        "run_multi_paper_comparison.py", "topic", "--per-source", "7", "--use-planner",
    ])

    comparison_runner.main()

    assert calls == [("topic", 7)]
    assert MultiPaperComparison.model_validate_json(
        output_path.read_text(encoding="utf-8")
    ).provenance is not None


@pytest.mark.parametrize("comparison_lineage", [None, "stale"])
def test_evidence_runner_rejects_missing_or_stale_gap_lineage(
    tmp_path, monkeypatch, comparison_lineage,
):
    comparison_path = tmp_path / "comparison.json"
    gap_path = tmp_path / "gaps.json"
    output_path = tmp_path / "validation.json"
    comparison_path.write_bytes(comparison().model_dump_json().encode("utf-8"))
    expected = hashlib.sha256(comparison_path.read_bytes()).hexdigest()
    fingerprint = None if comparison_lineage is None else (
        "b" * 64 if comparison_lineage == "stale" else expected
    )
    gap_result = ResearchGapDetectionResult(
        research_topic="topic", paper_count=2, paper_ids=["P1", "P2"],
        gaps=[], gap_category_counts={}, caveats=[],
        provenance=(ArtifactProvenance(
            stage="research_gap_detection",
            input_fingerprints={"comparison": fingerprint} if fingerprint else {},
        ) if comparison_lineage is not None else None),
    )
    gap_path.write_text(gap_result.model_dump_json(), encoding="utf-8")
    called = []

    class StubAgent:
        def __init__(self):
            pass

        def validate(self, _gaps):
            called.append(True)
            raise AssertionError("Validation must not run for stale inputs.")

    monkeypatch.setattr(evidence_runner, "EvidenceValidationAgent", StubAgent)
    monkeypatch.setattr(sys, "argv", [
        "run_evidence_validation.py", "--input", str(gap_path),
        "--comparison", str(comparison_path), "--output", str(output_path),
    ])
    with pytest.raises(SystemExit) as exit_info:
        evidence_runner.main()
    assert exit_info.value.code == 2
    assert called == []
    assert not output_path.exists()
