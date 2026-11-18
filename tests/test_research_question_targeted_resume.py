import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from experiments import run_research_questions as runner
from src.agents.evidence_validation_agent import EvidenceValidationResult, GapSupportStatus, GapValidationStatus
from src.agents.research_question_agent import (
    GapResearchQuestions,
    GenerationStatus,
    HypothesisStatus,
    ResearchQuestionResult,
    SkippedGap,
)
from experiments.run_langgraph import _model_fingerprint


# Sanitized copy of a saved PROMPT-001 critic-resume run in which GAP-004's
# question generation was skipped (HTTP 429); see tests/fixtures/README.md.
SOURCE_ARTIFACT = (Path(__file__).resolve().parent / "fixtures" / "langgraph_runs"
                   / "questions_gap004_skipped.json")


@pytest.fixture
def raw_artifact_copy(tmp_path):
    """Unmodified temporary copy of the saved run."""
    content = SOURCE_ARTIFACT.read_bytes()
    path = tmp_path / "saved-run.json"
    path.write_bytes(content)
    return path, content


def make_gap_accept_eligible(path, gap_id="GAP-004"):
    """In a TEMPORARY copy only: make the target gap pass the strict acceptance rule.

    The saved fixture's GAP-004 cites two PARTIALLY_SUPPORTED chunks, which the
    strict acceptance rule (correctly) refuses. To keep exercising the recovery
    mechanics, every cited record becomes SUPPORTED with a verified quote copied
    verbatim from the stored chunk text, the gap and joint status become
    FULLY_SUPPORTED, and the critic fingerprints are recomputed so the copy is
    internally consistent. The fixture file itself is never modified.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    chunk_text = {item["chunk_id"]: item["text"]
                  for analysis in data["paper_analyses"] for item in analysis["evidence"]}
    validation = EvidenceValidationResult.model_validate(data["evidence_validation"])
    target = next(item for item in validation.gap_validations if item.gap_id == gap_id)
    target.status = GapValidationStatus.FULLY_SUPPORTED
    target.joint_assessment.status = GapSupportStatus.FULLY_SUPPORTED
    for record in target.evidence_validations:
        record.status = runner.EvidenceValidationStatus.SUPPORTED
        text = chunk_text[record.chunk_id]
        record.supported_contribution_quote = text[:80].strip()
        assert record.supported_contribution_quote in text
    data["evidence_validation"] = validation.model_dump(mode="json")
    fingerprint = _model_fingerprint(validation)
    critic = runner.CriticResult.model_validate(data["critic_result"])
    critic.input_fingerprints["evidence_validation"] = fingerprint
    data["critic_result"] = critic.model_dump(mode="json")
    data["critic_input_fingerprints"]["evidence_validation"] = fingerprint
    path.write_text(json.dumps(data), encoding="utf-8")
    return path, path.read_bytes()


@pytest.fixture
def artifact_copy(raw_artifact_copy):
    """Temporary copy whose GAP-004 is accept-eligible (see make_gap_accept_eligible)."""
    return make_gap_accept_eligible(raw_artifact_copy[0])


def test_saved_partially_supported_target_is_refused_before_agent_call(raw_artifact_copy, monkeypatch):
    """The unmodified saved GAP-004 was ACCEPTED while citing PARTIALLY_SUPPORTED
    chunks; the strict acceptance rule refuses to recover questions for it."""
    source, original_bytes = raw_artifact_copy
    agent = Mock()
    monkeypatch.setattr(runner, "ResearchQuestionAgent", lambda: agent)

    with pytest.raises(ValueError, match="strict acceptance rule"):
        runner.recover_questions_for_gap(source, "GAP-004", artifact_path_base(source, "strict"))
    agent.generate_questions.assert_not_called()
    assert source.read_bytes() == original_bytes


def successful_target_result(topic):
    return ResearchQuestionResult(
        research_topic=topic,
        generation_status=GenerationStatus.COMPLETE,
        gap_results=[GapResearchQuestions(
            gap_id="GAP-004",
            gap_title="Targeted recovered gap",
            hypothesis_status=HypothesisStatus.INSUFFICIENT_EVIDENCE,
            hypothesis_limitation="A test placeholder for the mocked generation.",
            generation_status=GenerationStatus.COMPLETE,
        )],
    )


def install_success_agent(monkeypatch, calls):
    agent = Mock()

    def generate(critic, gaps, validation, *, critic_input_fingerprints):
        calls.append((critic, gaps, validation, critic_input_fingerprints))
        assert [gap.gap_id for gap in gaps.gaps] == ["GAP-004"]
        assert [review.gap_id for review in critic.gap_reviews] == ["GAP-004"]
        assert [item.gap_id for item in validation.gap_validations] == ["GAP-004"]
        return successful_target_result(gaps.research_topic)

    agent.generate_questions.side_effect = generate
    monkeypatch.setattr(runner, "ResearchQuestionAgent", lambda: agent)
    return agent


def test_cli_recovers_only_target_gap_merges_and_preserves_prior_results(
    artifact_copy, tmp_path, monkeypatch, capsys,
):
    source, original_bytes = artifact_copy
    original = json.loads(original_bytes)
    prior = ResearchQuestionResult.model_validate(original["research_questions"])
    prior_results = {
        item.gap_id: item.model_dump(mode="json") for item in prior.gap_results
    }
    calls = []
    agent = install_success_agent(monkeypatch, calls)
    output_base = tmp_path / "questions-recovery.json"
    monkeypatch.setattr(sys, "argv", [
        "run_research_questions.py", "--resume-from-langgraph", str(source),
        "--resume-research-questions", "GAP-004", "--output", str(output_base),
    ])

    runner.main()

    assert agent.generate_questions.call_count == 1
    assert len(calls) == 1
    output_line = next(line for line in capsys.readouterr().out.splitlines()
                       if line.startswith("Saved: "))
    output_path = Path(output_line.removeprefix("Saved: "))
    assert output_path.exists() and output_path != source
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    merged = ResearchQuestionResult.model_validate(saved["research_questions"])
    merged_by_id = {item.gap_id: item for item in merged.gap_results}
    assert merged.generation_status is GenerationStatus.COMPLETE
    assert saved["workflow_status"] == "COMPLETE"
    assert saved["stop_reason"] is None
    assert set(merged_by_id) == {"GAP-001", "GAP-002", "GAP-003", "GAP-004"}
    for gap_id, old_value in prior_results.items():
        assert merged_by_id[gap_id].model_dump(mode="json") == old_value
    assert not merged.skipped_gaps
    assert source.read_bytes() == original_bytes


def test_stale_provenance_is_rejected_before_agent_call(artifact_copy, monkeypatch):
    source, _ = artifact_copy
    data = json.loads(source.read_text(encoding="utf-8"))
    data["gap_detection"]["provenance"]["input_fingerprints"]["comparison"] = "0" * 64
    source.write_text(json.dumps(data), encoding="utf-8")
    agent = Mock()
    monkeypatch.setattr(runner, "ResearchQuestionAgent", lambda: agent)

    with pytest.raises(ValueError, match="stale"):
        runner.recover_questions_for_gap(source, "GAP-004", artifact_path_base(source, "stale"))
    agent.generate_questions.assert_not_called()


def artifact_path_base(source, name):
    return source.with_name(f"{name}.json")


def test_ineligible_target_evidence_is_rejected_before_agent_call(artifact_copy, monkeypatch):
    source, _ = artifact_copy
    data = json.loads(source.read_text(encoding="utf-8"))
    validation = EvidenceValidationResult.model_validate(data["evidence_validation"])
    target = next(item for item in validation.gap_validations if item.gap_id == "GAP-004")
    target.status = GapValidationStatus.UNSUPPORTED
    target.joint_assessment.status = GapSupportStatus.UNSUPPORTED
    for record in target.evidence_validations:
        record.status = runner.EvidenceValidationStatus.UNSUPPORTED
    data["evidence_validation"] = validation.model_dump(mode="json")
    critic = runner.CriticResult.model_validate(data["critic_result"])
    critic.input_fingerprints["evidence_validation"] = _model_fingerprint(validation)
    data["critic_result"] = critic.model_dump(mode="json")
    source.write_text(json.dumps(data), encoding="utf-8")
    agent = Mock()
    monkeypatch.setattr(runner, "ResearchQuestionAgent", lambda: agent)

    with pytest.raises(ValueError, match="not eligible"):
        runner.recover_questions_for_gap(source, "GAP-004", artifact_path_base(source, "unsupported"))
    agent.generate_questions.assert_not_called()


def test_api_failure_keeps_prior_results_and_target_skipped(artifact_copy, tmp_path, monkeypatch):
    source, original_bytes = artifact_copy
    original = json.loads(original_bytes)
    prior = ResearchQuestionResult.model_validate(original["research_questions"])
    prior_results = {item.gap_id: item.model_dump(mode="json") for item in prior.gap_results}
    failed = ResearchQuestionResult(
        research_topic=prior.research_topic,
        generation_status=GenerationStatus.INCOMPLETE,
        skipped_gaps=[SkippedGap(gap_id="GAP-004", reason="Model assessment failed: quota exhausted.")],
        errors=["GAP-004: Model assessment failed: quota exhausted."],
    )
    agent = Mock()
    agent.generate_questions.return_value = failed
    monkeypatch.setattr(runner, "ResearchQuestionAgent", lambda: agent)

    output_path, merged = runner.recover_questions_for_gap(
        source, "GAP-004", tmp_path / "retry-failure.json",
    )

    assert output_path.exists()
    assert merged.generation_status is GenerationStatus.INCOMPLETE
    assert [item.gap_id for item in merged.skipped_gaps] == ["GAP-004"]
    assert {item.gap_id: item.model_dump(mode="json") for item in merged.gap_results} == prior_results
    assert source.read_bytes() == original_bytes


def test_unique_output_collision_is_skipped_without_overwrite(artifact_copy, tmp_path, monkeypatch):
    source, _ = artifact_copy
    calls = []
    install_success_agent(monkeypatch, calls)
    occupied = tmp_path / "already-created.json"
    occupied.write_text("keep this file", encoding="utf-8")
    fresh = tmp_path / "new-unique-output.json"
    candidates = iter((occupied, fresh))
    monkeypatch.setattr(runner, "_evaluation_artifact_path", lambda *_args: next(candidates))

    output_path, merged = runner.recover_questions_for_gap(source, "GAP-004")

    assert output_path == fresh
    assert occupied.read_text(encoding="utf-8") == "keep this file"
    assert fresh.exists()
    assert merged.generation_status is GenerationStatus.COMPLETE


# ---------------------------------------------------------------------------
# Regression tests added during the stabilization audit
# ---------------------------------------------------------------------------

import hashlib

from src.agents.research_question_agent import (
    ResearchQuestionAgent as RealQuestionAgent,
    _LLMQuestionSet,
    _LLMResearchQuestion,
)


class _SequencedClient:
    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), []
        self.models = type("Models", (), {"generate_content": self.generate})()

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return type("Response", (), {"parsed": outcome})()


class _DailyQuotaError(Exception):
    code = 429

    def __init__(self):
        super().__init__("429 RESOURCE_EXHAUSTED quota exceeded per day")


def _install_real_agent(monkeypatch, client):
    monkeypatch.setattr(runner, "ResearchQuestionAgent",
                        lambda: RealQuestionAgent(client=client, max_retries=3))


def test_real_agent_quota_failure_makes_one_call_and_preserves_everything(
    artifact_copy, tmp_path, monkeypatch,
):
    source, original_bytes = artifact_copy
    prior = ResearchQuestionResult.model_validate(json.loads(original_bytes)["research_questions"])
    client = _SequencedClient([_DailyQuotaError()])
    _install_real_agent(monkeypatch, client)

    output_path, merged = runner.recover_questions_for_gap(source, "GAP-004", tmp_path / "out.json")

    assert len(client.calls) == 1                       # no retries on daily quota
    assert "GAP-001" not in client.calls[0]["contents"]  # only the target gap is sent
    assert merged.generation_status is GenerationStatus.INCOMPLETE
    assert merged.gap_results == prior.gap_results
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["workflow_status"] == "STOPPED"
    assert saved["stop_reason"] == runner.QUESTION_INCOMPLETE_STOP_REASON
    entry = saved["run_lineage"][-1]
    assert entry["mode"] == "targeted_question_recovery"
    assert entry["gap_id"] == "GAP-004"
    assert entry["outcome"] == "NOT_GENERATED"
    assert "HTTP 429" in entry["reason"]
    assert entry["source_artifact_sha256"] == hashlib.sha256(original_bytes).hexdigest()
    assert entry["preserved_gap_ids"] == ["GAP-001", "GAP-002", "GAP-003"]
    assert source.read_bytes() == original_bytes


def test_real_agent_success_merges_target_and_withholds_single_paper_hypothesis(
    artifact_copy, tmp_path, monkeypatch,
):
    source, original_bytes = artifact_copy
    prior = ResearchQuestionResult.model_validate(json.loads(original_bytes)["research_questions"])
    response = _LLMQuestionSet(research_questions=[_LLMResearchQuestion(
        question="How could a multi-module feedback framework be evaluated across both settings?",
        rationale="Bounded by the cited contributions.",
        evidence_labels=["P1-E1", "P2-E1"],
    )], hypothesis=None, hypothesis_limitation=None)
    client = _SequencedClient([response])
    _install_real_agent(monkeypatch, client)

    output_path, merged = runner.recover_questions_for_gap(source, "GAP-004", tmp_path / "ok.json")

    assert len(client.calls) == 1
    assert merged.generation_status is GenerationStatus.COMPLETE
    assert [item.gap_id for item in merged.gap_results] == ["GAP-001", "GAP-002", "GAP-003", "GAP-004"]
    assert merged.gap_results[:3] == prior.gap_results
    recovered = merged.gap_results[3]
    # GAP-004's SUPPORTED chunks all come from one paper: no hypothesis.
    assert recovered.hypothesis is None
    assert recovered.hypothesis_status is HypothesisStatus.INSUFFICIENT_EVIDENCE
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    for question in saved["research_questions"]["gap_results"][3]["research_questions"]:
        for ref in question["source_evidence"]:
            target_gap = next(g for g in saved["gap_detection"]["gaps"] if g["gap_id"] == "GAP-004")
            allowed = {(r["paper_id"], c) for r in target_gap["supporting_evidence"] for c in r["chunk_ids"]}
            assert all((ref["paper_id"], chunk) in allowed for chunk in ref["chunk_ids"])
    assert saved["workflow_status"] == "COMPLETE" and saved["stop_reason"] is None
    assert saved["run_lineage"][-1]["outcome"] == "GENERATED"
    assert source.read_bytes() == original_bytes


def test_recovery_can_be_chained_from_a_failed_recovery_artifact(artifact_copy, tmp_path, monkeypatch):
    source, _ = artifact_copy
    _install_real_agent(monkeypatch, _SequencedClient([_DailyQuotaError()]))
    first_path, _ = runner.recover_questions_for_gap(source, "GAP-004", tmp_path / "first.json")
    first_bytes = first_path.read_bytes()
    calls = []
    install_success_agent(monkeypatch, calls)

    second_path, merged = runner.recover_questions_for_gap(first_path, "GAP-004", tmp_path / "second.json")

    assert merged.generation_status is GenerationStatus.COMPLETE
    lineage = json.loads(second_path.read_text(encoding="utf-8"))["run_lineage"]
    assert [entry["outcome"] for entry in lineage] == ["NOT_GENERATED", "GENERATED"]
    assert lineage[1]["source_artifact"] == first_path.name
    assert first_path.read_bytes() == first_bytes


@pytest.mark.parametrize("gap_id", ["GAP-001", "GAP-999"])
def test_completed_or_unknown_gap_cannot_be_regenerated(artifact_copy, monkeypatch, gap_id):
    source, _ = artifact_copy
    agent = Mock()
    monkeypatch.setattr(runner, "ResearchQuestionAgent", lambda: agent)
    with pytest.raises(ValueError):
        runner.recover_questions_for_gap(source, gap_id, artifact_path_base(source, "x"))
    agent.generate_questions.assert_not_called()


def test_tampered_critic_fingerprint_is_rejected_before_agent_call(artifact_copy, monkeypatch):
    source, _ = artifact_copy
    data = json.loads(source.read_text(encoding="utf-8"))
    data["critic_result"]["input_fingerprints"]["evidence_validation"] = "f" * 64
    source.write_text(json.dumps(data), encoding="utf-8")
    agent = Mock()
    monkeypatch.setattr(runner, "ResearchQuestionAgent", lambda: agent)
    with pytest.raises(ValueError, match="stale"):
        runner.recover_questions_for_gap(source, "GAP-004", artifact_path_base(source, "t"))
    agent.generate_questions.assert_not_called()


def test_target_without_critic_accept_is_rejected_before_agent_call(artifact_copy, monkeypatch):
    source, _ = artifact_copy
    data = json.loads(source.read_text(encoding="utf-8"))
    for review in data["critic_result"]["gap_reviews"]:
        if review["gap_id"] == "GAP-004":
            review["decision"] = "REVISE"
    source.write_text(json.dumps(data), encoding="utf-8")
    agent = Mock()
    monkeypatch.setattr(runner, "ResearchQuestionAgent", lambda: agent)
    with pytest.raises(ValueError, match="ACCEPT"):
        runner.recover_questions_for_gap(source, "GAP-004", artifact_path_base(source, "r"))
    agent.generate_questions.assert_not_called()


def test_merge_refuses_to_alter_previous_gap_results():
    previous = ResearchQuestionResult(
        research_topic="t", generation_status=GenerationStatus.INCOMPLETE,
        gap_results=[successful_target_result("t").gap_results[0].model_copy(
            update={"gap_id": "GAP-001"})],
        skipped_gaps=[SkippedGap(gap_id="GAP-002", reason="quota")],
    )
    smuggled = successful_target_result("t").model_copy(update={"gap_results": [
        successful_target_result("t").gap_results[0].model_copy(update={"gap_id": "GAP-001"}),
    ]})
    with pytest.raises(ValueError, match="invalid per-gap result"):
        runner._merge_target_result(previous, smuggled, "GAP-002", ["GAP-001", "GAP-002"])


def test_recovered_gap_is_placed_in_original_gap_order(artifact_copy, tmp_path, monkeypatch):
    source, _ = artifact_copy
    data = json.loads(source.read_text(encoding="utf-8"))
    questions = data["research_questions"]
    gap2 = next(item for item in questions["gap_results"] if item["gap_id"] == "GAP-002")
    questions["gap_results"].remove(gap2)
    questions["skipped_gaps"].insert(0, {"gap_id": "GAP-002", "reason": "quota"})
    source.write_text(json.dumps(data), encoding="utf-8")
    agent = Mock()
    agent.generate_questions.side_effect = lambda critic, gaps, validation, **_: (
        successful_target_result(gaps.research_topic).model_copy(update={"gap_results": [
            successful_target_result(gaps.research_topic).gap_results[0].model_copy(
                update={"gap_id": "GAP-002"})]}))
    monkeypatch.setattr(runner, "ResearchQuestionAgent", lambda: agent)

    _, merged = runner.recover_questions_for_gap(source, "GAP-002", tmp_path / "order.json")

    assert [item.gap_id for item in merged.gap_results] == ["GAP-001", "GAP-002", "GAP-003"]
    assert [item.gap_id for item in merged.skipped_gaps] == ["GAP-004"]
    assert merged.generation_status is GenerationStatus.INCOMPLETE


def test_non_evaluation_langgraph_artifact_supports_targeted_recovery(
    artifact_copy, tmp_path, monkeypatch,
):
    source, _ = artifact_copy
    data = json.loads(source.read_text(encoding="utf-8"))
    for key in ("evaluation_prompt_id", "canonical_prompt_text", "evaluation_papers",
                "evaluation_expected_paper_ids", "evaluation_chroma_dir",
                "evaluation_chroma_collection"):
        data.pop(key, None)
    source.write_text(json.dumps(data), encoding="utf-8")
    calls = []
    install_success_agent(monkeypatch, calls)

    output_path, merged = runner.recover_questions_for_gap(source, "GAP-004", tmp_path / "plain.json")

    assert merged.generation_status is GenerationStatus.COMPLETE
    assert "-recovery-" in output_path.name
    assert len(calls) == 1


def test_previously_excluded_gaps_count_as_terminal_during_recovery(artifact_copy, tmp_path, monkeypatch):
    source, _ = artifact_copy
    data = json.loads(source.read_text(encoding="utf-8"))
    for review in data["critic_result"]["gap_reviews"]:
        if review["gap_id"] == "GAP-003":
            review["decision"] = "REJECT"
    questions = data["research_questions"]
    questions["gap_results"] = [item for item in questions["gap_results"] if item["gap_id"] != "GAP-003"]
    questions["excluded_gaps"] = [{"gap_id": "GAP-003", "reason": "Critic REJECT."}]
    source.write_text(json.dumps(data), encoding="utf-8")
    calls = []
    install_success_agent(monkeypatch, calls)

    _, merged = runner.recover_questions_for_gap(source, "GAP-004", tmp_path / "excl.json")

    assert merged.generation_status is GenerationStatus.COMPLETE
    assert [item.gap_id for item in merged.excluded_gaps] == ["GAP-003"]
    assert [item.gap_id for item in merged.gap_results] == ["GAP-001", "GAP-002", "GAP-004"]
