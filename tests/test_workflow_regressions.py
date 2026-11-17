"""Regression tests for defects fixed during the workflow stabilization audit.

All tests are offline: Gemini, retrieval and Chroma are mocked or stubbed, and
the shared conftest blocks network sockets.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments import run_langgraph
from src.agents import artifact_io
from src.agents.artifact_provenance import ArtifactProvenance
from src.agents.critic_agent import (
    CriticDecision, CriticResult, CriticReviewStatus, GapCriticReview,
    GapOriginAssessment, RevisionSuggestion,
)
from src.agents.cross_paper_comparison_agent import (
    CrossPaperComparisonAgent, EvidenceRef,
)
from src.agents.evidence_validation_agent import (
    EvidenceValidationResult, EvidenceValidationStatus, GapValidationStatus,
)
from src.agents.paper_analysis_agent import PaperAnalysis, PaperAnalysisResult
from src.agents.research_question_agent import (
    GenerationStatus, HypothesisStatus, ResearchQuestionAgent, ResearchQuestionResult,
)
from src.rag.models import RetrievedChunk
from src.workflow.langgraph_app import _apply_revision_suggestions, build_research_graph
from tests import test_research_question_agent as rq
from tests.test_langgraph_resume import resume_artifacts
from tests.test_langgraph_workflow import (
    PAPER_IDS, comparison, critic_result, gap_result, invoke, validation, workflow_agents,
)


# ---------------------------------------------------------------------------
# Research Question Agent: quota handling and eligibility
# ---------------------------------------------------------------------------


class DailyQuotaError(Exception):
    code = 429

    def __init__(self):
        super().__init__("429 RESOURCE_EXHAUSTED: quota exceeded for GenerateContentRequestsPerDay per day")


class MinuteQuotaError(Exception):
    code = 429

    def __init__(self, retry_after=None):
        super().__init__("429 RESOURCE_EXHAUSTED: rate limit per minute")
        if retry_after is not None:
            self.retry_after_seconds = retry_after


def two_accepted_gaps():
    first, second = rq.gap("GAP-001"), rq.gap("GAP-002")
    critic_result_value = CriticResult(
        research_topic="scientific discovery", review_status=CriticReviewStatus.COMPLETE,
        gap_reviews=[GapCriticReview(
            gap_id=gap_id, decision=CriticDecision.ACCEPT,
            review_status=CriticReviewStatus.COMPLETE, origin_assessment="INFERRED",
            origin_rationale="Synthesis.", reasoning="Accepted.",
        ) for gap_id in ("GAP-001", "GAP-002")],
    )
    validation_result = EvidenceValidationResult(
        research_topic="scientific discovery", paper_ids=[rq.P1, rq.P2],
        gap_validations=(rq.validation(gap_id="GAP-001").gap_validations
                         + rq.validation(gap_id="GAP-002").gap_validations),
        validation_counts={}, evidence_status_counts={},
    )
    return critic_result_value, rq.gap_result([first, second]), validation_result


def test_daily_quota_is_not_retried_and_later_gaps_are_not_attempted(monkeypatch):
    sleeps = []
    monkeypatch.setattr("src.agents.research_question_agent.time.sleep", sleeps.append)
    client = rq.SequencedClient([DailyQuotaError()])
    critic_value, gaps, validations = two_accepted_gaps()

    result = ResearchQuestionAgent(client=client, max_retries=5).generate_questions(
        critic_value, gaps, validations,
    )

    assert len(client.calls) == 1          # no retry, and GAP-002 never sent
    assert sleeps == []
    assert result.generation_status is GenerationStatus.INCOMPLETE
    assert [item.gap_id for item in result.skipped_gaps] == ["GAP-001", "GAP-002"]
    assert "HTTP 429" in result.skipped_gaps[0].reason
    assert "not attempted" in result.skipped_gaps[1].reason
    assert "RESOURCE_EXHAUSTED" not in json.dumps(result.model_dump(mode="json"))


def test_per_minute_quota_honours_provider_delay_then_succeeds(monkeypatch):
    sleeps = []
    monkeypatch.setattr("src.agents.research_question_agent.time.sleep", sleeps.append)
    client = rq.SequencedClient([MinuteQuotaError(retry_after=7), rq.llm_response()])

    result = ResearchQuestionAgent(client=client, max_retries=3).generate_questions(
        rq.critic(), rq.gap_result(), rq.validation(),
    )

    assert sleeps == [7.0]
    assert result.generation_status is GenerationStatus.COMPLETE


def test_provider_delay_beyond_bound_fails_closed_without_sleeping(monkeypatch):
    sleeps = []
    monkeypatch.setattr("src.agents.research_question_agent.time.sleep", sleeps.append)
    client = rq.SequencedClient([MinuteQuotaError(retry_after=999)])

    result = ResearchQuestionAgent(
        client=client, max_retries=3, max_backoff_seconds=60,
    ).generate_questions(rq.critic(), rq.gap_result(), rq.validation())

    assert sleeps == []
    assert len(client.calls) == 1
    assert result.generation_status is GenerationStatus.INCOMPLETE
    assert result.gap_results == []


def test_unsupported_cited_chunk_withholds_hypothesis_even_if_gap_labelled_fully_supported():
    validations = rq.validation()
    # Saved artifacts written before the stricter status rule may still label
    # such a gap FULLY_SUPPORTED; hypothesis eligibility must not trust that.
    item = validations.gap_validations[0].evidence_validations[1]
    validations.gap_validations[0].evidence_validations[1] = item.model_copy(
        update={"status": EvidenceValidationStatus.UNSUPPORTED})
    client = rq.FakeClient(rq.llm_response(question_labels=["P1-E1"]))

    result = ResearchQuestionAgent(client=client, max_retries=1).generate_questions(
        rq.critic(), rq.gap_result(), validations,
    )

    output = result.gap_results[0]
    assert output.hypothesis is None
    assert output.hypothesis_status is HypothesisStatus.INSUFFICIENT_EVIDENCE
    assert '"hypothesis_evidence_sufficient": false' in client.calls[0]["contents"]


def test_rejected_gap_is_excluded_and_does_not_block_accepted_gap():
    critic_value, gaps, validations = two_accepted_gaps()
    critic_value.gap_reviews[1] = critic_value.gap_reviews[1].model_copy(
        update={"decision": CriticDecision.REJECT})
    client = rq.FakeClient(rq.llm_response())

    result = ResearchQuestionAgent(client=client, max_retries=1).generate_questions(
        critic_value, gaps, validations,
    )

    assert result.generation_status is GenerationStatus.COMPLETE
    assert [item.gap_id for item in result.gap_results] == ["GAP-001"]
    assert [item.gap_id for item in result.excluded_gaps] == ["GAP-002"]
    assert result.skipped_gaps == []
    assert len(client.calls) == 1
    assert "GAP-002" not in client.calls[0]["contents"]


def test_all_rejected_gaps_are_not_reported_as_complete():
    critic_value, gaps, validations = two_accepted_gaps()
    critic_value.gap_reviews = [review.model_copy(update={"decision": CriticDecision.REJECT})
                                for review in critic_value.gap_reviews]
    client = rq.FakeClient(rq.llm_response())

    result = ResearchQuestionAgent(client=client, max_retries=1).generate_questions(
        critic_value, gaps, validations,
    )

    assert result.generation_status is GenerationStatus.INCOMPLETE
    assert result.gap_results == []
    assert not client.calls


# ---------------------------------------------------------------------------
# Critic revision safety
# ---------------------------------------------------------------------------


def _revise_review(*suggestions):
    return CriticResult(
        research_topic="topic", review_status=CriticReviewStatus.COMPLETE,
        gap_reviews=[GapCriticReview(
            gap_id="GAP-001", decision=CriticDecision.REVISE,
            review_status=CriticReviewStatus.COMPLETE,
            origin_assessment=GapOriginAssessment.INFERRED,
            origin_rationale="r", reasoning="r", revision_suggestions=list(suggestions),
        )],
    )


def test_revision_cannot_reduce_a_gap_to_single_paper_evidence():
    single_paper = RevisionSuggestion(
        field="supporting_evidence_labels", current_issue="narrow",
        proposed_text="P1-E1",
        source_evidence=[EvidenceRef(paper_id=PAPER_IDS[0], chunk_ids=["paper-1-chunk"])],
    )
    revised, changed = _apply_revision_suggestions(
        gap_result(), comparison(), _revise_review(single_paper),
    )
    assert changed == []
    assert revised.gaps[0].supporting_paper_ids == PAPER_IDS


def test_revision_keeps_text_citations_regardless_of_suggestion_order():
    text = RevisionSuggestion(
        field="description", current_issue="unclear", proposed_text="Clearer description.",
        source_evidence=[EvidenceRef(paper_id=PAPER_IDS[1], chunk_ids=["paper-2-chunk"])],
    )
    replace = RevisionSuggestion(
        field="supporting_evidence_labels", current_issue="extra ref", proposed_text="P1-E1",
        source_evidence=[EvidenceRef(paper_id=PAPER_IDS[0], chunk_ids=["paper-1-chunk"])],
    )
    outcomes = []
    for order in ((text, replace), (replace, text)):
        revised, changed = _apply_revision_suggestions(
            gap_result(), comparison(), _revise_review(*order),
        )
        assert changed == ["GAP-001"]
        outcomes.append(revised.gaps[0])
    assert outcomes[0].supporting_evidence == outcomes[1].supporting_evidence
    assert set(outcomes[0].supporting_paper_ids) == set(PAPER_IDS)
    assert outcomes[0].description == "Clearer description."


# ---------------------------------------------------------------------------
# Evidence-to-paper attribution
# ---------------------------------------------------------------------------


def _analysis_with_chunks(*owners):
    return PaperAnalysisResult(
        analysis=PaperAnalysis(problem="p", methodology="m", dataset="d", results="r",
                               limitations="l", future_work="f"),
        evidence=[RetrievedChunk(chunk_id=f"{owner}-chunk-{index}", paper_id=owner, text="t")
                  for index, owner in enumerate(owners)],
    )


def test_comparison_rejects_analysis_with_foreign_chunks():
    agent = CrossPaperComparisonAgent(client=object())
    with pytest.raises(ValueError, match="multiple or unknown papers"):
        agent._prepare([
            _analysis_with_chunks("paper-1", "paper-2"),
            _analysis_with_chunks("paper-2"),
        ])


def test_workflow_excludes_paper_whose_analysis_cites_another_papers_chunk():
    agents = workflow_agents(paper_ids=["paper-1", "paper-2", "paper-3"])
    leaking = {"paper-2": _analysis_with_chunks("paper-2", "paper-1")}
    agents.paper_analysis.analyze_paper.side_effect = lambda paper, **_kw: leaking.get(
        paper.source_id, _analysis_with_chunks(paper.source_id))

    result = invoke(agents, pdf_paths={pid: "p.pdf" for pid in ("paper-1", "paper-2", "paper-3")})

    analysed = [item.evidence[0].paper_id for item in result["paper_analyses"]]
    assert analysed == ["paper-1", "paper-3"]
    excluded = {item.paper_id: item.reason for item in result["comparison"].excluded_papers}
    assert "chunks owned by other papers" in excluded["paper-2"]


# ---------------------------------------------------------------------------
# Resume: stale downstream results and lineage
# ---------------------------------------------------------------------------


def test_evidence_resume_drops_stale_downstream_results(tmp_path):
    original_path, evidence_path, *_ = resume_artifacts(tmp_path)
    original = json.loads(original_path.read_text(encoding="utf-8"))
    original.update({
        "research_questions": {"generation_status": "COMPLETE", "gap_results": []},
        "critic_result": {"review_status": "COMPLETE"},
        "critic_input_fingerprints": {"comparison": "0" * 64},
        "workflow_status": "COMPLETE", "stop_reason": None,
    })
    original_path.write_text(json.dumps(original), encoding="utf-8")

    state = run_langgraph._prepare_evidence_resume(original_path, evidence_path)

    for key in ("research_questions", "critic_result", "critic_input_fingerprints",
                "workflow_status", "stop_reason"):
        assert key not in state


def test_stopped_evidence_resume_does_not_save_previous_questions(tmp_path, monkeypatch):
    original_path, evidence_path, *_ = resume_artifacts(tmp_path)
    original = json.loads(original_path.read_text(encoding="utf-8"))
    original["research_questions"] = ResearchQuestionResult(
        research_topic="topic", generation_status=GenerationStatus.COMPLETE,
    ).model_dump(mode="json")
    original_path.write_text(json.dumps(original), encoding="utf-8")
    agents = workflow_agents()
    agents.critic.critique.side_effect = lambda *_a, **kwargs: critic_result(
        CriticDecision.REJECT, kwargs["input_fingerprints"])
    monkeypatch.setattr(run_langgraph, "default_workflow_agents", lambda **_kw: agents)
    monkeypatch.setattr(sys, "argv", [
        "run_langgraph.py", "--resume-from-evidence", str(original_path),
        "--resume-evidence-validation", str(evidence_path),
        "--output", str(tmp_path / "stopped.json"),
    ])

    assert run_langgraph.main() == 1
    saved = json.loads(next(tmp_path.glob("stopped-resume-*.json")).read_text(encoding="utf-8"))
    assert saved["workflow_status"] == "STOPPED"
    assert "research_questions" not in saved
    lineage = saved["run_lineage"]
    assert lineage[-1]["mode"] == "resume_from_evidence"
    assert lineage[-1]["source_artifact"] == original_path.name
    assert len(lineage[-1]["source_artifact_sha256"]) == 64
    assert lineage[-1]["replacement_evidence_artifact"] == evidence_path.name


def test_graph_passes_run_lineage_through_unchanged():
    agents = workflow_agents()
    lineage = [{"mode": "resume_from_critic", "source_artifact": "x.json"}]
    result = invoke(agents, run_lineage=lineage)
    assert result["run_lineage"] == lineage


# ---------------------------------------------------------------------------
# Artifact writes
# ---------------------------------------------------------------------------


def test_no_overwrite_write_refuses_existing_file(tmp_path):
    target = tmp_path / "artifact.json"
    target.write_text("original", encoding="utf-8")
    with pytest.raises(FileExistsError):
        artifact_io.write_text_no_overwrite(target, "replacement")
    assert target.read_text(encoding="utf-8") == "original"
    assert not list(tmp_path.glob(".artifact.json.*.tmp"))


def test_no_overwrite_write_falls_back_when_hard_links_are_unsupported(tmp_path, monkeypatch):
    def no_links(*_args):
        raise PermissionError("hard links not supported")

    monkeypatch.setattr(artifact_io.os, "link", no_links)
    target = tmp_path / "artifact.json"
    artifact_io.write_text_no_overwrite(target, "content")
    assert target.read_text(encoding="utf-8") == "content"
    with pytest.raises(FileExistsError):
        artifact_io.write_text_no_overwrite(target, "replacement")
    assert target.read_text(encoding="utf-8") == "content"


def test_preserving_write_archives_previous_version(tmp_path):
    target = tmp_path / "critic_review.json"
    assert artifact_io.write_text_preserving_previous(target, "first") is None
    archived = artifact_io.write_text_preserving_previous(target, "second")
    assert archived is not None and archived.parent == tmp_path / "archive"
    assert archived.read_text(encoding="utf-8") == "first"
    assert target.read_text(encoding="utf-8") == "second"


def test_failed_serialization_never_touches_existing_output(tmp_path, monkeypatch):
    target = tmp_path / "critic_review.json"
    target.write_text("keep", encoding="utf-8")

    def broken_fsync(_fd):
        raise OSError("disk full")

    monkeypatch.setattr(artifact_io.os, "fsync", broken_fsync)
    with pytest.raises(OSError):
        artifact_io.write_text_preserving_previous(target, "new")
    assert target.read_text(encoding="utf-8") == "keep"
    assert not (tmp_path / "archive").exists()
    assert not list(tmp_path.glob(".critic_review.json.*.tmp"))


# ---------------------------------------------------------------------------
# Agent key ownership
# ---------------------------------------------------------------------------


def test_analysis_and_planner_agents_do_not_require_keys_until_used(monkeypatch):
    from src.agents import gemini_config
    from src.agents.paper_analysis_agent import PaperAnalysisAgent
    from src.agents.planner_agent import PlannerAgent

    monkeypatch.setattr(gemini_config, "load_dotenv", lambda: None)
    for name in (*gemini_config.GROUP_KEY_ENV.values(),
                 *gemini_config.LEGACY_GROUP_KEY_ENV.values(), "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    analysis, planner = PaperAnalysisAgent(), PlannerAgent()   # must not raise
    with pytest.raises(ValueError, match="GEMINI_API_KEY_ANALYSIS"):
        analysis.client
    with pytest.raises(ValueError, match="GEMINI_API_KEY_ANALYSIS"):
        planner.client
