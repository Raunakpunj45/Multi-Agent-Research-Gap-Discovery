"""Phase 1 reliability: fixed output budgets, six-paper default, stage
checkpoints with automatic resume, and the dry-run estimator. Offline only."""
import json

import pytest
from google.genai import types
from pydantic import BaseModel

from src.agents.critic_agent import CriticDecision
from src.agents.groq_client import GroqRequestTooLargeError, estimate_request_tokens, tpm_budget
from src.agents.output_budget import (
    TASK_OUTPUT_BUDGETS,
    generation_config,
    output_budget,
)
from src.workflow.checkpoint import (
    CheckpointError,
    RunCheckpoint,
    default_run_dir,
    deserialize_state,
    serialize_state,
)
from src.workflow.dry_run import DryRunAssumptions, RecordingClient, estimate_run, format_report
from src.workflow.langgraph_app import DEFAULT_MAX_PAPERS, build_research_graph
from src.workflow.runner import (
    RunInterrupted,
    TopicRunConfig,
    dry_run_topic,
    run_topic,
)
from tests.test_groq_provider import FakeGroqAPI, completion, groq_client
from tests.test_langgraph_workflow import analysis, paper_records, workflow_agents

MODEL = "openai/gpt-oss-120b"


class SimpleSchema(BaseModel):
    queries: list[str]


def config(**overrides):
    values = dict(research_topic="topic", analysis_query="query", max_papers=6)
    values.update(overrides)
    return TopicRunConfig(**values)


# ---------------------------------------------------------------------------
# Fixed output budgets
# ---------------------------------------------------------------------------

def test_every_agent_task_has_a_budget_and_env_override(monkeypatch):
    assert output_budget("gap_detection") == TASK_OUTPUT_BUDGETS["gap_detection"]
    monkeypatch.setenv("LLM_OUTPUT_BUDGET_GAP_DETECTION", "1800")
    assert output_budget("gap_detection") == 1800
    with pytest.raises(ValueError):
        output_budget("unknown_task")


def test_groq_configs_carry_the_task_budget_and_gemini_keeps_its_default(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    assert generation_config("critic").max_output_tokens == TASK_OUTPUT_BUDGETS["critic"]
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    assert generation_config("critic").max_output_tokens is None
    monkeypatch.setenv("LLM_OUTPUT_BUDGETS_FOR_GEMINI", "1")
    assert generation_config("critic").max_output_tokens == TASK_OUTPUT_BUDGETS["critic"]


def test_fixed_budget_is_reserved_exactly_not_shrunk(monkeypatch):
    api = FakeGroqAPI([completion('{"queries": []}')])
    client = groq_client(monkeypatch, api)
    client.models.generate_content(
        model=MODEL, contents="word " * 2000,
        config=types.GenerateContentConfig(response_schema=SimpleSchema, max_output_tokens=2048),
    )
    request = api.requests[0]
    assert request["max_completion_tokens"] == 2048
    assert estimate_request_tokens(request) + 2048 <= tpm_budget(MODEL)


def test_input_that_leaves_no_room_for_the_fixed_budget_is_never_sent(monkeypatch):
    api = FakeGroqAPI([completion('{"queries": []}')])
    client = groq_client(monkeypatch, api)
    # ~5k input tokens fit the old shrinking cap (>= 1k left) but not a 3k budget.
    with pytest.raises(GroqRequestTooLargeError) as raised:
        client.models.generate_content(
            model=MODEL, contents="word " * 4000,
            config=types.GenerateContentConfig(response_schema=SimpleSchema,
                                               max_output_tokens=3072),
        )
    assert api.requests == []
    assert "fixed output budget" in str(raised.value)


# ---------------------------------------------------------------------------
# Six-paper default and backfill
# ---------------------------------------------------------------------------

def _invoke(agents, **state):
    base = {"research_topic": "topic", "analysis_query": "query", "minimum_papers": 2,
            "max_revisions": 2, "revision_count": 0, "pdf_paths": {}}
    base.update(state)
    return build_research_graph(agents).invoke(base, config={"recursion_limit": 30})


def test_topic_runs_analyze_at_most_six_papers_by_default():
    ids = [f"paper-{n}" for n in range(1, 9)]
    agents = workflow_agents(papers=paper_records(8), paper_ids=ids)
    agents.paper_analysis._resolve_pdf_path.side_effect = lambda paper, pid, paths: "x.pdf"
    result = _invoke(agents)
    assert DEFAULT_MAX_PAPERS == 6
    assert agents.paper_analysis.analyze_paper.call_count == 6
    assert len(result["paper_analyses"]) == 6


def test_a_failed_paper_is_replaced_by_the_next_candidate():
    ids = [f"paper-{n}" for n in range(1, 9)]
    agents = workflow_agents(papers=paper_records(8), paper_ids=ids)
    agents.paper_analysis._resolve_pdf_path.side_effect = lambda paper, pid, paths: "x.pdf"

    def analyze(paper, **_):
        if paper.source_id == "paper-2":
            raise RuntimeError("no full text")
        return analysis(paper.source_id)

    agents.paper_analysis.analyze_paper.side_effect = analyze
    result = _invoke(agents)
    analyzed = [item.evidence[0].paper_id for item in result["paper_analyses"]]
    assert analyzed == ["paper-1", "paper-3", "paper-4", "paper-5", "paper-6", "paper-7"]
    assert [item.paper_id for item in result["excluded_papers"]] == ["paper-2"]


# ---------------------------------------------------------------------------
# Checkpoints and automatic resume
# ---------------------------------------------------------------------------

def test_state_round_trips_with_identical_artifacts(tmp_path):
    agents = workflow_agents()
    agents.paper_analysis._resolve_pdf_path.side_effect = lambda paper, pid, paths: "x.pdf"
    result = run_topic(config(), run_dir=tmp_path / "run", agents=agents).state
    restored = deserialize_state(json.loads(json.dumps(serialize_state(result))))
    for key in ("comparison", "gap_detection", "evidence_validation", "critic_result"):
        assert restored[key] == result[key]
    assert restored["papers"] == result["papers"]


def test_each_completed_step_is_checkpointed(tmp_path):
    agents = workflow_agents()
    agents.paper_analysis._resolve_pdf_path.side_effect = lambda paper, pid, paths: "x.pdf"
    outcome = run_topic(config(), run_dir=tmp_path / "run", agents=agents)
    assert outcome.state["workflow_status"] == "COMPLETE"
    checkpoint = RunCheckpoint(tmp_path / "run")
    state, stage = checkpoint.load()
    assert stage == "questions"
    # Seven steps saved (retrieve ... questions); each save archives the previous one.
    assert len(list((tmp_path / "run" / "archive").iterdir())) == 6
    assert state["research_questions"] == outcome.state["research_questions"]


def test_failed_step_resumes_without_repeating_completed_steps(tmp_path):
    agents = workflow_agents()
    agents.paper_analysis._resolve_pdf_path.side_effect = lambda paper, pid, paths: "x.pdf"
    good = agents.gap_detection.detect_gaps.return_value
    agents.gap_detection.detect_gaps.side_effect = RuntimeError("output truncated")

    with pytest.raises(RunInterrupted) as raised:
        run_topic(config(), run_dir=tmp_path / "run", agents=agents)
    assert raised.value.last_completed == "compare"

    agents.gap_detection.detect_gaps.side_effect = None
    agents.gap_detection.detect_gaps.return_value = good
    outcome = run_topic(config(), run_dir=tmp_path / "run", agents=agents)

    assert outcome.resumed_from == "compare"
    assert outcome.state["workflow_status"] == "COMPLETE"
    assert agents.retrieval.search.call_count == 1
    assert agents.paper_analysis.analyze_paper.call_count == 2
    assert agents.comparison.compare_many.call_count == 1
    lineage = outcome.state["run_lineage"]
    assert lineage[-1]["action"] == "auto_resume"
    assert lineage[-1]["resumed_after_step"] == "compare"
    assert len(lineage[-1]["checkpoint_sha256"]) == 64


class _Crash(BaseException):
    """Simulates the process dying mid-analysis (not caught per paper)."""


def test_interrupted_analysis_reanalyzes_only_missing_papers(tmp_path):
    ids = ["paper-1", "paper-2", "paper-3"]
    agents = workflow_agents(papers=paper_records(3), paper_ids=ids)
    agents.paper_analysis._resolve_pdf_path.side_effect = lambda paper, pid, paths: "x.pdf"
    calls = []

    def analyze(paper, **_):
        calls.append(paper.source_id)
        if paper.source_id == "paper-3" and calls.count("paper-3") == 1:
            raise _Crash()
        return analysis(paper.source_id)

    agents.paper_analysis.analyze_paper.side_effect = analyze
    with pytest.raises(_Crash):
        run_topic(config(), run_dir=tmp_path / "run", agents=agents)
    _, stage = RunCheckpoint(tmp_path / "run").load()
    assert stage == "analyze:partial"

    outcome = run_topic(config(), run_dir=tmp_path / "run", agents=agents)
    assert calls == ["paper-1", "paper-2", "paper-3", "paper-3"]
    assert [a.evidence[0].paper_id for a in outcome.state["paper_analyses"]] == ids
    assert outcome.state["workflow_status"] == "COMPLETE"


def test_completed_run_is_not_rerun_and_fresh_archives_it(tmp_path):
    agents = workflow_agents(critic_decisions=[CriticDecision.ACCEPT, CriticDecision.ACCEPT])
    agents.paper_analysis._resolve_pdf_path.side_effect = lambda paper, pid, paths: "x.pdf"
    run_topic(config(), run_dir=tmp_path / "run", agents=agents)
    again = run_topic(config(), run_dir=tmp_path / "run", agents=agents)
    assert again.state["workflow_status"] == "COMPLETE"
    assert agents.critic.critique.call_count == 1  # nothing re-requested

    archived_before = len(list((tmp_path / "run" / "archive").iterdir()))
    run_topic(config(), run_dir=tmp_path / "run", agents=agents, fresh=True)
    assert agents.retrieval.search.call_count == 2
    assert len(list((tmp_path / "run" / "archive").iterdir())) > archived_before


def test_research_outcome_stops_are_final_but_incomplete_work_is_retried(tmp_path):
    agents = workflow_agents(critic_decisions=[CriticDecision.REJECT])
    agents.paper_analysis._resolve_pdf_path.side_effect = lambda paper, pid, paths: "x.pdf"
    first = run_topic(config(), run_dir=tmp_path / "rejected", agents=agents)
    assert first.state["stop_reason"] == "Critic rejected all candidate gaps."
    run_topic(config(), run_dir=tmp_path / "rejected", agents=agents)
    assert agents.critic.critique.call_count == 1  # a rejection is not retried


def test_checkpoint_from_other_settings_is_refused(tmp_path):
    agents = workflow_agents()
    agents.paper_analysis._resolve_pdf_path.side_effect = lambda paper, pid, paths: "x.pdf"
    run_topic(config(), run_dir=tmp_path / "run", agents=agents)
    with pytest.raises(CheckpointError):
        run_topic(config(research_topic="another topic"), run_dir=tmp_path / "run", agents=agents)


def test_default_run_dir_is_stable_per_topic_and_settings(tmp_path):
    settings = {"research_topic": "Graph neural networks!", "max_papers": 6}
    first = default_run_dir(settings, tmp_path)
    assert first == default_run_dir(dict(settings), tmp_path)
    assert first.name.startswith("graph-neural-networks-")
    assert first != default_run_dir({**settings, "max_papers": 5}, tmp_path)


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def test_recording_client_sizes_requests_and_never_sends():
    client = RecordingClient()
    with pytest.raises(RuntimeError):
        client.models.generate_content(
            model=MODEL, contents="hello",
            config=generation_config("planner", response_schema=SimpleSchema),
        )
    call = client.calls[0]
    assert call.output_tokens == TASK_OUTPUT_BUDGETS["planner"]
    assert call.fits and call.input_tokens > 0


def test_dry_run_projects_every_stage_for_a_new_topic(tmp_path):
    report, stage, run_dir = dry_run_topic(config(), runs_dir=tmp_path)
    assert stage is None and not run_dir.exists()   # nothing written, nothing sent
    names = [s.stage for s in report.stages]
    assert names == ["planner", "paper analysis", "evidence matrix",
                     "gap detection", "evidence validation",
                     "contradiction adjudication (worst case)",   # projected worst case
                     "critic",
                     "research questions"]          # Phase 2: no revision rounds by default
    assert report.stages[0].basis == "measured"
    assert report.stages[2].basis == "deterministic" and report.stages[2].calls == 0
    assert all(s.basis == "projected" for s in report.stages[3:])
    text = format_report(report)
    assert "no request was sent" in text and "Assumptions" in text


def test_dry_run_measures_stages_from_checkpointed_artifacts(tmp_path):
    agents = workflow_agents()
    agents.paper_analysis._resolve_pdf_path.side_effect = lambda paper, pid, paths: "x.pdf"
    agents.gap_detection.detect_gaps.side_effect = RuntimeError("stop after compare")
    with pytest.raises(RunInterrupted):
        run_topic(config(), run_dir=tmp_path / "run", agents=agents)

    report, stage, _ = dry_run_topic(config(), run_dir=tmp_path / "run",
                                     assumptions=DryRunAssumptions(papers=2, max_revisions=0))
    by_name = {s.stage: s for s in report.stages}
    assert stage == "compare"
    assert by_name["paper analysis"].basis == "measured"
    assert by_name["paper analysis"].calls == 2
    assert by_name["gap detection"].basis == "measured"
    assert by_name["gap detection"].calls == 1
    assert "revision rounds (worst case)" not in by_name


def test_dry_run_totals_count_input_plus_output_budget():
    report = estimate_run({"research_topic": "topic"},
                          assumptions=DryRunAssumptions(max_revisions=0))
    for stage in report.stages:
        assert stage.total_tokens == stage.input_tokens + stage.output_tokens
    assert sum(m.tokens for m in report.models) == sum(s.total_tokens for s in report.stages)
