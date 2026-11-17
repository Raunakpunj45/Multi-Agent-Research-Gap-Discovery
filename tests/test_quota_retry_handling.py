"""Daily-quota vs transient retry classification across every Gemini agent.

Offline only: every client is a fake, sleeps are captured, and the shared
conftest blocks network sockets.
"""
from types import SimpleNamespace

import pytest
from google.genai import errors as genai_errors

from src.agents.critic_agent import CriticAgent, _LLMCriticResult
from src.agents.cross_paper_comparison_agent import CrossPaperComparisonAgent, LLMComparison
from src.agents.evidence_validation_agent import (
    EvidenceValidationAgent, EvidenceValidationStatus, GapValidationStatus,
)
from src.agents.paper_analysis_agent import (
    DAILY_QUOTA_NOT_ATTEMPTED, PaperAnalysis, PaperAnalysisAgent,
    _is_daily_quota_exhaustion, _is_retryable,
)
from src.agents.research_gap_detection_agent import LLMGapDetection, ResearchGapDetectionAgent
from src.agents.research_question_agent import ResearchQuestionAgent, _LLMQuestionSet
from tests.test_evidence_validation_agent import Client as ValidationClient
from tests.test_evidence_validation_agent import Store, input_result


# ---------------------------------------------------------------------------
# Provider-shaped errors
# ---------------------------------------------------------------------------


def daily_quota_error():
    """Real google-genai ClientError as returned for a per-day free-tier quota."""
    return genai_errors.ClientError(429, {"error": {
        "code": 429, "status": "RESOURCE_EXHAUSTED",
        "message": "You exceeded your current quota. Quota exceeded for metric: "
                   "generate_content_free_tier_requests, limit: 20",
        "details": [{"violations": [{
            "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}],
    }})


def minute_rate_limit_error():
    return genai_errors.ClientError(429, {"error": {
        "code": 429, "status": "RESOURCE_EXHAUSTED",
        "message": "Quota exceeded for metric: generate_content_free_tier_requests, limit: 10",
        "details": [{"violations": [{
            "quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}]}],
    }})


def unavailable_error():
    return genai_errors.ServerError(503, {"error": {
        "code": 503, "status": "UNAVAILABLE", "message": "The model is overloaded."}})


def invalid_argument_error():
    return genai_errors.ClientError(400, {"error": {
        "code": 400, "status": "INVALID_ARGUMENT", "message": "Request contains an invalid argument."}})


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("error, daily, retryable", [
    (daily_quota_error(), True, False),
    (type("E", (Exception,), {"code": 429})(
        "429 RESOURCE_EXHAUSTED: daily quota exceeded"), True, False),
    (minute_rate_limit_error(), False, True),
    (type("E", (Exception,), {"code": 429})("429 RESOURCE_EXHAUSTED: rate limit"), False, True),
    (unavailable_error(), False, True),
    (RuntimeError("503 UNAVAILABLE"), False, True),
    (type("E", (Exception,), {"code": 500})("internal"), False, True),
    (type("E", (Exception,), {"code": 504})("deadline"), False, True),
    (invalid_argument_error(), False, False),
    (ValueError("Gemini returned no parsed assessment."), False, False),
    # Digits inside identifiers are not HTTP status codes.
    (ValueError("Malformed response for chunk 1500 (paper W4402683817)"), False, False),
    # A per-day message without a 429 status is not treated as quota exhaustion.
    (ValueError("Results reported per day were inconsistent."), False, False),
])
def test_retry_classification(error, daily, retryable):
    assert _is_daily_quota_exhaustion(error) is daily
    assert _is_retryable(error) is retryable


def test_critic_reexports_the_same_shared_classifier():
    from src.agents import critic_agent, paper_analysis_agent
    assert critic_agent._is_daily_quota_exhaustion is paper_analysis_agent._is_daily_quota_exhaustion
    assert critic_agent._is_retryable is paper_analysis_agent._is_retryable


# ---------------------------------------------------------------------------
# Per-agent retry loops
# ---------------------------------------------------------------------------


class SequencedClient:
    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), 0
        self.models = SimpleNamespace(generate_content=self.generate)

    def generate(self, **_kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(parsed=outcome)


AGENTS = {
    "paper_analysis": (
        lambda client: PaperAnalysisAgent(client=client, max_retries=5),
        lambda agent: agent._generate_with_retry("instruction", "evidence"),
        PaperAnalysis.model_construct(),
    ),
    "comparison": (
        lambda client: CrossPaperComparisonAgent(client=client, max_retries=5),
        lambda agent: agent._generate_with_retry("contents"),
        LLMComparison.model_construct(),
    ),
    "gap_detection": (
        lambda client: ResearchGapDetectionAgent(client=client, max_retries=5),
        lambda agent: agent._generate_with_retry("contents"),
        LLMGapDetection.model_construct(),
    ),
    "critic": (
        lambda client: CriticAgent(client=client, max_retries=5),
        lambda agent: agent._generate("contents"),
        _LLMCriticResult.model_construct(),
    ),
    "question": (
        lambda client: ResearchQuestionAgent(client=client, max_retries=5),
        lambda agent: agent._generate("contents"),
        _LLMQuestionSet.model_construct(),
    ),
}


@pytest.fixture
def sleeps(monkeypatch):
    recorded = []
    monkeypatch.setattr("time.sleep", recorded.append)
    return recorded


@pytest.mark.parametrize("name", sorted(AGENTS))
def test_daily_quota_stops_immediately(name, sleeps):
    factory, call, _ = AGENTS[name]
    client = SequencedClient([daily_quota_error(), "must not be reached"])
    with pytest.raises(genai_errors.ClientError):
        call(factory(client))
    assert client.calls == 1
    assert sleeps == []


@pytest.mark.parametrize("name", sorted(AGENTS))
@pytest.mark.parametrize("make_error", [minute_rate_limit_error, unavailable_error],
                         ids=["transient_429", "http_503"])
def test_transient_errors_retry_with_bounded_backoff(name, make_error, sleeps):
    factory, call, success = AGENTS[name]
    client = SequencedClient([make_error(), make_error(), success])
    agent = factory(client)
    agent.base_backoff_seconds, agent.max_backoff_seconds = 2.0, 3.0

    assert call(agent) is success
    assert client.calls == 3
    assert sleeps == [2.0, 3.0]   # exponential, capped at max_backoff_seconds


@pytest.mark.parametrize("name", sorted(AGENTS))
def test_transient_errors_stop_at_max_retries(name, sleeps):
    factory, call, _ = AGENTS[name]
    client = SequencedClient([unavailable_error() for _ in range(5)])
    with pytest.raises(genai_errors.ServerError):
        call(factory(client))
    assert client.calls == 5
    assert len(sleeps) == 4


@pytest.mark.parametrize("name", sorted(AGENTS))
def test_non_retryable_error_is_not_retried(name, sleeps):
    factory, call, _ = AGENTS[name]
    client = SequencedClient([invalid_argument_error(), "must not be reached"])
    with pytest.raises(genai_errors.ClientError):
        call(factory(client))
    assert client.calls == 1
    assert sleeps == []


def test_paper_analysis_sends_no_further_papers_after_daily_quota(sleeps):
    client = SequencedClient([daily_quota_error(), PaperAnalysis.model_construct()])
    agent = PaperAnalysisAgent(client=client, max_retries=5)
    with pytest.raises(genai_errors.ClientError):
        agent._generate_with_retry("instruction", "paper one")
    with pytest.raises(RuntimeError, match="daily API quota"):
        agent._generate_with_retry("instruction", "paper two")
    assert client.calls == 1


def test_agents_keep_lazy_six_key_clients_and_injected_clients():
    injected = object()
    for agent, group in ((PaperAnalysisAgent(), "ANALYSIS"),
                         (CrossPaperComparisonAgent(), "COMPARISON"),
                         (ResearchGapDetectionAgent(), "GAP"),
                         (EvidenceValidationAgent(vector_store=Store()), "VALIDATION"),
                         (CriticAgent(), "CRITIC"),
                         (ResearchQuestionAgent(), "QUESTION")):
        assert agent._client is None, group   # nothing created at construction
    assert PaperAnalysisAgent(client=injected).client is injected
    assert EvidenceValidationAgent(vector_store=Store(), client=injected).client is injected


# ---------------------------------------------------------------------------
# Evidence Validation Agent end to end
# ---------------------------------------------------------------------------


class FailingFirstClient(ValidationClient):
    """Delegates to the standard fake after raising the given errors in order."""

    def __init__(self, errors, **kwargs):
        super().__init__(**kwargs)
        self.errors, self.calls = list(errors), 0

    def generate(self, **kwargs):
        self.calls += 1
        if self.errors:
            self.prompts.append((kwargs["config"].response_json_schema, kwargs["contents"]))
            raise self.errors.pop(0)
        return super().generate(**kwargs)


def validate_with(errors):
    client = FailingFirstClient(errors)
    agent = EvidenceValidationAgent(vector_store=Store(), client=client, max_retries=5)
    return agent, agent.validate(input_result()), client


def test_validation_daily_quota_makes_one_call_and_marks_everything_incomplete(sleeps):
    _, result, client = validate_with([daily_quota_error()])

    assert client.calls == 1           # chunk 2 and the joint assessment are not sent
    assert sleeps == []
    gap = result.gap_validations[0]
    assert gap.status is GapValidationStatus.VALIDATION_INCOMPLETE
    assert [item.status for item in gap.evidence_validations] == [None, None]
    assert gap.joint_assessment is None
    assert DAILY_QUOTA_NOT_ATTEMPTED in gap.evidence_validations[1].assessment_error
    assert any("joint Gemini assessment failed" in item and DAILY_QUOTA_NOT_ATTEMPTED in item
               for item in result.semantic_failures)


@pytest.mark.parametrize("make_error", [minute_rate_limit_error, unavailable_error],
                         ids=["transient_429", "http_503"])
def test_validation_transient_error_retries_and_completes(make_error, sleeps):
    _, result, client = validate_with([make_error()])

    assert client.calls == 4           # failed attempt + 2 chunks + joint
    assert sleeps == [2.0]
    gap = result.gap_validations[0]
    assert gap.status is not GapValidationStatus.VALIDATION_INCOMPLETE
    assert [item.status for item in gap.evidence_validations] == [
        EvidenceValidationStatus.SUPPORTED, EvidenceValidationStatus.PARTIALLY_SUPPORTED]
    assert result.semantic_failures == []


def test_validation_non_retryable_error_fails_only_that_assessment(sleeps):
    _, result, client = validate_with([invalid_argument_error()])

    assert sleeps == []
    assert client.calls == 3           # failed chunk, next chunk, joint
    gap = result.gap_validations[0]
    assert gap.status is GapValidationStatus.VALIDATION_INCOMPLETE
    assert gap.evidence_validations[0].status is None
    assert gap.evidence_validations[1].status is EvidenceValidationStatus.SUPPORTED


def test_validation_quota_flag_resets_for_a_later_run(sleeps):
    agent, first, client = validate_with([daily_quota_error()])
    assert first.gap_validations[0].status is GapValidationStatus.VALIDATION_INCOMPLETE
    client.chunk_statuses = iter(["SUPPORTED", "PARTIALLY_SUPPORTED"])

    second = agent.validate(input_result())   # e.g. after the quota reset

    assert second.gap_validations[0].status is not GapValidationStatus.VALIDATION_INCOMPLETE
    assert client.calls == 4


# ---------------------------------------------------------------------------
# Run-scoped daily-quota state for paper analysis
# ---------------------------------------------------------------------------

from unittest.mock import Mock

from src.agents.paper_analysis_pipeline import PaperAnalysisPipeline
from src.rag.models import RetrievedChunk
from src.workflow.langgraph_app import WorkflowAgents
from tests.test_langgraph_workflow import invoke, paper_records, workflow_agents


def paper_analysis_payload():
    return PaperAnalysis(problem="p", methodology="m", dataset="d", results="r",
                         limitations="l", future_work="f")


class StubRetriever:
    def retrieve(self, query, top_k, paper_id):
        return [RetrievedChunk(chunk_id=f"{paper_id}-c1", paper_id=paper_id, text="Evidence.")]


def real_pipeline(client):
    ingestion = Mock()
    ingestion.ingest.return_value = 1
    comparison_agent = Mock()
    comparison_agent.compare_many.side_effect = lambda results, research_topic=None: SimpleNamespace(
        analysed=[item.evidence[0].paper_id for item in results], excluded_papers=[])
    return PaperAnalysisPipeline(
        ingestion_pipeline=ingestion, rag_retriever=StubRetriever(),
        analysis_agent=PaperAnalysisAgent(client=client, max_retries=5),
        comparison_agent=comparison_agent, pdf_downloader=Mock(),
    )


def pdf_paths(tmp_path, count=3):
    paths = {}
    for index in range(1, count + 1):
        path = tmp_path / f"paper-{index}.pdf"
        path.write_bytes(b"%PDF-1.4 stub")
        paths[f"paper-{index}"] = str(path)
    return paths


def test_agent_quota_state_is_run_scoped(sleeps):
    client = SequencedClient([daily_quota_error(), paper_analysis_payload()])
    agent = PaperAnalysisAgent(client=client, max_retries=5)
    chunks = [RetrievedChunk(chunk_id="c1", paper_id="paper-1", text="Evidence.")]

    agent.begin_run()
    with pytest.raises(genai_errors.ClientError):
        agent.analyze(chunks)                      # paper 1: daily quota
    with pytest.raises(RuntimeError, match="daily API quota"):
        agent.analyze(chunks)                      # paper 2, same run: not sent
    assert client.calls == 1

    agent.begin_run()                              # next run, quota has reset
    assert agent.analyze(chunks).analysis == paper_analysis_payload()
    assert client.calls == 2
    assert sleeps == []


def test_pipeline_reuse_recovers_after_quota_reset(tmp_path, sleeps):
    client = SequencedClient([daily_quota_error()] + [paper_analysis_payload()] * 3)
    pipeline = real_pipeline(client)
    papers, paths = paper_records(3), pdf_paths(tmp_path)

    # Run 1: the first paper hits the daily quota; the other papers are not sent.
    with pytest.raises(ValueError, match="Only 0 paper"):
        pipeline.analyze_and_compare_papers(papers, "query", pdf_paths=paths)
    assert client.calls == 1

    # Run 2 with the same pipeline and agent instance succeeds.
    comparison = pipeline.analyze_and_compare_papers(papers, "query", pdf_paths=paths)
    assert comparison.analysed == ["paper-1", "paper-2", "paper-3"]
    assert client.calls == 4


def test_workflow_reuse_recovers_after_quota_reset(tmp_path, sleeps):
    client = SequencedClient([daily_quota_error()] + [paper_analysis_payload()] * 3)
    mocked = workflow_agents(paper_ids=["paper-1", "paper-2", "paper-3"])
    agents = WorkflowAgents(
        mocked.retrieval, real_pipeline(client), mocked.comparison,
        mocked.gap_detection, mocked.evidence_validation, mocked.critic,
        mocked.research_questions,
    )
    paths = pdf_paths(tmp_path)

    first = invoke(agents, pdf_paths=paths)
    assert first["workflow_status"] == "STOPPED"
    assert first["paper_analyses"] == []
    reasons = [item.reason for item in first["excluded_papers"]]
    assert "RESOURCE_EXHAUSTED" in reasons[0]
    assert all(DAILY_QUOTA_NOT_ATTEMPTED in reason for reason in reasons[1:])
    assert client.calls == 1

    second = invoke(agents, pdf_paths=paths)        # same agent objects
    assert [item.evidence[0].paper_id for item in second["paper_analyses"]] == [
        "paper-1", "paper-2", "paper-3"]
    assert second["workflow_status"] == "COMPLETE"
    assert client.calls == 4


def test_other_agents_do_not_leak_quota_state_across_runs(sleeps):
    # Evidence Validation: reset per validate() call.
    client = FailingFirstClient([daily_quota_error()])
    agent = EvidenceValidationAgent(vector_store=Store(), client=client, max_retries=5)
    assert agent.validate(input_result()).gap_validations[0].status is GapValidationStatus.VALIDATION_INCOMPLETE
    client.chunk_statuses = iter(["SUPPORTED", "PARTIALLY_SUPPORTED"])
    assert agent.validate(input_result()).gap_validations[0].status is not GapValidationStatus.VALIDATION_INCOMPLETE

    # Research Question Agent: reset per generate_questions() call.
    from tests import test_research_question_agent as rq
    question_client = SequencedClient([daily_quota_error(), rq.llm_response()])
    question_agent = ResearchQuestionAgent(client=question_client, max_retries=5)
    first = question_agent.generate_questions(rq.critic(), rq.gap_result(), rq.validation())
    second = question_agent.generate_questions(rq.critic(), rq.gap_result(), rq.validation())
    assert first.gap_results == [] and len(second.gap_results) == 1
    assert question_client.calls == 2
