"""Groq provider: configuration, structured output, errors and agent retry paths.

Offline only. Requests go through the real ``groq`` SDK into an in-memory
``httpx.MockTransport`` that returns recorded-shape Groq responses; nothing
leaves the process and the conftest socket guard stays active.
"""
import json

import groq
import httpx
import pytest
from google.genai import types
from pydantic import BaseModel

from src.agents import llm_provider
from src.agents.critic_agent import CriticAgent, _LLMCriticResult, _provider_retry_after
from src.agents.cross_paper_comparison_agent import (
    CrossPaperComparisonAgent, LLMComparison, _PaperGrounding, _StagedSynthesis,
)
from src.agents.evidence_validation_agent import (
    EvidenceValidationAgent, _ChunkAssessment, _ContradictionAdjudication, _JointAssessment,
    _gemini_compatible_schema,
)
from src.agents.groq_client import (
    GroqAPIError, GroqOutputTruncatedError, GroqStructuredClient, build_schema_plan,
    create_groq_client, strict_json_schema,
)
from src.agents.paper_analysis_agent import (
    DAILY_QUOTA_NOT_ATTEMPTED, PaperAnalysis, PaperAnalysisAgent,
    _is_daily_quota_exhaustion, _is_retryable,
)
from src.agents.planner_agent import PlannerAgent, SearchPlan
from src.agents.research_gap_detection_agent import LLMGapDetection, ResearchGapDetectionAgent
from src.agents.research_question_agent import ResearchQuestionAgent, _LLMQuestionSet
from src.evaluation.baselines import GeminiStructuredGenerator, _GeneratedGapSet
from src.evaluation.baselines import _gemini_compatible_schema as baseline_schema
from tests.test_evidence_validation_agent import Store

API_KEY = "gsk_unit_test_secret_value"
ORG = "org_01HZTESTORGANIZATION"


# ---------------------------------------------------------------------------
# Mock Groq HTTP API
# ---------------------------------------------------------------------------


def completion(content, model="openai/gpt-oss-120b", finish_reason="stop"):
    return 200, {
        "id": "chatcmpl-test", "object": "chat.completion", "created": 0, "model": model,
        "choices": [{"index": 0, "finish_reason": finish_reason,
                     "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }, {}


def rate_limit(window, code, retry_after):
    message = (
        f"Rate limit reached for model `openai/gpt-oss-120b` in organization `{ORG}` "
        f"service tier `on_demand` on {window} ({code}): Limit 1000, Used 1000, "
        "Requested 1. Please try again in 5m24.5s. Need more tokens? Upgrade to Dev "
        "Tier today at https://console.groq.com/settings/billing"
    )
    return 429, {"error": {"message": message, "type": window.split()[0],
                           "code": "rate_limit_exceeded"}}, {"retry-after": str(retry_after)}


def daily_tokens():
    return rate_limit("tokens per day", "TPD", 325)


def daily_requests():
    return rate_limit("requests per day", "RPD", 3600)


def minute_tokens():
    return rate_limit("tokens per minute", "TPM", 4)


def server_unavailable():
    return 503, {"error": {"message": "Service Unavailable", "type": "internal_server_error"}}, {}


def bad_request():
    return 400, {"error": {"message": "Generated JSON does not match the expected schema.",
                           "type": "invalid_request_error", "code": "json_validate_failed"}}, {}


class FakeGroqAPI:
    """Serves queued responses; an exception instance is raised as a transport error."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.requests, self.headers = [], []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        self.headers.append(dict(request.headers))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        status, body, headers = outcome
        return httpx.Response(status, json=body, headers=headers, request=request)


def groq_client(monkeypatch, api, group="CRITIC"):
    monkeypatch.setenv("GROQ_API_KEY", API_KEY)
    return create_groq_client(group, http_client=httpx.Client(transport=httpx.MockTransport(api)))


def minimal_json(schema):
    """A minimal instance of a (strict) JSON schema, used as a model response."""
    defs = schema.get("$defs", {})

    def build(node):
        if "$ref" in node:
            return build(defs[node["$ref"].rsplit("/", 1)[-1]])
        if "anyOf" in node:
            branches = [b for b in node["anyOf"] if b.get("type") != "null"]
            return build(branches[0]) if branches else None
        if "enum" in node:
            return node["enum"][0]
        kind = node.get("type")
        if isinstance(kind, list):
            kind = next(k for k in kind if k != "null")
        if kind == "object" or "properties" in node:
            return {name: build(sub) for name, sub in node.get("properties", {}).items()}
        return {"array": [], "string": "x", "boolean": False,
                "integer": 0, "number": 0}.get(kind)

    return build(schema)


def model_payload(model_class):
    schema, _, _ = strict_json_schema(model_class.model_json_schema())
    return json.dumps(minimal_json(schema))


@pytest.fixture
def sleeps(monkeypatch):
    recorded = []
    monkeypatch.setattr("time.sleep", recorded.append)
    return recorded


# ---------------------------------------------------------------------------
# Provider and model configuration
# ---------------------------------------------------------------------------


def test_groq_is_default_provider_and_agents_use_role_models():
    assert llm_provider.get_provider() == "groq"
    groq_models = llm_provider.DEFAULT_MODELS["groq"]
    agents = {
        "PLANNER": PlannerAgent(), "ANALYSIS": PaperAnalysisAgent(),
        "COMPARISON": CrossPaperComparisonAgent(), "GAP": ResearchGapDetectionAgent(),
        "VALIDATION": EvidenceValidationAgent(vector_store=Store()),
        "CRITIC": CriticAgent(), "QUESTION": ResearchQuestionAgent(),
        "BASELINE": GeminiStructuredGenerator(client=object()),
    }
    for role, agent in agents.items():
        assert agent.model == groq_models[role], role
        assert agent.model in {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}


def test_model_overrides_per_role_then_provider_wide_then_explicit_argument(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "provider/global-model")
    monkeypatch.setenv("GROQ_MODEL_CRITIC", "provider/critic-model")
    assert CriticAgent().model == "provider/critic-model"
    assert ResearchGapDetectionAgent().model == "provider/global-model"
    assert CriticAgent(model="explicit/model").model == "explicit/model"


def test_gemini_remains_available_with_its_previous_default_models(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    assert PaperAnalysisAgent().model == "gemini-3.5-flash-lite"
    assert CriticAgent().model == "gemini-3.5-flash"
    monkeypatch.setenv("GEMINI_MODEL_CRITIC", "gemini-custom")
    assert CriticAgent().model == "gemini-custom"


def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "other")
    with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
        llm_provider.get_provider()


# ---------------------------------------------------------------------------
# Keys and client construction
# ---------------------------------------------------------------------------


class RecordingGroq:
    created = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        RecordingGroq.created.append(self)


def test_every_agent_lazily_creates_its_own_groq_client_with_sdk_retries_disabled(monkeypatch):
    RecordingGroq.created = []
    monkeypatch.setattr(groq, "Groq", RecordingGroq)
    monkeypatch.setenv("GROQ_API_KEY", "global-secret")
    monkeypatch.setenv("GROQ_API_KEY_CRITIC", "critic-secret")

    agents = {
        "ANALYSIS": PaperAnalysisAgent(), "COMPARISON": CrossPaperComparisonAgent(),
        "GAP": ResearchGapDetectionAgent(),
        "VALIDATION": EvidenceValidationAgent(vector_store=Store()),
        "CRITIC": CriticAgent(), "QUESTION": ResearchQuestionAgent(),
    }
    assert RecordingGroq.created == []               # construction makes no client
    clients = {group: agent.client for group, agent in agents.items()}

    assert all(isinstance(client, GroqStructuredClient) for client in clients.values())
    assert len({id(client) for client in clients.values()}) == 6
    keys = {group: client._sdk.kwargs["api_key"] for group, client in clients.items()}
    assert keys["CRITIC"] == "critic-secret"         # per-group override first
    assert {keys[g] for g in keys if g != "CRITIC"} == {"global-secret"}
    assert all(sdk.kwargs["max_retries"] == 0 for sdk in RecordingGroq.created)
    assert all("secret" not in repr(client) for client in clients.values())


def test_missing_groq_key_names_variables_without_values(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "a-gemini-secret")   # never used for Groq
    agent = CriticAgent()                                      # lazy: no error yet
    with pytest.raises(ValueError) as error:
        agent.client
    message = str(error.value)
    assert "GROQ_API_KEY_CRITIC" in message and "GROQ_API_KEY" in message
    assert "secret" not in message


# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------


class _Plan(BaseModel):
    queries: list[str]
    note: str | None = None
    rank: int = 1


def test_request_uses_strict_json_schema_and_returns_validated_model(monkeypatch):
    api = FakeGroqAPI([completion('{"queries": ["a", "b"], "note": null, "rank": 2}')])
    client = groq_client(monkeypatch, api)
    response = client.models.generate_content(
        model="openai/gpt-oss-120b", contents="topic",
        config=types.GenerateContentConfig(
            system_instruction="system rules", response_mime_type="application/json",
            response_schema=_Plan, temperature=0.2,
        ),
    )

    assert response.parsed == _Plan(queries=["a", "b"], rank=2)
    request = api.requests[0]
    assert request["model"] == "openai/gpt-oss-120b"
    assert request["messages"] == [{"role": "system", "content": "system rules"},
                                   {"role": "user", "content": "topic"}]
    assert request["temperature"] == 0.2
    response_format = request["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    schema = response_format["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"queries", "note", "rank"}
    assert "default" not in json.dumps(schema)
    assert api.headers[0]["authorization"] == f"Bearer {API_KEY}"   # sent, never logged


def test_truncated_output_raises_a_clear_non_retryable_error(monkeypatch):
    api = FakeGroqAPI([completion('{"queries": ["a"', finish_reason="length")])
    client = groq_client(monkeypatch, api)
    with pytest.raises(GroqOutputTruncatedError) as error:
        client.models.generate_content(
            model="openai/gpt-oss-120b", contents="x",
            config=types.GenerateContentConfig(response_schema=_Plan),
        )
    assert "max_completion_tokens" in str(error.value)
    assert not _is_retryable(error.value)        # identical request would truncate again
    assert error.value.max_completion_tokens == api.requests[0]["max_completion_tokens"]


@pytest.mark.parametrize("content,finish", [
    ('{"queries": "not-a-list"}', "stop"),   # schema-invalid output
    ("{not json", "stop"),
    (None, "stop"),
])
def test_unusable_output_gives_parsed_none_like_the_gemini_sdk(monkeypatch, content, finish):
    client = groq_client(monkeypatch, FakeGroqAPI([completion(content, finish_reason=finish)]))
    response = client.models.generate_content(
        model="openai/gpt-oss-120b", contents="x",
        config=types.GenerateContentConfig(response_schema=_Plan),
    )
    assert response.parsed is None
    assert response.finish_reason == finish


AGENT_SCHEMAS = [PaperAnalysis, SearchPlan, LLMComparison, _StagedSynthesis,
                 _PaperGrounding, LLMGapDetection, _LLMCriticResult, _LLMQuestionSet]
EVIDENCE_SCHEMAS = [_ChunkAssessment, _JointAssessment, _ContradictionAdjudication]


@pytest.mark.parametrize("schema", (
    [model.model_json_schema() for model in AGENT_SCHEMAS]
    + [_gemini_compatible_schema(model) for model in EVIDENCE_SCHEMAS]
    + [baseline_schema(_GeneratedGapSet)]
), ids=[m.__name__ for m in AGENT_SCHEMAS] + [m.__name__ for m in EVIDENCE_SCHEMAS] + ["baseline"])
def test_every_agent_schema_is_valid_for_groq_strict_mode(schema):
    strict, _, compatible = strict_json_schema(schema)
    assert compatible
    allowed = {"type", "properties", "required", "items", "enum", "anyOf", "$ref", "$defs",
               "description", "title", "additionalProperties"}

    def check(node):
        if isinstance(node, list):
            for item in node:
                check(item)
        elif isinstance(node, dict):
            assert set(node) <= allowed, set(node) - allowed
            if "properties" in node:
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            for key in ("items", "anyOf"):
                if key in node:
                    check(node[key])
            for mapping in ("properties", "$defs"):
                for child in (node.get(mapping) or {}).values():
                    check(child)

    check(strict)


def test_free_form_objects_and_non_strict_models_use_best_effort_mode():
    class _FreeForm(BaseModel):
        notes: dict[str, str] = {}

    assert build_schema_plan(_FreeForm, "openai/gpt-oss-120b").strict is False
    assert build_schema_plan(_Plan, "llama-3.3-70b-versatile").strict is False
    assert build_schema_plan(_Plan, "openai/gpt-oss-20b").strict is True


def test_evidence_validation_through_groq_restores_absent_quotes(monkeypatch):
    payload = minimal_json(strict_json_schema(_gemini_compatible_schema(_ChunkAssessment))[0])
    payload["supported_contribution"] = None
    payload["supported_contribution_quote"] = None
    api = FakeGroqAPI([completion(json.dumps(payload))])
    agent = EvidenceValidationAgent(vector_store=Store(),
                                    client=groq_client(monkeypatch, api, "VALIDATION"))

    result = agent._generate("chunk", "instruction", _ChunkAssessment)

    assert isinstance(result, _ChunkAssessment)
    assert result.supported_contribution is None
    assert result.supported_contribution_quote is None
    assert api.requests[0]["response_format"]["json_schema"]["strict"] is True
    # Phase 2: per-gap evidence validation runs on the smaller model by default.
    assert api.requests[0]["model"] == "openai/gpt-oss-20b"


def test_reasoning_effort_is_opt_in_and_only_sent_to_gpt_oss(monkeypatch):
    api = FakeGroqAPI([completion('{"queries": []}', model=m)
                       for m in ("openai/gpt-oss-20b", "llama-3.3-70b-versatile")])
    client = groq_client(monkeypatch, api)
    monkeypatch.setenv("GROQ_REASONING_EFFORT", "low")
    config = types.GenerateContentConfig(response_schema=_Plan)
    client.models.generate_content(model="openai/gpt-oss-20b", contents="x", config=config)
    client.models.generate_content(model="llama-3.3-70b-versatile", contents="x", config=config)
    assert api.requests[0]["reasoning_effort"] == "low"
    assert "reasoning_effort" not in api.requests[1]
    monkeypatch.setenv("GROQ_REASONING_EFFORT", "maximum")
    with pytest.raises(ValueError, match="GROQ_REASONING_EFFORT"):
        client.models.generate_content(model="openai/gpt-oss-20b", contents="x", config=config)


def test_gemini_model_ids_are_rejected_under_groq(monkeypatch):
    client = groq_client(monkeypatch, FakeGroqAPI([]))
    with pytest.raises(ValueError, match="Gemini model"):
        client.models.generate_content(model="gemini-3.5-flash", contents="x")


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


def raised(monkeypatch, outcome):
    api = FakeGroqAPI([outcome])
    client = groq_client(monkeypatch, api)
    with pytest.raises(GroqAPIError) as error:
        client.models.generate_content(model="openai/gpt-oss-120b", contents="x")
    assert len(api.requests) == 1        # the SDK never retries on its own
    return error.value


@pytest.mark.parametrize("outcome", [daily_tokens(), daily_requests()], ids=["TPD", "RPD"])
def test_daily_limits_are_daily_quota_and_never_retried(monkeypatch, outcome):
    error = raised(monkeypatch, outcome)
    assert error.code == 429
    assert _is_daily_quota_exhaustion(error)
    assert not _is_retryable(error)
    assert "per day" in str(error)
    assert ORG not in str(error) and API_KEY not in str(error)


def test_per_minute_limit_is_transient_and_carries_provider_delay(monkeypatch):
    error = raised(monkeypatch, minute_tokens())
    assert not _is_daily_quota_exhaustion(error)
    assert _is_retryable(error)
    assert _provider_retry_after(error) == 4.0
    assert "day" not in str(error).lower()


@pytest.mark.parametrize("outcome,retryable", [
    (server_unavailable(), True),
    (bad_request(), False),
    (httpx.ConnectError("refused"), True),
    (httpx.ReadTimeout("slow"), True),
], ids=["http_503", "http_400", "connection", "timeout"])
def test_server_and_network_errors_retry_client_errors_do_not(monkeypatch, outcome, retryable):
    error = raised(monkeypatch, outcome)
    assert _is_retryable(error) is retryable
    assert not _is_daily_quota_exhaustion(error)


# ---------------------------------------------------------------------------
# Agent retry and daily-quota behaviour through the Groq adapter
# ---------------------------------------------------------------------------

AGENTS = {
    "paper_analysis": ("ANALYSIS", lambda c: PaperAnalysisAgent(client=c, max_retries=5),
                       lambda a: a._generate_with_retry("instruction", "evidence"), PaperAnalysis),
    "comparison": ("COMPARISON", lambda c: CrossPaperComparisonAgent(client=c, max_retries=5),
                   lambda a: a._generate_with_retry("contents"), LLMComparison),
    "gap_detection": ("GAP", lambda c: ResearchGapDetectionAgent(client=c, max_retries=5),
                      lambda a: a._generate_with_retry("contents"), LLMGapDetection),
    "evidence_validation": ("VALIDATION",
                            lambda c: EvidenceValidationAgent(vector_store=Store(), client=c,
                                                              max_retries=5),
                            lambda a: a._generate("chunk", "instruction", _JointAssessment),
                            _JointAssessment),
    "critic": ("CRITIC", lambda c: CriticAgent(client=c, max_retries=5),
               lambda a: a._generate("contents"), _LLMCriticResult),
    "question": ("QUESTION", lambda c: ResearchQuestionAgent(client=c, max_retries=5),
                 lambda a: a._generate("contents"), _LLMQuestionSet),
}


@pytest.mark.parametrize("name", sorted(AGENTS))
def test_each_agent_gets_validated_structured_output_from_groq(monkeypatch, name, sleeps):
    group, factory, call, schema = AGENTS[name]
    api = FakeGroqAPI([completion(model_payload(schema))])
    result = call(factory(groq_client(monkeypatch, api, group)))
    assert isinstance(result, schema)
    assert api.requests[0]["response_format"]["json_schema"]["strict"] is True
    assert sleeps == []


@pytest.mark.parametrize("name", sorted(AGENTS))
def test_each_agent_stops_immediately_on_groq_daily_limit(monkeypatch, name, sleeps):
    group, factory, call, _ = AGENTS[name]
    api = FakeGroqAPI([daily_tokens(), "must not be requested"])
    with pytest.raises(GroqAPIError):
        call(factory(groq_client(monkeypatch, api, group)))
    assert len(api.requests) == 1
    assert sleeps == []


@pytest.mark.parametrize("name", sorted(AGENTS))
def test_each_agent_retries_groq_transient_errors_then_succeeds(monkeypatch, name, sleeps):
    group, factory, call, schema = AGENTS[name]
    api = FakeGroqAPI([server_unavailable(), rate_limit("tokens per minute", "TPM", 7),
                       completion(model_payload(schema))])
    agent = factory(groq_client(monkeypatch, api, group))
    agent.base_backoff_seconds, agent.max_backoff_seconds = 2.0, 30.0
    assert isinstance(call(agent), schema)
    assert len(api.requests) == 3
    # 503 without guidance: exponential backoff (2s). TPM 429: every agent
    # waits exactly the Groq Retry-After (7s), not its own backoff (4s).
    assert sleeps == [2.0, 7.0]


@pytest.mark.parametrize("name", sorted(AGENTS))
def test_each_agent_fails_closed_when_groq_delay_exceeds_its_bound(monkeypatch, name, sleeps):
    group, factory, call, _ = AGENTS[name]
    api = FakeGroqAPI([rate_limit("tokens per minute", "TPM", 120), "must not be requested"])
    agent = factory(groq_client(monkeypatch, api, group))
    agent.max_backoff_seconds = 60.0
    with pytest.raises(GroqAPIError):
        call(agent)
    assert len(api.requests) == 1 and sleeps == []


def test_critic_fails_closed_when_groq_delay_exceeds_its_bound(monkeypatch, sleeps):
    api = FakeGroqAPI([rate_limit("tokens per minute", "TPM", 120), "must not be requested"])
    agent = CriticAgent(client=groq_client(monkeypatch, api), max_retries=5,
                        max_backoff_seconds=60.0)
    with pytest.raises(GroqAPIError):
        agent._generate("contents")
    assert len(api.requests) == 1 and sleeps == []


def test_paper_analysis_daily_limit_is_run_scoped_through_groq(monkeypatch, sleeps):
    api = FakeGroqAPI([daily_tokens(), completion(model_payload(PaperAnalysis),
                                                  model="openai/gpt-oss-20b")])
    agent = PaperAnalysisAgent(client=groq_client(monkeypatch, api, "ANALYSIS"), max_retries=5)
    assert agent.model == "openai/gpt-oss-20b"

    agent.begin_run()
    with pytest.raises(GroqAPIError):
        agent._generate_with_retry("instruction", "paper one")
    with pytest.raises(RuntimeError, match="daily API quota") as skipped:
        agent._generate_with_retry("instruction", "paper two")      # same run: not sent
    assert str(skipped.value) == DAILY_QUOTA_NOT_ATTEMPTED
    assert len(api.requests) == 1

    agent.begin_run()                                               # next run
    assert isinstance(agent._generate_with_retry("instruction", "paper three"), PaperAnalysis)
    assert len(api.requests) == 2
    assert sleeps == []
