"""Groq token budget: preflight, completion caps, pacing, staged comparison.

Offline only: requests go through the real ``groq`` SDK into an in-memory
transport (see tests/test_groq_provider.py); sleeps and the clock are faked.
"""
import json
import re
from types import SimpleNamespace

import httpx
import pytest
from google.genai import types
from pydantic import BaseModel

from src.agents.artifact_provenance import fingerprint_json
from src.agents.critic_agent import CriticAgent, _LLMCriticResult
from src.agents.cross_paper_comparison_agent import (
    ComparisonTooLargeError, CrossPaperComparisonAgent, EvidenceRef, LLMComparison,
    _PaperGrounding, _StagedItem,
)
from src.agents.groq_client import (
    MIN_COMPLETION_TOKENS, GroqRequestTooLargeError, estimate_request_tokens, tpm_budget,
)
from src.agents.paper_analysis_agent import (
    PaperAnalysis, PaperAnalysisAgent, _is_daily_quota_exhaustion, _is_retryable,
)
from src.rag import RetrievedChunk
from tests.test_groq_provider import (
    FakeGroqAPI, completion, groq_client, minute_tokens, model_payload,
)
from tests.test_multi_paper_comparison import FakeClient

MODEL = "openai/gpt-oss-120b"


class SimpleSchema(BaseModel):
    queries: list[str]


PAPERS = [f"https://openalex.org/W{n}000" for n in range(1, 7)]
FILLER = ("The method is evaluated against the stated baseline on the reported "
          "task, and the authors describe its assumptions and observed limits. ")


def long_result(paper_id, n_chunks=6):
    """A paper analysis whose evidence excerpts are longer than 700 chars."""
    return SimpleNamespace(
        analysis=SimpleNamespace(**{
            field: f"{paper_id} {field} statement" for field in (
                "problem", "methodology", "dataset", "results", "limitations", "future_work")
        }),
        evidence=[SimpleNamespace(
            paper_id=paper_id, chunk_id=f"{paper_id}_chunk_{i}",
            text=f"MARK[{paper_id}#{i}] " + FILLER * 12,
        ) for i in range(1, n_chunks + 1)],
    )


@pytest.fixture
def sleeps(monkeypatch):
    recorded = []
    monkeypatch.setattr("time.sleep", recorded.append)
    return recorded


def schema_name(request):
    return request["response_format"]["json_schema"]["name"]


def part(label, text="its part"):
    return {"paper_label": label, "part": text}


SYNTHESIS = {
    "dimension_summaries": [
        {"dimension": d, "summary": f"{d} across papers", "papers": [part("P1")]}
        for d in ("problem", "methodology", "dataset", "results", "limitations", "future_work")
    ],
    "recurring_patterns": [{"title": "Shared pattern", "description": "Both use it.",
                            "dimensions": ["methodology"],
                            "papers": [part("P1"), part("P2"), part("P9")]}],
    "common_limitations": [{"title": "Shared limit", "description": "Both limited.",
                            "dimensions": ["limitations"], "papers": [part("P1"), part("P3")]}],
    "conflicting_findings": [{"topic": "Result", "description": "They disagree.",
                              "position_a": "A", "position_a_papers": [part("P4")],
                              "position_b": "B", "position_b_papers": [part("P5")]}],
    "methodological_differences": [],
    "underexplored_directions": [{"direction": "Combine them", "rationale": "Both suggest it.",
                                  "papers": [part("P2"), part("P6")]}],
    "caveats": ["model caveat"],
}

# Per paper: item_id -> labels returned by the grounding step. Includes a
# foreign label, a non-existent label and an unknown item, all of which must
# be rejected in code.
GROUNDING = {
    "P1": {"R1": ["P1-E2"], "L1": ["P1-E1"], **{f"S{n}": ["P1-E1"] for n in range(1, 7)}},
    "P2": {"R1": ["P2-E3", "P1-E1", "P2-E99"], "D1": ["P2-E1"], "X9": ["P2-E2"]},
    "P3": {"L1": []},
    "P4": {"C1A": ["P4-E1"]},
    "P5": {"C1B": ["P5-E2"]},
    "P6": {"D1": ["P6-E4"]},
}


class StagedGroq(FakeGroqAPI):
    """Answers synthesis and grounding requests by schema, recording each."""

    def __init__(self):
        super().__init__([])

    def __call__(self, request):
        body = json.loads(request.content)
        self.requests.append(body)
        name = schema_name(body)
        if name == "_StagedSynthesis":
            content = json.dumps(SYNTHESIS)
        elif name == "_PaperGrounding":
            prompt = body["messages"][-1]["content"]
            label = re.match(r"Paper (P\d+)\.", prompt).group(1)
            item_ids = re.findall(r"^\[([A-Z]\d+[AB]?)\]", prompt, flags=re.M)
            answers = GROUNDING[label]
            attributions = [{"item_id": item, "evidence_labels": answers.get(item, [])}
                            for item in item_ids]
            if label == "P2":
                attributions.append({"item_id": "X9", "evidence_labels": ["P2-E2"]})
            content = json.dumps({"attributions": attributions})
        else:
            raise AssertionError(f"unexpected single-request comparison: {name}")
        status, payload, headers = completion(content)
        return httpx.Response(status, json=payload, headers=headers, request=request)


# ---------------------------------------------------------------------------
# Adapter preflight and completion cap
# ---------------------------------------------------------------------------


def test_oversized_request_is_refused_before_sending_and_never_retried(monkeypatch, sleeps):
    monkeypatch.setenv("GROQ_TPM_LIMIT", "2000")
    api = FakeGroqAPI(["must not be requested"])
    agent = CriticAgent(client=groq_client(monkeypatch, api), max_retries=5)
    with pytest.raises(GroqRequestTooLargeError) as error:
        agent._generate("evidence " * 4000)
    assert api.requests == [] and sleeps == []
    assert error.value.code == 413
    assert not _is_retryable(error.value) and not _is_daily_quota_exhaustion(error.value)
    assert not re.search(r"\b(429|500|502|503|504)\b", str(error.value))


def test_completion_cap_keeps_input_plus_output_within_the_budget(monkeypatch):
    api = FakeGroqAPI([completion('{"queries": []}'), completion('{"queries": []}')])
    client = groq_client(monkeypatch, api)
    config = types.GenerateContentConfig(response_schema=SimpleSchema)
    client.models.generate_content(model=MODEL, contents="word " * 4000, config=config)
    request = api.requests[0]
    budget = tpm_budget(MODEL)
    assert request["max_completion_tokens"] == min(4096, budget - estimate_request_tokens(request))
    assert estimate_request_tokens(request) + request["max_completion_tokens"] <= budget

    monkeypatch.setenv("GROQ_MAX_COMPLETION_TOKENS_OPENAI_GPT_OSS_120B", "1500")
    client.models.generate_content(model=MODEL, contents="short", config=config)
    assert api.requests[1]["max_completion_tokens"] == 1500


def test_tpm_management_can_be_disabled_for_higher_limits(monkeypatch):
    monkeypatch.setenv("GROQ_TPM_LIMIT", "0")
    api = FakeGroqAPI([completion('{"queries": []}')])
    client = groq_client(monkeypatch, api)
    client.models.generate_content(model=MODEL, contents="word " * 40000,
                                   config=types.GenerateContentConfig(response_schema=SimpleSchema))
    assert "max_completion_tokens" not in api.requests[0]
    assert client.max_input_tokens(MODEL) is None


def test_groq_413_is_not_retried(monkeypatch, sleeps):
    too_large = (413, {"error": {
        "message": "Request too large for model `openai/gpt-oss-120b` in organization "
                   "`org_01TEST` service tier `on_demand` on tokens per minute (TPM): "
                   "Limit 8000, Requested 8240, please reduce your message size and try again.",
        "type": "tokens", "code": "rate_limit_exceeded"}}, {})
    api = FakeGroqAPI([too_large, "must not be requested"])
    agent = CrossPaperComparisonAgent(client=groq_client(monkeypatch, api, "COMPARISON"),
                                      max_retries=5)
    with pytest.raises(Exception) as error:
        agent._generate_with_retry("contents")
    assert error.value.code == 413 and not _is_retryable(error.value)
    assert len(api.requests) == 1 and sleeps == []


# ---------------------------------------------------------------------------
# Pacing within the tokens-per-minute window
# ---------------------------------------------------------------------------


@pytest.fixture
def clock(monkeypatch):
    now = {"t": 1000.0}
    monkeypatch.setattr("time.monotonic", lambda: now["t"])
    return now


def usage_completion(total_tokens):
    status, payload, headers = completion('{"queries": []}')
    payload["usage"]["total_tokens"] = total_tokens
    return status, payload, headers


@pytest.mark.parametrize("first_usage,expect_wait", [(6000, True), (300, False)])
def test_requests_wait_for_capacity_in_the_token_window(monkeypatch, sleeps, clock,
                                                        first_usage, expect_wait):
    api = FakeGroqAPI([usage_completion(first_usage), usage_completion(300)])
    client = groq_client(monkeypatch, api)
    config = types.GenerateContentConfig(response_schema=SimpleSchema)
    client.models.generate_content(model=MODEL, contents="first", config=config)
    clock["t"] += 5            # second request 5 s later, same minute
    client.models.generate_content(model=MODEL, contents="second", config=config)
    assert len(api.requests) == 2
    if expect_wait:
        assert sleeps == [pytest.approx(55.0)]   # until the first request leaves the window
    else:
        assert sleeps == []    # actual usage, not the reservation, is kept


def test_failed_request_releases_its_reservation(monkeypatch, sleeps, clock):
    api = FakeGroqAPI([minute_tokens(), completion(model_payload(_LLMCriticResult))])
    agent = CriticAgent(client=groq_client(monkeypatch, api), max_retries=3)
    agent._generate("contents")
    assert len(api.requests) == 2
    assert sleeps == [4.0]      # only the provider Retry-After, no window wait


# ---------------------------------------------------------------------------
# Paper analysis prompt
# ---------------------------------------------------------------------------


def test_paper_analysis_sends_complete_evidence_text_without_unused_metadata(monkeypatch):
    chunks = [RetrievedChunk(chunk_id=f"https://openalex.org/W9_chunk_{i}",
                             paper_id="https://openalex.org/W9",
                             text=f"Evidence {i}\nline two  with   spacing {FILLER}",
                             distance=0.123456789)
              for i in range(1, 6)]
    payload = model_payload(PaperAnalysis)
    api = FakeGroqAPI([completion(payload, model="openai/gpt-oss-20b")])
    agent = PaperAnalysisAgent(client=groq_client(monkeypatch, api, "ANALYSIS"))
    result = agent.analyze(chunks)

    prompt = api.requests[0]["messages"][-1]["content"]
    for chunk in chunks:
        assert " ".join(chunk.text.split()) in prompt       # every word of every chunk
        assert chunk.chunk_id not in prompt
    assert "Distance" not in prompt and "Paper ID" not in prompt
    assert prompt.count("--- Evidence Chunk") == 5
    assert result.evidence == chunks                         # exact provenance kept


# ---------------------------------------------------------------------------
# Staged cross-paper comparison
# ---------------------------------------------------------------------------


def staged_run(monkeypatch):
    api = StagedGroq()
    agent = CrossPaperComparisonAgent(client=groq_client(monkeypatch, api, "COMPARISON"))
    results = [long_result(paper) for paper in PAPERS]
    return api, agent, results, agent.compare_many(results, research_topic="topic")


def test_comparison_that_fits_uses_one_request_with_unchanged_input(monkeypatch):
    api = FakeGroqAPI([completion(model_payload(LLMComparison))])
    agent = CrossPaperComparisonAgent(client=groq_client(monkeypatch, api, "COMPARISON"))
    results = [long_result(paper, n_chunks=1) for paper in PAPERS[:2]]
    comparison = agent.compare_many(results, research_topic="topic")

    _, text = agent._prepare(results)
    contents = agent._build_contents(text, 2, "topic")
    assert len(api.requests) == 1 and schema_name(api.requests[0]) == "LLMComparison"
    assert api.requests[0]["messages"][-1]["content"] == contents
    assert comparison.provenance.input_fingerprints["paper_analyses"] == fingerprint_json(contents)
    assert not any("Staged comparison" in caveat for caveat in comparison.caveats)


def test_oversized_comparison_is_staged_within_budget(monkeypatch):
    api, agent, results, _ = staged_run(monkeypatch)
    _, text = agent._prepare(results)
    single = agent._build_contents(text, len(results), "topic")
    limit = agent.client.max_input_tokens(MODEL)
    assert agent._estimate(single, agent.SYSTEM_INSTRUCTION, LLMComparison) > limit

    names = [schema_name(request) for request in api.requests]
    assert names[0] == "_StagedSynthesis" and set(names[1:]) == {"_PaperGrounding"}
    assert len(names) == 1 + len(PAPERS)                     # every paper is cited once
    for request in api.requests:                             # nothing near the limit was sent
        assert estimate_request_tokens(request) <= limit
        assert (estimate_request_tokens(request) + request["max_completion_tokens"]
                <= tpm_budget(MODEL))
        assert request["max_completion_tokens"] >= MIN_COMPLETION_TOKENS


def test_staged_requests_carry_each_papers_own_untruncated_excerpts(monkeypatch):
    api, agent, results, _ = staged_run(monkeypatch)
    synthesis_prompt = api.requests[0]["messages"][-1]["content"]
    assert "MARK[" not in synthesis_prompt                  # stage 1 sees analyses only

    grounding = {re.match(r"Paper (P\d+)\.", r["messages"][-1]["content"]).group(1):
                 r["messages"][-1]["content"] for r in api.requests[1:]}
    _, single_text = agent._prepare(results)
    for number, result in enumerate(results, start=1):
        label = f"P{number}"
        for chunk in result.evidence:
            excerpt = agent._excerpt(chunk)
            assert excerpt in single_text                    # same excerpt as one request
            assert excerpt in grounding[label]               # complete in its own request
            others = [p for other, p in grounding.items() if other != label]
            assert not any(f"MARK[{result.evidence[0].paper_id}#" in p for p in others)


def test_staged_comparison_maps_grounded_labels_to_exact_ids_and_rejects_others(monkeypatch):
    _, _, results, comparison = staged_run(monkeypatch)

    assert comparison.paper_ids == PAPERS
    pattern = comparison.recurring_patterns[0]
    assert pattern.evidence == [
        EvidenceRef(paper_id=PAPERS[0], chunk_ids=[f"{PAPERS[0]}_chunk_2"]),
        EvidenceRef(paper_id=PAPERS[1], chunk_ids=[f"{PAPERS[1]}_chunk_3"]),
    ]   # P2's foreign (P1-E1) and invented (P2-E99) labels and unknown P9 are rejected
    assert comparison.common_limitations == []               # P3 unsupported -> 1 paper -> dropped
    conflict = comparison.conflicting_findings[0]
    assert conflict.position_a_evidence == [EvidenceRef(paper_id=PAPERS[3],
                                                        chunk_ids=[f"{PAPERS[3]}_chunk_1"])]
    assert conflict.position_b_evidence == [EvidenceRef(paper_id=PAPERS[4],
                                                        chunk_ids=[f"{PAPERS[4]}_chunk_2"])]
    assert comparison.underexplored_directions[0].evidence == [
        EvidenceRef(paper_id=PAPERS[1], chunk_ids=[f"{PAPERS[1]}_chunk_1"]),
        EvidenceRef(paper_id=PAPERS[5], chunk_ids=[f"{PAPERS[5]}_chunk_4"]),
    ]
    assert len(comparison.dimension_summaries) == 6
    assert all(s.evidence == [EvidenceRef(paper_id=PAPERS[0], chunk_ids=[f"{PAPERS[0]}_chunk_1"])]
               for s in comparison.dimension_summaries)
    owners = {chunk.chunk_id: chunk.paper_id for r in results for chunk in r.evidence}
    for group in (comparison.recurring_patterns, comparison.underexplored_directions,
                  comparison.dimension_summaries):
        for item in group:
            for ref in item.evidence:
                assert all(owners[chunk] == ref.paper_id for chunk in ref.chunk_ids)
    assert any("'common limitations' were discarded" in c for c in comparison.caveats)
    assert any("Staged comparison" in c and "1 paper part(s)" in c for c in comparison.caveats)
    assert "model caveat" in comparison.caveats


def test_staged_comparison_keeps_the_canonical_provenance_fingerprint(monkeypatch):
    _, agent, results, comparison = staged_run(monkeypatch)
    _, text = agent._prepare(results)
    canonical = agent._build_contents(text, len(results), "topic")
    assert comparison.provenance.stage == "cross_paper_comparison"
    assert comparison.provenance.input_fingerprints == {
        "paper_analyses": fingerprint_json(canonical)}


def test_grounding_items_are_split_so_every_request_fits(monkeypatch):
    agent = CrossPaperComparisonAgent(client=groq_client(monkeypatch, FakeGroqAPI([]),
                                                         "COMPARISON"))
    excerpts = agent._paper_excerpts([long_result(PAPERS[0])])["P1"]
    items = [_StagedItem(f"R{n}", "P1", "Recurring pattern: " + "context " * 60, "part " * 40)
             for n in range(1, 13)]
    budget = agent._estimate(agent._grounding_contents("P1", excerpts, items[:3]),
                             agent.STAGED_GROUNDING_INSTRUCTION, _PaperGrounding)
    batches = agent._grounding_batches("P1", excerpts, items, budget)
    assert len(batches) >= 4
    assert [item for batch in batches for item in batch] == items   # nothing dropped
    for batch in batches:
        assert agent._estimate(agent._grounding_contents("P1", excerpts, batch),
                               agent.STAGED_GROUNDING_INSTRUCTION, _PaperGrounding) <= budget


def test_staged_comparison_that_still_cannot_fit_is_refused_before_sending(monkeypatch):
    monkeypatch.setenv("GROQ_TPM_LIMIT", "1500")
    api = StagedGroq()
    agent = CrossPaperComparisonAgent(client=groq_client(monkeypatch, api, "COMPARISON"))
    with pytest.raises(ComparisonTooLargeError, match="nothing was sent"):
        agent.compare_many([long_result(paper) for paper in PAPERS], research_topic="topic")
    assert api.requests == []


def test_clients_without_a_token_budget_keep_the_single_request(monkeypatch):
    parsed = LLMComparison.model_validate(json.loads(model_payload(LLMComparison)))
    client = FakeClient(parsed)          # e.g. Gemini or an injected test client
    agent = CrossPaperComparisonAgent(client=client)
    agent.compare_many([long_result(paper) for paper in PAPERS])
    assert len(client.prompts) == 1
