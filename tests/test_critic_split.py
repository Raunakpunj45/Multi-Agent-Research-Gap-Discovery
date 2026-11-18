"""Split Critic reviews: bounded sub-reviews, aggregation and integrity rules.

Offline only. Requests go through the real Groq adapter (preflight, completion
cap, strict-schema parsing) into a scripted SDK stub; the review chain is the
committed sanitized fixture questions_gap004_skipped.json (read-only).
"""
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments import run_langgraph
from src.agents.critic_agent import (
    CriticAgent, CriticDecision, CriticResult, CriticReviewStatus, _LLMCriticResult,
)
from src.agents.cross_paper_comparison_agent import MultiPaperComparison
from src.agents.evidence_validation_agent import EvidenceValidationResult
from src.agents.gap_acceptance import acceptance_blockers
from src.agents.groq_client import (
    MIN_COMPLETION_TOKENS, GroqAPIError, GroqStructuredClient, estimate_request_tokens,
    tpm_budget,
)
from src.agents.research_gap_detection_agent import ResearchGapDetectionResult
from src.agents.research_question_agent import ResearchQuestionAgent

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "langgraph_runs" / "questions_gap004_skipped.json"
MODEL = "openai/gpt-oss-120b"


def chain():
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return (MultiPaperComparison.model_validate(raw["comparison"]),
            ResearchGapDetectionResult.model_validate(raw["gap_detection"]),
            EvidenceValidationResult.model_validate(raw["evidence_validation"]),
            raw["critic_input_fingerprints"])


def schema_name(request):
    return request["response_format"]["json_schema"]["name"]


def prompt(request):
    return request["messages"][-1]["content"]


def gap_id_of(request):
    return re.search(r"^## (GAP-\d+):", prompt(request), re.M).group(1)


def evidence_labels_shown(request):
    return re.findall(r"^Individual evidence (P\d+-E\d+):", prompt(request), re.M)


def gap_labels_shown(request):
    line = re.search(r"^Gap evidence labels: \[(.*)\]$", prompt(request), re.M).group(1)
    return re.findall(r"'([^']+)'", line)


def criticism(labels, severity="MEDIUM"):
    return {"category": "WEAK_REASONING", "severity": severity, "affected_claim": "claim",
            "issue": "issue", "rationale": "rationale", "evidence_labels": labels}


class Script:
    """Default answers per step; tests override parts through hooks."""

    def __init__(self, decision="ACCEPT", verdict="NO_ISSUES"):
        self.decision, self.verdict = decision, verdict
        self.sub_review_hook = None
        self.decision_hook = None

    def __call__(self, request):
        name = schema_name(request)
        gap_id = gap_id_of(request)
        if name == "_LLMSubReview":
            shown = evidence_labels_shown(request) or gap_labels_shown(request)[:1]
            answer = {"gap_id": gap_id, "verdict": self.verdict, "summary": "Reviewed.",
                      "criticisms": [criticism(shown[:1])], "caveats": []}
            if self.sub_review_hook:
                answer = self.sub_review_hook(request, answer)
        elif name == "_LLMGapReview":
            labels = gap_labels_shown(request)[:1]
            answer = {"gap_id": gap_id, "decision": self.decision,
                      "origin_assessment": "INFERRED", "origin_rationale": "Inferred.",
                      "reasoning": "Sound within scope.", "criticisms": [],
                      "revision_suggestions": [{"field": "description", "current_issue": "x",
                                                "proposed_text": "y", "evidence_labels": labels}],
                      "caveats": ["decision caveat"]}
            if self.decision_hook:
                answer = self.decision_hook(request, answer)
        else:
            raise AssertionError(f"unexpected single-request critic call: {name}")
        if isinstance(answer, Exception):
            raise answer
        if isinstance(answer, tuple):          # (raw content, finish_reason)
            return answer
        return json.dumps(answer), "stop"


class ScriptedSDK:
    def __init__(self, script):
        self.script, self.requests = script, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        self.requests.append(request)
        content, finish = self.script(request)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content),
                                     finish_reason=finish)],
            usage=SimpleNamespace(total_tokens=50),
        )


def run(script, **agent_kwargs):
    sdk = ScriptedSDK(script)
    comparison, gaps, validation, fingerprints = chain()
    result = CriticAgent(review_mode="auto", client=GroqStructuredClient(sdk), max_retries=1, **agent_kwargs).critique(
        comparison, gaps, validation, input_fingerprints=fingerprints)
    return result, sdk.requests


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _seconds: None)


# ---------------------------------------------------------------------------
# Mode selection and bounded requests
# ---------------------------------------------------------------------------


def test_token_budgeted_client_uses_split_reviews_gemini_style_clients_do_not():
    result, requests = run(Script())
    assert {schema_name(r) for r in requests} == {"_LLMSubReview", "_LLMGapReview"}
    assert result.review_status is CriticReviewStatus.COMPLETE

    calls = []

    def single(**kwargs):   # an injected / Gemini-style client exposes no token budget
        calls.append(kwargs["config"].response_schema)
        gap_id = re.search(r"^## (GAP-\d+):", kwargs["contents"], re.M).group(1)
        return SimpleNamespace(parsed=_LLMCriticResult.model_validate({"gap_reviews": [{
            "gap_id": gap_id, "decision": "ACCEPT", "origin_assessment": "INFERRED",
            "origin_rationale": "x", "reasoning": "x"}]}))
    comparison, gaps, validation, fingerprints = chain()
    CriticAgent(review_mode="auto", client=SimpleNamespace(models=SimpleNamespace(generate_content=single))).critique(
        comparison, gaps, validation, input_fingerprints=fingerprints)
    assert calls == [_LLMCriticResult] * len(gaps.gaps)


@pytest.mark.parametrize("tpm_limit", [None, "5500"], ids=["default_8000", "tight_5500"])
def test_every_split_request_fits_the_token_budget(monkeypatch, tpm_limit):
    if tpm_limit:
        monkeypatch.setenv("GROQ_TPM_LIMIT", tpm_limit)   # forces several batches per step
    result, requests = run(Script())
    budget = tpm_budget(MODEL)
    assert result.review_status is CriticReviewStatus.COMPLETE
    for request in requests:
        estimated = estimate_request_tokens(request)
        assert estimated + request["max_completion_tokens"] <= budget
        assert request["max_completion_tokens"] >= MIN_COMPLETION_TOKENS
    if tpm_limit is None:
        # At the default 8,000 TPM every sub-review stays within the packing
        # target, leaving at least SPLIT_OUTPUT_RESERVE_TOKENS + 1,024 for output.
        target = budget - MIN_COMPLETION_TOKENS - CriticAgent.SPLIT_OUTPUT_RESERVE_TOKENS
        assert all(estimate_request_tokens(r) <= target for r in requests)


def test_each_assessment_and_comparison_item_is_reviewed_exactly_once(monkeypatch):
    monkeypatch.setenv("GROQ_TPM_LIMIT", "5500")
    comparison, gaps, validation, _ = chain()
    _, requests = run(Script())
    agent = CriticAgent(review_mode="auto", client=object())
    index = agent._build_evidence_index(comparison, gaps, validation)
    items = [line for line, _ in agent._comparison_entries(comparison, index)]
    for gap in gaps.gaps:
        subs = [r for r in requests if gap_id_of(r) == gap.gap_id and schema_name(r) == "_LLMSubReview"]
        evidence = [r for r in subs if "## Individual evidence assessments" in prompt(r)]
        cross = [r for r in subs if "## Cross-paper comparison items" in prompt(r)]
        assert len(evidence) + len(cross) == len(subs)
        record_lines = [agent._evidence_record_entry(item, index)[0]
                        for item in next(v for v in validation.gap_validations
                                         if v.gap_id == gap.gap_id).evidence_validations]
        for line in record_lines:
            assert sum(line in prompt(r) for r in evidence) == 1
        for line in items:
            assert sum(line in prompt(r) for r in cross) == 1
        assert len(cross) > 1                                   # the budget forced batching
        for request in cross:                                   # each keeps the joint context
            assert "Joint assessment status=" in prompt(request)
        decisions = [r for r in requests if gap_id_of(r) == gap.gap_id
                     and schema_name(r) == "_LLMGapReview"]
        assert len(decisions) == 1


# ---------------------------------------------------------------------------
# Aggregation, exact IDs and decisions
# ---------------------------------------------------------------------------


def test_aggregated_review_keeps_exact_ids_and_the_decision():
    comparison, gaps, validation, fingerprints = chain()
    result, requests = run(Script(decision="ACCEPT"))
    index = CriticAgent(review_mode="auto", client=object())._build_evidence_index(comparison, gaps, validation)
    assert result.input_fingerprints == fingerprints
    assert [r.gap_id for r in result.gap_reviews] == [g.gap_id for g in gaps.gaps]
    validation_by_id = {item.gap_id: item for item in validation.gap_validations}
    for gap, review in zip(gaps.gaps, result.gap_reviews):
        assert review.review_status is CriticReviewStatus.COMPLETE
        if acceptance_blockers(validation_by_id[gap.gap_id], gap):
            # The fixture's GAP-003/GAP-004 cite PARTIALLY_SUPPORTED chunks: the
            # guard turns the model ACCEPT into REVISE in legacy split mode.
            assert review.decision is CriticDecision.REVISE
            assert any("ACCEPT overridden to REVISE" in c for c in review.caveats)
        else:
            assert review.decision is CriticDecision.ACCEPT
        gap_chunks = {(ref.paper_id, chunk) for ref in gap.supporting_evidence
                      for chunk in ref.chunk_ids}
        cited = {(ref.paper_id, chunk) for c in review.criticisms
                 for ref in c.source_evidence for chunk in ref.chunk_ids}
        assert cited and cited <= gap_chunks                    # real chunks of this gap only
        for c in review.criticisms:
            for ref in c.source_evidence:
                for chunk in ref.chunk_ids:
                    assert index["by_key"][(ref.paper_id, chunk)]  # resolves to the index
        suggestion = review.revision_suggestions[0]
        first = index["by_label"][gap_labels_shown(next(
            r for r in requests if gap_id_of(r) == gap.gap_id))[0]]
        assert [(s.paper_id, s.chunk_ids) for s in suggestion.source_evidence] == [
            (first[0], [first[1]])]
        assert "decision caveat" in review.caveats
        assert any("Split critic review" in c for c in review.caveats)


@pytest.mark.parametrize("decision", ["REVISE", "REJECT"])
def test_revise_and_reject_decisions_are_preserved(decision):
    result, _ = run(Script(decision=decision))
    assert {r.decision for r in result.gap_reviews} == {CriticDecision[decision]}
    assert result.review_status is CriticReviewStatus.COMPLETE


def test_blocking_sub_review_prevents_accept():
    script = Script(decision="ACCEPT")
    script.sub_review_hook = lambda request, answer: (
        {**answer, "verdict": "BLOCKING_ISSUES"}
        if gap_id_of(request) == "GAP-002" and "## Cross-paper" in prompt(request) else answer)
    result, _ = run(script)
    reviews = {r.gap_id: r for r in result.gap_reviews}
    assert reviews["GAP-002"].decision is CriticDecision.REVISE
    assert reviews["GAP-002"].review_status is CriticReviewStatus.COMPLETE
    assert any("Decision changed from ACCEPT to REVISE" in c for c in reviews["GAP-002"].caveats)
    assert reviews["GAP-001"].decision is CriticDecision.ACCEPT


# ---------------------------------------------------------------------------
# Missing, failed, truncated or invalid sub-reviews never ACCEPT
# ---------------------------------------------------------------------------


def assert_incomplete(result, gap_id):
    review = next(r for r in result.gap_reviews if r.gap_id == gap_id)
    assert review.review_status is CriticReviewStatus.INCOMPLETE
    assert review.decision is not CriticDecision.ACCEPT
    assert result.review_status is CriticReviewStatus.INCOMPLETE
    assert any(error.startswith(f"{gap_id}:") for error in result.errors)
    return review


def decision_calls(requests, gap_id):
    return [r for r in requests if gap_id_of(r) == gap_id and schema_name(r) == "_LLMGapReview"]


def test_citing_a_label_the_step_was_not_shown_makes_the_review_incomplete():
    script = Script()
    # GAP-001's evidence step cites a real label that belongs to another gap.
    script.sub_review_hook = lambda request, answer: (
        {**answer, "criticisms": [criticism(["P1-E1", "P3-E2"])]}
        if gap_id_of(request) == "GAP-001" and evidence_labels_shown(request) else answer)
    result, requests = run(script)
    review = assert_incomplete(result, "GAP-001")
    assert any("it was not shown" in caveat for caveat in review.caveats)
    assert decision_calls(requests, "GAP-001") == []        # no decision without valid sub-reviews


def test_invented_label_in_the_decision_makes_the_review_incomplete():
    script = Script()
    script.decision_hook = lambda request, answer: (
        {**answer, "criticisms": [criticism(["P9-E99"])]}
        if gap_id_of(request) == "GAP-003" else answer)
    result, _ = run(script)
    assert_incomplete(result, "GAP-003")


def test_wrong_gap_id_from_a_sub_review_makes_the_review_incomplete():
    script = Script()
    script.sub_review_hook = lambda request, answer: (
        {**answer, "gap_id": "GAP-999"} if gap_id_of(request) == "GAP-002" else answer)
    result, requests = run(script)
    assert_incomplete(result, "GAP-002")
    assert decision_calls(requests, "GAP-002") == []


@pytest.mark.parametrize("failure", ["unparseable", "truncated", "provider_error"])
def test_a_failed_sub_review_leaves_the_gap_incomplete(failure):
    outcomes = {
        "unparseable": ("{not json", "stop"),
        "truncated": ('{"gap_id": "GAP-004", "verd', "length"),
        "provider_error": GroqAPIError("Groq API error (HTTP 400)", status_code=400),
    }
    script = Script()
    script.sub_review_hook = lambda request, answer: (
        outcomes[failure] if gap_id_of(request) == "GAP-004" else answer)
    result, requests = run(script)
    review = assert_incomplete(result, "GAP-004")
    if failure == "truncated":
        assert "truncated" in review.reasoning
    assert decision_calls(requests, "GAP-004") == []
    assert all(r.review_status is CriticReviewStatus.COMPLETE
               for r in result.gap_reviews if r.gap_id != "GAP-004")


def test_sub_review_that_cannot_fit_is_never_sent(monkeypatch):
    monkeypatch.setenv("GROQ_TPM_LIMIT", "2600")
    result, requests = run(Script())
    assert requests == []
    for review in result.gap_reviews:
        assert review.review_status is CriticReviewStatus.INCOMPLETE
        assert review.decision is not CriticDecision.ACCEPT
        assert "does not fit the configured token budget" in review.reasoning


def test_daily_quota_in_a_sub_review_stops_the_remaining_gaps():
    daily = GroqAPIError("Groq rate limit exceeded (HTTP 429) on tokens per day (TPD).",
                         status_code=429)
    script = Script()
    script.sub_review_hook = lambda request, answer: (
        daily if gap_id_of(request) == "GAP-002" else answer)
    result, requests = run(script)
    assert {gap_id_of(r) for r in requests} == {"GAP-001", "GAP-002"}
    statuses = {r.gap_id: r.review_status for r in result.gap_reviews}
    assert statuses["GAP-001"] is CriticReviewStatus.COMPLETE
    assert {statuses[g] for g in ("GAP-002", "GAP-003", "GAP-004")} == {
        CriticReviewStatus.INCOMPLETE}


def test_incomplete_evidence_validation_still_blocks_accept_in_split_mode():
    comparison, gaps, validation, fingerprints = chain()
    target = validation.gap_validations[0]
    broken = target.model_copy(update={"evidence_validations": [
        target.evidence_validations[0].model_copy(update={"assessment_error": "failed"}),
        *target.evidence_validations[1:]]})
    validation = validation.model_copy(update={
        "gap_validations": [broken, *validation.gap_validations[1:]]})
    sdk = ScriptedSDK(Script(decision="ACCEPT"))
    result = CriticAgent(review_mode="auto", client=GroqStructuredClient(sdk), max_retries=1).critique(
        comparison, gaps, validation, input_fingerprints=fingerprints)
    review = result.gap_reviews[0]
    assert review.review_status is CriticReviewStatus.INCOMPLETE
    assert review.decision is CriticDecision.REVISE


# ---------------------------------------------------------------------------
# Provenance, cached reviews, downstream integrity, forced modes
# ---------------------------------------------------------------------------


def test_matching_previous_reviews_are_reused_without_any_request():
    comparison, gaps, validation, fingerprints = chain()
    previous, _ = run(Script(decision="REVISE"))
    sdk = ScriptedSDK(Script())
    again = CriticAgent(review_mode="auto", client=GroqStructuredClient(sdk)).critique(
        comparison, gaps, validation, input_fingerprints=fingerprints, previous_result=previous)
    assert sdk.requests == []
    assert [r.decision for r in again.gap_reviews] == [CriticDecision.REVISE] * len(gaps.gaps)
    assert again.input_fingerprints == fingerprints


def test_split_result_passes_resume_lineage_and_downstream_integrity_checks(tmp_path):
    source = tmp_path / "run.json"
    source.write_bytes(FIXTURE.read_bytes())
    state = run_langgraph._prepare_critic_resume(source)
    sdk = ScriptedSDK(Script(decision="ACCEPT"))
    critic = CriticAgent(review_mode="auto", client=GroqStructuredClient(sdk)).critique(
        state["comparison"], state["gap_detection"], state["evidence_validation"],
        input_fingerprints=state["critic_input_fingerprints"])
    assert critic.input_fingerprints == {
        key: run_langgraph._model_fingerprint(state[key])
        for key in ("comparison", "gap_detection", "evidence_validation")}
    assert ResearchQuestionAgent._integrity_errors(
        critic, state["gap_detection"], state["evidence_validation"]) == []
    CriticResult.model_validate_json(critic.model_dump_json())   # same output schema


def test_forced_modes(monkeypatch):
    comparison, gaps, validation, fingerprints = chain()

    def single_answer(request):
        assert schema_name(request) == "_LLMCriticResult"
        return json.dumps({"gap_reviews": [{
            "gap_id": gap_id_of(request), "decision": "ACCEPT",
            "origin_assessment": "INFERRED", "origin_rationale": "x", "reasoning": "x",
            "criticisms": [], "revision_suggestions": [], "caveats": []}],
            "caveats": []}), "stop"
    # Forced single mode at the default 8,000 TPM: with the critic's fixed
    # output budget reserved, no whole-context review fits, so every request
    # is refused before sending (never sent with a shrunken, truncating cap)
    # and every gap stays INCOMPLETE.
    sdk = ScriptedSDK(single_answer)
    out = CriticAgent(client=GroqStructuredClient(sdk), review_mode="single").critique(
        comparison, gaps, validation, input_fingerprints=fingerprints)
    assert sdk.requests == []
    statuses = {r.gap_id: r.review_status for r in out.gap_reviews}
    assert set(statuses.values()) == {CriticReviewStatus.INCOMPLETE}
    assert any(e.startswith("GAP-004:") and "413" in e for e in out.errors)

    monkeypatch.setenv("GROQ_TPM_LIMIT", "20000")   # a plan with a higher limit
    sdk = ScriptedSDK(single_answer)
    out = CriticAgent(client=GroqStructuredClient(sdk), review_mode="single").critique(
        comparison, gaps, validation, input_fingerprints=fingerprints)
    assert out.review_status is CriticReviewStatus.COMPLETE
    assert [schema_name(r) for r in sdk.requests] == ["_LLMCriticResult"] * len(gaps.gaps)

    calls = []

    def split_without_budget(**kwargs):
        schema = kwargs["config"].response_schema
        calls.append(schema.__name__)
        gap_id = re.search(r"^## (GAP-\d+):", kwargs["contents"], re.M).group(1)
        payload = ({"gap_id": gap_id, "verdict": "NO_ISSUES", "summary": "ok"}
                   if schema.__name__ == "_LLMSubReview" else
                   {"gap_id": gap_id, "decision": "ACCEPT", "origin_assessment": "INFERRED",
                    "origin_rationale": "x", "reasoning": "x"})
        return SimpleNamespace(parsed=schema.model_validate(payload))
    forced = CriticAgent(client=SimpleNamespace(models=SimpleNamespace(
        generate_content=split_without_budget)), review_mode="split").critique(
        comparison, gaps, validation, input_fingerprints=fingerprints)
    assert forced.review_status is CriticReviewStatus.COMPLETE
    # No budget: one evidence and one cross-paper sub-review plus a decision per gap.
    assert calls == ["_LLMSubReview", "_LLMSubReview", "_LLMGapReview"] * len(gaps.gaps)
    with pytest.raises(ValueError, match="review_mode"):
        CriticAgent(client=object(), review_mode="parallel")
