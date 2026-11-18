"""Dry run: estimate every stage's LLM requests without sending any.

Each stage reports its calls, estimated input tokens, fixed output budget,
whether every request fits the per-request token budget (TPM fit) and its
share of the model's daily token budget. Estimates are made in one of three
ways, and each stage says which it used:

``measured``
    The stage's real agent builds its real requests from real checkpointed
    inputs; a recording client sizes each request with the same estimator and
    fixed output budget as the Groq preflight, then stops it before anything
    is sent.
``projected``
    The stage's inputs do not exist yet, so its real agent builds its real
    requests from synthetic inputs of the assumed size (placeholder text,
    worst-case evidence counts), sized the same way.
``deterministic``
    The stage makes no LLM request (the evidence matrix).

Totals count each request's input plus its full output budget, which is the
amount the provider's token-per-minute limit reserves. Real usage is lower
when outputs finish early. Nothing here calls a provider or the network.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from src.agents.artifact_provenance import ArtifactProvenance
from src.agents.groq_client import (
    GroqStructuredClient,
    _env_number,
    estimate_request_tokens,
    estimate_tokens,
    max_completion_tokens,
    tpm_budget,
    tpm_limit,
)
from src.agents.llm_provider import resolve_model
from src.agents.output_budget import output_budget
from src.rag.models import RetrievedChunk

# Groq free-tier daily token limits (Settings -> Limits in the Groq console),
# used when GROQ_TPD_LIMIT_<MODEL> is not set. Verify them for your account.
ASSUMED_DAILY_TOKEN_LIMITS = {
    "openai/gpt-oss-120b": 200_000,
    "openai/gpt-oss-20b": 200_000,
}
ASSUMED_DAILY_REQUEST_LIMITS = {
    "openai/gpt-oss-120b": 1_000,
    "openai/gpt-oss-20b": 1_000,
}


class DryRunIntercept(RuntimeError):
    """Raised by the recording client: the request was sized, never sent."""


_INTERCEPT_MESSAGE = "dry run: request recorded and not sent"


@dataclass
class RecordedCall:
    model: str
    input_tokens: int
    output_tokens: int
    per_request_budget: int | None

    @property
    def fits(self) -> bool:
        return (self.per_request_budget is None
                or self.input_tokens + self.output_tokens <= self.per_request_budget)


class RecordingClient:
    """Stands in for an LLM client; sizes each request and refuses to send it."""

    provider = "groq"

    def __init__(self):
        self._sizer = GroqStructuredClient(sdk_client=None)
        self.calls: list[RecordedCall] = []
        self.models = self

    def generate_content(self, *, model: str, contents, config=None):
        try:
            request, _ = self._sizer._build_request(model, contents, config)
            tokens = estimate_request_tokens(request)
            output = request.get("max_completion_tokens") or max_completion_tokens(model)
        except ValueError:  # non-Groq model IDs: same character-based estimate
            system = getattr(config, "system_instruction", None) or ""
            tokens = estimate_tokens(str(system)) + estimate_tokens(str(contents))
            output = getattr(config, "max_output_tokens", None) or max_completion_tokens(model)
        self.calls.append(RecordedCall(model, tokens, int(output), tpm_budget(model)))
        raise DryRunIntercept(_INTERCEPT_MESSAGE)

    def max_input_tokens(self, model: str, output_tokens: int | None = None):
        return self._sizer.max_input_tokens(model, output_tokens=output_tokens)

    def estimate_input_tokens(self, *, model: str, contents, config=None) -> int:
        try:
            return self._sizer.estimate_input_tokens(model=model, contents=contents, config=config)
        except ValueError:
            return estimate_tokens(str(contents))


@dataclass
class DryRunAssumptions:
    """Explicit assumptions for stages whose inputs do not exist yet."""

    papers: int = 6
    analysis_top_k: int = 5
    chunk_words: int = 180
    gaps: int = 3
    papers_per_gap: int = 3
    chunks_per_gap: int = 6
    clusters: int = 12
    papers_per_cluster: int = 3
    max_revisions: int = 0
    # Legacy LLM comparison only (COMPARISON_MODE=llm).
    comparison_items_per_paper: int = 6
    # Contradictions the validation model proposes per validated gap; each is
    # adjudicated by one request. None = the agent's per-gap maximum (worst
    # case); larger values are capped at that maximum; 0 = none proposed.
    contradictions_per_gap: int | None = None

    def describe(self) -> list[str]:
        return [
            f"{self.papers} papers analyzed, {self.analysis_top_k} retrieved chunks of "
            f"~{self.chunk_words} words each per paper",
            f"evidence matrix with {self.clusters} cross-paper clusters of "
            f"{self.papers_per_cluster} papers (gap detection shows as many as fit)",
            f"{self.gaps} candidate gaps (the maximum), each citing {self.chunks_per_gap} "
            f"chunks from {self.papers_per_gap} papers (the per-gap cap, worst case)",
            "every gap passes validation and is accepted by the critic "
            "(worst case for critic and question generation)",
            f"critic revision rounds: {self.max_revisions}",
        ] + ([
            f"up to {_adjudications_per_gap(self)} proposed contradiction(s) per validated "
            "gap, each adjudicated by one validation-model request (projected worst case; "
            "requests are sent only when contradictions are proposed)",
        ] if _adjudications_per_gap(self) else [])


@dataclass
class StageEstimate:
    stage: str
    model: str
    calls: int
    input_tokens: int
    output_tokens: int
    largest_request: int
    per_request_budget: int | None
    all_fit: bool
    basis: str
    notes: list[str] = field(default_factory=list)
    output_budget_per_call: int | None = None
    daily_token_limit: int | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def daily_share(self) -> float | None:
        if not self.daily_token_limit:
            return None
        return self.total_tokens / self.daily_token_limit


@dataclass
class ModelTotals:
    model: str
    calls: int
    tokens: int
    tpm_limit: int
    min_minutes: float | None
    daily_token_limit: int | None
    daily_request_limit: int | None
    daily_limit_assumed: bool = False

    @property
    def within_daily_budget(self) -> bool | None:
        if not self.daily_token_limit:
            return None
        return (self.tokens <= self.daily_token_limit
                and (not self.daily_request_limit or self.calls <= self.daily_request_limit))


@dataclass
class DryRunReport:
    stages: list[StageEstimate]
    models: list[ModelTotals]
    assumptions: list[str]

    @property
    def all_fit(self) -> bool:
        return all(stage.all_fit for stage in self.stages)

    @property
    def within_daily_budget(self) -> bool:
        return all(totals.within_daily_budget is not False for totals in self.models)

    def stage(self, name: str) -> StageEstimate:
        return next(stage for stage in self.stages if stage.stage == name)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _stage_from_calls(stage: str, model: str, calls: list[RecordedCall], basis: str,
                      notes: list[str] | None = None) -> StageEstimate:
    outputs = {call.output_tokens for call in calls}
    return StageEstimate(
        stage=stage, model=model, calls=len(calls),
        input_tokens=sum(call.input_tokens for call in calls),
        output_tokens=sum(call.output_tokens for call in calls),
        largest_request=max((call.input_tokens + call.output_tokens for call in calls), default=0),
        per_request_budget=tpm_budget(model),
        all_fit=all(call.fits for call in calls),
        basis=basis, notes=list(notes or []),
        output_budget_per_call=outputs.pop() if len(outputs) == 1 else None,
    )


def _projected(model: str, count: int, input_tokens: int, task: str) -> list[RecordedCall]:
    return [RecordedCall(model, int(input_tokens), output_budget(task), tpm_budget(model))
            for _ in range(max(0, int(count)))]


def _capture(run) -> list[RecordedCall]:
    """Run an agent call with a recording client; return what it tried to send."""
    client = RecordingClient()
    try:
        run(client)
    except Exception:  # agents stop or record a failure once a request is refused
        pass
    return client.calls


def _words(count: int, word: str = "lorem") -> str:
    return " ".join([word] * max(1, int(count)))


def _placeholder_chunks(count: int, words: int, paper_id: str = "placeholder") -> list[RetrievedChunk]:
    return [RetrievedChunk(chunk_id=f"{paper_id}_chunk_{index}", paper_id=paper_id,
                           text=_words(words))
            for index in range(count)]


def _daily_limit(model: str) -> tuple[int | None, bool]:
    configured = _env_number("GROQ_TPD_LIMIT", model, 0)
    if configured > 0:
        return int(configured), False
    assumed = ASSUMED_DAILY_TOKEN_LIMITS.get(model)
    return assumed, assumed is not None


def _daily_requests(model: str) -> int | None:
    configured = _env_number("GROQ_RPD_LIMIT", model, 0)
    if configured > 0:
        return int(configured)
    return ASSUMED_DAILY_REQUEST_LIMITS.get(model)


# ---------------------------------------------------------------------------
# synthetic inputs of the assumed size (for projected stages)
# ---------------------------------------------------------------------------

class _SyntheticStore:
    """In-memory exact-ID chunk store for projected evidence validation."""

    def __init__(self, chunk_words: int):
        self.chunk_words = chunk_words

    def get_chunks_by_ids(self, chunk_ids):
        from types import SimpleNamespace

        chunks = []
        for chunk_id in chunk_ids:
            paper_id = chunk_id.rsplit("_chunk_", 1)[0]
            chunks.append({"chunk_id": chunk_id, "text": _words(self.chunk_words),
                           "metadata": {"paper_id": paper_id, "chunk_id": chunk_id}})
        return SimpleNamespace(chunks=chunks, missing_ids=[])


def _synthetic_comparison(a: DryRunAssumptions):
    """Matrix comparison with ``a.clusters`` clusters of realistic text sizes."""
    from src.agents.evidence_matrix import EvidenceMatrixComparisonAgent
    from src.agents.evidence_matrix_models import (
        CLUSTER_DIMENSIONS,
        DIMENSION_CODES,
        MATRIX_DIMENSIONS,
        ClusterMember,
        EvidenceCluster,
        EvidenceMatrix,
        MatrixCell,
    )

    papers = [f"paper_{n:02d}" for n in range(1, max(2, a.papers) + 1)]
    statement, quote = _words(25), _words(40)
    cells = [MatrixCell(paper_id=paper, dimension=dimension, statement=statement,
                        chunk_id=f"{paper}_chunk_{index % a.analysis_top_k}", quote=quote,
                        grounding_similarity=0.8, grounded=True)
             for paper in papers for index, dimension in enumerate(MATRIX_DIMENSIONS)]
    clusters = []
    per_dimension: dict[str, int] = {}
    for number in range(a.clusters):
        dimension = CLUSTER_DIMENSIONS[number % len(CLUSTER_DIMENSIONS)]
        per_dimension[dimension] = per_dimension.get(dimension, 0) + 1
        members = [papers[(number + offset) % len(papers)]
                   for offset in range(min(a.papers_per_cluster, len(papers)))]
        chunk = (number // len(papers)) % a.analysis_top_k
        clusters.append(EvidenceCluster(
            cluster_id=f"{DIMENSION_CODES[dimension]}-{per_dimension[dimension]}",
            dimension=dimension, paper_ids=members,
            members=[ClusterMember(paper_id=paper, chunk_id=f"{paper}_chunk_{chunk}",
                                   statement=statement, quote=quote) for paper in members],
            representative_statement=statement, mean_similarity=0.75,
        ))
    matrix = EvidenceMatrix(
        paper_ids=papers, dimensions=list(MATRIX_DIMENSIONS), cells=cells, clusters=clusters,
        embedding_model="synthetic", grounding_threshold=0.55, similarity_threshold=0.7,
        min_supporting_papers=2,
    )
    comparison = EvidenceMatrixComparisonAgent(embedding_model=object())._to_comparison(matrix)
    comparison.provenance = ArtifactProvenance(
        stage="evidence_matrix", input_fingerprints={"paper_analyses": "0" * 64})
    return comparison


def _synthetic_gaps(a: DryRunAssumptions, comparison, gap_count: int):
    from src.agents.cross_paper_comparison_agent import EvidenceRef
    from src.agents.research_gap_detection_agent import (
        GapCategory,
        GapConfidence,
        GapOrigin,
        ResearchGap,
        ResearchGapDetectionResult,
    )

    available: dict[str, list[str]] = {}
    for cluster in comparison.evidence_matrix.clusters:
        for member in cluster.members:
            chunks = available.setdefault(member.paper_id, [])
            if member.chunk_id not in chunks:
                chunks.append(member.chunk_id)
    papers = list(available)
    gaps = []
    for number in range(gap_count):
        chosen = [papers[(number + offset) % len(papers)]
                  for offset in range(min(a.papers_per_gap, len(papers)))]
        refs: list[EvidenceRef] = []
        per_paper = max(1, a.chunks_per_gap // len(chosen))
        for paper in chosen:
            refs.append(EvidenceRef(paper_id=paper, chunk_ids=available[paper][:per_paper]))
        text = _words(45)
        gaps.append(ResearchGap(
            gap_id=f"GAP-{number + 1:03d}", title=_words(10), description=text,
            gap_category=GapCategory.EMPIRICAL, reasoning=text,
            supporting_paper_ids=[ref.paper_id for ref in refs],
            supporting_chunk_ids=[chunk for ref in refs for chunk in ref.chunk_ids],
            supporting_evidence=refs, evidence_summary=text, gap_origin=GapOrigin.INFERRED,
            confidence=GapConfidence.MEDIUM, confidence_reason=_words(20),
            research_opportunity=_words(30),
        ))
    result = ResearchGapDetectionResult(
        research_topic="topic", paper_count=comparison.paper_count,
        paper_ids=list(comparison.paper_ids), gaps=gaps,
        gap_category_counts={GapCategory.EMPIRICAL.value: len(gaps)}, caveats=[],
    )
    return result


def _synthetic_validation(gaps):
    from src.agents.cross_paper_comparison_agent import EvidenceRef
    from src.agents.evidence_validation_agent import (
        DeterministicCheck,
        EvidenceValidation,
        EvidenceValidationResult,
        EvidenceValidationStatus,
        GapSupportStatus,
        GapValidation,
        GapValidationStatus,
        JointGapAssessment,
    )

    quote = _words(25)
    validations = []
    for gap in gaps.gaps:
        records = []
        for ref in gap.supporting_evidence:
            for chunk_id in ref.chunk_ids:
                finding = {"status": "SUPPORTED", "attributed_contribution": _words(30),
                           "supported_contribution": _words(30),
                           "supported_contribution_quote": quote,
                           "limitations": _words(20), "author_stated": False}
                records.append(EvidenceValidation(
                    paper_id=ref.paper_id, chunk_id=chunk_id,
                    status=EvidenceValidationStatus.SUPPORTED,
                    findings=json.dumps(finding), supported_contribution=_words(30),
                    supported_contribution_quote=quote,
                    claim_evidence_alignment=_words(25),
                    evidence_summary_accuracy=_words(20),
                ))
        text = _words(50)
        validations.append(GapValidation(
            gap_id=gap.gap_id, status=GapValidationStatus.FULLY_SUPPORTED, explanation=text,
            evidence_validations=records,
            joint_assessment=JointGapAssessment(
                status=GapSupportStatus.FULLY_SUPPORTED, complete_gap_assessment=text,
                cross_paper_inference=text, evidence_summary_accuracy=text,
                research_opportunity=text, gap_origin=text, overgeneralization=text,
                limitations=text, counterevidence=text,
                source_evidence=[EvidenceRef(paper_id=r.paper_id, chunk_ids=list(r.chunk_ids))
                                 for r in gap.supporting_evidence],
            ),
            deterministic_check=DeterministicCheck(passed=True, checks=["synthetic"]),
        ))
    return EvidenceValidationResult(
        research_topic=gaps.research_topic, paper_ids=list(gaps.paper_ids),
        gap_validations=validations,
        validation_counts={"FULLY_SUPPORTED": len(validations)},
        evidence_status_counts={}, validation_mode="joint",
    )


def _synthetic_critic(gaps, fingerprints):
    from src.agents.critic_agent import (
        CriticDecision,
        CriticResult,
        CriticReviewStatus,
        GapCriticReview,
        GapOriginAssessment,
    )

    return CriticResult(
        research_topic=gaps.research_topic, review_status=CriticReviewStatus.COMPLETE,
        gap_reviews=[GapCriticReview(
            gap_id=gap.gap_id, decision=CriticDecision.ACCEPT,
            review_status=CriticReviewStatus.COMPLETE,
            origin_assessment=GapOriginAssessment.INFERRED,
            origin_rationale="synthetic", reasoning="synthetic",
            cited_evidence=list(gap.supporting_evidence),
        ) for gap in gaps.gaps],
        input_fingerprints=fingerprints, review_mode="verdict",
    )


def _fingerprints(comparison, gaps, validation) -> dict[str, str]:
    from src.agents.artifact_provenance import fingerprint_json

    def fp(model):
        return fingerprint_json(model.model_dump(mode="json", exclude={"provenance"}))

    return {"comparison": fp(comparison), "gap_detection": fp(gaps),
            "evidence_validation": fp(validation)}


# ---------------------------------------------------------------------------
# per-stage estimates
# ---------------------------------------------------------------------------

def _estimate_planner(state) -> StageEstimate:
    from src.agents.planner_agent import PlannerAgent

    model = resolve_model("PLANNER")
    calls = _capture(lambda client: PlannerAgent(model=model, client=client)
                     .create_search_plan(state["research_topic"]))
    return _stage_from_calls("planner", model, calls, "measured")


def _estimate_analysis(state, assumptions: DryRunAssumptions) -> StageEstimate:
    from src.agents.paper_analysis_agent import PaperAnalysisAgent

    model = resolve_model("ANALYSIS")
    analyses = state.get("paper_analyses") or []
    measured: list[RecordedCall] = []
    for result in analyses:
        measured += _capture(lambda client, result=result: PaperAnalysisAgent(
            model=model, client=client).analyze(list(result.evidence)))
    remaining = max(0, assumptions.papers - len(analyses))
    projected: list[RecordedCall] = []
    if remaining:
        typical = _capture(lambda client: PaperAnalysisAgent(model=model, client=client).analyze(
            _placeholder_chunks(assumptions.analysis_top_k, assumptions.chunk_words)))
        projected = typical * remaining
    basis = "measured" if not projected else ("partly measured" if measured else "projected")
    notes = [] if not projected else [
        f"{remaining} paper(s) projected with the real analysis prompt and "
        f"{assumptions.analysis_top_k} placeholder chunks of {assumptions.chunk_words} words."]
    return _stage_from_calls("paper analysis", model, measured + projected, basis, notes)


def _estimate_comparison(state, assumptions: DryRunAssumptions) -> StageEstimate:
    from src.agents.evidence_matrix import comparison_mode

    if comparison_mode() == "matrix":
        return StageEstimate(
            stage="evidence matrix", model="local embeddings", calls=0,
            input_tokens=0, output_tokens=0, largest_request=0, per_request_budget=None,
            all_fit=True, basis="deterministic",
            notes=["No LLM request: statements are grounded and clustered with local "
                   "embeddings (set COMPARISON_MODE=llm for the legacy LLM comparison)."],
        )
    return _estimate_legacy_comparison(state, assumptions)


def _estimate_legacy_comparison(state, assumptions: DryRunAssumptions) -> StageEstimate:
    from src.agents.cross_paper_comparison_agent import (
        CrossPaperComparisonAgent,
        LLMComparison,
    )

    name = "cross-paper comparison (legacy)"
    model = resolve_model("COMPARISON")
    analyses = state.get("paper_analyses") or []
    papers = len(analyses) or assumptions.papers
    if len(analyses) >= 2:
        first = _capture(lambda client: CrossPaperComparisonAgent(model=model, client=client)
                         .compare_many(analyses, research_topic=state.get("research_topic")))
        agent = CrossPaperComparisonAgent(model=model, client=RecordingClient())
        index, papers_text = agent._prepare(analyses)
        single = agent._build_contents(papers_text, len(analyses), state.get("research_topic"))
        budget = agent._input_budget()
        staged = budget is not None and agent._estimate(
            single, agent.SYSTEM_INSTRUCTION, LLMComparison) > budget
        if not staged:
            return _stage_from_calls(name, model, first, "measured")
        grounding = []
        for result in analyses:
            tokens = estimate_tokens(" ".join(
                " ".join(chunk.text.split())[:agent.max_evidence_chars] for chunk in result.evidence))
            grounding += _projected(model, 1, tokens + 150 * assumptions.comparison_items_per_paper
                                    + estimate_tokens(agent.STAGED_GROUNDING_INSTRUCTION),
                                    "comparison_grounding")
        return _stage_from_calls(name, model, first + grounding, "partly measured",
                                 ["Staged: synthesis measured; one grounding request per paper projected."])
    synthesis = _projected(model, 1, 650 * papers, "comparison_synthesis")
    grounding = _projected(model, papers, 1400, "comparison_grounding")
    return _stage_from_calls(name, model, synthesis + grounding, "projected",
                             ["Staged comparison assumed (six analyses exceed one request)."])


def _estimate_gaps(state, assumptions: DryRunAssumptions) -> StageEstimate:
    from src.agents.evidence_matrix import comparison_mode
    from src.agents.research_gap_detection_agent import ResearchGapDetectionAgent

    model = resolve_model("GAP")
    comparison, basis, notes = state.get("comparison"), "measured", []
    if comparison is None:
        if comparison_mode() == "llm":
            instruction = estimate_tokens(ResearchGapDetectionAgent.SYSTEM_INSTRUCTION)
            return _stage_from_calls("gap detection", model, _projected(
                model, 1, instruction + 700 * assumptions.papers, "gap_detection"), "projected")
        comparison, basis = _synthetic_comparison(assumptions), "projected"
        notes.append(f"Real gap-detection prompt over a synthetic matrix of "
                     f"{assumptions.clusters} clusters, trimmed to the token budget.")
    calls = _capture(lambda client: ResearchGapDetectionAgent(model=model, client=client)
                     .detect_gaps(comparison, research_topic=state.get("research_topic")))
    if not calls:
        notes.append("No request: the matrix has no cross-paper cluster, or the smallest "
                     "request does not fit the budget (refused before sending).")
    return _stage_from_calls("gap detection", model, calls, basis, notes)


def _gap_inputs(state, assumptions: DryRunAssumptions, vector_store):
    """(comparison, gaps, store, basis) for validation: real or synthetic."""
    gaps, comparison = state.get("gap_detection"), state.get("comparison")
    if gaps is not None and comparison is not None and vector_store is not None:
        return comparison, gaps, vector_store, "measured"
    count = len(gaps.gaps) if gaps is not None else assumptions.gaps
    synthetic = _synthetic_comparison(assumptions)
    return (synthetic, _synthetic_gaps(assumptions, synthetic, count),
            _SyntheticStore(assumptions.chunk_words), "projected")


def _estimate_validation(state, assumptions: DryRunAssumptions, vector_store) -> StageEstimate:
    from src.agents.evidence_validation_agent import EvidenceValidationAgent

    model = resolve_model("VALIDATION")
    comparison, gaps, store, basis = _gap_inputs(state, assumptions, vector_store)
    matrix = getattr(comparison, "evidence_matrix", None)
    calls = _capture(lambda client: EvidenceValidationAgent(
        vector_store=store, model=model, client=client).validate(gaps, evidence_matrix=matrix))
    notes = ["One joint request per gap after deterministic checks; no per-chunk requests."]
    if basis == "projected":
        notes.append(f"Real validation prompt with {assumptions.chunks_per_gap} placeholder "
                     f"chunks of {assumptions.chunk_words} words per gap.")
    if len(calls) < len(gaps.gaps):
        notes.append("Fewer requests than gaps: deterministic checks failed, so the real run "
                     "would stop before any semantic request.")
    return _stage_from_calls("evidence validation", model, calls, basis, notes)


def _estimate_critic(state, assumptions: DryRunAssumptions, vector_store) -> StageEstimate:
    from src.agents.critic_agent import CriticAgent

    model = resolve_model("CRITIC")
    if (state.get("evidence_validation") is not None and state.get("gap_detection") is not None
            and state.get("comparison") is not None):
        comparison, gaps = state["comparison"], state["gap_detection"]
        validation, basis = state["evidence_validation"], "measured"
    else:
        comparison, gaps, _, _ = _gap_inputs(state, assumptions, None)
        validation, basis = _synthetic_validation(gaps), "projected"
    calls = _capture(lambda client: CriticAgent(model=model, client=client).critique(
        comparison, gaps, validation,
        input_fingerprints=_fingerprints(comparison, gaps, validation)))
    return _stage_from_calls(
        "critic", model, calls, basis,
        ["One ACCEPT/REJECT request per validated gap; unsupported gaps are rejected "
         "without a request; no split reviews."])


def _estimate_questions(state, assumptions: DryRunAssumptions) -> StageEstimate:
    from src.agents.research_question_agent import ResearchQuestionAgent

    model = resolve_model("QUESTION")
    if state.get("critic_result") is not None and state.get("critic_input_fingerprints"):
        calls = _capture(lambda client: ResearchQuestionAgent(model=model, client=client)
                         .generate_questions(state["critic_result"], state["gap_detection"],
                                             state["evidence_validation"],
                                             critic_input_fingerprints=state["critic_input_fingerprints"]))
        return _stage_from_calls("research questions", model, calls, "measured")
    comparison, gaps, _, _ = _gap_inputs(state, assumptions, None)
    validation = _synthetic_validation(gaps)
    fingerprints = _fingerprints(comparison, gaps, validation)
    critic = _synthetic_critic(gaps, fingerprints)
    calls = _capture(lambda client: ResearchQuestionAgent(model=model, client=client)
                     .generate_questions(critic, gaps, validation,
                                         critic_input_fingerprints=fingerprints))
    return _stage_from_calls("research questions", model, calls, "projected",
                             ["One request per critic-accepted gap; all gaps assumed accepted."])


def _adjudications_per_gap(assumptions: DryRunAssumptions) -> int:
    """Worst-case adjudication requests per validated gap (0 when none proposed)."""
    from src.agents.evidence_validation_agent import MAX_ADJUDICATIONS_PER_GAP

    value = assumptions.contradictions_per_gap
    if value is None:
        return MAX_ADJUDICATIONS_PER_GAP
    # More proposals than the maximum make the gap incomplete without any
    # adjudication request, so the maximum is also the worst case.
    return max(0, min(int(value), MAX_ADJUDICATIONS_PER_GAP))


ADJUDICATION_PROPOSITION_WORDS = 40
ADJUDICATION_STAGE = "contradiction adjudication (worst case)"


def _estimate_adjudication(validation: StageEstimate,
                           assumptions: DryRunAssumptions) -> StageEstimate | None:
    """Projected worst-case contradiction-adjudication requests (None when none)."""
    from src.agents.cross_paper_comparison_agent import EvidenceRef
    from src.agents.evidence_validation_agent import (
        MAX_ADJUDICATIONS_PER_GAP,
        EvidenceContradiction,
        EvidenceValidationAgent,
    )

    per_gap = _adjudications_per_gap(assumptions)
    if per_gap <= 0 or validation.calls <= 0:
        return None
    model = validation.model
    # Worst-case size: each side quotes a whole chunk.
    proposal = EvidenceContradiction(
        proposition_a=_words(ADJUDICATION_PROPOSITION_WORDS),
        source_a=EvidenceRef(paper_id="placeholder_a", chunk_ids=["placeholder_a_chunk_0"]),
        supporting_quote_a=_words(assumptions.chunk_words),
        proposition_b=_words(ADJUDICATION_PROPOSITION_WORDS),
        source_b=EvidenceRef(paper_id="placeholder_b", chunk_ids=["placeholder_b_chunk_0"]),
        supporting_quote_b=_words(assumptions.chunk_words),
        explanation=_words(20),
    )
    one = _capture(lambda client: EvidenceValidationAgent(
        vector_store=_SyntheticStore(assumptions.chunk_words), model=model,
        client=client)._adjudicate_contradiction(proposal))
    calls = one * (per_gap * validation.calls)
    return _stage_from_calls(
        ADJUDICATION_STAGE, model, calls, "projected",
        [f"Projected worst case: {per_gap} adjudication request(s) for each of "
         f"{validation.calls} validated gap(s) (agent maximum {MAX_ADJUDICATIONS_PER_GAP} "
         "per gap). Requests are sent only when the validation model proposes "
         "contradictions; usually there are fewer or none.",
         f"Real adjudication prompt with two {ADJUDICATION_PROPOSITION_WORDS}-word "
         f"propositions and two whole-chunk quotes of {assumptions.chunk_words} words."])


def _estimate_revisions(validation: StageEstimate, critic: StageEstimate,
                        assumptions: DryRunAssumptions) -> StageEstimate | None:
    rounds = assumptions.max_revisions
    if rounds <= 0:
        return None
    calls = rounds * (validation.calls + critic.calls)
    model = critic.model if critic.model == validation.model else f"{validation.model} + {critic.model}"
    return StageEstimate(
        stage="revision rounds (worst case)", model=model, calls=calls,
        input_tokens=rounds * (validation.input_tokens + critic.input_tokens),
        output_tokens=rounds * (validation.output_tokens + critic.output_tokens),
        largest_request=max(validation.largest_request, critic.largest_request),
        per_request_budget=critic.per_request_budget,
        all_fit=validation.all_fit and critic.all_fit, basis="projected",
        notes=["Each round re-validates and re-reviews revised gaps; worst case "
               "assumes every gap is revised every round. The default critic returns "
               "only ACCEPT/REJECT, so revisions apply only to legacy critic modes."],
    )


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def estimate_run(state: dict, *, assumptions: DryRunAssumptions | None = None,
                 vector_store=None) -> DryRunReport:
    """Estimate every stage of a topic run from its (possibly partial) state."""
    assumptions = assumptions or DryRunAssumptions()
    validation = _estimate_validation(state, assumptions, vector_store)
    adjudication = _estimate_adjudication(validation, assumptions)
    critic = _estimate_critic(state, assumptions, vector_store)
    stages = [
        _estimate_planner(state),
        _estimate_analysis(state, assumptions),
        _estimate_comparison(state, assumptions),
        _estimate_gaps(state, assumptions),
        validation,
        *([adjudication] if adjudication is not None else []),
        critic,
        _estimate_questions(state, assumptions),
    ]
    revisions = _estimate_revisions(validation, critic, assumptions)
    if revisions is not None:
        stages.append(revisions)

    per_model: dict[str, list[int]] = {}

    def add(model: str, calls: int, tokens: int) -> None:
        entry = per_model.setdefault(model, [0, 0])
        entry[0] += calls
        entry[1] += tokens

    for stage in stages:
        if stage is not revisions and stage.calls:
            add(stage.model, stage.calls, stage.total_tokens)
    if revisions is not None:  # attribute worst-case rounds to their own models
        for row in (validation, critic):
            add(row.model, assumptions.max_revisions * row.calls,
                assumptions.max_revisions * row.total_tokens)
    for stage in stages:
        if stage.calls and stage is not revisions:
            stage.daily_token_limit = _daily_limit(stage.model)[0]
    models = []
    for model, (calls, tokens) in per_model.items():
        limit = tpm_limit(model)
        daily, assumed = _daily_limit(model)
        models.append(ModelTotals(
            model=model, calls=calls, tokens=tokens, tpm_limit=limit,
            min_minutes=(tokens / limit) if limit else None,
            daily_token_limit=daily, daily_request_limit=_daily_requests(model),
            daily_limit_assumed=assumed,
        ))
    return DryRunReport(stages=stages, models=models, assumptions=assumptions.describe())


def format_report(report: DryRunReport) -> str:
    """Human-readable dry-run table."""
    lines = ["Dry run: no request was sent. Tokens = estimated input + fixed output budget.", ""]
    budgets = sorted({(stage.model, stage.per_request_budget) for stage in report.stages
                      if stage.calls and stage.per_request_budget})
    for model, budget in budgets:
        lines.append(f"Per-request budget for {model}: {budget:,} tokens "
                     f"(90% of {tpm_limit(model):,} TPM).")
    lines.append("")
    header = (f"{'stage':<30}{'model':<22}{'calls':>6}{'input':>9}{'out/call':>9}"
              f"{'total':>9}{'largest':>9}  fits  {'day%':>5}  basis")
    lines += [header, "-" * len(header)]
    for stage in report.stages:
        model = stage.model if len(stage.model) <= 21 else stage.model[:20] + "~"
        per_call = f"{stage.output_budget_per_call:,}" if stage.output_budget_per_call else "-"
        share = f"{stage.daily_share:.0%}" if stage.daily_share is not None else "-"
        fits = ("yes " if stage.all_fit else "NO  ") if stage.calls else "n/a "
        lines.append(
            f"{stage.stage:<30}{model:<22}{stage.calls:>6}{stage.input_tokens:>9,}"
            f"{per_call:>9}{stage.total_tokens:>9,}{stage.largest_request:>9,}  "
            f"{fits}  {share:>5}  {stage.basis}"
        )
        for note in stage.notes:
            lines.append(f"{'':<4}- {note}")
    lines += ["", "Per model (one run):"]
    for totals in report.models:
        minutes = (f"{totals.min_minutes:.1f} min at {totals.tpm_limit:,} TPM"
                   if totals.min_minutes else "unmanaged")
        if totals.daily_token_limit:
            source = "assumed free tier" if totals.daily_limit_assumed else "configured"
            daily = (f"{totals.tokens / totals.daily_token_limit:.0%} of "
                     f"{totals.daily_token_limit:,} tokens/day ({source})")
            if totals.daily_request_limit:
                daily += f", {totals.calls} of {totals.daily_request_limit:,} requests/day"
            daily += "; within daily budget" if totals.within_daily_budget else "; OVER daily budget"
        else:
            daily = "daily limit unknown (set GROQ_TPD_LIMIT_<MODEL> from your Groq console)"
        lines.append(f"  {totals.model}: {totals.calls} calls, {totals.tokens:,} tokens; "
                     f"minimum pacing {minutes}; {daily}")
    if not report.all_fit:
        lines += ["", "WARNING: at least one request exceeds its per-request token budget and "
                      "would be refused before sending."]
    if not report.within_daily_budget:
        lines += ["", "WARNING: this run needs more than a model's daily token or request budget."]
    lines += ["", "Assumptions for projected stages:"] + [f"  - {item}" for item in report.assumptions]
    return "\n".join(lines)
