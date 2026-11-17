"""LangGraph orchestration for the existing research-gap agents.

The graph owns routing and provenance handoffs only. All model-backed work is
delegated to the existing agent implementations.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from src.agents.artifact_provenance import (
    ArtifactProvenance,
    fingerprint_json,
    require_fingerprint,
    require_recorded_fingerprint,
)
from src.agents.critic_agent import (
    CriticAgent,
    CriticDecision,
    CriticResult,
    CriticReviewStatus,
)
from src.agents.cross_paper_comparison_agent import (
    CrossPaperComparisonAgent,
    EvidenceRef,
    ExcludedPaper,
    MultiPaperComparison,
)
from src.agents.evidence_validation_agent import (
    EvidenceValidationAgent,
    EvidenceValidationResult,
    EvidenceValidationStatus,
    GapSupportStatus,
    GapValidationStatus,
)
from src.agents.paper_analysis_agent import PaperAnalysisResult
from src.agents.paper_analysis_pipeline import PaperAnalysisPipeline
from src.agents.planner_agent import SearchPlan
from src.agents.planner_retrieval_agent import PlannerRetrievalAgent
from src.agents.research_gap_detection_agent import (
    GapOrigin,
    ResearchGap,
    ResearchGapDetectionAgent,
    ResearchGapDetectionResult,
)
from src.agents.research_question_agent import ResearchQuestionAgent, ResearchQuestionResult
from src.embeddings import LocalEmbeddingModel
from src.evaluation.identity import validate_exact_paper_ids
from src.rag import PaperIngestionPipeline, RAGRetriever
from src.retrieval import PaperRecord
from src.vectorstore import ChromaVectorStore
from src.vectorstore.chroma_store import collection_name_for


class ResearchWorkflowState(TypedDict, total=False):
    """Typed state; structured artifacts use the project's existing models."""

    research_topic: str
    evaluation_prompt_id: str
    canonical_prompt_text: str
    analysis_query: str
    max_results_per_source: int
    minimum_papers: int
    max_papers: int | None
    max_revisions: int
    revision_count: int
    pdf_paths: dict[str, str]
    evaluation_papers: list[PaperRecord]
    evaluation_expected_paper_ids: list[str]
    evaluation_chroma_dir: str
    evaluation_chroma_collection: str
    search_plan: SearchPlan
    papers: list[PaperRecord]
    paper_analyses: list[PaperAnalysisResult]
    excluded_papers: list[ExcludedPaper]
    comparison: MultiPaperComparison
    gap_detection: ResearchGapDetectionResult
    evidence_validation: EvidenceValidationResult
    critic_result: CriticResult
    research_questions: ResearchQuestionResult
    critic_input_fingerprints: dict[str, str]
    revision_applied: bool
    workflow_status: str
    stop_reason: str | None
    resume_from_evidence: bool
    resume_from_critic: bool
    revised_gap_ids: list[str]
    # Append-only record of resume/recovery steps that produced this artifact.
    run_lineage: list[dict]


@dataclass
class WorkflowAgents:
    """Injectable existing agents/pipeline; useful for offline tests."""

    retrieval: PlannerRetrievalAgent | None
    paper_analysis: PaperAnalysisPipeline
    comparison: CrossPaperComparisonAgent
    gap_detection: ResearchGapDetectionAgent
    evidence_validation: EvidenceValidationAgent
    critic: CriticAgent
    research_questions: ResearchQuestionAgent


def default_workflow_agents(
    *,
    evaluation_chroma_dir: Path | None = None,
    evaluation_chroma_collection: str | None = None,
) -> WorkflowAgents:
    """Construct project agents, isolating storage for frozen-corpus evaluation."""
    evaluation_mode = evaluation_chroma_dir is not None or evaluation_chroma_collection is not None
    if evaluation_mode:
        if evaluation_chroma_dir is None or not evaluation_chroma_collection:
            raise ValueError("Evaluation Chroma directory and collection must both be configured.")
        evaluation_directory = Path(evaluation_chroma_dir).resolve()
        conventional_directory = (Path(__file__).resolve().parents[2] / "chroma_db").resolve()
        conventional_collection = collection_name_for("BAAI/bge-small-en-v1.5")
        if evaluation_directory == conventional_directory:
            raise ValueError("Multi-Agent evaluation Chroma directory must be separate from Conventional RAG.")
        if evaluation_chroma_collection == conventional_collection:
            raise ValueError("Multi-Agent evaluation Chroma collection must be separate from Conventional RAG.")

        embedding_model = LocalEmbeddingModel()
        vector_store = ChromaVectorStore.for_embedder(
            embedding_model,
            persist_directory=str(evaluation_directory),
            collection_name=evaluation_chroma_collection,
        )
        ingestion = PaperIngestionPipeline(
            embedding_model=embedding_model,
            vector_store=vector_store,
        )
        retriever = RAGRetriever(
            embedding_model=embedding_model,
            vector_store=vector_store,
        )
        pipeline = PaperAnalysisPipeline(
            ingestion_pipeline=ingestion,
            rag_retriever=retriever,
        )
        evidence_validation = EvidenceValidationAgent(vector_store=vector_store)
        planner_retrieval = None
    else:
        planner_retrieval = PlannerRetrievalAgent()
        pipeline = PaperAnalysisPipeline()
        evidence_validation = EvidenceValidationAgent()
    return WorkflowAgents(
        retrieval=planner_retrieval,
        paper_analysis=pipeline,
        comparison=pipeline.comparison_agent,
        gap_detection=ResearchGapDetectionAgent(),
        evidence_validation=evidence_validation,
        critic=CriticAgent(),
        research_questions=ResearchQuestionAgent(),
    )


def _model_fingerprint(model) -> str:
    return fingerprint_json(model.model_dump(mode="json", exclude={"provenance"}))


def _critic_fingerprints(state: ResearchWorkflowState) -> dict[str, str]:
    return {
        "comparison": _model_fingerprint(state["comparison"]),
        "gap_detection": _model_fingerprint(state["gap_detection"]),
        "evidence_validation": _model_fingerprint(state["evidence_validation"]),
    }


def _paper_id(paper: PaperRecord) -> str:
    return paper.source_id or paper.doi or ""


def _all_comparison_evidence(comparison: MultiPaperComparison) -> set[tuple[str, str]]:
    refs: list[EvidenceRef] = []
    for item in comparison.dimension_summaries:
        refs.extend(item.evidence)
    for group in (
        comparison.recurring_patterns,
        comparison.common_limitations,
        comparison.methodological_differences,
    ):
        for item in group:
            refs.extend(item.evidence)
    for item in comparison.conflicting_findings:
        refs.extend(item.position_a_evidence)
        refs.extend(item.position_b_evidence)
    for item in comparison.underexplored_directions:
        refs.extend(item.evidence)
    return {(ref.paper_id, chunk_id) for ref in refs for chunk_id in ref.chunk_ids}


def _evidence_validation_complete(
    gap_result: ResearchGapDetectionResult,
    validation: EvidenceValidationResult,
) -> bool:
    """Check completion and reference integrity without judging support strength."""
    if validation.retrieval_failures or validation.semantic_failures:
        return False
    if not validation.gap_validations:
        return False
    validation_ids = [item.gap_id for item in validation.gap_validations]
    if len(validation_ids) != len(set(validation_ids)):
        return False
    if set(validation_ids) != {gap.gap_id for gap in gap_result.gaps}:
        return False
    if set(validation.paper_ids) != set(gap_result.paper_ids):
        return False
    gap_by_id = {gap.gap_id: gap for gap in gap_result.gaps}
    for validated_gap in validation.gap_validations:
        if (validated_gap.status is GapValidationStatus.VALIDATION_INCOMPLETE
                or validated_gap.joint_assessment is None):
            return False
        expected = [
            (ref.paper_id, chunk_id)
            for ref in gap_by_id[validated_gap.gap_id].supporting_evidence
            for chunk_id in ref.chunk_ids
        ]
        actual = [(item.paper_id, item.chunk_id)
                  for item in validated_gap.evidence_validations]
        if (not expected or len(expected) != len(set(expected))
                or len(actual) != len(set(actual)) or set(expected) != set(actual)):
            return False
        if any(item.status not in (
                EvidenceValidationStatus.SUPPORTED,
                EvidenceValidationStatus.PARTIALLY_SUPPORTED,
                EvidenceValidationStatus.UNSUPPORTED,
        ) or item.retrieval_error is not None or item.assessment_error is not None
               for item in validated_gap.evidence_validations):
            return False
    return True


def _questions_evidence_eligible(
    gap_result: ResearchGapDetectionResult,
    validation: EvidenceValidationResult,
    gap_ids: set[str] | None = None,
) -> bool:
    """Reject question generation where any selected gap has unsupported evidence.

    Validation completeness is always checked for every gap. Support strength is
    checked for ``gap_ids`` (the critic-accepted gaps), or for all gaps when no
    subset is given; critic-rejected gaps never receive questions.
    """
    if not _evidence_validation_complete(gap_result, validation):
        return False
    by_id = {item.gap_id: item for item in validation.gap_validations}
    selected = [gap for gap in gap_result.gaps
                if gap_ids is None or gap.gap_id in gap_ids]
    if not selected:
        return False
    for gap in selected:
        item = by_id[gap.gap_id]
        if (item.status is GapValidationStatus.UNSUPPORTED
                or item.joint_assessment.status is GapSupportStatus.UNSUPPORTED
                or not any(record.status in (
                    EvidenceValidationStatus.SUPPORTED,
                    EvidenceValidationStatus.PARTIALLY_SUPPORTED,
                ) for record in item.evidence_validations)):
            return False
    return True


MIN_GAP_SUPPORTING_PAPERS = 2


def _apply_revision_suggestions(
    gap_result: ResearchGapDetectionResult,
    comparison: MultiPaperComparison,
    critic: CriticResult,
) -> tuple[ResearchGapDetectionResult, list[str]]:
    """Apply only constrained critic text revisions, retaining verified evidence."""
    safe_fields = {
        "title",
        "description",
        "reasoning",
        "evidence_summary",
        "confidence_reason",
        "research_opportunity",
    }
    origin_fields = {"origin", "origin_assessment", "gap_origin"}
    evidence_label_fields = {"gap_evidence_labels", "supporting_evidence_labels"}
    allowed_refs = _all_comparison_evidence(comparison)
    reviews = {review.gap_id: review for review in critic.gap_reviews}
    updated_gaps: list[ResearchGap] = []
    changed_gap_ids: list[str] = []

    for gap in gap_result.gaps:
        review = reviews.get(gap.gap_id)
        updates: dict = {}
        # Evidence cited by applied text/origin suggestions is always retained,
        # independently of the order in which a label replacement appears.
        added_refs: list[EvidenceRef] = []
        replacement_refs: list[EvidenceRef] | None = None
        if (review and review.review_status is CriticReviewStatus.COMPLETE
                and review.decision is CriticDecision.REVISE):
            for suggestion in review.revision_suggestions:
                field = suggestion.field.strip().lower().replace("-", "_").replace(" ", "_")
                proposed = suggestion.proposed_text.strip()
                if not proposed:
                    continue
                valid_refs = not any(
                    (ref.paper_id, chunk_id) not in allowed_refs
                    for ref in suggestion.source_evidence
                    for chunk_id in ref.chunk_ids
                )
                if not valid_refs:
                    continue
                if field in safe_fields:
                    updates[field] = proposed
                    added_refs.extend(suggestion.source_evidence)
                elif field in origin_fields:
                    origin_value = proposed.strip().lower().replace(" ", "_")
                    if origin_value in {"inferred", "inferred_cross_paper"}:
                        updates["gap_origin"] = GapOrigin.INFERRED
                    elif origin_value in {"explicit", "explicitly_stated", "author_stated"}:
                        updates["gap_origin"] = GapOrigin.EXPLICIT
                    else:
                        continue
                    added_refs.extend(suggestion.source_evidence)
                elif field in evidence_label_fields:
                    # Labels are presentation aliases; the Critic's exact source
                    # references are the canonical IDs/chunks to retain.
                    if not suggestion.source_evidence:
                        continue
                    replacement_refs = list(suggestion.source_evidence)

        if updates or replacement_refs is not None:
            refs = list(replacement_refs if replacement_refs is not None
                        else gap.supporting_evidence)
            seen = {(ref.paper_id, chunk) for ref in refs for chunk in ref.chunk_ids}
            for ref in added_refs:
                new_chunks = [chunk for chunk in ref.chunk_ids if (ref.paper_id, chunk) not in seen]
                if new_chunks:
                    refs.append(EvidenceRef(paper_id=ref.paper_id, chunk_ids=new_chunks))
                    seen.update((ref.paper_id, chunk) for chunk in new_chunks)
            revised_paper_ids = list(dict.fromkeys(ref.paper_id for ref in refs))
            if len(revised_paper_ids) < MIN_GAP_SUPPORTING_PAPERS:
                # A revision may not turn a cross-paper gap into a single-paper
                # limitation; leave this gap unchanged so the critic decision
                # remains unresolved instead of silently narrowing the evidence.
                updated_gaps.append(gap)
                continue
            updates["supporting_evidence"] = refs
            updates["supporting_paper_ids"] = revised_paper_ids
            updates["supporting_chunk_ids"] = list(dict.fromkeys(
                chunk for ref in refs for chunk in ref.chunk_ids
            ))
            updated_gaps.append(gap.model_copy(update=updates))
            changed_gap_ids.append(gap.gap_id)
        else:
            updated_gaps.append(gap)

    if not changed_gap_ids:
        return gap_result, []
    revised = gap_result.model_copy(update={
        "gaps": updated_gaps,
        "provenance": ArtifactProvenance(
            stage="critic_revision",
            input_fingerprints={
                "comparison": _model_fingerprint(comparison),
                "prior_gap_detection": _model_fingerprint(gap_result),
                "critic": fingerprint_json(critic.model_dump(mode="json")),
            },
        ),
        "caveats": list(dict.fromkeys(gap_result.caveats + [
            "Gap text was updated from critic suggestions; all evidence references remain tied to comparison-owned chunks and are revalidated.",
        ])),
    })
    return revised, changed_gap_ids


def build_research_graph(agents: WorkflowAgents | None = None):
    """Create the compiled LangGraph workflow using existing agent instances."""
    agents = agents or default_workflow_agents()

    def retrieve(state: ResearchWorkflowState) -> dict:
        if state.get("evaluation_prompt_id"):
            canonical_text = state.get("canonical_prompt_text")
            if not canonical_text or state["research_topic"] != canonical_text or state.get("analysis_query") != canonical_text:
                raise ValueError("Evaluation workflow topic and analysis query must equal the canonical prompt text.")
            expected_ids = state.get("evaluation_expected_paper_ids", [])
            papers = state.get("evaluation_papers", [])
            validate_exact_paper_ids(
                [_paper_id(paper) for paper in papers], expected_ids,
                "Frozen evaluation input",
            )
            plan = SearchPlan(
                original_topic=canonical_text,
                research_domain="Frozen six-paper evaluation corpus",
                search_queries=[],
                target_modalities=[],
                evaluation_focus=["Analyze only the six manifest-included papers."],
            )
            return {
                "search_plan": plan,
                "papers": papers,
                "workflow_status": "RUNNING",
                "stop_reason": None,
            }
        if agents.retrieval is None:
            raise ValueError("Normal LangGraph retrieval is not configured.")
        plan, papers = agents.retrieval.search(
            state["research_topic"],
            max_results_per_source=state.get("max_results_per_source", 5),
        )
        return {
            "search_plan": plan,
            "papers": papers,
            "workflow_status": "RUNNING",
            "stop_reason": None,
        }

    def analyze_papers(state: ResearchWorkflowState) -> dict:
        paths = state.get("pdf_paths", {})
        evaluation_mode = bool(state.get("evaluation_prompt_id"))
        expected_ids = state.get("evaluation_expected_paper_ids", [])
        if evaluation_mode:
            validate_exact_paper_ids(
                [_paper_id(paper) for paper in state.get("papers", [])],
                expected_ids, "Frozen evaluation paper input",
            )
        max_papers = None if evaluation_mode else state.get("max_papers")
        selected = state["papers"][:max_papers] if max_papers else state["papers"]
        analyses: list[PaperAnalysisResult] = []
        excluded: list[ExcludedPaper] = []
        # Each workflow run starts with fresh run-scoped quota state; it is
        # not cleared between papers, so a daily 429 stops this run's papers.
        begin_run = getattr(agents.paper_analysis, "begin_run", None)
        if callable(begin_run):
            begin_run()
        for paper in selected:
            paper_id = _paper_id(paper)
            try:
                resolved_paper_id = agents.paper_analysis._get_paper_id(paper)
                if evaluation_mode and resolved_paper_id != paper_id:
                    raise ValueError(
                        f"Analysis pipeline resolved {resolved_paper_id!r} instead of frozen ID {paper_id!r}."
                    )
                paper_id = resolved_paper_id
                path = paths.get(paper_id)
                if not path:
                    path = agents.paper_analysis._resolve_pdf_path(paper, paper_id, paths)
                analysis = agents.paper_analysis.analyze_paper(
                    paper=paper,
                    pdf_path=str(Path(path)),
                    analysis_query=state["analysis_query"],
                )
                evidence_owners = {item.paper_id for item in analysis.evidence}
                if not evidence_owners or evidence_owners != {paper_id}:
                    raise ValueError(
                        f"Analysis evidence for {paper_id!r} is missing or contains "
                        f"chunks owned by other papers: {sorted(evidence_owners)}."
                    )
                analyses.append(analysis)
            except Exception as error:
                if evaluation_mode:
                    raise ValueError(
                        f"Frozen evaluation paper {paper_id!r} could not be analyzed: {error}"
                    ) from error
                excluded.append(ExcludedPaper(
                    paper_id=paper.source_id or paper.doi or paper.title or "unknown",
                    reason=f"{type(error).__name__}: {error}",
                ))
        if evaluation_mode:
            analyzed_ids = []
            for paper, analysis in zip(selected, analyses):
                paper_id = _paper_id(paper)
                evidence_ids = [item.paper_id for item in analysis.evidence]
                if not evidence_ids or set(evidence_ids) != {paper_id}:
                    raise ValueError(
                        f"Analysis evidence for {paper_id!r} is missing or contains an unexpected paper."
                    )
                analyzed_ids.append(paper_id)
            validate_exact_paper_ids(analyzed_ids, expected_ids, "Frozen evaluation analyses")
        return {"paper_analyses": analyses, "excluded_papers": excluded}

    def compare_papers(state: ResearchWorkflowState) -> dict:
        comparison = agents.comparison.compare_many(
            state["paper_analyses"], research_topic=state["research_topic"]
        )
        if state.get("evaluation_prompt_id"):
            validate_exact_paper_ids(
                comparison.paper_ids,
                state.get("evaluation_expected_paper_ids", []),
                "Evaluation comparison artifact",
            )
        require_recorded_fingerprint(
            comparison.provenance, "paper_analyses", "Comparison artifact"
        )
        comparison = comparison.model_copy(update={
            "excluded_papers": list(comparison.excluded_papers) + state.get("excluded_papers", []),
        })
        return {"comparison": comparison}

    def detect_gaps(state: ResearchWorkflowState) -> dict:
        comparison = state["comparison"]
        require_recorded_fingerprint(
            comparison.provenance, "paper_analyses", "Comparison artifact"
        )
        result = agents.gap_detection.detect_gaps(
            comparison, research_topic=state["research_topic"]
        )
        if set(result.paper_ids) != set(comparison.paper_ids):
            raise ValueError("Gap detection changed the comparison paper IDs.")
        if state.get("evaluation_prompt_id"):
            validate_exact_paper_ids(
                result.paper_ids, state.get("evaluation_expected_paper_ids", []),
                "Evaluation gap-detection artifact",
            )
        result.provenance = ArtifactProvenance(
            stage="research_gap_detection",
            input_fingerprints={"comparison": _model_fingerprint(comparison)},
        )
        return {"gap_detection": result}

    def validate_evidence(state: ResearchWorkflowState) -> dict:
        gap_result = state["gap_detection"]
        require_fingerprint(
            gap_result.provenance,
            "comparison",
            _model_fingerprint(state["comparison"]),
            "Gap detection artifact",
        )
        revised_gap_ids = state.get("revised_gap_ids", [])
        prior_validation = state.get("evidence_validation")
        if state.get("revision_applied") and revised_gap_ids and prior_validation is not None:
            revised_set = set(revised_gap_ids)
            focused_gap_result = gap_result.model_copy(update={
                "gaps": [gap for gap in gap_result.gaps if gap.gap_id in revised_set],
            })
            focused_result = agents.evidence_validation.validate(focused_gap_result)
            new_by_id = {item.gap_id: item for item in focused_result.gap_validations}
            if (len(new_by_id) != len(focused_result.gap_validations)
                    or set(new_by_id) != revised_set):
                raise ValueError("Focused evidence revalidation did not return exactly the revised gaps.")
            old_by_id = {item.gap_id: item for item in prior_validation.gap_validations}
            if (len(old_by_id) != len(prior_validation.gap_validations)
                    or set(old_by_id) != {gap.gap_id for gap in gap_result.gaps}):
                raise ValueError("Prior evidence validation does not exactly match the revised gap set.")
            merged = [new_by_id[gap.gap_id] if gap.gap_id in revised_set else old_by_id[gap.gap_id]
                      for gap in gap_result.gaps]
            gap_statuses = [item.status.value for item in merged]
            evidence_statuses = [
                record.status.value for item in merged for record in item.evidence_validations
                if record.status is not None
            ]
            result = focused_result.model_copy(update={
                "gap_validations": merged,
                "validation_counts": {status.value: gap_statuses.count(status.value)
                                      for status in GapValidationStatus},
                "evidence_status_counts": {
                    status.value: evidence_statuses.count(status.value)
                    for status in EvidenceValidationStatus
                },
                "retrieval_failures": list(prior_validation.retrieval_failures)
                                      + list(focused_result.retrieval_failures),
                "semantic_failures": list(prior_validation.semantic_failures)
                                     + list(focused_result.semantic_failures),
                "caveats": list(dict.fromkeys(prior_validation.caveats + focused_result.caveats)),
            })
        else:
            result = agents.evidence_validation.validate(gap_result)
        if set(result.paper_ids) != set(gap_result.paper_ids):
            raise ValueError("Evidence validation changed the gap-detection paper IDs.")
        if state.get("evaluation_prompt_id"):
            validate_exact_paper_ids(
                result.paper_ids, state.get("evaluation_expected_paper_ids", []),
                "Evaluation evidence-validation artifact",
            )
        result.provenance = ArtifactProvenance(
            stage="evidence_validation",
            input_fingerprints={"gap_detection": _model_fingerprint(gap_result)},
        )
        return {"evidence_validation": result}

    def review_with_critic(state: ResearchWorkflowState) -> dict:
        comparison = state["comparison"]
        gap_result = state["gap_detection"]
        validation = state["evidence_validation"]
        require_recorded_fingerprint(
            comparison.provenance, "paper_analyses", "Comparison artifact"
        )
        require_fingerprint(
            gap_result.provenance, "comparison", _model_fingerprint(comparison),
            "Gap detection artifact",
        )
        require_fingerprint(
            validation.provenance, "gap_detection", _model_fingerprint(gap_result),
            "Evidence validation artifact",
        )
        fingerprints = _critic_fingerprints(state)
        result = agents.critic.critique(
            comparison,
            gap_result,
            validation,
            input_fingerprints=fingerprints,
            previous_result=state.get("critic_result"),
        )
        return {"critic_result": result, "critic_input_fingerprints": fingerprints}

    def revise_gaps(state: ResearchWorkflowState) -> dict:
        result, changed_gap_ids = _apply_revision_suggestions(
            state["gap_detection"], state["comparison"], state["critic_result"]
        )
        if changed_gap_ids:
            result.provenance = ArtifactProvenance(
                stage="research_gap_detection",
                input_fingerprints={"comparison": _model_fingerprint(state["comparison"])},
            )
        return {
            "gap_detection": result,
            "revision_count": state.get("revision_count", 0) + 1,
            "revision_applied": bool(changed_gap_ids),
            "revised_gap_ids": changed_gap_ids,
        }

    def generate_questions(state: ResearchWorkflowState) -> dict:
        result = agents.research_questions.generate_questions(
            state["critic_result"],
            state["gap_detection"],
            state["evidence_validation"],
            critic_input_fingerprints=state["critic_input_fingerprints"],
        )
        complete = result.generation_status.value == "COMPLETE"
        return {
            "research_questions": result,
            "workflow_status": "COMPLETE" if complete else "STOPPED",
            "stop_reason": None if complete else (
                "Research question/hypothesis generation was incomplete."
            ),
        }

    def route_after_retrieval(state: ResearchWorkflowState) -> str:
        return "analyze" if len(state.get("papers", [])) >= state.get("minimum_papers", 2) else "stop_insufficient_retrieval"

    def route_after_analysis(state: ResearchWorkflowState) -> str:
        count = len(state.get("paper_analyses", []))
        if count < state.get("minimum_papers", 2):
            return "stop_insufficient_analysis"
        return "compare"

    def route_after_gaps(state: ResearchWorkflowState) -> str:
        return "validate" if state["gap_detection"].gaps else "stop_no_gaps"

    def route_after_validation(state: ResearchWorkflowState) -> str:
        return "critic" if _evidence_validation_complete(
            state["gap_detection"], state["evidence_validation"]
        ) else "stop_unsupported_evidence"

    def route_after_critic(state: ResearchWorkflowState) -> str:
        critic = state["critic_result"]
        gap_ids = [gap.gap_id for gap in state["gap_detection"].gaps]
        review_ids = [review.gap_id for review in critic.gap_reviews]
        if (critic.review_status is not CriticReviewStatus.COMPLETE
                or critic.errors
                or critic.input_fingerprints != _critic_fingerprints(state)
                or len(review_ids) != len(set(review_ids))
                or set(review_ids) != set(gap_ids)
                or any(review.review_status is not CriticReviewStatus.COMPLETE
                       for review in critic.gap_reviews)):
            return "stop_incomplete_critic"
        if any(review.decision is CriticDecision.REVISE for review in critic.gap_reviews):
            if state.get("revision_count", 0) < state.get("max_revisions", 2):
                return "revise"
            return "stop_revision_limit"
        accepted = {review.gap_id for review in critic.gap_reviews
                    if review.decision is CriticDecision.ACCEPT}
        if not accepted:
            return "stop_rejected"
        # Critic-rejected gaps are terminal exclusions; they are recorded by the
        # question agent and never block questions for accepted gaps. Every
        # accepted gap must still be evidence-eligible, otherwise stop.
        if not _questions_evidence_eligible(
                state["gap_detection"], state["evidence_validation"], accepted):
            return "stop_unsupported_evidence"
        return "questions"

    def route_after_revision(state: ResearchWorkflowState) -> str:
        return "validate" if state.get("revision_applied") else "stop_no_applicable_revision"

    def stop(state: ResearchWorkflowState, reason: str) -> dict:
        return {"workflow_status": "STOPPED", "stop_reason": reason}

    def stop_insufficient_retrieval(state):
        return stop(state, "Insufficient papers were retrieved.")

    def stop_insufficient_analysis(state):
        return stop(state, "Fewer than the required number of papers could be analyzed.")

    def stop_no_gaps(state):
        return stop(state, "No candidate research gaps were produced.")

    def stop_unsupported_evidence(state):
        return stop(state, "Evidence validation was incomplete, or a critic-accepted gap "
                           "has no supported evidence.")

    def stop_incomplete_critic(state):
        return stop(state, "Critic review was incomplete or reported errors.")

    def stop_revision_limit(state):
        return stop(state, "Critic revision limit reached before all remaining decisions were resolved.")

    def stop_rejected(state):
        return stop(state, "Critic rejected all candidate gaps.")

    def stop_no_applicable_revision(state):
        return stop(state, "Critic requested revision, but no safe evidence-grounded suggestion could be applied.")

    def route_start(state: ResearchWorkflowState) -> str:
        if state.get("resume_from_critic"):
            critic = state["critic_result"]
            if (critic.review_status is not CriticReviewStatus.COMPLETE
                    or any(review.review_status is not CriticReviewStatus.COMPLETE
                           for review in critic.gap_reviews)):
                return "critic"
            return route_after_critic(state)
        return "critic" if state.get("resume_from_evidence") else "retrieve"

    graph = StateGraph(ResearchWorkflowState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("analyze", analyze_papers)
    graph.add_node("compare", compare_papers)
    graph.add_node("detect_gaps", detect_gaps)
    graph.add_node("validate", validate_evidence)
    graph.add_node("critic", review_with_critic)
    graph.add_node("revise", revise_gaps)
    graph.add_node("questions", generate_questions)
    for name, node in (
        ("stop_insufficient_retrieval", stop_insufficient_retrieval),
        ("stop_insufficient_analysis", stop_insufficient_analysis),
        ("stop_no_gaps", stop_no_gaps),
        ("stop_unsupported_evidence", stop_unsupported_evidence),
        ("stop_incomplete_critic", stop_incomplete_critic),
        ("stop_revision_limit", stop_revision_limit),
        ("stop_rejected", stop_rejected),
        ("stop_no_applicable_revision", stop_no_applicable_revision),
    ):
        graph.add_node(name, node)

    graph.add_conditional_edges(START, route_start, {
        "retrieve": "retrieve", "critic": "critic",
        "revise": "revise", "questions": "questions",
        "stop_incomplete_critic": "stop_incomplete_critic",
        "stop_revision_limit": "stop_revision_limit",
        "stop_rejected": "stop_rejected",
        "stop_unsupported_evidence": "stop_unsupported_evidence",
    })
    graph.add_conditional_edges("retrieve", route_after_retrieval, {
        "analyze": "analyze", "stop_insufficient_retrieval": "stop_insufficient_retrieval",
    })
    graph.add_conditional_edges("analyze", route_after_analysis, {
        "compare": "compare", "stop_insufficient_analysis": "stop_insufficient_analysis",
    })
    graph.add_edge("compare", "detect_gaps")
    graph.add_conditional_edges("detect_gaps", route_after_gaps, {
        "validate": "validate", "stop_no_gaps": "stop_no_gaps",
    })
    graph.add_conditional_edges("validate", route_after_validation, {
        "critic": "critic", "stop_unsupported_evidence": "stop_unsupported_evidence",
    })
    graph.add_conditional_edges("critic", route_after_critic, {
        "revise": "revise", "questions": "questions",
        "stop_incomplete_critic": "stop_incomplete_critic",
        "stop_unsupported_evidence": "stop_unsupported_evidence",
        "stop_revision_limit": "stop_revision_limit", "stop_rejected": "stop_rejected",
    })
    graph.add_conditional_edges("revise", route_after_revision, {
        "validate": "validate", "stop_no_applicable_revision": "stop_no_applicable_revision",
    })
    graph.add_edge("questions", END)
    for name in (
        "stop_insufficient_retrieval", "stop_insufficient_analysis", "stop_no_gaps",
        "stop_unsupported_evidence", "stop_incomplete_critic", "stop_revision_limit",
        "stop_rejected", "stop_no_applicable_revision",
    ):
        graph.add_edge(name, END)
    return graph.compile()
