import json
from pathlib import Path
from types import SimpleNamespace
import sys
from unittest.mock import Mock

import pytest

from src.agents.artifact_provenance import ArtifactProvenance
from src.agents.critic_agent import (
    CriticDecision,
    CriticResult,
    CriticReviewStatus,
    GapCriticReview,
    GapOriginAssessment,
    RevisionSuggestion,
)
from src.agents.cross_paper_comparison_agent import (
    DimensionSummary,
    EvidenceRef,
    MultiPaperComparison,
)
from src.agents.evidence_validation_agent import (
    EvidenceValidation,
    EvidenceValidationResult,
    EvidenceValidationStatus,
    GapValidation,
    GapValidationStatus,
)
from src.agents.paper_analysis_agent import PaperAnalysis, PaperAnalysisResult
from src.agents.planner_agent import SearchPlan
from src.agents.research_gap_detection_agent import (
    GapCategory,
    GapConfidence,
    GapOrigin,
    ResearchGap,
    ResearchGapDetectionResult,
)
from src.agents.research_question_agent import GenerationStatus, ResearchQuestionResult
from src.rag.models import RetrievedChunk
from src.retrieval.models import PaperRecord
from src.workflow.langgraph_app import WorkflowAgents, build_research_graph
from src.evaluation.identity import validate_exact_paper_ids


PAPER_IDS = ["paper-1", "paper-2"]
CHUNKS = ["paper-1-chunk", "paper-2-chunk"]


def search_plan():
    return SearchPlan(
        original_topic="topic", research_domain="domain", search_queries=["q"],
        target_modalities=[], evaluation_focus=[],
    )


def paper_records(count=2):
    return [PaperRecord(
        title=f"Paper {index}", authors=[], year=2024, doi=None, abstract=None,
        source="mock", source_id=f"paper-{index}", landing_url=None, pdf_url=None,
    ) for index in range(1, count + 1)]


def analysis(paper_id):
    return PaperAnalysisResult(
        analysis=PaperAnalysis(
            problem="p", methodology="m", dataset="d", results="r",
            limitations="l", future_work="f",
        ),
        evidence=[RetrievedChunk(
            chunk_id=f"{paper_id}-chunk", paper_id=paper_id, text="source text",
        )],
    )


def comparison(paper_ids=PAPER_IDS):
    refs = [EvidenceRef(paper_id=paper_id, chunk_ids=[f"{paper_id}-chunk"])
            for paper_id in paper_ids]
    return MultiPaperComparison(
        paper_ids=paper_ids, paper_count=len(paper_ids),
        dimension_summaries=[DimensionSummary(dimension="method", summary="summary", evidence=refs)],
        recurring_patterns=[], common_limitations=[], conflicting_findings=[],
        methodological_differences=[], underexplored_directions=[], caveats=[],
        provenance=ArtifactProvenance(
            stage="cross_paper_comparison", input_fingerprints={"paper_analyses": "a" * 64},
        ),
    )


def gap_result(paper_ids=PAPER_IDS):
    refs = [EvidenceRef(paper_id=paper_id, chunk_ids=[f"{paper_id}-chunk"])
            for paper_id in paper_ids]
    gap = ResearchGap(
        gap_id="GAP-001", title="Candidate", description="Initial description",
        gap_category=GapCategory.METHODOLOGICAL, reasoning="reason",
        supporting_paper_ids=paper_ids,
        supporting_chunk_ids=[f"{paper_id}-chunk" for paper_id in paper_ids],
        supporting_evidence=refs, evidence_summary="summary",
        gap_origin=GapOrigin.INFERRED, confidence=GapConfidence.MEDIUM,
        confidence_reason="reason", research_opportunity="opportunity",
    )
    return ResearchGapDetectionResult(
        research_topic="topic", paper_count=len(paper_ids), paper_ids=paper_ids, gaps=[gap],
        gap_category_counts={GapCategory.METHODOLOGICAL.value: 1}, caveats=[],
    )


def validation(status=GapValidationStatus.FULLY_SUPPORTED, paper_ids=PAPER_IDS):
    evidence = [EvidenceValidation(
        paper_id=paper_id, chunk_id=f"{paper_id}-chunk",
        status=(EvidenceValidationStatus.SUPPORTED if status is GapValidationStatus.FULLY_SUPPORTED
                else EvidenceValidationStatus.UNSUPPORTED),
        findings="validated",
    ) for paper_id in paper_ids]
    return EvidenceValidationResult(
        research_topic="topic", paper_ids=paper_ids,
        gap_validations=[GapValidation(
            gap_id="GAP-001", status=status, explanation="checked",
            evidence_validations=evidence,
        )],
        validation_counts={status.value: 1},
        evidence_status_counts={item.status.value: 1 for item in evidence},
    )


def critic_result(decision, fingerprints=None, with_revision=False):
    review = GapCriticReview(
        gap_id="GAP-001", decision=decision, review_status=CriticReviewStatus.COMPLETE,
        origin_assessment=GapOriginAssessment.INFERRED, origin_rationale="rationale",
        reasoning="reasoning",
        revision_suggestions=([
            RevisionSuggestion(
                field="description", current_issue="unclear",
                proposed_text="Revised description",
                source_evidence=[
                    EvidenceRef(paper_id=PAPER_IDS[0], chunk_ids=[CHUNKS[0]])
                ],
            )
        ] if with_revision else []),
    )
    return CriticResult(
        research_topic="topic", review_status=CriticReviewStatus.COMPLETE,
        gap_reviews=[review], input_fingerprints=fingerprints or {},
    )


def workflow_agents(papers=None, paper_ids=PAPER_IDS,
                    evidence_status=GapValidationStatus.FULLY_SUPPORTED,
                    critic_decisions=None):
    retrieval = Mock()
    retrieval.search.return_value = (search_plan(), paper_records(len(paper_ids)) if papers is None else papers)

    analysis_pipeline = Mock()
    analysis_pipeline._get_paper_id.side_effect = lambda paper: paper.source_id
    analysis_pipeline._resolve_pdf_path.side_effect = lambda paper, paper_id, paths: paths[paper_id]
    analysis_pipeline.analyze_paper.side_effect = lambda paper, **kwargs: analysis(paper.source_id)

    compare_agent = Mock()
    compare_agent.compare_many.return_value = comparison(paper_ids)

    gap_agent = Mock()
    gap_agent.detect_gaps.return_value = gap_result(paper_ids)

    validation_agent = Mock()
    validation_agent.validate.return_value = validation(evidence_status, paper_ids)

    critic_agent = Mock()
    decisions = iter(critic_decisions or [CriticDecision.ACCEPT])

    def critique(*args, **kwargs):
        decision = next(decisions)
        return critic_result(
            decision, kwargs["input_fingerprints"],
            with_revision=decision is CriticDecision.REVISE,
        )

    critic_agent.critique.side_effect = critique
    question_agent = Mock()
    question_agent.generate_questions.return_value = ResearchQuestionResult(
        research_topic="topic", generation_status=GenerationStatus.COMPLETE,
    )
    return WorkflowAgents(
        retrieval, analysis_pipeline, compare_agent, gap_agent, validation_agent,
        critic_agent, question_agent,
    )


def invoke(agents, **overrides):
    state = {
        "research_topic": "topic", "analysis_query": "query",
        "pdf_paths": {paper_id: "paper.pdf" for paper_id in PAPER_IDS},
        "minimum_papers": 2, "max_revisions": 2, "revision_count": 0,
    }
    state.update(overrides)
    return build_research_graph(agents).invoke(state, config={"recursion_limit": 30})


def test_evaluation_agents_use_chroma_separate_from_conventional_rag(monkeypatch, tmp_path):
    import src.workflow.langgraph_app as workflow
    from src.vectorstore.chroma_store import collection_name_for

    created = {}
    vector_store = object()
    embedding_model = SimpleNamespace(model_name="BAAI/bge-small-en-v1.5", dimension=384)
    monkeypatch.setattr(workflow, "LocalEmbeddingModel", lambda: embedding_model)

    def make_store(embedder, persist_directory, collection_name):
        created.update(directory=persist_directory, collection=collection_name, embedder=embedder)
        return vector_store

    monkeypatch.setattr(
        workflow, "ChromaVectorStore",
        SimpleNamespace(for_embedder=make_store),
    )
    pipeline_options = {}
    pipeline = SimpleNamespace(comparison_agent=object())

    def make_pipeline(**kwargs):
        pipeline_options.update(kwargs)
        return pipeline

    monkeypatch.setattr(workflow, "PaperAnalysisPipeline", make_pipeline)
    monkeypatch.setattr(workflow, "ResearchGapDetectionAgent", object)
    validation_options = {}
    monkeypatch.setattr(
        workflow, "EvidenceValidationAgent",
        lambda **kwargs: validation_options.update(kwargs) or object(),
    )
    monkeypatch.setattr(workflow, "CriticAgent", object)
    monkeypatch.setattr(workflow, "ResearchQuestionAgent", object)
    monkeypatch.setattr(
        workflow, "PlannerRetrievalAgent",
        lambda: pytest.fail("Evaluation setup must not construct external paper retrieval."),
    )

    evaluation_dir = tmp_path / "multi-agent-evaluation"
    evaluation_collection = "multi_agent_evaluation"
    agents = workflow.default_workflow_agents(
        evaluation_chroma_dir=evaluation_dir,
        evaluation_chroma_collection=evaluation_collection,
    )

    assert agents.retrieval is None
    assert Path(created["directory"]) == evaluation_dir.resolve()
    assert created["directory"] != str((Path(__file__).resolve().parents[1] / "chroma_db").resolve())
    assert created["collection"] == evaluation_collection
    assert created["collection"] != collection_name_for("BAAI/bge-small-en-v1.5")
    assert pipeline_options["ingestion_pipeline"].vector_store is vector_store
    assert pipeline_options["rag_retriever"].vector_store is vector_store
    assert pipeline_options["ingestion_pipeline"].embedding_model is embedding_model
    assert pipeline_options["rag_retriever"].embedding_model is embedding_model
    assert validation_options["vector_store"] is vector_store


def test_insufficient_retrieval_stops_before_analysis():
    agents = workflow_agents(papers=paper_records(1))
    result = invoke(agents)
    assert result["workflow_status"] == "STOPPED"
    assert "Insufficient papers" in result["stop_reason"]
    agents.paper_analysis.analyze_paper.assert_not_called()


def test_supported_accept_routes_to_existing_question_agent():
    agents = workflow_agents()
    result = invoke(agents)
    assert result["workflow_status"] == "COMPLETE"
    agents.critic.critique.assert_called_once()
    agents.research_questions.generate_questions.assert_called_once()
    agents.retrieval.search.assert_called_once_with("topic", max_results_per_source=5)
    assert agents.paper_analysis.analyze_paper.call_count == 2
    agents.comparison.compare_many.assert_called_once()
    agents.gap_detection.detect_gaps.assert_called_once()
    agents.evidence_validation.validate.assert_called_once()
    assert result["comparison"].paper_ids == PAPER_IDS
    assert result["gap_detection"].provenance.input_fingerprints["comparison"]
    assert result["evidence_validation"].provenance.input_fingerprints["gap_detection"]
    assert result["critic_result"].input_fingerprints == result["critic_input_fingerprints"]


def test_evaluation_prompt_is_the_actual_retrieval_and_analysis_input():
    expected_ids = [f"frozen-paper-{index}" for index in range(1, 7)]
    frozen_papers = [PaperRecord(
        title=f"Frozen Paper {index}", authors=[], year=2024, doi=None,
        abstract=None, source="frozen_evaluation_manifest",
        source_id=paper_id, landing_url=None, pdf_url=None,
    ) for index, paper_id in enumerate(expected_ids, start=1)]
    agents = workflow_agents(papers=frozen_papers, paper_ids=expected_ids)
    prompt = "canonical frozen prompt"
    result = invoke(
        agents, research_topic=prompt, analysis_query=prompt,
        evaluation_prompt_id="PROMPT-001", canonical_prompt_text=prompt,
        evaluation_papers=frozen_papers,
        evaluation_expected_paper_ids=expected_ids,
        pdf_paths={paper_id: "frozen.pdf" for paper_id in expected_ids},
        minimum_papers=6,
    )
    assert result["workflow_status"] == "COMPLETE"
    agents.retrieval.search.assert_not_called()
    assert [paper.source_id for paper in result["papers"]] == expected_ids
    assert result["comparison"].paper_ids == expected_ids
    assert result["gap_detection"].paper_ids == expected_ids
    assert result["evidence_validation"].paper_ids == expected_ids
    assert all(call.kwargs["analysis_query"] == prompt
               for call in agents.paper_analysis.analyze_paper.call_args_list)
    assert agents.paper_analysis.analyze_paper.call_count == 6
    assert result["evaluation_prompt_id"] == "PROMPT-001"
    assert result["canonical_prompt_text"] == prompt


def test_evaluation_workflow_rejects_unbound_prompt_before_retrieval():
    agents = workflow_agents()
    with pytest.raises(ValueError, match="must equal the canonical prompt text"):
        invoke(
            agents, evaluation_prompt_id="PROMPT-001",
            canonical_prompt_text="canonical frozen prompt",
        )
    agents.retrieval.search.assert_not_called()


@pytest.mark.parametrize("observed, expected_message", [
    (PAPER_IDS[:1], "missing"),
    ([*PAPER_IDS, *PAPER_IDS[:1], *[f"other-{i}" for i in range(3, 7)]], "duplicate"),
    ([*PAPER_IDS[:1], *[f"other-{i}" for i in range(2, 7)]], "unexpected"),
])
def test_frozen_evaluation_identity_rejects_missing_duplicate_and_unexpected_papers(
    observed, expected_message,
):
    expected = [f"frozen-{index}" for index in range(6)]
    if expected_message == "missing":
        observed = expected[:-1]
    elif expected_message == "duplicate":
        observed = [*expected[:-1], expected[0]]
    else:
        observed = [*expected[:-1], "unexpected-paper"]
    with pytest.raises(ValueError, match=expected_message):
        validate_exact_paper_ids(observed, expected, "test artifact")


def test_unsupported_evidence_stops_before_critic():
    agents = workflow_agents(evidence_status=GapValidationStatus.PARTIALLY_SUPPORTED)
    result = invoke(agents)
    assert result["workflow_status"] == "STOPPED"
    assert "Evidence validation" in result["stop_reason"]
    agents.critic.critique.assert_not_called()


def test_evidence_validation_errors_stop_even_if_gap_status_is_fully_supported():
    agents = workflow_agents()
    agents.evidence_validation.validate.return_value.semantic_failures = ["failed assessment"]
    result = invoke(agents)
    assert result["workflow_status"] == "STOPPED"
    agents.critic.critique.assert_not_called()


def test_critic_revision_revalidates_and_obeys_maximum_revision_count():
    agents = workflow_agents(critic_decisions=[CriticDecision.REVISE, CriticDecision.REVISE])
    result = invoke(agents, max_revisions=1)
    assert result["workflow_status"] == "STOPPED"
    assert "revision limit" in result["stop_reason"]
    assert result["revision_count"] == 1
    assert result["gap_detection"].gaps[0].description == "Revised description"
    assert agents.evidence_validation.validate.call_count == 2
    assert agents.critic.critique.call_count == 2
    agents.research_questions.generate_questions.assert_not_called()


def test_critic_revision_can_reassess_then_accept():
    agents = workflow_agents(critic_decisions=[CriticDecision.REVISE, CriticDecision.ACCEPT])
    result = invoke(agents, max_revisions=1)
    assert result["workflow_status"] == "COMPLETE"
    assert result["revision_count"] == 1
    assert agents.evidence_validation.validate.call_count == 2
    assert agents.critic.critique.call_count == 2
    agents.research_questions.generate_questions.assert_called_once()


def test_critic_reject_stops_without_question_generation():
    agents = workflow_agents(critic_decisions=[CriticDecision.REJECT])
    result = invoke(agents)
    assert result["workflow_status"] == "STOPPED"
    assert "rejected" in result["stop_reason"]
    agents.research_questions.generate_questions.assert_not_called()


def test_critic_with_missing_input_fingerprints_cannot_route_to_questions():
    agents = workflow_agents()
    agents.critic.critique.side_effect = lambda *args, **kwargs: critic_result(
        CriticDecision.ACCEPT, fingerprints={},
    )
    result = invoke(agents)
    assert result["workflow_status"] == "STOPPED"
    assert "Critic review" in result["stop_reason"]
    agents.research_questions.generate_questions.assert_not_called()


def test_incomplete_question_generation_does_not_mark_graph_complete():
    agents = workflow_agents()
    agents.research_questions.generate_questions.return_value = ResearchQuestionResult(
        research_topic="topic", generation_status=GenerationStatus.INCOMPLETE,
        errors=["mocked incomplete generation"],
    )
    result = invoke(agents)
    assert result["workflow_status"] == "STOPPED"
    assert "generation was incomplete" in result["stop_reason"]


def test_dedicated_runner_invokes_compiled_graph_and_writes_configured_output(
    tmp_path, monkeypatch, capsys,
):
    from experiments import run_langgraph

    agents = workflow_agents()
    monkeypatch.setattr(
        run_langgraph, "build_research_graph",
        lambda: build_research_graph(agents),
    )
    output = tmp_path / "offline-run.json"
    monkeypatch.setattr(sys, "argv", [
        "run_langgraph.py", "--topic", "topic", "--output", str(output),
        "--paper-pdf", "paper-1=first.pdf", "--paper-pdf", "paper-2=second.pdf",
    ])

    assert run_langgraph.main() == 0
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["workflow_status"] == "COMPLETE"
    assert saved["critic_result"]["gap_reviews"][0]["decision"] == "ACCEPT"
    assert "Saved:" in capsys.readouterr().out


def test_evaluation_runner_uses_manifest_papers_and_unique_prompt_artifacts(tmp_path, monkeypatch, capsys):
    from experiments import run_langgraph

    prompt = "canonical evaluation prompt"
    expected_ids = [f"frozen-paper-{index}" for index in range(1, 7)]
    corpus = SimpleNamespace(
        prompts={"PROMPT-002": {"text": prompt}},
        papers=[SimpleNamespace(
            paper_id=paper_id, title=f"Paper {index}", authors=[f"Author {index}"],
            year=2024, external_ids={"doi": None},
            relative_path=f"data/papers/paper-{index}.pdf",
        ) for index, paper_id in enumerate(expected_ids, start=1)],
    )
    monkeypatch.setattr(run_langgraph, "load_frozen_evaluation_corpus", lambda _: corpus)
    agent_options = []
    monkeypatch.setattr(run_langgraph, "default_workflow_agents", lambda **kwargs: agent_options.append(kwargs))
    invocations = []

    class CapturingGraph:
        def invoke(self, initial, config):
            invocations.append((initial, config))
            return {
                "workflow_status": "COMPLETE",
                "evaluation_prompt_id": initial["evaluation_prompt_id"],
                "canonical_prompt_text": initial["canonical_prompt_text"],
                "evaluation_chroma_dir": initial["evaluation_chroma_dir"],
                "evaluation_chroma_collection": initial["evaluation_chroma_collection"],
                "papers": initial["evaluation_papers"],
                "paper_analyses": [
                    {"evidence": [{"paper_id": paper_id}]}
                    for paper_id in initial["evaluation_expected_paper_ids"]
                ],
                "comparison": {"paper_ids": expected_ids},
                "gap_detection": {"paper_ids": expected_ids},
                "evidence_validation": {"paper_ids": expected_ids},
            }

    monkeypatch.setattr(run_langgraph, "build_research_graph", lambda agents: CapturingGraph())
    requested = tmp_path / "artifacts.json"
    evaluation_store = tmp_path / "evaluation-chroma"
    for _ in range(2):
        monkeypatch.setattr(sys, "argv", [
            "run_langgraph.py", "--evaluation-prompt-id", "PROMPT-002",
            "--output", str(requested), "--evaluation-chroma-dir", str(evaluation_store),
            "--evaluation-chroma-collection", "prompt-isolated-evaluation",
        ])
        assert run_langgraph.main() == 0
    saved_paths = [
        Path(line.removeprefix("Saved: "))
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("Saved: ")
    ]
    assert len(saved_paths) == 2
    assert saved_paths[0] != saved_paths[1]
    assert all(path.is_file() for path in saved_paths)
    assert all("PROMPT-002" in path.name for path in saved_paths)
    saved = json.loads(saved_paths[0].read_text(encoding="utf-8"))
    assert not requested.exists()
    assert saved["evaluation_prompt_id"] == "PROMPT-002"
    assert saved["canonical_prompt_text"] == prompt
    initial, _ = invocations[0]
    assert initial["research_topic"] == prompt
    assert initial["analysis_query"] == prompt
    assert [paper.source_id for paper in initial["evaluation_papers"]] == expected_ids
    assert list(initial["pdf_paths"]) == expected_ids
    assert initial["max_papers"] == initial["minimum_papers"] == 6
    assert all(options["evaluation_chroma_dir"] == evaluation_store for options in agent_options)
