"""Regression tests built from the failed Phase 3 live run (data/runs/phase3-two-paper).

That run retrieved 39 candidates but analyzed the first two that happened to
download (MuaLLM and a translation-agent review), built one topical cluster
(MET-1), let the model draw three gaps from the same evidence pair, and the
Critic ACCEPTED GAP-001 although its validation was PARTIALLY_SUPPORTED.

Every paper ID, chunk ID, title and quote below is copied exactly from that
run. All tests are offline: fake clients, fake embedders, no network.
"""
import re
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.agents.critic_agent import (
    CriticAgent,
    CriticDecision,
    CriticResult,
    CriticReviewStatus,
    GapCriticReview,
    GapOriginAssessment,
    _LLMCriticResult,
    _LLMGapReview,
    _LLMGapVerdict,
)
from src.agents.artifact_provenance import ArtifactProvenance
from src.agents.cross_paper_comparison_agent import EvidenceRef
from src.agents.evidence_matrix import EvidenceMatrixComparisonAgent, split_statements
from src.agents.evidence_matrix_models import (
    ClusterMember,
    EvidenceCluster,
    EvidenceMatrix,
    MatrixCell,
)
from src.agents.evidence_validation_agent import (
    DeterministicCheck,
    EvidenceContradiction,
    EvidenceValidation,
    EvidenceValidationAgent,
    EvidenceValidationResult,
    EvidenceValidationStatus as EV,
    GapSupportStatus,
    GapValidation,
    GapValidationStatus,
    JointGapAssessment,
)
from src.agents.gap_acceptance import acceptance_blockers
from src.agents.planner_agent import SearchPlan
from src.agents.planner_retrieval_agent import PlannerRetrievalAgent
from src.agents.quote_match import locate_quote
from src.agents.research_gap_detection_agent import (
    GapCategory,
    GapConfidence,
    GapOrigin,
    LLMGapDetection,
    ResearchGap,
    ResearchGapDetectionAgent,
    ResearchGapDetectionResult,
)
from src.agents.research_question_agent import GenerationStatus, ResearchQuestionAgent
from src.retrieval.models import PaperRecord
from src.retrieval.ranking import (
    DEFAULT_RELEVANCE_THRESHOLD,
    PaperRelevanceRanker,
)
from src.retrieval.relevance_filter import RelevanceFilter
from src.workflow.report import render_report
from tests import test_langgraph_workflow as wf

# ---------------------------------------------------------------------------
# Exact data of the failed run
# ---------------------------------------------------------------------------

TOPIC = ("Multi-agent LLM systems for scientific literature review and automated "
         "research gap discovery")
MUA = "c1e3fe3a8802f593b2c6f899a9528032c8e2078c"          # MuaLLM (#7)
TRA = "be529e144c3de710ab49d465fb34ac80ec3544ac"          # Translation agents (#10)
MUA_CHUNK = f"{MUA}_chunk_1"
TRA_CHUNK = f"{TRA}_chunk_2"
MUA_QUOTE = ("MuaLLM: A Multimodal Large Language Model Agent for Circuit Design Assistance "
             "with Hybrid Contextual Retrieval-Augmented Generation Pravallika Abbineni∗, "
             "Saoud Aldowaish∗, Colin Liechty∗, Soroosh Noorzad∗, Ali "
             "Ghazizadeh†, Morteza Fayazi∗ Abstract—Conducting a comprehensive "
             "literature review is")
TRA_QUOTE = ("It traces their technological evolution and theoretical mapping, with particular "
             "attention to single prompting with retrieval-augmented generation, multi-agent "
             "collaboration, and multi-level memory mechanisms.")
MUA_STATEMENT = ("MuaLLM is an open‑source multimodal Large Language Model agent that "
                 "combines a hybrid Retrieval‑Augmented Generation (RAG) framework with a "
                 "Reason + Act (ReAct) workflow.")
TRA_STATEMENT = ("It reviews recent studies on TAs, focusing on single prompting with "
                 "retrieval‑augmented generation (RAG), multi‑agent collaboration "
                 "(e.g., TransAgents, multi‑role frameworks), and multi‑level memory "
                 "mechanisms.")
# Verbatim excerpt of MuaLLM chunk_1 (stored text uses ASCII hyphens).
MUA_CHUNK_TEXT = (
    MUA_QUOTE + " crucial for advancing circuit design methodologies. In this paper, we "
    "propose MuaLLM, an open-source multimodal Large Language Model (LLM) agent for circuit "
    "design assistance that integrates a hybrid Retrieval-Augmented Generation (RAG) "
    "framework with an adaptive vector database of circuit design research papers. Unlike "
    "conventional LLMs, the MuaLLM agent employs a Reason + Act (ReAct) workflow for "
    "iterative reasoning, goal- setting, and multi-step information retrieval."
)
GAP_TITLES = [
    "Evaluation of Hybrid RAG + ReAct Agents in Multi‑Agent Collaborative Settings",
    "Memory Mechanism Design for Multi‑Modal LLM Agents in Literature Review",
    "Benchmarking Multimodal vs. Text‑Only Agents for Circuit Design Literature Review",
]
GAP_DESCRIPTIONS = [
    "There is limited empirical evidence on how hybrid Retrieval‑Augmented Generation "
    "(RAG) combined with Reason + Act (ReAct) agents performs when integrated into "
    "multi‑agent collaboration frameworks for literature review tasks.",
    "The design and impact of multi‑level memory mechanisms for multimodal LLM agents "
    "used in literature review remain underexplored.",
    "A systematic benchmark comparing multimodal LLM agents (e.g., MuaLLM) with "
    "text‑only agents for reviewing circuit design literature is missing.",
]


def pair_refs():
    return [EvidenceRef(paper_id=MUA, chunk_ids=[MUA_CHUNK]),
            EvidenceRef(paper_id=TRA, chunk_ids=[TRA_CHUNK])]


def met1_matrix():
    members = [
        ClusterMember(paper_id=MUA, chunk_id=MUA_CHUNK, statement=MUA_STATEMENT, quote=MUA_QUOTE),
        ClusterMember(paper_id=TRA, chunk_id=TRA_CHUNK, statement=TRA_STATEMENT, quote=TRA_QUOTE),
    ]
    return EvidenceMatrix(
        paper_ids=[MUA, TRA], dimensions=["methodology"],
        cells=[MatrixCell(paper_id=m.paper_id, dimension="methodology", statement=m.statement,
                          chunk_id=m.chunk_id, quote=m.quote, grounded=True,
                          grounding_similarity=0.86) for m in members],
        clusters=[EvidenceCluster(
            cluster_id="MET-1", dimension="methodology", paper_ids=[MUA, TRA],
            members=members, representative_statement=MUA_STATEMENT, mean_similarity=0.7083,
        )],
        embedding_model="BAAI/bge-small-en-v1.5", grounding_threshold=0.55,
        similarity_threshold=0.7, min_supporting_papers=2,
    )


def met1_comparison():
    comparison = EvidenceMatrixComparisonAgent(embedding_model=object())._to_comparison(met1_matrix())
    comparison.provenance = ArtifactProvenance(
        stage="evidence_matrix", input_fingerprints={"paper_analyses": "a" * 64})
    return comparison


def run_gap(index):
    """The three gaps of the run, all citing the same MET-1 evidence pair."""
    categories = [GapCategory.EMPIRICAL, GapCategory.METHODOLOGICAL,
                  next(c for c in GapCategory if "Dataset" in c.value)]
    return ResearchGap(
        gap_id=f"GAP-00{index + 1}", title=GAP_TITLES[index],
        description=GAP_DESCRIPTIONS[index], gap_category=categories[index],
        reasoning="Inferred across P1 and P2.", supporting_paper_ids=[MUA, TRA],
        supporting_chunk_ids=[MUA_CHUNK, TRA_CHUNK], supporting_evidence=pair_refs(),
        evidence_summary="P1-E1 and P2-E1.", gap_origin=GapOrigin.INFERRED,
        confidence=GapConfidence.MEDIUM if index == 0 else GapConfidence.LOW,
        confidence_reason="Two papers.", research_opportunity="Tentative opportunity.",
    )


def run_gap_detection(count=3):
    gaps = [run_gap(i) for i in range(count)]
    return ResearchGapDetectionResult(
        research_topic=TOPIC, paper_count=2, paper_ids=[MUA, TRA], gaps=gaps,
        gap_category_counts={}, caveats=[],
    )


def joint(status, contradictions=None):
    return JointGapAssessment(
        status=status, complete_gap_assessment="assessed", cross_paper_inference="inferred",
        evidence_summary_accuracy="accurate", research_opportunity="tentative",
        gap_origin="Inferred across papers.", overgeneralization="none", limitations="two papers",
        counterevidence="none", source_evidence=pair_refs(), contradictions=contradictions or [],
    )


def record(paper_id, chunk_id, status, quote=None):
    return EvidenceValidation(
        paper_id=paper_id, chunk_id=chunk_id, status=status, findings="{}",
        supported_contribution=("supported" if quote else None),
        supported_contribution_quote=quote,
    )


def gap_validation(gap_id, status, joint_status, statuses, quotes=(None, None)):
    return GapValidation(
        gap_id=gap_id, status=status, explanation="checked",
        evidence_validations=[record(MUA, MUA_CHUNK, statuses[0], quotes[0]),
                              record(TRA, TRA_CHUNK, statuses[1], quotes[1])],
        joint_assessment=joint(joint_status),
        deterministic_check=DeterministicCheck(passed=True, checks=["2 of 2 cited chunk(s) exist"]),
    )


def gap001_validation():
    """Exact GAP-001 record: PARTIAL gap/joint, both items SUPPORTED, no verified quote."""
    return gap_validation("GAP-001", GapValidationStatus.PARTIALLY_SUPPORTED,
                          GapSupportStatus.PARTIALLY_SUPPORTED, (EV.SUPPORTED, EV.SUPPORTED))


def gap002_validation():
    return gap_validation("GAP-002", GapValidationStatus.UNSUPPORTED,
                          GapSupportStatus.UNSUPPORTED, (EV.UNSUPPORTED, EV.UNSUPPORTED))


def gap003_validation():
    return gap_validation("GAP-003", GapValidationStatus.PARTIALLY_SUPPORTED,
                          GapSupportStatus.PARTIALLY_SUPPORTED, (EV.SUPPORTED, EV.UNSUPPORTED))


def fully_supported_validation(gap_id="GAP-001"):
    """Positive control: FULLY_SUPPORTED, both chunks SUPPORTED with verified quotes."""
    return gap_validation(gap_id, GapValidationStatus.FULLY_SUPPORTED,
                          GapSupportStatus.FULLY_SUPPORTED, (EV.SUPPORTED, EV.SUPPORTED),
                          ("employs a Reason + Act (ReAct) workflow", TRA_QUOTE))


def validation_result(gap_validations):
    return EvidenceValidationResult(
        research_topic=TOPIC, paper_ids=[MUA, TRA], gap_validations=gap_validations,
        validation_counts={}, evidence_status_counts={},
    )


def run_validation():
    return validation_result([gap001_validation(), gap002_validation(), gap003_validation()])


FINGERPRINTS = {"comparison": "a" * 64, "gap_detection": "b" * 64,
                "evidence_validation": "c" * 64}


class VerdictClient:
    """Fake verdict-mode critic model that always says ACCEPT with the shown labels."""

    def __init__(self, decision="ACCEPT"):
        self.decision, self.calls = decision, []
        self.models = SimpleNamespace(generate_content=self.generate)

    def generate(self, *, model, contents, config=None):
        self.calls.append(contents)
        gap_id = re.search(r"(GAP-\d+)", contents).group(1)
        labels = sorted(set(re.findall(r"\bP\d+-E\d+\b", contents)))
        return SimpleNamespace(parsed=_LLMGapVerdict(
            gap_id=gap_id, decision=self.decision, origin_assessment="INFERRED",
            origin_rationale="Cross-paper synthesis.", reasoning="Model reasoning.",
            evidence_labels=labels,
        ))


# ---------------------------------------------------------------------------
# 1. Relevance filter
# ---------------------------------------------------------------------------

def paper(source_id, title, abstract=None):
    return PaperRecord(title=title, authors=[], year=2026, doi=None, abstract=abstract,
                       source="test", source_id=source_id, landing_url=None, pdf_url=None)


NAOH = paper("cdc0249c7056a6325f799cdc29ae163bc4cf833a",
             "Comparative assessment of NaOH and NH4OH in alkaline extraction of lentil and "
             "chickpea sidestream proteins: impacts on protein quality and process sustainability")
BLOWING = paper("https://openalex.org/W4398189926",
                "Development of Reagents - Blowing Agents Based on Local Raw Materials and their "
                "Research",
                "The development and search for effective reagents and optimal technological "
                "conditions is one of the main factors for improving the coal flotation process.")
GAP_RAG = paper("7508356493137d93c8f830e97d3864c58ebc7531",
                "Automating Research Gap Identification Using Retrieval-Augmented Generation and "
                "Chain-of-Thought Prompting",
                "Since the number of scientific publications has increased rapidly, finding "
                "research gaps has become much more difficult and time-consuming.")
ASSISTANT = paper("10.13052/rp-9788743817109a015",
                  "LLM-Driven Research Assistant for Automated Literature\nAnalysis and Research "
                  "Gap Detection")
EVO = paper("d05a5180798bbbdc411aad66e441c053b382ea44",
            "EvoResearch: A Multi-Agent AI Framework for Automated Paper Analysis",
            "The rapid growth of scientific publications has intensified the challenge of "
            "efficiently extracting essential knowledge from large sets of research papers.")
MMAGENT = paper("10.1038/s41598-025-96458-w",
                "MMAgentRec, a personalized multi-modal recommendation agent with large language model")


def test_relevance_filter_no_longer_keeps_papers_that_only_contain_and():
    query = '"LLM" AND "gap detection" AND "research"'
    kept = RelevanceFilter().filter([NAOH, BLOWING, GAP_RAG], query)
    # The old filter kept all three: its only unquoted token was "and".
    assert kept == [GAP_RAG]


def test_relevance_filter_strips_boolean_syntax_and_quotes():
    query = '"agent-based" AND "scientific review" AND "automated"'
    kept = RelevanceFilter().filter([BLOWING, EVO], query)
    assert kept == [EVO]   # "Blowing Agents Based ..." matches only "agent"


# ---------------------------------------------------------------------------
# 2. Ranking against the original topic
# ---------------------------------------------------------------------------

VOCAB = ("research", "gap", "literature", "review", "automated", "multi", "agent", "llm",
         "scientific", "discovery", "analysis", "paper", "publication")


class FakeEmbedder:
    """Deterministic bag-of-words embedder over a fixed vocabulary (no model load)."""

    def __init__(self):
        self.query_calls, self.document_calls = [], []

    @staticmethod
    def _vector(text):
        tokens = re.findall(r"[a-z]+", text.lower())
        return [float(sum(1 for t in tokens if t.startswith(term))) for term in VOCAB]

    def embed_query(self, text):
        self.query_calls.append(text)
        return self._vector(text)

    def embed_documents(self, texts):
        self.document_calls.append(list(texts))
        return [self._vector(text) for text in texts]


RETRIEVAL_ORDER = [MMAGENT, NAOH, ASSISTANT, GAP_RAG, BLOWING, EVO]


def test_ranking_puts_on_topic_papers_above_unrelated_ones_deterministically():
    ranker = PaperRelevanceRanker(embedding_model=FakeEmbedder())
    first = ranker.rank(TOPIC, RETRIEVAL_ORDER)
    second = ranker.rank(TOPIC, list(RETRIEVAL_ORDER))
    order = [item.paper.source_id for item in first]
    assert order == [item.paper.source_id for item in second]
    on_topic = {GAP_RAG.source_id, ASSISTANT.source_id, EVO.source_id}
    unrelated = {MMAGENT.source_id, NAOH.source_id, BLOWING.source_id}
    worst_on_topic = max(order.index(pid) for pid in on_topic)
    best_unrelated = min(order.index(pid) for pid in unrelated)
    assert worst_on_topic < best_unrelated
    assert dict((i.paper.source_id, i.score) for i in first)[NAOH.source_id] == 0.0


def test_ranking_ties_keep_retrieval_order():
    twin_a, twin_b = paper("a", "Same title"), paper("b", "Same title")
    ranked = PaperRelevanceRanker(embedding_model=FakeEmbedder()).rank(TOPIC, [twin_a, twin_b])
    assert [item.paper.source_id for item in ranked] == ["a", "b"]


def test_planner_retrieval_ranks_against_the_original_topic_not_sub_queries():
    plan = SearchPlan(original_topic=TOPIC, research_domain="AI",
                      search_queries=['"LLM" AND "gap detection"', '"multi-agent"'],
                      target_modalities=[], evaluation_focus=[])
    planner = Mock()
    planner.create_search_plan.return_value = plan
    retriever = Mock()
    retriever.search.side_effect = [[MMAGENT, NAOH], [ASSISTANT, GAP_RAG, BLOWING, EVO]]
    embedder = FakeEmbedder()
    agent = PlannerRetrievalAgent(planner=planner, retriever=retriever,
                                  ranker=PaperRelevanceRanker(embedding_model=embedder))

    returned_plan, papers, scores = agent.search(TOPIC, max_results_per_source=5,
                                                 with_relevance=True)
    assert returned_plan is plan
    assert embedder.query_calls == [TOPIC]
    assert scores == sorted(scores, reverse=True)
    assert papers[0].source_id in {GAP_RAG.source_id, ASSISTANT.source_id, EVO.source_id}
    # Backward-compatible default: (plan, papers), same ranked order.
    retriever.search.side_effect = [[MMAGENT, NAOH], [ASSISTANT, GAP_RAG, BLOWING, EVO]]
    assert agent.search(TOPIC)[1] == papers
    # PaperRecord carries no score: saved paper records are unchanged.
    assert not hasattr(papers[0], "relevance")


# ---------------------------------------------------------------------------
# 3. Deterministic selection with max_papers=2 and honest stopping
# ---------------------------------------------------------------------------

def run_retrieval_candidates():
    """Retrieval-order candidates of the run with their calibrated relevance scores."""
    candidates = [
        (MMAGENT, 0.692), (paper(MUA, "MuaLLM"), 0.762), (paper(TRA, "Translation agents"), 0.757),
        (ASSISTANT, 0.885), (GAP_RAG, 0.814), (NAOH, 0.521), (BLOWING, 0.554), (EVO, 0.860),
    ]
    return [p for p, _ in candidates], [s for _, s in candidates]


def selection_agents(papers, scores, failing):
    agents = wf.workflow_agents()
    agents.retrieval.search.return_value = (wf.search_plan(), papers, scores)

    def resolve(paper_record, paper_id, paths):
        if paper_id in failing:
            raise failing[paper_id]
        return "paper.pdf"

    agents.paper_analysis._resolve_pdf_path.side_effect = resolve
    return agents


def test_max_papers_selects_top_ranked_accessible_papers_and_labels_every_outcome():
    papers, scores = run_retrieval_candidates()
    failing = {ASSISTANT.source_id: ValueError("No PDF path or PDF URL available.")}
    agents = selection_agents(papers, scores, failing)
    result = wf.invoke(agents, max_papers=2, pdf_paths={})

    outcomes = {entry["paper_id"]: entry["outcome"] for entry in result["paper_selection"]}
    analyzed = [call.kwargs["paper"].source_id
                for call in agents.paper_analysis.analyze_paper.call_args_list]
    # Ranked: #12 (no PDF), #39, #16 ... not the first two downloadable in retrieval order.
    assert analyzed == [EVO.source_id, GAP_RAG.source_id]
    assert outcomes == {
        ASSISTANT.source_id: "failed",
        EVO.source_id: "analyzed",
        GAP_RAG.source_id: "analyzed",
        MUA: "not_attempted_max_papers",
        TRA: "not_attempted_max_papers",
        MMAGENT.source_id: "not_attempted_below_relevance",
        BLOWING.source_id: "not_attempted_below_relevance",
        NAOH.source_id: "not_attempted_below_relevance",
    }
    # Only real failures are excluded papers; not-attempted papers are not failures.
    assert [item.paper_id for item in result["excluded_papers"]] == [ASSISTANT.source_id]
    assert [entry["rank"] for entry in result["paper_selection"]] == list(range(1, 9))
    assert result["relevance_threshold"] == DEFAULT_RELEVANCE_THRESHOLD
    assert result["paper_relevance"][EVO.source_id] == 0.860

    report = render_report(result)
    assert "not attempted (max_papers reached)" in report
    assert "not attempted (below relevance threshold)" in report
    assert "failed (download or analysis)" in report
    assert "0.860" in report


def test_no_sufficiently_relevant_accessible_papers_stops_honestly():
    papers, scores = [NAOH, BLOWING, MMAGENT], [0.521, 0.554, 0.692]
    agents = selection_agents(papers, scores, {})
    result = wf.invoke(agents, max_papers=2, pdf_paths={})

    assert result["workflow_status"] == "STOPPED"
    assert "Insufficient evidence: 0 of 3 retrieved papers met the relevance threshold" \
        in result["stop_reason"]
    agents.paper_analysis.analyze_paper.assert_not_called()
    agents.comparison.compare_many.assert_not_called()
    assert {e["outcome"] for e in result["paper_selection"]} == {"not_attempted_below_relevance"}


def test_relevant_but_inaccessible_papers_also_stop_honestly():
    papers, scores = [ASSISTANT, NAOH], [0.885, 0.521]
    agents = selection_agents(papers, scores, {
        ASSISTANT.source_id: ValueError("No PDF path or PDF URL available.")})
    result = wf.invoke(agents, max_papers=2, pdf_paths={})
    assert result["workflow_status"] == "STOPPED"
    assert "1 of 2 retrieved papers met the relevance threshold" in result["stop_reason"]
    assert "0 of them could be analyzed" in result["stop_reason"]


def test_state_without_relevance_keeps_legacy_selection():
    """Checkpoints written before ranking (no paper_relevance) resume unchanged."""
    agents = wf.workflow_agents()   # retrieval returns (plan, papers): no scores
    result = wf.invoke(agents)
    assert "paper_relevance" not in result
    assert [e["outcome"] for e in result["paper_selection"]] == ["analyzed", "analyzed"]


# ---------------------------------------------------------------------------
# 4/5. One evidence set may ground only one gap
# ---------------------------------------------------------------------------

class GapClient:
    def __init__(self, gaps):
        self.gaps, self.contents = gaps, []
        self.models = self

    def generate_content(self, *, model, contents, config=None):
        self.contents.append(contents)
        return SimpleNamespace(parsed=LLMGapDetection(gaps=self.gaps, caveats=[]))


def llm_gap(index, evidence):
    categories = ["Empirical Gap", "Methodological Gap", "Dataset / Evaluation Gap"]
    return {"title": GAP_TITLES[index], "description": GAP_DESCRIPTIONS[index],
            "gap_category": categories[index], "reasoning": "r", "evidence": evidence,
            "evidence_summary": "s", "gap_origin": "inferred_cross_paper",
            "confidence": "medium" if index == 0 else "low", "confidence_reason": "c",
            "research_opportunity": "o"}


PAIR = [{"paper_label": "P1", "evidence_labels": ["P1-E1"]},
        {"paper_label": "P2", "evidence_labels": ["P2-E1"]}]


def test_three_gaps_from_the_single_met1_pair_keep_only_one():
    client = GapClient([llm_gap(i, PAIR) for i in range(3)])
    result = ResearchGapDetectionAgent(client=client).detect_gaps(met1_comparison(), TOPIC)

    assert "Return at most 1 candidate research gaps" in client.contents[0]
    assert [gap.title for gap in result.gaps] == [GAP_TITLES[0]]
    gap = result.gaps[0]
    assert gap.gap_id == "GAP-001"
    assert gap.supporting_evidence == pair_refs()
    assert gap.supporting_chunk_ids == [MUA_CHUNK, TRA_CHUNK]
    reasons = [item.reason for item in result.excluded_candidates]
    assert len(reasons) == 2
    assert all(reason.startswith("Reuses the evidence of GAP-001") for reason in reasons)


def test_contained_evidence_sets_are_also_rejected():
    """With two clusters, a gap whose evidence is a subset or superset of a kept
    gap's evidence is still rejected; only genuinely different evidence survives."""
    matrix = met1_matrix()
    second_mua, second_tra = f"{MUA}_chunk_2", f"{TRA}_chunk_9"
    matrix.clusters.append(EvidenceCluster(
        cluster_id="RES-1", dimension="methodology", paper_ids=[MUA, TRA],
        members=[ClusterMember(paper_id=MUA, chunk_id=second_mua, statement="s1", quote="q1"),
                 ClusterMember(paper_id=TRA, chunk_id=second_tra, statement="s2", quote="q2")],
        representative_statement="s1", mean_similarity=0.75,
    ))
    comparison = EvidenceMatrixComparisonAgent(embedding_model=object())._to_comparison(matrix)
    comparison.provenance = ArtifactProvenance(
        stage="evidence_matrix", input_fingerprints={"paper_analyses": "a" * 64})
    superset = [{"paper_label": "P1", "evidence_labels": ["P1-E1", "P1-E2"]},
                {"paper_label": "P2", "evidence_labels": ["P2-E1"]}]
    other = [{"paper_label": "P1", "evidence_labels": ["P1-E2"]},
             {"paper_label": "P2", "evidence_labels": ["P2-E2"]}]
    client = GapClient([llm_gap(0, PAIR), llm_gap(1, superset), llm_gap(2, other)])
    result = ResearchGapDetectionAgent(client=client).detect_gaps(comparison, TOPIC)

    assert [gap.title for gap in result.gaps] == [GAP_TITLES[0], GAP_TITLES[2]]
    assert [gap.gap_id for gap in result.gaps] == ["GAP-001", "GAP-002"]
    assert result.excluded_candidates[0].reason.startswith("Reuses the evidence of GAP-001")


# ---------------------------------------------------------------------------
# 6/7. Strict acceptance rule
# ---------------------------------------------------------------------------

def test_gap001_record_is_not_accept_eligible():
    blockers = acceptance_blockers(gap001_validation(), run_gap(0))
    text = " ".join(blockers)
    assert "gap-level validation is PARTIALLY_SUPPORTED" in text
    assert "joint assessment is PARTIALLY_SUPPORTED" in text
    assert f"{MUA}/{MUA_CHUNK} has no verified verbatim quote" in text
    assert f"{TRA}/{TRA_CHUNK} has no verified verbatim quote" in text


@pytest.mark.parametrize("mutate, expected", [
    (lambda v: v.evidence_validations[1].__setattr__("status", EV.PARTIALLY_SUPPORTED),
     "is PARTIALLY_SUPPORTED, not SUPPORTED"),
    (lambda v: v.evidence_validations[0].__setattr__("supported_contribution_quote", None),
     "no verified verbatim quote"),
    (lambda v: v.evidence_validations[1].__setattr__("paper_id", MUA),
     "does not exactly match"),
    (lambda v: v.contradictions.append(EvidenceContradiction(
        proposition_a="a", source_a=pair_refs()[0], supporting_quote_a="q",
        proposition_b="b", source_b=pair_refs()[1], supporting_quote_b="q",
        explanation="conflict", unresolved=True)), "unresolved contradiction"),
    (lambda v: v.__setattr__("deterministic_check", DeterministicCheck(passed=False)),
     "deterministic evidence checks failed"),
])
def test_each_requirement_of_the_acceptance_rule_blocks(mutate, expected):
    validation = fully_supported_validation()
    assert acceptance_blockers(validation, run_gap(0)) == []
    mutate(validation)
    assert any(expected in item for item in acceptance_blockers(validation, run_gap(0)))


def test_supported_evidence_from_one_paper_is_not_enough():
    validation = fully_supported_validation()
    validation.evidence_validations[1].paper_id = MUA
    validation.evidence_validations[1].chunk_id = f"{MUA}_chunk_2"
    blockers = acceptance_blockers(validation)
    assert any("from 1 distinct paper(s)" in item for item in blockers)


# ---------------------------------------------------------------------------
# 6. Critic replay of the failed run (verdict mode)
# ---------------------------------------------------------------------------

def critique(validations, client, gaps=None, **kwargs):
    gap_detection = gaps or run_gap_detection(len(validations))
    return CriticAgent(client=client, max_retries=1).critique(
        met1_comparison(), gap_detection, validation_result(validations),
        input_fingerprints=FINGERPRINTS, **kwargs)


def test_gap001_replay_partial_gap_is_rejected_without_a_model_call():
    client = VerdictClient("ACCEPT")
    result = critique([gap001_validation()], client)

    review = result.gap_reviews[0]
    assert client.calls == []
    assert review.decision is CriticDecision.REJECT
    assert review.review_status is CriticReviewStatus.COMPLETE
    assert result.review_status is CriticReviewStatus.COMPLETE
    assert any("Deterministic REJECT" in c for c in review.caveats)
    assert any("PARTIALLY_SUPPORTED" in c for c in review.caveats)
    assert review.cited_evidence == pair_refs()


def test_full_run_replay_gives_no_accept_and_no_model_call():
    client = VerdictClient("ACCEPT")
    result = critique([gap001_validation(), gap002_validation(), gap003_validation()], client)
    assert [r.decision for r in result.gap_reviews] == [CriticDecision.REJECT] * 3
    assert client.calls == []
    # GAP-002 keeps its UNSUPPORTED path; GAP-003 is rejected by the strict rule.
    assert "Deterministic REJECT: evidence validation was UNSUPPORTED." in result.gap_reviews[1].caveats
    assert any(f"{TRA}/{TRA_CHUNK} is UNSUPPORTED" in c for c in result.gap_reviews[2].caveats)


def test_fully_supported_gap_with_one_partial_chunk_is_rejected():
    validation = fully_supported_validation()
    validation.evidence_validations[1].status = EV.PARTIALLY_SUPPORTED
    client = VerdictClient("ACCEPT")
    result = critique([validation], client)
    assert result.gap_reviews[0].decision is CriticDecision.REJECT
    assert client.calls == []


def test_positive_control_fully_supported_two_paper_evidence_permits_accept():
    client = VerdictClient("ACCEPT")
    validation = fully_supported_validation()
    before = validation.model_dump(mode="json")
    result = critique([validation], client)

    review = result.gap_reviews[0]
    assert len(client.calls) == 1
    assert review.decision is CriticDecision.ACCEPT
    assert review.review_status is CriticReviewStatus.COMPLETE
    assert {ref.paper_id for ref in review.cited_evidence} == {MUA, TRA}
    # Provenance: inputs untouched, fingerprints passed through exactly.
    assert validation.model_dump(mode="json") == before
    assert result.input_fingerprints == FINGERPRINTS


def test_positive_control_model_reject_is_respected():
    client = VerdictClient("REJECT")
    result = critique([fully_supported_validation()], client)
    assert result.gap_reviews[0].decision is CriticDecision.REJECT


# ---------------------------------------------------------------------------
# 8. Cached and legacy-mode ACCEPTs are guarded too
# ---------------------------------------------------------------------------

def test_cached_accept_for_a_partial_gap_is_downgraded():
    previous = CriticResult(
        research_topic=TOPIC, review_status=CriticReviewStatus.COMPLETE,
        gap_reviews=[GapCriticReview(
            gap_id="GAP-001", decision=CriticDecision.ACCEPT,
            review_status=CriticReviewStatus.COMPLETE,
            origin_assessment=GapOriginAssessment.INFERRED,
            origin_rationale="old", reasoning="Old ACCEPT saved before the strict rule.",
            cited_evidence=pair_refs(),
        )],
        input_fingerprints=FINGERPRINTS,
    )
    client = VerdictClient("ACCEPT")
    result = critique([gap001_validation()], client, previous_result=previous)

    review = result.gap_reviews[0]
    assert client.calls == []
    assert review.decision is CriticDecision.REJECT
    assert any("Reused a previously successful review" in c for c in review.caveats)
    assert any("ACCEPT overridden to REJECT" in c for c in review.caveats)
    assert review.cited_evidence == pair_refs()


@pytest.mark.parametrize("mode", ["single", "auto"])
def test_legacy_mode_accept_for_a_partial_gap_becomes_revise(mode):
    parsed = _LLMCriticResult(gap_reviews=[_LLMGapReview(
        gap_id="GAP-001", decision=CriticDecision.ACCEPT, origin_assessment="INFERRED",
        origin_rationale="r", reasoning="Legacy ACCEPT.",
    )])
    client = SimpleNamespace(calls=[])
    client.models = SimpleNamespace(generate_content=lambda **kw: (
        client.calls.append(kw), SimpleNamespace(parsed=parsed))[1])
    result = CriticAgent(client=client, max_retries=1, review_mode=mode).critique(
        met1_comparison(), run_gap_detection(1), validation_result([gap001_validation()]),
        input_fingerprints=FINGERPRINTS)

    review = result.gap_reviews[0]
    assert client.calls                       # the legacy model was asked ...
    assert review.decision is CriticDecision.REVISE   # ... but its ACCEPT does not stand
    assert any("ACCEPT overridden to REVISE" in c for c in review.caveats)


# ---------------------------------------------------------------------------
# 9/10. Workflow and question agent second-line enforcement
# ---------------------------------------------------------------------------

def replay_agents(critic_decisions):
    agents = wf.workflow_agents(paper_ids=[MUA, TRA])
    agents.retrieval.search.return_value = (
        wf.search_plan(), [paper(MUA, "MuaLLM"), paper(TRA, "Translation agents")])
    agents.comparison.compare_many.return_value = met1_comparison()
    agents.gap_detection.detect_gaps.return_value = run_gap_detection()
    agents.evidence_validation.validate.return_value = run_validation()

    def injected_critic(*_args, **kwargs):
        return CriticResult(
            research_topic=TOPIC, review_status=CriticReviewStatus.COMPLETE,
            gap_reviews=[GapCriticReview(
                gap_id=f"GAP-00{n}", decision=decision,
                review_status=CriticReviewStatus.COMPLETE,
                origin_assessment=GapOriginAssessment.INFERRED,
                origin_rationale="r", reasoning="Injected decision.",
            ) for n, decision in enumerate(critic_decisions, start=1)],
            input_fingerprints=kwargs["input_fingerprints"],
        )

    agents.critic.critique.side_effect = injected_critic
    return agents


def test_workflow_stops_when_a_partial_gap_is_accepted():
    # Exactly the run's decisions: ACCEPT, REJECT, REJECT.
    agents = replay_agents([CriticDecision.ACCEPT, CriticDecision.REJECT, CriticDecision.REJECT])
    result = wf.invoke(agents, pdf_paths={MUA: "a.pdf", TRA: "b.pdf"})

    assert result["workflow_status"] == "STOPPED"
    assert "strict acceptance rule" in result["stop_reason"]
    agents.research_questions.generate_questions.assert_not_called()
    assert "research_questions" not in result
    # Exact IDs preserved through the stop.
    gap = result["gap_detection"].gaps[0]
    assert gap.supporting_chunk_ids == [MUA_CHUNK, TRA_CHUNK]


def test_question_agent_skips_partial_accept_without_model_call():
    client = SimpleNamespace(calls=[])
    client.models = SimpleNamespace(generate_content=lambda **kw: client.calls.append(kw))
    critic_result = CriticResult(
        research_topic=TOPIC, review_status=CriticReviewStatus.COMPLETE,
        gap_reviews=[GapCriticReview(
            gap_id=f"GAP-00{n}", decision=decision, review_status=CriticReviewStatus.COMPLETE,
            origin_assessment=GapOriginAssessment.INFERRED, origin_rationale="r", reasoning="r",
        ) for n, decision in enumerate(
            [CriticDecision.ACCEPT, CriticDecision.REJECT, CriticDecision.REJECT], start=1)],
    )
    result = ResearchQuestionAgent(client=client, max_retries=1).generate_questions(
        critic_result, run_gap_detection(), run_validation())

    assert client.calls == []
    assert result.gap_results == []
    assert result.generation_status is GenerationStatus.INCOMPLETE
    assert [item.gap_id for item in result.skipped_gaps] == ["GAP-001"]
    assert "strict acceptance rule" in result.skipped_gaps[0].reason
    assert [item.gap_id for item in result.excluded_gaps] == ["GAP-002", "GAP-003"]


# ---------------------------------------------------------------------------
# 11. Quote verification tolerant of typographic variants, storing exact text
# ---------------------------------------------------------------------------

def test_non_breaking_hyphen_quote_is_located_and_stored_as_exact_chunk_text():
    model_quote = "integrates a hybrid Retrieval‑Augmented Generation (RAG) framework"
    exact = locate_quote(model_quote, MUA_CHUNK_TEXT)
    assert exact == "integrates a hybrid Retrieval-Augmented Generation (RAG) framework"
    assert exact in MUA_CHUNK_TEXT


@pytest.mark.parametrize("model_quote, stored", [
    ("employs a Reason + Act (ReAct) workflow", "employs a Reason + Act (ReAct) workflow"),
    ("goal-setting, and multi–step information retrieval",
     "goal- setting, and multi-step information retrieval"),
    ("an  open-source   multimodal", "an open-source multimodal"),
])
def test_dash_and_whitespace_variants_are_located(model_quote, stored):
    assert locate_quote(model_quote, MUA_CHUNK_TEXT) == stored


def test_curly_quotes_are_located():
    text = 'The authors call it "translation-reflection-improvement" loop.'
    assert locate_quote("“translation‑reflection‑improvement” loop",
                        text) == '"translation-reflection-improvement" loop'


@pytest.mark.parametrize("model_quote", [
    "integrates a hybrid Retrieval-Augmented Generation (RAG) pipeline",   # different word
    "MuaLLM employs ReAct",                                                  # paraphrase
    "",
    None,
])
def test_non_verbatim_quotes_still_fail(model_quote):
    assert locate_quote(model_quote, MUA_CHUNK_TEXT) is None


def test_validation_keeps_the_exact_chunk_substring_not_the_model_text():
    assessment = SimpleNamespace(
        supported_contribution="MuaLLM integrates hybrid RAG.",
        supported_contribution_quote="integrates a hybrid Retrieval‑Augmented Generation (RAG) framework",
    )
    flags = EvidenceValidationAgent._verified_contribution(assessment, MUA_CHUNK_TEXT)
    assert flags["supported_contribution_quote"] == (
        "integrates a hybrid Retrieval-Augmented Generation (RAG) framework")
    assert flags["supported_contribution"] == "MuaLLM integrates hybrid RAG."
    failed = EvidenceValidationAgent._verified_contribution(
        SimpleNamespace(supported_contribution="x", supported_contribution_quote="not in chunk"),
        MUA_CHUNK_TEXT)
    assert failed == {"supported_contribution": None, "supported_contribution_quote": None}


# ---------------------------------------------------------------------------
# 12. Evidence-matrix placeholders and cluster wording
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("placeholder", [
    "The evidence does not provide explicit limitations, weaknesses, or unresolved issues "
    "reported by the authors.",
    "No future research directions are explicitly mentioned or supported by the provided "
    "evidence.",
    "No specific datasets, experimental data, benchmarks, or evaluation resources are "
    "reported in the provided evidence.",
])
def test_run_placeholder_statements_are_not_matrix_findings(placeholder):
    assert split_statements(placeholder, 4) == []


@pytest.mark.parametrize("finding", [
    "On RAG‑250, MuaLLM achieves 90.1 % recall, indicating strong multimodal retrieval "
    "and citation accuracy.",
    "Evaluation agents still miss deep cross‑cultural metaphors and may exhibit evaluative bias.",
    "Future research should develop robust evaluation mechanisms.",
])
def test_real_findings_are_kept(finding):
    assert split_statements(finding, 4) == [finding]


def test_cluster_wording_says_topical_similarity_not_agreement():
    comparison = met1_comparison()
    description = comparison.recurring_patterns[0].description
    assert "not verified agreement" in description
    assert "pattern stated by" not in description
    assert comparison.recurring_patterns[0].evidence == pair_refs()


def test_selection_state_round_trips_through_checkpoints():
    from src.workflow.checkpoint import deserialize_state, serialize_state

    papers, scores = run_retrieval_candidates()
    agents = selection_agents(papers, scores, {
        ASSISTANT.source_id: ValueError("No PDF path or PDF URL available.")})
    result = wf.invoke(agents, max_papers=2, pdf_paths={})
    restored = deserialize_state(serialize_state(result))
    assert restored["paper_selection"] == result["paper_selection"]
    assert restored["paper_relevance"] == result["paper_relevance"]
    assert restored["relevance_threshold"] == result["relevance_threshold"]
    assert [p.source_id for p in restored["papers"]] == [p.source_id for p in result["papers"]]
