"""Retrieval recall regressions from the Phase 3 read-only inspection.

The per-query lexical filter needed two literal content words from EACH
sub-query, so clearly relevant candidates were at risk:

* #30 AIssistant (relevance 0.7829) has no abstract and matched only one word
  of every planner query, so it was dropped whichever query found it;
* #12 LLM-Driven Research Assistant (0.8898) passed only 2 of 5 queries because
  "LLM" did not match "large language model";
* #3 Large Language Model Applied in Multi-agent System (0.7816) passes only one
  query lexically.

Now the planner path applies no per-query filter: candidates are aggregated,
deduplicated, scored against the original topic and excluded only by the
explicit relevance threshold, which is recorded and reported. All metadata is
the saved run's (tests/phase3_candidates.py); everything runs offline.
"""
import re

import pytest

from src.agents.planner_agent import SearchPlan
from src.agents.planner_retrieval_agent import PlannerRetrievalAgent
from src.retrieval.models import PaperRecord
from src.retrieval.multi_source import MultiSourceRetriever
from src.retrieval.ranking import (
    DEFAULT_RELEVANCE_THRESHOLD,
    RANKING_METHOD,
    PaperRelevanceRanker,
    paper_text,
)
from src.retrieval.relevance_filter import RelevanceFilter, content_terms
from src.workflow.report import render_report
from tests import test_langgraph_workflow as wf
from tests.phase3_candidates import QUERIES, RECORDS, RELEVANCE, TOPIC

RELEVANT = (12, 30, 3)
IRRELEVANT = (21, 34, 18, 22, 32)


def record(number, **changes):
    return PaperRecord(**{**RECORDS[number], **changes})


def sid(number):
    return RECORDS[number]["source_id"]


# ---------------------------------------------------------------------------
# Lexical matching: equivalences and title-only records
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("left, right", [
    ("LLM", "large language model"),
    ("LLMs", "Large Language Models"),
    ("multi-agent", "multi agent"),
    ("multi-agent", "multiagent"),
    ("retrieval-augmented generation", "RAG"),
])
def test_acronym_and_phrase_equivalences_match_both_ways(left, right):
    assert content_terms(left) & content_terms(right)


def test_multi_agent_still_matches_an_agent_query_term():
    assert "agent" in content_terms("Multi-agent System")


def test_markup_is_not_matched_or_scored():
    assert "jat" not in content_terms(RECORDS[3]["abstract"])
    assert "<jats:p>" not in paper_text(record(3))
    assert paper_text(record(3)).endswith("open questions in this evolving field.")


@pytest.mark.parametrize("number", [12, 30])
@pytest.mark.parametrize("query", QUERIES)
def test_title_only_relevant_records_pass_every_saved_planner_query(number, query):
    assert RECORDS[number]["abstract"] is None
    assert RelevanceFilter().is_relevant(record(number), query)


def test_llm_equivalence_is_what_lets_12_match_the_literature_review_query():
    matched = RelevanceFilter().matched_terms(record(12), QUERIES[0])
    assert "llm" in matched and "literature" in matched


def test_3_is_kept_by_its_matching_query():
    assert RelevanceFilter().is_relevant(record(3), QUERIES[0])


@pytest.mark.parametrize("number", [18])
@pytest.mark.parametrize("query", QUERIES)
def test_clearly_irrelevant_record_fails_every_query(number, query):
    assert not RelevanceFilter().is_relevant(record(number), query)


def test_title_only_irrelevant_record_is_dropped_unless_a_word_coincides():
    # NaOH extraction paper: title-only, no query word except "extraction" (q3).
    kept = [n for n, query in enumerate(QUERIES, 1)
            if RelevanceFilter().is_relevant(record(21), query)]
    assert kept == [3]


# ---------------------------------------------------------------------------
# Planner path: aggregate, deduplicate, score; no per-query lexical filter
# ---------------------------------------------------------------------------

class FakeSource:
    """A scholarly source returning fixed saved records per query (no network)."""

    def __init__(self, results_by_query):
        self.results_by_query = results_by_query

    def search(self, query, max_results, only_journal_conference):
        return list(self.results_by_query.get(query, []))


class FakeEmbedder:
    VOCAB = ("research", "gap", "literature", "review", "automated", "multi", "agent",
             "llm", "large", "language", "scientific", "discovery", "assistant", "survey")

    def _vector(self, text):
        tokens = re.findall(r"[a-z]+", text.lower())
        return [float(sum(1 for t in tokens if t.startswith(term))) for term in self.VOCAB]

    def embed_query(self, text):
        return self._vector(text)

    def embed_documents(self, texts):
        return [self._vector(text) for text in texts]


def planner_agent(results_by_query):
    plan = SearchPlan(original_topic=TOPIC, research_domain="AI", search_queries=QUERIES,
                      target_modalities=[], evaluation_focus=[])
    planner = type("Planner", (), {"create_search_plan": lambda self, topic: plan})()
    retriever = MultiSourceRetriever(peer_reviewed_only=True)
    retriever.retrievers = [FakeSource(results_by_query)]
    return PlannerRetrievalAgent(planner=planner, retriever=retriever,
                                 ranker=PaperRelevanceRanker(embedding_model=FakeEmbedder()))


def test_planner_keeps_candidates_whatever_query_found_them():
    # Each relevant record is returned by a query it does NOT lexically match
    # (the worst case the old per-query filter could not survive).
    results = {
        QUERIES[1]: [record(3)],              # #3 fails q2 lexically
        QUERIES[2]: [record(12), record(21)],  # NaOH also returned
        QUERIES[4]: [record(30), record(18)],
    }
    assert not RelevanceFilter().is_relevant(record(3), QUERIES[1])
    _, papers, scores = planner_agent(results).search(TOPIC, with_relevance=True)

    found = {paper.source_id for paper in papers}
    assert found == {sid(n) for n in (3, 12, 30, 21, 18)}   # nothing silently dropped
    assert len(papers) == len(scores) == 5
    assert scores == sorted(scores, reverse=True)


def test_cross_query_duplicates_are_merged_before_scoring():
    # #12 is title-only; a second copy carrying an abstract fills it in.
    with_abstract = record(12, abstract="An LLM research assistant for literature review.",
                           source="semantic_scholar", source_id="s2-duplicate")
    results = {QUERIES[0]: [record(12)], QUERIES[3]: [with_abstract]}
    _, papers = planner_agent(results).search(TOPIC)
    assert len(papers) == 1
    assert papers[0].source_id == sid(12)            # first-seen record and ID kept
    assert papers[0].abstract == with_abstract.abstract
    assert RECORDS[12]["abstract"] is None           # saved record not mutated


def test_standalone_multi_source_retriever_still_filters_by_default():
    retriever = MultiSourceRetriever(peer_reviewed_only=True)
    retriever.retrievers = [FakeSource({QUERIES[3]: [record(12), record(18)]})]
    assert [p.source_id for p in retriever.search(QUERIES[3])] == [sid(12)]
    unfiltered = retriever.search(QUERIES[3], apply_relevance_filter=False)
    assert [p.source_id for p in unfiltered] == [sid(12), sid(18)]


def test_ranking_is_deterministic_over_saved_records():
    papers = [record(n) for n in (21, 30, 3, 12, 34, 18, 22, 32)]
    ranker = PaperRelevanceRanker(embedding_model=FakeEmbedder())
    first = [(i.paper.source_id, i.score) for i in ranker.rank(TOPIC, papers)]
    assert first == [(i.paper.source_id, i.score) for i in ranker.rank(TOPIC, list(papers))]


# ---------------------------------------------------------------------------
# Explicit threshold: recorded, reported, configurable
# ---------------------------------------------------------------------------

def run_with_saved_scores(**overrides):
    order = (21, 30, 3, 12, 34, 18, 22, 32)
    agents = wf.workflow_agents()
    agents.retrieval.search.return_value = (
        wf.search_plan(), [record(n) for n in order], [RELEVANCE[n] for n in order])
    agents.paper_analysis._resolve_pdf_path.side_effect = lambda paper, paper_id, paths: "x.pdf"
    return agents, wf.invoke(agents, max_papers=6, pdf_paths={}, **overrides)


def test_threshold_excludes_only_irrelevant_candidates_and_reports_each_one():
    agents, result = run_with_saved_scores()
    outcomes = {entry["paper_id"]: entry for entry in result["paper_selection"]}

    for number in RELEVANT:
        assert outcomes[sid(number)]["outcome"] == "analyzed"
        assert outcomes[sid(number)]["relevance"] == RELEVANCE[number]
    for number in IRRELEVANT:
        assert outcomes[sid(number)]["outcome"] == "not_attempted_below_relevance"
        assert outcomes[sid(number)]["relevance"] == RELEVANCE[number]
    assert result["relevance_threshold"] == DEFAULT_RELEVANCE_THRESHOLD
    assert result["relevance_method"] == RANKING_METHOD
    # Every candidate was scored and kept in state; none removed before scoring.
    assert len(result["papers"]) == len(result["paper_relevance"]) == 8
    analyzed = {call.kwargs["paper"].source_id
                for call in agents.paper_analysis.analyze_paper.call_args_list}
    assert analyzed == {sid(n) for n in RELEVANT}

    report = render_report(result)
    assert "no per-query lexical filter was applied" in report
    assert "Excluded below the relevance threshold 0.75 (not analyzed):" in report
    for number in IRRELEVANT:
        assert f"`{sid(number)}` ({RELEVANCE[number]:.3f})" in report


def test_threshold_is_configurable_and_recorded(monkeypatch):
    monkeypatch.setenv("PAPER_RELEVANCE_THRESHOLD", "0.6")
    _, result = run_with_saved_scores()
    outcomes = {entry["paper_id"]: entry["outcome"] for entry in result["paper_selection"]}
    assert result["relevance_threshold"] == 0.6
    assert outcomes[sid(18)] == "analyzed"           # 0.6148 >= 0.6
    assert outcomes[sid(32)] == "not_attempted_below_relevance"   # 0.5910 < 0.6


def test_invalid_threshold_fails_closed(monkeypatch):
    monkeypatch.setenv("PAPER_RELEVANCE_THRESHOLD", "high")
    with pytest.raises(ValueError, match="PAPER_RELEVANCE_THRESHOLD must be a number"):
        run_with_saved_scores()
