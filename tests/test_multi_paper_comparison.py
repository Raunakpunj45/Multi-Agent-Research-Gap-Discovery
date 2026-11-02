from types import SimpleNamespace

import pytest

from src.agents import PaperAnalysisPipeline
from src.agents.cross_paper_comparison_agent import (
    CrossPaperComparisonAgent,
    LLMComparison,
    LLMConflict,
    LLMDirection,
    LLMDimensionSummary,
    LLMEvidenceRef,
    LLMFinding,
)
from src.retrieval import PaperRecord


def make_result(paper_id, n_chunks=2):
    return SimpleNamespace(
        analysis=SimpleNamespace(
            problem=f"{paper_id} problem",
            methodology=f"{paper_id} method",
            dataset=f"{paper_id} data",
            results=f"{paper_id} results",
            limitations=f"{paper_id} limits",
            future_work=f"{paper_id} future",
        ),
        evidence=[
            SimpleNamespace(
                paper_id=paper_id,
                chunk_id=f"{paper_id}_chunk_{i}",
                text=f"text {i}",
            )
            for i in range(1, n_chunks + 1)
        ],
    )


def ref(paper, *chunks):
    return LLMEvidenceRef(paper_label=paper, evidence_labels=list(chunks))


class FakeClient:
    def __init__(self, parsed):
        self.prompts = []
        self.models = SimpleNamespace(generate_content=self._generate)
        self._parsed = parsed

    def _generate(self, model, contents, config):
        self.prompts.append(contents)
        return SimpleNamespace(parsed=self._parsed)


# Paper IDs deliberately look like real IDs (URLs / DOIs), not P1/P2.
IDS = ["https://openalex.org/W1", "10.1/abc", "paper three"]


def comparison_for(parsed):
    agent = CrossPaperComparisonAgent(client=FakeClient(parsed))
    return agent.compare_many([make_result(i) for i in IDS])


def test_labels_map_back_to_exact_ids_and_unsupported_items_are_dropped():
    parsed = LLMComparison(
        dimension_summaries=[
            LLMDimensionSummary(
                dimension="Future Work", summary="s",
                evidence=[ref("P1", "P1-E1")]),
            LLMDimensionSummary(  # invented chunk label -> no evidence -> dropped
                dimension="results", summary="s", evidence=[ref("P1", "P1-E9")]),
        ],
        recurring_patterns=[
            LLMFinding(  # supported by 2 papers; P2-E2 is valid, P1-E2 mismatched below
                title="t", description="d", dimensions=["methodology"],
                evidence=[ref("P1", "P1-E2"), ref("P2", "P2-E1", "P2-E2")]),
            LLMFinding(  # single paper only -> dropped
                title="t2", description="d2", dimensions=["results"],
                evidence=[ref("P1", "P1-E1")]),
            LLMFinding(  # chunk of another paper cited under P3 -> dropped
                title="t3", description="d3", dimensions=["results"],
                evidence=[ref("P3", "P1-E1"), ref("P9", "P9-E1")]),
        ],
        common_limitations=[],
        conflicting_findings=[
            LLMConflict(
                topic="x", description="d", position_a="a",
                position_a_evidence=[ref("P1", "P1-E1")],
                position_b="b", position_b_evidence=[ref("P2", "P2-E1")]),
            LLMConflict(  # same paper on both sides -> dropped
                topic="y", description="d", position_a="a",
                position_a_evidence=[ref("P1", "P1-E1")],
                position_b="b", position_b_evidence=[ref("P1", "P1-E2")]),
        ],
        methodological_differences=[],
        underexplored_directions=[
            LLMDirection(direction="dir", rationale="r",
                         evidence=[ref("P2", "P2-E1"), ref("P3", "P3-E2")]),
            LLMDirection(direction="weak", rationale="r",
                         evidence=[ref("P3", "P3-E1")]),
        ],
        caveats=["model caveat"],
    )

    result = comparison_for(parsed)

    assert result.paper_ids == IDS and result.paper_count == 3

    assert [s.dimension for s in result.dimension_summaries] == ["future_work"]

    assert len(result.recurring_patterns) == 1
    refs = {r.paper_id: r.chunk_ids for r in result.recurring_patterns[0].evidence}
    assert refs == {
        "https://openalex.org/W1": ["https://openalex.org/W1_chunk_2"],
        "10.1/abc": ["10.1/abc_chunk_1", "10.1/abc_chunk_2"],
    }

    assert len(result.conflicting_findings) == 1
    assert len(result.underexplored_directions) == 1
    assert result.underexplored_directions[0].status == "candidate"
    assert "model caveat" in result.caveats
    assert any("recurring patterns" in c and "discarded" in c for c in result.caveats)


def test_input_validation():
    agent = CrossPaperComparisonAgent(client=FakeClient(None))

    with pytest.raises(ValueError):
        agent.compare_many([make_result("a")])
    with pytest.raises(ValueError, match="Duplicate"):
        agent.compare_many([make_result("a"), make_result("a")])
    with pytest.raises(ValueError, match="no evidence"):
        agent.compare_many([make_result("a"), make_result("b", n_chunks=0)])
    with pytest.raises(RuntimeError):
        agent.compare_many([make_result("a"), make_result("b")])  # parsed is None


def test_two_paper_agent_still_works_without_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    result = CrossPaperComparisonAgent().compare(make_result("p1"), make_result("p2"))
    assert (result.paper_1_id, result.paper_2_id) == ("p1", "p2")


def test_pipeline_skips_failed_papers_and_returns_one_comparison(tmp_path):
    ok_pdf = tmp_path / "ok.pdf"
    ok_pdf.write_bytes(b"%PDF")

    class Analysis:
        def analyze(self, evidence_chunks):
            return make_result(evidence_chunks[0].paper_id)

    class Retriever:
        def retrieve(self, query, top_k, paper_id):
            return [SimpleNamespace(paper_id=paper_id, chunk_id=f"{paper_id}_c1", text="t")]

    seen_inputs = []

    class Comparer:
        def compare_many(self, results, research_topic=None):
            seen_inputs.append([r.evidence[0].paper_id for r in results])
            return SimpleNamespace(excluded_papers=[])

    pipeline = PaperAnalysisPipeline(
        ingestion_pipeline=SimpleNamespace(ingest=lambda pdf_path, paper_id: 1),
        rag_retriever=Retriever(),
        analysis_agent=Analysis(),
        comparison_agent=Comparer(),
        pdf_downloader=SimpleNamespace(output_dir=tmp_path),
    )

    def paper(sid, pdf=True):
        return PaperRecord(title=sid, authors=[], year=None, doi=None, abstract=None,
                           source="t", source_id=sid, landing_url=None,
                           pdf_url="https://x/a.pdf" if pdf else None)

    papers = [paper("A"), paper("B", pdf=False), paper("C"), paper("A")]
    paths = {"A": str(ok_pdf), "C": str(ok_pdf)}

    comparison = pipeline.analyze_and_compare_papers(
        papers, "q", pdf_paths=paths)

    assert seen_inputs == [["A", "C"]]  # duplicate A ignored, B skipped
    assert [e.paper_id for e in comparison.excluded_papers] == ["B"]

    with pytest.raises(ValueError, match="at least"):
        pipeline.analyze_and_compare_papers([paper("A")], "q", pdf_paths=paths)
