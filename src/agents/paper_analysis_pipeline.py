
from pathlib import Path

from src.agents.paper_analysis_agent import (
    PaperAnalysisAgent,
    PaperAnalysisResult,
)
from src.agents.cross_paper_comparison_agent import (
    CrossPaperComparison,
    CrossPaperComparisonAgent,
)
from src.rag import (
    PaperIngestionPipeline,
    RAGRetriever,
)
from src.retrieval import PaperRecord


class PaperAnalysisPipeline:
    def __init__(
        self,
        ingestion_pipeline: PaperIngestionPipeline | None = None,
        rag_retriever: RAGRetriever | None = None,
        analysis_agent: PaperAnalysisAgent | None = None,
        comparison_agent: CrossPaperComparisonAgent | None = None,
    ):
        self.ingestion_pipeline = (
            ingestion_pipeline or PaperIngestionPipeline()
        )
        self.rag_retriever = (
            rag_retriever or RAGRetriever()
        )
        self.analysis_agent = (
            analysis_agent or PaperAnalysisAgent()
        )
        self.comparison_agent = (
            comparison_agent or CrossPaperComparisonAgent()
        )

    def analyze_paper(
        self,
        paper: PaperRecord,
        pdf_path: str,
        analysis_query: str,
        top_k: int = 5,
    ) -> PaperAnalysisResult:

        if not pdf_path.strip():
            raise ValueError("PDF path cannot be empty.")

        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if not analysis_query.strip():
            raise ValueError("Analysis query cannot be empty.")

        paper_id = self._get_paper_id(paper)

        print(f"\n=== INGESTING: {paper.title} ===")

        chunk_count = self.ingestion_pipeline.ingest(
            pdf_path=pdf_path,
            paper_id=paper_id,
        )

        print(f"Ingested chunks: {chunk_count}")

        print(f"\n=== RETRIEVING EVIDENCE: {paper_id} ===")

        evidence = self.rag_retriever.retrieve(
            query=analysis_query,
            top_k=top_k,
            paper_id=paper_id,
        )

        print(f"Retrieved evidence chunks: {len(evidence)}")

        if not evidence:
            raise ValueError(
                f"No evidence retrieved for paper: {paper_id}"
            )

        print(f"\n=== ANALYZING: {paper_id} ===")

        return self.analysis_agent.analyze(
            evidence_chunks=evidence
        )

    def compare_two_papers(
        self,
        paper_1: PaperRecord,
        pdf_path_1: str,
        paper_2: PaperRecord,
        pdf_path_2: str,
        analysis_query: str,
        top_k: int = 5,
    ) -> CrossPaperComparison:

        paper_1_id = self._get_paper_id(paper_1)
        paper_2_id = self._get_paper_id(paper_2)

        if paper_1_id == paper_2_id:
            raise ValueError(
                "Both papers have the same identifier."
            )

        print("\n========== PAPER 1 ANALYSIS ==========")

        result_1 = self.analyze_paper(
            paper=paper_1,
            pdf_path=pdf_path_1,
            analysis_query=analysis_query,
            top_k=top_k,
        )

        print("\n========== PAPER 2 ANALYSIS ==========")

        result_2 = self.analyze_paper(
            paper=paper_2,
            pdf_path=pdf_path_2,
            analysis_query=analysis_query,
            top_k=top_k,
        )

        print("\n========== CROSS-PAPER COMPARISON ==========")

        comparison = self.comparison_agent.compare(
            paper_1=result_1,
            paper_2=result_2,
        )

        return comparison

    @staticmethod
    def _get_paper_id(paper: PaperRecord) -> str:

        if paper.source_id:
            return paper.source_id

        if paper.doi:
            return paper.doi

        normalized_title = (
            paper.title.lower().strip().replace(" ", "_")
        )

        if normalized_title:
            return normalized_title

        raise ValueError(
            "Paper does not contain a usable identifier."
        )
