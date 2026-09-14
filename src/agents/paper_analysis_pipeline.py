from pathlib import Path

from src.agents.paper_analysis_agent import (
    PaperAnalysis,
    PaperAnalysisAgent,
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
    ):
        self.ingestion_pipeline = (
            ingestion_pipeline
            or PaperIngestionPipeline()
        )

        self.rag_retriever = (
            rag_retriever
            or RAGRetriever()
        )

        self.analysis_agent = (
            analysis_agent
            or PaperAnalysisAgent()
        )

    def analyze_paper(
        self,
        paper: PaperRecord,
        pdf_path: str,
        analysis_query: str,
        top_k: int = 5,
    ) -> PaperAnalysis:

        if not pdf_path.strip():
            raise ValueError(
                "PDF path cannot be empty."
            )

        if not Path(pdf_path).exists():
            raise FileNotFoundError(
                f"PDF not found: {pdf_path}"
            )

        if not analysis_query.strip():
            raise ValueError(
                "Analysis query cannot be empty."
            )

        paper_id = self._get_paper_id(
            paper
        )

        print("\n=== PAPER INGESTION ===")

        chunk_count = (
            self.ingestion_pipeline.ingest(
                pdf_path=pdf_path,
                paper_id=paper_id,
            )
        )

        print(
            f"Ingested chunks: {chunk_count}"
        )

        print("\n=== EVIDENCE RETRIEVAL ===")

        evidence = self.rag_retriever.retrieve(
            query=analysis_query,
            top_k=top_k,
            paper_id=paper_id,
        )

        print(
            f"Retrieved evidence chunks: "
            f"{len(evidence)}"
        )

        print("\n=== PAPER ANALYSIS ===")

        analysis = self.analysis_agent.analyze(
            evidence_chunks=evidence
        )

        return analysis

    @staticmethod
    def _get_paper_id(
        paper: PaperRecord,
    ) -> str:

        if paper.source_id:
            return paper.source_id

        if paper.doi:
            return paper.doi

        normalized_title = (
            paper.title
            .lower()
            .strip()
            .replace(" ", "_")
        )

        if normalized_title:
            return normalized_title

        raise ValueError(
            "Paper does not contain a usable identifier."
        )