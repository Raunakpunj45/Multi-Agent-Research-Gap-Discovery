
import hashlib
import re
from pathlib import Path

from src.agents.paper_analysis_agent import (
    PaperAnalysisAgent,
    PaperAnalysisResult,
)
from src.agents.cross_paper_comparison_agent import (
    CrossPaperComparison,
    CrossPaperComparisonAgent,
    ExcludedPaper,
    MultiPaperComparison,
)
from src.processing import PDFDownloader
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
        pdf_downloader: PDFDownloader | None = None,
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
        self.pdf_downloader = (
            pdf_downloader or PDFDownloader()
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

    def analyze_and_compare_papers(
        self,
        papers: list[PaperRecord],
        analysis_query: str,
        pdf_paths: dict[str, str] | None = None,
        top_k: int = 5,
        research_topic: str | None = None,
        max_successful: int | None = None,
    ) -> MultiPaperComparison:
        """Analyze each paper, then produce ONE collective comparison.

        A paper that fails (no PDF, download/ingestion/analysis error) is
        skipped and listed in `excluded_papers`; it never aborts the run.
        `pdf_paths` maps paper ID (see _get_paper_id) to a local PDF; papers
        without an entry are downloaded from `paper.pdf_url`.
        `max_successful` stops once that many papers were analyzed.
        """

        if not analysis_query.strip():
            raise ValueError("Analysis query cannot be empty.")

        results: list[PaperAnalysisResult] = []
        excluded: list[ExcludedPaper] = []
        seen_ids: set[str] = set()

        for paper in papers:
            if max_successful is not None and len(results) >= max_successful:
                break

            try:
                paper_id = self._get_paper_id(paper)
            except ValueError as error:
                excluded.append(
                    ExcludedPaper(
                        paper_id=paper.title or "unknown",
                        reason=str(error),
                    )
                )
                continue

            if paper_id in seen_ids:
                continue

            seen_ids.add(paper_id)

            try:
                pdf_path = self._resolve_pdf_path(
                    paper, paper_id, pdf_paths
                )

                results.append(
                    self.analyze_paper(
                        paper=paper,
                        pdf_path=pdf_path,
                        analysis_query=analysis_query,
                        top_k=top_k,
                    )
                )
            except Exception as error:
                print(f"SKIP {paper_id}: {type(error).__name__}: {error}")
                excluded.append(
                    ExcludedPaper(
                        paper_id=paper_id,
                        reason=f"{type(error).__name__}: {error}",
                    )
                )

        if len(results) < 2:
            raise ValueError(
                f"Only {len(results)} paper(s) could be analyzed; at least "
                f"2 are required. Exclusions: "
                f"{[(e.paper_id, e.reason) for e in excluded]}"
            )

        print(
            f"\n========== COLLECTIVE COMPARISON "
            f"({len(results)} papers) =========="
        )

        comparison = self.comparison_agent.compare_many(
            results,
            research_topic=research_topic,
        )

        comparison.excluded_papers.extend(excluded)

        return comparison

    def _resolve_pdf_path(
        self,
        paper: PaperRecord,
        paper_id: str,
        pdf_paths: dict[str, str] | None,
    ) -> str:

        if pdf_paths and pdf_paths.get(paper_id):
            return pdf_paths[paper_id]

        if not paper.pdf_url:
            raise ValueError("No PDF path or PDF URL available.")

        # Paper IDs can be URLs or DOIs, so use a filesystem-safe file name.
        slug = re.sub(r"[^A-Za-z0-9]+", "_", paper_id).strip("_")[:60]
        digest = hashlib.sha1(paper_id.encode("utf-8")).hexdigest()[:8]
        file_id = f"{slug or 'paper'}_{digest}"

        existing = self.pdf_downloader.output_dir / f"{file_id}.pdf"

        if existing.exists():
            return str(existing)

        return str(
            self.pdf_downloader.download(
                pdf_url=paper.pdf_url,
                paper_id=file_id,
            )
        )

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
