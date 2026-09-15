"""
Month 1 Demo (MULTI-PAPER) — Multi-Agent Research Gap Discovery
================================================================

Runs the full Month 1 pipeline end-to-end across MULTIPLE papers, so the
presentation can show several real analyses rather than just one:

  1. Multi-source retrieval (OpenAlex, Crossref, Semantic Scholar) —
     journal/conference papers only.
  2. Try papers one by one (in order) until TARGET_PAPER_COUNT of them
     have been successfully downloaded, ingested, and analyzed. Papers
     that fail (no PDF, download error, extraction error, etc.) are
     skipped and logged, and the next candidate is tried — one bad
     paper does NOT kill the whole demo run.
  3. For each successful paper: download PDF -> extract -> clean ->
     chunk -> embed -> store in ChromaDB -> RAG retrieve evidence ->
     LLM-based structured Paper Analysis Agent output.
  4. Prints a full per-paper breakdown AND a compact summary table at
     the end, and also writes everything to a Markdown report file you
     can screenshot / paste into slides.

Usage:
    python demo_month1_pipeline_multi.py

Before running:
  - .env must have GEMINI_API_KEY set.
  - requirements.txt packages must be installed.
  - Run this from the project root.

Presentation-day safety net:
    python demo_month1_pipeline_multi.py | tee demo_output.txt
This saves a full transcript in case live wifi/API access is flaky.
The script ALSO writes demo_analysis_report.md on its own regardless of
tee, so you always have a saved copy either way.
"""

import re
import sys
from pathlib import Path
from typing import Optional

from src.agents.paper_analysis_agent import PaperAnalysisAgent
from src.processing import PDFDownloader
from src.rag import PaperIngestionPipeline, RAGRetriever
from src.retrieval import MultiSourceRetriever, PaperRecord


ANALYSIS_QUERY = (
    "What are the problem, methodology, dataset, results, "
    "limitations, and future work discussed in this paper?"
)

TARGET_PAPER_COUNT = 2
MAX_CANDIDATES_TO_TRY = 8
MAX_RESULTS_PER_SOURCE = 8

REPORT_PATH = Path("demo_analysis_report.md")


def slugify(text: str, max_length: int = 60) -> str:
    """Filesystem- and ChromaDB-safe identifier (no slashes/spaces/colons)."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text[:max_length] or "paper"


def get_safe_paper_id(paper: PaperRecord) -> str:
    if paper.doi:
        return slugify(paper.doi)
    if paper.source_id:
        return slugify(paper.source_id)
    return slugify(paper.title)


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def process_single_paper(paper: PaperRecord) -> Optional[dict]:
    """
    Runs download -> ingest -> RAG retrieve -> analyze for one paper.
    Returns a result dict on success, or None if this paper should be
    skipped (any step failed).
    """

    if not paper.pdf_url:
        print(f"  SKIP (no PDF URL): {paper.title}")
        return None

    paper_id = get_safe_paper_id(paper)

    try:
        downloader = PDFDownloader()
        pdf_path = downloader.download(
            pdf_url=paper.pdf_url,
            paper_id=paper_id
        )
    except Exception as error:
        print(
            f"  SKIP (download failed: {error}): "
            f"{paper.title}"
        )
        return None

    try:
        ingestion_pipeline = PaperIngestionPipeline()
        chunk_count = ingestion_pipeline.ingest(
            pdf_path=str(pdf_path),
            paper_id=paper_id,
        )
    except Exception as error:
        print(
            f"  SKIP (ingestion failed: {error}): "
            f"{paper.title}"
        )
        return None

    try:
        rag_retriever = RAGRetriever()
        evidence = rag_retriever.retrieve(
            query=ANALYSIS_QUERY,
            top_k=5,
            paper_id=paper_id,
        )
    except Exception as error:
        print(
            f"  SKIP (RAG retrieval failed: {error}): "
            f"{paper.title}"
        )
        return None

    if not evidence:
        print(
            f"  SKIP (no evidence chunks retrieved): "
            f"{paper.title}"
        )
        return None

    try:
        agent = PaperAnalysisAgent()
        result = agent.analyze(
            evidence_chunks=evidence
        )
    except Exception as error:
        print(
            f"  SKIP (analysis failed: {error}): "
            f"{paper.title}"
        )
        return None

    print(
        f"  OK ({chunk_count} chunks): "
        f"{paper.title}"
    )

    return {
        "paper": paper,
        "paper_id": paper_id,
        "chunk_count": chunk_count,
        "evidence": evidence,
        "analysis": result.analysis,
    }


def write_markdown_report(results: list[dict], topic: str) -> None:
    lines = [
        "# Month 1 Demo — Multi-Paper Analysis Report",
        "",
        f"Topic: **{topic}**",
        "",
        f"Papers successfully analyzed: **{len(results)}**",
        "",
    ]

    for i, item in enumerate(results, start=1):
        paper: PaperRecord = item["paper"]
        analysis = item["analysis"]

        lines.append(f"## {i}. {paper.title}")
        lines.append("")
        lines.append(f"- **Source:** {paper.source}")
        lines.append(f"- **Year:** {paper.year}")
        lines.append(f"- **DOI:** {paper.doi or 'N/A'}")
        lines.append(
            f"- **Chunks embedded:** {item['chunk_count']}"
        )
        lines.append("")
        lines.append(
            f"**Problem:** {analysis.problem}"
        )
        lines.append("")
        lines.append(
            f"**Methodology:** {analysis.methodology}"
        )
        lines.append("")
        lines.append(
            f"**Dataset:** {analysis.dataset}"
        )
        lines.append("")
        lines.append(
            f"**Results:** {analysis.results}"
        )
        lines.append("")
        lines.append(
            f"**Limitations:** {analysis.limitations}"
        )
        lines.append("")
        lines.append(
            f"**Future Work:** {analysis.future_work}"
        )
        lines.append("")
        lines.append("---")
        lines.append("")

    REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )


def main() -> None:

    # -----------------------------------------------------------
    # STEP 0: USER INPUT
    # -----------------------------------------------------------
    topic = input(
        "\nEnter research topic: "
    ).strip()

    if not topic:
        print("Research topic cannot be empty.")
        sys.exit(1)

    # -----------------------------------------------------------
    # STEP 1: RETRIEVAL
    # -----------------------------------------------------------
    section("STEP 1: RESEARCH PAPER RETRIEVAL")
    print(f"Query: {topic}\n")

    try:
        retriever = MultiSourceRetriever(
            peer_reviewed_only=True
        )
        papers = retriever.search(
            query=topic,
            max_results_per_source=MAX_RESULTS_PER_SOURCE,
        )
    except Exception as error:
        print(f"Retrieval failed: {error}")
        sys.exit(1)

    if not papers:
        print(
            "No papers retrieved. "
            "Check network/API availability and retry."
        )
        sys.exit(1)

    print(
        f"Total relevant journal/conference papers found: "
        f"{len(papers)}\n"
    )

    for i, paper in enumerate(
        papers[:MAX_CANDIDATES_TO_TRY],
        start=1,
    ):
        has_pdf = (
            "PDF available"
            if paper.pdf_url
            else "no PDF"
        )

        print(
            f"{i}. {paper.title} "
            f"[{paper.source}, {paper.year}] "
            f"({has_pdf})"
        )

    # -----------------------------------------------------------
    # STEP 2: PROCESS PAPERS ONE BY ONE UNTIL WE HIT THE TARGET COUNT
    # -----------------------------------------------------------
    section(
        f"STEP 2: PROCESSING PAPERS "
        f"(target: {TARGET_PAPER_COUNT} successful)"
    )

    results: list[dict] = []
    candidates = papers[:MAX_CANDIDATES_TO_TRY]

    for paper in candidates:
        if len(results) >= TARGET_PAPER_COUNT:
            break

        print(f"\nTrying: {paper.title}")

        outcome = process_single_paper(paper)

        if outcome is not None:
            results.append(outcome)

    if not results:
        print(
            "\nNo papers could be fully processed. "
            "Cannot continue."
        )
        sys.exit(1)

    print(
        f"\nSuccessfully analyzed "
        f"{len(results)}/{TARGET_PAPER_COUNT} target papers."
    )

    # -----------------------------------------------------------
    # STEP 3: PRINT FULL PER-PAPER ANALYSIS
    # -----------------------------------------------------------
    for i, item in enumerate(
        results,
        start=1,
    ):
        paper: PaperRecord = item["paper"]
        analysis = item["analysis"]

        section(
            f"PAPER {i}: {paper.title}"
        )

        print(
            f"Source: {paper.source} | "
            f"Year: {paper.year} | "
            f"DOI: {paper.doi or 'N/A'}"
        )

        print(
            f"Chunks embedded: "
            f"{item['chunk_count']}"
        )

        print(
            "\nProblem:\n"
            + analysis.problem
        )

        print(
            "\nMethodology:\n"
            + analysis.methodology
        )

        print(
            "\nDataset:\n"
            + analysis.dataset
        )

        print(
            "\nResults:\n"
            + analysis.results
        )

        print(
            "\nLimitations:\n"
            + analysis.limitations
        )

        print(
            "\nFuture Work:\n"
            + analysis.future_work
        )

    # -----------------------------------------------------------
    # STEP 4: SUMMARY TABLE
    # -----------------------------------------------------------
    section("SUMMARY TABLE (for slides)")

    for i, item in enumerate(
        results,
        start=1,
    ):
        paper: PaperRecord = item["paper"]

        problem_preview = (
            item["analysis"]
            .problem[:100]
            .replace("\n", " ")
        )

        print(
            f"{i}. {paper.title} "
            f"({paper.year}, {paper.source})"
        )

        print(
            f"   Problem: "
            f"{problem_preview}..."
        )

    # -----------------------------------------------------------
    # STEP 5: SAVE MARKDOWN REPORT
    # -----------------------------------------------------------
    write_markdown_report(
        results,
        topic
    )

    section(
        "DEMO COMPLETE — MONTH 1 PIPELINE "
        "(MULTI-PAPER)"
    )

    print(
        f"Full written report saved to: "
        f"{REPORT_PATH.resolve()}"
    )

    print(
        "This run demonstrated the full Month 1 "
        "scope across multiple papers:\n"
        "  Multi-source retrieval "
        "(journal/conference filtered)\n"
        "  -> PDF download\n"
        "  -> text extraction & cleaning\n"
        "  -> chunking\n"
        "  -> embedding generation\n"
        "  -> ChromaDB vector storage\n"
        "  -> semantic RAG retrieval\n"
        "  -> LLM-based structured paper analysis\n"
        f"...repeated across {len(results)} "
        "independent papers."
    )


if __name__ == "__main__":
    main()