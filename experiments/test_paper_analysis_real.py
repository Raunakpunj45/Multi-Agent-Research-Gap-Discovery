from pathlib import Path

from src.agents import PaperAnalysisAgent
from src.processing import PDFExtractor


PDF_PATH = Path(
    "data/papers/test_paper.pdf"
)


def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}"
        )

    extractor = PDFExtractor()

    paper_text = extractor.extract_text(
        str(PDF_PATH)
    )

    print(
        f"\nExtracted characters: "
        f"{len(paper_text)}"
    )

    agent = PaperAnalysisAgent()

    analysis = agent.analyze(
        paper_text=paper_text
    )

    print("\n=== REAL PAPER ANALYSIS ===")

    print("\nProblem:")
    print(analysis.problem)

    print("\nMethodology:")
    print(analysis.methodology)

    print("\nDataset:")
    print(analysis.dataset)

    print("\nResults:")
    print(analysis.results)

    print("\nLimitations:")
    print(analysis.limitations)

    print("\nFuture Work:")
    print(analysis.future_work)


if __name__ == "__main__":
    main()