from src.agents import PaperAnalysisPipeline
from src.retrieval import PaperRecord


PDF_PATH = "data/papers/test_paper.pdf"


def main():
    paper = PaperRecord(
        title="Test Paper",
        authors=[],
        year=None,
        doi=None,
        abstract=None,
        source="test",
        source_id="pipeline_test_paper",
        landing_url=None,
        pdf_url=None,
    )

    pipeline = PaperAnalysisPipeline()

    analysis = pipeline.analyze_paper(
        paper=paper,
        pdf_path=PDF_PATH,
        analysis_query=(
            "What are the main problem, methodology, "
            "results, limitations, and future work "
            "discussed in the paper?"
        ),
        top_k=5,
    )

    print("\n=== FINAL PAPER ANALYSIS ===")

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