from pathlib import Path

from src.rag import PaperIngestionPipeline


PDF_PATH = Path(
    "data/papers/test_paper.pdf"
)


def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}"
        )

    pipeline = PaperIngestionPipeline()

    print("\n=== PAPER INGESTION ===")

    chunk_count = pipeline.ingest(
        pdf_path=str(PDF_PATH),
        paper_id="test_paper_ingested",
    )

    print(
        f"\nSuccessfully ingested "
        f"{chunk_count} chunks."
    )

    print(
        f"Total chunks in ChromaDB: "
        f"{pipeline.vector_store.count()}"
    )


if __name__ == "__main__":
    main()