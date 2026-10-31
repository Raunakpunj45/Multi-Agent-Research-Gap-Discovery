
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.embeddings import LocalEmbeddingModel
from src.rag import PaperIngestionPipeline, RAGRetriever
from src.vectorstore import ChromaVectorStore


PAPERS_DIR = ROOT_DIR / "data" / "papers"

PAPERS = [
    ("10_1038_s41746_023_00811_0.pdf", "paper_nature_2023"),
    ("10_1109_access_2023_3313977.pdf", "paper_ieee_2023"),
]

QUERY = "What methods, limitations, and challenges are discussed in these research papers?"
TOP_K = 10


def main():
    embedding_model = LocalEmbeddingModel()
    vector_store = ChromaVectorStore.for_embedder(
        embedding_model,
        persist_directory=str(ROOT_DIR / "chroma_db"),
    )
    ingestion = PaperIngestionPipeline(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )
    retriever = RAGRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    print("\n=== MULTI-PAPER INGESTION ===")

    for filename, paper_id in PAPERS:
        pdf_path = PAPERS_DIR / filename

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        count = ingestion.ingest(str(pdf_path), paper_id)
        print(f"{paper_id}: {count} chunks")

    print("\n=== COLLECTION-WIDE RETRIEVAL ===")
    print(f"Query: {QUERY}")

    results = retriever.retrieve(query=QUERY, top_k=TOP_K)

    if not results:
        print("No results returned.")
        return

    distinct_papers = set()

    for rank, chunk in enumerate(results, start=1):
        distinct_papers.add(chunk.paper_id)
        print(f"\n--- Result {rank} ---")
        print(f"Paper ID: {chunk.paper_id}")
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Cosine distance: {chunk.distance}")
        print(f"Evidence: {chunk.text[:700].replace(chr(10), ' ')}")

    print("\n=== SUMMARY ===")
    print(f"Results returned: {len(results)}")
    print(f"Distinct papers represented: {len(distinct_papers)}")
    print(f"Paper IDs: {', '.join(sorted(distinct_papers))}")

    if len(distinct_papers) == len(PAPERS):
        print("Both test papers are represented in the results.")
    else:
        print(
            "Not all test papers appear in the top results. "
            "This is not automatically a failure; inspect query relevance "
            "and consider increasing TOP_K."
        )


if __name__ == "__main__":
    main()
