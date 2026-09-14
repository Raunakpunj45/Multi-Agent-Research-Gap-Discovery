from src.rag import PaperIngestionPipeline, RAGRetriever


def main():
    print("\n=== INGESTION → RETRIEVAL TEST ===")

    retriever = RAGRetriever()

    query = "What are the limitations of LLM-driven hypothesis generation?"

    results = retriever.retrieve(
        query=query,
        top_k=5,
    )

    print(
        f"\nRetrieved chunks: {len(results)}"
    )

    for index, chunk in enumerate(
        results,
        start=1,
    ):
        print(f"\n--- Result {index} ---")
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Paper ID: {chunk.paper_id}")
        print(f"Distance: {chunk.distance}")
        print(f"Text: {chunk.text[:300]}...")


if __name__ == "__main__":
    main()