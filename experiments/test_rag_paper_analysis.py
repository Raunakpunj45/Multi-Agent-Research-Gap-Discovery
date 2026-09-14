from src.agents import PaperAnalysisAgent
from src.rag import RAGRetriever


def main():
    retriever = RAGRetriever()

    query = "What problem does the paper address?"

    print("\n=== RAG RETRIEVAL ===")

    evidence = retriever.retrieve(
        query=query,
        top_k=3,
    )

    print(
        f"\nRetrieved evidence chunks: "
        f"{len(evidence)}"
    )

    for index, chunk in enumerate(
        evidence,
        start=1,
    ):
        print(
            f"\n--- Evidence {index} ---"
        )
        print(
            f"Chunk ID: {chunk.chunk_id}"
        )
        print(
            f"Distance: {chunk.distance}"
        )
        print(
            f"Text: {chunk.text[:500]}..."
        )

    print("\n=== PAPER ANALYSIS ===")

    agent = PaperAnalysisAgent()

    analysis = agent.analyze(
        evidence_chunks=evidence
    )

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