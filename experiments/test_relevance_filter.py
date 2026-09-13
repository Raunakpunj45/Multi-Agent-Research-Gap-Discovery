from src.retrieval import MultiSourceRetriever


retriever = MultiSourceRetriever()

query = "large language models scientific discovery"

papers = retriever.search(
    query=query,
    max_results_per_source=2,
)

print(f"\nFinal relevant papers: {len(papers)}\n")

for i, paper in enumerate(papers, start=1):
    print(f"{i}. {paper.title}")
    print(f"   Source: {paper.source}")
    print(f"   Year: {paper.year}")
    print()