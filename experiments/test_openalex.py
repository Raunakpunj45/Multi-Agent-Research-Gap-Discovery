from src.retrieval import OpenAlexRetriever


def main():
    retriever = OpenAlexRetriever()

    papers = retriever.search(
        query="large language models scientific discovery",
        max_results=5
    )

    print(f"\nRetrieved {len(papers)} papers.\n")

    for index, paper in enumerate(papers, start=1):
        print(f"{index}. {paper.title}")
        print(f"   Year: {paper.year}")
        print(f"   Authors: {', '.join(paper.authors[:3])}")
        print(f"   DOI: {paper.doi}")
        print(f"   Source: {paper.source}")
        print()


if __name__ == "__main__":
    main()