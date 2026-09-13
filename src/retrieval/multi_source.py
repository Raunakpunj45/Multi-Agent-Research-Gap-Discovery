from .arxiv import ArxivRetriever
from .crossref import CrossrefRetriever
from .deduplicator import PaperDeduplicator
from .models import PaperRecord
from .openalex import OpenAlexRetriever
from .semantic_scholar import SemanticScholarRetriever


class MultiSourceRetriever:
    def __init__(self):
        self.retrievers = [
            OpenAlexRetriever(),
            CrossrefRetriever(),
            SemanticScholarRetriever(),
            ArxivRetriever(),
        ]

        self.deduplicator = PaperDeduplicator()

    def search(
        self,
        query: str,
        max_results_per_source: int = 5,
    ) -> list[PaperRecord]:

        all_papers = []

        for retriever in self.retrievers:
            source_name = retriever.__class__.__name__

            try:
                papers = retriever.search(
                    query=query,
                    max_results=max_results_per_source,
                )

                all_papers.extend(papers)

                print(
                    f"{source_name}: "
                    f"{len(papers)} papers retrieved."
                )

            except Exception as error:
                print(
                    f"{source_name}: unavailable — {error}"
                )

        print(f"\nBefore deduplication: {len(all_papers)} papers")

        unique_papers = self.deduplicator.deduplicate(all_papers)

        print(
            f"After deduplication: "
            f"{len(unique_papers)} papers"
        )

        return unique_papers