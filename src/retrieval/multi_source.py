from .arxiv import ArxivRetriever
from .crossref import CrossrefRetriever
from .deduplicator import PaperDeduplicator
from .models import PaperRecord
from .openalex import OpenAlexRetriever
from .relevance_filter import RelevanceFilter
from .semantic_scholar import SemanticScholarRetriever
 
 
class MultiSourceRetriever:
    def __init__(self, peer_reviewed_only: bool = True):
        # arXiv is a preprint server: it has no journal/conference type
        # field to filter on, so when peer_reviewed_only is True it is
        # excluded entirely rather than silently ignoring the filter.
        self.peer_reviewed_only = peer_reviewed_only
 
        self.retrievers = [
            OpenAlexRetriever(),
            CrossrefRetriever(),
            SemanticScholarRetriever(),
        ]
 
        if not peer_reviewed_only:
            self.retrievers.append(ArxivRetriever())
 
        self.deduplicator = PaperDeduplicator()
        self.relevance_filter = RelevanceFilter()
 
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
                    only_journal_conference=self.peer_reviewed_only,
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
 
        relevant_papers = self.relevance_filter.filter(
            papers=unique_papers,
            query=query,
        )
 
        print(
            f"After relevance filtering: "
            f"{len(relevant_papers)} papers"
        )
 
        return relevant_papers
 