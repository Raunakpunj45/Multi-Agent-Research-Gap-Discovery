from src.agents.planner_agent import PlannerAgent
from src.retrieval import MultiSourceRetriever, PaperDeduplicator, PaperRecord
from src.retrieval.ranking import PaperRelevanceRanker


class PlannerRetrievalAgent:
    def __init__(
        self,
        planner: PlannerAgent | None = None,
        retriever: MultiSourceRetriever | None = None,
        ranker: PaperRelevanceRanker | None = None,
    ):
        self.planner = planner or PlannerAgent()
        self.retriever = retriever or MultiSourceRetriever()
        self.ranker = ranker or PaperRelevanceRanker()

    def search(
        self,
        topic: str,
        max_results_per_source: int = 5,
        with_relevance: bool = False,
    ) -> tuple:
        """Plan, retrieve, deduplicate and rank candidates against ``topic``.

        No lexical filter is applied per query: a relevant paper must not be
        lost because one sub-query's words do not appear in its title or
        abstract (title-only records, "LLM" vs "large language model"). Every
        query's results are aggregated and deduplicated first, and every
        candidate is then scored against the original topic; exclusion happens
        only later, by the explicit relevance threshold, and is reported.

        Papers are returned ranked by relevance to the ORIGINAL topic (score
        descending, then retrieval order), so paper selection does not depend
        on the order of sub-queries or sources. Returns ``(plan, papers)``, or
        ``(plan, papers, scores)`` with ``with_relevance=True``, where
        ``scores[i]`` is the relevance score of ``papers[i]``. Scores are kept
        outside ``PaperRecord`` so saved paper records are unchanged.
        """

        if not topic.strip():
            raise ValueError(
                "Research topic cannot be empty."
            )

        print("\n=== PLANNER AGENT ===")

        search_plan = self.planner.create_search_plan(
            topic
        )

        print(
            f"\nResearch Domain: "
            f"{search_plan.research_domain}"
        )

        print("\nGenerated Search Queries:")

        for index, query in enumerate(
            search_plan.search_queries,
            start=1,
        ):
            print(f"{index}. {query}")

        all_papers: list[PaperRecord] = []

        print("\n=== RETRIEVAL ===")

        for index, query in enumerate(
            search_plan.search_queries,
            start=1,
        ):
            print(
                f"\nSearching Query {index}: "
                f"{query}"
            )

            papers = self.retriever.search(
                query=query,
                max_results_per_source=max_results_per_source,
                apply_relevance_filter=False,
            )

            all_papers.extend(papers)

            print(
                f"Candidates returned: "
                f"{len(papers)}"
            )

        unique_papers = self._deduplicate_results(
            all_papers
        )

        print(
            f"\nTotal papers after cross-query "
            f"deduplication: {len(unique_papers)}"
        )

        ranked = self.ranker.rank(topic, unique_papers)
        papers = [item.paper for item in ranked]
        if with_relevance:
            return search_plan, papers, [item.score for item in ranked]
        return search_plan, papers

    @staticmethod
    def _deduplicate_results(
        papers: list[PaperRecord],
    ) -> list[PaperRecord]:
        """Cross-query deduplication by normalized DOI or title, in first-seen order.

        Duplicates fill fields the kept record lacks (abstract, PDF URL, ...),
        so a title-only record found again with an abstract is scored on it.
        """
        return PaperDeduplicator().deduplicate(papers)
