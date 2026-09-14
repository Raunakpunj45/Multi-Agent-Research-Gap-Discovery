from src.agents.planner_agent import PlannerAgent
from src.retrieval import MultiSourceRetriever, PaperRecord


class PlannerRetrievalAgent:
    def __init__(
        self,
        planner: PlannerAgent | None = None,
        retriever: MultiSourceRetriever | None = None,
    ):
        self.planner = planner or PlannerAgent()
        self.retriever = retriever or MultiSourceRetriever()

    def search(
        self,
        topic: str,
        max_results_per_source: int = 5,
    ) -> tuple:

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
            )

            all_papers.extend(papers)

            print(
                f"Relevant papers returned: "
                f"{len(papers)}"
            )

        unique_papers = self._deduplicate_results(
            all_papers
        )

        print(
            f"\nTotal papers after cross-query "
            f"deduplication: {len(unique_papers)}"
        )

        return search_plan, unique_papers

    @staticmethod
    def _deduplicate_results(
        papers: list[PaperRecord],
    ) -> list[PaperRecord]:

        unique_papers = []
        seen_keys = set()

        for paper in papers:

            if paper.doi:
                key = (
                    "doi",
                    paper.doi.lower().strip()
                )

            elif paper.title:
                key = (
                    "title",
                    paper.title.lower().strip()
                )

            else:
                key = (
                    "source_id",
                    paper.source_id,
                )

            if key in seen_keys:
                continue

            seen_keys.add(key)
            unique_papers.append(paper)

        return unique_papers