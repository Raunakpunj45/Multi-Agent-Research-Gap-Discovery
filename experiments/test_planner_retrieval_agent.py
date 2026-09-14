from src.agents.planner_retrieval_agent import (
    PlannerRetrievalAgent,
)


agent = PlannerRetrievalAgent()

topic = "Large language models for scientific discovery"

search_plan, papers = agent.search(
    topic=topic,
    max_results_per_source=3,
)

print("\n=== FINAL RETRIEVED PAPERS ===")

for index, paper in enumerate(
    papers[:10],
    start=1,
):
    print(f"\n--- Paper {index} ---")
    print(f"Title: {paper.title}")
    print(f"Year: {paper.year}")
    print(f"Source: {paper.source}")
    print(f"DOI: {paper.doi}")