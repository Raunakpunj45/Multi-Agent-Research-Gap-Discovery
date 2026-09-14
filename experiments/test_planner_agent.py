from src.agents.planner_agent import PlannerAgent


agent = PlannerAgent()

topic = "Large language models for scientific discovery"

plan = agent.create_search_plan(topic)

print("\n=== PLANNER AGENT OUTPUT ===")

print("\nOriginal Topic:")
print(plan.original_topic)

print("\nResearch Domain:")
print(plan.research_domain)

print("\nSearch Queries:")
for query in plan.search_queries:
    print(f" - {query}")

print("\nTarget Modalities:")
for modality in plan.target_modalities:
    print(f" - {modality}")

print("\nEvaluation Focus:")
for item in plan.evaluation_focus:
    print(f" - {item}")