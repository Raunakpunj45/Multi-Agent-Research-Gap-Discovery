from src.agents.paper_analysis_agent import PaperAnalysisAgent


agent = PaperAnalysisAgent()

paper_text = """
Large language models have recently been explored for scientific
discovery. This study proposes an LLM-based framework for generating
scientific hypotheses from existing literature. The method retrieves
relevant scientific papers and uses a language model to identify
potential research directions.

The framework is evaluated on a collection of scientific literature.
The results show that the approach can generate potentially useful
hypotheses, although the generated hypotheses require expert
validation.

A limitation of the approach is that language models may generate
unsupported or incorrect scientific claims. Future work will focus
on improving factual grounding and validating generated hypotheses
through experiments.
"""

analysis = agent.analyze(paper_text)

print("\n=== PAPER ANALYSIS ===")

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