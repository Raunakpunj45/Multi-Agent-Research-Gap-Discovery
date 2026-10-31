
from types import SimpleNamespace

import pytest

from src.agents import CrossPaperComparisonAgent


def make_result(paper_id, **analysis_fields):
    defaults = {
        "problem": "Research problem",
        "methodology": "Method A",
        "dataset": "Dataset A",
        "results": "Results not reported",
        "limitations": "Limited evaluation",
        "future_work": "More evaluation needed",
    }
    defaults.update(analysis_fields)

    chunks = [
        SimpleNamespace(
            paper_id=paper_id,
            chunk_id=f"{paper_id}_chunk_1",
        )
    ]

    return SimpleNamespace(
        analysis=SimpleNamespace(**defaults),
        evidence=chunks,
    )


def test_compare_returns_all_six_dimensions():
    agent = CrossPaperComparisonAgent()

    result = agent.compare(
        make_result("paper_1"),
        make_result("paper_2", methodology="Method B"),
    )

    assert result.paper_1_id == "paper_1"
    assert result.paper_2_id == "paper_2"
    assert len(result.comparisons) == 6
    assert {item.dimension for item in result.comparisons} == {
        "problem",
        "methodology",
        "dataset",
        "results",
        "limitations",
        "future_work",
    }


def test_comparison_records_differences_without_claiming_a_gap():
    agent = CrossPaperComparisonAgent()

    result = agent.compare(
        make_result("paper_1", methodology="CNN"),
        make_result("paper_2", methodology="Transformer"),
    )

    methodology = next(
        item for item in result.comparisons
        if item.dimension == "methodology"
    )

    assert methodology.paper_1_observation == "CNN"
    assert methodology.paper_2_observation == "Transformer"
    assert methodology.comparison
    assert result.potential_research_opportunities == []


def test_comparison_handles_missing_evidence():
    agent = CrossPaperComparisonAgent()

    paper_1 = SimpleNamespace(
        analysis=SimpleNamespace(
            problem="Problem A",
            methodology="Method A",
            dataset="Dataset A",
            results="Results A",
            limitations="Limitations A",
            future_work="Future work A",
        ),
        evidence=[],
    )

    with pytest.raises((ValueError, IndexError)):
        agent.compare(paper_1, make_result("paper_2"))
