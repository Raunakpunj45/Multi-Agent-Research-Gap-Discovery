"""Dry-run estimates of the contradiction-adjudication requests (offline).

Evidence validation now sends one adjudication request per proposed
contradiction (at most MAX_ADJUDICATIONS_PER_GAP per gap). The dry run projects
them as a worst-case stage on the validation model; with zero proposed
contradictions the report is exactly what it was without this stage.
"""
import pytest

from src.agents.evidence_validation_agent import MAX_ADJUDICATIONS_PER_GAP
from src.agents.llm_provider import resolve_model
from src.agents.output_budget import output_budget
from src.workflow.dry_run import (
    ADJUDICATION_STAGE,
    DryRunAssumptions,
    estimate_run,
    format_report,
)

BASE_STAGES = ["planner", "paper analysis", "evidence matrix", "gap detection",
               "evidence validation", "critic", "research questions"]


def report(per_gap, **assumptions):
    return estimate_run({"research_topic": "topic"},
                        assumptions=DryRunAssumptions(contradictions_per_gap=per_gap,
                                                      **assumptions))


def stages(rep):
    return {stage.stage: stage for stage in rep.stages}


def rows(rep, exclude=()):
    return [(s.stage, s.model, s.calls, s.input_tokens, s.output_tokens, s.largest_request,
             s.all_fit, s.basis, tuple(s.notes), s.daily_token_limit)
            for s in rep.stages if s.stage not in exclude]


def totals(rep):
    return {m.model: (m.calls, m.tokens) for m in rep.models}


# ---------------------------------------------------------------------------
# zero proposed contradictions: nothing changes
# ---------------------------------------------------------------------------

def test_zero_contradictions_adds_no_stage_assumption_or_tokens():
    zero = report(0)
    assert [s.stage for s in zero.stages] == BASE_STAGES
    assert not any("contradiction" in line for line in zero.assumptions)
    assert ADJUDICATION_STAGE not in format_report(zero)
    assert sum(m.tokens for m in zero.models) == sum(s.total_tokens for s in zero.stages)


def test_every_other_stage_is_identical_with_or_without_adjudication():
    zero, worst = report(0), report(3)
    assert rows(zero) == rows(worst, exclude=(ADJUDICATION_STAGE,))
    added = stages(worst)[ADJUDICATION_STAGE]
    before, after = totals(zero), totals(worst)
    model = added.model
    assert after[model] == (before[model][0] + added.calls,
                            before[model][1] + added.total_tokens)
    assert {k: v for k, v in after.items() if k != model} == \
        {k: v for k, v in before.items() if k != model}


# ---------------------------------------------------------------------------
# one and three proposed contradictions per gap
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("per_gap", [1, 3])
def test_adjudication_stage_projects_per_gap_calls_on_the_validation_model(per_gap):
    rep = report(per_gap)
    by_name = stages(rep)
    stage, validation = by_name[ADJUDICATION_STAGE], by_name["evidence validation"]
    names = [s.stage for s in rep.stages]
    assert names.index(ADJUDICATION_STAGE) == names.index("evidence validation") + 1

    assert stage.model == resolve_model("VALIDATION") == validation.model
    assert stage.calls == per_gap * validation.calls == per_gap * 3   # three gaps assumed
    assert stage.output_budget_per_call == output_budget("evidence_adjudication")
    assert stage.output_tokens == stage.calls * output_budget("evidence_adjudication")
    assert stage.input_tokens > 0 and stage.all_fit and stage.basis == "projected"
    assert stage.daily_token_limit is not None
    assert "(worst case)" in stage.stage
    assert stage.notes[0].startswith(f"Projected worst case: {per_gap} adjudication request(s)")
    assert any(f"up to {per_gap} proposed contradiction(s) per validated gap" in line
               and "projected worst case" in line for line in rep.assumptions)


def test_three_per_gap_is_three_times_one_per_gap():
    one, three = stages(report(1))[ADJUDICATION_STAGE], stages(report(3))[ADJUDICATION_STAGE]
    assert three.calls == 3 * one.calls
    assert three.input_tokens == 3 * one.input_tokens
    assert three.total_tokens == 3 * one.total_tokens
    assert three.largest_request == one.largest_request


def test_default_and_larger_values_use_the_agent_maximum():
    worst = stages(report(MAX_ADJUDICATIONS_PER_GAP))[ADJUDICATION_STAGE]
    default = stages(estimate_run({"research_topic": "topic"}))[ADJUDICATION_STAGE]
    above = stages(report(MAX_ADJUDICATIONS_PER_GAP + 5))[ADJUDICATION_STAGE]
    assert MAX_ADJUDICATIONS_PER_GAP == 3
    assert default.calls == worst.calls == above.calls
    assert default.total_tokens == worst.total_tokens == above.total_tokens


def test_quote_size_drives_the_projected_request_size():
    small = stages(report(1, chunk_words=60))[ADJUDICATION_STAGE]
    large = stages(report(1, chunk_words=180))[ADJUDICATION_STAGE]
    assert large.input_tokens > small.input_tokens


def test_no_validation_request_means_no_adjudication_request():
    # 360-word chunks make every projected validation request exceed the token
    # budget, so no gap is validated and nothing can be adjudicated.
    by_name = stages(report(3, chunk_words=360))
    assert by_name["evidence validation"].calls == 0
    assert ADJUDICATION_STAGE not in by_name


def test_report_labels_the_rows_as_projected_worst_case():
    text = format_report(report(3))
    assert ADJUDICATION_STAGE in text
    assert "Projected worst case: 3 adjudication request(s) for each of 3 validated gap(s)" in text
    assert "projected" in text.split(ADJUDICATION_STAGE, 1)[1].splitlines()[0]
