"""Deterministic Markdown report of a workflow run (no LLM call).

Renders whatever the run produced, complete or stopped: topic, retrieved
papers, the evidence matrix, candidate gaps, validation results, critic
decisions, research questions and hypotheses, citations and limitations.
Every cited source is an exact (paper ID, chunk ID) pair from the artifacts;
quotes are shown only where a verbatim quote was verified.
"""
from __future__ import annotations

import json

from src.workflow.summary import paper_counts

NOVELTY_SCOPE = (
    "Novelty is limited to the retrieved corpus: every gap, question and "
    "hypothesis below is derived only from the papers listed in this report. "
    "Absence of a topic from these papers does not show that it is unstudied "
    "in the wider literature."
)

INSUFFICIENT_EVIDENCE = (
    "Insufficient evidence: no candidate gap passed deterministic validation, "
    "semantic validation and a Critic ACCEPT decision, so no research question "
    "or hypothesis was generated."
)

_DIMENSION_TITLES = {
    "problem": "Problem",
    "methodology": "Methodology",
    "dataset": "Dataset",
    "results": "Results",
    "limitations": "Limitations",
    "future_work": "Future work",
}


def _cell(text, limit: int = 160) -> str:
    """Markdown-table-safe, single-line, bounded text."""
    value = " ".join(str(text if text is not None else "").split())
    if len(value) > limit:
        value = value[:limit - 1].rstrip() + "…"
    return value.replace("|", "\\|") or "—"


def _value(item, name, default=None):
    return getattr(item, name, default) if item is not None else default


def _enum(value) -> str:
    return getattr(value, "value", value) if value is not None else "—"


class _Citations:
    """Numbered registry of exact (paper ID, chunk ID) citations."""

    def __init__(self):
        self.order: list[tuple[str, str]] = []
        self.quotes: dict[tuple[str, str], str] = {}

    def cite(self, paper_id: str, chunk_id: str, quote: str | None = None) -> str:
        key = (paper_id, chunk_id)
        if key not in self.order:
            self.order.append(key)
        if quote and key not in self.quotes:
            self.quotes[key] = quote
        return f"[{self.order.index(key) + 1}]"

    def cite_refs(self, refs) -> str:
        marks = [self.cite(ref.paper_id, chunk) for ref in refs or [] for chunk in ref.chunk_ids]
        return " ".join(dict.fromkeys(marks)) or "—"


def _paper_id(paper) -> str:
    return _value(paper, "source_id") or _value(paper, "doi") or _value(paper, "title") or "unknown"


def render_report(state: dict) -> str:
    citations = _Citations()
    lines: list[str] = []
    topic = state.get("research_topic") or "Not supplied"
    status = state.get("workflow_status") or "UNKNOWN"
    lines += [f"# Research gap report: {topic}", ""]
    lines.append(f"**Workflow status:** {status}")
    if state.get("stop_reason"):
        lines.append(f"**Stop reason:** {state['stop_reason']}")
    lines += ["", f"> {NOVELTY_SCOPE}", ""]

    lines += _papers_section(state)
    lines += _matrix_section(state, citations)
    lines += _gaps_section(state, citations)
    lines += _validation_section(state, citations)
    lines += _critic_section(state, citations)
    lines += _questions_section(state, citations)
    lines += _citations_section(state, citations)
    lines += _limitations_section(state)
    return "\n".join(lines).rstrip() + "\n"


def _papers_section(state) -> list[str]:
    papers = state.get("papers") or []
    analyzed = {
        item.evidence[0].paper_id for item in state.get("paper_analyses") or [] if item.evidence
    }
    lines = ["## Retrieved papers", ""]
    if not papers:
        return lines + ["No papers were retrieved.", ""]
    selection = state.get("paper_selection")
    if selection:
        return lines + _count_lines(state) + _selection_lines(state, papers, selection)
    lines += ["| # | Paper ID | Title | Year | Source | Analyzed |",
              "|---|---|---|---|---|---|"]
    for number, paper in enumerate(papers, start=1):
        paper_id = _paper_id(paper)
        lines.append(
            f"| {number} | {_cell(paper_id, 60)} | {_cell(_value(paper, 'title'), 90)} | "
            f"{_value(paper, 'year') or '—'} | {_cell(_value(paper, 'source'), 30)} | "
            f"{'yes' if paper_id in analyzed else 'no'} |"
        )
    excluded = state.get("excluded_papers") or []
    if excluded:
        lines += ["", "Papers not analyzed:"]
        lines += [f"- `{item.paper_id}`: {_cell(item.reason, 200)}" for item in excluded]
    return lines + [""]


_OUTCOME_TEXT = {
    "analyzed": "analyzed",
    "failed": "failed (download or analysis)",
    "not_attempted_max_papers": "not attempted (max_papers reached)",
    "not_attempted_below_relevance": "not attempted (below relevance threshold)",
    "not_attempted_quota": "not attempted (daily quota exhausted)",
}


def _count_lines(state) -> list[str]:
    counts = paper_counts(state)
    lines = [
        f"- Retrieved: {counts.retrieved}"
        + (f" ({counts.relevant} at or above the relevance threshold)"
           if counts.relevant is not None else ""),
        f"- Downloaded (PDF obtained): {counts.downloaded}",
        f"- Analyzed: {counts.analyzed}",
        f"- Failed: {counts.failed} (download {counts.failed_download}, "
        f"analysis {counts.failed_analysis}"
        + (f", step not recorded {counts.failed_unknown_step}" if counts.failed_unknown_step else "")
        + ")",
        f"- Skipped: {counts.skipped} (below relevance {counts.skipped_below_relevance}, "
        f"max_papers reached {counts.skipped_max_papers}, quota {counts.skipped_quota})",
    ]
    target = state.get("max_papers")
    if target and counts.analyzed < target:
        lines.append(f"- Target of {target} analyzed papers was not reached: only "
                     f"{counts.analyzed} relevant paper(s) could be downloaded and analyzed.")
    return lines + [""]


def _selection_lines(state, papers, selection) -> list[str]:
    by_id = {entry["paper_id"]: entry for entry in selection}
    threshold = state.get("relevance_threshold")
    lines = []
    if threshold is not None:
        method = state.get("relevance_method")
        lines += [f"All {len(papers)} retrieved candidates were deduplicated and scored "
                  "against the research topic (local embedding cosine similarity"
                  + (f", `{method}`" if method else "") + ") before any exclusion; no "
                  f"per-query lexical filter was applied. Papers below the threshold "
                  f"{threshold} are not analyzed and are listed below.", ""]
    lines += ["| Rank | Paper ID | Title | Year | Source | Relevance | Outcome |",
              "|---|---|---|---|---|---|---|"]
    for number, paper in enumerate(papers, start=1):
        paper_id = (_value(paper, "source_id") or _value(paper, "doi")
                    or _value(paper, "title") or "unknown")
        entry = by_id.get(paper_id, {})
        score = entry.get("relevance")
        outcome = _OUTCOME_TEXT.get(entry.get("outcome"), entry.get("outcome") or "—")
        lines.append(
            f"| {entry.get('rank', number)} | {_cell(paper_id, 60)} | "
            f"{_cell(_value(paper, 'title'), 90)} | {_value(paper, 'year') or '—'} | "
            f"{_cell(_value(paper, 'source'), 30)} | "
            f"{f'{score:.3f}' if isinstance(score, (int, float)) else '—'} | {outcome} |"
        )
    counts: dict[str, int] = {}
    for entry in selection:
        counts[entry["outcome"]] = counts.get(entry["outcome"], 0) + 1
    lines += ["", "Selection summary: " + "; ".join(
        f"{_OUTCOME_TEXT.get(name, name)}: {count}" for name, count in counts.items())]
    titles = {(_value(p, "source_id") or _value(p, "doi") or _value(p, "title") or "unknown"):
              _value(p, "title") for p in papers}
    below = [entry for entry in selection
             if entry["outcome"] == "not_attempted_below_relevance"]
    if below:
        lines += ["", f"Excluded below the relevance threshold {threshold} (not analyzed):"]
        for entry in below:
            score = entry.get("relevance")
            shown = f"{score:.3f}" if isinstance(score, (int, float)) else "no score"
            lines.append(f"- `{entry['paper_id']}` ({shown}): "
                         f"{_cell(titles.get(entry['paper_id']), 120)}")
    failed = [entry for entry in selection if entry["outcome"] == "failed"]
    if failed:
        lines += ["", "Papers that failed download or analysis:"]
        lines += [f"- `{entry['paper_id']}`"
                  + (f" ({entry['failure_step']} step)" if entry.get("failure_step") else "")
                  + f": {_cell(entry.get('reason'), 200)}" for entry in failed]
    return lines + [""]


def _matrix_section(state, citations: _Citations) -> list[str]:
    comparison = state.get("comparison")
    lines = ["## Evidence matrix", ""]
    if comparison is None:
        return lines + ["The run stopped before papers were compared.", ""]
    matrix = getattr(comparison, "evidence_matrix", None)
    if matrix is None:
        return lines + [
            "No evidence matrix: this run used the legacy LLM cross-paper comparison "
            f"({len(comparison.paper_ids)} papers).", "",
        ]
    lines += [
        "Grounded statements per paper and research dimension (each grounded in the "
        "paper's own retrieved chunk; ungrounded statements in parentheses).", "",
        "| Paper | " + " | ".join(_DIMENSION_TITLES.get(d, d) for d in matrix.dimensions) + " |",
        "|---|" + "---|" * len(matrix.dimensions),
    ]
    for paper_id in matrix.paper_ids:
        row = []
        for dimension in matrix.dimensions:
            cells = [c for c in matrix.cells if c.paper_id == paper_id and c.dimension == dimension]
            grounded = sum(1 for c in cells if c.grounded)
            ungrounded = len(cells) - grounded
            row.append(f"{grounded}" + (f" ({ungrounded})" if ungrounded else "") if cells else "—")
        lines.append(f"| {_cell(paper_id, 60)} | " + " | ".join(row) + " |")
    lines += ["", f"### Cross-paper clusters (supported by at least "
                  f"{matrix.min_supporting_papers} distinct papers)", "",
              "Clusters group similar statements by embedding similarity: they show "
              "topical similarity, not verified agreement or a shared method.", ""]
    if not matrix.clusters:
        lines.append("No cross-paper pattern was supported by at least "
                     f"{matrix.min_supporting_papers} distinct papers.")
    for cluster in matrix.clusters:
        lines.append(f"**{cluster.cluster_id}** ({_DIMENSION_TITLES.get(cluster.dimension, cluster.dimension)}, "
                     f"{len(cluster.paper_ids)} papers, mean similarity {cluster.mean_similarity:.2f}): "
                     f"{_cell(cluster.representative_statement, 200)}")
        for member in cluster.members:
            mark = citations.cite(member.paper_id, member.chunk_id, member.quote)
            lines.append(f"- `{member.paper_id}` {mark}: \"{_cell(member.quote, 240)}\"")
        lines.append("")
    return lines + [""]


def _gaps_section(state, citations: _Citations) -> list[str]:
    result = state.get("gap_detection")
    lines = ["## Candidate gaps", ""]
    if result is None:
        return lines + ["The run stopped before gap detection.", ""]
    if not result.gaps:
        lines.append("No candidate gap was produced.")
    for gap in result.gaps:
        lines += [
            f"### {gap.gap_id}: {_cell(gap.title, 200)}", "",
            f"- Category: {_enum(gap.gap_category)}; declared origin: {_enum(gap.gap_origin)}; "
            f"confidence: {_enum(gap.confidence)}",
            f"- Description: {_cell(gap.description, 600)}",
            f"- Reasoning: {_cell(gap.reasoning, 600)}",
            f"- Tentative opportunity: {_cell(gap.research_opportunity, 400)}",
            f"- Supporting evidence: {citations.cite_refs(gap.supporting_evidence)} "
            f"({len(gap.supporting_paper_ids)} papers)",
            "",
        ]
    if result.excluded_candidates:
        lines.append("Excluded candidates:")
        lines += [f"- {_cell(item.title, 120)}: {_cell(item.reason, 200)}"
                  for item in result.excluded_candidates]
    return lines + [""]


def _validation_section(state, citations: _Citations) -> list[str]:
    validation = state.get("evidence_validation")
    lines = ["## Evidence validation", ""]
    if validation is None:
        return lines + ["The run stopped before evidence validation.", ""]
    for item in validation.gap_validations:
        lines.append(f"### {item.gap_id}: {_enum(item.status)}")
        check = getattr(item, "deterministic_check", None)
        if check is not None:
            lines.append(f"- Deterministic checks: {'passed' if check.passed else 'FAILED'} — "
                         + _cell("; ".join(check.checks + check.failures), 400))
        if item.joint_assessment is not None:
            joint = item.joint_assessment
            lines += [
                f"- Joint assessment: {_enum(joint.status)} — {_cell(joint.complete_gap_assessment, 400)}",
                f"- Origin: {_cell(joint.gap_origin, 300)}",
            ]
        elif item.joint_assessment_error:
            lines.append(f"- Joint assessment unavailable: {_cell(item.joint_assessment_error, 300)}")
        if item.contradiction_warnings:
            lines += [f"- Contradiction: {_cell(text, 300)}" for text in item.contradiction_warnings]
        for proposed in getattr(item, "unverified_contradictions", None) or []:
            lines.append(
                f"- Unverified proposed contradiction (not confirmed; not a blocker): "
                f"{_cell(proposed.proposition_a, 120)!r} vs {_cell(proposed.proposition_b, 120)!r}. "
                f"Adjudication: {_cell(proposed.adjudication, 200)}")
        lines += ["", "| Evidence | Status | Author-stated | Verified quote |", "|---|---|---|---|"]
        for record in item.evidence_validations:
            mark = citations.cite(record.paper_id, record.chunk_id,
                                  record.supported_contribution_quote)
            try:
                finding = json.loads(record.findings)
            except (TypeError, json.JSONDecodeError):
                finding = {}
            author = finding.get("author_stated") if isinstance(finding, dict) else None
            lines.append(
                f"| {mark} `{_cell(record.paper_id, 40)}` | {_enum(record.status) if record.status else 'NOT ASSESSED'} | "
                f"{'yes' if author else 'no' if author is False else '—'} | "
                f"{_cell(record.supported_contribution_quote, 160)} |"
            )
        lines.append("")
    failures = list(validation.retrieval_failures) + list(validation.semantic_failures)
    if failures:
        lines.append("Validation failures:")
        lines += [f"- {_cell(text, 300)}" for text in failures]
    return lines + [""]


def _critic_section(state, citations: _Citations) -> list[str]:
    critic = state.get("critic_result")
    lines = ["## Critic decisions", ""]
    if critic is None:
        return lines + ["The run stopped before the critic review.", ""]
    lines += ["| Gap | Decision | Review | Origin | Grounding evidence |",
              "|---|---|---|---|---|"]
    for review in critic.gap_reviews:
        cited = getattr(review, "cited_evidence", None)
        lines.append(
            f"| {review.gap_id} | {_enum(review.decision)} | {_enum(review.review_status)} | "
            f"{_enum(review.origin_assessment)} | {citations.cite_refs(cited) if cited else '—'} |"
        )
    lines.append("")
    for review in critic.gap_reviews:
        lines.append(f"- **{review.gap_id}** reasoning: {_cell(review.reasoning, 600)}")
    if critic.errors:
        lines += ["", "Critic errors:"] + [f"- {_cell(text, 300)}" for text in critic.errors]
    return lines + [""]


def _questions_section(state, citations: _Citations) -> list[str]:
    result = state.get("research_questions")
    lines = ["## Research questions and hypotheses", ""]
    if result is None or not result.gap_results:
        if state.get("critic_result") is not None or result is not None:
            lines.append(INSUFFICIENT_EVIDENCE)
        else:
            lines.append("The run stopped before question and hypothesis generation.")
        if result is not None and result.skipped_gaps:
            lines += [""] + [f"- {item.gap_id} skipped: {_cell(item.reason, 300)}"
                             for item in result.skipped_gaps]
        return lines + [""]
    for gap in result.gap_results:
        lines += [f"### {gap.gap_id}: {_cell(gap.gap_title, 200)}", ""]
        for number, question in enumerate(gap.research_questions, start=1):
            lines.append(f"{number}. {_cell(question.question, 400)} "
                         f"{citations.cite_refs(question.source_evidence)}")
        lines.append("")
        hypothesis = gap.hypothesis
        if hypothesis is not None:
            lines += [
                f"- **H1:** {_cell(hypothesis.alternative_hypothesis, 400)}",
                f"- **H0:** {_cell(hypothesis.null_hypothesis, 400)}",
                f"- Independent variable: {_cell(hypothesis.independent_variable)}; dependent "
                f"variable: {_cell(hypothesis.dependent_variable)}; comparator: "
                f"{_cell(hypothesis.comparator)}; measurable outcome: "
                f"{_cell(hypothesis.measurable_outcome)}",
                f"- Evidence: {citations.cite_refs(hypothesis.source_evidence)}",
            ]
        else:
            lines.append(f"- Hypothesis: insufficient evidence — "
                         f"{_cell(gap.hypothesis_limitation or 'not generated', 400)}")
        lines.append("")
    for item in result.excluded_gaps:
        lines.append(f"- {item.gap_id} excluded: {_cell(item.reason, 300)}")
    for item in result.skipped_gaps:
        lines.append(f"- {item.gap_id} skipped: {_cell(item.reason, 300)}")
    return lines + [""]


def _citations_section(state, citations: _Citations) -> list[str]:
    titles = {_paper_id(paper): _value(paper, "title") for paper in state.get("papers") or []}
    lines = ["## Citations", ""]
    if not citations.order:
        return lines + ["No evidence was cited.", ""]
    for number, (paper_id, chunk_id) in enumerate(citations.order, start=1):
        title = titles.get(paper_id)
        quote = citations.quotes.get((paper_id, chunk_id))
        text = f"[{number}] `{paper_id}`" + (f" — {_cell(title, 120)}" if title else "")
        text += f", chunk `{chunk_id}`"
        if quote:
            text += f": \"{_cell(quote, 300)}\""
        lines.append(text)
    return lines + [""]


def _limitations_section(state) -> list[str]:
    lines = ["## Limitations", "", f"- {NOVELTY_SCOPE}",
             "- Only retrieved chunks of each paper were analyzed, not full texts."]
    comparison = state.get("comparison")
    matrix = getattr(comparison, "evidence_matrix", None) if comparison is not None else None
    if matrix is not None:
        lines += [f"- {caveat}" for caveat in matrix.caveats]
    lines.append("- Model judgements (gap detection, validation, critic, hypotheses) are "
                 "constrained to cited evidence but are not peer review; verify each "
                 "citation before relying on it.")
    return lines + [""]
