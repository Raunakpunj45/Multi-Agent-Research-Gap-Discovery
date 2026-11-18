"""Regression tests for the evidence-grounding precision fixes.

Every case is taken from the saved 10-paper live run (tests/phase5_ten_paper_case.py):

* GAP-001 cited P4 chunk_14 for "agent coordination remains challenging"; the phrase
  is actually a literature-table row about another work in P4 chunk_19.
* GAP-002 relied on the same P4 secondary text; P8's genuine statement failed quote
  verification because the stored text hyphenates "chal- lenge" / "evalu- ation".
* GAP-003 was built only from absence statements ("The paper does not specify a
  particular dataset or benchmark") grounded in a header, a table row and an
  unrelated sentence, and carried an unadjudicated "contradiction" whose quotes
  state neither proposition.

All tests are offline: fake embedders and fake model clients only.
"""
import json
import re
from types import SimpleNamespace

import pytest

from src.agents.artifact_provenance import fingerprint_json
from src.agents.cross_paper_comparison_agent import EvidenceRef
from src.agents.evidence_matrix import EvidenceMatrixComparisonAgent, split_statements
from src.agents.evidence_matrix_models import EvidenceMatrix
from src.agents.evidence_precision import (
    claim_coverage,
    contains_claim,
    is_absence_claim,
    is_secondary,
    non_evidence_reason,
    quote_states_absence,
    split_compound,
)
from src.agents.evidence_validation_agent import (
    MAX_ADJUDICATIONS_PER_GAP,
    EvidenceValidationAgent,
    EvidenceValidationStatus as EV,
    GapValidation,
    GapValidationStatus,
    _ContradictionAdjudication,
)
from src.agents.gap_acceptance import acceptance_blockers
from src.agents.paper_analysis_agent import PaperAnalysis, PaperAnalysisResult
from src.agents.quote_match import locate_quote
from src.agents.research_gap_detection_agent import (
    ResearchGap,
    ResearchGapDetectionAgent,
    ResearchGapDetectionResult,
)
from src.rag.models import RetrievedChunk
from tests.phase5_ten_paper_case import ANALYSES, GAPS, LABELS, SAVED_MATRIX, SAVED_VALIDATIONS, TOPIC

P2, P4, P5, P8, P10 = (f"https://openalex.org/{w}" for w in (
    "W7222723116", "W7214432694", "W7171304406", "W7166808050", "W7126221181"))
CHUNK = {e["chunk_id"]: e["text"] for a in ANALYSES for e in a["evidence"]}
DAT2_STATEMENTS = [
    "The paper does not specify a particular dataset or benchmark",
    "No specific experimental dataset is introduced",
    "The paper does not report the use of a specific experimental dataset",
]
P4_COMPOUND = ("The evidence indicates that agent coordination and evaluation remain challenging, "
               "automatic knowledge‑graph construction can introduce incorrect or duplicate "
               "relationships, and long‑context processing still limits information utilization.")
P5_STATEMENT = ("and the current lack of effective multi‑agent coordination mechanisms for "
                "complex scientific goals.")
P5_QUOTE = ("One of the other significant weaknesses is the absence of an effective mechanism of "
            "multi-agent coordination that can break down the complex scientific goals")
P4_TABLE_ROW = "Agent coordination and evaluation remain challenging"
P8_STORED = ("A key chal- lenge in achieving such continuous evolution lies in establishing more "
             "objective and high-quality evalu- ation mechanisms")
P8_DEHYPHENATED = ("A key challenge in achieving such continuous evolution lies in establishing more "
                   "objective and high-quality evaluation mechanisms")


class FlatEmbedder:
    """Every text gets the same vector: similarity never decides grounding here,
    so only the deterministic precision rules do."""

    model_name = "flat-test-embedder"

    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]


def results():
    return [PaperAnalysisResult.model_validate(a) for a in ANALYSES]


@pytest.fixture(scope="module")
def matrix():
    return EvidenceMatrixComparisonAgent(embedding_model=FlatEmbedder()).build_matrix(results())


def cells_of(matrix, paper_id, dimension=None):
    return [c for c in matrix.cells
            if c.paper_id == paper_id and (dimension is None or c.dimension == dimension)]


def clustered(matrix):
    return {(m.paper_id, m.chunk_id, m.statement) for c in matrix.clusters for m in c.members}


# ---------------------------------------------------------------------------
# 1. Absence claims
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("statement", DAT2_STATEMENTS + [
    "Specific datasets are not named",
    "The dataset is not mentioned in the paper.",
    "No standardized benchmark is used.",
    "The authors did not release their code.",
])
def test_absence_statements_are_detected(statement):
    assert is_absence_claim(statement)


@pytest.mark.parametrize("statement", [
    P5_STATEMENT,                                              # a substantive limitation
    "Experiments were conducted on 1,200 PubMed articles drawn from three distinct scientific domains.",
    "The system lacks robustness to noisy citations.",
])
def test_substantive_statements_are_not_absence_claims(statement):
    assert not is_absence_claim(statement)


def test_gap003_dat2_absence_statements_never_reach_a_cluster(matrix):
    statements = {s for _, _, s in clustered(matrix)}
    assert not statements & set(DAT2_STATEMENTS)
    for paper in (P2, P4, P10):
        for cell in cells_of(matrix, paper, "dataset"):
            if cell.statement in DAT2_STATEMENTS:
                # Not grounded at all (no quote contains the claim) or excluded.
                assert not cell.grounded or cell.cluster_exclusion
    assert not any(c.dimension == "dataset" and {P2, P4, P10} <= set(c.paper_ids)
                   for c in matrix.clusters)


def test_author_stated_absence_is_kept():
    text = "No public benchmark dataset was released for this task. Results follow."
    result = PaperAnalysisResult(
        analysis=PaperAnalysis(problem="p", methodology="m",
                               dataset="No public benchmark dataset is released.",
                               results="r", limitations="l", future_work="f"),
        evidence=[RetrievedChunk(chunk_id="x_c1", paper_id="x", text=text)])
    other = PaperAnalysisResult(
        analysis=PaperAnalysis(problem="p", methodology="m", dataset="d", results="r",
                               limitations="l", future_work="f"),
        evidence=[RetrievedChunk(chunk_id="y_c1", paper_id="y", text="Unrelated text here.")])
    built = EvidenceMatrixComparisonAgent(embedding_model=FlatEmbedder()).build_matrix([result, other])
    cell = next(c for c in built.cells if c.paper_id == "x" and c.dimension == "dataset")
    assert cell.grounded and cell.cluster_exclusion is None
    assert quote_states_absence(cell.quote) and cell.quote in text


def test_absence_without_an_author_statement_is_excluded():
    text = "Our pipeline retrieves papers and summarizes datasets reported in each study."
    result = PaperAnalysisResult(
        analysis=PaperAnalysis(problem="p", methodology="m",
                               dataset="The paper does not report the datasets it summarizes.",
                               results="r", limitations="l", future_work="f"),
        evidence=[RetrievedChunk(chunk_id="x_c1", paper_id="x", text=text)])
    other = PaperAnalysisResult(
        analysis=PaperAnalysis(problem="p", methodology="m", dataset="d", results="r",
                               limitations="l", future_work="f"),
        evidence=[RetrievedChunk(chunk_id="y_c1", paper_id="y", text="Unrelated text here.")])
    built = EvidenceMatrixComparisonAgent(embedding_model=FlatEmbedder()).build_matrix([result, other])
    cell = next(c for c in built.cells if c.paper_id == "x" and c.dimension == "dataset")
    assert cell.grounded and cell.cluster_exclusion.startswith("absence claim")
    assert any("missing information in retrieved chunks is not evidence" in c for c in built.caveats)


# ---------------------------------------------------------------------------
# 2. Statement-to-quote grounding
# ---------------------------------------------------------------------------

def test_gap001_p4_quote_does_not_contain_the_claim():
    chunk14_quote = ("Looking at this, whether agent coordination increases evidence coverage "
                     "and decreases unsubstantiated claims.")
    assert chunk14_quote in CHUNK[f"{P4}_chunk_14"]
    assert claim_coverage(P4_COMPOUND, chunk14_quote) < 0.5
    assert not contains_claim(P4_COMPOUND, chunk14_quote)


def test_p5_genuine_quote_contains_its_claim():
    assert P5_QUOTE in CHUNK[f"{P5}_chunk_8"]
    assert contains_claim(P5_STATEMENT, P5_QUOTE)


def test_compound_statement_is_split_into_independent_claims():
    assert split_compound(P4_COMPOUND) == [
        "The evidence indicates that agent coordination and evaluation remain challenging",
        "Automatic knowledge‑graph construction can introduce incorrect or duplicate relationships",
        "Long‑context processing still limits information utilization.",
    ]
    noun_list = ("No specific datasets, experimental data, benchmarks, or evaluation resources "
                 "are reported in the provided evidence.")
    assert split_compound(noun_list) == [noun_list]
    assert split_statements(P4_COMPOUND, 6) == split_compound(P4_COMPOUND)


@pytest.mark.parametrize("quote, reason", [
    # GAP-003's P2 quote: the paper's author/affiliation header.
    ("0 International LLM-Driven Research Assistant for Automated Literature Analysis and "
     "Research Gap Detection Bijoyini Bagchi1, Kaushik Banerjee1, Tunir De1, Trina Chakraborty1, "
     "Sayudh Chakraborty1 Department of Computer Science", "author/title header"),
    ("Jinheon Baek, Sujay Kumar Jauhar, Silviu Cucerzan, and Sung Ju Hwang. 2024. Researchagent: "
     "Iter- ative research idea generation over scientific liter- ature with large language "
     "models. arXiv preprint arXiv:2404.07738.", "bibliographic entry"),
    ("Large Language Models (LLMs), Semantic Retrieval, Research Gap Detection, Multi-Agent "
     "Systems, Academic Literature Analysis.", "keyword list"),
])
def test_headers_bibliography_and_keyword_lines_are_never_evidence(quote, reason):
    assert non_evidence_reason(quote) == reason


def test_genuine_sentences_are_evidence():
    assert non_evidence_reason(P5_QUOTE) is None
    assert non_evidence_reason(P8_STORED) is None


def test_every_grounded_quote_is_exact_chunk_text_and_never_a_header(matrix):
    by_chunk = {e.chunk_id: e.text for r in results() for e in r.evidence}
    grounded = [c for c in matrix.cells if c.grounded]
    assert grounded
    for cell in grounded:
        assert cell.quote in by_chunk[cell.chunk_id]
        assert cell.chunk_id.startswith(cell.paper_id)
        assert non_evidence_reason(cell.quote) is None
        assert contains_claim(cell.statement, cell.quote)


def test_p5_coordination_limitation_is_grounded_in_its_own_sentence(matrix):
    cell = next(c for c in cells_of(matrix, P5, "limitations")
                if "coordination" in c.statement)
    assert cell.grounded and cell.chunk_id == f"{P5}_chunk_8"
    assert "multi-agent coordination" in cell.quote
    assert cell.secondary is None and cell.cluster_exclusion is None


# ---------------------------------------------------------------------------
# 3. Secondary evidence
# ---------------------------------------------------------------------------

def test_literature_table_row_is_secondary():
    row = CHUNK[f"{P4}_chunk_19"]
    start = row.index("Singh et al. [14]")
    assert is_secondary(row[start:start + 190])
    assert not is_secondary(P5_QUOTE)
    assert is_secondary("as shown by (Smith et al., 2021) in prior work")


def test_quote_tail_of_a_cited_table_row_is_secondary_in_context():
    from src.agents.evidence_precision import is_secondary_in_context

    text = CHUNK[f"{P4}_chunk_19"]
    assert P4_TABLE_ROW in text and not is_secondary(P4_TABLE_ROW)
    assert is_secondary_in_context(P4_TABLE_ROW, text)


def test_a_sentence_after_a_cited_sentence_is_not_secondary():
    from src.agents.evidence_precision import is_secondary_in_context

    text = "Prior work reported gains [3]. Our system lacks a coordination mechanism."
    assert not is_secondary_in_context("Our system lacks a coordination mechanism.", text)


def test_gap001_p4_claims_ground_in_chunk19_as_secondary_and_are_not_clustered(matrix):
    p4 = [c for c in cells_of(matrix, P4, "limitations") if "coordination" in c.statement]
    assert p4, "the coordination clause must still be recorded"
    for cell in p4:
        assert cell.grounded and cell.chunk_id == f"{P4}_chunk_19"
        assert cell.secondary is True
        assert cell.cluster_exclusion.startswith("secondary evidence")
    assert not any(m.paper_id == P4 and m.chunk_id in (f"{P4}_chunk_14", f"{P4}_chunk_19")
                   for c in matrix.clusters if c.dimension == "limitations" for m in c.members)
    assert any("not counted as the papers' own findings" in c for c in matrix.caveats)


def test_validation_never_marks_secondary_text_author_stated():
    quote = "Agent coordination and evaluation remain challenging"
    response = assessment(gap003(), ["SUPPORTED", "SUPPORTED", "SUPPORTED"],
                          ["In this project, an AI-driven Research Assistant is introduced",
                           quote, "These capabilities enhance efficiency"],
                          "PARTIALLY_SUPPORTED", author_stated=True)
    item = validate(Client([response]), gap003()).gap_validations[0]
    record = next(r for r in item.evidence_validations if r.chunk_id == f"{P4}_chunk_19")
    finding = json.loads(record.findings)
    assert finding["author_stated"] is False and finding["secondary_evidence"] is True
    assert record.gap_origin_assessment == "INFERRED"
    first = next(r for r in item.evidence_validations if r.chunk_id == f"{P2}_chunk_1")
    assert json.loads(first.findings)["author_stated"] is True     # primary text unaffected


# ---------------------------------------------------------------------------
# 4/5. Prompts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("phrase", [
    "traceable to the exact quote of a label you cite",
    "never generalize to 'current frameworks', 'existing systems', 'the field'",
    "Never propose a gap from silence or missing information",
    "Do not invent agreement between papers",
    "are secondary: never present them as the citing paper's own finding",
])
def test_matrix_gap_prompt_rules(phrase):
    assert phrase in ResearchGapDetectionAgent.MATRIX_SYSTEM_INSTRUCTION


def test_legacy_gap_prompt_has_the_same_core_rules():
    text = ResearchGapDetectionAgent.SYSTEM_INSTRUCTION
    assert "never generalize to the whole field" in text
    assert "Never propose a gap from silence or missing information" in text
    assert "Do not invent agreement between papers" in text


def test_gap_detection_never_sees_p4_secondary_or_dat2_evidence():
    comparison = EvidenceMatrixComparisonAgent(embedding_model=FlatEmbedder()).compare_many(
        results(), research_topic=TOPIC)
    agent = ResearchGapDetectionAgent(client=object())
    clusters = agent._ranked_clusters(comparison.evidence_matrix)
    index = agent._matrix_index(comparison, clusters)
    contents = agent._matrix_contents(comparison, clusters, index, TOPIC, len(clusters))
    shown = set(index.label_by_chunk)
    assert (P4, f"{P4}_chunk_19") not in shown
    for statement in DAT2_STATEMENTS:
        assert statement not in contents
    assert P4_TABLE_ROW not in contents


def test_validation_prompt_states_silence_is_not_support():
    for text in (EvidenceValidationAgent.GAP_SYSTEM_INSTRUCTION,
                 EvidenceValidationAgent.CHUNK_SYSTEM_INSTRUCTION):
        assert "absence of mention is never evidence of absence" in text
    client = Client([gap003_partial_response()])
    validate(client, gap003())
    assert "cannot support a claim that the paper lacks" in client.requests[0]["instruction"]


# ---------------------------------------------------------------------------
# 6. Contradiction adjudication in the default per-gap mode
# ---------------------------------------------------------------------------

def gap003():
    return ResearchGap.model_validate(next(g for g in GAPS if g["gap_id"] == "GAP-003"))


def gap_detection(*gaps):
    return ResearchGapDetectionResult(
        research_topic=TOPIC, paper_count=len(LABELS), paper_ids=list(LABELS), gaps=list(gaps),
        gap_category_counts={}, caveats=[])


class Store:
    def get_chunks_by_ids(self, chunk_ids):
        return SimpleNamespace(chunks=[
            {"chunk_id": cid, "text": CHUNK[cid], "metadata": {"paper_id": cid.split("_chunk_")[0]}}
            for cid in chunk_ids if cid in CHUNK])


class Client:
    """Scripted gap assessments; adjudications come from ``adjudications``."""

    def __init__(self, assessments, adjudications=()):
        self.assessments, self.adjudications = list(assessments), list(adjudications)
        self.requests = []
        self.models = SimpleNamespace(generate_content=self.generate)

    def generate(self, *, model, contents, config=None):
        name = (config.response_json_schema or {}).get("title")
        self.requests.append({"schema": name, "contents": contents,
                              "instruction": config.system_instruction})
        if name == "_ContradictionAdjudication":
            outcome = self.adjudications.pop(0)
        else:
            outcome = self.assessments.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(parsed=outcome)


def labels_for(gap):
    order = {pid: n for n, pid in enumerate(LABELS, 1)}
    return [f"P{order[ref.paper_id]}-E{i}" for ref in gap.supporting_evidence
            for i, _ in enumerate(ref.chunk_ids, 1)]


def assessment(gap, statuses, quotes, joint_status, contradictions=(), author_stated=False):
    evidence = [{
        "evidence_label": label, "status": status,
        "attributed_contribution": f"Contribution of {label}.",
        "claim_evidence_alignment": f"Alignment of {label}.",
        "supported_contribution": f"Supported part of {label}." if quote else None,
        "supported_contribution_quote": quote, "author_stated": author_stated,
        "limitations": f"Limits of {label}.",
    } for label, status, quote in zip(labels_for(gap), statuses, quotes)]
    return {
        "status": joint_status, "complete_gap_assessment": "Assessed.",
        "cross_paper_inference": "Inferred.", "evidence_summary_accuracy": "Partly.",
        "research_opportunity": "Tentative.", "gap_origin": "Inferred across papers.",
        "overgeneralization": "Field-wide claim not supported.", "limitations": "Few chunks.",
        "counterevidence": "None.", "evidence": evidence, "contradictions": list(contradictions),
    }


def validate(client, *gaps):
    agent = EvidenceValidationAgent(vector_store=Store(), client=client, max_retries=1,
                                    base_backoff_seconds=0, max_backoff_seconds=0)
    return agent.validate(gap_detection(*gaps))


SAVED_GAP3_CONTRADICTION = SAVED_VALIDATIONS["GAP-003"]["contradictions"][0]


def gap003_proposal():
    """The run's proposed contradiction, with its exact propositions and quotes."""
    saved = SAVED_GAP3_CONTRADICTION
    # GAP-003 cites P2 chunk_1, P4 chunk_19, P10 chunk_1 in that order; side A is
    # P2's chunk and side B is P4's chunk, exactly as in the saved record.
    label_p2, label_p4, _ = labels_for(gap003())
    assert saved["source_a"]["paper_id"] == P2 and saved["source_b"]["paper_id"] == P4
    return {
        "proposition_a": saved["proposition_a"], "evidence_label_a": label_p2,
        "supporting_quote_a": saved["supporting_quote_a"],
        "proposition_b": saved["proposition_b"], "evidence_label_b": label_p4,
        "supporting_quote_b": saved["supporting_quote_b"], "explanation": saved["explanation"],
    }


def gap003_partial_response(contradictions=()):
    saved = SAVED_VALIDATIONS["GAP-003"]["evidence_validations"]
    return assessment(gap003(), [r["status"] for r in saved],
                      [r["supported_contribution_quote"] for r in saved],
                      "PARTIALLY_SUPPORTED", contradictions)


def adjudication(a=True, b=True, same=True, conflict=True, confidence="HIGH"):
    return _ContradictionAdjudication(
        proposition_a_supported=a, proposition_b_supported=b, same_subject=same,
        substantively_incompatible=conflict, confidence=confidence,
        explanation="Adjudicated from the two quotes only.")


def test_gap003_contradiction_replay_is_unverified_not_a_blocker():
    client = Client([gap003_partial_response([gap003_proposal()])],
                    [adjudication(a=False, b=False)])
    item = validate(client, gap003()).gap_validations[0]

    assert [r["schema"] for r in client.requests] == [
        "_GapEvidenceAssessment", "_ContradictionAdjudication"]
    assert item.contradictions == [] and item.joint_assessment.contradictions == []
    assert item.contradiction_warnings == []
    [unverified] = item.unverified_contradictions
    assert unverified.proposition_a == SAVED_GAP3_CONTRADICTION["proposition_a"]
    assert unverified.supporting_quote_a in CHUNK[f"{P2}_chunk_1"]
    assert unverified.adjudication.startswith(
        "proposition A is not stated by quote A; proposition B is not stated by quote B")
    blockers = acceptance_blockers(item, gap003())
    assert not any("contradiction" in b for b in blockers)
    assert any("PARTIALLY_SUPPORTED" in b for b in blockers)        # still rejected


def test_confirmed_contradiction_still_blocks_full_support():
    gap = gap003()
    quotes = ["In this project, an AI-driven Research Assistant is introduced",
              "It is a reporting guideline rather than an AI implementation",
              "These capabilities enhance efficiency"]
    response = assessment(gap, ["SUPPORTED"] * 3, quotes, "FULLY_SUPPORTED", [gap003_proposal()])
    item = validate(Client([response], [adjudication()]), gap).gap_validations[0]
    assert len(item.contradictions) == 1 and item.contradictions[0].unresolved
    assert item.status is GapValidationStatus.PARTIALLY_SUPPORTED
    assert any("unresolved contradiction" in b for b in acceptance_blockers(item, gap))


@pytest.mark.parametrize("ruling", [
    adjudication(a=False), adjudication(b=False), adjudication(same=False),
    adjudication(conflict=False), adjudication(confidence="MEDIUM"),
])
def test_any_failed_adjudication_condition_leaves_the_contradiction_unverified(ruling):
    gap = gap003()
    response = assessment(gap, ["SUPPORTED"] * 3,
                          ["In this project, an AI-driven Research Assistant is introduced",
                           "It is a reporting guideline rather than an AI implementation",
                           "These capabilities enhance efficiency"],
                          "FULLY_SUPPORTED", [gap003_proposal()])
    item = validate(Client([response], [ruling]), gap).gap_validations[0]
    assert item.contradictions == [] and len(item.unverified_contradictions) == 1


def test_adjudication_failure_makes_the_gap_incomplete():
    client = Client([gap003_partial_response([gap003_proposal()])],
                    [RuntimeError("adjudicator unavailable")])
    result = validate(client, gap003())
    item = result.gap_validations[0]
    assert item.status is GapValidationStatus.VALIDATION_INCOMPLETE
    assert "adjudicator unavailable" in result.semantic_failures[0]


def test_too_many_proposed_contradictions_make_the_gap_incomplete():
    client = Client([gap003_partial_response([gap003_proposal()] * (MAX_ADJUDICATIONS_PER_GAP + 1))])
    result = validate(client, gap003())
    assert result.gap_validations[0].status is GapValidationStatus.VALIDATION_INCOMPLETE
    assert "exceed the adjudication limit" in result.semantic_failures[0]
    assert all(r["schema"] != "_ContradictionAdjudication" for r in client.requests)


def test_saved_gap003_record_still_blocks_and_round_trips():
    saved = GapValidation.model_validate(SAVED_VALIDATIONS["GAP-003"])
    assert saved.unverified_contradictions is None
    assert fingerprint_json(saved.model_dump(mode="json")) == fingerprint_json(SAVED_VALIDATIONS["GAP-003"])
    assert any("unresolved contradiction" in b for b in acceptance_blockers(saved, gap003()))


# ---------------------------------------------------------------------------
# 7. PDF line-break hyphenation
# ---------------------------------------------------------------------------

def test_gap002_p8_dehyphenated_quote_locates_exact_stored_text():
    stored = CHUNK[f"{P8}_chunk_38"]
    assert P8_STORED in stored and P8_DEHYPHENATED not in stored
    assert locate_quote(P8_DEHYPHENATED, stored) == P8_STORED


@pytest.mark.parametrize("quote", [
    "A key challenge in achieving such continuous evolution lies in establishing less objective",
    "A key chal lenge",                                             # a space is not a hyphen break
])
def test_hyphenation_fallback_never_matches_different_text(quote):
    assert locate_quote(quote, CHUNK[f"{P8}_chunk_38"]) is None


def test_hyphenation_aware_claim_matching():
    statement = ("The paper notes that establishing objective and high-quality evaluation "
                 "mechanisms is challenging")
    assert contains_claim(statement, P8_STORED)


def test_validation_stores_the_hyphenated_source_substring():
    gap = ResearchGap.model_validate(next(g for g in GAPS if g["gap_id"] == "GAP-002"))
    response = assessment(gap, ["PARTIALLY_SUPPORTED", "SUPPORTED"],
                          ["intermediate validation at various stages", P8_DEHYPHENATED],
                          "PARTIALLY_SUPPORTED")
    item = validate(Client([response]), gap).gap_validations[0]
    record = next(r for r in item.evidence_validations if r.paper_id == P8)
    assert record.supported_contribution_quote == P8_STORED       # exact source text
    assert CHUNK[f"{P8}_chunk_38"].count(record.supported_contribution_quote) == 1
    assert item.status is GapValidationStatus.PARTIALLY_SUPPORTED   # still rejected


# ---------------------------------------------------------------------------
# Acceptance unchanged; saved artifacts unchanged
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gap_id", ["GAP-001", "GAP-002", "GAP-003"])
def test_saved_gaps_remain_ineligible(gap_id):
    gap = ResearchGap.model_validate(next(g for g in GAPS if g["gap_id"] == gap_id))
    saved = GapValidation.model_validate(SAVED_VALIDATIONS[gap_id])
    assert saved.status is GapValidationStatus.PARTIALLY_SUPPORTED
    assert acceptance_blockers(saved, gap)


def test_saved_matrix_round_trips_with_identical_fingerprint():
    again = EvidenceMatrix.model_validate(SAVED_MATRIX).model_dump(mode="json")
    assert fingerprint_json(again) == fingerprint_json(SAVED_MATRIX)
    assert all("secondary" not in c and "cluster_exclusion" not in c for c in again["cells"])


def test_matrix_method_version_was_bumped():
    from src.agents.evidence_matrix import MATRIX_METHOD_VERSION

    assert MATRIX_METHOD_VERSION == "evidence_matrix_v3"
