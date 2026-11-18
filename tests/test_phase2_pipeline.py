"""Phase 2: the simplified free-tier pipeline. Offline only.

Covers the deterministic evidence matrix, matrix-based gap detection, per-gap
evidence validation with deterministic checks, the ACCEPT/REJECT critic, the
Markdown report, the dry run, and backward compatibility of saved artifacts.
"""
import json
import re
import zlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.agents.artifact_provenance import ArtifactProvenance, fingerprint_json
from src.agents.critic_agent import (
    CriticAgent,
    CriticDecision,
    CriticResult,
    CriticReviewStatus,
    _BinaryDecision,
    _LLMGapVerdict,
)
from src.agents.cross_paper_comparison_agent import EvidenceRef, MultiPaperComparison
from src.agents.evidence_matrix import (
    EvidenceMatrixComparisonAgent,
    quote_candidates,
    split_statements,
)
from src.agents.evidence_matrix_models import (
    ClusterMember,
    EvidenceCluster,
    EvidenceMatrix,
    MatrixCell,
)
from src.agents.evidence_validation_agent import (
    EvidenceValidationAgent,
    EvidenceValidationResult,
    EvidenceValidationStatus,
    GapSupportStatus,
    GapValidationStatus,
    _GapEvidenceAssessment,
)
from src.agents.llm_provider import resolve_model
from src.agents.paper_analysis_agent import PaperAnalysis, PaperAnalysisResult
from src.agents.research_gap_detection_agent import (
    LLMGapDetection,
    GapDetectionTooLargeError,
    GapOrigin,
    ResearchGapDetectionAgent,
    ResearchGapDetectionResult,
)
from src.agents.research_question_agent import ResearchQuestionAgent, _LLMQuestionSet
from src.rag.models import RetrievedChunk
from src.retrieval.models import PaperRecord
from src.workflow.checkpoint import RunCheckpoint
from src.workflow.dry_run import DryRunAssumptions, RecordingClient, estimate_run, format_report
from src.workflow.langgraph_app import WorkflowAgents, _model_fingerprint
from src.workflow.report import INSUFFICIENT_EVIDENCE, NOVELTY_SCOPE, render_report
from src.workflow.runner import RunInterrupted, TopicRunConfig, run_topic
from tests.conftest import LANGGRAPH_RUN_FIXTURES
from tests.test_langgraph_workflow import search_plan

GPT_120B = "openai/gpt-oss-120b"
GPT_20B = "openai/gpt-oss-20b"

# ---------------------------------------------------------------------------
# Deterministic offline embeddings and a three-paper corpus
# ---------------------------------------------------------------------------

_STOP = {"a", "an", "the", "of", "on", "in", "to", "we", "our", "and", "is", "are",
         "will", "should", "this", "that", "with", "for", "be", "work"}


class BagOfWordsEmbedder:
    """Hashed bag-of-words vectors: shared content words -> high cosine."""

    model_name = "test/bag-of-words"

    def embed_documents(self, texts):
        vectors = []
        for text in texts:
            vector = [0.0] * 128
            for word in re.findall(r"[a-z]+", text.lower()):
                if word not in _STOP:
                    vector[zlib.crc32(word.encode()) % 128] += 1.0
            vectors.append(vector)
        return vectors


CHUNKS = {
    "p1": ("p1_c0", "We train a graph neural network on molecular graphs. "
                    "The evaluation uses only synthetic benchmark data. "
                    "Future studies should test real clinical patient data."),
    "p2": ("p2_c0", "A transformer predicts adverse drug events. "
                    "The evaluation uses only synthetic benchmark records. "
                    "We plan to test real clinical patient data next."),
    "p3": ("p3_c0", "Federated privacy training is expensive on large GPU clusters."),
}


def analysis(paper_id, **fields):
    chunk_id, text = CHUNKS[paper_id]
    values = dict(problem="Not available in the provided evidence.",
                  methodology="Not available in the provided evidence.",
                  dataset="Not available in the provided evidence.",
                  results="Not available in the provided evidence.",
                  limitations="Not available in the provided evidence.",
                  future_work="Not available in the provided evidence.")
    values.update(fields)
    return PaperAnalysisResult(
        analysis=PaperAnalysis(**values),
        evidence=[RetrievedChunk(chunk_id=chunk_id, paper_id=paper_id, text=text)],
    )


def corpus():
    return [
        analysis("p1", methodology="A graph neural network on molecular graphs.",
                 limitations="The evaluation uses only synthetic benchmark data.",
                 future_work="Test real clinical patient data."),
        analysis("p2", methodology="A transformer predicts adverse drug events.",
                 limitations="The evaluation uses only synthetic benchmark records.",
                 future_work="Test real clinical patient data."),
        analysis("p3", limitations="Federated privacy training is expensive on GPU clusters."),
    ]


def matrix_agent():
    return EvidenceMatrixComparisonAgent(embedding_model=BagOfWordsEmbedder())


class Store:
    """Exact-ID chunk store with the stored text and owner of every chunk."""

    def __init__(self, overrides=None):
        self.records = {chunk_id: (paper_id, text) for paper_id, (chunk_id, text) in CHUNKS.items()}
        self.records.update(overrides or {})

    def get_chunks_by_ids(self, chunk_ids):
        chunks = [{"chunk_id": cid, "text": self.records[cid][1],
                   "metadata": {"paper_id": self.records[cid][0], "chunk_id": cid}}
                  for cid in chunk_ids if cid in self.records]
        return SimpleNamespace(chunks=chunks, missing_ids=[c for c in chunk_ids if c not in self.records])


# ---------------------------------------------------------------------------
# A scripted LLM that answers each agent from the labels it was shown
# ---------------------------------------------------------------------------

def _labels(text):
    return list(dict.fromkeys(re.findall(r"\bP\d+-E\d+\b", text)))


class ScriptedLLM:
    """Fake provider client. Records each request by task; never networked."""

    def __init__(self, critic_decision="ACCEPT", fail_tasks=()):
        self.calls: list[str] = []
        self.critic_decision = critic_decision
        self.fail_tasks = set(fail_tasks)
        self.models = self

    def task(self, config):
        instruction = config.system_instruction
        if instruction == ResearchGapDetectionAgent.MATRIX_SYSTEM_INSTRUCTION:
            return "gap"
        # Validation instructions carry the generated output contract after the base text.
        if instruction.startswith(EvidenceValidationAgent.GAP_SYSTEM_INSTRUCTION):
            return "validation"
        if instruction == CriticAgent.VERDICT_INSTRUCTION:
            return "critic"
        if instruction == ResearchQuestionAgent.SYSTEM_INSTRUCTION:
            return "questions"
        return "other"

    def generate_content(self, *, model, contents, config=None):
        task = self.task(config)
        self.calls.append(task)
        if task in self.fail_tasks:
            raise ValueError(f"scripted {task} failure")
        return SimpleNamespace(parsed=getattr(self, f"_{task}")(contents))

    def _gap(self, contents):
        labels = [label for label in _labels(contents) if "-E" in label]
        return LLMGapDetection(gaps=[{
            "title": "Real clinical evaluation of synthetic-benchmark models",
            "description": "Two papers evaluate only on synthetic benchmarks.",
            "gap_category": "Empirical Gap", "reasoning": "Both quotes state synthetic-only evaluation.",
            "evidence": [{"paper_label": label.split("-")[0], "evidence_labels": [label]}
                         for label in labels],
            "evidence_summary": "Each paper evaluates on synthetic benchmark data only.",
            "gap_origin": "explicitly_stated", "confidence": "medium",
            "confidence_reason": "Two papers.", "research_opportunity": "Evaluate on real clinical data.",
        }], caveats=[])

    def _validation(self, contents):
        labels = re.search(r"exactly one entry each\): (.*)", contents).group(1).split(", ")
        blocks = dict(re.findall(r"\[(P\d+-E\d+)\]\n(.*?)(?=\n\n---|\Z)", contents, re.S))
        evidence = []
        for label in labels:
            sentence = re.search(r"[^.]*synthetic[^.]*\.", blocks[label])
            evidence.append({
                "evidence_label": label, "status": "SUPPORTED",
                "attributed_contribution": "Synthetic-only evaluation.",
                "claim_evidence_alignment": "The chunk states it.",
                "supported_contribution": "Evaluation uses only synthetic benchmarks.",
                "supported_contribution_quote": sentence.group(0).strip() if sentence else None,
                "author_stated": True, "limitations": "None identified.",
            })
        return {
            "status": "FULLY_SUPPORTED", "complete_gap_assessment": "Supported by both papers.",
            "cross_paper_inference": "Consistent.", "evidence_summary_accuracy": "Accurate.",
            "research_opportunity": "Real-data evaluation.", "gap_origin": "author stated",
            "overgeneralization": "None.", "limitations": "Two papers only.",
            "counterevidence": "None.", "evidence": evidence, "contradictions": [],
        }

    def _critic(self, contents):
        gap_id = re.search(r"## (GAP-\d+):", contents).group(1)
        labels = json.loads(re.search(r"Gap evidence labels: (\[.*?\])", contents)
                            .group(1).replace("'", '"'))
        return _LLMGapVerdict(
            gap_id=gap_id, decision=self.critic_decision, origin_assessment="AUTHOR_STATED",
            origin_rationale="Verified author statements.", reasoning="Grounded in both quotes.",
            evidence_labels=labels[:2],
        )

    def _questions(self, contents):
        payload = json.loads(contents.split("\n", 1)[1])
        labels = [item["evidence_label"] for item in payload["evidence"]]
        return _LLMQuestionSet(research_questions=[{
            "question": "Does accuracy hold on real clinical records?",
            "rationale": "Synthetic-only evaluation.", "evidence_labels": labels,
        }], hypothesis={
            "alternative_hypothesis": "Accuracy drops on real clinical records.",
            "null_hypothesis": "Accuracy does not differ between synthetic and real records.",
            "rationale": "Both papers evaluate synthetically.", "evidence_labels": labels,
            "independent_variable": "data source", "dependent_variable": "accuracy",
            "comparator": "synthetic benchmark", "measurable_outcome": "AUROC difference",
        } if payload["hypothesis_evidence_sufficient"] else None)


# ---------------------------------------------------------------------------
# Evidence matrix
# ---------------------------------------------------------------------------

def test_matrix_clusters_only_cross_paper_patterns_with_exact_quotes_and_ids():
    comparison = matrix_agent().compare_many(corpus(), research_topic="clinical ML")
    matrix = comparison.evidence_matrix

    assert comparison.provenance.stage == "evidence_matrix"
    assert len(comparison.provenance.input_fingerprints["paper_analyses"]) == 64
    assert matrix.paper_ids == ["p1", "p2", "p3"] == comparison.paper_ids
    by_dimension = {cluster.dimension: cluster for cluster in matrix.clusters}
    assert by_dimension["limitations"].paper_ids == ["p1", "p2"]
    assert by_dimension["future_work"].paper_ids == ["p1", "p2"]
    # p3's single-paper limitation is never surfaced as a pattern.
    assert all("p3" not in cluster.paper_ids for cluster in matrix.clusters)
    texts = {chunk_id: text for chunk_id, text in CHUNKS.values()}
    owners = {chunk_id: paper for paper, (chunk_id, _) in CHUNKS.items()}
    for cluster in matrix.clusters:
        assert len(set(cluster.paper_ids)) >= 2
        for member in cluster.members:
            assert member.quote in texts[member.chunk_id]          # exact substring
            assert owners[member.chunk_id] == member.paper_id      # own paper only
    # The comparison artifact carries the same exact references.
    refs = {(ref.paper_id, chunk) for finding in comparison.common_limitations
            for ref in finding.evidence for chunk in ref.chunk_ids}
    assert refs == {("p1", "p1_c0"), ("p2", "p2_c0")}
    assert comparison.conflicting_findings == [] and comparison.dimension_summaries == []


def test_matrix_is_deterministic_and_makes_no_llm_call():
    agent = matrix_agent()
    assert not hasattr(agent, "client")
    first = agent.compare_many(corpus(), research_topic="t")
    second = matrix_agent().compare_many(corpus(), research_topic="t")
    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_missing_information_statements_are_not_matrix_cells():
    statements = split_statements(
        "Not available in the provided evidence. Uses ImageNet. "
        "Results on large graphs are not reported.", limit=4)
    assert statements == ["Uses ImageNet.", "Results on large graphs are not reported."]


def test_quote_candidates_are_exact_substrings_even_without_punctuation():
    text = " ".join(f"word{n}" for n in range(120))
    candidates = quote_candidates(text, max_chars=300)
    assert len(candidates) > 1 and all(candidate in text for candidate in candidates)


@pytest.mark.parametrize("mutate, message", [
    (lambda r: r[0].evidence.append(RetrievedChunk(chunk_id="x", paper_id="p2", text="t")),
     "multiple or unknown"),
    (lambda r: r.append(r[0]), "Duplicate paper ID"),
    (lambda r: r.__delitem__(slice(1, None)), "At least two"),
])
def test_matrix_rejects_foreign_duplicate_or_insufficient_inputs(mutate, message):
    results = corpus()
    mutate(results)
    with pytest.raises(ValueError, match=message):
        matrix_agent().compare_many(results)


def test_single_paper_patterns_produce_no_clusters_and_no_gap_request():
    results = [analysis("p1", limitations="The evaluation uses only synthetic benchmark data."),
               analysis("p3", limitations="Federated privacy training is expensive on GPU clusters.")]
    comparison = matrix_agent().compare_many(results)
    assert comparison.evidence_matrix.clusters == []

    exploding = SimpleNamespace(models=SimpleNamespace(
        generate_content=lambda **_: pytest.fail("No gap request may be sent.")))
    result = ResearchGapDetectionAgent(client=exploding).detect_gaps(comparison)
    assert result.gaps == []
    assert any("no gap detection request was sent" in c for c in result.caveats)


# ---------------------------------------------------------------------------
# Gap detection over the matrix
# ---------------------------------------------------------------------------

def synthetic_matrix_comparison(clusters=1, chunks_per_paper=1, papers=("p1", "p2", "p3")):
    """Matrix comparison built directly (for exact control of clusters/chunks)."""
    members_by_cluster = []
    for number in range(clusters):
        chunk = number % chunks_per_paper
        members_by_cluster.append([ClusterMember(
            paper_id=paper, chunk_id=f"{paper}_c{chunk}", statement="lorem " * 10,
            quote="lorem " * 30) for paper in papers])
    matrix = EvidenceMatrix(
        paper_ids=list(papers), dimensions=["limitations"],
        cells=[MatrixCell(paper_id=p, dimension="limitations", statement="s", chunk_id=f"{p}_c0",
                          quote="q", grounded=True) for p in papers],
        clusters=[EvidenceCluster(
            cluster_id=f"LIM-{n + 1}", dimension="limitations", paper_ids=list(papers),
            members=members, representative_statement="lorem", mean_similarity=0.8)
            for n, members in enumerate(members_by_cluster)],
        embedding_model="test", grounding_threshold=0.55, similarity_threshold=0.7,
        min_supporting_papers=2,
    )
    comparison = EvidenceMatrixComparisonAgent(embedding_model=object())._to_comparison(matrix)
    comparison.provenance = ArtifactProvenance(stage="evidence_matrix",
                                               input_fingerprints={"paper_analyses": "a" * 64})
    return comparison


class GapClient:
    def __init__(self, gaps):
        self.gaps, self.contents = gaps, []
        self.models = self

    def generate_content(self, *, model, contents, config=None):
        self.contents.append(contents)
        return SimpleNamespace(parsed=LLMGapDetection(gaps=self.gaps, caveats=[]))


def llm_gap(evidence, title="Gap"):
    return {"title": title, "description": "d", "gap_category": "Empirical Gap",
            "reasoning": "r", "evidence": evidence, "evidence_summary": "s",
            "gap_origin": "inferred_cross_paper", "confidence": "high",
            "confidence_reason": "c", "research_opportunity": "o"}


def test_matrix_gap_detection_defaults_to_three_gaps_and_one_bounded_request():
    # Five independent clusters, each on its own chunk, so five candidates
    # citing five distinct evidence sets: only the default cap of 3 applies.
    comparison = synthetic_matrix_comparison(clusters=5, chunks_per_paper=5)
    client = GapClient([llm_gap(
        [{"paper_label": "P1", "evidence_labels": [f"P1-E{n}"]},
         {"paper_label": "P2", "evidence_labels": [f"P2-E{n}"]}], f"Gap {n}")
        for n in range(1, 6)])
    result = ResearchGapDetectionAgent(client=client).detect_gaps(comparison, "topic")

    assert len(client.contents) == 1
    assert "Return at most 3 candidate research gaps" in client.contents[0]
    assert [gap.gap_id for gap in result.gaps] == ["GAP-001", "GAP-002", "GAP-003"]
    assert sum("Over the limit of 3" in item.reason for item in result.excluded_candidates) == 2
    assert result.gaps[0].supporting_evidence == [
        EvidenceRef(paper_id="p1", chunk_ids=["p1_c0"]),
        EvidenceRef(paper_id="p2", chunk_ids=["p2_c0"])]


def test_matrix_gap_detection_cannot_cite_unshown_or_invented_evidence():
    comparison = synthetic_matrix_comparison()
    invented = [{"paper_label": "P1", "evidence_labels": ["P1-E9"]},   # not shown
                {"paper_label": "P7", "evidence_labels": ["P7-E1"]},   # no such paper
                {"paper_label": "P2", "evidence_labels": ["P2-E1"]}]
    result = ResearchGapDetectionAgent(client=GapClient([llm_gap(invented)])).detect_gaps(comparison)
    assert result.gaps == []
    assert "at least 2 required" in result.excluded_candidates[0].reason


def test_matrix_gap_evidence_is_capped_per_paper_and_per_gap():
    comparison = synthetic_matrix_comparison(clusters=3, chunks_per_paper=3,
                                             papers=("p1", "p2", "p3", "p4"))
    evidence = [{"paper_label": f"P{n}", "evidence_labels": [f"P{n}-E1", f"P{n}-E2", f"P{n}-E3"]}
                for n in range(1, 5)]
    result = ResearchGapDetectionAgent(client=GapClient([llm_gap(evidence)])).detect_gaps(comparison)
    gap = result.gaps[0]
    assert len(gap.supporting_chunk_ids) == 6                     # per-gap cap
    assert all(len(ref.chunk_ids) <= 2 for ref in gap.supporting_evidence)
    assert {ref.paper_id for ref in gap.supporting_evidence} == {"p1", "p2", "p3", "p4"}
    assert any("were dropped" in caveat for caveat in result.caveats)


def test_matrix_gap_request_is_trimmed_to_fit_the_token_budget(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    comparison = synthetic_matrix_comparison(clusters=40)
    client = RecordingClient()
    with pytest.raises(Exception):
        ResearchGapDetectionAgent(model=GPT_120B, client=client).detect_gaps(comparison)
    call = client.calls[0]
    assert call.fits and call.output_tokens == 3072

    agent = ResearchGapDetectionAgent(model=GPT_120B, client=RecordingClient())
    shown, _, contents = agent._fit_matrix_request(comparison, None)
    assert 0 < len(shown) < 40 and f"of 40 shown" in contents


def test_gap_request_that_cannot_fit_is_refused_before_sending(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_TPM_LIMIT", "4000")    # 3,600 budget < 3,072 output + input
    client = RecordingClient()
    with pytest.raises(GapDetectionTooLargeError):
        ResearchGapDetectionAgent(model=GPT_120B, client=client).detect_gaps(
            synthetic_matrix_comparison())
    assert client.calls == []


# ---------------------------------------------------------------------------
# Evidence validation: deterministic checks, then one 20B request per gap
# ---------------------------------------------------------------------------

def detected_gaps(comparison):
    client = ScriptedLLM()
    return ResearchGapDetectionAgent(client=client).detect_gaps(comparison, "topic")


def test_validation_defaults_to_one_20b_request_per_gap_and_keeps_exact_records():
    assert resolve_model("VALIDATION") == GPT_20B
    comparison = matrix_agent().compare_many(corpus())
    gaps = detected_gaps(comparison)
    client = ScriptedLLM()
    result = EvidenceValidationAgent(vector_store=Store(), client=client).validate(
        gaps, evidence_matrix=comparison.evidence_matrix)

    assert client.calls == ["validation"] * len(gaps.gaps)       # no per-chunk requests
    item = result.gap_validations[0]
    assert result.validation_mode == "joint" and item.deterministic_check.passed
    assert item.status is GapValidationStatus.FULLY_SUPPORTED
    assert [(r.paper_id, r.chunk_id) for r in item.evidence_validations] == [
        ("p1", "p1_c0"), ("p2", "p2_c0")]
    for record in item.evidence_validations:
        assert record.supported_contribution_quote in dict(CHUNKS.values())[record.chunk_id]
        assert record.gap_origin_assessment == "AUTHOR_STATED"
        assert json.loads(record.findings)["author_stated"] is True
    assert "Author-stated in" in item.joint_assessment.gap_origin


@pytest.mark.parametrize("override, failure", [
    ({"p2_c0": ("p2", "Different stored text with no matrix quote at all.")}, "verbatim"),
    ({"p2_c0": ("p3", CHUNKS["p2"][1])}, "ownership mismatch"),
    ({"p2_c0": None}, "not found"),
])
def test_deterministic_failures_send_no_request_for_any_gap(override, failure):
    comparison = matrix_agent().compare_many(corpus())
    gaps = detected_gaps(comparison)
    store = Store()
    for chunk_id, record in override.items():
        if record is None:
            del store.records[chunk_id]
        else:
            store.records[chunk_id] = record
    client = ScriptedLLM()
    result = EvidenceValidationAgent(vector_store=store, client=client).validate(
        gaps, evidence_matrix=comparison.evidence_matrix)

    assert client.calls == []
    item = result.gap_validations[0]
    assert item.status is GapValidationStatus.VALIDATION_INCOMPLETE
    assert not item.deterministic_check.passed
    assert any(failure in message for message in item.deterministic_check.failures)
    assert result.retrieval_failures


def test_unverified_author_statement_is_not_author_stated_and_origin_is_flagged():
    comparison = matrix_agent().compare_many(corpus())
    gaps = detected_gaps(comparison)
    assert gaps.gaps[0].gap_origin is GapOrigin.EXPLICIT
    client = ScriptedLLM()
    original = client._validation

    def paraphrased_quotes(contents):
        payload = original(contents)
        for item in payload["evidence"]:
            item["supported_contribution_quote"] = "A paraphrase that is not in the chunk."
        return payload

    client._validation = paraphrased_quotes
    result = EvidenceValidationAgent(vector_store=Store(), client=client).validate(
        gaps, evidence_matrix=comparison.evidence_matrix)
    item = result.gap_validations[0]
    for record in item.evidence_validations:
        assert record.supported_contribution is None and record.supported_contribution_quote is None
        assert record.gap_origin_assessment == "INFERRED"
    assert "treat the gap as inferred" in item.joint_assessment.gap_origin


@pytest.mark.parametrize("edit", ["unknown_label", "missing_label", "duplicate_label"])
def test_label_errors_in_the_joint_response_leave_the_gap_incomplete(edit):
    comparison = matrix_agent().compare_many(corpus())
    gaps = detected_gaps(comparison)
    client = ScriptedLLM()
    original = client._validation

    def broken(contents):
        payload = original(contents)
        if edit == "unknown_label":
            payload["evidence"][0]["evidence_label"] = "P9-E9"
        elif edit == "missing_label":
            payload["evidence"].pop()
        else:
            payload["evidence"].append(dict(payload["evidence"][0]))
        return payload

    client._validation = broken
    result = EvidenceValidationAgent(vector_store=Store(), client=client).validate(
        gaps, evidence_matrix=comparison.evidence_matrix)
    assert result.gap_validations[0].status is GapValidationStatus.VALIDATION_INCOMPLETE
    assert result.semantic_failures


def test_validation_schema_has_no_per_chunk_requests_on_the_default_path():
    agent = EvidenceValidationAgent(vector_store=Store(), client=object())
    assert agent.mode == "joint"
    assert "evidence" in _GapEvidenceAssessment.model_fields
    with pytest.raises(ValueError, match="mode"):
        EvidenceValidationAgent(vector_store=Store(), client=object(), mode="batch")


# ---------------------------------------------------------------------------
# Critic: one ACCEPT/REJECT request per validated gap
# ---------------------------------------------------------------------------

def validated_chain():
    comparison = matrix_agent().compare_many(corpus())
    gaps = detected_gaps(comparison)
    validation = EvidenceValidationAgent(vector_store=Store(), client=ScriptedLLM()).validate(
        gaps, evidence_matrix=comparison.evidence_matrix)
    return comparison, gaps, validation


def test_critic_verdict_schema_has_no_revise_option():
    assert {member.value for member in _BinaryDecision} == {"ACCEPT", "REJECT"}
    schema = json.dumps(_LLMGapVerdict.model_json_schema())
    assert "REVISE" not in schema
    assert CriticAgent(client=object()).review_mode == "verdict"


@pytest.mark.parametrize("decision", ["ACCEPT", "REJECT"])
def test_critic_makes_one_request_per_gap_and_never_rewrites(decision):
    comparison, gaps, validation = validated_chain()
    client = ScriptedLLM(critic_decision=decision)
    result = CriticAgent(client=client).critique(comparison, gaps, validation)

    assert client.calls == ["critic"] * len(gaps.gaps)
    review = result.gap_reviews[0]
    assert result.review_status is CriticReviewStatus.COMPLETE and result.review_mode == "verdict"
    assert review.decision is CriticDecision(decision)
    assert review.revision_suggestions == []
    assert {ref.paper_id for ref in review.cited_evidence} == {"p1", "p2"}


def test_unsupported_gap_is_rejected_without_a_request():
    comparison, gaps, validation = validated_chain()
    item = validation.gap_validations[0]
    validation.gap_validations[0] = item.model_copy(update={
        "status": GapValidationStatus.UNSUPPORTED,
        "joint_assessment": item.joint_assessment.model_copy(
            update={"status": GapSupportStatus.UNSUPPORTED}),
    })
    client = ScriptedLLM()
    result = CriticAgent(client=client).critique(comparison, gaps, validation)
    assert client.calls == []
    assert result.gap_reviews[0].decision is CriticDecision.REJECT
    assert result.review_status is CriticReviewStatus.COMPLETE


@pytest.mark.parametrize("labels, message", [
    ([], "cited no shown evidence"),
    (["P3-E1"], "not shown"),
])
def test_ungrounded_accept_is_incomplete_not_accepted(labels, message):
    comparison, gaps, validation = validated_chain()
    client = ScriptedLLM()
    original = client._critic
    client._critic = lambda contents: original(contents).model_copy(update={"evidence_labels": labels})
    result = CriticAgent(client=client).critique(comparison, gaps, validation)
    review = result.gap_reviews[0]
    assert review.review_status is CriticReviewStatus.INCOMPLETE
    assert review.decision is not CriticDecision.ACCEPT
    assert any(message in error for error in result.errors)


def test_critic_request_fits_the_budget_and_carries_only_the_gaps_clusters(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    comparison, gaps, validation = validated_chain()
    client = RecordingClient()
    CriticAgent(model=GPT_120B, client=client).critique(comparison, gaps, validation)
    assert len(client.calls) == len(gaps.gaps)
    assert all(call.fits and call.output_tokens == 2048 for call in client.calls)


# ---------------------------------------------------------------------------
# Complete topic-to-hypothesis workflow: connected, resumable, reported
# ---------------------------------------------------------------------------

def papers():
    return [PaperRecord(title=f"Paper {pid}", authors=[], year=2024, doi=None, abstract=None,
                        source="mock", source_id=pid, landing_url=None, pdf_url=None)
            for pid in ("p1", "p2", "p3")]


def phase2_agents(client):
    retrieval = Mock()
    retrieval.search.return_value = (search_plan(), papers())
    by_id = {result.evidence[0].paper_id: result for result in corpus()}
    pipeline = Mock()
    pipeline._get_paper_id.side_effect = lambda paper: paper.source_id
    pipeline._resolve_pdf_path.side_effect = lambda paper, pid, paths: "x.pdf"
    pipeline.analyze_paper.side_effect = lambda paper, **_: by_id[paper.source_id]
    return WorkflowAgents(
        retrieval, pipeline, matrix_agent(),
        ResearchGapDetectionAgent(client=client),
        EvidenceValidationAgent(vector_store=Store(), client=client),
        CriticAgent(client=client), ResearchQuestionAgent(client=client),
    )


def test_topic_to_hypothesis_runs_end_to_end_with_the_simplified_calls(tmp_path):
    client = ScriptedLLM()
    outcome = run_topic(TopicRunConfig(research_topic="clinical ML", max_papers=6),
                        run_dir=tmp_path / "run", agents=phase2_agents(client))
    state = outcome.state

    assert state["workflow_status"] == "COMPLETE"
    assert client.calls == ["gap", "validation", "critic", "questions"]
    assert state["max_revisions"] == 0 and state["revision_count"] == 0
    assert state["comparison"].evidence_matrix is not None
    gap_result = state["research_questions"].gap_results[0]
    assert gap_result.hypothesis is not None
    assert {ref.paper_id for ref in gap_result.hypothesis.source_evidence} == {"p1", "p2"}

    report = outcome.report_path.read_text(encoding="utf-8")
    for heading in ("## Retrieved papers", "## Evidence matrix", "## Candidate gaps",
                    "## Evidence validation", "## Critic decisions",
                    "## Research questions and hypotheses", "## Citations", "## Limitations"):
        assert heading in report
    assert NOVELTY_SCOPE in report and "**H1:**" in report
    assert "`p1`" in report and "chunk `p1_c0`" in report


def test_failed_critic_stops_and_resume_retries_only_the_critic(tmp_path):
    config = TopicRunConfig(research_topic="clinical ML", max_papers=6)
    failing = ScriptedLLM(fail_tasks={"critic"})
    first = run_topic(config, run_dir=tmp_path / "run", agents=phase2_agents(failing))
    assert failing.calls == ["gap", "validation", "critic"]
    assert first.state["workflow_status"] == "STOPPED"
    assert RunCheckpoint(tmp_path / "run").load()[1] == "stop_incomplete_critic"

    working = ScriptedLLM()
    outcome = run_topic(config, run_dir=tmp_path / "run", agents=phase2_agents(working))
    assert outcome.resumed_from == "stop_incomplete_critic"
    assert working.calls == ["critic", "questions"]          # no gap/validation re-spend
    assert outcome.state["workflow_status"] == "COMPLETE"


def test_interrupted_gap_detection_resumes_after_the_saved_matrix(tmp_path):
    config = TopicRunConfig(research_topic="clinical ML", max_papers=6)
    failing = ScriptedLLM(fail_tasks={"gap"})
    with pytest.raises(RunInterrupted):
        run_topic(config, run_dir=tmp_path / "run", agents=phase2_agents(failing))
    state, stage = RunCheckpoint(tmp_path / "run").load()
    assert stage == "compare" and state["comparison"].evidence_matrix is not None

    working = ScriptedLLM()
    agents = phase2_agents(working)
    agents.comparison = Mock(wraps=agents.comparison)
    outcome = run_topic(config, run_dir=tmp_path / "run", agents=agents)
    assert outcome.resumed_from == "compare"
    agents.comparison.compare_many.assert_not_called()       # matrix reused, lineage checked
    agents.paper_analysis.analyze_paper.assert_not_called()
    assert working.calls == ["gap", "validation", "critic", "questions"]
    assert outcome.state["workflow_status"] == "COMPLETE"


def test_all_rejected_reports_insufficient_evidence_instead_of_a_hypothesis(tmp_path):
    client = ScriptedLLM(critic_decision="REJECT")
    outcome = run_topic(TopicRunConfig(research_topic="clinical ML"),
                        run_dir=tmp_path / "run", agents=phase2_agents(client))
    assert outcome.state["workflow_status"] == "STOPPED"
    assert "questions" not in client.calls
    report = outcome.report_path.read_text(encoding="utf-8")
    assert INSUFFICIENT_EVIDENCE in report and "**H1:**" not in report


def test_report_renders_a_run_that_stopped_before_comparison():
    text = render_report({"research_topic": "t", "workflow_status": "STOPPED",
                          "stop_reason": "Insufficient papers were retrieved.", "papers": []})
    assert "Insufficient papers were retrieved." in text
    assert "The run stopped before papers were compared." in text
    assert NOVELTY_SCOPE in text


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

@pytest.fixture
def groq_defaults(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")


def test_six_paper_dry_run_fits_8k_tpm_and_the_daily_budget(groq_defaults):
    report = estimate_run({"research_topic": "topic"}, assumptions=DryRunAssumptions(papers=6))
    stages = {stage.stage: stage for stage in report.stages}

    assert report.all_fit                                         # no oversized request
    assert all(s.largest_request <= 7200 for s in report.stages)
    assert stages["evidence matrix"].calls == 0                   # no LLM comparison
    assert stages["gap detection"].calls == 1
    assert stages["evidence validation"].calls == 3               # one per gap, no per-chunk
    assert stages["critic"].calls == 3                            # one per gap, no split
    assert "revision rounds (worst case)" not in stages
    for name in ("gap detection", "evidence validation", "critic", "research questions"):
        assert stages[name].output_budget_per_call is not None
        assert stages[name].daily_share is not None
    totals = {item.model: item for item in report.models}
    assert totals[GPT_120B].calls == 7
    assert totals[GPT_120B].tokens < 0.25 * totals[GPT_120B].daily_token_limit
    assert report.within_daily_budget
    text = format_report(report)
    assert "no request was sent" in text and "within daily budget" in text
    assert "7,200 tokens" in text


def test_dry_run_uses_a_configured_daily_limit(groq_defaults, monkeypatch):
    monkeypatch.setenv("GROQ_TPD_LIMIT_OPENAI_GPT_OSS_120B", "20000")
    report = estimate_run({"research_topic": "topic"})
    totals = {item.model: item for item in report.models}[GPT_120B]
    assert totals.daily_token_limit == 20000 and not totals.daily_limit_assumed
    assert totals.within_daily_budget is False and not report.within_daily_budget
    assert "OVER daily budget" in format_report(report)


def test_dry_run_reports_the_legacy_comparison_only_when_selected(groq_defaults, monkeypatch):
    monkeypatch.setenv("COMPARISON_MODE", "llm")
    names = [stage.stage for stage in estimate_run({"research_topic": "topic"}).stages]
    assert "cross-paper comparison (legacy)" in names and "evidence matrix" not in names


# ---------------------------------------------------------------------------
# Backward compatibility of saved artifacts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture", sorted(p.name for p in LANGGRAPH_RUN_FIXTURES.glob("*.json")))
def test_saved_artifacts_round_trip_with_unchanged_fingerprints(fixture):
    saved = json.loads((LANGGRAPH_RUN_FIXTURES / fixture).read_text(encoding="utf-8"))
    for key, model in (("comparison", MultiPaperComparison),
                       ("gap_detection", ResearchGapDetectionResult),
                       ("evidence_validation", EvidenceValidationResult),
                       ("critic_result", CriticResult)):
        if saved.get(key) is None:
            continue
        loaded = model.model_validate(saved[key])
        # New optional fields are absent from legacy artifacts' serialized form.
        assert loaded.model_dump(mode="json") == saved[key]
    critic_fingerprints = saved.get("critic_input_fingerprints") or {}
    if saved.get("comparison") and critic_fingerprints.get("comparison"):
        comparison = MultiPaperComparison.model_validate(saved["comparison"])
        assert _model_fingerprint(comparison) == critic_fingerprints["comparison"]


def test_new_fields_are_serialized_only_when_set():
    comparison = matrix_agent().compare_many(corpus())
    assert "evidence_matrix" in comparison.model_dump(mode="json")
    legacy = comparison.model_copy(update={"evidence_matrix": None})
    assert "evidence_matrix" not in legacy.model_dump(mode="json")
    assert fingerprint_json(legacy.model_dump(mode="json")) == fingerprint_json(
        MultiPaperComparison.model_validate(legacy.model_dump(mode="json")).model_dump(mode="json"))


# ---------------------------------------------------------------------------
# Default wiring: no LLM comparison, per-chunk validation or split critic
# ---------------------------------------------------------------------------

def _patched_defaults(monkeypatch):
    import src.workflow.langgraph_app as workflow

    real_validation = workflow.EvidenceValidationAgent
    legacy_comparison = object()
    monkeypatch.setattr(workflow, "PlannerRetrievalAgent", lambda: object())
    monkeypatch.setattr(workflow, "PaperAnalysisPipeline",
                        lambda **_: SimpleNamespace(comparison_agent=legacy_comparison))
    monkeypatch.setattr(workflow, "EvidenceValidationAgent",
                        lambda **kw: real_validation(vector_store=object(), **kw))
    return workflow, legacy_comparison


def test_default_workflow_agents_use_only_the_simplified_path(monkeypatch):
    workflow, _ = _patched_defaults(monkeypatch)
    agents = workflow.default_workflow_agents()
    assert isinstance(agents.comparison, EvidenceMatrixComparisonAgent)
    assert agents.evidence_validation.mode == "joint"
    assert agents.critic.review_mode == "verdict"
    assert agents.gap_detection.max_gaps == 3
    assert workflow.DEFAULT_MAX_REVISIONS == 0 and TopicRunConfig("t").max_revisions == 0


def test_legacy_llm_comparison_is_available_behind_the_flag(monkeypatch):
    workflow, legacy = _patched_defaults(monkeypatch)
    monkeypatch.setenv("COMPARISON_MODE", "llm")
    assert workflow.default_workflow_agents().comparison is legacy
    monkeypatch.setenv("COMPARISON_MODE", "other")
    with pytest.raises(ValueError, match="COMPARISON_MODE"):
        workflow.default_workflow_agents()
