import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from src.evaluation.baselines import (
    ConventionalRAGBaseline,
    EvaluationCorpus,
    SingleLLMBaseline,
    build_read_only_rag_retriever,
    load_frozen_evaluation_corpus,
)
from src.evaluation.models import OutputCompletionStatus
from src.evaluation.runner import DatasetHeader, DatasetManifest, DatasetPaper
from src.evaluation.runner import create_imported_run
from src.rag.models import RetrievedChunk
from src.processing import TextChunker


PROMPT_ID = "PROMPT-001"
PAPER_IDS = [f"https://openalex.org/W{i}" for i in range(6)]


def corpus():
    prompts = [
        {"prompt_id": f"PROMPT-00{i}", "text": f"Prompt {i}", "status": "FINALIZED_SHARED_PROMPT"}
        for i in range(1, 4)
    ]
    header = DatasetHeader(
        record_type="dataset", dataset_version="evaluation-corpus-v1",
        curation_status="FINALIZED_EVALUATION_CONFIGURATION",
        research_prompts=prompts,
        evaluation_configuration={
            "purpose": "Evaluation/testing only; does not change the main project scope or objectives.",
            "frozen": True, "included_paper_count": 6,
            "shared_prompts":[p["prompt_id"] for p in prompts],
        },
    )
    papers = [DatasetPaper(
        record_type="paper", paper_id=paper_id,
        external_ids={"doi": None, "openalex": f"W{i}"},
        title=f"Paper {i}", authors=[f"Author {i}"], year=2024,
        relative_path=f"data/papers/paper-{i}.pdf", sha256="a" * 64,
        research_prompt_ids=[p["prompt_id"] for p in prompts], split=None,
        inclusion_decision={"decision":"INCLUDE_IN_PRIMARY_EVALUATION","status":"FINALIZED"},
        curation_status="FINALIZED_EVALUATION_CONFIGURATION",
    ) for i, paper_id in enumerate(PAPER_IDS)]
    papers.extend(DatasetPaper(
        record_type="paper", paper_id=f"excluded-{i}",
        relative_path=f"excluded-{i}.pdf", sha256="b" * 64,
        inclusion_decision={"decision":"EXCLUDE_FROM_PRIMARY_EVALUATION","status":"FINALIZED"},
        curation_status="FINALIZED_EVALUATION_CONFIGURATION",
    ) for i in range(2))
    return EvaluationCorpus(
        manifest=DatasetManifest(header=header, papers=papers, manifest_hash="c" * 64),
        papers=papers[:6], prompts={p["prompt_id"]: p for p in prompts},
    )


def generated_gap(paper_id, chunk_id, quote="exact source quote"):
    return {"gaps": [{
        "title": "Candidate method gap",
        "description": "The evaluation leaves an evidence-bounded comparison open.",
        "evidence": [{"paper_id": paper_id, "chunk_id": chunk_id, "quote": quote}],
    }]}


class RecordingGenerator:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate(self, prompt, response_schema):
        self.calls.append((prompt, response_schema))
        return self.response


def test_single_llm_makes_one_call_over_full_paper_chunks_and_preserves_exact_ids(tmp_path):
    data = corpus()
    extractor = Mock()
    extractor.extract_text.return_value = "exact source quote plus context"
    generator = RecordingGenerator(generated_gap(PAPER_IDS[0], f"{PAPER_IDS[0]}_chunk_1"))
    baseline = SingleLLMBaseline(generator, extractor=extractor, chunker=TextChunker())

    output = baseline.generate(data, PROMPT_ID, tmp_path)

    assert len(generator.calls) == 1
    prompt, schema = generator.calls[0]
    assert schema.__name__ == "_GeneratedGapSet"
    assert all(paper_id in prompt for paper_id in PAPER_IDS)
    assert extractor.extract_text.call_count == 6
    assert output.paper_ids == PAPER_IDS
    assert output.research_prompt_id == PROMPT_ID
    assert output.candidates[0].citations[0].paper_id == PAPER_IDS[0]
    assert output.candidates[0].citations[0].chunk_id == f"{PAPER_IDS[0]}_chunk_1"
    assert output.candidates[0].citations[0].quote == "exact source quote"
    assert output.completion_status is OutputCompletionStatus.COMPLETE


def test_conventional_rag_retrieves_per_paper_then_generates_once():
    data = corpus()
    retriever = Mock()
    retriever.retrieve.side_effect = lambda query, top_k, paper_id: [RetrievedChunk(
        chunk_id=f"{paper_id}_chunk_1", paper_id=paper_id,
        text="exact source quote plus context",
    )]
    generator = RecordingGenerator(generated_gap(PAPER_IDS[2], f"{PAPER_IDS[2]}_chunk_1"))
    baseline = ConventionalRAGBaseline(generator, retriever, top_k=4)

    output = baseline.generate(data, PROMPT_ID)

    assert retriever.retrieve.call_count == 6
    assert all(call.kwargs["query"] == data.prompts[PROMPT_ID]["text"] for call in retriever.retrieve.call_args_list)
    assert all(call.kwargs["top_k"] == 4 for call in retriever.retrieve.call_args_list)
    assert {call.kwargs["paper_id"] for call in retriever.retrieve.call_args_list} == set(PAPER_IDS)
    assert len(generator.calls) == 1
    assert output.paper_ids == PAPER_IDS
    assert output.candidates[0].citations[0].paper_id == PAPER_IDS[2]
    assert output.candidates[0].citations[0].source_provenance == "conventional_rag_retrieved_chunk"


def test_unknown_or_wrong_owner_evidence_is_dropped_and_marks_output_incomplete(tmp_path):
    data = corpus()
    extractor = Mock()
    extractor.extract_text.return_value = "source text"
    generator = RecordingGenerator(generated_gap("made-up-paper", "made-up-chunk"))
    output = SingleLLMBaseline(generator, extractor=extractor).generate(data, PROMPT_ID, tmp_path)
    assert output.completion_status is OutputCompletionStatus.INCOMPLETE
    assert output.candidates == []
    assert any("unknown or mismatched" in warning for warning in output.warnings)


def test_nonverbatim_model_quote_is_removed_and_output_is_marked_incomplete(tmp_path):
    data = corpus()
    extractor = Mock()
    extractor.extract_text.return_value = "source text without that quote"
    generator = RecordingGenerator(generated_gap(
        PAPER_IDS[0], f"{PAPER_IDS[0]}_chunk_1", quote="invented quote",
    ))
    output = SingleLLMBaseline(generator, extractor=extractor).generate(data, PROMPT_ID, tmp_path)
    assert output.completion_status is OutputCompletionStatus.INCOMPLETE
    assert output.candidates[0].citations[0].quote is None
    assert any("non-verbatim quote" in warning for warning in output.warnings)


def test_read_only_chroma_builder_gets_existing_collection_without_create(monkeypatch, tmp_path):
    import src.evaluation.baselines as baselines

    class FakeEmbedder:
        def __init__(self, model_name, dimension):
            self.model_name = model_name
            self.dimension = dimension

        def embed_query(self, query):
            return [0.0] * self.dimension

    class FakeCollection:
        metadata = {"embedding_model":"model-x", "embedding_dimension":2}

        def query(self, **kwargs):
            self.query_kwargs = kwargs
            return {
                "ids":[["id-1"]], "documents":[["retrieved text"]],
                "metadatas":[[{"paper_id":PAPER_IDS[0],"chunk_id":"id-1"}]],
                "distances":[[0.1]],
            }

    collection = FakeCollection()
    class FakeClient:
        def get_collection(self, name):
            self.name = name
            return collection

    client = FakeClient()
    monkeypatch.setattr(baselines, "LocalEmbeddingModel", FakeEmbedder)
    monkeypatch.setitem(__import__("sys").modules, "chromadb", SimpleNamespace(PersistentClient=lambda path: client))

    retriever = build_read_only_rag_retriever(
        tmp_path, embedding_model="model-x", embedding_dimension=2,
    )
    chunks = retriever.retrieve("query", top_k=3, paper_id=PAPER_IDS[0])

    assert client.name == "research_papers__model_x"
    assert chunks == [RetrievedChunk(chunk_id="id-1", paper_id=PAPER_IDS[0], text="retrieved text", distance=0.1)]
    assert collection.query_kwargs["where"] == {"paper_id": PAPER_IDS[0]}


def test_real_manifest_exposes_the_same_finalized_six_papers_and_three_prompts():
    root = Path(__file__).resolve().parents[1]
    loaded = load_frozen_evaluation_corpus(root / "data" / "evaluation" / "dataset_manifest.jsonl")
    assert len(loaded.papers) == 6
    assert list(loaded.prompts) == ["PROMPT-001", "PROMPT-002", "PROMPT-003"]


def test_baseline_output_imports_through_the_existing_evaluation_runner(tmp_path):
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "data" / "evaluation" / "dataset_manifest.jsonl"
    loaded = load_frozen_evaluation_corpus(manifest_path)
    first = loaded.papers[0]
    extractor = Mock()
    extractor.extract_text.return_value = "exact source quote plus context"
    generator = RecordingGenerator(generated_gap(first.paper_id, f"{first.paper_id}_chunk_1"))
    output = SingleLLMBaseline(generator, extractor=extractor).generate(loaded, PROMPT_ID, root)
    owners = {citation.chunk_id: citation.paper_id
              for candidate in output.candidates for citation in candidate.citations}
    normalized_path = tmp_path / "normalized.json"
    owners_path = tmp_path / "owners.json"
    prompt_path = tmp_path / "prompt.txt"
    normalized_path.write_text(output.model_dump_json(), encoding="utf-8")
    owners_path.write_text(json.dumps(owners), encoding="utf-8")
    prompt_path.write_text(loaded.prompts[PROMPT_ID]["text"], encoding="utf-8")

    run_dir = create_imported_run(
        normalized_path, manifest_path, tmp_path / "runs",
        system_name="single_llm", prompt_path=prompt_path,
        citation_index_path=owners_path,
        research_prompt_id=PROMPT_ID,
        canonical_prompt_text=loaded.prompts[PROMPT_ID]["text"],
    )

    saved = json.loads((run_dir / "outputs.json").read_text(encoding="utf-8"))
    assert saved["system_name"] == "single_llm"
    assert saved["paper_ids"] == [paper.paper_id for paper in loaded.papers]
    assert saved["research_prompt_id"] == PROMPT_ID
