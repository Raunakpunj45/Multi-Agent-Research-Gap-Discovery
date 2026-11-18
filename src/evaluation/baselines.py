"""Independent single-LLM and conventional-RAG evaluation baselines.

Both systems make one structured generation call for a finalized prompt. They
share the frozen manifest papers and prompt text, but have independent context
construction and use no production agents, critic, or LangGraph.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from src.embeddings import LocalEmbeddingModel
from src.evaluation.adapters import AdapterError, normalize_common_output
from src.evaluation.models import (
    CandidateGap,
    Citation,
    EvaluationSystemOutput,
    NoveltyAssessment,
    NoveltyIndication,
    OutputCompletionStatus,
)
from src.evaluation.runner import DatasetManifest, DatasetPaper, load_dataset_manifest
from src.processing import PDFExtractor, TextChunker, TextCleaner
from src.rag import RAGRetriever, RetrievedChunk
from src.vectorstore.chroma_store import collection_name_for

load_dotenv()


BASELINE_INSTRUCTIONS = (
    "Identify evidence-grounded candidate research gaps using only the supplied "
    "evaluation corpus and the research prompt. Do not claim global novelty. "
    "For each candidate, provide concise title and description and cite only "
    "paper_id/chunk_id pairs present in the supplied context. A quote, if given, "
    "must be copied verbatim from that chunk. Do not invent papers, IDs, quotes, "
    "results, annotations, or validation labels. Return an empty gaps list when "
    "the context does not support a candidate. Do not generate research questions "
    "or hypotheses."
)


class GapEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    paper_id: str
    chunk_id: str
    quote: str | None = None


class _GeneratedGap(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str
    description: str
    evidence: list[GapEvidence] = Field(default_factory=list)


class _GeneratedGapSet(BaseModel):
    model_config = ConfigDict(extra="forbid")
    gaps: list[_GeneratedGap] = Field(default_factory=list)


class StructuredGenerator(Protocol):
    def generate(self, prompt: str, response_schema: type[BaseModel]) -> BaseModel: ...


def _gemini_compatible_schema(model: type[BaseModel]) -> dict:
    """Return JSON Schema without keywords unsupported by Gemini response_schema."""
    schema = model.model_json_schema()

    def remove_unsupported(value):
        if isinstance(value, dict):
            value.pop("additionalProperties", None)
            value.pop("additional_properties", None)
            for nested in value.values():
                remove_unsupported(nested)
        elif isinstance(value, list):
            for nested in value:
                remove_unsupported(nested)

    remove_unsupported(schema)
    return schema


class GeminiStructuredGenerator:
    """Structured baseline generator for the configured LLM provider.

    The historical class name is kept for compatibility; the client comes from
    ``LLM_PROVIDER`` (Groq by default, Gemini optional). Construction makes no
    request.
    """

    def __init__(self, model: str | None = None, client=None):
        from src.agents.llm_provider import create_llm_client, resolve_model

        self.model = model or resolve_model("BASELINE")
        self._client = client
        if self._client is None:
            # The standalone structured baseline belongs to the analysis stage.
            self._client = create_llm_client("ANALYSIS")

    def generate(self, prompt: str, response_schema: type[BaseModel]) -> BaseModel:
        from src.agents.output_budget import generation_config

        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=generation_config(
                "baseline",
                temperature=0,
                response_mime_type="application/json",
                response_schema=_gemini_compatible_schema(response_schema),
            ),
        )
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, response_schema):
            return parsed
        if parsed is not None:
            return response_schema.model_validate(parsed, strict=True)
        text = getattr(response, "text", None)
        if not text:
            raise ValueError("LLM returned no structured baseline output.")
        return response_schema.model_validate_json(text, strict=True)


@dataclass(frozen=True)
class EvaluationCorpus:
    manifest: DatasetManifest
    papers: list[DatasetPaper]
    prompts: dict[str, dict]


def load_frozen_evaluation_corpus(manifest_path: Path, *,
                                  repo_root: Path | None = None) -> EvaluationCorpus:
    """Load only the six finalized inclusion records and their shared prompts.

    ``repo_root`` is the directory manifest paper paths resolve against; it
    defaults to this repository.
    """
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    manifest = load_dataset_manifest(manifest_path, root)
    configuration = manifest.header.evaluation_configuration or {}
    if configuration.get("frozen") is not True:
        raise ValueError("Evaluation manifest does not declare a frozen corpus.")
    if configuration.get("purpose") != "Evaluation/testing only; does not change the main project scope or objectives.":
        raise ValueError("Evaluation manifest is missing the evaluation-only scope statement.")

    papers = [
        paper for paper in manifest.papers
        if (paper.inclusion_decision or {}).get("decision") == "INCLUDE_IN_PRIMARY_EVALUATION"
    ]
    if len(papers) != configuration.get("included_paper_count") or len(papers) != 6:
        raise ValueError("Expected the finalized six-paper primary evaluation corpus.")

    raw_prompts = manifest.header.research_prompts
    prompts = {
        item["prompt_id"]: item for item in raw_prompts
        if item.get("status") == "FINALIZED_SHARED_PROMPT"
    }
    shared_ids = configuration.get("shared_prompts", [])
    if (len(prompts) != 3 or list(prompts) != shared_ids
            or any((paper.research_prompt_ids or []) != shared_ids for paper in papers)):
        raise ValueError("Evaluation papers and systems must share the same three finalized prompts.")
    return EvaluationCorpus(manifest=manifest, papers=papers, prompts=prompts)


class SingleLLMBaseline:
    """One direct LLM call over all supplied full-paper chunks per prompt."""

    system_name = "single_llm"

    def __init__(
        self,
        generator: StructuredGenerator,
        *,
        extractor=None,
        cleaner=None,
        chunker=None,
    ):
        self.generator = generator
        self.extractor = extractor or PDFExtractor()
        self.cleaner = cleaner or TextCleaner()
        self.chunker = chunker or TextChunker()

    def generate(
        self,
        corpus: EvaluationCorpus,
        prompt_id: str,
        repo_root: Path,
    ) -> EvaluationSystemOutput:
        prompt = _get_prompt(corpus, prompt_id)
        context: list[dict[str, str]] = []
        owners: dict[str, str] = {}
        chunk_texts: dict[str, str] = {}
        for paper in corpus.papers:
            raw_text = self.extractor.extract_text(repo_root / paper.relative_path)
            cleaned = self.cleaner.clean(raw_text)
            chunks = self.chunker.chunk(cleaned, paper.paper_id)
            for chunk in chunks:
                context.append({
                    "paper_id": paper.paper_id,
                    "title": paper.title or "",
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                })
                owners[chunk.chunk_id] = paper.paper_id
                chunk_texts[chunk.chunk_id] = chunk.text
        if not context:
            return _empty_output(corpus, prompt_id, self.system_name, "No paper chunks were available.")
        response = self.generator.generate(
            _generation_prompt(prompt, context), _GeneratedGapSet
        )
        return _normalize_generated(
            response, corpus, prompt_id, self.system_name, owners, chunk_texts,
            input_chunk_count=len(context),
        )


class ConventionalRAGBaseline:
    """Per-paper Chroma retrieval followed by one direct generation call."""

    system_name = "conventional_rag"

    def __init__(self, generator: StructuredGenerator, retriever, top_k: int = 8):
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        self.generator = generator
        self.retriever = retriever
        self.top_k = top_k

    def generate(
        self,
        corpus: EvaluationCorpus,
        prompt_id: str,
        repo_root: Path | None = None,
    ) -> EvaluationSystemOutput:
        del repo_root  # The RAG baseline consumes the existing indexed corpus.
        prompt = _get_prompt(corpus, prompt_id)
        context: list[dict[str, str]] = []
        owners: dict[str, str] = {}
        chunk_texts: dict[str, str] = {}
        warnings: list[str] = []
        incomplete = False
        for paper in corpus.papers:
            chunks = self.retriever.retrieve(
                query=prompt["text"], top_k=self.top_k, paper_id=paper.paper_id,
            )
            paper_chunks = 0
            for chunk in chunks:
                chunk_id, paper_id, text = _chunk_fields(chunk)
                if (paper_id != paper.paper_id or not isinstance(chunk_id, str)
                        or not chunk_id or not isinstance(text, str) or not text.strip()):
                    warnings.append(f"Discarded malformed or cross-paper retrieval result for {paper.paper_id}.")
                    incomplete = True
                    continue
                if chunk_id in owners and owners[chunk_id] != paper_id:
                    warnings.append(f"Discarded retrieval result with conflicting owner for {chunk_id}.")
                    incomplete = True
                    continue
                if chunk_id in chunk_texts:
                    if chunk_texts[chunk_id] != text:
                        warnings.append(f"Discarded duplicate chunk with conflicting text for {chunk_id}.")
                        incomplete = True
                    else:
                        paper_chunks += 1
                    continue
                context.append({
                    "paper_id": paper.paper_id,
                    "title": paper.title or "",
                    "chunk_id": chunk_id,
                    "text": text,
                })
                owners[chunk_id] = paper_id
                chunk_texts[chunk_id] = text
                paper_chunks += 1
            if not paper_chunks:
                warnings.append(f"No indexed chunks were retrieved for {paper.paper_id}.")
                incomplete = True

        if not context:
            output = _empty_output(corpus, prompt_id, self.system_name, "No indexed chunks were retrieved.")
            output.warnings = warnings or output.warnings
            return output
        response = self.generator.generate(
            _generation_prompt(prompt, context), _GeneratedGapSet
        )
        return _normalize_generated(
            response, corpus, prompt_id, self.system_name, owners, chunk_texts,
            input_chunk_count=len(context), initial_warnings=warnings,
            initially_incomplete=incomplete,
        )


def build_read_only_rag_retriever(
    persist_directory: Path,
    *,
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    embedding_dimension: int = 384,
):
    """Open the existing collection with `get_collection`; never create/upsert it."""
    import chromadb

    from src.rag import RAGRetriever

    embedder = LocalEmbeddingModel(
        model_name=embedding_model, dimension=embedding_dimension,
    )
    collection_name = collection_name_for(embedding_model)
    persist_directory = Path(persist_directory)
    if not persist_directory.is_dir():
        raise FileNotFoundError(f"Existing Chroma directory not found: {persist_directory}")
    client = chromadb.PersistentClient(path=str(persist_directory))
    collection = client.get_collection(name=collection_name)
    metadata = collection.metadata or {}
    if (metadata.get("embedding_model") != embedding_model
            or metadata.get("embedding_dimension") != embedding_dimension):
        raise ValueError(
            f"Chroma collection {collection_name!r} does not match the requested embedding model/dimension."
        )
    return RAGRetriever(
        embedding_model=embedder,
        vector_store=_ReadOnlyCollection(collection, embedding_dimension),
    )


class _ReadOnlyCollection:
    def __init__(self, collection, embedding_dimension: int):
        self.collection = collection
        self.embedding_dimension = embedding_dimension

    def search(self, query_embedding: list[float], n_results: int = 3,
               paper_id: str | None = None) -> dict:
        if len(query_embedding) != self.embedding_dimension:
            raise ValueError("Embedding dimension does not match the existing Chroma collection.")
        kwargs = {"query_embeddings": [query_embedding], "n_results": n_results}
        if paper_id:
            kwargs["where"] = {"paper_id": paper_id}
        return self.collection.query(**kwargs)


def _get_prompt(corpus: EvaluationCorpus, prompt_id: str) -> dict:
    try:
        return corpus.prompts[prompt_id]
    except KeyError as error:
        raise ValueError(f"Unknown finalized evaluation prompt: {prompt_id}") from error


def _generation_prompt(prompt: dict, context: list[dict[str, str]]) -> str:
    payload = {
        "baseline_instructions": BASELINE_INSTRUCTIONS,
        "research_prompt_id": prompt["prompt_id"],
        "research_prompt": prompt["text"],
        "evidence_context": context,
        "required_response_schema": _GeneratedGapSet.model_json_schema(),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _normalize_generated(
    response: BaseModel | dict,
    corpus: EvaluationCorpus,
    prompt_id: str,
    system_name: str,
    owners: dict[str, str],
    chunk_texts: dict[str, str],
    *,
    input_chunk_count: int,
    initial_warnings: list[str] | None = None,
    initially_incomplete: bool = False,
) -> EvaluationSystemOutput:
    parsed = response if isinstance(response, _GeneratedGapSet) else _GeneratedGapSet.model_validate(response)
    warnings = list(initial_warnings or [])
    incomplete = initially_incomplete
    candidates: list[CandidateGap] = []
    paper_ids = [paper.paper_id for paper in corpus.papers]
    for index, gap in enumerate(parsed.gaps, start=1):
        title, description = gap.title.strip(), gap.description.strip()
        if not title or not description:
            warnings.append(f"Dropped generated candidate {index} with an empty title or description.")
            incomplete = True
            continue
        citations: list[Citation] = []
        seen: set[tuple[str, str]] = set()
        for reference in gap.evidence:
            owner = owners.get(reference.chunk_id)
            if owner is None or owner != reference.paper_id or owner not in paper_ids:
                warnings.append(f"Dropped unknown or mismatched evidence reference {reference.paper_id}/{reference.chunk_id}.")
                incomplete = True
                continue
            key = (owner, reference.chunk_id)
            if key in seen:
                continue
            seen.add(key)
            quote = reference.quote
            if quote and quote not in chunk_texts[reference.chunk_id]:
                warnings.append(f"Removed non-verbatim quote for {reference.chunk_id}.")
                quote = None
                incomplete = True
            citations.append(Citation(
                paper_id=owner,
                chunk_id=reference.chunk_id,
                quote=quote,
                source_provenance=(
                    "single_llm_full_paper_chunks" if system_name == "single_llm"
                    else "conventional_rag_retrieved_chunk"
                ),
            ))
        if not citations:
            warnings.append(f"Dropped generated candidate {index} because it had no verifiable evidence references.")
            incomplete = True
            continue
        candidates.append(CandidateGap(
            gap_id=f"{system_name}-{prompt_id}-GAP-{len(candidates) + 1:03d}",
            title=title,
            description=description,
            citations=citations,
            novelty=NoveltyAssessment(
                indication=NoveltyIndication.NOT_ASSESSED,
                corpus_scope="Frozen evaluation corpus only.",
                limitations=["No claim of global novelty is made by this baseline."],
            ),
            source_provenance={
                "baseline": system_name,
                "research_prompt_id": prompt_id,
            },
        ))

    output = EvaluationSystemOutput(
        system_name=system_name,
        dataset_version=corpus.manifest.header.dataset_version,
        research_prompt_id=prompt_id,
        paper_ids=paper_ids,
        completion_status=(
            OutputCompletionStatus.INCOMPLETE if incomplete
            else OutputCompletionStatus.COMPLETE
        ),
        candidates=candidates,
        source_provenance={
            "baseline": system_name,
            "prompt_id": prompt_id,
            "input_chunk_count": str(input_chunk_count),
        },
        warnings=warnings,
    )
    try:
        return normalize_common_output(output.model_dump(mode="json"), owners)
    except AdapterError as error:
        raise ValueError(f"Baseline output failed the common evaluation adapter: {error}") from error


def _empty_output(
    corpus: EvaluationCorpus, prompt_id: str, system_name: str, warning: str,
) -> EvaluationSystemOutput:
    return EvaluationSystemOutput(
        system_name=system_name,
        dataset_version=corpus.manifest.header.dataset_version,
        research_prompt_id=prompt_id,
        paper_ids=[paper.paper_id for paper in corpus.papers],
        completion_status=OutputCompletionStatus.INCOMPLETE,
        source_provenance={"baseline": system_name, "prompt_id": prompt_id},
        warnings=[warning],
    )


def _chunk_fields(chunk) -> tuple[str, str, str]:
    if isinstance(chunk, dict):
        return chunk.get("chunk_id", ""), chunk.get("paper_id", ""), chunk.get("text", "")
    return chunk.chunk_id, chunk.paper_id, chunk.text
