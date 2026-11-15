"""Reproducible read-only citation ownership indexes from the existing Chroma DB."""
from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import tempfile

from src.vectorstore.chroma_store import collection_name_for
from src.processing import PDFExtractor, TextChunker, TextCleaner

from .runner import load_dataset_manifest


def build_chunk_ownership_index(
    manifest_path: Path,
    persist_directory: Path,
    *,
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    embedding_dimension: int = 384,
    page_size: int = 1000,
) -> dict[str, str]:
    """Read existing records; reject missing, malformed, or conflicting ownership."""
    if page_size < 1:
        raise ValueError("page_size must be at least 1.")
    root = Path(__file__).resolve().parents[2]
    manifest = load_dataset_manifest(manifest_path, root)
    included = {
        paper.paper_id for paper in manifest.papers
        if isinstance(paper.inclusion_decision, dict)
        and paper.inclusion_decision.get("decision") == "INCLUDE_IN_PRIMARY_EVALUATION"
    }
    if len(included) != 6:
        raise ValueError("The frozen evaluation manifest must contain exactly six included papers.")

    directory = Path(persist_directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Existing Chroma directory not found: {directory}")
    import chromadb

    collection = chromadb.PersistentClient(path=str(directory)).get_collection(
        name=collection_name_for(embedding_model),
    )
    metadata = collection.metadata or {}
    if (metadata.get("embedding_model") != embedding_model
            or metadata.get("embedding_dimension") != embedding_dimension):
        raise ValueError("Existing Chroma collection model/dimension metadata does not match.")

    expected_chunks: dict[str, tuple[str, str]] = {}
    extractor, cleaner, chunker = PDFExtractor(), TextCleaner(), TextChunker()
    for paper in manifest.papers:
        if paper.paper_id not in included:
            continue
        text = cleaner.clean(extractor.extract_text(root / paper.relative_path))
        for chunk in chunker.chunk(text, paper.paper_id):
            expected_chunks[chunk.chunk_id] = (
                paper.paper_id,
                hashlib.sha256(chunk.text.encode("utf-8")).hexdigest(),
            )
    if not expected_chunks:
        raise ValueError("No expected chunks could be derived from the included evaluation PDFs.")

    owners: dict[str, str] = {}
    offset = 0
    while True:
        page = collection.get(limit=page_size, offset=offset, include=["metadatas"])
        ids = page.get("ids", [])
        metadata_rows = page.get("metadatas", [])
        if len(ids) != len(metadata_rows):
            raise ValueError("Chroma returned inconsistent IDs and metadata rows.")
        for record_id, record_metadata in zip(ids, metadata_rows):
            if not isinstance(record_metadata, dict):
                raise ValueError(f"Chroma record {record_id!r} has no ownership metadata.")
            paper_id = record_metadata.get("paper_id")
            chunk_id = record_metadata.get("chunk_id")
            if not isinstance(paper_id, str) or not isinstance(chunk_id, str) or not chunk_id:
                raise ValueError(f"Chroma record {record_id!r} has malformed paper/chunk metadata.")
            if chunk_id != record_id:
                raise ValueError(f"Chroma record ID and metadata chunk_id disagree for {record_id!r}.")
            if paper_id not in included:
                continue
            expected = expected_chunks.get(chunk_id)
            if expected is None:
                raise ValueError(f"Chroma contains an unrecognized or stale evaluation chunk {chunk_id!r}.")
            if expected[0] != paper_id:
                raise ValueError(f"Chroma paper ownership conflicts with the local PDF for chunk {chunk_id!r}.")
            if record_metadata.get("text_hash") != expected[1]:
                raise ValueError(f"Chroma text hash does not match the local PDF chunk {chunk_id!r}.")
            previous = owners.get(chunk_id)
            if previous is not None and previous != paper_id:
                raise ValueError(f"Conflicting Chroma owners for chunk {chunk_id!r}.")
            owners[chunk_id] = paper_id
        offset += len(ids)
        if len(ids) < page_size:
            break

    missing = set(expected_chunks) - set(owners)
    if missing:
        raise ValueError(f"Chroma is missing local evaluation chunks: {sorted(missing)[:10]}")
    return dict(sorted(owners.items()))


def write_chunk_ownership_index(index: dict[str, str], output_path: Path) -> Path:
    """Atomically serialize an ownership index in stable key order."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temp_name = handle.name
            json.dump(dict(sorted(index.items())), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)
    return path
