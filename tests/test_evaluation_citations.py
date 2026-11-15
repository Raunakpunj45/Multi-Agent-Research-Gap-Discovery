import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.evaluation.citations import build_chunk_ownership_index, write_chunk_ownership_index
from src.evaluation.runner import load_dataset_manifest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "evaluation" / "dataset_manifest.jsonl"


class FakeCollection:
    metadata = {
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "embedding_dimension": 384,
    }

    def __init__(self, rows):
        self.rows = rows

    def get(self, *, limit, offset, include):
        page = self.rows[offset:offset + limit]
        return {
            "ids": [chunk_id for chunk_id, _ in page],
            "metadatas": [metadata for _, metadata in page],
        }


def install_fake_chroma(monkeypatch, collection):
    class Client:
        def get_collection(self, name):
            assert name == "research_papers__bge_small_en_v1_5"
            return collection

    monkeypatch.setitem(
        __import__("sys").modules, "chromadb",
        SimpleNamespace(PersistentClient=lambda path: Client()),
    )


def rows_for_included():
    manifest = load_dataset_manifest(MANIFEST, ROOT)
    from src.processing import PDFExtractor, TextChunker, TextCleaner
    import hashlib

    rows = []
    for paper in manifest.papers:
        if paper.inclusion_decision["decision"] != "INCLUDE_IN_PRIMARY_EVALUATION":
            continue
        text = TextCleaner().clean(PDFExtractor().extract_text(ROOT / paper.relative_path))
        for chunk in TextChunker().chunk(text, paper.paper_id):
            rows.append((chunk.chunk_id, {
                "paper_id": paper.paper_id,
                "chunk_id": chunk.chunk_id,
                "text_hash": hashlib.sha256(chunk.text.encode("utf-8")).hexdigest(),
            }))
    return rows


def test_builds_stable_ownership_index_from_existing_chroma_records(monkeypatch, tmp_path):
    rows = rows_for_included()
    install_fake_chroma(monkeypatch, FakeCollection(rows))
    directory = tmp_path / "chroma"
    directory.mkdir()

    index = build_chunk_ownership_index(MANIFEST, directory, page_size=2)
    output = write_chunk_ownership_index(index, tmp_path / "owners.json")

    assert len(index) > 6
    assert set(index.values()) == {metadata["paper_id"] for _, metadata in rows}
    assert json.loads(output.read_text(encoding="utf-8")) == index


def test_ownership_index_rejects_missing_paper_coverage_and_fabricated_rows(monkeypatch, tmp_path):
    rows = rows_for_included()
    install_fake_chroma(monkeypatch, FakeCollection(rows[:-1]))
    directory = tmp_path / "chroma"
    directory.mkdir()
    with pytest.raises(ValueError, match="Chroma is missing local evaluation chunks"):
        build_chunk_ownership_index(MANIFEST, directory, page_size=2)

    malformed = rows_for_included()
    malformed[0] = (malformed[0][0], {"paper_id": malformed[0][1]["paper_id"], "chunk_id": "invented"})
    install_fake_chroma(monkeypatch, FakeCollection(malformed))
    with pytest.raises(ValueError, match="record ID and metadata chunk_id disagree"):
        build_chunk_ownership_index(MANIFEST, directory, page_size=2)
