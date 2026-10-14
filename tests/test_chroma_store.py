import chromadb
import pytest

from src.vectorstore.chroma_store import ChromaVectorStore, collection_name_for

MODEL, DIM = "BAAI/bge-small-en-v1.5", 3


def store(path, model=MODEL, dim=DIM):
    return ChromaVectorStore(model, dim, persist_directory=str(path))


def put(s, ids, vectors=None, paper_id="p1"):
    s.upsert(
        ids=ids,
        documents=[f"text {i}" for i in ids],
        embeddings=vectors or [[1.0, 0.0, 0.0]] * len(ids),
        metadatas=[{"paper_id": paper_id, "chunk_id": i, "text_hash": f"h{i}"} for i in ids],
    )


def test_collection_name_is_deterministic_and_per_model():
    assert collection_name_for(MODEL) == "research_papers__bge_small_en_v1_5"
    assert collection_name_for(MODEL) != collection_name_for("other/model-x")


def test_upsert_is_idempotent_and_survives_reopen(tmp_path):
    s = store(tmp_path)
    put(s, ["c1", "c2"])
    put(s, ["c1", "c2"])
    assert s.count() == 2

    reopened = store(tmp_path)  # fresh object, same persisted directory
    assert reopened.count() == 2
    assert reopened.collection.metadata["embedding_model"] == MODEL


def test_model_or_dimension_mismatch_is_rejected(tmp_path):
    name = collection_name_for(MODEL)
    store(tmp_path)

    with pytest.raises(ValueError):
        ChromaVectorStore("other/model", DIM, persist_directory=str(tmp_path), collection_name=name)
    with pytest.raises(ValueError):
        ChromaVectorStore(MODEL, 8, persist_directory=str(tmp_path), collection_name=name)


def test_legacy_gemini_collection_is_isolated_and_untouched(tmp_path):
    raw = chromadb.PersistentClient(path=str(tmp_path))
    legacy = raw.get_or_create_collection("research_papers")
    legacy.add(ids=["old1"], documents=["old"], embeddings=[[0.1] * 8])

    s = store(tmp_path)  # new per-model collection in the same database
    put(s, ["c1"])

    with pytest.raises(ValueError):  # opening legacy as local is refused
        ChromaVectorStore(MODEL, DIM, persist_directory=str(tmp_path), collection_name="research_papers")

    assert s.collection_name != "research_papers"
    assert legacy.count() == 1
    assert chromadb.PersistentClient(path=str(tmp_path)).get_collection("research_papers").count() == 1


def test_wrong_dimension_vectors_are_rejected(tmp_path):
    s = store(tmp_path)

    with pytest.raises(ValueError):
        put(s, ["c1"], vectors=[[1.0, 0.0]])
    with pytest.raises(ValueError):
        s.search([1.0, 0.0], n_results=1)


def test_paper_filter_hashes_and_delete(tmp_path):
    s = store(tmp_path)
    put(s, ["a1", "a2"], paper_id="A")
    put(s, ["b1"], paper_id="B")

    assert s.get_chunk_hashes("A") == {"a1": "ha1", "a2": "ha2"}
    result = s.search([1.0, 0.0, 0.0], n_results=5, paper_id="B")
    assert result["ids"][0] == ["b1"]

    s.delete_paper("A")
    assert s.count() == 1 and s.get_chunk_hashes("A") == {}
