import zlib

from src.processing import TextChunker
from src.rag import PaperIngestionPipeline, RAGRetriever
from src.vectorstore import ChromaVectorStore

DIM = 256


class FakeEmbedder:
    """Deterministic bag-of-words vectors; counts calls. No model, no network."""

    model_name = "fake/bow"
    dimension = DIM

    def __init__(self):
        self.document_calls = []
        self.query_calls = 0

    def _vec(self, text):
        v = [0.0] * DIM
        for word in text.lower().split():
            v[zlib.crc32(word.encode()) % DIM] += 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts):
        self.document_calls.append(len(texts))
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        self.query_calls += 1
        return self._vec(text)


class FakeExtractor:
    def __init__(self, text):
        self.text = text

    def extract_text(self, path):
        return self.text


QUANTUM = "quantum qubit entanglement superposition decoherence".split() * 4
PROTEIN = "protein enzyme folding ribosome peptide".split() * 4
GALAXY = "galaxy nebula pulsar quasar orbit".split() * 4
TEXT = " ".join(QUANTUM + PROTEIN + GALAXY)  # 60 words -> 4 chunks of 20, overlap 5


def build(tmp_path, text=TEXT, embedder=None):
    embedder = embedder or FakeEmbedder()
    pipeline = PaperIngestionPipeline(
        extractor=FakeExtractor(text),
        chunker=TextChunker(chunk_size=20, overlap=5),
        embedding_model=embedder,
        vector_store=ChromaVectorStore.for_embedder(embedder, persist_directory=str(tmp_path)),
    )
    return pipeline, embedder


def test_ingest_batches_embeddings_and_stores_chunks(tmp_path):
    pipeline, embedder = build(tmp_path)

    assert pipeline.ingest("any.pdf", "paperA") == 4
    assert embedder.document_calls == [4]  # one batched call, not one per chunk
    assert pipeline.vector_store.count() == 4


def test_reingest_unchanged_paper_skips_embedding_and_adds_no_duplicates(tmp_path):
    pipeline, embedder = build(tmp_path)
    pipeline.ingest("any.pdf", "paperA")
    pipeline.ingest("any.pdf", "paperA")

    assert embedder.document_calls == [4]
    assert pipeline.vector_store.count() == 4


def test_changed_paper_replaces_stale_chunks(tmp_path):
    pipeline, embedder = build(tmp_path)
    pipeline.ingest("any.pdf", "paperA")

    pipeline.extractor = FakeExtractor(" ".join(QUANTUM))  # shorter paper: 20 words
    assert pipeline.ingest("any.pdf", "paperA") == 2
    assert pipeline.vector_store.count() == 2
    assert embedder.document_calls == [4, 2]


def test_retrieval_uses_query_embedding_and_finds_relevant_chunk(tmp_path):
    pipeline, embedder = build(tmp_path)
    pipeline.ingest("any.pdf", "paperA")
    pipeline.ingest("any.pdf", "paperB")  # same text under another id

    retriever = RAGRetriever(embedding_model=embedder, vector_store=pipeline.vector_store)
    hits = retriever.retrieve("protein enzyme ribosome", top_k=2, paper_id="paperA")

    assert embedder.query_calls == 1
    assert {h.paper_id for h in hits} == {"paperA"}
    assert hits[0].chunk_id == "paperA_chunk_2" and "protein" in hits[0].text


def test_local_collection_survives_reopening_in_fresh_objects(tmp_path):
    pipeline, _ = build(tmp_path)
    pipeline.ingest("any.pdf", "paperA")

    embedder = FakeEmbedder()
    retriever = RAGRetriever(
        embedding_model=embedder,
        vector_store=ChromaVectorStore.for_embedder(embedder, persist_directory=str(tmp_path)),
    )

    assert retriever.retrieve("galaxy pulsar", top_k=1)[0].paper_id == "paperA"


def test_embedding_failure_preserves_existing_chunks(tmp_path):
    pipeline, embedder = build(tmp_path)

    # First ingestion succeeds; save the existing paper.
    pipeline.ingest("any.pdf", "paperA")
    old_hashes = pipeline.vector_store.get_chunk_hashes("paperA")
    old_count = pipeline.vector_store.count()

    # Change the paper and make the next embedding call fail.
    pipeline.extractor = FakeExtractor(
        " ".join(QUANTUM)
    )

    def failing_embed_documents(texts):
        raise RuntimeError("Simulated embedding failure")

    embedder.embed_documents = failing_embed_documents

    import pytest

    with pytest.raises(
        RuntimeError,
        match="Simulated embedding failure",
    ):
        pipeline.ingest("any.pdf", "paperA")

    # Previous chunks must remain untouched.
    assert pipeline.vector_store.count() == old_count
    assert pipeline.vector_store.get_chunk_hashes("paperA") == old_hashes
