"""Local end-to-end smoke test (REAL model, REAL ChromaDB, no Gemini embedding calls).

Run from the project root:
    python experiments/smoke_local_embeddings.py [path/to/paper.pdf]

Writes only to the per-model collection; every other collection (including the
old Gemini one) is counted before and after to show it is unchanged.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chromadb

from src.embeddings import LocalEmbeddingModel
from src.rag import PaperIngestionPipeline, RAGRetriever
from src.vectorstore import ChromaVectorStore

PDF = sys.argv[1] if len(sys.argv) > 1 else "data/papers/test_paper.pdf"
PAPER_ID = "smoke_test_paper"
QUERY = "What are the limitations of the approach?"


def other_collections(persist="chroma_db", exclude=""):
    client = chromadb.PersistentClient(path=persist)
    names = [c if isinstance(c, str) else c.name for c in client.list_collections()]
    return {n: client.get_collection(n).count() for n in names if n != exclude}


def main():
    if not Path(PDF).exists():
        raise SystemExit(f"PDF not found: {PDF}")

    embedder = LocalEmbeddingModel()
    store = ChromaVectorStore.for_embedder(embedder)
    before = other_collections(exclude=store.collection_name)

    pipeline = PaperIngestionPipeline(embedding_model=embedder, vector_store=store)
    chunks = pipeline.ingest(PDF, PAPER_ID)
    stored = len(store.get_chunk_hashes(PAPER_ID))
    print(f"collection: {store.collection_name} | chunks ingested: {chunks} | stored: {stored}")

    sample = store.collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    print(f"stored vector dimension: {len(sample)} (expected {embedder.dimension})")

    pipeline.ingest(PDF, PAPER_ID)  # must be skipped, no duplicates
    print(f"after re-ingest: {store.count()} chunks total (should be unchanged)")

    reopened = ChromaVectorStore.for_embedder(embedder)  # fresh store object
    hits = RAGRetriever(embedding_model=embedder, vector_store=reopened).retrieve(
        QUERY, top_k=3, paper_id=PAPER_ID
    )
    for h in hits:
        print(f"\n[{h.chunk_id}] distance={h.distance:.4f}\n{h.text[:300]}...")

    after = other_collections(exclude=store.collection_name)
    print(f"\nother collections before: {before}\nother collections after:  {after}")
    print("UNCHANGED" if before == after else "CHANGED - investigate")


if __name__ == "__main__":
    main()
