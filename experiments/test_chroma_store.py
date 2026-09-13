from src.embeddings.gemini_embeddings import GeminiEmbeddingModel
from src.vectorstore.chroma_store import ChromaVectorStore


embedding_model = GeminiEmbeddingModel()
vector_store = ChromaVectorStore(
    persist_directory="chroma_db_test",
    collection_name="test_collection",
)

documents = [
    "Large language models can assist scientific discovery.",
    "Machine learning can help predict molecular properties.",
]

ids = [
    "test_chunk_1",
    "test_chunk_2",
]

metadatas = [
    {
        "paper_id": "test_paper_1",
        "chunk_id": "test_chunk_1",
    },
    {
        "paper_id": "test_paper_2",
        "chunk_id": "test_chunk_2",
    },
]

embeddings = [
    embedding_model.embed(documents[0]),
    embedding_model.embed(documents[1]),
]

vector_store.add(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas,
)

print(f"Documents stored: {vector_store.count()}")