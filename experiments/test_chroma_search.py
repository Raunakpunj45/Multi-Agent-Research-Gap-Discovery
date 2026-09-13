from src.embeddings.gemini_embeddings import GeminiEmbeddingModel
from src.vectorstore.chroma_store import ChromaVectorStore


embedding_model = GeminiEmbeddingModel()

vector_store = ChromaVectorStore(
    persist_directory="chroma_db_test",
    collection_name="test_collection",
)

query = "How can large language models help scientific discovery?"

query_embedding = embedding_model.embed(query)

results = vector_store.search(
    query_embedding=query_embedding,
    n_results=2,
)

print("\nQuery:")
print(query)

print("\nRetrieved documents:")

for i, document in enumerate(results["documents"][0], start=1):
    print(f"\n--- Result {i} ---")
    print(document)

print("\nDistances:")
print(results["distances"][0])

print("\nMetadata:")
print(results["metadatas"][0])