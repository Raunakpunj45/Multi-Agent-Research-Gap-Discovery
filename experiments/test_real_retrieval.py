from src.embeddings.gemini_embeddings import GeminiEmbeddingModel
from src.vectorstore.chroma_store import ChromaVectorStore


embedding_model = GeminiEmbeddingModel()

vector_store = ChromaVectorStore(
    persist_directory="chroma_db",
    collection_name="research_papers",
)

query = "How does the paper generate scientific hypotheses?"

query_embedding = embedding_model.embed(query)

results = vector_store.search(
    query_embedding=query_embedding,
    n_results=3,
)

print("\nQuery:")
print(query)

print("\nRetrieved chunks:")

for i, document in enumerate(
    results["documents"][0],
    start=1,
):
    print(f"\n--- Result {i} ---")
    print(f"Distance: {results['distances'][0][i - 1]}")
    print(f"Metadata: {results['metadatas'][0][i - 1]}")
    print(document[:1000])