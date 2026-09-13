from src.embeddings.gemini_embeddings import GeminiEmbeddingModel


embedding_model = GeminiEmbeddingModel()

text = "Large language models can assist scientific discovery."

vector = embedding_model.embed(text)

print(f"Embedding dimensions: {len(vector)}")
print(f"First 10 values: {vector[:10]}")