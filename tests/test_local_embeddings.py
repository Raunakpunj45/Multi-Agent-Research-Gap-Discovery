import numpy as np
import pytest

from src.embeddings.local_embeddings import BGE_QUERY_PREFIX, LocalEmbeddingModel


class FakeSentenceTransformer:
    """Stands in for SentenceTransformer: no download, no torch."""

    def __init__(self, dimension=4):
        self.dimension = dimension
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        return np.ones((len(texts), self.dimension), dtype="float32")


def make(dimension=4, model_dimension=4):
    fake = FakeSentenceTransformer(model_dimension)
    return LocalEmbeddingModel(model_name="fake/model", dimension=dimension, model=fake), fake


def test_documents_are_embedded_in_one_batched_call():
    model, fake = make()
    vectors = model.embed_documents(["a", "b", "c"])

    assert len(vectors) == 3 and all(len(v) == 4 for v in vectors)
    assert len(fake.calls) == 1
    assert fake.calls[0][1]["normalize_embeddings"] is True
    assert fake.calls[0][1]["batch_size"] == 32
    assert BGE_QUERY_PREFIX not in fake.calls[0][0][0]


def test_query_gets_prefix_and_has_declared_dimension():
    model, fake = make()
    vector = model.embed_query("what are the limitations?")

    assert len(vector) == model.dimension == 4
    assert fake.calls[0][0] == [BGE_QUERY_PREFIX + "what are the limitations?"]


def test_empty_and_invalid_inputs():
    model, fake = make()

    assert model.embed_documents([]) == []
    for bad in ([""], ["ok", "   "], [None], [123]):
        with pytest.raises(ValueError):
            model.embed_documents(bad)
    with pytest.raises(TypeError):
        model.embed_documents("a single string")
    with pytest.raises(ValueError):
        model.embed_query("  ")
    with pytest.raises(ValueError):
        LocalEmbeddingModel(batch_size=0)
    assert fake.calls == []  # nothing invalid reached the model


def test_dimension_mismatch_is_detected():
    model, _ = make(dimension=384, model_dimension=4)

    with pytest.raises(RuntimeError, match="expected 384"):
        model.embed_query("hello")


def test_constructing_does_not_load_the_real_model():
    # No sentence_transformers import / download until the first embed call.
    model = LocalEmbeddingModel()
    assert model._model is None
