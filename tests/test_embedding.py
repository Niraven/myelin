"""Embedding provider tests."""

from myelin.memory.embedding import NoOpEmbedding, get_embedding_provider


def test_no_embedding_provider_is_default_without_optional_dependencies():
    provider = get_embedding_provider()
    assert isinstance(provider, NoOpEmbedding)
    assert provider.embed("anything") == []
    assert provider.embed_batch(["a", "b"]) == [[], []]
