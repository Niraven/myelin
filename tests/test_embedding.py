"""Tests for embedding providers."""

from __future__ import annotations

import pytest

from myelin.memory.embedding import (
    ApiEmbedding,
    LocalEmbedding,
    NoOpEmbedding,
    get_embedding_provider,
)


class TestNoOpEmbedding:
    def test_returns_empty_list(self):
        e = NoOpEmbedding()
        assert e.embed("hello") == []
        assert e.embed_batch(["a", "b"]) == [[], []]
        assert e.dimension == 768


class TestLocalEmbedding:
    """Lightweight construction tests — model loads lazily on first embed."""

    def test_constructs_immediately(self):
        """Construction is instant (model loads lazily)."""
        e = LocalEmbedding()
        assert e._model is None  # not loaded yet

    def test_dimension_property_returns_default_without_loading(self):
        """Accessing dimension does NOT trigger model load."""
        e = LocalEmbedding()
        assert e.dimension == 768  # returns default without loading
        assert e._model is None  # still no load


class TestFactory:
    def test_none(self):
        e = get_embedding_provider("none")
        assert isinstance(e, NoOpEmbedding)

    def test_local(self):
        e = get_embedding_provider("local")
        assert isinstance(e, LocalEmbedding)

    def test_api_http_shorthand(self):
        e = get_embedding_provider("http://localhost:11434/api/embeddings")
        assert isinstance(e, ApiEmbedding)

    def test_api_prefix(self):
        e = get_embedding_provider("api:http://localhost:8000/v1/embeddings")
        assert isinstance(e, ApiEmbedding)

    def test_api_custom_model_name(self):
        e = get_embedding_provider(
            "api:http://localhost:11434/api/embeddings",
            model_name="nomic-embed-text",
        )
        assert isinstance(e, ApiEmbedding)
        assert e._model == "nomic-embed-text"

    def test_local_custom_model_name(self):
        e = get_embedding_provider("local", model_name="custom/local-model")
        assert isinstance(e, LocalEmbedding)
        assert e._model_name == "custom/local-model"

    def test_invalid(self):
        with pytest.raises(ValueError):
            get_embedding_provider("bogus")


class TestApiEmbedding:
    """Live remote-endpoint tests require a running server — skip by default."""

    def test_ollama_endpoint_detection(self):
        e = ApiEmbedding("http://localhost:11434/api/embeddings")
        assert e._is_ollama is True

    def test_openai_endpoint_detection(self):
        e = ApiEmbedding("http://localhost:8000/v1/embeddings")
        assert e._is_ollama is False

    def test_caches_identical_texts(self):
        e = ApiEmbedding("http://localhost:8000/v1/embeddings")
        # Simulate a cached result being returned
        text = "hello world"
        # Both calls should go through the same cache path
        e._cache[text] = [0.1, 0.2, 0.3]
        assert e.embed(text) == [0.1, 0.2, 0.3]

    def test_cache_miss_calls_api(self):
        e = ApiEmbedding("http://localhost:8000/v1/embeddings")
        assert e._cache == {}

    def test_api_key_from_env_openai(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test123")
        e = ApiEmbedding("https://api.openai.com/v1/embeddings")
        assert e._api_key == "sk-test123"

    def test_api_key_from_env_embedding(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_API_KEY", "emb-test-key")
        e = ApiEmbedding("http://localhost:8000/v1/embeddings")
        assert e._api_key == "emb-test-key"

    def test_api_key_prefers_openai(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-prefer")
        monkeypatch.setenv("EMBEDDING_API_KEY", "emb-fallback")
        e = ApiEmbedding("http://localhost:8000/v1/embeddings")
        assert e._api_key == "sk-prefer"

    def test_no_api_key_returns_empty_string(self):
        e = ApiEmbedding("http://localhost:8000/v1/embeddings")
        assert e._api_key == ""
