"""Embedding generation for semantic search."""

from __future__ import annotations

from typing import Protocol


class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
    @property
    def dimension(self) -> int: ...


class NoOpEmbedding:
    """Placeholder when no embedding model is configured."""

    @property
    def dimension(self) -> int:
        return 768

    def embed(self, text: str) -> list[float]:
        return []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[] for _ in texts]


class LocalEmbedding:
    """Local embedding using sentence-transformers (nomic-embed-text-v1.5)."""

    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5"):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name, trust_remote_code=True)
            self._dim = self._model.get_sentence_embedding_dimension()
        except ImportError:
            raise ImportError(
                "sentence-transformers required. Install with: pip install myelin[embeddings]"
            )

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()


def get_embedding_provider(provider: str = "none", **kwargs) -> EmbeddingProvider:
    if provider == "local":
        return LocalEmbedding(**kwargs)
    return NoOpEmbedding()
