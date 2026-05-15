"""Embedding generation for semantic search.

Supports three modes:
  - none:  NoOpEmbedding (placeholder)
  - local: sentence-transformers with torch backend (nomic-embed-text-v1.5, lazy)
  - api:   HTTP POST to any OpenAI-compatible / Ollama embedding endpoint
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Any, Protocol, cast

log = logging.getLogger("myelin.embedding")

# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
    @property
    def dimension(self) -> int: ...


# ---------------------------------------------------------------------------
# No-op
# ---------------------------------------------------------------------------


class NoOpEmbedding:
    """Placeholder when no embedding model is configured."""

    @property
    def dimension(self) -> int:
        return 768

    def embed(self, text: str) -> list[float]:
        return []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[] for _ in texts]


# ---------------------------------------------------------------------------
# Lazy torch-backed local embedding (sentence-transformers)
# ---------------------------------------------------------------------------


class LocalEmbedding:
    """Local embedding using sentence-transformers (torch backend, nomic-embed-text-v1.5).

    *Completely lazy* — the constructor is instant. The first ``embed()`` call
    imports sentence-transformers, downloads the model if needed, and caches
    it for subsequent calls.
    """

    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5"):
        self._model_name = model_name
        self._model: Any = None
        self._dim: int = 768

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        t0 = time.time()
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name, trust_remote_code=True)
            self._dim = int(self._model.get_embedding_dimension() or 768)
        except ImportError as err:
            raise ImportError(
                "sentence-transformers required. Install with: pip install myelin[embeddings]"
            ) from err
        log.info("LocalEmbedding loaded in %.2fs", time.time() - t0)

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        self._ensure_loaded()
        return cast(list[float], self._model.encode(text, normalize_embeddings=True).tolist())

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self._ensure_loaded()
        return cast(
            list[list[float]],
            self._model.encode(texts, normalize_embeddings=True).tolist(),
        )


# ---------------------------------------------------------------------------
# Remote API embedding (OpenAI-compatible / Ollama)
# ---------------------------------------------------------------------------


class ApiEmbedding:
    """Remote embedding via HTTP POST (OpenAI-compatible or Ollama endpoint).

    Accepts ``api:http://…`` or a bare ``http://…`` / ``https://…`` URL.

    Features:
    - Lazy API key loading from ``OPENAI_API_KEY`` or ``EMBEDDING_API_KEY`` env vars
    - Caches results for identical texts within the same session
    - Supports OpenAI and Ollama response formats
    """

    def __init__(self, endpoint: str, model: str = "nomic-embed-text-v1.5"):
        self._endpoint = endpoint.rstrip("/")
        self._model = model
        self._is_ollama = "/api/" in self._endpoint
        self._dim: int = 768
        self._cache: dict[str, list[float]] = {}
        # Lazy-load API key from environment
        self._api_key: str = self._load_api_key()

    @staticmethod
    def _load_api_key() -> str:
        """Load API key from environment, preferring OPENAI_API_KEY."""
        return os.environ.get("OPENAI_API_KEY") or os.environ.get("EMBEDDING_API_KEY") or ""

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        # Check cache first
        if text in self._cache:
            return self._cache[text]
        result = self._call(text)
        self._cache[text] = result
        return result

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        uncached: list[str] = []
        uncached_idx: list[int] = []

        for i, t in enumerate(texts):
            if t in self._cache:
                results.append(self._cache[t])
            else:
                results.append([])  # placeholder
                uncached.append(t)
                uncached_idx.append(i)

        if uncached:
            # For batch, call the API for each uncached text
            # (some providers support true batch, but we handle sequentially for simplicity)
            for j, t in zip(uncached_idx, uncached, strict=False):
                emb = self._call(t)
                self._cache[t] = emb
                results[j] = emb

        return results

    def _call(self, text: str) -> list[float]:
        headers = {"Content-Type": "application/json"}

        if self._is_ollama:
            payload = json.dumps({"model": self._model, "prompt": text}).encode("utf-8")
        else:
            # OpenAI-compatible — include API key if available
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            payload = json.dumps({"model": self._model, "input": text}).encode("utf-8")

        req = urllib.request.Request(
            self._endpoint,
            data=payload,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Embedding API error: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"Embedding API error: {exc}") from exc

        # OpenAI-compatible: {"data": [{"embedding": […]}]}
        if "data" in body and isinstance(body["data"], list) and len(body["data"]) > 0:
            emb = body["data"][0].get("embedding", [])
        # Ollama: {"embeddings": [[…]]}  or  {"embedding": […]}
        elif "embeddings" in body:
            emb = body["embeddings"][0] if body["embeddings"] else []
        elif "embedding" in body:
            emb = body["embedding"]
        else:
            raise RuntimeError(f"Unknown embedding response shape: {list(body.keys())}")

        if not emb:
            return []
        self._dim = len(emb)
        return [float(v) for v in emb]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_embedding_provider(provider: str = "none", **kwargs: Any) -> EmbeddingProvider:
    """Resolve a provider string to an ``EmbeddingProvider`` instance.

    Recognised values:

    * ``\"none\"``           → :class:`NoOpEmbedding`
    * ``\"local\"``          → :class:`LocalEmbedding` (lazy, torch-backed sentence-transformers)
    * ``\"api:…\"``          → :class:`ApiEmbedding`  (remote HTTP endpoint)
    * ``\"http://…\"`` / ``\"https://…\"`` → shorthand for ``api:<url>``
    """
    p = provider.strip()

    if p == "none":
        return NoOpEmbedding()

    if p == "local":
        return LocalEmbedding(**kwargs)

    if p.startswith("api:"):
        endpoint = p[4:]
        return ApiEmbedding(endpoint, **kwargs)

    if p.startswith("http://") or p.startswith("https://"):
        return ApiEmbedding(p, **kwargs)

    raise ValueError(
        f"Unknown embedding provider: '{provider}'. "
        "Expected one of: none, local, api:<url>, or a bare http(s) URL."
    )
