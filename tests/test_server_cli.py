"""Tests for Myelin's stdio server command-line contract."""

from __future__ import annotations

import sys

from myelin import server


def test_embedding_model_name_reaches_server(monkeypatch):
    captured: dict[str, str | None] = {}

    async def fake_run_server(
        db_path: str | None = None,
        embedding_provider: str = "none",
        llm_extraction: str | None = None,
        synthesis_model: str | None = None,
        embedding_model_name: str | None = None,
    ) -> None:
        captured.update(
            db_path=db_path,
            embedding_provider=embedding_provider,
            llm_extraction=llm_extraction,
            synthesis_model=synthesis_model,
            embedding_model_name=embedding_model_name,
        )

    monkeypatch.setattr(server, "run_server", fake_run_server)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "myelin",
            "--db",
            "/tmp/myelin.db",
            "--embedding-model",
            "api:http://127.0.0.1:11434/api/embeddings",
            "--embedding-model-name",
            "nomic-embed-text",
        ],
    )

    server.main()

    assert captured == {
        "db_path": "/tmp/myelin.db",
        "embedding_provider": "api:http://127.0.0.1:11434/api/embeddings",
        "llm_extraction": None,
        "synthesis_model": None,
        "embedding_model_name": "nomic-embed-text",
    }
