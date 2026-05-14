"""Shared test fixtures."""

import os
import tempfile

import pytest

from myelin.core.database import Database
from myelin.memory.embedding import NoOpEmbedding
from myelin.memory.episodic import EpisodicMemory
from myelin.memory.procedural import ProceduralMemory
from myelin.memory.semantic import SemanticMemory


@pytest.fixture
def tmp_db(tmp_path):
    db_path = tmp_path / "test.db"
    db = Database(path=db_path, enable_vec=False)
    _ = db.conn
    yield db
    db.close()


@pytest.fixture
def episodic(tmp_db):
    return EpisodicMemory(tmp_db)


@pytest.fixture
def semantic(tmp_db):
    return SemanticMemory(tmp_db)


@pytest.fixture
def procedural(tmp_db):
    return ProceduralMemory(tmp_db)


@pytest.fixture
def embedder():
    return NoOpEmbedding()
