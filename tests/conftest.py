"""Shared test fixtures."""

import pytest

from myelin.core.database import Database
from myelin.intelligence.context import ContextAssembler
from myelin.knowledge.entities import EntityStore
from myelin.knowledge.graph import KnowledgeGraph
from myelin.knowledge.temporal import TemporalIndex
from myelin.memory.embedding import NoOpEmbedding
from myelin.memory.episodic import EpisodicMemory
from myelin.memory.procedural import ProceduralMemory
from myelin.memory.retriever import MultiSignalRetriever
from myelin.memory.semantic import SemanticMemory
from myelin.metacognition.confidence import ConfidenceMap
from myelin.transfer.profiling import AgentProfiler
from myelin.transfer.protocol import TransferProtocol


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


@pytest.fixture
def entity_store(tmp_db):
    return EntityStore(tmp_db)


@pytest.fixture
def graph(tmp_db):
    return KnowledgeGraph(tmp_db)


@pytest.fixture
def temporal(tmp_db):
    return TemporalIndex(tmp_db)


@pytest.fixture
def retriever(tmp_db, entity_store, graph, temporal):
    return MultiSignalRetriever(tmp_db, entity_store, graph, temporal)


@pytest.fixture
def confidence_map(tmp_db):
    return ConfidenceMap(tmp_db)


@pytest.fixture
def profiler(tmp_db):
    return AgentProfiler(tmp_db)


@pytest.fixture
def transfer_protocol(tmp_db, procedural):
    return TransferProtocol(tmp_db, procedural)


@pytest.fixture
def assembler(tmp_db, retriever, entity_store, graph, temporal, procedural, confidence_map):
    return ContextAssembler(
        tmp_db, retriever, entity_store, graph, temporal, procedural, confidence_map
    )
