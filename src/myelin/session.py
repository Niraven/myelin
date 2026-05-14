"""Session lifecycle management.

Handles session start, tracking, and end with cognitive process triggers.
"""

from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from .cognitive.orchestrator import CognitiveOrchestrator
from .core.database import Database
from .core.models import ActionType, Episode
from .knowledge.entities import EntityStore
from .knowledge.graph import KnowledgeGraph
from .knowledge.temporal import TemporalIndex
from .memory.embedding import EmbeddingProvider, NoOpEmbedding
from .memory.episodic import EpisodicMemory
from .memory.procedural import ProceduralMemory
from .memory.retriever import MultiSignalRetriever
from .memory.semantic import SemanticMemory
from .metacognition.confidence import ConfidenceMap
from .metacognition.impasse import ImpasseDetector


class Session:
    """Manages a single agent session with full cognitive loop."""

    def __init__(
        self,
        db: Database,
        agent_id: str,
        embedder: EmbeddingProvider | None = None,
        session_id: str | None = None,
    ):
        self.db = db
        self.agent_id = agent_id
        self.session_id = session_id or uuid4().hex[:16]
        self.embedder = embedder or NoOpEmbedding()

        self.episodic = EpisodicMemory(db)
        self.semantic = SemanticMemory(db)
        self.procedural = ProceduralMemory(db)
        self.confidence_map = ConfidenceMap(db)
        self.impasse_detector = ImpasseDetector(db)
        self.entity_store = EntityStore(db)
        self.graph = KnowledgeGraph(db)
        self.temporal = TemporalIndex(db)
        self.retriever = MultiSignalRetriever(
            db, self.entity_store, self.graph, self.temporal
        )
        self.orchestrator = CognitiveOrchestrator(
            db, self.episodic, self.semantic, self.procedural
        )

        self._started_at = time.time()
        self._episode_count = 0

    async def observe(
        self,
        action: str,
        action_type: str,
        content_text: str,
        input_context: dict | None = None,
        output_result: dict | None = None,
        success: bool = True,
        domain: str | None = None,
        tags: list[str] | None = None,
    ) -> str:
        """Record an observation and trigger background processes if needed."""
        embedding = self.embedder.embed(content_text) or None

        episode = Episode(
            agent_id=self.agent_id,
            session_id=self.session_id,
            action=action,
            action_type=ActionType(action_type),
            content_text=content_text,
            input_context=input_context,
            output_result=output_result,
            success=success,
            embedding=embedding,
            domain=domain,
            tags=tags or [],
        )

        episode_id = self.episodic.record(episode)
        self._episode_count += 1
        self.orchestrator.on_write()

        self.entity_store.process_episode(
            episode_id=episode_id,
            content_text=content_text,
            action=action,
            action_type=action_type,
            domain=domain,
        )

        if domain:
            self.confidence_map.update_domain(domain, episode_delta=1)
            self.impasse_detector.check(domain, self.agent_id)

        await self.orchestrator.check_triggers()

        return episode_id

    async def end(self) -> dict[str, Any]:
        """End the session and run all session-end cognitive processes."""
        results = await self.orchestrator.on_session_end()

        return {
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "duration_seconds": time.time() - self._started_at,
            "episodes_recorded": self._episode_count,
            "cognitive_results": results,
        }

    def get_stats(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "episodes": self._episode_count,
            "uptime_seconds": time.time() - self._started_at,
            "orchestrator": self.orchestrator.get_status(),
        }
