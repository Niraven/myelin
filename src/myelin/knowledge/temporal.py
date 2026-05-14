"""Temporal reasoning for time-aware retrieval.

mem0's April 2026 upgrade added temporal reasoning that ranks the right
dated instance for queries about current state, past events, and upcoming
plans. We go further: our temporal index tracks state transitions over
time, so we can answer "what changed" and "when did this start failing"
in addition to "what is the current state."

Key capabilities:
- Current state resolution: what is true NOW about an entity
- Historical queries: what WAS true at time T
- Change detection: when did state X transition to state Y
- Temporal decay: recent states weighted higher in retrieval
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any
from uuid import uuid4

from ..core.database import Database
from ..core.models import TemporalState


def _new_id() -> str:
    return uuid4().hex[:16]


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


class TemporalIndex:
    """Time-aware state tracking for entities and facts."""

    def __init__(self, db: Database):
        self.db = db

    def record_state(
        self,
        state_description: str,
        entity_id: str | None = None,
        semantic_node_id: str | None = None,
        source_episode_id: str | None = None,
        domain: str | None = None,
        confidence: float = 0.5,
    ) -> str:
        """Record a new state, closing the previous one for the same entity."""
        if entity_id:
            self.db.execute(
                "UPDATE temporal_states SET valid_until = ? "
                "WHERE entity_id = ? AND valid_until IS NULL",
                (_now_iso(), entity_id),
            )
            self.db.commit()

        state_id = _new_id()
        state = TemporalState(
            id=state_id,
            entity_id=entity_id,
            semantic_node_id=semantic_node_id,
            state_description=state_description,
            valid_from=_now_iso(),
            confidence=confidence,
            source_episode_id=source_episode_id,
            domain=domain,
        )
        self.db.insert("temporal_states", state.model_dump())
        return state_id

    def get_current_state(self, entity_id: str) -> dict[str, Any] | None:
        """Get the currently valid state for an entity."""
        return self.db.fetchone(
            "SELECT * FROM temporal_states "
            "WHERE entity_id = ? AND valid_until IS NULL "
            "ORDER BY valid_from DESC LIMIT 1",
            (entity_id,),
        )

    def get_state_at(self, entity_id: str, timestamp: str) -> dict[str, Any] | None:
        """Get the state that was valid at a specific time."""
        return self.db.fetchone(
            "SELECT * FROM temporal_states "
            "WHERE entity_id = ? AND valid_from <= ? "
            "AND (valid_until IS NULL OR valid_until > ?) "
            "ORDER BY valid_from DESC LIMIT 1",
            (entity_id, timestamp, timestamp),
        )

    def get_state_history(
        self,
        entity_id: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get the full state history for an entity, newest first."""
        return self.db.fetchall(
            "SELECT * FROM temporal_states "
            "WHERE entity_id = ? "
            "ORDER BY valid_from DESC LIMIT ?",
            (entity_id, limit),
        )

    def get_state_transitions(
        self,
        entity_id: str,
    ) -> list[dict[str, Any]]:
        """Get pairs of (old_state, new_state) transitions for an entity."""
        history = self.get_state_history(entity_id, limit=100)
        if len(history) < 2:
            return []

        transitions = []
        for i in range(len(history) - 1):
            transitions.append({
                "from_state": history[i + 1]["state_description"],
                "to_state": history[i]["state_description"],
                "changed_at": history[i]["valid_from"],
                "confidence": history[i]["confidence"],
            })
        return transitions

    def get_current_states_for_domain(
        self,
        domain: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get all currently valid states for a domain."""
        return self.db.fetchall(
            "SELECT ts.*, e.name as entity_name, e.entity_type "
            "FROM temporal_states ts "
            "LEFT JOIN entities e ON e.id = ts.entity_id "
            "WHERE ts.domain = ? AND ts.valid_until IS NULL "
            "ORDER BY ts.valid_from DESC LIMIT ?",
            (domain, limit),
        )

    def get_recent_changes(
        self,
        hours: float = 24.0,
        domain: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get state changes within the last N hours."""
        cutoff = datetime.utcnow()
        from datetime import timedelta
        cutoff_iso = (cutoff - timedelta(hours=hours)).isoformat()

        if domain:
            return self.db.fetchall(
                "SELECT ts.*, e.name as entity_name, e.entity_type "
                "FROM temporal_states ts "
                "LEFT JOIN entities e ON e.id = ts.entity_id "
                "WHERE ts.valid_from >= ? AND ts.domain = ? "
                "ORDER BY ts.valid_from DESC LIMIT ?",
                (cutoff_iso, domain, limit),
            )

        return self.db.fetchall(
            "SELECT ts.*, e.name as entity_name, e.entity_type "
            "FROM temporal_states ts "
            "LEFT JOIN entities e ON e.id = ts.entity_id "
            "WHERE ts.valid_from >= ? "
            "ORDER BY ts.valid_from DESC LIMIT ?",
            (cutoff_iso, limit),
        )

    def temporal_score(
        self,
        state: dict[str, Any],
        query_time: str | None = None,
    ) -> float:
        """Score a temporal state for relevance.

        Factors:
        - Recency: more recent states score higher
        - Validity: currently valid states get a boost
        - Confidence: higher confidence scores higher
        """
        confidence = state.get("confidence", 0.5)

        is_current = state.get("valid_until") is None
        currency_boost = 0.3 if is_current else 0.0

        try:
            valid_from = datetime.fromisoformat(state["valid_from"])
            age_hours = (datetime.utcnow() - valid_from).total_seconds() / 3600
            recency = 1.0 / (1.0 + age_hours / 24.0)
        except (ValueError, KeyError):
            recency = 0.5

        return confidence * 0.4 + recency * 0.3 + currency_boost

    def count(self) -> int:
        row = self.db.fetchone("SELECT COUNT(*) as cnt FROM temporal_states")
        return row["cnt"] if row else 0
