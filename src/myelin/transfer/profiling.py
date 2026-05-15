"""Agent profiling and similarity scoring for cross-agent transfer."""

from __future__ import annotations

import time
from typing import Any

from ..core.activation import agent_similarity, transfer_confidence
from ..core.database import Database
from ..core.models import AgentProfile, TransferRecord


class AgentProfiler:
    def __init__(self, db: Database):
        self.db = db

    def register(self, profile: AgentProfile) -> None:
        existing = self.db.fetchone(
            "SELECT agent_id FROM agent_profiles WHERE agent_id = ?",
            (profile.agent_id,),
        )
        data = profile.model_dump()
        if existing:
            data.pop("first_seen", None)
            data["last_seen"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            self.db.update("agent_profiles", profile.agent_id, data)
        else:
            self.db.insert("agent_profiles", data)

    def get(self, agent_id: str) -> dict[str, Any] | None:
        return self.db.fetchone("SELECT * FROM agent_profiles WHERE agent_id = ?", (agent_id,))

    def compute_similarity(self, agent_a: str, agent_b: str) -> float:
        profile_a = self.get(agent_a)
        profile_b = self.get(agent_b)
        if not profile_a or not profile_b:
            return 0.3

        import json

        tools_a = set(
            json.loads(profile_a["tools"])
            if isinstance(profile_a["tools"], str)
            else profile_a["tools"]
        )
        tools_b = set(
            json.loads(profile_b["tools"])
            if isinstance(profile_b["tools"], str)
            else profile_b["tools"]
        )

        return agent_similarity(
            tools_a,
            tools_b,
            profile_a.get("context_format", ""),
            profile_b.get("context_format", ""),
            profile_a.get("model_family", ""),
            profile_b.get("model_family", ""),
        )

    def compute_transfer_confidence(
        self, source_confidence: float, source_agent: str, target_agent: str
    ) -> float:
        similarity = self.compute_similarity(source_agent, target_agent)
        return transfer_confidence(source_confidence, similarity)

    def record_transfer(
        self,
        procedure_id: str,
        source_agent: str,
        target_agent: str,
        source_confidence: float,
    ) -> TransferRecord:
        similarity = self.compute_similarity(source_agent, target_agent)
        conf = transfer_confidence(source_confidence, similarity)

        record = TransferRecord(
            procedure_id=procedure_id,
            source_agent=source_agent,
            target_agent=target_agent,
            similarity_score=similarity,
            transfer_confidence=conf,
        )
        self.db.insert("transfer_log", record.model_dump())
        return record
