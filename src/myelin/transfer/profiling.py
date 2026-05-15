"""Agent profiling and similarity scoring for cross-agent transfer."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from ..core.activation import agent_similarity, transfer_confidence
from ..core.database import Database
from ..core.models import AgentProfile, TransferRecord
from ..knowledge.entities import TOOL_PATTERNS
from .tool_map import find_alternative_tool


@dataclass
class AgentCapability:
    tool_name: str
    tool_type: str = "terminal"
    usage_count: int = 0
    last_used: str | None = None


def extract_tools_from_text(text: str) -> list[str]:
    """Extract tool names from episode text using TOOL_PATTERNS.

    Catches: git, docker, npm, python, kubectl, etc.
    Uses existing TOOL_PATTERNS from entities.py.
    """
    tools: set[str] = set()
    for pattern in TOOL_PATTERNS:
        for match in pattern.finditer(text):
            tool = match.group(1).strip().lower()
            base_tool = tool.split()[0] if " " in tool else tool
            tools.add(base_tool)
    return sorted(tools)


class AgentProfiler:
    def __init__(self, db: Database):
        self.db = db

    def get_or_create_profile(self, agent_id: str) -> dict[str, Any]:
        profile = self.get(agent_id)
        if profile:
            return profile
        self.register(AgentProfile(agent_id=agent_id))
        profile = self.get(agent_id)
        assert profile is not None, "Profile should exist after registration"
        return profile

    def register(self, profile: AgentProfile) -> None:
        existing = self.db.fetchone(
            "SELECT agent_id FROM agent_profiles WHERE agent_id = ?",
            (profile.agent_id,),
        )
        data = profile.model_dump()
        if existing:
            data.pop("first_seen", None)
            data["last_seen"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            processed: dict[str, Any] = {}
            for k, v in data.items():
                if k == "id":
                    continue
                if isinstance(v, (list, dict)):
                    processed[k] = json.dumps(v)
                elif isinstance(v, bool):
                    processed[k] = int(v)
                else:
                    processed[k] = v
            sets = ", ".join(f"{k} = :{k}" for k in processed)
            processed["_agent_id"] = profile.agent_id
            sql = f"UPDATE agent_profiles SET {sets} WHERE agent_id = :_agent_id"
            self.db.conn.execute(sql, processed)
            self.db._commit_if_needed()
        else:
            self.db.insert("agent_profiles", data)

    def get(self, agent_id: str) -> dict[str, Any] | None:
        return self.db.fetchone("SELECT * FROM agent_profiles WHERE agent_id = ?", (agent_id,))

    def learn_from_episode(self, episode: dict) -> None:
        """Extract tool usage from an episode and update agent profile."""
        tools = extract_tools_from_text(
            f"{episode.get('action', '')} {episode.get('content_text', '')}"
        )
        agent_id = episode.get("agent_id", "unknown")
        for tool in tools:
            self.record_tool_usage(agent_id=agent_id, tool_name=tool)

    def record_tool_usage(self, agent_id: str, tool_name: str) -> None:
        """Increment tool usage count, update last_seen, and update agent profile tools list."""
        profile = self.get_or_create_profile(agent_id)

        existing = self.db.fetchone(
            "SELECT usage_count FROM tool_usage WHERE agent_id = ? AND tool_name = ?",
            (agent_id, tool_name),
        )
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        if existing:
            self.db.conn.execute(
                "UPDATE tool_usage SET usage_count = usage_count + 1, last_seen = ? "
                "WHERE agent_id = ? AND tool_name = ?",
                (now, agent_id, tool_name),
            )
        else:
            self.db.conn.execute(
                "INSERT INTO tool_usage (agent_id, tool_name, usage_count, last_seen) "
                "VALUES (?, ?, 1, ?)",
                (agent_id, tool_name, now),
            )
        self.db._commit_if_needed()

        raw_tools = profile.get("tools", "[]")
        current_tools = json.loads(raw_tools) if isinstance(raw_tools, str) else raw_tools
        if tool_name not in current_tools:
            current_tools.append(tool_name)
            self.db.conn.execute(
                "UPDATE agent_profiles SET tools = ? WHERE agent_id = ?",
                (json.dumps(current_tools), agent_id),
            )
            self.db._commit_if_needed()

    def get_toolset(self, agent_id: str, min_usage: int = 1) -> list[AgentCapability]:
        """Get tools this agent uses, ranked by frequency.

        Sources from tool_usage table (auto-learned). Falls back to
        profile.tools only for tools with no usage tracking yet.
        """
        profile = self.get(agent_id)
        capabilities: list[AgentCapability] = []

        rows = self.db.fetchall(
            "SELECT tool_name, usage_count, last_seen FROM tool_usage "
            "WHERE agent_id = ? AND usage_count >= ? "
            "ORDER BY usage_count DESC, tool_name",
            (agent_id, min_usage),
        )
        for row in rows:
            capabilities.append(
                AgentCapability(
                    tool_name=row["tool_name"],
                    tool_type=self._infer_tool_type(row["tool_name"]),
                    usage_count=row["usage_count"],
                    last_used=row.get("last_seen"),
                )
            )

        # Only fall back to profile.tools when min_usage is 1
        # (no usage threshold to meet)
        if profile and min_usage <= 1:
            raw_tools = profile.get("tools", "[]")
            if isinstance(raw_tools, str):
                profile_tools = json.loads(raw_tools)
            elif isinstance(raw_tools, list):
                profile_tools = raw_tools
            else:
                profile_tools = []
            existing = {c.tool_name for c in capabilities}
            for tool in profile_tools:
                tool_name = tool.lower().strip() if isinstance(tool, str) else str(tool)
                if tool_name not in existing:
                    capabilities.append(
                        AgentCapability(
                            tool_name=tool_name,
                            tool_type=self._infer_tool_type(tool_name),
                            usage_count=1,
                            last_used=profile.get("last_seen"),
                        )
                    )

        return sorted(capabilities, key=lambda c: c.usage_count, reverse=True)

    def has_tool(self, agent_id: str, tool_name: str) -> bool:
        """Check if agent has a specific tool capability."""
        toolset = self.get_toolset(agent_id, min_usage=1)
        target_set = {c.tool_name for c in toolset}
        return tool_name.lower().strip() in target_set

    def find_alternative(self, required_tool: str, target_agent: str) -> str | None:
        """Find the closest available tool on the target agent."""
        toolset = self.get_toolset(target_agent, min_usage=1)
        target_set = {c.tool_name for c in toolset}
        return find_alternative_tool(required_tool, target_set)

    def compute_similarity(self, agent_a: str, agent_b: str) -> float:
        profile_a = self.get(agent_a)
        profile_b = self.get(agent_b)
        if not profile_a or not profile_b:
            return 0.3

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
        adapted: bool = False,
        adaptation_details: dict[str, Any] | None = None,
    ) -> TransferRecord:
        similarity = self.compute_similarity(source_agent, target_agent)
        conf = transfer_confidence(source_confidence, similarity)

        record = TransferRecord(
            procedure_id=procedure_id,
            source_agent=source_agent,
            target_agent=target_agent,
            similarity_score=similarity,
            transfer_confidence=conf,
            adapted=adapted,
            adaptation_details=adaptation_details or {},
        )
        self.db.insert("transfer_log", record.model_dump())
        return record

    def _infer_tool_type(self, tool_name: str) -> str:
        """Infer the type of a tool from its name."""
        name = tool_name.lower()
        if name in {"git", "gh", "github-cli"}:
            return "vcs"
        if name in {"docker", "podman", "nerdctl"}:
            return "container"
        if name in {"npm", "yarn", "pnpm", "pip", "pip3", "conda", "cargo"}:
            return "package_manager"
        if name in {"kubectl", "oc", "k"}:
            return "orchestrator"
        if name in {"psql", "mysql", "sqlite3"}:
            return "database"
        if name in {"aws", "gcloud", "az"}:
            return "cloud"
        if name in {"curl", "wget", "httpie"}:
            return "network"
        if name in {"pytest", "unittest", "nose"}:
            return "testing"
        return "terminal"
