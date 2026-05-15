"""ReconsolidationEngine: biologically-inspired reconsolidation with lability windows.

Inspired by Nader (2000), Agenternal, and ZenBrain.
Trigger: manual (MCP tool) or automatic (on retrieval/lability window expiry).

Algorithm:
1. When a memory is RETRIEVED, open a 6h lability window
2. When new evidence arrives, compute prediction error (Jaccard distance)
3. Select update mode based on effective PE
4. Apply stability protection before modifying old memories
5. Apply contradiction penalty (Agenternal β) to confidence
6. Log every reconsolidation event with before/after snapshots
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

from ..core.database import Database
from ..core.models import ProcessName
from ..memory.episodic import EpisodicMemory
from ..memory.procedural import ProceduralMemory
from ..memory.semantic import SemanticMemory
from .base import CognitiveProcess

# ── Constants ──────────────────────────────────────────────────

LABILITY_WINDOW_HOURS = 6
MAX_LABILE_MEMORIES = 10
DEFAULT_NE = 1.0
DEFAULT_5HT = 0.5

PE_CONFIRMED = 0.1
PE_SELECTIVE_EDIT = 0.3
PE_INTEGRATION = 0.7

BETA_MIN = 0.2
BETA_MAX = 0.85

CONTRADICTION_C_NEW = {
    "decision": 1.0,
    "fact": 0.9,
    "preference": 0.75,
    "opinion": 0.6,
}
DEFAULT_C_NEW = 0.85

LABILE_COLUMNS = {
    "episode": "labile_until",
    "semantic_node": "labile_until",
    "procedure": None,
}

MEMORY_TABLE = {
    "episode": "episodes",
    "semantic_node": "semantic_nodes",
    "procedure": "procedures",
}


def _new_id() -> str:
    return uuid4().hex[:16]


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _parse_iso_or_none(val: str | None) -> datetime | None:
    if not val:
        return None
    try:
        return datetime.fromisoformat(val)
    except (ValueError, TypeError):
        return None


def _jaccard_distance(text_a: str, text_b: str) -> float:
    """Compute Jaccard distance between two texts (word-level).

    Returns 0.0 if identical, 1.0 if completely different.
    J(A, B) = 1 - |intersection| / |union|
    """
    if not text_a and not text_b:
        return 0.0
    if not text_a or not text_b:
        return 1.0

    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())

    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b

    return 1.0 - (len(intersection) / max(len(union), 1))


class ReconsolidationEngine(CognitiveProcess):
    """Biologically-inspired reconsolidation with lability windows.

    Manages memory destabilization and restabilization when new evidence
    arrives, with prediction error computation, update mode selection,
    stability protection, and contradiction penalty.
    """

    name = ProcessName.RECONSOLIDATOR

    def __init__(
        self,
        db: Database,
        episodic: EpisodicMemory,
        semantic: SemanticMemory,
        procedural: ProceduralMemory,
    ):
        super().__init__(db)
        self.episodic = episodic
        self.semantic = semantic
        self.procedural = procedural

    # ── CognitiveProcess interface ─────────────────────────────

    def should_run(self) -> bool:
        """Check if there are labile memories that need processing."""
        count = self.db.fetchone(
            "SELECT COUNT(*) as cnt FROM episodes "
            "WHERE labile_until IS NOT NULL "
            "AND labile_until > datetime('now')"
        )
        episode_labile = count["cnt"] if count else 0

        count = self.db.fetchone(
            "SELECT COUNT(*) as cnt FROM semantic_nodes "
            "WHERE labile_until IS NOT NULL "
            "AND labile_until > datetime('now')"
        )
        semantic_labile = count["cnt"] if count else 0

        return (episode_labile + semantic_labile) > 0

    async def execute(self) -> dict[str, Any]:
        """Process expired lability windows — finalize any pending updates.

        Currently a no-op since reconsolidation is triggered manually
        or on retrieval. Future: auto-process expired labile memories.
        """
        processed = 0
        modified = 0

        episode_expired = self.db.execute(
            "UPDATE episodes SET labile_until = NULL WHERE labile_until < datetime('now')",
        )
        processed += episode_expired.rowcount

        semantic_expired = self.db.execute(
            "UPDATE semantic_nodes SET labile_until = NULL WHERE labile_until < datetime('now')",
        )
        processed += semantic_expired.rowcount

        self.db.commit()

        return {
            "processed": processed,
            "modified": modified,
            "expired_windows_cleaned": processed,
        }

    # ── Lability Window Management ─────────────────────────────

    def open_lability_window(self, memory_type: str, memory_id: str) -> str | None:
        """Open or extend a 6-hour lability window for a memory.

        If the memory already has a labile window, extend it to
        max(current, now + 6h). Enforces the MAX_LABILE_MEMORIES cap.
        Returns the labile_until ISO string, or None if type doesn't
        support lability windows.
        """
        labile_col = LABILE_COLUMNS.get(memory_type)
        if not labile_col:
            return None

        table = MEMORY_TABLE.get(memory_type)
        if not table:
            return None

        # Enforce cap: oldest evicted first
        self._evict_oldest_labile(table, labile_col)

        now = datetime.utcnow()
        new_labile = now + timedelta(hours=LABILITY_WINDOW_HOURS)
        new_labile_iso = new_labile.isoformat()

        row = self.db.fetchone(
            f"SELECT {labile_col} FROM {table} WHERE id = ?",
            (memory_id,),
        )
        if row and row.get(labile_col):
            current_labile = _parse_iso_or_none(row[labile_col])
            if current_labile and current_labile > now:
                # Extend: take the max of current and new
                extended = max(current_labile, new_labile)
                new_labile_iso = extended.isoformat()

        self.db.update(table, memory_id, {labile_col: new_labile_iso})
        return new_labile_iso

    def _evict_oldest_labile(self, table: str, labile_col: str) -> None:
        """Enforce MAX_LABILE_MEMORIES cap by clearing the oldest window."""
        labile_count = self.db.fetchone(
            f"SELECT COUNT(*) as cnt FROM {table} "
            f"WHERE {labile_col} IS NOT NULL "
            f"AND {labile_col} > datetime('now')",
        )
        current = labile_count["cnt"] if labile_count else 0

        while current >= MAX_LABILE_MEMORIES:
            oldest = self.db.fetchone(
                f"SELECT id FROM {table} "
                f"WHERE {labile_col} IS NOT NULL "
                f"AND {labile_col} > datetime('now') "
                f"ORDER BY {labile_col} ASC LIMIT 1",
            )
            if not oldest:
                break
            self.db.update(table, oldest["id"], {labile_col: None})
            current -= 1

    def get_labile_memories(
        self,
        memory_type: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Query all labile memories with active lability windows.

        Args:
            memory_type: 'episode', 'semantic_node', or None for all.
            limit: Max results per type.

        Returns:
            List of dicts with 'memory_type', 'memory_id', 'labile_until',
            and the full row data.
        """
        results: list[dict[str, Any]] = []

        if memory_type is None or memory_type == "episode":
            episodes = self.db.fetchall(
                "SELECT *, 'episode' as _memory_type FROM episodes "
                "WHERE labile_until IS NOT NULL "
                "AND labile_until > datetime('now') "
                "ORDER BY labile_until ASC LIMIT ?",
                (limit,),
            )
            for ep in episodes:
                results.append(
                    {
                        "memory_type": "episode",
                        "memory_id": ep["id"],
                        "labile_until": ep.get("labile_until"),
                        "data": ep,
                    }
                )

        if memory_type is None or memory_type == "semantic_node":
            nodes = self.db.fetchall(
                "SELECT *, 'semantic_node' as _memory_type FROM semantic_nodes "
                "WHERE labile_until IS NOT NULL "
                "AND labile_until > datetime('now') "
                "ORDER BY labile_until ASC LIMIT ?",
                (limit,),
            )
            for node in nodes:
                results.append(
                    {
                        "memory_type": "semantic_node",
                        "memory_id": node["id"],
                        "labile_until": node.get("labile_until"),
                        "data": node,
                    }
                )

        return results

    # ── Prediction Error Computation ───────────────────────────

    def compute_pe(
        self,
        existing_content: str,
        new_content: str,
        action_type: str | None = None,
        success: int | bool | None = None,
        ne: float = DEFAULT_NE,
        ht5: float = DEFAULT_5HT,
    ) -> tuple[float, float]:
        """Compute prediction error between existing and new content.

        Args:
            existing_content: Current memory content text.
            new_content: New evidence content text.
            action_type: If 'error', adds contradiction bonus.
            success: If 0/False, adds contradiction bonus.
            ne: Norepinephrine level (default 1.0).
            ht5: Serotonin level (default 0.5).

        Returns:
            Tuple of (pe_raw, pe_eff).
        """
        pe_raw = _jaccard_distance(existing_content, new_content)

        # Contradiction bonus
        contradiction_bonus = 0.0
        if action_type == "error":
            contradiction_bonus += 0.2
        if success is not None and not success:
            contradiction_bonus += 0.2

        pe_raw = min(1.0, pe_raw + contradiction_bonus)

        # Neuromodulation: PE_eff = PE_raw * (1 + 0.3*NE - 0.2*5HT)
        pe_eff = pe_raw * (1.0 + 0.3 * ne - 0.2 * ht5)
        pe_eff = max(0.0, min(1.0, pe_eff))

        return pe_raw, pe_eff

    # ── Update Mode Selection ──────────────────────────────────

    def select_update_mode(self, pe_eff: float) -> str:
        """Select update mode based on effective prediction error.

        Args:
            pe_eff: Effective prediction error (0.0-1.0).

        Returns:
            One of 'confirmed', 'selective_edit', 'integration', 'new_episode'.
        """
        if pe_eff < PE_CONFIRMED:
            return "confirmed"
        if pe_eff < PE_SELECTIVE_EDIT:
            return "selective_edit"
        if pe_eff < PE_INTEGRATION:
            return "integration"
        return "new_episode"

    # ── Stability Protector ────────────────────────────────────

    def stability_protector(
        self,
        memory_row: dict[str, Any],
    ) -> tuple[float, float, float]:
        """Determine if a memory is stable enough to resist update.

        Lock score L = min(access_count / 10, 1.0)
        Rigidity factor ρ = min(days_since_creation / 30, 1.0)
        Update threshold = 0.5 + 0.3 * L * ρ

        Only update if PE_eff > threshold.

        Args:
            memory_row: The memory row dict with 'access_count' and 'created_at'.

        Returns:
            Tuple of (threshold, lock_score, rigidity).
        """
        access_count = memory_row.get("access_count", 0) or 0
        created_at_raw = memory_row.get("created_at", _now_iso())

        lock_score = min(access_count / 10.0, 1.0)

        created_dt = _parse_iso_or_none(created_at_raw)
        if created_dt:
            days_since = (datetime.utcnow() - created_dt).total_seconds() / 86400.0
        else:
            days_since = 0.0

        rigidity = min(days_since / 30.0, 1.0)
        threshold = 0.5 + 0.3 * lock_score * rigidity

        return (threshold, lock_score, rigidity)

    # ── Contradiction Penalty ──────────────────────────────────

    def contradiction_penalty(
        self,
        content_type: str,
        old_confidence: float,
    ) -> float:
        """Compute Agenternal β contradiction penalty.

        β = 0.2 + (0.85 - 0.2) * exp(-c_new / s_old)

        Where c_new depends on content type (decision=1.0, fact=0.9,
        preference=0.75, opinion=0.6) and s_old is the current confidence.

        Returns new_confidence = max(0.05, old_confidence * β).
        """
        c_new = CONTRADICTION_C_NEW.get(content_type, DEFAULT_C_NEW)
        s_old = max(old_confidence, 0.01)

        beta = BETA_MIN + (BETA_MAX - BETA_MIN) * math.exp(-c_new / s_old)
        new_confidence = max(0.05, old_confidence * beta)

        return new_confidence

    # ── Snapshot ───────────────────────────────────────────────

    def _snapshot(self, memory_type: str, memory_id: str) -> dict[str, Any] | None:
        """Capture current state of a memory as a JSON-serializable dict."""
        table = MEMORY_TABLE.get(memory_type)
        if not table:
            return None

        row = self.db.fetchone(f"SELECT * FROM {table} WHERE id = ?", (memory_id,))
        if not row:
            return None

        snapshot: dict[str, Any] = {}
        for k, v in row.items():
            if isinstance(v, bytes):
                snapshot[k] = "<binary>"
            else:
                try:
                    json.dumps(v)
                    snapshot[k] = v
                except (TypeError, ValueError):
                    snapshot[k] = str(v)
        return snapshot

    # ── Logging ─────────────────────────────────────────────────

    def _log_reconsolidation(
        self,
        memory_type: str,
        memory_id: str,
        pe_raw: float,
        pe_eff: float,
        update_mode: str,
        labile_duration_minutes: float | None,
        snapshot_before: dict[str, Any] | None,
        snapshot_after: dict[str, Any] | None,
        trigger_episode_id: str | None = None,
        agent_id: str | None = None,
    ) -> str:
        """Record a reconsolidation event in the reconsolidation_log table."""
        log_id = _new_id()
        self.db.insert(
            "reconsolidation_log",
            {
                "id": log_id,
                "memory_type": memory_type,
                "memory_id": memory_id,
                "pe_raw": pe_raw,
                "pe_eff": pe_eff,
                "update_mode": update_mode,
                "labile_duration_minutes": labile_duration_minutes,
                "snapshot_before": json.dumps(snapshot_before) if snapshot_before else None,
                "snapshot_after": json.dumps(snapshot_after) if snapshot_after else None,
                "trigger_episode_id": trigger_episode_id,
                "agent_id": agent_id,
                "timestamp": _now_iso(),
            },
        )
        return log_id

    # ── Core Reconsolidation Logic ─────────────────────────────

    def process_new_evidence(
        self,
        memory_type: str,
        memory_id: str,
        new_content: str,
        action_type: str | None = None,
        success: int | bool | None = None,
        trigger_episode_id: str | None = None,
        agent_id: str | None = None,
        ne: float = DEFAULT_NE,
        ht5: float = DEFAULT_5HT,
        content_type: str | None = None,
    ) -> dict[str, Any]:
        """Process new evidence against an existing memory.

        Full reconsolidation pipeline:
        1. Open lability window (if supported)
        2. Compute prediction error
        3. Select update mode
        4. Check stability protection
        5. Apply contradiction penalty
        6. Execute update
        7. Log everything

        Args:
            memory_type: 'episode', 'semantic_node', or 'procedure'.
            memory_id: The ID of the existing memory.
            new_content: The new evidence content text.
            action_type: Action type for contradiction bonus.
            success: Success flag for contradiction bonus.
            trigger_episode_id: Episode that triggered this reconsolidation.
            agent_id: ID of the agent performing the reconsolidation.
            ne: Norepinephrine level.
            ht5: Serotonin level.
            content_type: Type for contradiction penalty (decision, fact, etc.).

        Returns:
            Dict with reconsolidation results.
        """
        table = MEMORY_TABLE.get(memory_type)
        if not table:
            return {"status": "error", "message": f"Unknown memory type: {memory_type}"}

        # 1. Open lability window
        self.open_lability_window(memory_type, memory_id)

        # 2. Fetch existing memory and compute PE
        memory_row = self.db.fetchone(
            f"SELECT * FROM {table} WHERE id = ?",
            (memory_id,),
        )
        if not memory_row:
            return {"status": "error", "message": "Memory not found"}

        # Get the content text field for PE computation
        if memory_type == "episode":
            existing_content = memory_row.get("content_text", "") or ""
            existing_content_full = existing_content + " " + (memory_row.get("action", "") or "")
        elif memory_type == "semantic_node":
            existing_content = memory_row.get("content", "") or ""
            existing_content_full = existing_content
        elif memory_type == "procedure":
            existing_content = memory_row.get("description", "") or ""
            steps = memory_row.get("steps", "[]")
            if isinstance(steps, str):
                try:
                    steps_list = json.loads(steps)
                    step_texts = [s.get("description", "") for s in steps_list]
                    existing_content_full = existing_content + " " + " ".join(step_texts)
                except (json.JSONDecodeError, TypeError):
                    existing_content_full = existing_content
            else:
                existing_content_full = existing_content
        else:
            existing_content_full = str(memory_row.get("content", ""))

        # Compute prediction error
        pe_raw, pe_eff = self.compute_pe(
            existing_content_full,
            new_content,
            action_type=action_type,
            success=success,
            ne=ne,
            ht5=ht5,
        )

        # 3. Select update mode
        update_mode = self.select_update_mode(pe_eff)

        # 4. Stability protection check
        stab_threshold, lock_score, rigidity = self.stability_protector(memory_row)

        # Snapshot before any modification
        snapshot_before = self._snapshot(memory_type, memory_id)

        # 5. Determine if we should actually modify the memory
        did_update = False
        snapshot_after: dict[str, Any] | None = None
        new_confidence: float | None = None

        # Confirmed mode always applies (non-destructive confidence boost)
        # Other modes respect stability protection
        should_update = (
            update_mode == "confirmed" or pe_eff > stab_threshold or update_mode == "new_episode"
        )

        if should_update:
            did_update = True

            # 6. Apply update based on mode
            if update_mode == "confirmed":
                # Boost confidence
                if memory_type == "semantic_node":
                    old_conf = memory_row.get("confidence", 0.5) or 0.5
                    new_confidence = min(1.0, old_conf + 0.1)
                    self.db.update(
                        "semantic_nodes",
                        memory_id,
                        {"confidence": new_confidence},
                    )
                elif memory_type == "episode":
                    old_imp = memory_row.get("importance_score", 0.5) or 0.5
                    new_imp = min(1.0, old_imp + 0.05)
                    self.db.update(
                        "episodes",
                        memory_id,
                        {"importance_score": new_imp},
                    )

            elif update_mode == "selective_edit":
                # Merge specific attributes (append content, update metadata)
                self._apply_selective_edit(memory_type, memory_id, memory_row, new_content)

                # Apply contradiction penalty to confidence
                if content_type and memory_type == "semantic_node":
                    old_conf = memory_row.get("confidence", 0.5) or 0.5
                    new_confidence = self.contradiction_penalty(content_type, old_conf)
                    self.db.update(
                        "semantic_nodes",
                        memory_id,
                        {"confidence": new_confidence},
                    )

            elif update_mode == "integration":
                # Full context merge
                self._apply_integration(memory_type, memory_id, memory_row, new_content)

                # Apply contradiction penalty
                if content_type and memory_type == "semantic_node":
                    old_conf = memory_row.get("confidence", 0.5) or 0.5
                    new_confidence = self.contradiction_penalty(content_type, old_conf)
                    self.db.update(
                        "semantic_nodes",
                        memory_id,
                        {"confidence": new_confidence},
                    )

            elif update_mode == "new_episode":
                # Create a separate new memory entry
                new_id = self._create_new_from_evidence(
                    memory_type,
                    memory_row,
                    new_content,
                    agent_id,
                )
                snapshot_after = {"new_memory_id": new_id}

            # Update PE tracking fields
            self._update_pe_fields(memory_type, memory_id, pe_raw, update_mode)

            # Snapshot after modification
            if update_mode != "new_episode":
                snapshot_after = self._snapshot(memory_type, memory_id)

            self.db.commit()

        # Calculate labile duration
        labile_minutes: float | None = None
        labile_col = LABILE_COLUMNS.get(memory_type)
        if labile_col:
            labile_val = memory_row.get(labile_col)
            if labile_val:
                labile_dt = _parse_iso_or_none(labile_val)
                if labile_dt:
                    labile_minutes = max(
                        0.0,
                        (datetime.utcnow() - labile_dt).total_seconds() / 60.0,
                    )

        # 7. Log the reconsolidation event
        log_id = self._log_reconsolidation(
            memory_type=memory_type,
            memory_id=memory_id,
            pe_raw=pe_raw,
            pe_eff=pe_eff,
            update_mode=update_mode,
            labile_duration_minutes=labile_minutes,
            snapshot_before=snapshot_before,
            snapshot_after=snapshot_after,
            trigger_episode_id=trigger_episode_id,
            agent_id=agent_id,
        )

        return {
            "status": "completed" if did_update else "skipped",
            "log_id": log_id,
            "memory_id": memory_id,
            "memory_type": memory_type,
            "pe_raw": round(pe_raw, 4),
            "pe_eff": round(pe_eff, 4),
            "update_mode": update_mode,
            "stability_threshold": round(stab_threshold, 4),
            "lock_score": round(lock_score, 4),
            "rigidity": round(rigidity, 4),
            "did_update": did_update,
            "new_confidence": round(new_confidence, 4) if new_confidence is not None else None,
        }

    def _apply_selective_edit(
        self,
        memory_type: str,
        memory_id: str,
        memory_row: dict[str, Any],
        new_content: str,
    ) -> None:
        """Merge specific attributes from new content into existing memory."""
        if memory_type == "episode":
            existing_text = memory_row.get("content_text", "") or ""
            merged = existing_text + " | " + new_content
            self.db.update("episodes", memory_id, {"content_text": merged})

        elif memory_type == "semantic_node":
            existing_content = memory_row.get("content", "") or ""
            merged = existing_content + " | " + new_content
            self.db.update(
                "semantic_nodes",
                memory_id,
                {
                    "content": merged,
                    "updated_at": _now_iso(),
                },
            )

        elif memory_type == "procedure":
            existing_desc = memory_row.get("description", "") or ""
            self.db.update(
                "procedures",
                memory_id,
                {
                    "description": existing_desc + " | " + new_content,
                    "updated_at": _now_iso(),
                },
            )

    def _apply_integration(
        self,
        memory_type: str,
        memory_id: str,
        memory_row: dict[str, Any],
        new_content: str,
    ) -> None:
        """Full context merge: combine content and update metadata."""
        if memory_type == "episode":
            existing_text = memory_row.get("content_text", "") or ""
            merged = f"[Integrated] {new_content} | {existing_text}"
            self.db.update(
                "episodes",
                memory_id,
                {
                    "content_text": merged,
                    "importance_score": min(
                        1.0,
                        (memory_row.get("importance_score", 0.5) or 0.5) + 0.05,
                    ),
                },
            )

        elif memory_type == "semantic_node":
            existing_content = memory_row.get("content", "") or ""
            merged = f"[Integrated] {new_content} | {existing_content}"
            old_conf = memory_row.get("confidence", 0.5) or 0.5
            new_conf = max(0.1, old_conf * 0.85)
            self.db.update(
                "semantic_nodes",
                memory_id,
                {
                    "content": merged,
                    "confidence": new_conf,
                    "updated_at": _now_iso(),
                },
            )

        elif memory_type == "procedure":
            existing_desc = memory_row.get("description", "") or ""
            self.db.update(
                "procedures",
                memory_id,
                {
                    "description": f"[Integrated] {new_content} | {existing_desc}",
                    "modify_count": (memory_row.get("modify_count", 0) or 0) + 1,
                    "updated_at": _now_iso(),
                },
            )

    def _create_new_from_evidence(
        self,
        memory_type: str,
        memory_row: dict[str, Any],
        new_content: str,
        agent_id: str | None = None,
    ) -> str:
        """Create a new memory entry from the evidence (PE ≥ 0.7 case)."""
        new_id = _new_id()

        if memory_type == "episode":
            from ..core.models import ActionType, Episode

            ep = Episode(
                id=new_id,
                agent_id=memory_row.get("agent_id", agent_id or "unknown"),
                session_id=memory_row.get("session_id", ""),
                action="reconsolidation: " + new_content[:80],
                action_type=ActionType.TOOL_CALL,
                content_text=new_content,
                domain=memory_row.get("domain"),
            )
            self.episodic.record(ep)

        elif memory_type == "semantic_node":
            from ..core.models import NodeType, SemanticNode, SourceType

            node = SemanticNode(
                id=new_id,
                node_type=NodeType.FACT,
                content=new_content,
                source_type=SourceType.OBSERVATION,
                source_ids=json.loads(memory_row.get("source_ids", "[]")),
                domain=memory_row.get("domain"),
                confidence=0.4,
            )
            self.semantic.store(node)

        return new_id

    def _update_pe_fields(
        self,
        memory_type: str,
        memory_id: str,
        pe_raw: float,
        update_mode: str,
    ) -> None:
        """Update prediction error tracking fields on the memory."""
        if memory_type == "semantic_node":
            self.db.update(
                "semantic_nodes",
                memory_id,
                {
                    "prediction_error": pe_raw,
                    "last_pe_raw": pe_raw,
                    "last_update_mode": update_mode,
                },
            )
        elif memory_type == "episode":
            self.db.update(
                "episodes",
                memory_id,
                {
                    "surprise_score": pe_raw,
                    "td_error": pe_raw,
                },
            )
        elif memory_type == "procedure":
            row = self.db.fetchone(
                "SELECT total_pe_sum, pe_count FROM procedures WHERE id = ?",
                (memory_id,),
            )
            if row:
                old_sum = row["total_pe_sum"] or 0.0
                old_count = row["pe_count"] or 0
                self.db.update(
                    "procedures",
                    memory_id,
                    {
                        "prediction_error": pe_raw,
                        "surprise_score": pe_raw,
                        "total_pe_sum": old_sum + pe_raw,
                        "pe_count": old_count + 1,
                    },
                )

    # ── MCP Tool Handler ───────────────────────────────────────

    async def myelin_reconsolidate(
        self,
        memory_id: str,
        memory_type: str,
        new_content: str,
        action_type: str | None = None,
        success: bool | None = None,
        content_type: str | None = None,
        agent_id: str | None = None,
        trigger_episode_id: str | None = None,
    ) -> dict[str, Any]:
        """Manually trigger reconsolidation on a specific memory.

        This is the MCP tool handler for myelin_reconsolidate.

        Args:
            memory_id: ID of the memory to reconsolidate.
            memory_type: 'episode', 'semantic_node', or 'procedure'.
            new_content: New evidence content to reconcile against.
            action_type: Action type ('error', 'tool_call', etc.) for PE bonus.
            success: Whether the action succeeded, for PE bonus.
            content_type: Content category (decision, fact, preference, opinion).
            agent_id: Optional agent identifier for audit trail.
            trigger_episode_id: Optional episode ID that triggered this.

        Returns:
            Dict with reconsolidation results.
        """
        if memory_type not in ("episode", "semantic_node", "procedure"):
            return {
                "tool": "myelin_reconsolidate",
                "result": {
                    "status": "error",
                    "message": f"Invalid memory_type '{memory_type}'. "
                    f"Must be 'episode', 'semantic_node', or 'procedure'.",
                },
            }

        if not new_content or not new_content.strip():
            return {
                "tool": "myelin_reconsolidate",
                "result": {
                    "status": "error",
                    "message": "new_content is required and must be non-empty.",
                },
            }

        table = MEMORY_TABLE.get(memory_type)
        if not table:
            return {
                "tool": "myelin_reconsolidate",
                "result": {
                    "status": "error",
                    "message": f"No table for memory type: {memory_type}",
                },
            }

        exists = self.db.fetchone(
            f"SELECT id FROM {table} WHERE id = ?",
            (memory_id,),
        )
        if not exists:
            return {
                "tool": "myelin_reconsolidate",
                "result": {
                    "status": "error",
                    "message": f"Memory {memory_id} not found in {table}.",
                },
            }

        result = self.process_new_evidence(
            memory_type=memory_type,
            memory_id=memory_id,
            new_content=new_content,
            action_type=action_type,
            success=success,
            trigger_episode_id=trigger_episode_id,
            agent_id=agent_id,
            content_type=content_type,
        )

        return {
            "tool": "myelin_reconsolidate",
            "result": result,
        }
