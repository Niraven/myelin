"""Entity extraction and linking engine.

Extracts structured entities from episode text, deduplicates them into
canonical forms, and links them across episodes. This is what mem0 does
with LLM-powered extraction, but we do it with pattern-based NER that
requires zero API calls and works offline.

The key insight from mem0's April 2026 upgrade: entity linking across
memories is worth +20 points on LoCoMo. We get the same benefit by
extracting entities from action sequences where the structure is
predictable (tool names, file paths, services, commands).
"""

from __future__ import annotations

import re
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from ..core.database import Database
from ..core.models import Entity, EntityMention, EntityType


def _new_id() -> str:
    return uuid4().hex[:16]


TOOL_PATTERNS = [
    re.compile(r"\b(git\s+\w+)\b", re.IGNORECASE),
    re.compile(r"\b(npm\s+\w+)\b", re.IGNORECASE),
    re.compile(r"\b(docker\s+\w+)\b", re.IGNORECASE),
    re.compile(r"\b(kubectl\s+\w+)\b", re.IGNORECASE),
    re.compile(r"\b(pip\s+\w+)\b", re.IGNORECASE),
    re.compile(r"\b(cargo\s+\w+)\b", re.IGNORECASE),
    re.compile(r"\b(make\s+\w+)\b", re.IGNORECASE),
    re.compile(r"\b(pytest|jest|mocha|vitest)\b", re.IGNORECASE),
    re.compile(r"\b(webpack|vite|esbuild|rollup|turbopack)\b", re.IGNORECASE),
    re.compile(r"\b(postgres(?:ql)?|mysql|redis|mongodb|sqlite)\b", re.IGNORECASE),
    re.compile(r"\b(playwright|puppeteer|selenium)\b", re.IGNORECASE),
    re.compile(r"\b(cloudflared|ngrok|tailscale|zerotier)\b", re.IGNORECASE),
    re.compile(r"\b(hermes|myelin)\b", re.IGNORECASE),
    re.compile(r"\b(obsidian|notion|remio)\b", re.IGNORECASE),
    re.compile(r"\b(kanban|trello|asana|linear|jira|notion)\b", re.IGNORECASE),
    re.compile(r"\b(pantheon|slack|discord|telegram|signal|whatsapp)\b", re.IGNORECASE),
]

FILE_PATTERN = re.compile(
    r"(?:^|\s)([\w./\-]+\.(?:py|js|ts|tsx|jsx|rs|go|java|yml|yaml|json|toml|md|sql|sh|css|html))\b"
)

SERVICE_PATTERNS = [
    re.compile(r"\b((?:AWS|GCP|Azure)\s*\w*)\b"),
    re.compile(r"\b(S3|EC2|Lambda|ECS|RDS|DynamoDB|CloudFront|SQS|SNS)\b"),
    re.compile(
        r"\b(Slack|GitHub|GitLab|Jira|Linear|Vercel|Netlify|Heroku|Railway)\b", re.IGNORECASE
    ),
]

ERROR_PATTERN = re.compile(r"\b(\w*Error|\w*Exception|\w*Fault|ENOENT|EACCES|ENOMEM|SIGKILL|OOM)\b")

CONFIG_PATTERN = re.compile(r"\b(\w+(?:_\w+){2,})\b")

COMMAND_PATTERN = re.compile(r"(?:^|\s)((?:sudo\s+)?[\w./\-]+(?:\s+[\w\-]+){0,3})\s*$")


def extract_entities_from_text(
    text: str,
    action: str = "",
    action_type: str = "",
) -> list[dict[str, str]]:
    """Extract entities from episode text using pattern-based NER.

    Returns list of dicts with 'name', 'entity_type', 'canonical_name'.
    """
    entities: list[dict[str, str]] = []
    seen_canonicals: set[str] = set()
    combined = f"{action} {text}"

    for pattern in TOOL_PATTERNS:
        for match in pattern.finditer(combined):
            raw = match.group(1).strip()
            canonical = _canonicalize(raw, "tool")
            if canonical and canonical not in seen_canonicals:
                entities.append(
                    {
                        "name": raw,
                        "entity_type": EntityType.TOOL.value,
                        "canonical_name": canonical,
                    }
                )
                seen_canonicals.add(canonical)

    for match in FILE_PATTERN.finditer(combined):
        raw = match.group(1).strip()
        canonical = _canonicalize(raw, "file")
        if canonical and canonical not in seen_canonicals:
            entities.append(
                {
                    "name": raw,
                    "entity_type": EntityType.FILE.value,
                    "canonical_name": canonical,
                }
            )
            seen_canonicals.add(canonical)

    for pattern in SERVICE_PATTERNS:
        for match in pattern.finditer(combined):
            raw = match.group(1).strip()
            canonical = _canonicalize(raw, "service")
            if canonical and canonical not in seen_canonicals:
                entities.append(
                    {
                        "name": raw,
                        "entity_type": EntityType.SERVICE.value,
                        "canonical_name": canonical,
                    }
                )
                seen_canonicals.add(canonical)

    for match in ERROR_PATTERN.finditer(combined):
        raw = match.group(1).strip()
        canonical = _canonicalize(raw, "error")
        if canonical and canonical not in seen_canonicals:
            entities.append(
                {
                    "name": raw,
                    "entity_type": EntityType.ERROR.value,
                    "canonical_name": canonical,
                }
            )
            seen_canonicals.add(canonical)

    return entities


def _canonicalize(name: str, entity_type: str) -> str:
    """Normalize entity name to canonical form for deduplication.

    Always lowercased to prevent case mismatches (GitHub vs github).
    """
    name = name.strip().lower()
    if not name or len(name) < 2:
        return ""

    return name


def extract_relations_from_sequence(
    episodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Infer relationships between entities from action sequences.

    If entity A appears before entity B in the same session,
    and B consistently follows A, infer A -> triggers -> B.
    If two entities always co-occur, infer A -> related_to -> B.
    """
    relations: list[dict[str, Any]] = []
    sessions: dict[str, list[dict]] = defaultdict(list)
    for ep in episodes:
        sessions[ep.get("session_id", "unknown")].append(ep)

    for _sid, session_eps in sessions.items():
        session_eps.sort(key=lambda e: e.get("timestamp", ""))
        session_entities: list[str] = []

        for ep in session_eps:
            ep_entities = extract_entities_from_text(
                ep.get("content_text", ""),
                ep.get("action", ""),
                ep.get("action_type", ""),
            )
            for ent in ep_entities:
                session_entities.append(ent["canonical_name"])

    co_occurrence: Counter[tuple[str, str]] = Counter()
    sequence_pairs: Counter[tuple[str, str]] = Counter()

    for _sid, session_eps in sessions.items():
        session_eps.sort(key=lambda e: e.get("timestamp", ""))
        prev_entities: set[str] = set()

        for ep in session_eps:
            ep_entities = extract_entities_from_text(
                ep.get("content_text", ""),
                ep.get("action", ""),
            )
            current = {e["canonical_name"] for e in ep_entities}

            for a in current:
                for b in current:
                    if a < b:
                        co_occurrence[(a, b)] += 1

            for prev in prev_entities:
                for curr in current:
                    if prev != curr:
                        sequence_pairs[(prev, curr)] += 1

            prev_entities = current

    for (a, b), count in sequence_pairs.items():
        if count >= 2:
            relations.append(
                {
                    "source": a,
                    "target": b,
                    "relation_type": "triggers",
                    "evidence_count": count,
                }
            )

    for (a, b), count in co_occurrence.items():
        if count >= 3:
            relations.append(
                {
                    "source": a,
                    "target": b,
                    "relation_type": "related_to",
                    "evidence_count": count,
                }
            )

    return relations


class EntityStore:
    """Manages entity storage, deduplication, and linking."""

    def __init__(self, db: Database):
        self.db = db

    def upsert_entity(
        self,
        name: str,
        entity_type: str,
        canonical_name: str,
        episode_id: str | None = None,
        domain: str | None = None,
        description: str | None = None,
    ) -> str:
        """Insert or update an entity, returning its ID."""
        existing = self.db.fetchone(
            "SELECT id, mention_count, source_episodes FROM entities "
            "WHERE canonical_name = ? AND entity_type = ?",
            (canonical_name, entity_type),
        )

        if existing:
            entity_id = str(existing["id"])
            import json

            episodes = json.loads(existing["source_episodes"] or "[]")
            if episode_id and episode_id not in episodes:
                episodes.append(episode_id)
            self.db.update(
                "entities",
                entity_id,
                {
                    "mention_count": existing["mention_count"] + 1,
                    "last_seen": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "source_episodes": episodes,
                },
            )
            return entity_id

        entity_id = _new_id()
        # LLM-backed extraction can return free-form types (e.g. "workflow",
        # "method", "database") that are not valid EntityType values. Normalize
        # unknown types to CONCEPT so extraction can never crash the write path.
        try:
            normalized_type = EntityType(entity_type)
        except ValueError:
            normalized_type = EntityType.CONCEPT
        entity = Entity(
            id=entity_id,
            name=name,
            entity_type=normalized_type,
            canonical_name=canonical_name,
            description=description,
            domain=domain,
            source_episodes=[episode_id] if episode_id else [],
        )
        self.db.insert("entities", entity.model_dump())
        return entity_id

    def add_mention(
        self,
        entity_id: str,
        source_type: str,
        source_id: str,
        context_snippet: str | None = None,
        role: str = "subject",
    ) -> str:
        mention_id = _new_id()
        mention = EntityMention(
            id=mention_id,
            entity_id=entity_id,
            source_type=source_type,
            source_id=source_id,
            context_snippet=context_snippet,
            role=role,
        )
        self.db.insert("entity_mentions", mention.model_dump())
        return mention_id

    def has_entity_mentions(self, source_id: str, source_type: str = "episode") -> int:
        """Return the count of existing entity mentions for a source.

        Returns 0 if no mentions exist. Used by callers to avoid
        re-extracting entities from an episode that already has them.
        """
        row = self.db.fetchone(
            "SELECT COUNT(*) as cnt FROM entity_mentions WHERE source_id = ? AND source_type = ?",
            (source_id, source_type),
        )
        return row["cnt"] if row else 0

    def get_entity(self, entity_id: str) -> dict[str, Any] | None:
        return self.db.fetchone("SELECT * FROM entities WHERE id = ?", (entity_id,))

    def find_by_canonical(self, canonical_name: str, entity_type: str) -> dict[str, Any] | None:
        return self.db.fetchone(
            "SELECT * FROM entities WHERE canonical_name = ? AND entity_type = ?",
            (canonical_name, entity_type),
        )

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        return self.db.fts_search("entities", "entities_fts", query, limit=limit)

    def get_entity_episodes(self, entity_id: str) -> list[dict[str, Any]]:
        mentions = self.db.fetchall(
            "SELECT source_id FROM entity_mentions WHERE entity_id = ? AND source_type = 'episode'",
            (entity_id,),
        )
        if not mentions:
            return []
        ids = [m["source_id"] for m in mentions]
        placeholders = ",".join("?" * len(ids))
        return self.db.fetchall(
            f"SELECT * FROM episodes WHERE id IN ({placeholders})",
            tuple(ids),
        )

    def get_co_occurring_entities(self, entity_id: str, limit: int = 10) -> list[dict[str, Any]]:
        """Find entities that co-occur with the given entity in the same episodes."""
        return self.db.fetchall(
            """
            SELECT e.*, COUNT(*) as co_count
            FROM entity_mentions m1
            JOIN entity_mentions m2 ON m1.source_id = m2.source_id
                AND m1.source_type = m2.source_type
                AND m1.entity_id != m2.entity_id
            JOIN entities e ON e.id = m2.entity_id
            WHERE m1.entity_id = ?
            GROUP BY e.id
            ORDER BY co_count DESC
            LIMIT ?
            """,
            (entity_id, limit),
        )

    def get_top_entities(self, domain: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        if domain:
            return self.db.fetchall(
                "SELECT * FROM entities WHERE domain = ? ORDER BY mention_count DESC LIMIT ?",
                (domain, limit),
            )
        return self.db.fetchall(
            "SELECT * FROM entities ORDER BY mention_count DESC LIMIT ?",
            (limit,),
        )

    def count(self) -> int:
        row = self.db.fetchone("SELECT COUNT(*) as cnt FROM entities")
        return row["cnt"] if row else 0

    def process_episode(
        self,
        episode_id: str,
        content_text: str,
        action: str = "",
        action_type: str = "",
        domain: str | None = None,
    ) -> list[str]:
        """Extract entities from an episode and store them. Returns entity IDs.

        If entity mentions already exist for this episode, skips re-extraction
        and returns existing entity IDs (dedup guard).
        """
        existing = self.has_entity_mentions(episode_id)
        if existing > 0:
            rows = self.db.fetchall(
                "SELECT DISTINCT entity_id FROM entity_mentions "
                "WHERE source_id = ? AND source_type = 'episode'",
                (episode_id,),
            )
            return [r["entity_id"] for r in rows]

        raw_entities = extract_entities_from_text(content_text, action, action_type)
        entity_ids = []

        for raw in raw_entities:
            entity_id = self.upsert_entity(
                name=raw["name"],
                entity_type=raw["entity_type"],
                canonical_name=raw["canonical_name"],
                episode_id=episode_id,
                domain=domain,
            )
            self.add_mention(
                entity_id=entity_id,
                source_type="episode",
                source_id=episode_id,
                context_snippet=content_text[:200],
            )
            entity_ids.append(entity_id)

        return entity_ids


class PatternExtractor:
    """Fast regex-based entity extraction. <1ms per text."""

    def extract(
        self,
        text: str,
        action: str = "",
        action_type: str = "",
    ) -> list[dict[str, str]]:
        return extract_entities_from_text(text, action, action_type)


class HybridEntityExtractor:
    """Two-tier entity extraction: regex (fast) + LLM (deep)."""

    def __init__(self, llm_extract: Callable[[list[str]], list[dict[str, str]]] | None = None):
        self.regex = PatternExtractor()
        self.llm = llm_extract

    def extract_fast(
        self,
        text: str,
        action: str = "",
        action_type: str = "",
    ) -> list[dict[str, str]]:
        """Regex-only extraction. Called at write time. <1ms."""
        return self.regex.extract(text, action, action_type)

    def extract_concepts(self, texts: list[str]) -> list[dict[str, str]]:
        """LLM-based concept extraction. Called during sleep.

        Takes batch of episode texts, returns new entity candidates.
        Each candidate has: name, entity_type, canonical_name.
        """
        if not self.llm:
            return []
        return self.llm(texts)
