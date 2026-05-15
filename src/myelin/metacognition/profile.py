"""User profiling: learns preferences, communication style, and priorities from episodes.
Uses a static/dynamic split inspired by Supermemory.
Static facts: stable preferences, habits, long-term traits.
Dynamic context: recent activity, current projects, short-term state.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..core.database import Database

CATEGORY_PATTERNS: dict[str, list[str]] = {
    "preference": [
        "prefer", "like", "favorite", "rather", "better", "preferred",
        "tend to", "usually", "typically", "always use", "always do",
        "prefer not", "don't like", "avoid",
    ],
    "style": [
        "style", "approach", "way of", "method", "philosophy",
        "concise", "verbose", "detailed", "thorough", "minimal",
        "writing", "format", "organized", "structure",
    ],
    "priority": [
        "priority", "important", "urgent", "critical", "blocker",
        "need to", "must", "should focus", "goal", "aim",
        "top of mind", "working on", "currently", "next",
    ],
    "skill": [
        "good at", "expert", "experienced", "skill", "familiar",
        "know how", "can handle", "comfortable", "proficient",
    ],
    "fact": [
        "used", "use", "work", "works", "working",
        "project", "repo", "repository", "environment",
        "setup", "config", "configured", "running",
    ],
}

MAX_STATIC_FACTS = 50
MAX_DYNAMIC_FACTS = 20
STATIC_CONFIDENCE_THRESHOLD = 0.75


@dataclass
class ProfileFact:
    id: str = ""
    agent_id: str = ""
    fact: str = ""
    category: str = "fact"  # 'preference', 'style', 'priority', 'skill', 'fact'
    confidence: float = 0.5
    is_static: bool = False
    created_at: str = ""
    last_observed: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "fact": self.fact,
            "category": self.category,
            "confidence": round(self.confidence, 3),
            "is_static": int(self.is_static),
            "created_at": self.created_at,
            "last_observed": self.last_observed,
        }


class UserProfiler:
    """Learns user preferences, style, and priorities from episodes.
    
    Maintains a split profile:
    - static_facts: High-confidence, stable traits that persist across sessions.
    - dynamic_context: Recent observations, current projects, temporary state.
    """

    def __init__(self, db: Database):
        self.db = db

    # ── Public API ─────────────────────────────────────────────────

    def learn_from_episode(self, episode: dict) -> list[ProfileFact]:
        """Extract profile-worthy facts from an episode text.
        
        Scans the action + content_text for signals about preferences,
        communication style, priorities, skills, and general facts.
        Returns list of newly created or updated ProfileFact objects.
        """
        agent_id = episode.get("agent_id", "unknown")
        text = f"{episode.get('action', '')} {episode.get('content_text', '')}"
        now = time.strftime("%Y-%m-%dT%H:%M:%S")

        if not text.strip():
            return []

        facts = self._extract_facts(text, agent_id)

        created_or_updated: list[ProfileFact] = []
        for fact_data in facts:
            fact = self._upsert_fact(agent_id, fact_data, now)
            if fact:
                created_or_updated.append(fact)

        self._graduate_static_facts(agent_id)

        return created_or_updated

    def get_profile(self, agent_id: str) -> dict[str, Any]:
        """Return current profile as a structured dict.
        
        Returns:
            {
                "static_facts": [...high-confidence stable facts...],
                "dynamic_context": [...recent/dynamic facts...],
                "last_updated": "...",
                "fact_count": int,
                "confidence_summary": {...}
            }
        """
        static = self._get_facts(agent_id, is_static=True)
        dynamic = self._get_facts(agent_id, is_static=False)
        profile = self._get_profile_row(agent_id)

        categories: dict[str, int] = {}
        confidence_summary: dict[str, float] = {}
        for fact_list, is_static_label in [(static, "static"), (dynamic, "dynamic")]:
            for f in fact_list:
                cat = f["category"]
                categories[cat] = categories.get(cat, 0) + 1
                key = f"{cat}_{is_static_label}"
                confidence_summary[key] = max(
                    confidence_summary.get(key, 0.0), f["confidence"]
                )

        return {
            "agent_id": agent_id,
            "static_facts": static,
            "dynamic_context": dynamic,
            "last_updated": profile.get("last_updated", "") if profile else "",
            "fact_count": len(static) + len(dynamic),
            "static_count": len(static),
            "dynamic_count": len(dynamic),
            "category_breakdown": categories,
            "confidence_summary": confidence_summary,
        }

    def get_facts_by_category(
        self, agent_id: str, category: str, min_confidence: float = 0.0
    ) -> list[dict[str, Any]]:
        """Get facts for a specific category (preference, style, priority, skill, fact)."""
        rows = self.db.fetchall(
            """SELECT * FROM profile_facts
               WHERE agent_id = ? AND category = ? AND confidence >= ?
               ORDER BY confidence DESC, last_observed DESC""",
            (agent_id, category, min_confidence),
        )
        return [dict(r) for r in rows]

    def invalidate_fact(self, fact_id: str) -> None:
        """Remove a fact (e.g., if it's outdated or wrong)."""
        self.db.delete("profile_facts", fact_id)

    # ── Fact extraction ────────────────────────────────────────────

    def _extract_facts(self, text: str, agent_id: str) -> list[dict[str, Any]]:
        """Extract candidate facts from text using pattern matching.
        
        Returns list of {fact, category, confidence} dicts.
        """
        text_lower = text.lower()
        found: list[dict[str, Any]] = []

        for category, patterns in CATEGORY_PATTERNS.items():
            for pattern in patterns:
                if pattern in text_lower:
                    fact = self._extract_sentence(text, pattern)
                    if fact and len(fact) > 10:
                        base_confidence = {
                            "preference": 0.6,
                            "style": 0.5,
                            "priority": 0.7,
                            "skill": 0.5,
                            "fact": 0.4,
                        }.get(category, 0.5)

                        found.append({
                            "fact": fact.strip(),
                            "category": category,
                            "confidence": base_confidence,
                        })
                        break  # One match per category per episode

        # Also extract tool/framework usage as skill facts
        self._extract_tool_skills(text_lower, agent_id, found)

        return found

    def _extract_sentence(self, text: str, pattern: str) -> str | None:
        """Extract the sentence containing the pattern match."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sentence in sentences:
            if pattern in sentence.lower():
                return sentence
        return None

    def _extract_tool_skills(
        self, text_lower: str, agent_id: str, found: list[dict[str, Any]]
    ) -> None:
        """Extract tool/tech usage as skill facts."""
        tech_patterns = [
            (r'\b(python|javascript|typescript|rust|go|golang|java|c\+\+|ruby)\b', "language"),
            (r'\b(django|flask|fastapi|react|vue|angular|spring|rails)\b', "framework"),
            (r'\b(postgresql|mysql|sqlite|mongodb|redis)\b', "database"),
            (r'\b(docker|kubernetes|k8s|aws|gcp|azure)\b', "infra"),
            (r'\b(git|github|gitlab|ci/cd|jenkins|github actions)\b', "devops"),
            (r'\b(pytest|jest|mocha|unittest|playwright|cypress)\b', "testing"),
            (r'\b(linux|ubuntu|debian|alpine|macos)\b', "os"),
        ]
        for regex_str, tech_category in tech_patterns:
            matches = re.findall(regex_str, text_lower)
            for match in matches:
                fact_text = f"works with {match} ({tech_category})"
                # Deduplicate
                if not any(f["fact"].lower() == fact_text for f in found):
                    found.append({
                        "fact": fact_text,
                        "category": "skill",
                        "confidence": 0.5,
                    })

    # ── Database operations ────────────────────────────────────────

    def _upsert_fact(
        self, agent_id: str, fact_data: dict[str, Any], now: str
    ) -> ProfileFact | None:
        """Insert or update a profile fact. If a similar fact exists, boost confidence."""
        fact_text = fact_data["fact"]
        category = fact_data["category"]
        incoming_confidence = fact_data["confidence"]

        # Look for existing fact with similar text and same category
        existing = self.db.fetchone(
            """SELECT * FROM profile_facts
               WHERE agent_id = ? AND category = ? AND fact = ?""",
            (agent_id, category, fact_text),
        )

        if existing:
            # Boost confidence on re-observation (diminishing returns)
            old_conf = existing["confidence"]
            new_conf = min(1.0, old_conf + (incoming_confidence * (1.0 - old_conf) * 0.5))
            self.db.update(
                "profile_facts",
                existing["id"],
                {
                    "confidence": new_conf,
                    "last_observed": now,
                },
            )
            return ProfileFact(
                id=existing["id"],
                agent_id=agent_id,
                fact=fact_text,
                category=category,
                confidence=new_conf,
                is_static=bool(existing["is_static"]),
                created_at=existing["created_at"],
                last_observed=now,
            )
        else:
            fact_id = str(uuid.uuid4())
            fact_obj = ProfileFact(
                id=fact_id,
                agent_id=agent_id,
                fact=fact_text,
                category=category,
                confidence=incoming_confidence,
                is_static=False,
                created_at=now,
                last_observed=now,
            )
            self.db.insert("profile_facts", fact_obj.to_dict())
            return fact_obj

    def _graduate_static_facts(self, agent_id: str) -> None:
        """Promote high-confidence dynamic facts to static.
        
        Facts with confidence >= STATIC_CONFIDENCE_THRESHOLD are graduated
        to the static set, meaning they represent stable preferences or traits.
        """
        candidates = self.db.fetchall(
            """SELECT * FROM profile_facts
               WHERE agent_id = ? AND is_static = 0 AND confidence >= ?
               ORDER BY confidence DESC""",
            (agent_id, STATIC_CONFIDENCE_THRESHOLD),
        )

        current_static_count = len(
            self.db.fetchall(
                "SELECT id FROM profile_facts WHERE agent_id = ? AND is_static = 1",
                (agent_id,),
            )
        )

        for candidate in candidates:
            if current_static_count >= MAX_STATIC_FACTS:
                # Replace lowest-confidence static fact
                lowest = self.db.fetchone(
                    """SELECT * FROM profile_facts
                       WHERE agent_id = ? AND is_static = 1
                       ORDER BY confidence ASC
                       LIMIT 1""",
                    (agent_id,),
                )
                if lowest and lowest["confidence"] < candidate["confidence"]:
                    self.db.conn.execute(
                        "UPDATE profile_facts SET is_static = 0 WHERE id = ?",
                        (lowest["id"],),
                    )
                    self.db.conn.execute(
                        "UPDATE profile_facts SET is_static = 1 WHERE id = ?",
                        (candidate["id"],),
                    )
                    self.db._commit_if_needed()
            else:
                self.db.conn.execute(
                    "UPDATE profile_facts SET is_static = 1 WHERE id = ?",
                    (candidate["id"],),
                )
                self.db._commit_if_needed()
                current_static_count += 1

    def _get_facts(self, agent_id: str, is_static: bool) -> list[dict[str, Any]]:
        """Get facts for an agent, optionally filtered by static/dynamic."""
        rows = self.db.fetchall(
            """SELECT * FROM profile_facts
               WHERE agent_id = ? AND is_static = ?
               ORDER BY confidence DESC, last_observed DESC""",
            (agent_id, int(is_static)),
        )
        result = []
        for r in rows:
            r["confidence"] = round(r["confidence"], 3)
            r["is_static"] = bool(r["is_static"])
            result.append(r)
        return result

    def _get_profile_row(self, agent_id: str) -> dict[str, Any] | None:
        """Get the agent_profiles row for this agent."""
        return self.db.fetchone(
            "SELECT * FROM agent_profiles WHERE agent_id = ?", (agent_id,)
        )

    def _update_static_facts_json(self, agent_id: str) -> None:
        """Sync the static_facts JSON column on agent_profiles from profile_facts table."""
        static = self._get_facts(agent_id, is_static=True)
        dynamic = self._get_facts(agent_id, is_static=False)
        now = time.strftime("%Y-%m-%dT%H:%M:%S")

        self.db.conn.execute(
            """UPDATE agent_profiles
               SET static_facts = ?, dynamic_context = ?, last_updated = ?
               WHERE agent_id = ?""",
            (json.dumps(static), json.dumps(dynamic), now, agent_id),
        )
        self.db._commit_if_needed()
