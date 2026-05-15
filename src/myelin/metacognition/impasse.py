"""SOAR-inspired impasse detection and learning goal creation."""

from __future__ import annotations

import time
from typing import Any

from ..core.database import Database
from ..core.models import GoalStatus, LearningGoal


class ImpasseDetector:
    def __init__(self, db: Database):
        self.db = db

    def check(self, domain: str, agent_id: str) -> LearningGoal | None:
        """Check if we're at an impasse in this domain and create a learning goal."""
        confidence = self.db.fetchone(
            "SELECT confidence FROM confidence_map WHERE domain = ?", (domain,)
        )
        conf_value = confidence["confidence"] if confidence else 0.0

        if conf_value >= 0.4:
            return None

        procedures = self.db.fetchone(
            "SELECT COUNT(*) as cnt FROM procedures WHERE domain = ? AND status != 'archived'",
            (domain,),
        )
        if procedures and procedures["cnt"] > 0:
            return None

        existing_goal = self.db.fetchone(
            "SELECT * FROM learning_goals WHERE domain = ? AND status = ?",
            (domain, GoalStatus.ACTIVE.value),
        )
        if existing_goal:
            return None

        goal = LearningGoal(
            domain=domain,
            goal=f"Learn how agent handles {domain} tasks",
            strategy="Record with maximum detail next time",
            priority=max(0.3, 1.0 - conf_value),
        )
        self.db.insert("learning_goals", goal.model_dump())
        return goal

    def get_active_goals(self) -> list[dict[str, Any]]:
        return self.db.fetchall(
            "SELECT * FROM learning_goals WHERE status = ? ORDER BY priority DESC",
            (GoalStatus.ACTIVE.value,),
        )

    def resolve_goal(self, goal_id: str) -> None:
        self.db.update(
            "learning_goals",
            goal_id,
            {
                "status": GoalStatus.ACHIEVED.value,
                "resolved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )

    def abandon_goal(self, goal_id: str) -> None:
        self.db.update(
            "learning_goals",
            goal_id,
            {
                "status": GoalStatus.ABANDONED.value,
                "resolved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )
