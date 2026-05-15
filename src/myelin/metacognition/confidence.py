"""Metacognitive confidence map with calibration tracking."""

from __future__ import annotations

import time
from typing import Any

from ..core.database import Database
from ..core.models import DomainConfidence


class ConfidenceMap:
    def __init__(self, db: Database):
        self.db = db

    def update_domain(self, domain: str, episode_delta: int = 0, procedure_delta: int = 0) -> None:
        existing = self.db.fetchone("SELECT * FROM confidence_map WHERE domain = ?", (domain,))

        if existing:
            new_ep = existing["episode_count"] + episode_delta
            new_proc = existing["procedure_count"] + procedure_delta
            confidence = self._compute_domain_confidence(new_ep, new_proc)
            self.db.update(
                "confidence_map",
                existing["id"],
                {
                    "confidence": confidence,
                    "episode_count": new_ep,
                    "procedure_count": new_proc,
                    "last_activity": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
            )
        else:
            dc = DomainConfidence(
                domain=domain,
                confidence=self._compute_domain_confidence(episode_delta, procedure_delta),
                episode_count=max(0, episode_delta),
                procedure_count=max(0, procedure_delta),
                last_activity=time.strftime("%Y-%m-%dT%H:%M:%S"),
            )
            self.db.insert("confidence_map", dc.model_dump())

    def get_domain(self, domain: str) -> dict[str, Any] | None:
        return self.db.fetchone("SELECT * FROM confidence_map WHERE domain = ?", (domain,))

    def get_all(self) -> list[dict[str, Any]]:
        return self.db.fetchall("SELECT * FROM confidence_map ORDER BY confidence DESC")

    def get_weak_domains(self, threshold: float = 0.4) -> list[dict[str, Any]]:
        return self.db.fetchall(
            "SELECT * FROM confidence_map WHERE confidence < ? AND episode_count > 0 ORDER BY confidence ASC",
            (threshold,),
        )

    def _compute_domain_confidence(self, episodes: int, procedures: int) -> float:
        """Heuristic: confidence grows with evidence, procedures are worth more."""
        ep_score = min(1.0, episodes / 20.0) * 0.4
        proc_score = min(1.0, procedures / 3.0) * 0.6
        return min(1.0, ep_score + proc_score)
