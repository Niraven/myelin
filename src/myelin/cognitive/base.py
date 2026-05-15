"""Base class for cognitive background processes."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from ..core.database import Database
from ..core.models import ProcessName, ProcessRun


class CognitiveProcess(ABC):
    """Base for all 6 cognitive background processes.

    Each process runs asynchronously, never blocking the agent.
    Execution is tracked in the process_runs table.
    """

    name: ProcessName

    def __init__(self, db: Database):
        self.db = db

    async def run(self) -> dict[str, Any]:
        run = ProcessRun(process_name=self.name)
        self.db.insert("process_runs", run.model_dump())

        try:
            result = await self.execute()
            self.db.update(
                "process_runs",
                run.id,
                {
                    "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "status": "completed",
                    "items_processed": result.get("processed", 0),
                    "items_created": result.get("created", 0),
                    "items_modified": result.get("modified", 0),
                    "details": result,
                },
            )
            return result
        except Exception as e:
            self.db.update(
                "process_runs",
                run.id,
                {
                    "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "status": "failed",
                    "error": str(e),
                },
            )
            raise

    @abstractmethod
    async def execute(self) -> dict[str, Any]:
        """Implement the cognitive process logic. Returns metrics dict."""
        ...

    @abstractmethod
    def should_run(self) -> bool:
        """Check if this process should run based on triggers."""
        ...
