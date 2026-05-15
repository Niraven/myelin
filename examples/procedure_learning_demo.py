"""Demonstrate Myelin learning a deploy procedure from repeated agent behavior."""

from __future__ import annotations

import argparse
import asyncio
import tempfile
from pathlib import Path
from typing import Any

from myelin.core.database import Database
from myelin.memory.embedding import NoOpEmbedding
from myelin.memory.episodic import EpisodicMemory
from myelin.memory.procedural import ProceduralMemory
from myelin.memory.semantic import SemanticMemory
from myelin.session import Session
from myelin.tools.handlers import ToolHandlers

WORKFLOW = [
    "git pull origin main",
    "npm test",
    "docker build myelin:latest",
    "docker push registry/myelin:latest",
    "kubectl rollout restart deployment/myelin",
]


async def run_demo(db_path: Path) -> dict[str, Any]:
    if db_path.exists():
        db_path.unlink()

    db = Database(db_path, enable_vec=False)
    handlers = ToolHandlers(
        EpisodicMemory(db),
        SemanticMemory(db),
        ProceduralMemory(db),
        NoOpEmbedding(),
    )

    try:
        for run in range(5):
            session = Session(db, agent_id="demo-agent", session_id=f"deploy-run-{run + 1}")
            for action in WORKFLOW:
                await session.observe(
                    action=action,
                    action_type="tool_call",
                    content_text=f"{action} during production deployment",
                    domain="deployment",
                )

        final_session = Session(db, agent_id="demo-agent", session_id="launch-proof")
        cognitive = await final_session.end()
        promoter_result = next(
            item for item in cognitive["cognitive_results"] if item["process"] == "promoter"
        )

        execution = await handlers.execute_procedure(
            query="deployment workflow",
            agent_id="demo-agent",
        )

        feedback = None
        if execution["found"]:
            feedback = await handlers.procedure_feedback(
                procedure_id=execution["procedure_id"],
                success=True,
                notes="Demo run completed successfully.",
            )

        context = await handlers.context(
            query="deployment workflow",
            domain="deployment",
            agent_id="demo-agent",
            max_memories=3,
            max_procedures=1,
        )

        return {
            "episodes_observed": len(WORKFLOW) * 5,
            "procedures_created": promoter_result.get("created", 0),
            "execution": execution,
            "feedback": feedback,
            "context": context["assembled_text"],
        }
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show Myelin promoting repeated deployment actions into a procedure."
    )
    parser.add_argument("--db", type=Path, help="SQLite database path for the demo")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        db_path = args.db or Path(tmp) / "myelin-demo.db"
        result = asyncio.run(run_demo(db_path))

    print("Myelin procedure-learning demo")
    print("=" * 32)
    print(f"Episodes observed: {result['episodes_observed']}")
    print(f"Procedures created: {result['procedures_created']}")

    execution = result["execution"]
    if not execution["found"]:
        print("No executable procedure was found.")
        return

    print(f"Learned procedure: {execution['name']}")
    print(f"Initial confidence: {execution['confidence']:.0%}")
    print(f"Trust level: {execution['trust_level']}")
    print(f"Recommendation: {execution['recommendation']}")
    print("Steps:")
    for index, step in enumerate(execution["steps"], start=1):
        print(f"  {index}. {step['description']}")

    feedback = result["feedback"]
    if feedback:
        print(f"Confidence after success feedback: {feedback['new_confidence']:.0%}")
        print(f"Trust after feedback: {feedback['trust_level']}")

    print("\nAssembled context:")
    print(result["context"])


if __name__ == "__main__":
    main()
