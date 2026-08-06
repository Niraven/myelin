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

# Deterministic verified-feedback loop length. Each iteration executes the
# procedure (emitting a fresh prediction id) and reports success back through
# that prediction id, marking the evidence as *verified*. Three verified
# successful executions are required to promote stored trust to "trusted".
VERIFIED_FEEDBACK_CYCLES = 3


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
        initial_confidence = execution["confidence"] if execution["found"] else 0.0

        # Deterministic verified feedback loop: 3 × (execute → bound feedback).
        # The first execution above supplies the first prediction; subsequent
        # cycles execute again so every success is bound to a fresh prediction.
        feedback = None
        for cycle in range(VERIFIED_FEEDBACK_CYCLES):
            if cycle:
                execution = await handlers.execute_procedure(
                    query="deployment workflow",
                    agent_id="demo-agent",
                )
            feedback = await handlers.procedure_feedback(
                procedure_id=execution["procedure_id"],
                success=True,
                notes="Demo run completed successfully.",
                prediction_id=execution["prediction_id"],
            )

        assert feedback is not None
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
            "initial_confidence": initial_confidence,
            "prediction_id": feedback["prediction_id"],
            "evidence_quality": feedback["evidence_quality"],
            "trust_state": feedback["trust_state"],
            "execution": execution,
            "feedback": feedback,
            "matching_procedures": [p["name"] for p in context.get("matching_procedures", [])],
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
    print(f"Initial confidence: {result['initial_confidence']:.0%}")
    print(f"Verified feedback loop ({VERIFIED_FEEDBACK_CYCLES} × execute → bound feedback):")
    print(f"  evidence_quality: {result['evidence_quality']}")
    print(f"  stored trust_state: {result['trust_state']}")
    print("Steps:")
    for index, step in enumerate(execution["steps"], start=1):
        print(f"  {index}. {step['description']}")

    feedback = result["feedback"]
    if feedback:
        print(f"Confidence after verified feedback: {feedback['new_confidence']:.0%}")

    if result["matching_procedures"]:
        print(f"\nSame-domain context includes procedure: {result['matching_procedures'][0]}")

    print("\nAssembled context:")
    print(result["context"])


if __name__ == "__main__":
    main()
