"""Demonstrate Myelin learning from Hermes-style orchestrated agent runs."""

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

WORKFLOWS = [
    [
        ("hermes/research", "inspect failing GitHub Actions job"),
        ("hermes/build", "run ruff check src tests"),
        ("hermes/build", "apply formatting fixes"),
        ("hermes/build", "run pytest tests -q"),
        ("hermes/release", "push branch after green checks"),
    ],
    [
        ("hermes/research", "inspect failing CI logs"),
        ("hermes/build", "run ruff check src tests"),
        ("hermes/build", "apply formatting fixes"),
        ("hermes/build", "run pytest tests -q"),
        ("hermes/release", "push branch after green checks"),
    ],
    [
        ("hermes/research", "inspect failing GitHub Actions job"),
        ("hermes/build", "run ruff format --check src tests"),
        ("hermes/build", "apply formatting fixes"),
        ("hermes/build", "run pytest tests -q"),
        ("hermes/release", "push branch after green checks"),
    ],
    [
        ("hermes/research", "inspect failing CI logs"),
        ("hermes/build", "run ruff check src tests"),
        ("hermes/build", "apply formatting fixes"),
        ("hermes/build", "run mypy src/myelin"),
        ("hermes/build", "run pytest tests -q"),
        ("hermes/release", "push branch after green checks"),
    ],
    [
        ("hermes/research", "inspect failing GitHub Actions job"),
        ("hermes/build", "run ruff check src tests"),
        ("hermes/build", "apply formatting fixes"),
        ("hermes/build", "run pytest tests -q"),
        ("hermes/release", "push branch after green checks"),
    ],
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
        episodes = 0
        for run, workflow in enumerate(WORKFLOWS, start=1):
            session_id = f"hermes-ci-run-{run}"
            for agent_id, action in workflow:
                await handlers.observe(
                    agent_id=agent_id,
                    session_id=session_id,
                    action=action,
                    action_type="tool_call",
                    content_text=f"{agent_id} performed '{action}' during Hermes CI recovery.",
                    success=True,
                    domain="ci",
                    tags=["hermes", "orchestrated", "ci"],
                    input_context={
                        "orchestrator": "hermes",
                        "swarm_id": "ci-recovery-team",
                        "agent_role": agent_id.split("/")[-1],
                        "task_id": "repair-failing-ci",
                    },
                )
                episodes += 1

        final_session = Session(db, agent_id="hermes", session_id="hermes-proof")
        cognitive = await final_session.end()
        promoter_result = next(
            item for item in cognitive["cognitive_results"] if item["process"] == "promoter"
        )

        execution = await handlers.execute_procedure(
            query="ci workflow",
            agent_id="hermes",
        )

        feedback = None
        if execution["found"]:
            feedback = await handlers.procedure_feedback(
                procedure_id=execution["procedure_id"],
                success=True,
                notes="Hermes completed the CI repair workflow using the suggested procedure.",
            )

        return {
            "episodes_observed": episodes,
            "procedures_created": promoter_result.get("created", 0),
            "execution": execution,
            "feedback": feedback,
        }
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show Myelin learning from Hermes-style orchestrated workflows."
    )
    parser.add_argument("--db", type=Path, help="SQLite database path for the demo")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        db_path = args.db or Path(tmp) / "myelin-hermes-demo.db"
        result = asyncio.run(run_demo(db_path))

    print("Hermes + Myelin procedure-learning demo")
    print("=" * 41)
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
        print(f"  {index}. {step['description']} [{step['step_type']}]")

    feedback = result["feedback"]
    if feedback:
        print(f"Confidence after success feedback: {feedback['new_confidence']:.0%}")
        print(f"Trust after feedback: {feedback['trust_level']}")


if __name__ == "__main__":
    main()
