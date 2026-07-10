"""Repeatable local benchmark harness for Myelin."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

from .core.database import Database
from .memory.embedding import NoOpEmbedding
from .memory.episodic import EpisodicMemory
from .memory.procedural import ProceduralMemory
from .memory.semantic import SemanticMemory
from .session import Session
from .tools.handlers import ToolHandlers

try:
    from benchmarks.ci_subset import run_ci_subset
    from benchmarks.nightly import run_nightly
    from benchmarks.longmemeval.dataset import LongMemEvalDataset
    from benchmarks.longmemeval.harness import evaluate
    from benchmarks.locomo.harness import evaluate_locomo

    _EVAL_AVAILABLE = True
except ImportError:
    _EVAL_AVAILABLE = False

WORKFLOW = [
    "git pull origin main",
    "npm test",
    "docker build myelin:latest",
    "docker push registry/myelin:latest",
    "kubectl rollout restart deployment/myelin",
]


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "avg_ms": 0.0}
    p95 = statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values)
    return {
        "p50_ms": round(statistics.median(values), 3),
        "p95_ms": round(p95, 3),
        "avg_ms": round(statistics.mean(values), 3),
    }


def _make_handlers(db: Database) -> ToolHandlers:
    return ToolHandlers(
        EpisodicMemory(db),
        SemanticMemory(db),
        ProceduralMemory(db),
        NoOpEmbedding(),
    )


def _make_events(count: int) -> list[dict[str, Any]]:
    events = []
    for index in range(count):
        step = WORKFLOW[index % len(WORKFLOW)]
        service = f"service-{index % 50}"
        events.append(
            {
                "agent_id": "bench-agent",
                "session_id": f"bench-session-{index // len(WORKFLOW)}",
                "action": step,
                "action_type": "tool_call",
                "content_text": f"{step} for {service} using git npm docker kubectl",
                "success": True,
                "domain": "deployment",
                "tags": ["benchmark", service],
            }
        )
    return events


async def run_benchmark(count: int, db_path: Path | None = None) -> dict[str, Any]:
    if db_path and db_path.exists():
        db_path.unlink()

    db = Database(db_path, enable_vec=False)
    handlers = _make_handlers(db)

    try:
        events = _make_events(count)
        store_times = []
        batch_size = 250
        for offset in range(0, len(events), batch_size):
            batch = events[offset : offset + batch_size]
            started = time.perf_counter()
            await handlers.observe_batch(batch)
            elapsed = (time.perf_counter() - started) * 1000
            per_event = elapsed / max(len(batch), 1)
            store_times.extend([per_event] * len(batch))

        recall_times = []
        context_times = []
        execute_times = []
        for _ in range(100):
            started = time.perf_counter()
            await handlers.recall("deploy service-7 docker kubectl", limit=5)
            recall_times.append((time.perf_counter() - started) * 1000)

            started = time.perf_counter()
            await handlers.context("deployment workflow service-7", domain="deployment")
            context_times.append((time.perf_counter() - started) * 1000)

            started = time.perf_counter()
            await handlers.execute_procedure("deployment workflow", agent_id="bench-agent")
            execute_times.append((time.perf_counter() - started) * 1000)

        promotion_started = time.perf_counter()
        result = await Session(db, agent_id="bench-agent", session_id="bench-final").end()
        promotion_ms = (time.perf_counter() - promotion_started) * 1000
        promoter_result = next(
            item for item in result["cognitive_results"] if item["process"] == "promoter"
        )

        procedure_hits = 0
        procedure_checks = 20
        steps_with_myelin = 0
        for _ in range(procedure_checks):
            execution = await handlers.execute_procedure(
                "deployment workflow",
                agent_id="bench-agent",
            )
            if execution["found"]:
                procedure_hits += 1
                steps_with_myelin += len(execution["steps"])

        cold_steps = len(WORKFLOW) + 3  # planning, investigation, validation overhead
        learned_steps = steps_with_myelin / procedure_hits if procedure_hits else len(WORKFLOW)
        agent_steps_saved = max(0.0, cold_steps - learned_steps)

        return {
            "count": count,
            "mode": "fast_trace_no_embeddings",
            "store": _stats(store_times),
            "recall": _stats(recall_times),
            "context": _stats(context_times),
            "execute_procedure": _stats(execute_times),
            "promotion_ms": round(promotion_ms, 3),
            "procedures_created": promoter_result.get("created", 0),
            "procedure_hit_rate": round(procedure_hits / procedure_checks, 3),
            "agent_steps_saved": round(agent_steps_saved, 3),
        }
    finally:
        db.close()


async def _run(args: argparse.Namespace) -> list[dict[str, Any]]:
    counts = [int(item.strip()) for item in args.counts.split(",") if item.strip()]
    results = []

    if args.db:
        db_base = Path(args.db)
        db_base.parent.mkdir(parents=True, exist_ok=True)
        for count in counts:
            results.append(await run_benchmark(count, db_base.with_suffix(f".{count}.db")))
        return results

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for count in counts:
            results.append(await run_benchmark(count, tmp_path / f"myelin-bench-{count}.db"))
    return results


def _run_eval(args: argparse.Namespace) -> dict[str, Any]:
    """Run evaluation harness benchmarks."""
    results: dict[str, Any] = {}

    if args.nightly:
        results["nightly"] = run_nightly()
        return results

    if args.longmemeval:
        dataset = LongMemEvalDataset()
        episodes = dataset.generate_episodes()
        questions = dataset.generate_questions()
        report = evaluate(questions, episodes)
        results["longmemeval"] = report

    if args.ci_subset:
        results["ci_subset"] = run_ci_subset()

    if args.locomo:
        results["locomo"] = evaluate_locomo()

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Myelin local memory operations.")
    parser.add_argument(
        "--counts",
        default="1000",
        help="Comma-separated observation counts. Use 1000,10000,50000 for a full run.",
    )
    parser.add_argument("--db", type=str, help="Optional base path for benchmark databases.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    parser.add_argument(
        "--longmemeval",
        action="store_true",
        help="Run LongMemEval-S synthetic benchmark (500 episodes, 100 questions).",
    )
    parser.add_argument(
        "--ci-subset",
        action="store_true",
        help="Run CI subset of LongMemEval (50 questions, <30s).",
    )
    parser.add_argument(
        "--nightly",
        action="store_true",
        help="Run full nightly eval suite (LongMemEval + LoCoMo).",
    )
    parser.add_argument(
        "--locomo",
        action="store_true",
        help="Run LoCoMo-S adversarial reasoning benchmark (50 questions).",
    )
    args = parser.parse_args()

    # ---- Eval harness modes ----
    if _EVAL_AVAILABLE and (args.longmemeval or args.ci_subset or args.nightly or args.locomo):
        results = _run_eval(args)
        if args.json:
            print(json.dumps(results, indent=2))
            return
        for name, report in results.items():
            print(f"\n{'=' * 60}")
            print(f"  {name}")
            print(f"{'=' * 60}")
            print(json.dumps(report, indent=2))
        return

    results = asyncio.run(_run(args))

    if args.json:
        print(json.dumps({"benchmarks": results}, indent=2))
        return

    print("Myelin local benchmark")
    print("======================")
    for result in results:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
