#!/usr/bin/env python3
"""Myelin Live Demo — 30-second tour of procedure learning, knowledge graph,
query, cross-agent transfer, and agent profiling. Fully self-contained,
no external services or MCP server needed."""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path

from myelin.core.database import Database
from myelin.memory.embedding import NoOpEmbedding
from myelin.memory.episodic import EpisodicMemory
from myelin.memory.procedural import ProceduralMemory
from myelin.memory.semantic import SemanticMemory
from myelin.tools.handlers import ToolHandlers

# ── Demo configuration ──────────────────────────────────────────
AGENT_A = "ops-agent"
AGENT_B = "dev-agent"

DOMAINS = {
    "deployment": [
        "git pull origin main",
        "docker build myelin-app:latest",
        "docker push registry/myelin-app:latest",
        "kubectl apply -f k8s/deployment.yaml",
        "kubectl rollout status deployment/myelin-app",
    ],
    "development": [
        "git checkout -b feature/experiment",
        "python -m pytest tests/ -q",
        "ruff check src/ --fix",
        "git commit -m 'fix lint issues'",
        "git push origin feature/experiment",
    ],
    "infrastructure": [
        "kubectl get pods --all-namespaces",
        "kubectl describe pod myelin-app-7d8f9",
        "docker logs myelin-app --tail 100",
        "systemctl status docker",
        "df -h && free -m",
    ],
}

# How many times to repeat each domain's workflow
REPEATS = 4  # 4 × 5 actions × 3 domains = 60 episodes


def emoji(s: str) -> str:
    """Return emoji for section label."""
    return {
        "🚀": "🚀",
        "🧠": "🧠",
        "💤": "💤",
        "🔍": "🔍",
        "🕸️": "🕸️",
        "📤": "📤",
        "📥": "📥",
        "📊": "📊",
        "✅": "✅",
    }.get(s, s)


async def run_demo() -> None:
    t0 = time.perf_counter()

    # ── 0. Fresh in-memory SQLite database ──────────────────────
    db = Database(":memory:", enable_vec=False)
    embedder = NoOpEmbedding()

    episodic = EpisodicMemory(db)
    semantic = SemanticMemory(db)
    procedural = ProceduralMemory(db)
    handlers = ToolHandlers(episodic, semantic, procedural, embedder)

    print("╔══════════════════════════════════════════════════╗")
    print("║     🧠  Myelin — Procedural Learning Demo      ║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    # ── 1. Simulate episodes ────────────────────────────────────
    section_start = time.perf_counter()
    print(f"{emoji('🚀')} Phase 1: Learning from behavior...")
    total_episodes = 0

    for domain, actions in DOMAINS.items():
        for run in range(1, REPEATS + 1):
            session = f"{domain}-run-{run}"
            for action in actions:
                await handlers.observe(
                    agent_id=AGENT_A,
                    session_id=session,
                    action=action,
                    action_type="tool_call",
                    content_text=f"{action} during {domain} run #{run}",
                    domain=domain,
                    success=True,
                    tags=[domain, "automated"],
                )
                total_episodes += 1

    print(f"   ✅ {total_episodes} episodes recorded across {len(DOMAINS)} domains")
    print(f"   ⏱️  {time.perf_counter() - section_start:.2f}s")

    # ── 2. Sleep consolidation ──────────────────────────────────
    section_start = time.perf_counter()
    print(f"\n{emoji('💤')} Phase 2: Sleep consolidation...")
    sleep_result = await handlers.trigger_sleep()
    print(f"   ✅ Entities extracted, relationships inferred, procedures promoted")
    for key in ("entities_extracted", "relationships_created", "entities_merged", "promoter"):
        val = sleep_result.get(key, sleep_result.get("created", 0))
        if isinstance(val, dict):
            val = val.get("created", 0)
        if val:
            print(f"      • {key.replace('_', ' ').title()}: {val}")
    print(f"   ⏱️  {time.perf_counter() - section_start:.2f}s")

    # ── 3. Query — "deployment workflow" ──────────────────────
    section_start = time.perf_counter()
    print(f"\n{emoji('🔍')} Phase 3: Query → 'deployment workflow'")
    exec_result = await handlers.execute_procedure(
        query="deployment workflow",
        agent_id=AGENT_A,
    )
    if exec_result["found"]:
        proc = exec_result
        print(f"   ✅ Found: {proc['name']}")
        print(f"      Confidence: {proc['confidence']:.0%}")
        print(f"      Trust: {proc['trust_level']}")
        print(f"      Steps:")
        for i, step in enumerate(proc["steps"], 1):
            desc = step["description"] if isinstance(step, dict) else str(step)
            print(f"         {i}. {desc}")
    else:
        print(f"   ⚠️  {exec_result['message']}")
    print(f"   ⏱️  {time.perf_counter() - section_start:.2f}s")

    # ── 4. Knowledge graph ──────────────────────────────────────
    section_start = time.perf_counter()
    print(f"\n{emoji('🕸️')} Phase 4: Knowledge graph around 'kubectl'")
    graph = await handlers.graph_query(entity_name="kubectl")

    boxes = [
        f"   ✅ Found entity: {graph['entity']['name']} ({graph['entity']['type']})"
        f" — {graph['entity']['mention_count']} mentions"
    ]
    if graph.get("neighbors"):
        for n in graph["neighbors"][:6]:
            boxes.append(f"      ● {n['name']} [{n['type']}]  ─{n['relation']}→  strength={n['strength']:.1f}")
    if graph.get("subgraph"):
        boxes.append(f"      Subgraph: {graph['subgraph']['node_count']} nodes, {graph['subgraph']['edge_count']} edges")
    print("\n".join(boxes))
    print(f"   ⏱️  {time.perf_counter() - section_start:.2f}s")

    # ── 5. Transfer to second agent ──────────────────────────────
    section_start = time.perf_counter()
    print(f"\n{emoji('📤')} Phase 5: Cross-agent transfer {AGENT_A} → {AGENT_B}")

    # Register agent B with some tool overlap
    from myelin.transfer.profiling import AgentProfiler
    profiler = AgentProfiler(db)
    profiler.get_or_create_profile(AGENT_B)
    for tool in ("git", "docker", "kubectl", "python"):
        profiler.record_tool_usage(agent_id=AGENT_B, tool_name=tool)

    # Discover transferable procedures
    discover = await handlers.transfer_discover(
        source_agent=AGENT_A,
        target_agent=AGENT_B,
        min_confidence=0.3,
    )
    print(f"   📋 Found {discover['count']} transferable procedure(s)")

    if discover["transferable_procedures"]:
        best = discover["transferable_procedures"][0]
        pid = best["procedure_id"]

        # Export
        pkg = await handlers.transfer_export(
            procedure_id=pid,
            source_agent=AGENT_A,
            target_agent=AGENT_B,
        )
        print(f"   📤 Exported: {pkg.get('procedure_name', 'unknown')}")
        print(f"      Agent similarity: {pkg.get('agent_similarity', 0):.2f}")
        print(f"      Transfer confidence: {pkg.get('transfer_confidence', 0):.2f}")

        # Import
        imp = await handlers.transfer_import(package=pkg, agent_id=AGENT_B)
        print(f"   📥 Imported by {AGENT_B} (confidence: {imp['transfer_confidence']:.1%})")
        print(f"      Status: {imp['status']}")
    print(f"   ⏱️  {time.perf_counter() - section_start:.2f}s")

    # ── 6. Agent profile ────────────────────────────────────────
    section_start = time.perf_counter()
    print(f"\n{emoji('📊')} Phase 6: Agent profiles after learning")

    for agent_id in (AGENT_A, AGENT_B):
        profile = profiler.get(agent_id)
        toolset = profiler.get_toolset(agent_id, min_usage=1)
        if profile:
            tools_used = json.loads(profile["tools"]) if isinstance(profile["tools"], str) else profile["tools"]
            print(f"   🤖 {agent_id}")
            print(f"      Tools: {', '.join(sorted(tools_used)) if tools_used else '(none)'}")
            print(f"      Tool usage: {len(toolset)} tools tracked")
            print(f"      Last seen: {profile.get('last_seen', 'N/A')}")

    # ── Summary ─────────────────────────────────────────────────
    total_time = time.perf_counter() - t0
    status = await handlers.status(agent_id=AGENT_A)

    print()
    print("╔══════════════════════════════════════════════════╗")
    print(f"║  ✅  Demo complete in {total_time:.1f}s              ║")
    print("╠══════════════════════════════════════════════════╣")
    print(f"║  Episodes:      {status['episodes']:>4}                        ║")
    print(f"║  Procedures:    {status['procedures']['total']:>4} ({status['procedures']['active']} active)        ║")
    print(f"║  Entities:      {status['entities']:>4}                        ║")
    print(f"║  Relationships: {status['relationships']:>4}                        ║")
    print(f"║  Temporal:      {status['temporal_states']:>4}                        ║")
    print("╚══════════════════════════════════════════════════╝")
    print()


def main() -> None:
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
