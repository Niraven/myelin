"""MCP tool handlers for Myelin tools."""

from __future__ import annotations

import json
import time
from typing import Any

from ..core.activation import procedure_recommendation, procedure_trust_level
from ..core.models import (
    ActionType,
    Episode,
    GoalStatus,
    Procedure,
    ProcedureStatus,
    ProcedureStep,
    PromotionMethod,
    StepType,
)
from ..intelligence.context import ContextAssembler
from ..intelligence.synthesizer import Synthesizer
from ..knowledge.entities import EntityStore, HybridEntityExtractor
from ..knowledge.graph import KnowledgeGraph
from ..knowledge.temporal import TemporalIndex
from ..memory.embedding import EmbeddingProvider
from ..memory.episodic import EpisodicMemory
from ..memory.procedural import ProceduralMemory
from ..memory.retriever import MultiSignalRetriever
from ..memory.semantic import SemanticMemory
from ..metacognition.confidence import ConfidenceMap
from ..metacognition.profile import UserProfiler
from ..transfer.profiling import AgentProfiler
from ..transfer.protocol import TransferProtocol
from .visualize import Visualizer


class ToolHandlers:
    """Implements Myelin MCP tool operations."""

    def __init__(
        self,
        episodic: EpisodicMemory,
        semantic: SemanticMemory,
        procedural: ProceduralMemory,
        embedder: EmbeddingProvider,
        entity_store: EntityStore | None = None,
        graph: KnowledgeGraph | None = None,
        temporal: TemporalIndex | None = None,
        retriever: MultiSignalRetriever | None = None,
        context_assembler: ContextAssembler | None = None,
        transfer_protocol: TransferProtocol | None = None,
        confidence_map: ConfidenceMap | None = None,
        agent_profiler: AgentProfiler | None = None,
        synthesizer: Synthesizer | None = None,
        hybrid_extractor: HybridEntityExtractor | None = None,
    ):
        self.episodic = episodic
        self.semantic = semantic
        self.procedural = procedural
        self.embedder = embedder
        self.db = episodic.db
        self.synthesizer = synthesizer

        self.entities = entity_store or EntityStore(self.db)
        self.graph = graph or KnowledgeGraph(self.db)
        self.temporal = temporal or TemporalIndex(self.db)
        self.retriever = retriever or MultiSignalRetriever(
            self.db, self.entities, self.graph, self.temporal
        )
        self.confidence_map = confidence_map or ConfidenceMap(self.db)
        self.profiler = agent_profiler or AgentProfiler(self.db)
        self.user_profiler = UserProfiler(self.db)
        self.transfer = transfer_protocol or TransferProtocol(self.db, self.procedural)
        self.assembler = context_assembler or ContextAssembler(
            self.db,
            self.retriever,
            self.entities,
            self.graph,
            self.temporal,
            self.procedural,
            self.confidence_map,
            self.embedder,
        )
        self.hybrid_extractor = hybrid_extractor

    # ── 1. myelin_observe ─────────────────────────────────────

    async def observe(
        self,
        agent_id: str,
        session_id: str,
        action: str,
        action_type: str,
        content_text: str,
        input_context: dict | None = None,
        output_result: dict | None = None,
        success: bool = True,
        domain: str | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        embedding = self.embedder.embed(content_text) or None
        episode = self._build_episode(
            agent_id=agent_id,
            session_id=session_id,
            action=action,
            action_type=action_type,
            content_text=content_text,
            input_context=input_context,
            output_result=output_result,
            success=success,
            domain=domain,
            tags=tags,
            embedding=embedding,
        )

        episode_id = self._record_episode(episode)

        return {
            "episode_id": episode_id,
            "status": "recorded",
            "total_episodes": self.episodic.count(agent_id),
        }

    # ── 2. myelin_observe_batch ───────────────────────────────

    async def observe_batch(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        """Record many observations in one transaction."""
        started = time.perf_counter()
        failed: list[dict[str, Any]] = []
        prepared: list[Episode] = []
        texts: list[str] = []

        required = {"agent_id", "session_id", "action", "action_type", "content_text"}
        failed_indices: set[int] = set()
        for index, event in enumerate(events):
            missing = sorted(required - set(event))
            if missing:
                failed_indices.add(index)
                failed.append(
                    {
                        "index": index,
                        "error": f"Missing required fields: {', '.join(missing)}",
                    }
                )
                continue
            texts.append(str(event["content_text"]))

        embeddings = self.embedder.embed_batch(texts) if texts else []
        embedding_index = 0
        for index, event in enumerate(events):
            if index in failed_indices:
                continue
            embedding = embeddings[embedding_index] if embedding_index < len(embeddings) else []
            embedding_index += 1
            try:
                prepared.append(
                    self._build_episode(
                        agent_id=str(event["agent_id"]),
                        session_id=str(event["session_id"]),
                        action=str(event["action"]),
                        action_type=str(event["action_type"]),
                        content_text=str(event["content_text"]),
                        input_context=event.get("input_context"),
                        output_result=event.get("output_result"),
                        success=bool(event.get("success", True)),
                        domain=event.get("domain"),
                        tags=event.get("tags"),
                        embedding=embedding or None,
                    )
                )
            except (TypeError, ValueError) as exc:
                failed_indices.add(index)
                failed.append({"index": index, "error": str(exc)})

        episode_ids: list[str] = []
        if prepared:
            with self.db.transaction():
                for episode in prepared:
                    episode_ids.append(self._record_episode(episode))

        return {
            "recorded": len(episode_ids),
            "failed": failed,
            "episode_ids": episode_ids,
            "duration_ms": round((time.perf_counter() - started) * 1000, 3),
        }

    def _build_episode(
        self,
        agent_id: str,
        session_id: str,
        action: str,
        action_type: str,
        content_text: str,
        input_context: dict | None = None,
        output_result: dict | None = None,
        success: bool = True,
        domain: str | None = None,
        tags: list[str] | None = None,
        embedding: list[float] | None = None,
    ) -> Episode:
        episode = Episode(
            agent_id=agent_id,
            session_id=session_id,
            action=action,
            action_type=ActionType(action_type),
            content_text=content_text,
            input_context=input_context,
            output_result=output_result,
            success=success,
            embedding=embedding,
            domain=domain,
            tags=tags or [],
        )
        return episode

    def _record_episode(self, episode: Episode) -> str:
        episode_id = self.episodic.record(episode)

        self.entities.process_episode(
            episode_id=episode_id,
            content_text=episode.content_text,
            action=episode.action,
            action_type=episode.action_type.value,
            domain=episode.domain,
        )

        self.profiler.learn_from_episode(
            {
                "agent_id": episode.agent_id,
                "action": episode.action,
                "content_text": episode.content_text,
            }
        )

        self.user_profiler.learn_from_episode(
            {
                "agent_id": episode.agent_id,
                "action": episode.action,
                "content_text": episode.content_text,
            }
        )

        if episode.domain:
            self.confidence_map.update_domain(episode.domain, episode_delta=1)

        self._check_learning_goals(episode.domain, episode.agent_id)

        return episode_id

    # ── 3. myelin_recall ──────────────────────────────────────

    async def recall(
        self,
        query: str,
        limit: int = 10,
        memory_types: list[str] | None = None,
        domain: str | None = None,
        min_confidence: float = 0.0,
    ) -> dict[str, Any]:
        types = memory_types or ["episodic", "semantic", "procedural"]
        query_vec = self.embedder.embed(query) or None
        results: dict[str, list] = {}

        if "episodic" in types:
            episodes = self.episodic.search_hybrid(query, query_vec, limit=limit)
            for ep in episodes:
                self.episodic.access(ep["id"])
            results["episodes"] = episodes

        if "semantic" in types:
            nodes = self.semantic.search_hybrid(query, query_vec, limit=limit)
            for node in nodes:
                self.semantic.access(node["id"])
            results["semantic"] = [n for n in nodes if n.get("confidence", 0) >= min_confidence]

        if "procedural" in types:
            procedures = self.procedural.find_matching(
                query, query_vec, limit=limit, min_confidence=min_confidence
            )
            results["procedures"] = procedures

        return {
            "query": query,
            "results": results,
            "total_results": sum(len(v) for v in results.values()),
        }

    # ── 3. myelin_context ─────────────────────────────────────

    async def context(
        self,
        query: str,
        domain: str | None = None,
        agent_id: str | None = None,
        max_memories: int = 10,
        max_procedures: int = 3,
        agent_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Assemble complete context for the current situation."""
        return self.assembler.assemble(
            query=query,
            domain=domain,
            agent_id=agent_id,
            max_memories=max_memories,
            max_procedures=max_procedures,
            agent_ids=agent_ids,
        )

    # ── 4. myelin_execute_procedure ───────────────────────────

    async def execute_procedure(
        self,
        query: str,
        agent_id: str,
        context: dict | None = None,
    ) -> dict[str, Any]:
        query_vec = self.embedder.embed(query) or None
        matches = self.procedural.find_matching(query, query_vec, limit=3)

        if not matches:
            return {
                "found": False,
                "message": "No matching procedure found.",
                "suggestion": "Try myelin_context for broader intelligence.",
            }

        best = matches[0]
        steps = best["steps"] if isinstance(best["steps"], list) else json.loads(best["steps"])
        trust_level = procedure_trust_level(
            best["confidence"],
            best.get("success_count", 0),
            best.get("failure_count", 0),
        )

        return {
            "found": True,
            "procedure_id": best["id"],
            "name": best["name"],
            "confidence": best["confidence"],
            "trust_level": trust_level,
            "recommendation": procedure_recommendation(trust_level),
            "calibration_offset": best.get("calibration_offset", 0.0),
            "steps": steps,
            "preconditions": best.get("preconditions", []),
            "alternatives": [
                {"id": m["id"], "name": m["name"], "confidence": m["confidence"]}
                for m in matches[1:]
            ],
        }

    # ── 5. myelin_procedure_feedback ──────────────────────────

    async def procedure_feedback(
        self,
        procedure_id: str,
        success: bool,
        modifications: list[dict] | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        new_confidence = self.procedural.record_execution(procedure_id, success)

        if modifications:
            self.procedural.record_modification(procedure_id, modifications)

        # Recalculate trust state based on accumulated evidence
        trust_state = self.procedural.update_trust_state(procedure_id)

        proc = self.procedural.get(procedure_id)
        trust_level = (
            procedure_trust_level(
                new_confidence,
                proc["success_count"] if proc else 0,
                proc["failure_count"] if proc else 0,
            )
            if proc
            else "unknown"
        )
        return {
            "procedure_id": procedure_id,
            "new_confidence": new_confidence,
            "trust_level": trust_level,
            "trust_state": trust_state,
            "recommendation": procedure_recommendation(trust_level)
            if trust_level != "unknown"
            else "procedure_not_found",
            "success_count": proc["success_count"] if proc else 0,
            "failure_count": proc["failure_count"] if proc else 0,
            "status": proc["status"] if proc else "unknown",
        }

    # ── 6. myelin_confidence ──────────────────────────────────

    async def confidence(
        self,
        domain: str | None = None,
        procedure_id: str | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}

        if procedure_id:
            proc = self.procedural.get(procedure_id)
            if proc:
                result["procedure"] = {
                    "id": proc["id"],
                    "name": proc["name"],
                    "confidence": proc["confidence"],
                    "trust_level": procedure_trust_level(
                        proc["confidence"],
                        proc["success_count"],
                        proc["failure_count"],
                    ),
                    "actual_success_rate": proc.get("actual_success_rate"),
                    "calibration_offset": proc.get("calibration_offset", 0.0),
                    "executions": proc["success_count"] + proc["failure_count"],
                }

        if domain:
            row = self.db.fetchone("SELECT * FROM confidence_map WHERE domain = ?", (domain,))
            if row:
                result["domain"] = dict(row)

            procedures = self.procedural.get_by_domain(domain)
            result["domain_procedures"] = len(procedures)

        if not domain and not procedure_id:
            domains = self.db.fetchall(
                "SELECT * FROM confidence_map ORDER BY confidence DESC LIMIT 20"
            )
            result["all_domains"] = [dict(d) for d in domains]
            result["total_procedures"] = self.procedural.count()
            result["active_procedures"] = self.procedural.count(ProcedureStatus.ACTIVE)

        return result

    # ── 7. myelin_teach ───────────────────────────────────────

    async def teach(
        self,
        name: str,
        trigger_pattern: str,
        steps: list[dict],
        agent_id: str,
        description: str | None = None,
        preconditions: list[str] | None = None,
        postconditions: list[str] | None = None,
        domain: str | None = None,
    ) -> dict[str, Any]:
        trigger_embedding = self.embedder.embed(trigger_pattern) or None

        procedure = Procedure(
            name=name,
            description=description,
            trigger_pattern=trigger_pattern,
            trigger_embedding=trigger_embedding,
            steps=[
                ProcedureStep(
                    order=i,
                    description=s.get("description", s.get("step", "")),
                    step_type=StepType(s.get("type", "core")),
                    variants=s.get("variants", []),
                    condition=s.get("condition"),
                )
                for i, s in enumerate(steps)
            ],
            preconditions=preconditions or [],
            postconditions=postconditions or [],
            confidence=0.7,
            source_agent=agent_id,
            promotion_method=PromotionMethod.TAUGHT,
            status=ProcedureStatus.ACTIVE,
            domain=domain,
        )

        proc_id = self.procedural.store(procedure)

        return {
            "procedure_id": proc_id,
            "name": name,
            "status": "taught",
            "confidence": 0.7,
            "steps_count": len(steps),
        }

    # ── 8. myelin_status ──────────────────────────────────────

    async def status(self, agent_id: str | None = None) -> dict[str, Any]:
        total_episodes = self.episodic.count(agent_id)
        total_semantic = self.semantic.count()
        total_procedures = self.procedural.count()
        active_procedures = self.procedural.count(ProcedureStatus.ACTIVE)

        goals = self.db.fetchall(
            "SELECT * FROM learning_goals WHERE status = ?",
            (GoalStatus.ACTIVE.value,),
        )

        last_run = self.db.fetchone("SELECT * FROM process_runs ORDER BY started_at DESC LIMIT 1")

        return {
            "episodes": total_episodes,
            "semantic_nodes": total_semantic,
            "procedures": {
                "total": total_procedures,
                "active": active_procedures,
                "draft": self.procedural.count(ProcedureStatus.DRAFT),
                "archived": self.procedural.count(ProcedureStatus.ARCHIVED),
            },
            "entities": self.entities.count(),
            "relationships": self.graph.count_relationships(),
            "temporal_states": self.temporal.count(),
            "learning_goals": len(goals),
            "last_cognitive_process": dict(last_run) if last_run else None,
        }

    # ── 9. myelin_query ───────────────────────────────────────

    async def query(
        self,
        query: str,
        limit: int = 10,
        domain: str | None = None,
        weights: dict[str, float] | None = None,
        synthesize: bool = False,
        agent_ids: list[str] | None = None,
        agent_id: str | None = None,
    ) -> dict[str, Any]:
        """Multi-signal retrieval across all memory types."""
        query_vec = self.embedder.embed(query) or None
        results = self.retriever.retrieve(
            query,
            query_embedding=query_vec,
            domain=domain,
            limit=limit,
            weights=weights,
            agent_ids=agent_ids,
            querying_agent_id=agent_id,
        )
        raw_results = [
            {
                "id": r.get("id"),
                "source_type": r.get("_source_type"),
                "content": r.get("content_text") or r.get("content") or r.get("name", ""),
                "composite_score": r.get("_composite_score", 0),
                "scores": r.get("_scores", {}),
                "source_agent": r.get("source_agent", "unknown"),
            }
            for r in results
        ]

        if synthesize and self.synthesizer:
            return self.synthesizer.synthesize(
                query=query,
                results=raw_results,
            )

        return {
            "query": query,
            "results": raw_results,
            "total": len(results),
        }

    # ── 10. myelin_graph_query ────────────────────────────────

    async def graph_query(
        self,
        entity_name: str | None = None,
        entity_id: str | None = None,
        direction: str = "both",
        relation_types: list[str] | None = None,
        max_depth: int = 2,
    ) -> dict[str, Any]:
        """Explore the knowledge graph around an entity."""
        eid = entity_id
        if not eid and entity_name:
            found = self.entities.search(entity_name)
            if found:
                eid = found[0]["id"]

        if not eid:
            return {"found": False, "error": "Entity not found"}

        entity = self.entities.get_entity(eid)
        neighbors = self.graph.get_neighbors(
            eid, relation_types=relation_types, direction=direction, limit=20
        )
        subgraph = self.graph.bfs_subgraph(eid, max_depth=max_depth, max_nodes=30)

        return {
            "found": True,
            "entity": {
                "id": eid,
                "name": entity["canonical_name"] if entity else "",
                "type": entity["entity_type"] if entity else "",
                "mention_count": entity.get("mention_count", 0) if entity else 0,
            },
            "neighbors": [
                {
                    "id": n["id"],
                    "name": n.get("canonical_name", ""),
                    "type": n.get("entity_type", ""),
                    "relation": n.get("relation_type", ""),
                    "strength": n.get("strength", 1.0),
                }
                for n in neighbors
            ],
            "subgraph": {
                "node_count": len(subgraph["nodes"]),
                "edge_count": len(subgraph["edges"]),
                "nodes": [
                    {"id": n["id"], "name": n.get("canonical_name", "")} for n in subgraph["nodes"]
                ],
            },
        }

    # ── 11. myelin_temporal ───────────────────────────────────

    async def temporal_query(
        self,
        entity_name: str | None = None,
        entity_id: str | None = None,
        domain: str | None = None,
    ) -> dict[str, Any]:
        """Query temporal state of entities or domains."""
        if domain:
            states = self.temporal.get_current_states_for_domain(domain)
            return {
                "domain": domain,
                "current_states": [
                    {
                        "entity_id": s.get("entity_id"),
                        "state": s["state_description"],
                        "since": s.get("valid_from"),
                        "confidence": s.get("confidence", 0.5),
                    }
                    for s in states
                ],
            }

        eid = entity_id
        if not eid and entity_name:
            found = self.entities.search(entity_name)
            if found:
                eid = found[0]["id"]

        if not eid:
            return {"found": False, "error": "Entity not found"}

        current = self.temporal.get_current_state(eid)
        history = self.temporal.get_state_history(eid)
        transitions = self.temporal.get_state_transitions(eid)

        return {
            "found": True,
            "entity_id": eid,
            "current_state": {
                "description": current["state_description"],
                "since": current.get("valid_from"),
                "confidence": current.get("confidence", 0.5),
            }
            if current
            else None,
            "history": [
                {
                    "state": h["state_description"],
                    "from": h["valid_from"],
                    "until": h["valid_until"],
                }
                for h in history[:10]
            ],
            "transitions": [
                {
                    "from_state": t["from_state"],
                    "to_state": t["to_state"],
                    "when": t.get("changed_at"),
                }
                for t in transitions[:10]
            ],
        }

    # ── 12. myelin_what_changed ───────────────────────────────

    async def what_changed(
        self,
        domain: str,
        since: str,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Return state changes in a domain since a timestamp."""
        transitions = self.temporal.get_domain_transitions_since(domain, since, limit=limit)

        if not transitions:
            return {
                "found": True,
                "domain": domain,
                "since": since,
                "change_count": 0,
                "changes": [],
                "markdown": f"## No changes\nNo temporal changes found for `{domain}` since `{since}`.",
            }

        rows = []
        for t in transitions:
            rows.append(
                {
                    "entity": t.get("entity_name") or t.get("entity_id") or "unknown",
                    "from": t.get("from_state") or "(initial state)",
                    "to": t.get("to_state"),
                    "changed_at": t.get("changed_at"),
                    "confidence": t.get("confidence", 0.5),
                }
            )

        markdown_rows = [
            "## Temporal Changes",
            f"Domain: `{domain}` · Since: `{since}`",
            "",
            "| Entity | From | To | Changed At | Confidence |",
            "| --- | --- | --- | --- | --- |",
        ]
        for row in rows:
            from_state = str(row["from"]).replace("|", "\\|")
            to_state = str(row["to"]).replace("|", "\\|")
            entity = str(row["entity"]).replace("|", "\\|")
            markdown_rows.append(
                f"| {entity} | {from_state} | {to_state} | {row['changed_at']} | "
                f"{float(row['confidence']):.2f} |"
            )

        return {
            "found": True,
            "domain": domain,
            "since": since,
            "change_count": len(rows),
            "changes": rows,
            "markdown": "\n".join(markdown_rows),
        }

    async def entity_status(self, entity_name: str) -> dict[str, Any]:
        """Get current state and recent transitions for an entity."""
        found = self.entities.search(entity_name)
        if not found:
            return {
                "found": False,
                "entity_name": entity_name,
                "error": f"Entity '{entity_name}' not found",
                "markdown": f"## Entity Not Found\nNo entity matching `{entity_name}` exists in the knowledge graph.",
            }

        entity = found[0]
        eid = entity["id"]
        current = self.temporal.get_current_state(eid)
        history = self.temporal.get_state_history(eid, limit=5)
        transitions = self.temporal.get_state_transitions(eid)[:5]

        current_entry = None
        if current:
            current_entry = {
                "description": current["state_description"],
                "since": current.get("valid_from"),
                "confidence": current.get("confidence", 0.5),
            }

        markdown = [
            f"## Entity Status: `{entity['canonical_name']}`",
            f"Type: `{entity.get('entity_type', 'unknown')}`  |  Mentions: `{entity.get('mention_count', 0)}`",
            "",
            "### Current",
            (
                f"- **State:** {current_entry['description']} (since {current_entry['since']}, "
                f"confidence {current_entry['confidence']:.2f})"
                if current_entry
                else "- **State:** unknown (no active state recorded)"
            ),
            "",
            "### Recent Transitions",
            "| From | To | When |",
            "| --- | --- | --- |",
        ]

        for t in transitions:
            from_state = str(t.get("from_state")).replace("|", "\\|")
            to_state = str(t.get("to_state")).replace("|", "\\|")
            markdown.append(f"| {from_state} | {to_state} | {t.get('changed_at')} |")

        return {
            "found": True,
            "entity": {
                "id": entity["id"],
                "name": entity["canonical_name"],
                "type": entity.get("entity_type", "unknown"),
                "mention_count": entity.get("mention_count", 0),
            },
            "current_state": current_entry,
            "recent_history": [
                {
                    "state": h["state_description"],
                    "from": h.get("valid_from"),
                    "until": h.get("valid_until"),
                }
                for h in history
            ],
            "transitions": [
                {
                    "from_state": t["from_state"],
                    "to_state": t["to_state"],
                    "when": t.get("changed_at"),
                    "confidence": t.get("confidence", 0.5),
                }
                for t in transitions
            ],
            "markdown": "\n".join(markdown),
        }

    # ── 13. myelin_entities_query ───────────────────────────────

    async def entities_query(
        self,
        search: str | None = None,
        entity_type: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Search and browse extracted entities."""
        if search:
            results = self.entities.search(search)
            return {
                "query": search,
                "entities": [
                    {
                        "id": e["id"],
                        "name": e["canonical_name"],
                        "type": e["entity_type"],
                        "mention_count": e.get("mention_count", 1),
                    }
                    for e in results[:limit]
                ],
            }

        top = self.entities.get_top_entities(limit=limit)
        if entity_type:
            top = [e for e in top if e.get("entity_type") == entity_type]

        return {
            "entities": [
                {
                    "id": e["id"],
                    "name": e["canonical_name"],
                    "type": e["entity_type"],
                    "mention_count": e.get("mention_count", 1),
                }
                for e in top
            ],
            "total": self.entities.count(),
        }

    # ── 13. myelin_transfer_export ────────────────────────────

    async def transfer_export(
        self,
        procedure_id: str,
        source_agent: str,
        target_agent: str,
    ) -> dict[str, Any]:
        """Package a procedure for transfer to another agent."""
        return self.transfer.export_procedure(procedure_id, source_agent, target_agent)

    # ── 14. myelin_transfer_import ────────────────────────────

    async def transfer_import(
        self,
        package: dict[str, Any],
        agent_id: str,
    ) -> dict[str, Any]:
        """Import a procedure from another agent."""
        return self.transfer.import_procedure(package, agent_id)

    # ── 15. myelin_transfer_discover ──────────────────────────

    async def transfer_discover(
        self,
        source_agent: str,
        target_agent: str,
        min_confidence: float = 0.6,
    ) -> dict[str, Any]:
        """Discover transferable procedures between agents."""
        available = self.transfer.get_transferable_procedures(
            source_agent, target_agent, min_confidence
        )
        return {
            "source_agent": source_agent,
            "target_agent": target_agent,
            "transferable_procedures": available,
            "count": len(available),
        }

    # ── 16. myelin_visualize ──────────────────────────────────

    async def visualize(
        self,
        entity_name: str | None = None,
        format: str = "mermaid",
        depth: int = 2,
    ) -> dict[str, Any]:
        """Export the knowledge graph as Mermaid.js or D3.js JSON."""
        viz = Visualizer(self.entities, self.graph)
        output_format = format.lower()

        if output_format == "mermaid":
            mermaid_result = viz.export_mermaid(entity_name, depth)
            return {
                "format": "mermaid",
                "mermaid": mermaid_result,
                "markdown": f"```mermaid\n{mermaid_result}```",
                "node_count": sum(1 for _ in mermaid_result.split("\n") if "-->" in _ or '["' in _),
            }
        elif output_format in ("d3_json", "d3json", "d3"):
            graph_result = viz.export_d3_json(entity_name, depth)
            return {
                "format": "d3_json",
                "graph": graph_result,
                "node_count": len(graph_result.get("nodes", [])),
                "edge_count": len(graph_result.get("links", [])),
            }
        else:
            return {
                "error": f"Unknown format '{format}'. Supported: mermaid, d3_json",
            }

    # ── 17. myelin_sleep ──────────────────────────────────────

    async def trigger_sleep(self, agent_id: str | None = None) -> dict[str, Any]:
        """Trigger sleep consolidation and procedure promotion manually."""
        from ..cognitive.promoter import Promoter
        from ..cognitive.sleep import SleepCycle

        try:
            sleep = SleepCycle(self.db, hybrid_extractor=self.hybrid_extractor)
            sleep_result = await sleep.run()
        except Exception as e:
            sleep_result = {
                "error": str(e),
                "entities_extracted": 0,
                "relationships_created": 0,
                "entities_merged": 0,
                "temporal_states_updated": 0,
                "cross_domain_links": 0,
                "stale_flagged": 0,
                "importance_scores_updated": 0,
                "nrem": {},
                "rem": {},
            }

        try:
            promoter = Promoter(self.db, self.episodic, self.procedural)
            promoter_result = await promoter.run()
        except Exception as e:
            promoter_result = {"error": str(e), "processed": 0, "created": 0}

        return {
            "status": "completed",
            "process": "sleep",
            **sleep_result,
            "sleep": sleep_result,
            "promoter": promoter_result,
        }

    # ── 18. myelin_profile ─────────────────────────────────────

    async def profile(self, agent_id: str) -> dict[str, Any]:
        """Return the learned user profile for an agent.

        Returns static (stable preferences) and dynamic (recent context)
        profile sections with confidence scores and category breakdown.
        """
        result = self.user_profiler.get_profile(agent_id)
        result["markdown"] = self._format_profile_markdown(result)
        return result

    # ── 19. myelin_memorize ──────────────────────────────────

    async def memorize(
        self,
        agent_id: str,
        key: str,
        value: str,
        domain: str | None = None,
        ttl_days: int | None = None,
    ) -> dict[str, Any]:
        """Store a durable semantic fact (key-value). Upserts if agent_id+key already exists."""
        import uuid
        import datetime

        now = datetime.datetime.utcnow()

        # Check if fact already exists
        existing = self.db.fetchone(
            "SELECT id, value FROM semantic_facts WHERE agent_id = ? AND key = ? AND deleted_at IS NULL",
            (agent_id, key),
        )

        if existing:
            # Update existing fact
            updates: dict[str, Any] = {"value": value, "access_count": 0}
            if domain is not None:
                updates["domain"] = domain
            if ttl_days is not None:
                expiry = now + datetime.timedelta(days=ttl_days)
                updates["expires_at"] = expiry.strftime("%Y-%m-%dT%H:%M:%S")
            self.db.update("semantic_facts", existing["id"], updates)
            fact_id = existing["id"]
        else:
            # Insert new fact
            fact_id = str(uuid.uuid4())
            fact = {
                "id": fact_id,
                "agent_id": agent_id,
                "key": key,
                "value": value,
                "domain": domain,
                "created_at": now.strftime("%Y-%m-%dT%H:%M:%S"),
                "access_count": 0,
            }
            if ttl_days is not None:
                expiry = now + datetime.timedelta(days=ttl_days)
                fact["expires_at"] = expiry.strftime("%Y-%m-%dT%H:%M:%S")
            self.db.insert("semantic_facts", fact)

        return {"fact_id": fact_id, "status": "stored" if not existing else "updated"}

    # ── 20. myelin_update ────────────────────────────────────

    async def update(
        self,
        memory_id: str,
        memory_type: str = "episode",
        content_text: str | None = None,
        action: str | None = None,
        value: str | None = None,
    ) -> dict[str, Any]:
        """Update an existing memory (episodic observation or semantic fact) by ID."""
        if memory_type == "episode":
            existing = self.db.fetchone(
                "SELECT id FROM episodes WHERE id = ? AND deleted_at IS NULL",
                (memory_id,),
            )
            if not existing:
                return {"success": False, "error": f"Episode {memory_id} not found"}
            updates: dict[str, Any] = {}
            if content_text is not None:
                updates["content_text"] = content_text
            if action is not None:
                updates["action"] = action
            if not updates:
                return {"success": False, "error": "No fields to update"}
            self.db.update("episodes", memory_id, updates)
            return {"success": True, "memory_id": memory_id, "memory_type": "episode"}

        elif memory_type == "semantic":
            existing = self.db.fetchone(
                "SELECT id FROM semantic_facts WHERE id = ? AND deleted_at IS NULL",
                (memory_id,),
            )
            if not existing:
                return {"success": False, "error": f"Semantic fact {memory_id} not found"}
            if value is not None:
                self.db.update("semantic_facts", memory_id, {"value": value})
                return {"success": True, "memory_id": memory_id, "memory_type": "semantic"}
            return {"success": False, "error": "No fields to update"}

        return {"success": False, "error": f"Unknown memory_type: {memory_type}"}

    # ── 21. myelin_forget ────────────────────────────────────

    async def forget(
        self,
        memory_id: str,
        memory_type: str = "episode",
    ) -> dict[str, Any]:
        """Soft-delete a memory by ID. Marks as deleted rather than removing rows."""
        import datetime

        now = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")

        if memory_type == "episode":
            existing = self.db.fetchone(
                "SELECT id FROM episodes WHERE id = ? AND deleted_at IS NULL",
                (memory_id,),
            )
            if not existing:
                return {"success": False, "error": f"Episode {memory_id} not found or already deleted"}
            self.db.update("episodes", memory_id, {"deleted_at": now})

        elif memory_type == "semantic":
            existing = self.db.fetchone(
                "SELECT id FROM semantic_facts WHERE id = ? AND deleted_at IS NULL",
                (memory_id,),
            )
            if not existing:
                return {"success": False, "error": f"Semantic fact {memory_id} not found or already deleted"}
            self.db.update("semantic_facts", memory_id, {"deleted_at": now})

        elif memory_type == "procedure":
            existing = self.db.fetchone(
                "SELECT id FROM procedures WHERE id = ?",
                (memory_id,),
            )
            if not existing:
                return {"success": False, "error": f"Procedure {memory_id} not found"}
            self.db.update("procedures", memory_id, {"deleted_at": now})

        else:
            return {"success": False, "error": f"Unknown memory_type: {memory_type}"}

        return {"success": True, "memory_id": memory_id, "memory_type": memory_type}

    # ── 22. myelin_facts ─────────────────────────────────────

    async def facts(
        self,
        agent_id: str,
        key_prefix: str | None = None,
        domain: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Query stored semantic facts for an agent."""
        conditions = ["agent_id = ?", "deleted_at IS NULL"]
        params: list[Any] = [agent_id]

        if key_prefix:
            conditions.append("key LIKE ?")
            params.append(f"{key_prefix}%")

        if domain:
            conditions.append("domain = ?")
            params.append(domain)

        sql = f"SELECT * FROM semantic_facts WHERE {' AND '.join(conditions)} ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = self.db.fetchall(sql, tuple(params))

        import datetime
        now = datetime.datetime.utcnow()

        facts_list = []
        for row in rows:
            expired = False
            if row.get("expires_at"):
                try:
                    expiry = datetime.datetime.strptime(row["expires_at"], "%Y-%m-%dT%H:%M:%S")
                    expired = expiry < now
                except (ValueError, TypeError):
                    pass
            facts_list.append({
                "id": row["id"],
                "key": row["key"],
                "value": row["value"],
                "domain": row.get("domain"),
                "created_at": row.get("created_at"),
                "expired": expired,
                "expires_at": row.get("expires_at"),
                "access_count": row.get("access_count", 0),
            })

        return {"agent_id": agent_id, "facts": facts_list, "count": len(facts_list)}

    # ── Internal helpers ───────────────────────────────────────

    def _format_profile_markdown(self, profile: dict) -> str:
        lines = [
            f"## Profile: `{profile['agent_id']}`",
            f"Facts: {profile['fact_count']} ({profile['static_count']} static, {profile['dynamic_count']} dynamic)",
            "",
        ]
        if profile["static_facts"]:
            lines.append("### Static Facts (Stable Traits)")
            for f in profile["static_facts"]:
                cat = f["category"].title()
                conf = f["confidence"]
                lines.append(f"- [{cat}] {f['fact']} (confidence: {conf:.2f})")
            lines.append("")

        if profile["dynamic_context"]:
            lines.append("### Dynamic Context (Recent Activity)")
            for f in profile["dynamic_context"]:
                cat = f["category"].title()
                conf = f["confidence"]
                lines.append(f"- [{cat}] {f['fact']} (confidence: {conf:.2f})")
            lines.append("")

        if profile.get("category_breakdown"):
            lines.append("### Category Breakdown")
            for cat, count in sorted(profile["category_breakdown"].items()):
                lines.append(f"- **{cat.title()}**: {count} facts")
        return "\n".join(lines)

    def _check_learning_goals(self, domain: str | None, agent_id: str) -> None:
        if not domain:
            return
        goals = self.db.fetchall(
            "SELECT * FROM learning_goals WHERE domain = ? AND status = ?",
            (domain, GoalStatus.ACTIVE.value),
        )
        for goal in goals:
            new_count = goal["episodes_collected"] + 1
            updates: dict[str, Any] = {"episodes_collected": new_count}
            if new_count >= goal["episodes_needed"]:
                updates["status"] = GoalStatus.ACHIEVED.value
                updates["resolved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            self.db.update("learning_goals", goal["id"], updates)
