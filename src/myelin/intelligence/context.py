"""Context assembly engine.

The thing that makes Myelin actually useful. When an agent needs to act,
this assembles the optimal context from every memory signal into a single
structured block the agent can consume directly.

No other system does this. mem0 returns flat search results. hermes-lcm
compresses context. Myelin *assembles* it: relevant memories, matching
procedures, entity relationships, temporal state, confidence assessment,
and suggested actions, all ranked and budgeted.
"""

from __future__ import annotations

import json
from typing import Any

from ..core.database import Database
from ..core.models import TrustState
from ..knowledge.entities import EntityStore, extract_entities_from_text
from ..knowledge.graph import KnowledgeGraph
from ..knowledge.temporal import TemporalIndex
from ..memory.embedding import EmbeddingProvider, NoOpEmbedding
from ..memory.procedural import ProceduralMemory
from ..memory.retriever import MultiSignalRetriever
from ..metacognition.confidence import ConfidenceMap


class ContextAssembler:
    """Builds complete context blocks from all memory signals."""

    def __init__(
        self,
        db: Database,
        retriever: MultiSignalRetriever,
        entity_store: EntityStore,
        graph: KnowledgeGraph,
        temporal: TemporalIndex,
        procedural: ProceduralMemory,
        confidence_map: ConfidenceMap,
        embedder: EmbeddingProvider | None = None,
    ):
        self.db = db
        self.retriever = retriever
        self.entities = entity_store
        self.graph = graph
        self.temporal = temporal
        self.procedural = procedural
        self.confidence = confidence_map
        self.embedder = embedder or NoOpEmbedding()

    def assemble(
        self,
        query: str,
        domain: str | None = None,
        agent_id: str | None = None,
        max_memories: int = 10,
        max_procedures: int = 3,
        max_entities: int = 8,
        include_graph: bool = True,
        include_temporal: bool = True,
        include_confidence: bool = True,
        agent_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Assemble a complete context block for the given query.

        Returns a structured dict with all relevant context an agent needs
        to make informed decisions.
        """
        query_embedding = self.embedder.embed(query) or None

        memories = self.retriever.retrieve(
            query,
            query_embedding=query_embedding,
            domain=domain,
            limit=max_memories,
            agent_ids=agent_ids,
            querying_agent_id=agent_id,
        )

        # Context shield: the retriever surfaces procedures too, so apply the
        # same validated/trusted + exact-domain gate to procedure memories to
        # keep them out of relevant_memories / assembled_text.
        allowed_trust = {TrustState.VALIDATED.value, TrustState.TRUSTED.value}
        memories = [
            m
            for m in memories
            if m.get("_source_type") != "procedure"
            or (
                m.get("trust_state") in allowed_trust
                and (domain is None or m.get("domain") == domain)
            )
        ]

        procedures = self._find_procedures(
            query, query_embedding, max_procedures, domain=domain, agent_ids=agent_ids
        )

        query_entities = extract_entities_from_text(query)
        entity_context = self._build_entity_context(
            query_entities, include_graph, include_temporal, max_entities
        )

        domain_confidence = None
        if include_confidence and domain:
            domain_confidence = self._get_domain_confidence(domain)

        suggested_actions = self._derive_suggestions(procedures, memories, entity_context)

        assembled_text = self._render_text(
            query, memories, procedures, entity_context, domain_confidence, suggested_actions
        )

        return {
            "query": query,
            "domain": domain,
            "relevant_memories": [
                {
                    "id": m.get("id"),
                    "source_type": m.get("_source_type", "unknown"),
                    "source_id": m.get("source_id", ""),
                    "source_timestamp": m.get("source_timestamp"),
                    "content": m.get("content_text") or m.get("content") or m.get("name", ""),
                    "score": m.get("_composite_score", 0.0),
                    "scores": m.get("_scores", {}),
                    "source_agent": m.get("source_agent", "unknown"),
                    "provenance": m.get("_provenance"),
                }
                for m in memories
            ],
            "matching_procedures": procedures,
            "entity_context": entity_context,
            "domain_confidence": domain_confidence,
            "suggested_actions": suggested_actions,
            "assembled_text": assembled_text,
            "stats": {
                "memories_retrieved": len(memories),
                "procedures_matched": len(procedures),
                "entities_resolved": len(entity_context),
            },
        }

    def _find_procedures(
        self,
        query: str,
        embedding: list[float] | None,
        limit: int,
        domain: str | None = None,
        agent_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        # Default-context safety gate (issue #2): only validated/trusted
        # procedures may enter assembled context. Candidate, seed, stale, and
        # archived/superseded procedures are excluded at the SQL layer.
        matches = self.procedural.find_matching(
            query,
            embedding,
            limit=limit,
            agent_ids=agent_ids,
            domain=domain,
            trust_states=[TrustState.VALIDATED.value, TrustState.TRUSTED.value],
        )
        results = []
        for m in matches:
            steps = m.get("steps", "[]")
            if isinstance(steps, str):
                try:
                    steps = json.loads(steps)
                except (json.JSONDecodeError, TypeError):
                    steps = []

            results.append(
                {
                    "id": m["id"],
                    "name": m["name"],
                    "confidence": m.get("confidence", 0.5),
                    "status": m.get("status", "draft"),
                    "trust_state": m.get("trust_state", TrustState.SEED.value),
                    "steps": steps,
                    "preconditions": _parse_json_field(m.get("preconditions", "[]")),
                    "postconditions": _parse_json_field(m.get("postconditions", "[]")),
                    "success_rate": m.get("actual_success_rate"),
                    "execution_count": (m.get("success_count", 0) + m.get("failure_count", 0)),
                }
            )
        return results

    def _build_entity_context(
        self,
        query_entities: list[dict[str, Any]],
        include_graph: bool,
        include_temporal: bool,
        max_entities: int,
    ) -> list[dict[str, Any]]:
        context = []
        for qe in query_entities[:max_entities]:
            found = self.entities.find_by_canonical(qe["canonical_name"], qe["entity_type"])
            if not found:
                continue

            entry: dict[str, Any] = {
                "id": found["id"],
                "name": found["canonical_name"],
                "type": found["entity_type"],
                "mention_count": found.get("mention_count", 1),
            }

            if include_graph:
                neighbors = self.graph.get_neighbors(found["id"], limit=5)
                entry["related_entities"] = [
                    {
                        "id": n["id"],
                        "name": n.get("canonical_name", ""),
                        "relation": n.get("relation_type", "related_to"),
                        "strength": n.get("strength", 1.0),
                    }
                    for n in neighbors
                ]

            if include_temporal:
                current_state = self.temporal.get_current_state(found["id"])
                if current_state:
                    entry["temporal_state"] = {
                        "description": current_state["state_description"],
                        "since": current_state.get("valid_from"),
                        "confidence": current_state.get("confidence", 0.5),
                    }

                transitions = self.temporal.get_state_transitions(found["id"])
                if transitions:
                    entry["recent_transitions"] = [
                        {
                            "from": t["from_state"],
                            "to": t["to_state"],
                            "when": t.get("changed_at"),
                        }
                        for t in transitions[:3]
                    ]

            context.append(entry)
        return context

    def _get_domain_confidence(self, domain: str) -> dict[str, Any] | None:
        row = self.db.fetchone("SELECT * FROM confidence_map WHERE domain = ?", (domain,))
        if not row:
            return None
        return {
            "domain": domain,
            "confidence": row["confidence"],
            "trend": row.get("trend", "stable"),
            "episode_count": row.get("episode_count", 0),
            "procedure_count": row.get("procedure_count", 0),
        }

    def _derive_suggestions(
        self,
        procedures: list[dict],
        memories: list[dict],
        entity_context: list[dict],
    ) -> list[str]:
        suggestions = []

        for proc in procedures:
            if proc["confidence"] >= 0.7:
                steps_preview = ""
                if proc["steps"]:
                    first_step = proc["steps"][0]
                    if isinstance(first_step, dict):
                        steps_preview = f": {first_step.get('description', '')}"
                    elif isinstance(first_step, str):
                        steps_preview = f": {first_step}"
                suggestions.append(
                    f"Follow procedure '{proc['name']}' "
                    f"(confidence: {proc['confidence']:.0%}){steps_preview}"
                )

        for entity in entity_context:
            state = entity.get("temporal_state")
            if state and state.get("confidence", 0) < 0.5:
                suggestions.append(
                    f"Verify state of {entity['name']}: '{state['description']}' has low confidence"
                )

        return suggestions

    def _render_text(
        self,
        query: str,
        memories: list[dict],
        procedures: list[dict],
        entity_context: list[dict],
        domain_confidence: dict | None,
        suggestions: list[str],
    ) -> str:
        sections = []

        if memories:
            lines = ["## Relevant Memories"]
            for m in memories[:5]:
                content = m.get("content_text") or m.get("content") or m.get("name", "")
                score = m.get("_composite_score", 0)
                source = m.get("_source_type", "")
                preview = content[:120].replace("\n", " ")
                lines.append(f"- [{source}] (score: {score:.2f}) {preview}")
            sections.append("\n".join(lines))

        if procedures:
            lines = ["## Matching Procedures"]
            for p in procedures:
                lines.append(
                    f"- **{p['name']}** (confidence: {p['confidence']:.0%}, "
                    f"executions: {p['execution_count']})"
                )
                for i, step in enumerate(p["steps"][:5]):
                    desc = step.get("description", step) if isinstance(step, dict) else step
                    lines.append(f"  {i + 1}. {desc}")
            sections.append("\n".join(lines))

        if entity_context:
            lines = ["## Entity Context"]
            for e in entity_context:
                line = f"- **{e['name']}** ({e['type']}, mentioned {e['mention_count']}x)"
                state = e.get("temporal_state")
                if state:
                    line += f" | current state: {state['description']}"
                lines.append(line)
                for rel in e.get("related_entities", [])[:3]:
                    lines.append(
                        f"  -> {rel['relation']} {rel['name']} (strength: {rel['strength']:.1f})"
                    )
            sections.append("\n".join(lines))

        if domain_confidence:
            sections.append(
                f"## Domain Confidence\n"
                f"- {domain_confidence['domain']}: {domain_confidence['confidence']:.0%} "
                f"({domain_confidence['trend']}, "
                f"{domain_confidence['episode_count']} episodes)"
            )

        if suggestions:
            lines = ["## Suggested Actions"]
            for s in suggestions:
                lines.append(f"- {s}")
            sections.append("\n".join(lines))

        if not sections:
            return "No relevant context found."

        return "\n\n".join(sections)


def _parse_json_field(value: Any) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return list(parsed) if isinstance(parsed, list) else []
        except (json.JSONDecodeError, TypeError):
            return []
    return []
