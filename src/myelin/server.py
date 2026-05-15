"""Myelin MCP server. Exposes tools for agent procedural learning and context."""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .core.database import Database
from .intelligence.context import ContextAssembler
from .intelligence.synthesizer import Synthesizer
from .knowledge.entities import EntityStore
from .knowledge.graph import KnowledgeGraph
from .knowledge.temporal import TemporalIndex
from .memory.embedding import get_embedding_provider
from .memory.episodic import EpisodicMemory
from .memory.procedural import ProceduralMemory
from .memory.retriever import MultiSignalRetriever
from .memory.semantic import SemanticMemory
from .metacognition.confidence import ConfidenceMap
from .tools.handlers import ToolHandlers
from .transfer.protocol import TransferProtocol


def _call_llm(endpoint: str, prompt: str) -> str:
    """Minimal LLM call using stdlib urllib."""
    import urllib.request
    import urllib.error

    data = json.dumps(
        {"model": "default", "messages": [{"role": "user", "content": prompt}]}
    ).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            # Attempt common response shapes (OpenAI-compatible / Ollama)
            if "choices" in payload:
                return str(payload["choices"][0]["message"]["content"])
            if "message" in payload:
                return str(payload["message"]["content"])
            if "response" in payload:
                return str(payload["response"])
            return str(payload)
    except urllib.error.URLError as exc:
        return f"[LLM error: {exc}]"
    except Exception as exc:
        return f"[LLM error: {exc}]"

TOOLS = [
    Tool(
        name="myelin_observe",
        description="Record an agent action as an episodic memory with automatic entity extraction. Call this for every significant action.",
        inputSchema={
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "Unique identifier for the calling agent",
                },
                "session_id": {"type": "string", "description": "Current session identifier"},
                "action": {"type": "string", "description": "What the agent did"},
                "action_type": {
                    "type": "string",
                    "enum": ["tool_call", "response", "error", "user_input"],
                },
                "content_text": {
                    "type": "string",
                    "description": "Full text description for search indexing",
                },
                "input_context": {"type": "object", "description": "What triggered this action"},
                "output_result": {"type": "object", "description": "What the action produced"},
                "success": {"type": "boolean", "default": True},
                "domain": {
                    "type": "string",
                    "description": "Inferred domain (e.g. 'deployment', 'testing')",
                },
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["agent_id", "session_id", "action", "action_type", "content_text"],
        },
    ),
    Tool(
        name="myelin_observe_batch",
        description="Record many agent actions in one transaction. Use this when orchestrators emit bursts of events from a workflow, team, or swarm.",
        inputSchema={
            "type": "object",
            "properties": {
                "events": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "agent_id": {
                                "type": "string",
                                "description": "Unique identifier for the acting agent",
                            },
                            "session_id": {
                                "type": "string",
                                "description": "Shared workflow/session identifier",
                            },
                            "action": {"type": "string", "description": "What the agent did"},
                            "action_type": {
                                "type": "string",
                                "enum": ["tool_call", "response", "error", "user_input"],
                            },
                            "content_text": {
                                "type": "string",
                                "description": "Full text description for search indexing",
                            },
                            "input_context": {"type": "object"},
                            "output_result": {"type": "object"},
                            "success": {"type": "boolean", "default": True},
                            "domain": {"type": "string"},
                            "tags": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": [
                            "agent_id",
                            "session_id",
                            "action",
                            "action_type",
                            "content_text",
                        ],
                    },
                }
            },
            "required": ["events"],
        },
    ),
    Tool(
        name="myelin_recall",
        description="Search across all memory types (episodic, semantic, procedural). Use myelin_context for richer results.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 10},
                "memory_types": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["episodic", "semantic", "procedural"]},
                },
                "domain": {"type": "string"},
                "min_confidence": {"type": "number", "default": 0.0},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="myelin_context",
        description="Assemble complete context for a situation: relevant memories, matching procedures, entity relationships, temporal state, confidence, and suggested actions. This is the primary intelligence tool.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What you need context for"},
                "domain": {"type": "string"},
                "agent_id": {"type": "string"},
                "max_memories": {"type": "integer", "default": 10},
                "max_procedures": {"type": "integer", "default": 3},
                "agent_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by agent IDs. Omit or ['*'] for all agents.",
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="myelin_execute_procedure",
        description="Find the best matching learned procedure for a task. Returns step-by-step instructions with confidence.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Description of what you want to do"},
                "agent_id": {"type": "string"},
                "context": {"type": "object"},
            },
            "required": ["query", "agent_id"],
        },
    ),
    Tool(
        name="myelin_procedure_feedback",
        description="Report whether a procedure execution succeeded or failed. Updates Bayesian confidence.",
        inputSchema={
            "type": "object",
            "properties": {
                "procedure_id": {"type": "string"},
                "success": {"type": "boolean"},
                "modifications": {"type": "array", "items": {"type": "object"}},
                "notes": {"type": "string"},
            },
            "required": ["procedure_id", "success"],
        },
    ),
    Tool(
        name="myelin_confidence",
        description="Query confidence levels across domains or for a specific procedure. Includes calibration data.",
        inputSchema={
            "type": "object",
            "properties": {
                "domain": {"type": "string"},
                "procedure_id": {"type": "string"},
            },
        },
    ),
    Tool(
        name="myelin_teach",
        description="Manually teach a procedure. Starts at 0.7 confidence (higher than auto-promoted).",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "trigger_pattern": {
                    "type": "string",
                    "description": "When to suggest this procedure",
                },
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "type": {"type": "string", "enum": ["core", "variant", "optional"]},
                            "variants": {"type": "array", "items": {"type": "string"}},
                            "condition": {"type": "string"},
                        },
                        "required": ["description"],
                    },
                },
                "agent_id": {"type": "string"},
                "description": {"type": "string"},
                "preconditions": {"type": "array", "items": {"type": "string"}},
                "postconditions": {"type": "array", "items": {"type": "string"}},
                "domain": {"type": "string"},
            },
            "required": ["name", "trigger_pattern", "steps", "agent_id"],
        },
    ),
    Tool(
        name="myelin_status",
        description="Get system status: episode counts, procedures, entities, relationships, temporal states, learning goals.",
        inputSchema={
            "type": "object",
            "properties": {"agent_id": {"type": "string"}},
        },
    ),
    Tool(
        name="myelin_query",
        description="Multi-signal retrieval fusing text search, vector similarity, entity graph, temporal recency, and ACT-R activation into a single ranked result.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 10},
                "domain": {"type": "string"},
                "weights": {
                    "type": "object",
                    "description": "Signal weights: text, vector, entity, temporal, activation (0-1 each)",
                },
                "synthesize": {
                    "type": "boolean",
                    "default": False,
                    "description": "When true, synthesize top results into a concise answer with citations",
                },
                "agent_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by agent IDs. Omit or ['*'] for all agents.",
                },
                "agent_id": {
                    "type": "string",
                    "description": "Querying agent ID for confidence calibration.",
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="myelin_graph_query",
        description="Explore the knowledge graph around an entity. Returns neighbors, relationships, and subgraph structure.",
        inputSchema={
            "type": "object",
            "properties": {
                "entity_name": {"type": "string", "description": "Entity name to search for"},
                "entity_id": {"type": "string", "description": "Or provide entity ID directly"},
                "direction": {"type": "string", "enum": ["in", "out", "both"], "default": "both"},
                "relation_types": {"type": "array", "items": {"type": "string"}},
                "max_depth": {"type": "integer", "default": 2},
            },
        },
    ),
    Tool(
        name="myelin_temporal",
        description="Query temporal state of entities or domains. Shows current state, history, and transitions over time.",
        inputSchema={
            "type": "object",
            "properties": {
                "entity_name": {"type": "string"},
                "entity_id": {"type": "string"},
                "domain": {"type": "string"},
            },
        },
    ),
    Tool(
        name="myelin_what_changed",
        description="Get state transitions in a domain since a timestamp. Useful for answering what changed and when.",
        inputSchema={
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Domain to query (e.g. infrastructure, deployment)."},
                "since": {
                    "type": "string",
                    "description": "ISO timestamp or date from which to show changes (e.g. 2026-05-14 or 2026-05-14T09:00:00)",
                },
            },
            "required": ["domain", "since"],
        },
    ),
    Tool(
        name="myelin_entity_status",
        description="Get current state and recent transitions for a named entity.",
        inputSchema={
            "type": "object",
            "properties": {
                "entity_name": {"type": "string", "description": "Canonical or alias entity name."},
            },
            "required": ["entity_name"],
        },
    ),
    Tool(
        name="myelin_entities",
        description="Search and browse extracted entities (tools, services, files, errors, concepts).",
        inputSchema={
            "type": "object",
            "properties": {
                "search": {"type": "string", "description": "Search term"},
                "entity_type": {
                    "type": "string",
                    "enum": [
                        "tool",
                        "service",
                        "concept",
                        "file",
                        "person",
                        "config",
                        "error",
                        "command",
                        "pattern",
                    ],
                },
                "limit": {"type": "integer", "default": 20},
            },
        },
    ),
    Tool(
        name="myelin_transfer_export",
        description="Package a learned procedure for transfer to another agent. Adapts steps based on target agent capabilities.",
        inputSchema={
            "type": "object",
            "properties": {
                "procedure_id": {"type": "string"},
                "source_agent": {"type": "string"},
                "target_agent": {"type": "string"},
            },
            "required": ["procedure_id", "source_agent", "target_agent"],
        },
    ),
    Tool(
        name="myelin_transfer_import",
        description="Import a procedure from another agent. Creates a draft procedure with discounted confidence.",
        inputSchema={
            "type": "object",
            "properties": {
                "package": {
                    "type": "object",
                    "description": "Transfer package from myelin_transfer_export",
                },
                "agent_id": {"type": "string", "description": "Receiving agent ID"},
            },
            "required": ["package", "agent_id"],
        },
    ),
    Tool(
        name="myelin_transfer_discover",
        description="Discover procedures that could be transferred between agents. Shows compatibility scores.",
        inputSchema={
            "type": "object",
            "properties": {
                "source_agent": {"type": "string"},
                "target_agent": {"type": "string"},
                "min_confidence": {"type": "number", "default": 0.6},
            },
            "required": ["source_agent", "target_agent"],
        },
    ),
    Tool(
        name="myelin_visualize",
        description="Export the knowledge graph as Mermaid.js diagram or D3.js force-directed JSON. Make the brain visible.",
        inputSchema={
            "type": "object",
            "properties": {
                "entity_name": {
                    "type": "string",
                    "description": "Optional: focus on a specific entity and its neighborhood. Omit for full graph.",
                },
                "format": {
                    "type": "string",
                    "enum": ["mermaid", "d3_json", "d3"],
                    "default": "mermaid",
                    "description": "Output format: mermaid (diagram-as-code), d3_json (force-directed graph JSON)",
                },
                "depth": {
                    "type": "integer",
                    "default": 2,
                    "description": "Traversal depth when entity_name is set (1=direct neighbors, 2=two hops).",
                },
            },
        },
    ),
    Tool(
        name="myelin_sleep",
        description="Trigger a sleep consolidation cycle: entity extraction, relationship inference, graph merging, temporal updates, cross-domain linking, and staleness detection.",
        inputSchema={
            "type": "object",
            "properties": {"agent_id": {"type": "string"}},
        },
    ),
    Tool(
        name="myelin_profile",
        description="Get the learned user profile for an agent. Returns static (stable preferences, habits) and dynamic (recent activity, current projects) sections with confidence scores.",
        inputSchema={
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "Agent ID to get profile for",
                },
            },
            "required": ["agent_id"],
        },
    ),
]


def create_server(
    db_path: str | None = None,
    embedding_provider: str = "none",
    llm_extraction: str | None = None,
    synthesis_model: str | None = None,
) -> Server:
    server = Server("myelin")
    db = Database(db_path)
    # Provider is resolved now, but LocalEmbedding won't load the model until
    # the first embed() call (lazy loading).
    embedder = get_embedding_provider(embedding_provider)

    episodic = EpisodicMemory(db)
    semantic = SemanticMemory(db)
    procedural = ProceduralMemory(db)

    entity_store = EntityStore(db)
    graph = KnowledgeGraph(db)
    temporal = TemporalIndex(db)
    confidence_map = ConfidenceMap(db)

    from .knowledge.concept_extractor import ConceptExtractor
    from .knowledge.entities import HybridEntityExtractor

    if llm_extraction:
        extractor = ConceptExtractor(provider=llm_extraction)
    else:
        extractor = None

    hybrid_extractor = HybridEntityExtractor(
        llm_extract=extractor.extract_concepts if extractor else None
    )

    if synthesis_model:
        llm = lambda prompt: _call_llm(synthesis_model, prompt)
        synthesizer = Synthesizer(llm_complete=llm)
    else:
        synthesizer = Synthesizer()

    retriever = MultiSignalRetriever(db, entity_store, graph, temporal)
    assembler = ContextAssembler(
        db, retriever, entity_store, graph, temporal, procedural, confidence_map, embedder
    )
    transfer = TransferProtocol(db, procedural)

    handlers = ToolHandlers(
        episodic,
        semantic,
        procedural,
        embedder,
        entity_store=entity_store,
        graph=graph,
        temporal=temporal,
        retriever=retriever,
        context_assembler=assembler,
        transfer_protocol=transfer,
        confidence_map=confidence_map,
        synthesizer=synthesizer,
        hybrid_extractor=hybrid_extractor,
    )

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return TOOLS

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        handler_map: dict[str, Any] = {
            "myelin_observe": handlers.observe,
            "myelin_observe_batch": handlers.observe_batch,
            "myelin_recall": handlers.recall,
            "myelin_context": handlers.context,
            "myelin_execute_procedure": handlers.execute_procedure,
            "myelin_procedure_feedback": handlers.procedure_feedback,
            "myelin_confidence": handlers.confidence,
            "myelin_teach": handlers.teach,
            "myelin_status": handlers.status,
            "myelin_query": handlers.query,
            "myelin_graph_query": handlers.graph_query,
            "myelin_temporal": handlers.temporal_query,
            "myelin_what_changed": handlers.what_changed,
            "myelin_entity_status": handlers.entity_status,
            "myelin_entities": handlers.entities_query,
            "myelin_transfer_export": handlers.transfer_export,
            "myelin_transfer_import": handlers.transfer_import,
            "myelin_transfer_discover": handlers.transfer_discover,
            "myelin_sleep": handlers.trigger_sleep,
            "myelin_visualize": handlers.visualize,
            "myelin_profile": handlers.profile,
        }

        handler = handler_map.get(name)
        if not handler:
            return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]

        result = await handler(**arguments)
        return [TextContent(type="text", text=json.dumps(result, default=str))]

    return server


async def run_server(
    db_path: str | None = None,
    embedding_provider: str = "none",
    llm_extraction: str | None = None,
    synthesis_model: str | None = None,
):
    server = create_server(db_path, embedding_provider, llm_extraction, synthesis_model)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main():
    parser = argparse.ArgumentParser(description="Myelin MCP Server")
    parser.add_argument("--db", type=str, help="Path to SQLite database", default=None)
    parser.add_argument(
        "--embeddings",
        type=str,
        choices=["none", "local"],
        default="none",
        help=argparse.SUPPRESS,  # deprecated, use --embedding-model
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help=(
            "Embedding mode: 'none' (text search only), 'local' (torch nomic-embed), "
            "'onnx' (lazy int8-quantized ONNX), or 'api:<url>' (remote HTTP endpoint). "
            "Overrides --embeddings when set."
        ),
    )
    parser.add_argument(
        "--llm-extraction",
        type=str,
        default=None,
        help="LLM endpoint/config for concept extraction (e.g. http://localhost:11434/v1/chat/completions)",
    )
    parser.add_argument(
        "--synthesis-model",
        type=str,
        default=None,
        help="LLM endpoint for query-time synthesis (e.g. http://localhost:11434/v1/chat/completions)",
    )
    args = parser.parse_args()
    # --embedding-model takes precedence over deprecated --embeddings
    provider = args.embedding_model if args.embedding_model is not None else args.embeddings
    asyncio.run(run_server(args.db, provider, args.llm_extraction, args.synthesis_model))


if __name__ == "__main__":
    main()
