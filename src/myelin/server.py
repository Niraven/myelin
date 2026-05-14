"""Myelin MCP server. Exposes 7 tools for agent memory."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .core.database import Database
from .memory.embedding import get_embedding_provider
from .memory.episodic import EpisodicMemory
from .memory.procedural import ProceduralMemory
from .memory.semantic import SemanticMemory
from .tools.handlers import ToolHandlers

TOOLS = [
    Tool(
        name="myelin_observe",
        description="Record an agent action as an episodic memory. Call this for every significant action the agent takes.",
        inputSchema={
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "Unique identifier for the calling agent"},
                "session_id": {"type": "string", "description": "Current session identifier"},
                "action": {"type": "string", "description": "What the agent did"},
                "action_type": {"type": "string", "enum": ["tool_call", "response", "error", "user_input"]},
                "content_text": {"type": "string", "description": "Full text description for search indexing"},
                "input_context": {"type": "object", "description": "What triggered this action"},
                "output_result": {"type": "object", "description": "What the action produced"},
                "success": {"type": "boolean", "default": True},
                "domain": {"type": "string", "description": "Inferred domain (e.g. 'deployment', 'testing')"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["agent_id", "session_id", "action", "action_type", "content_text"],
        },
    ),
    Tool(
        name="myelin_recall",
        description="Search across all memory types (episodic, semantic, procedural) for relevant knowledge.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for"},
                "limit": {"type": "integer", "default": 10},
                "memory_types": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["episodic", "semantic", "procedural"]},
                    "description": "Which memory types to search. Default: all.",
                },
                "domain": {"type": "string"},
                "min_confidence": {"type": "number", "default": 0.0},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="myelin_execute_procedure",
        description="Find and return the best matching learned procedure for a task. Returns step-by-step instructions.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Description of what you want to do"},
                "agent_id": {"type": "string"},
                "context": {"type": "object", "description": "Current execution context"},
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
                "modifications": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Steps that were modified during execution",
                },
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
        description="Manually teach Myelin a procedure. Starts at 0.7 confidence (higher than auto-promoted).",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Procedure name"},
                "trigger_pattern": {"type": "string", "description": "When to suggest this procedure"},
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
        description="Get overall Myelin system status: episode counts, procedures, learning goals, last process run.",
        inputSchema={
            "type": "object",
            "properties": {
                "agent_id": {"type": "string"},
            },
        },
    ),
]


def create_server(db_path: str | None = None, embedding_provider: str = "none") -> Server:
    server = Server("myelin")
    db = Database(db_path)
    embedder = get_embedding_provider(embedding_provider)

    episodic = EpisodicMemory(db)
    semantic = SemanticMemory(db)
    procedural = ProceduralMemory(db)
    handlers = ToolHandlers(episodic, semantic, procedural, embedder)

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return TOOLS

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        handler_map = {
            "myelin_observe": handlers.observe,
            "myelin_recall": handlers.recall,
            "myelin_execute_procedure": handlers.execute_procedure,
            "myelin_procedure_feedback": handlers.procedure_feedback,
            "myelin_confidence": handlers.confidence,
            "myelin_teach": handlers.teach,
            "myelin_status": handlers.status,
        }

        handler = handler_map.get(name)
        if not handler:
            return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]

        result = await handler(**arguments)
        return [TextContent(type="text", text=json.dumps(result, default=str))]

    return server


async def run_server(db_path: str | None = None, embedding_provider: str = "none"):
    server = create_server(db_path, embedding_provider)
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
        help="Embedding provider (none=text search only, local=nomic-embed)",
    )
    args = parser.parse_args()
    asyncio.run(run_server(args.db, args.embeddings))


if __name__ == "__main__":
    main()
