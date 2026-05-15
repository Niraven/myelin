# Myelin Launch Kit

## One-Liner

Myelin is a local-first procedural memory layer for AI agents: it watches repeated workflows, learns executable procedures, and improves them with feedback.

## Short Pitch

Most agent memory systems retrieve text. Myelin learns how work gets done. It observes agent actions, clusters repeated workflows, uses ClustalW-inspired sequence alignment to extract procedure steps, and exposes the learned workflow through MCP.

Sigil was the broad research prototype. Myelin is the focused product: procedural memory for AI agents.

## Demo Command

```bash
uv run --python /Users/niamamor/.local/bin/python3.11 --with-editable ".[dev]" \
  python examples/procedure_learning_demo.py
```

## Hacker News Draft

Title:

```text
Show HN: Myelin, procedural memory for AI agents
```

Post:

```text
Myelin is a local-first memory layer for AI agents, but the main idea is procedural memory rather than fact recall.

The demo observes five repeated deployment workflows, promotes the shared action sequence into an executable procedure, and updates confidence after success feedback. Core learning runs locally on SQLite/FTS5 and does not require an LLM call.

The architecture combines episodic, semantic, and procedural memory with entity extraction, temporal state, context assembly, and MCP tools.

The positioning is simple: mem0 remembers. Myelin learns.
```

## X / LinkedIn Draft

```text
Most agent memory stores facts.

Myelin learns procedures.

It watches repeated agent workflows, aligns action sequences with a ClustalW-inspired algorithm, promotes consensus steps into executable procedures, and updates confidence from feedback.

Local-first. SQLite. MCP-native. No LLM required for core learning.

Demo: python examples/procedure_learning_demo.py
```

## GitHub Repo Settings

Recommended description:

```text
Cognitive procedural memory for AI agents. MCP-native, local-first, SQLite-backed.
```

Recommended topics:

```text
ai-agents, mcp, memory, procedural-memory, cognitive-architecture, sqlite, agent-framework, llm-tools
```

Use `assets/brand/social-preview.png` for the repository social preview. GitHub repository settings require uploading the image manually.
