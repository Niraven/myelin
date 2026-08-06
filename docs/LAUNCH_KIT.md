# Myelin Launch Kit

## One-Liner

Agents should not relearn the same workflow twice. Myelin turns repeated agent behavior into reusable procedures.

For orchestrators:

> Hermes operates. Myelin learns the operating procedures.

## Short Pitch

Most agent memory systems retrieve text. Myelin learns how work gets done. It observes agent actions, clusters repeated workflows, uses ClustalW-inspired sequence alignment to extract procedure steps, and exposes the learned workflow through MCP.

Sigil was the broad research prototype. Myelin is the focused product: procedural learning for AI agents.

## Demo Command

```bash
python examples/procedure_learning_demo.py
```

Requires a source checkout (`pip install -e ".[dev]"`). Hermes demo:

```bash
python examples/hermes_procedure_demo.py
```

## Hacker News Draft

Title:

```text
Show HN: Myelin, procedural memory for AI agents
```

Post:

```text
Myelin is a local-first procedural learning layer for AI agents. The main idea is workflow learning rather than fact recall.

The demo observes five repeated deployment workflows, promotes the shared action sequence into an executable procedure, and updates confidence after success feedback. Core learning runs locally on SQLite/FTS5 and does not require an LLM call.

The architecture combines episodic, semantic, and procedural memory with entity extraction, temporal state, context assembly, and MCP tools.

The positioning is simple: mem0 remembers. Myelin learns.
```

## X / LinkedIn Draft

```text
Most agent memory stores facts.

Myelin learns procedures.

It watches repeated agent workflows, aligns action sequences with a ClustalW-inspired algorithm, promotes consensus steps into executable procedures, and updates confidence from feedback.

Local-first. SQLite. stdio MCP. Core learning runs without an LLM.

Demo: python examples/procedure_learning_demo.py

Hermes demo: python examples/hermes_procedure_demo.py
```

## GitHub Repo Settings

Recommended description:

```text
Agents should not relearn workflows. Myelin turns repeated agent behavior into reusable procedures via MCP.
```

Recommended topics:

```text
ai-agents, mcp, memory, procedural-memory, cognitive-architecture, sqlite, agent-framework, llm-tools
```

Use `assets/brand/social-preview.png` for the repository social preview. It is the deterministic flat export of `social-preview.svg`, sized at 1280x640. GitHub repository settings require uploading the image manually.

## HyperFrames Video

Use `assets/hyperframes/myelin-launch/` as the source for a short launch video.

Rendered assets:

- Video: `assets/hyperframes/myelin-launch/renders/myelin-launch.mp4`
- Opening slide: `assets/brand/myelin-launch-slide.png`
- Closing slide: `assets/brand/myelin-launch-close.png`

Validation commands:

```bash
cd assets/hyperframes/myelin-launch
npx hyperframes lint --verbose
npx hyperframes inspect --samples 8 --json
npx hyperframes render --output renders/myelin-launch.mp4 --quality draft --fps 30
```

Recommended title:

```text
Myelin: agents should not relearn the same workflow twice
```

Recommended caption:

```text
Most memory tools retrieve facts. Myelin learns procedures.

It watches repeated agent workflows, aligns action sequences, promotes the shared pattern into an executable procedure, and updates confidence with feedback.

Local-first. SQLite-backed. stdio MCP. Core learning runs without an LLM.
```

Recommended structure:

1. Brand hook: `mem0 remembers. Myelin learns.`
2. Problem: agents repeat deploy, CI, research, and release workflows.
3. Mechanism: observe -> align -> promote -> execute -> feedback.
4. Distribution: Hermes, Codex, Claude Code, OpenClaw, and generic MCP clients.
5. Close: local-first procedural learning for AI agents.
