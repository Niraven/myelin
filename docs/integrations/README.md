# Agent Integration Guide

Myelin is a local procedural learning layer for MCP-compatible agents. The agent or orchestrator remains in charge of planning, tools, approvals, scheduling, messages, and user interaction. Myelin learns from structured observations and returns context or procedures when the agent asks for them.

Current transport: Myelin is a stdio MCP server launched with `python -m myelin.server`. It does not currently expose a built-in port, SSE endpoint, or Streamable HTTP endpoint.

Remote HTTP or Streamable HTTP is useful for shared network services and agents on other machines, but that is future transport work unless you place Myelin behind an external MCP bridge.

## Integration Shape

```text
Agent runtime or orchestrator
  -> stdio MCP client
  -> python -m myelin.server
  -> SQLite database
  -> learned procedures, context, graph, feedback, sleep
```

Myelin does not automatically watch shells, browsers, tool calls, or agent logs. The calling agent must emit observations through `myelin_observe` or `myelin_observe_batch`.

## Minimal MCP Config

Use a separate database while testing a new agent integration:

```json
{
  "mcpServers": {
    "myelin": {
      "command": "python",
      "args": [
        "-m",
        "myelin.server",
        "--db",
        "/absolute/path/to/myelin-agent-test.db",
        "--embedding-model",
        "none"
      ],
      "env": {}
    }
  }
}
```

Use `--embedding-model none` for fast action traces. Add local embeddings later only when semantic recall quality matters.

## Two Loops

Learning loop:

1. Call `myelin_context` at the start of important work.
2. Emit actions with `myelin_observe_batch` during the workflow.
3. Call `myelin_sleep` at session end or during maintenance.
4. Call `myelin_execute_procedure` when a similar task appears.
5. Call `myelin_procedure_feedback` after the agent uses a procedure.

Transfer loop:

1. A source agent learns and validates a procedure.
2. `myelin_transfer_discover` checks target fit.
3. `myelin_transfer_export` packages procedure intent, steps, evidence, and capability assumptions.
4. `myelin_transfer_import` creates a target-agent draft with discounted confidence.
5. The target agent runs it, sends feedback, and builds its own confidence.

Transfer is capability-aware. It should describe intent and gaps for the target agent, not blindly copy raw commands from the source agent.

## Tool Exposure

Start small. Expose the tools required for the base learning loop first.

| Stage | Tools | Notes |
|---|---|---|
| Starter | `myelin_context`, `myelin_observe`, `myelin_observe_batch`, `myelin_execute_procedure`, `myelin_procedure_feedback`, `myelin_status`, `myelin_sleep` | Enough for observe, suggest, execute, and calibrate. |
| Expanded | `myelin_query`, `myelin_recall`, `myelin_confidence`, `myelin_temporal`, `myelin_what_changed`, `myelin_entity_status`, `myelin_entities` | Add after the agent handles concise outputs well. |
| Gated | `myelin_transfer_export`, `myelin_transfer_import`, `myelin_transfer_discover`, `myelin_graph_query`, `myelin_visualize`, `myelin_teach`, `myelin_profile` | Use with approval, admin-only channels, or trusted operators. |

## Call Cadence

- Call `myelin_context` at task boundaries, not every model turn.
- Prefer `myelin_observe_batch` for bursts from orchestrators, swarms, and multi-agent workflows.
- Keep `content_text` concise and put large raw tool outputs elsewhere.
- Call `myelin_sleep` at session end, after a workflow batch, or during scheduled maintenance.
- Call `myelin_execute_procedure` only when the current task resembles a repeated workflow.
- Call `myelin_procedure_feedback` every time an agent follows a procedure.

## Trust Bands

Agents should treat learned procedures according to trust level:

| Trust level | Agent behavior |
|---|---|
| `unvalidated` | Keep observing. Do not execute. |
| `candidate` | Suggest only. Require review. |
| `validated` | Use with normal tool approvals. |
| `trusted` | Prefer for matching workflows, still respecting dangerous-tool approvals. |
| `low_confidence` | Treat as historical context. |

See [Observation Schema](../OBSERVATION_SCHEMA.md) for event fields and trust semantics.

## Client Guides

- [Hermes](hermes.md): flagship orchestrator integration.
- [Codex](codex.md): coding-agent integration and AGENTS.md guidance.
- [Claude Code](claude-code.md): CLI MCP setup and output-budget cautions.
- [OpenClaw](openclaw.md): OpenClaw MCP registry setup.
- [Generic MCP clients](generic-mcp.md): portable stdio configuration.

Related client docs:

- [Hermes MCP](https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp)
- [Claude Code MCP](https://docs.anthropic.com/en/docs/claude-code/mcp)
- [OpenClaw MCP](https://docs.openclaw.ai/cli/mcp)
- [OpenAI Docs MCP](https://developers.openai.com/learn/docs-mcp)
