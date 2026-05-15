# OpenClaw Integration

Use Myelin with OpenClaw when OpenClaw-managed agents should learn from repeated workflows without turning Myelin into the orchestrator. OpenClaw remains responsible for channels, gateway/runtime behavior, approvals, and agent routing.

Reference: [OpenClaw MCP docs](https://docs.openclaw.ai/cli/mcp).

## MCP Registry Setup

OpenClaw documentation describes `openclaw mcp set` as a saved MCP server registry for runtimes that consume configured servers. Store Myelin as a stdio server:

```bash
openclaw mcp set myelin '{
  "command": "python",
  "args": [
    "-m",
    "myelin.server",
    "--db",
    "/absolute/path/to/openclaw-myelin.db",
    "--embedding-model",
    "none"
  ]
}'
```

Then inspect the saved entry:

```bash
openclaw mcp show myelin --json
```

This stores configuration. It does not by itself prove that a downstream OpenClaw runtime has started a live MCP client session or that Myelin is reachable.

## Transport Note

Myelin currently exposes stdio MCP. Do not configure it as `transport: "streamable-http"` or an HTTP URL unless you run a separate MCP bridge.

## Recommended Tool Allowlist

Start with:

- `myelin_context`
- `myelin_observe`
- `myelin_observe_batch`
- `myelin_execute_procedure`
- `myelin_procedure_feedback`
- `myelin_status`
- `myelin_sleep`

Use transfer tools only after a source agent has validated useful procedures and the target agent capabilities are known.

## Agent ID Pattern

Use stable actor IDs so cross-agent learning stays readable:

```text
openclaw/<workspace>/<role>
openclaw/security-reviewer
openclaw/release-operator
openclaw/browser-agent
```

Use one `session_id` for the shared workflow run so Myelin can cluster the sequence.
