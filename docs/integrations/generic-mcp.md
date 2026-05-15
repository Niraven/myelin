# Generic MCP Client Integration

Any MCP client that supports stdio servers can connect to Myelin.

## Config

```json
{
  "mcpServers": {
    "myelin": {
      "command": "python",
      "args": [
        "-m",
        "myelin.server",
        "--db",
        "/absolute/path/to/myelin.db",
        "--embedding-model",
        "none"
      ],
      "env": {}
    }
  }
}
```

Some clients use `servers` instead of `mcpServers`, or require a `type: "stdio"` field. Keep the command and args the same.

## First Tool Set

Expose only the base loop first:

- `myelin_context`
- `myelin_observe`
- `myelin_observe_batch`
- `myelin_execute_procedure`
- `myelin_procedure_feedback`
- `myelin_status`
- `myelin_sleep`

Add the rest after you trust the agent's behavior and output handling.

## Observation Contract

Every meaningful event should include:

- stable `agent_id`
- shared `session_id`
- normalized `action`
- `action_type`
- concise `content_text`
- `success`
- `domain`

For teams and swarms, keep a shared `session_id` across actors and distinct `agent_id` values for each actor.

## When Not To Call Myelin

- Do not call `myelin_context` on every model turn.
- Do not store raw secrets, full environment dumps, or private credentials in observations.
- Do not auto-execute `candidate` procedures without review.
- Do not expose transfer/admin tools to untrusted channels.

