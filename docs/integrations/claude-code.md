# Claude Code Integration

Use Myelin with Claude Code when you want repeated coding workflows to become reusable procedures. Claude Code remains responsible for editing, shell execution, approvals, and project instructions. Myelin supplies learned context and procedure suggestions through MCP.

Reference: [Claude Code MCP docs](https://docs.anthropic.com/en/docs/claude-code/mcp).

## MCP Setup

Add Myelin as a stdio MCP server:

```bash
claude mcp add --transport stdio --scope user myelin -- \
  python -m myelin.server \
  --db /absolute/path/to/claude-code-myelin.db \
  --embedding-model none
```

Then open Claude Code and check `/mcp` to confirm the server is connected and exposing tools.

Project-scoped configuration is useful when a team wants the same server entry in the repository. User scope is usually better for a personal local memory database.

## Output Budget

Claude Code warns on large MCP outputs and has configurable MCP output limits. Keep Myelin outputs concise:

- Call `myelin_context` at task boundaries instead of every turn.
- Keep `max_memories` and `max_procedures` low until the integration feels useful.
- Do not store full command logs in `content_text`; store summaries and references.
- Prefer `myelin_observe_batch` for bursts of events.

## Recommended Tool Allowlist

Start with the base loop:

- `myelin_context`
- `myelin_observe`
- `myelin_observe_batch`
- `myelin_execute_procedure`
- `myelin_procedure_feedback`
- `myelin_status`
- `myelin_sleep`

Gate transfer tools and graph exploration until the local database contains useful, trusted procedures.

## Good First Tests

Use a disposable database and run a repeated workflow:

1. Ask Claude Code to run tests and fix one small lint or type issue.
2. Emit observations for each meaningful tool step.
3. Run `myelin_sleep`.
4. Repeat a similar workflow.
5. Ask `myelin_execute_procedure` for the task and send feedback after trying it.
