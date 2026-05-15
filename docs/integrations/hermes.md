# Hermes Integration

Hermes operates. Myelin learns the operating procedures.

Use Myelin as a procedural learning layer beside Hermes, not as a replacement for Hermes memory, task management, cron, Kanban, skills, or routing.

## Integration Boundary

Hermes should own:

- cron and always-on execution
- message gateway and notifications
- task routing and Kanban state
- skill selection and execution
- external memory provider selection
- human approval gates

Myelin should own:

- procedure learning from repeated Hermes runs
- context and procedure suggestions before a task
- structured observation storage during a task
- confidence updates after success or failure
- cross-agent/team procedure transfer after trust is established

## Install

From PyPI after release:

```bash
pip install myelin-memory
```

From a checkout:

```bash
pip install -e ".[dev]"
```

Verify the package:

```bash
python -m myelin.server --help
```

## MCP Server

Start with a local SQLite database dedicated to Hermes:

```json
{
  "mcpServers": {
    "myelin": {
      "command": "python",
      "args": [
        "-m",
        "myelin.server",
        "--db",
        "/absolute/path/to/hermes-myelin.db",
        "--embeddings",
        "none"
      ],
      "env": {}
    }
  }
}
```

If Hermes supports tool allowlists in your active config, start with this minimal set:

```yaml
tools:
  include:
    - myelin_context
    - myelin_observe
    - myelin_execute_procedure
    - myelin_procedure_feedback
    - myelin_status
    - myelin_sleep
prompts: false
resources: false
sampling:
  enabled: false
timeout: 10
```

Defer these until the base loop is proven:

- `myelin_transfer_export`
- `myelin_transfer_import`
- `myelin_transfer_discover`
- broad graph/entity exploration tools for untrusted channels

## Hermes Call Pattern

Before a task:

```json
{
  "tool": "myelin_context",
  "arguments": {
    "query": "fix failing CI for Myelin",
    "domain": "ci",
    "agent_id": "hermes"
  }
}
```

During the task:

```json
{
  "tool": "myelin_observe",
  "arguments": {
    "agent_id": "hermes/codex-build",
    "session_id": "ci-run-20260515-001",
    "action": "ruff check src tests",
    "action_type": "tool_call",
    "content_text": "Ran ruff check on src and tests while repairing CI.",
    "success": true,
    "domain": "ci",
    "tags": ["lint", "release"],
    "input_context": {
      "orchestrator": "hermes",
      "agent_role": "build",
      "task_id": "repair-myelin-ci"
    }
  }
}
```

After a task:

```json
{
  "tool": "myelin_procedure_feedback",
  "arguments": {
    "procedure_id": "returned-procedure-id",
    "success": true,
    "notes": "Hermes completed the workflow without manual correction."
  }
}
```

At session end or daily maintenance:

```json
{
  "tool": "myelin_sleep",
  "arguments": {
    "agent_id": "hermes"
  }
}
```

## Routing Rules

Hermes should treat Myelin responses by trust band:

| Trust level | Hermes action |
|---|---|
| `candidate` | Present as a suggested procedure; do not auto-run. |
| `validated` | Use with light review and normal tool approvals. |
| `trusted` | Prefer as default workflow for matching tasks. |
| `low_confidence` | Mention only as historical context. |
| `unvalidated` | Keep observing. |

## Procedure To Skill Bridge

Myelin should not write live Hermes skills automatically at first. The safer bridge is:

1. Myelin learns a procedure.
2. Hermes or Codex reviews the procedure.
3. Codex exports it into a `SKILL.md` candidate.
4. Hermes runs the skill tests or dry-run.
5. The user approves installation.

Recommended future command:

```bash
myelin export-skill PROCEDURE_ID --target hermes --out ./skills/generated/
```

That bridge turns learned procedures into durable Hermes skills without letting untrusted observations mutate the agent runtime directly.

## Verification

Run the local proof:

```bash
python examples/hermes_procedure_demo.py
```

Expected shape:

```text
Hermes + Myelin procedure-learning demo
Episodes observed: 26
Procedures created: 1
Trust level: candidate
Recommendation: suggest_only_review_before_execution
Trust after feedback: validated
```

If this works, wire Hermes to the MCP server and let it observe low-risk workflows first: repo CI, recurring research summaries, build-log-to-brand, and deployment dry-runs.
