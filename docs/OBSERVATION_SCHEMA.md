# Observation Schema

Myelin learns from structured observations. It can observe a single agent, a delegated team, or a swarm, but only when the orchestrator emits actions into `myelin_observe`.

Myelin does not watch processes automatically and it does not replace the orchestrator.

> Orchestrators coordinate agents. Myelin learns from what agents repeatedly do.

## Required Fields

These fields map directly to `myelin_observe`. For orchestrators emitting many events, wrap the same objects in `{"events": [...]}` and call `myelin_observe_batch`.

| Field | Type | Purpose |
|---|---:|---|
| `agent_id` | string | Stable ID for the actor that performed the action. |
| `session_id` | string | Stable ID for the workflow run. Myelin clusters by session. |
| `action` | string | Short normalized action, usually a command, tool name, or step label. |
| `action_type` | enum | `tool_call`, `response`, `error`, or `user_input`. |
| `content_text` | string | Searchable description of what happened. |
| `success` | boolean | Whether the action succeeded. |
| `domain` | string | Workflow family such as `deployment`, `research`, `triage`, or `security`. |
| `tags` | string[] | Optional labels for routing and later filtering. |
| `input_context` | object | Trigger, task metadata, tool input, or upstream state. |
| `output_result` | object | Result summary, tool output metadata, or downstream state. |

## Recommended Orchestrator Fields

Put these in `input_context` or `output_result` when the caller is Hermes, LangGraph, CrewAI, AutoGen, Codex, or another orchestrator.

```json
{
  "run_id": "hermes-20260515-ci-001",
  "swarm_id": "release-team",
  "task_id": "fix-ci",
  "parent_task_id": "launch-myelin",
  "agent_role": "build-agent",
  "orchestrator": "hermes",
  "tool_name": "shell",
  "tool_scope": "repo",
  "risk_level": "low",
  "requires_human_approval": false
}
```

## Single-Agent Pattern

```json
{
  "agent_id": "codex",
  "session_id": "myelin-ci-fix-20260515",
  "action": "ruff check src tests",
  "action_type": "tool_call",
  "content_text": "Ran ruff check across source and tests before pushing.",
  "success": true,
  "domain": "ci",
  "tags": ["lint", "release"]
}
```

## Team Or Swarm Pattern

Use one `session_id` for the shared workflow and distinct `agent_id` values for each actor.

```json
[
  {
    "agent_id": "hermes/researcher",
    "session_id": "launch-run-42",
    "action": "scan competitor docs",
    "action_type": "tool_call",
    "content_text": "Researched current memory providers before updating comparisons.",
    "success": true,
    "domain": "launch",
    "input_context": {"swarm_id": "launch-team", "agent_role": "research"}
  },
  {
    "agent_id": "codex/builder",
    "session_id": "launch-run-42",
    "action": "patch README",
    "action_type": "tool_call",
    "content_text": "Updated README positioning and launch proof sections.",
    "success": true,
    "domain": "launch",
    "input_context": {"swarm_id": "launch-team", "agent_role": "implementation"}
  }
]
```

## Trust Bands

Myelin separates procedure discovery from procedure trust.

| Trust level | Meaning | Agent behavior |
|---|---|---|
| `unvalidated` | Pattern exists but evidence is weak or unused. | Observe more before relying on it. |
| `candidate` | Repeated sessions support the procedure, but execution feedback is missing. | Suggest only; review before execution. |
| `validated` | The procedure has success feedback and enough confidence for light review. | Use with normal agent checks. |
| `trusted` | High confidence plus multiple successful executions. | Use as a default workflow, still respecting tool approvals. |
| `low_confidence` | Feedback or failures weakened the procedure. | Do not execute without review. |

Hermes should route `candidate` procedures into review mode and reserve automatic use for `validated` or `trusted` procedures.

## Cold Start

Myelin needs repeated sessions before it can promote a procedure. The practical integration pattern is:

1. Install Myelin quietly.
2. Call `myelin_context` before important work.
3. Emit observations during the run.
4. Call `myelin_procedure_feedback` after execution.
5. Let Myelin promote only repeated, similar workflows.

The first sessions should feel like passive telemetry. Value appears when the same workflow family repeats.
