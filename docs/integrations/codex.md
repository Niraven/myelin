# Codex Integration

Use Myelin with Codex when you want coding sessions to accumulate procedural knowledge: CI repair sequences, release checklists, migration workflows, security review loops, and repeated repository maintenance.

Myelin does not replace `AGENTS.md`, skills, local instructions, or Codex approval modes. It sits underneath Codex as a learning layer for repeated workflows.

Reference: [OpenAI Docs MCP](https://developers.openai.com/learn/docs-mcp).

## MCP Setup

Current Myelin transport is stdio MCP:

```toml
[mcp_servers.myelin]
command = "python"
args = [
  "-m",
  "myelin.server",
  "--db",
  "/absolute/path/to/codex-myelin.db",
  "--embedding-model",
  "none"
]
```

If your Codex CLI supports adding stdio MCP servers from the command line, the equivalent shape is:

```bash
codex mcp add myelin -- python -m myelin.server \
  --db /absolute/path/to/codex-myelin.db \
  --embedding-model none
```

Verify with your Codex MCP status command before relying on the tools. Local Codex installations can vary, so the checked-in config shape is the durable reference.

## AGENTS.md Guidance

Add a short instruction where Codex will read it:

```md
Use Myelin for procedural learning. Call `myelin_context` before important repeatable work, emit concise observations with `myelin_observe_batch`, and send `myelin_procedure_feedback` after following a learned procedure. Do not expose transfer/admin tools unless the task explicitly needs them.
```

## Recommended Tool Allowlist

Start with:

- `myelin_context`
- `myelin_observe`
- `myelin_observe_batch`
- `myelin_execute_procedure`
- `myelin_procedure_feedback`
- `myelin_status`
- `myelin_sleep`

Keep transfer, graph, teach, and profile tools gated until the workflow is proven.

## Useful Workflows

- CI repair: observe test, lint, type-check, patch, rerun sequence.
- Release prep: observe version bump, docs, changelog, demo, checks.
- Security pass: observe scan, triage, fix, validation sequence.
- Repo maintenance: observe repeated dependency or migration work.

The goal is not to retrieve old chat. The goal is for Codex to recognize "we have done this workflow before" and reuse the procedure with calibrated confidence.
