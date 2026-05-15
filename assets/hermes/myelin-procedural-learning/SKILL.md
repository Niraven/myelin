---
name: myelin-procedural-learning
description: Use Myelin as Hermes' procedural learning layer for repeated agent, team, and swarm workflows.
---

# Myelin Procedural Learning

Use this skill when Hermes is about to run a repeated workflow or when a workflow finishes and should become training signal for Myelin.

## Principle

Hermes operates. Myelin learns the operating procedures.

Do not treat Myelin as the canonical task manager, message router, skill router, or general memory provider. Use it as a quiet procedure-learning substrate.

## Before A Task

Call `myelin_context` with:

- `query`: the task in plain language
- `domain`: stable workflow family such as `ci`, `deployment`, `research`, `content`, or `security`
- `agent_id`: `hermes` or the delegated agent ID

If `myelin_execute_procedure` returns:

- `trusted`: prefer the procedure with normal tool approval checks
- `validated`: use with light review
- `candidate`: suggest only and review before execution
- `low_confidence` or `unvalidated`: use as history, not an execution plan

## During A Task

Call `myelin_observe` for important actions:

- tool calls
- command executions
- edits
- delegated agent handoffs
- failures or retries
- final responses that changed the task state

Use the same `session_id` for the whole workflow. Use distinct `agent_id` values for delegated agents.

## After A Task

If a Myelin procedure was used, call `myelin_procedure_feedback`.

Set `success` to true only when the workflow completed without major manual correction. Add `notes` when the procedure needed edits, skipped steps, or human intervention.

## Session End

Call `myelin_sleep` after a batch of related work or at the end of the day to trigger consolidation and promotion.

## Safety

Never let a learned procedure bypass Hermes approval gates. Myelin suggests how work has been done; Hermes remains responsible for permissions, scheduling, routing, and execution.
