# Myelin Launch Plan

## Positioning

Myelin should launch as a procedural memory layer for AI agents, not as another vector-memory product.

Primary line:

> mem0 remembers. Myelin learns.

The first public proof should focus on one claim: repeated agent behavior becomes an executable procedure through local, deterministic learning.

## Highest-Leverage Next Moves

1. **Ship the proof demo.** Use `examples/procedure_learning_demo.py` as the canonical terminal demo and record it as a short GIF or video.
2. **Publish only verified claims.** Avoid live star counts, benchmark rankings, or competitor claims unless they are re-checked immediately before launch.
3. **Make install friction boring.** Keep `pip install myelin-memory`, Docker, and MCP config examples as the launch-readiness bar.
4. **Prioritize Hermes, Claude Code, and Codex integration docs.** Hermes has the cleanest memory-provider fit; Claude Code and Codex have the largest cross-session memory pain.
5. **Delay Phase 2 features.** Predictive Procedures is the best next feature after launch because it makes Myelin feel proactive. Surprise-Driven Learning should build on it. Causal graph work should wait until after usage proves demand.

## Launch Checklist

- CI green: tests, ruff, format check, and mypy.
- README leads with procedural learning and includes a one-command demo.
- Demo output shows observed episodes, created procedure, executable steps, and confidence update.
- GitHub repository contains `.github/workflows/ci.yml` at the correct path.
- PyPI package metadata is correct for `myelin-memory`.
- First launch post links to the demo, architecture, and source.

## Revenue Path

The open-source repo should earn trust with local-first memory. The paid product should focus on teams:

- Managed shared memory across multiple agents.
- Procedure library with approval, versioning, and transfer analytics.
- Admin dashboard for confidence trends and learned workflows.
- Enterprise deployment with SSO/RBAC, audit logs, and data residency.

The first monetizable signal is not stars; it is a team asking to share learned procedures across more than one agent or repo.
