# Changelog

All notable changes to Myelin are documented here.

## Unreleased

- Shielded automatically assembled context to validated/trusted procedures with exact domain matching, while keeping explicit diagnostic search available.
- Added prediction-linked verified feedback with mismatch protection, replay idempotency, atomic outcome/evidence updates, and non-destructive provenance migration.
- Preserved legacy unbound feedback for compatibility while preventing it from promoting procedure trust.
- Bounded the MCP dependency to the compatible 1.x API and updated both public demos to prove verified feedback through trusted same-domain context.
- Rewrote the README as a concise, developer-first page: source-install quickstart, deterministic demo command, stdio MCP setup, verified trust boundary, and integration links.
- Redesigned the social preview with the own-product headline "Procedural learning for AI agents" and `stdio MCP` / `SQLite local` / `LLM optional` badges.
- Clarified the feedback contract: `myelin_procedure_feedback` bound via `prediction_id` is verified, idempotent, and atomic and can promote trust; unbound feedback updates confidence but cannot promote trust.
- Documented that the MCP server runs sleep consolidation and promotion only when the caller invokes `myelin_sleep` or maintenance, not via an automatic background daemon.
- Fixed the documented MCP tool count to 25 and corrected benchmark `--counts` invocation examples.

## 0.3.0 - Learning OS Stabilization

- Stabilized the Learning OS architecture with schema migration support, cognitive trigger fixes, and CI-clean validation.
- Expanded the MCP surface to 25 tools, including batch observation, profile, transfer, visualization, sleep, temporal, confidence, facts, and update workflows.
- Added agent integration guidance for Hermes, Codex, Claude Code, OpenClaw, and generic MCP clients.
- Clarified the public transport story: current Myelin is a stdio MCP server, with HTTP/Streamable HTTP left as future transport work.
- Added local performance guidance focused on agent acceleration and token-budget discipline.
- Verified the release baseline with ruff, mypy, the test suite, and the procedure-learning/Hermes demos.

## 0.2.x - Learning OS Expansion

- Added reconsolidation, prediction error learning, two-phase sleep, prioritized replay, schema learning, LLM consolidation/reflection, curiosity, FSRS scheduling, and self-model calibration.
- Added task specs documenting the Learning OS subsystems.
- Added cross-agent transfer planning and procedure adaptation direction.

## 0.1.x - Procedural Memory Foundation

- Added episodic, semantic, and procedural memory layers.
- Added SQLite/FTS5 storage, entity extraction, temporal state, knowledge graph support, ACT-R activation, and Bayesian confidence.
- Added the first MCP server interface and proof demos for automatic procedure promotion.

