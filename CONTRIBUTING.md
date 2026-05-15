# Contributing to Myelin

Myelin is built around one core product promise: agents should learn reusable procedures from behavior, not only recall text.

## Development Setup

```bash
git clone https://github.com/Niraven/myelin.git
cd myelin
pip install -e ".[dev]"
```

## Validation

Run the same checks used for release readiness:

```bash
ruff format --check src/ tests/ examples/
ruff check src/ tests/ examples/
mypy src/myelin/ --ignore-missing-imports
pytest tests/ -q
python examples/procedure_learning_demo.py
```

## Contribution Priorities

High-value contributions:

- Better procedure-learning demos from real agent workflows.
- MCP client integration guides.
- Reliability fixes for SQLite, FTS5, and optional sqlite-vec behavior.
- Tests for promotion, transfer, context assembly, and confidence updates.
- Documentation that makes local-first usage easier.

Please avoid adding speculative cognitive features before the current procedural-learning flow is easy to install, understand, and verify.

## Security

Do not commit secrets, private memory databases, browser session files, `.env` files, or real user/agent transcripts. Use synthetic examples in tests and docs.
