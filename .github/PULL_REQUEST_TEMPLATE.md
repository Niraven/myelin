## Summary

Describe what changed and why.

## Validation

```bash
ruff format --check src/ tests/ examples/
ruff check src/ tests/ examples/
mypy src/myelin/ --ignore-missing-imports
pytest tests/ -q
python examples/procedure_learning_demo.py
```

## Risk

Note any changes to memory schema, MCP tool behavior, procedure promotion, transfer, or persistence.

## Checklist

- [ ] Tests added or updated for behavior changes.
- [ ] README/docs updated when public behavior changed.
- [ ] No secrets, credentials, or private memory data included.
