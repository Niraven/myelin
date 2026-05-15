# Security Policy

Myelin stores agent observations, learned procedures, entity mentions, and temporal state. Treat Myelin databases as sensitive operational memory.

## Reporting

Please report security issues privately by opening a minimal GitHub issue that requests a private disclosure channel. Do not include exploit details, private memory contents, credentials, or tokens in a public issue.

## Data Handling

- Do not commit real Myelin SQLite databases.
- Do not commit `.env` files, API keys, tokens, browser session files, or MCP credentials.
- Prefer synthetic traces in demos and tests.
- Use least-privilege MCP configurations when connecting Myelin to agent frameworks.

## Integration Guidance

When integrating Myelin with an agent runtime:

- Scope database paths per project or team.
- Review what action/output text is sent to `myelin_observe`.
- Avoid recording secrets from command output.
- Run Myelin locally unless shared memory is explicitly needed.
- Treat transfer packages as potentially sensitive because they can encode operational workflows.
