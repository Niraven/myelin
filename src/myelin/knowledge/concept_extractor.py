"""LLM-based concept extraction for the hybrid entity extraction pipeline.

Takes N episode texts, returns structured entity candidates that regex patterns miss.
Designed for low cost (< 200 tokens per episode batch) and structured JSON output.
"""

from __future__ import annotations

import json
import re
import urllib.request
from typing import Callable


class ConceptExtractor:
    """Extracts conceptual entities from episode text using an LLM.

    Takes N episode texts, returns structured entity candidates.
    Uses the configured LLM provider or an injected client callable.

    Prompt designed for:
    - Low cost (< 200 tokens per episode batch)
    - Structured JSON output
    - Focus on user-specific tools, services, workflows
    """

    PROMPT = """Extract named entities from these agent episode descriptions.
Focus on tools, services, workflows, and concepts that a regex would miss
(e.g. "Google Drive", "project sync", "deployment pipeline").

Episodes:
{episodes}

Return JSON array: [{{"name": "...", "entity_type": "concept|tool|service"}}]
Only return entities NOT already in this list: {existing_entities}
"""

    def __init__(self, provider: str | None = None, client: Callable | None = None):
        self.provider = provider
        self.client = client

    def extract_concepts(self, texts: list[str]) -> list[dict[str, str]]:
        """Extract conceptual entities from a batch of episode texts.

        Returns list of dicts with 'name', 'entity_type', 'canonical_name'.
        """
        if not texts:
            return []
        if self.client:
            return self.client(texts)
        if not self.provider:
            return []
        return self._call_llm(texts)

    def _call_llm(self, texts: list[str]) -> list[dict[str, str]]:
        provider = self.provider
        if not provider or not provider.startswith(("http://", "https://")):
            return []

        episodes = "\n".join(f"- {t}" for t in texts)
        prompt = self.PROMPT.format(episodes=episodes, existing_entities="[]")

        payload = {
            "model": "default",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
        }

        try:
            req = urllib.request.Request(
                provider,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
                content = self._parse_response(data)
                return self._parse_entities(content)
        except Exception:
            return []

    def _parse_response(self, data: dict) -> str:
        """Parse LLM response into raw text."""
        # OpenAI format
        if "choices" in data and data["choices"]:
            return data["choices"][0].get("message", {}).get("content", "")
        # Ollama format
        if "message" in data:
            return data["message"].get("content", "")
        return ""

    def _parse_entities(self, content: str) -> list[dict[str, str]]:
        """Parse JSON array from LLM response text."""
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if not match:
            return []
        try:
            items = json.loads(match.group())
            results: list[dict[str, str]] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                name = item.get("name", "")
                entity_type = item.get("entity_type", "concept")
                if name:
                    results.append(
                        {
                            "name": name,
                            "entity_type": entity_type,
                            "canonical_name": name.lower().strip(),
                        }
                    )
            return results
        except Exception:
            return []
