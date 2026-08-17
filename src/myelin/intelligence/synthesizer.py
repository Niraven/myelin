from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SearchResult:
    """Canonical result object used by query-time synthesis."""

    title: str
    url: str
    snippet: str
    score: float | None = None
    source: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, item: Mapping[str, Any]) -> SearchResult:
        if isinstance(item, SearchResult):
            return item
        if not isinstance(item, Mapping):
            raise TypeError("Search results must be Mapping or SearchResult instances.")

        title = (
            item.get("title")
            or item.get("name")
            or item.get("headline")
            or item.get("node_type")
            or item.get("action")
            or "Untitled result"
        )
        url = item.get("url") or item.get("link") or item.get("source_url") or ""
        snippet = (
            item.get("snippet")
            or item.get("excerpt")
            or item.get("description")
            or item.get("content")
            or item.get("content_text")
            or ""
        )
        score = item.get("score")
        if score is None:
            score = item.get("rank")
        if score is None:
            score = item.get("composite_score")
        if score is None:
            score = item.get("_composite_score")
        try:
            score = None if score is None else float(score)
        except (TypeError, ValueError):
            score = None

        source = item.get("source")
        return cls(
            title=str(title).strip(),
            url=str(url).strip(),
            snippet=str(snippet).strip(),
            score=score,
            source=str(source).strip() if source is not None else None,
            metadata={
                k: v
                for k, v in item.items()
                if k
                not in {
                    "title",
                    "url",
                    "link",
                    "source_url",
                    "snippet",
                    "excerpt",
                    "description",
                    "content",
                    "score",
                    "rank",
                    "source",
                }
            },
        )


def _split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    # Split with lightweight punctuation heuristics; good enough for snippets.
    parts = re.split(r"(?<=[\.\!?])\s+", text)
    out = [p.strip() for p in parts if p.strip()]
    if not out:
        out = [text]
    return out


def _first_sentence(text: str, max_words: int = 28) -> str:
    sentences = _split_sentences(text)
    if not sentences:
        return ""
    candidate = sentences[0]
    return _truncate_words(candidate, max_words)


def _truncate_words(text: str, max_words: int) -> str:
    words = re.split(r"\s+", text.strip())
    if not words:
        return ""
    if len(words) <= max_words:
        return text.strip().rstrip(".?") + "."
    return " ".join(words[:max_words]).rstrip(".,;:") + "..."


def _truncate_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    words = re.split(r"\s+", text.strip())
    if len(words) <= max_tokens:
        return text.strip()
    return " ".join(words[:max_tokens]).rstrip(".,;:") + " ..."


def _dedupe_by_url(items: Sequence[SearchResult]) -> list[SearchResult]:
    seen = set()
    out: list[SearchResult] = []
    for item in items:
        key = (item.url or item.title).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _ranked_items(results: Sequence[Any], top_n: int) -> list[SearchResult]:
    normalized: list[tuple[int, SearchResult]] = []
    for idx, result in enumerate(results):
        normalized.append((idx, SearchResult.from_mapping(result)))

    ranked = [
        item for _, item in sorted(normalized, key=lambda item: (-(item[1].score or -1), item[0]))
    ]
    ranked = _dedupe_by_url(ranked)
    return ranked[: max(0, top_n)]


def summarize_top_results(
    *,
    query: str,
    results: Sequence[Any],
    top_n: int = 5,
    max_sentences: int = 5,
    max_tokens: int = 500,
) -> str:
    """Return a 3-5 sentence query-time synthesis with citations.

    The function summarizes only the first ``top_n`` results (defaults to 5), and
    constrains the final response to ``max_tokens`` token budget.
    """
    ranked = _ranked_items(results, top_n=top_n)
    if not ranked:
        return "No results were available to synthesize."

    sentence_budget = max(1, min(max_sentences, max(3, min(len(ranked), 5))))
    selected = ranked[:sentence_budget]

    lines: list[str] = []
    for idx, result in enumerate(selected, start=1):
        snippet = _first_sentence(result.snippet or "", max_words=30)
        if snippet:
            line = (
                f"[{idx}] {snippet[0].lower()}{snippet[1:]}"
                if snippet[0].isalpha()
                else f"[{idx}] {snippet}"
            )
            if "appears" not in line:
                line = f"{line.rstrip()} ({result.title})"
        else:
            line = f"[{idx}] {result.title} appears relevant to this query."

        lines.append(line if line.endswith(".") else line + ".")

    heading = f"Top findings for: '{query}'. " if query else "Top findings: "
    base = heading + " ".join(lines)
    return _truncate_tokens(base, max_tokens)


def format_raw_results(
    *,
    query: str,
    results: Sequence[Any],
    top_n: int = 5,
    max_tokens: int = 500,
) -> str:
    """Fallback raw formatter used when synthesis is disabled."""
    ranked = _ranked_items(results, top_n=top_n)
    if not ranked:
        return "No results were available."

    lines = [f"Top results for: {query}".strip() or "Search results"]
    for idx, result in enumerate(ranked, start=1):
        title = result.title or "Untitled result"
        url = result.url or "(no-url)"
        snippet = _truncate_words(result.snippet or "", 28)
        lines.append(f"{idx}. {title}\n   {url}\n   {snippet}")

    raw = "\n".join(lines)
    return _truncate_tokens(raw, max_tokens)


def query_with_fallback(
    *,
    query: str,
    results: Sequence[Any],
    synthesize: bool,
    top_n: int = 5,
    max_tokens: int = 500,
    max_sentences: int = 5,
) -> str:
    """Single entry point combining both modes."""
    if synthesize:
        return summarize_top_results(
            query=query,
            results=results,
            top_n=top_n,
            max_sentences=max_sentences,
            max_tokens=max_tokens,
        )
    return format_raw_results(
        query=query,
        results=results,
        top_n=top_n,
        max_tokens=max_tokens,
    )


class Synthesizer:
    """Synthesizes query results into concise answers with citations.

    When ``llm_complete`` is provided, it will be used for LLM-based synthesis.
    Otherwise, a lightweight rule-based summarizer is used so the server works
    without any additional dependencies.
    """

    def __init__(self, llm_complete: Callable[[str], str] | None = None):
        self.llm = llm_complete

    def synthesize(
        self,
        query: str,
        results: list[dict],
        *,
        max_sources: int = 5,
        max_tokens: int = 500,
        max_sentences: int = 5,
    ) -> dict[str, Any]:
        """Produce a synthesized answer from ranked retrieval results.

        Returns a dict with keys:
            - synthesis: str (or None when falling back to raw)
            - sources: list[dict]  (top results used)
            - source_count: int
            - mode: "synthesized" | "raw"
            - results: list[dict] (present only in raw fallback)
            - message: str (present only in raw fallback)
        """
        if not results:
            return self._fallback(query, results)

        top = results[:max_sources]

        if self.llm:
            prompt = self._build_prompt(query, top)
            summary = self.llm(prompt)
            sources = [
                {
                    "id": r.get("id"),
                    "content": (r.get("content_text") or r.get("content") or "")[:200],
                    "score": r.get("_composite_score", 0),
                }
                for r in top
            ]
            return {
                "synthesis": summary,
                "sources": sources,
                "source_count": len(sources),
                "mode": "synthesized",
            }

        # Rule-based fallback when no LLM is configured.
        summary = summarize_top_results(
            query=query,
            results=results,
            top_n=max_sources,
            max_sentences=max_sentences,
            max_tokens=max_tokens,
        )
        sources = [
            {
                "id": r.get("id"),
                "content": (r.get("content_text") or r.get("content") or "")[:200],
                "score": r.get("_composite_score", 0),
            }
            for r in top
        ]
        return {
            "synthesis": summary,
            "sources": sources,
            "source_count": len(sources),
            "mode": "synthesized",
        }

    def _fallback(self, query: str, results: list[dict]) -> dict[str, Any]:
        """Raw results when synthesis is unavailable."""
        return {
            "synthesis": None,
            "results": results,
            "source_count": len(results),
            "mode": "raw",
            "message": "Synthesis unavailable. Returning raw results.",
        }

    @staticmethod
    def _build_prompt(query: str, top: list[dict]) -> str:
        lines = [f"Given this query: {query}", "And these relevant episodes:"]
        for idx, r in enumerate(top, start=1):
            content = (r.get("content_text") or r.get("content") or "")[:300]
            lines.append(f"  [{idx}] id={r.get('id')} — {content}")
        lines.append(
            "Produce a 3-5 sentence summary of what happened, citing episode IDs. "
            "Focus on: what was done, what changed, what entities were involved."
        )
        return "\n".join(lines)
