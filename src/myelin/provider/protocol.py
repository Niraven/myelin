"""MemoryProvider protocol — abstract base class defining the Hermes memory surface.

This protocol defines the contract that Myelin and Mem0 providers both satisfy,
enabling the dual-write / shadow-read architecture.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchResult:
    """A single memory result in mem0-compatible shape."""

    id: str
    memory: str
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Result of comparing two provider search outputs (shadow-read)."""

    overlap: float = 0.0
    precision: float = 0.0
    extra_in_primary: list[SearchResult] = field(default_factory=list)
    extra_in_shadow: list[SearchResult] = field(default_factory=list)
    quality: str = "unknown"  # match | myelin_better | mem0_better | divergence


class MemoryProvider(ABC):
    """Abstract interface for a Hermes memory provider.

    Implementations wrap Myelin's MCP tools or Mem0's direct API behind
    a uniform 4-method surface: add, search, update, delete.
    """

    @abstractmethod
    async def add(
        self,
        content: str,
        *,
        agent_id: str = "hermes",
        user_id: str = "default",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Store a memory fact.

        Returns ``{"result": "Fact stored.", "event_id": "<uuid>"}``
        or equivalent provider-specific shape.
        """

    @abstractmethod
    async def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        agent_id: str = "hermes",
        user_id: str = "default",
    ) -> list[SearchResult]:
        """Search stored memories, returning ranked results."""

    @abstractmethod
    async def update(
        self,
        memory_id: str,
        text: str,
        *,
        agent_id: str = "hermes",
    ) -> dict[str, Any]:
        """Update an existing memory by ID.

        Returns ``{"memory_id": memory_id}`` on success.
        """

    @abstractmethod
    async def delete(
        self,
        memory_id: str,
        *,
        agent_id: str = "hermes",
    ) -> dict[str, Any]:
        """Delete a memory by ID.

        Returns ``{"memory_id": memory_id}`` on success.
        """

    @abstractmethod
    async def system_prompt_block(
        self,
        *,
        agent_id: str = "hermes",
        user_id: str = "default",
    ) -> str:
        """Return a system-prompt injection block describing active memory."""

    @abstractmethod
    async def health(self) -> dict[str, Any]:
        """Return provider health status.

        Returns at minimum:
            ``{"ok": bool, "detail": str}``
        """


def compare_search_results(
    primary: list[SearchResult],
    shadow: list[SearchResult],
    *,
    iou_threshold: float = 0.3,
) -> ComparisonResult:
    """Compare two sets of search results (primary vs. shadow).

    Uses token-overlap IoU on ``.memory`` text to determine matching
    results.  Returns a ``ComparisonResult`` with quality classification.
    """
    if not primary and not shadow:
        return ComparisonResult(overlap=1.0, precision=1.0, quality="match")

    if not primary:
        return ComparisonResult(
            overlap=0.0,
            precision=0.0,
            extra_in_shadow=shadow,
            quality="divergence",
        )

    if not shadow:
        return ComparisonResult(
            overlap=0.0,
            precision=0.0,
            extra_in_primary=primary,
            quality="divergence",
        )

    # Build token sets for IoU
    def _token_set(text: str) -> set[str]:
        return set(text.lower().split())

    matched_primary: set[int] = set()
    matched_shadow: set[int] = set()

    for pi, p in enumerate(primary):
        p_tokens = _token_set(p.memory)
        for si, s in enumerate(shadow):
            if si in matched_shadow:
                continue
            s_tokens = _token_set(s.memory)
            union = p_tokens | s_tokens
            if not union:
                continue
            iou = len(p_tokens & s_tokens) / len(union)
            if iou >= iou_threshold:
                matched_primary.add(pi)
                matched_shadow.add(si)
                break

    overlap_count = len(matched_primary)  # same as len(matched_shadow)
    total = max(len(primary), len(shadow))
    overlap = overlap_count / total if total else 0.0
    precision = overlap_count / len(shadow) if shadow else 0.0

    extra_p = [r for i, r in enumerate(primary) if i not in matched_primary]
    extra_s = [r for i, r in enumerate(shadow) if i not in matched_shadow]

    # Classify quality
    if overlap >= 0.8 and len(extra_p) == 0 and len(extra_s) == 0:
        quality = "match"
    elif overlap >= 0.5 and len(extra_p) <= len(extra_s):
        quality = "myelin_better"
    elif overlap >= 0.5 and len(extra_s) < len(extra_p):
        quality = "mem0_better"
    else:
        quality = "divergence"

    return ComparisonResult(
        overlap=round(overlap, 4),
        precision=round(precision, 4),
        extra_in_primary=extra_p,
        extra_in_shadow=extra_s,
        quality=quality,
    )
