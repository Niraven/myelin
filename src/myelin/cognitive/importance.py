"""Retriever importance scoring.

Computes per-cluster importance using three signals:
1. cluster frequency
2. consequence (success rate)
3. recency (temporal decay)

Weights are configurable and intentionally default to the prior behaviour (frequency-only)
for zero regression.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from .reflector import build_cluster_stats


@dataclass(frozen=True)
class ImportanceWeights:
    """Configurable weights for the three signal components."""

    frequency: float = 1.0
    consequence: float = 0.0
    recency: float = 0.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, float] | "ImportanceWeights" | None) -> "ImportanceWeights":
        """Build weights from a mapping or environment fallbacks."""
        if isinstance(values, ImportanceWeights):
            return values
        if values is None:
            values = {}

        env_weight_json = os.environ.get("IMPORTANCE_WEIGHTS")
        if env_weight_json:
            try:
                env_values = json.loads(env_weight_json)
                values = {**env_values, **values}
            except json.JSONDecodeError:
                # Preserve backward compatibility: malformed env config should not crash callers.
                values = values

        env_weights = {
            "frequency": os.environ.get("IMPORTANCE_WEIGHT_FREQUENCY"),
            "consequence": os.environ.get("IMPORTANCE_WEIGHT_CONSEQUENCE"),
            "recency": os.environ.get("IMPORTANCE_WEIGHT_RECENCY"),
        }
        final_values = {}
        for key, env_value in env_weights.items():
            if env_value is not None:
                try:
                    final_values[key] = float(env_value)
                except ValueError:
                    final_values[key] = float(values.get(key, 0.0))
            elif key in values:
                final_values[key] = float(values[key])

        return cls(
            frequency=float(final_values.get("frequency", values.get("frequency", 1.0))),
            consequence=float(final_values.get("consequence", values.get("consequence", 0.0))),
            recency=float(final_values.get("recency", values.get("recency", 0.0))),
        )

    def normalized(self) -> "ImportanceWeights":
        """Normalize weights to sum to 1 while keeping all-zero semantics stable."""
        total = self.frequency + self.consequence + self.recency
        if total <= 0:
            return ImportanceWeights(frequency=1.0, consequence=0.0, recency=0.0)
        return ImportanceWeights(
            frequency=self.frequency / total,
            consequence=self.consequence / total,
            recency=self.recency / total,
        )

    def is_frequency_only(self) -> bool:
        return self.frequency > 0 and self.consequence == 0 and self.recency == 0


def temporal_decay(last_seen_timestamp: float, now: Optional[float] = None, *, half_life_seconds: float = 7 * 24 * 3600) -> float:
    """Exponential temporal decay based on how long ago a cluster was last seen.

    decay = 2 ^ (-age / half_life)
    """
    if last_seen_timestamp <= 0:
        return 0.0

    if now is None:
        now = datetime.now(timezone.utc).timestamp()

    age = max(0.0, now - float(last_seen_timestamp))
    if age <= 0:
        return 1.0

    if half_life_seconds <= 0:
        return 0.0 if age > 0 else 1.0

    return math.pow(2.0, -age / half_life_seconds)


def _normalize(values: Sequence[float]) -> list[float]:
    if not values:
        return []

    max_value = max(values)
    if max_value <= 0:
        return [0.0 for _ in values]
    return [v / max_value for v in values]


def score_clusters(
    episodes: Iterable[Mapping[str, Any]],
    *,
    weights: Mapping[str, float] | ImportanceWeights | None = None,
    now: Optional[float] = None,
    half_life_seconds: float = 7 * 24 * 3600,
    use_normalized_default: bool = False,
) -> Dict[str, float]:
    """Compute a per-cluster importance score.

    Returns
    -------
    dict[str, float]
        Mapping of cluster_id -> importance score.
    """
    resolved_weights = ImportanceWeights.from_mapping(weights)

    # Backward-compatible path: old behaviour was frequency-only and unnormalized.
    if resolved_weights.is_frequency_only() and not use_normalized_default:
        metrics = build_cluster_stats(episodes)
        return {cluster_id: float(stats.frequency) for cluster_id, stats in metrics.items()}

    normalized_weights = resolved_weights.normalized()
    metrics = build_cluster_stats(episodes)

    if not metrics:
        return {}

    frequencies = [stats.frequency for stats in metrics.values()]
    consequences = [stats.success_rate for stats in metrics.values()]
    recencies = [temporal_decay(stats.last_seen_ts, now=now, half_life_seconds=half_life_seconds) for stats in metrics.values()]

    norm_frequency = _normalize(frequencies)
    norm_consequence = _normalize(consequences)
    norm_recency = _normalize(recencies)

    scores: Dict[str, float] = {}
    for idx, (cluster_id, stats) in enumerate(metrics.items()):
        score = (
            normalized_weights.frequency * norm_frequency[idx]
            + normalized_weights.consequence * norm_consequence[idx]
            + normalized_weights.recency * norm_recency[idx]
        )
        scores[cluster_id] = float(score)

    return scores


def rank_clusters(
    episodes: Iterable[Mapping[str, Any]],
    *,
    weights: Mapping[str, float] | ImportanceWeights | None = None,
    now: Optional[float] = None,
    half_life_seconds: float = 7 * 24 * 3600,
    descending: bool = True,
    limit: Optional[int] = None,
    use_normalized_default: bool = False,
) -> list[tuple[str, float]]:
    """Return clusters sorted by computed importance."""
    scores = score_clusters(
        episodes,
        weights=weights,
        now=now,
        half_life_seconds=half_life_seconds,
        use_normalized_default=use_normalized_default,
    )

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=descending)
    if limit is not None:
        ranked = ranked[:limit]
    return ranked


def score_episodes(
    episodes: Iterable[Mapping[str, Any]],
    *,
    weights: Mapping[str, float] | ImportanceWeights | None = None,
    now: Optional[float] = None,
    half_life_seconds: float = 7 * 24 * 3600,
    use_normalized_default: bool = False,
) -> list[tuple[str, float]]:
    """Compatibility wrapper expected by older callers: returns ranked scores."""
    return rank_clusters(
        episodes,
        weights=weights,
        now=now,
        half_life_seconds=half_life_seconds,
        use_normalized_default=use_normalized_default,
    )
