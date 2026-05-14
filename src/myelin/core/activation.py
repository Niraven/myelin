"""ACT-R activation math and Bayesian confidence updating.

Based on Anderson et al. (2004) ACT-R 6.0 and the base-level learning equation.
"""

from __future__ import annotations

import math
import time


def base_level_activation(
    access_times: list[float],
    decay: float = 0.5,
    now: float | None = None,
) -> float:
    """ACT-R base-level activation: B_i = ln(sum_j(t_j^(-d)))

    Higher activation = more recently and frequently accessed.
    Returns -inf if no access times.
    """
    if not access_times:
        return float("-inf")

    now = now or time.time()
    total = 0.0
    for t in access_times:
        age = max(now - t, 0.001)
        total += age ** (-decay)

    if total <= 0:
        return float("-inf")
    return math.log(total)


def should_promote(
    access_times: list[float],
    threshold: float = 1.0,
    decay: float = 0.5,
    min_episodes: int = 2,
) -> bool:
    """Determine if a cluster of episodes should be promoted to a procedure."""
    if len(access_times) < min_episodes:
        return False
    return base_level_activation(access_times, decay) > threshold


def bayesian_confidence_update(
    current: float,
    success: bool,
    learning_rate: float = 0.15,
) -> float:
    """Bayesian-inspired confidence update.

    On success: confidence approaches 1.0 asymptotically
    On failure: confidence decays proportionally
    """
    if success:
        new = current + (1.0 - current) * learning_rate
    else:
        new = current * (1.0 - learning_rate)
    return max(0.0, min(1.0, new))


def calibration_offset(
    predicted: list[float],
    outcomes: list[bool],
) -> float:
    """Compute calibration offset: how much our predictions overshoot/undershoot reality.

    Positive offset = overconfident. Negative = underconfident.
    """
    if not predicted or not outcomes:
        return 0.0

    avg_predicted = sum(predicted) / len(predicted)
    avg_actual = sum(1.0 if o else 0.0 for o in outcomes) / len(outcomes)
    return avg_predicted - avg_actual


def ebbinghaus_decay(
    confidence: float,
    hours_since_use: float,
    stability: float = 1.0,
) -> float:
    """Ebbinghaus forgetting curve: R = e^(-t/S)

    stability: higher = slower decay. Increases with successful retrievals.
    """
    retention = math.exp(-hours_since_use / max(stability, 0.01))
    return confidence * retention


def agent_similarity(
    tools_a: set[str],
    tools_b: set[str],
    format_a: str,
    format_b: str,
    model_a: str,
    model_b: str,
    weights: tuple[float, float, float] = (0.4, 0.3, 0.3),
) -> float:
    """Compute similarity between two agent profiles for transfer scoring."""
    union = tools_a | tools_b
    tool_overlap = len(tools_a & tools_b) / len(union) if union else 1.0

    if format_a == format_b:
        format_match = 1.0
    elif {format_a, format_b} <= {"mcp_stdio", "mcp_sse"}:
        format_match = 0.5
    else:
        format_match = 0.1

    if model_a == model_b:
        model_match = 1.0
    elif model_a.split("-")[0] == model_b.split("-")[0]:
        model_match = 0.7
    else:
        model_match = 0.3

    w_tool, w_format, w_model = weights
    return w_tool * tool_overlap + w_format * format_match + w_model * model_match


def transfer_confidence(
    source_confidence: float,
    similarity: float,
) -> float:
    """Compute confidence for a transferred procedure."""
    return source_confidence * similarity
