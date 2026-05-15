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


def initial_procedure_confidence(
    session_count: int,
    core_step_count: int,
    variant_step_count: int = 0,
) -> float:
    """Estimate starting confidence for an auto-promoted procedure.

    Repeated independent sessions and stable core steps should not surface as
    coin-flip guidance. Variant-heavy procedures get a small penalty because
    they need human review.
    """
    session_score = min(session_count, 5) * 0.04
    core_score = min(core_step_count, 5) * 0.02
    variant_penalty = min(variant_step_count, 4) * 0.02
    confidence = 0.45 + session_score + core_score - variant_penalty
    return max(0.5, min(0.75, confidence))


def procedure_trust_level(confidence: float, success_count: int = 0, failure_count: int = 0) -> str:
    """Human-readable trust band for procedure suggestions."""
    executions = success_count + failure_count
    if confidence >= 0.85 and success_count >= 3:
        return "trusted"
    if confidence >= 0.7 and success_count >= 1:
        return "validated"
    if confidence >= 0.6:
        return "candidate"
    if executions > 0:
        return "low_confidence"
    return "unvalidated"


def procedure_recommendation(trust_level: str) -> str:
    """Action guidance for agents consuming a learned procedure."""
    if trust_level == "trusted":
        return "safe_to_use_with_normal_checks"
    if trust_level == "validated":
        return "use_with_light_review"
    if trust_level == "candidate":
        return "suggest_only_review_before_execution"
    if trust_level == "low_confidence":
        return "do_not_execute_without_human_or_agent_review"
    return "observe_more_runs_before_relying_on_this"


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
