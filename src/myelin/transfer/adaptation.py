"""Step-level adaptation engine for cross-agent transfer.

Adapts procedure steps for the target agent's toolset, rewriting or flagging
steps when tools don't match. Produces confidence-calibrated scores.
"""

from __future__ import annotations

import re
from typing import Any

from .tool_map import extract_tool_from_step, find_alternative_tool


class AdaptationResult:
    """Result of adapting a single step."""

    def __init__(
        self,
        step: dict[str, Any],
        quality: float = 1.0,
        changed: bool = False,
        flag: bool = False,
        note: str = "",
    ):
        self.step = step
        self.quality = quality
        self.changed = changed
        self.flag = flag
        self.note = note


class StepAdaptationEngine:
    """Adapts procedure steps for different target agent tool sets."""

    def adapt(
        self,
        step: dict[str, Any],
        target_tools: list[str],
    ) -> AdaptationResult:
        """Adapt a single step for the target agent's toolset.

        Returns an AdaptationResult with:
        - adapted step dict
        - quality score (0.0-1.0)
        - whether the step was changed
        - whether it needs human review flag
        - adaptation note
        """
        description = step.get("description", "")
        if not description:
            return AdaptationResult(step=dict(step), quality=1.0, note="Empty step")

        tool_name = extract_tool_from_step(description)
        if not tool_name:
            # No tool reference found — keep as-is
            return AdaptationResult(
                step=dict(step),
                quality=1.0,
                note="No tool reference found",
            )

        target_set = {t.lower().strip() for t in target_tools}
        desc_lower = description.lower()

        # Quick check: does any target tool phrase appear directly in the step?
        for t in target_set:
            if t in desc_lower:
                return AdaptationResult(
                    step=dict(step),
                    quality=1.0,
                    note=f"Tool '{t}' available on target",
                )

        alt = find_alternative_tool(tool_name, target_set)

        if tool_name.lower().strip() in target_set or alt == tool_name.lower().strip():
            # Tool is available directly
            return AdaptationResult(
                step=dict(step),
                quality=1.0,
                note=f"Tool '{tool_name}' available on target",
            )

        if alt:
            # Rewrite step with alternative tool
            new_desc = self._rewrite_step(description, tool_name, alt)
            adapted_step = dict(step)
            adapted_step["description"] = new_desc
            adapted_step["_original_tool"] = tool_name
            adapted_step["_adapted_tool"] = alt
            adapted_step["type"] = "variant"
            return AdaptationResult(
                step=adapted_step,
                quality=0.8,
                changed=True,
                note=f"Rewrote '{tool_name}' -> '{alt}'",
            )

        # No alternative found — flag for review
        flagged_step = dict(step)
        flagged_step["_missing_tool"] = tool_name
        flagged_step["type"] = "variant"
        flagged_step["_flagged"] = True
        return AdaptationResult(
            step=flagged_step,
            quality=0.4,
            flag=True,
            note=f"Tool '{tool_name}' not available on target — no alternative found",
        )

    def analyze_requirements(self, procedure: dict[str, Any]) -> list[str]:
        """Extract tool requirements from procedure steps."""
        steps = procedure.get("steps", [])
        requirements = []
        for step in steps:
            if isinstance(step, str):
                step = {"description": step}
            desc = step.get("description", "")
            tool = extract_tool_from_step(desc)
            if tool and tool not in requirements:
                requirements.append(tool)
        return requirements

    def calculate_confidence_discount(
        self,
        base_confidence: float,
        adaptations: list[AdaptationResult],
    ) -> tuple[float, str]:
        """Calculate post-adaptation confidence and status.

        Discount rules:
        - 1.0 (no changes) → confidence preserved
        - 0.8 (minor: 1-2 substitutions) → slight discount
        - 0.6 (multiple: 3+ substitutions) → major discount, stays draft
        - 0.4 (any flagged step) → heavy discount, flagged for review

        Returns (discounted_confidence, status).
        """
        if not adaptations:
            return base_confidence, "active"

        has_flagged = any(a.flag for a in adaptations)
        changed_count = sum(1 for a in adaptations if a.changed)
        unchanged_count = sum(1 for a in adaptations if not a.changed and not a.flag)

        if has_flagged:
            discount = 0.4
            status = "draft"
        elif changed_count == 0:
            discount = 1.0
            status = "active"
        elif changed_count <= 2:
            discount = 0.8
            status = "draft"
        else:
            discount = 0.6
            status = "draft"

        return base_confidence * discount, status

    def adapt_procedure(
        self,
        steps: list[Any],
        target_tools: list[str],
    ) -> tuple[list[dict[str, Any]], list[AdaptationResult], list[str]]:
        """Adapt all steps in a procedure.

        Returns (adapted_steps, adaptation_results, notes).
        """
        adapted_steps = []
        results = []
        notes = []

        for step in steps:
            if isinstance(step, str):
                step = {"description": step, "type": "core"}
            result = self.adapt(step, target_tools)
            adapted_steps.append(result.step)
            results.append(result)
            if result.note:
                notes.append(result.note)

        if not notes:
            notes.append("All steps compatible with target agent")

        return adapted_steps, results, notes

    def _rewrite_step(
        self,
        description: str,
        original_tool: str,
        new_tool: str,
    ) -> str:
        """Rewrite a step description replacing the original tool with the alternative."""
        desc = description
        # Replace case-insensitive whole-word or common patterns
        patterns = [
            rf"\b{re.escape(original_tool)}\b",
        ]
        for pat in patterns:
            desc = re.sub(pat, new_tool, desc, flags=re.IGNORECASE)
        return desc
