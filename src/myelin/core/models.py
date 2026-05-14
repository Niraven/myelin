"""Pydantic data models for all Myelin memory types."""

from __future__ import annotations

import time
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def _new_id() -> str:
    return uuid4().hex[:16]


def _now() -> datetime:
    return datetime.utcnow()


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


# ── Enums ──────────────────────────────────────────────────────


class ActionType(str, Enum):
    TOOL_CALL = "tool_call"
    RESPONSE = "response"
    ERROR = "error"
    USER_INPUT = "user_input"


class NodeType(str, Enum):
    FACT = "fact"
    REFLECTION = "reflection"
    META_REFLECTION = "meta_reflection"
    PREFERENCE = "preference"


class SourceType(str, Enum):
    OBSERVATION = "observation"
    REFLECTION = "reflection"
    TEACHING = "teaching"
    TRANSFER = "transfer"


class ProcedureStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    REFLEXIVE = "reflexive"
    ARCHIVED = "archived"


class PromotionMethod(str, Enum):
    AUTO = "auto"
    TAUGHT = "taught"
    TRANSFERRED = "transferred"
    COMPOSED = "composed"


class StepType(str, Enum):
    CORE = "core"
    VARIANT = "variant"
    OPTIONAL = "optional"


class GoalStatus(str, Enum):
    ACTIVE = "active"
    ACHIEVED = "achieved"
    ABANDONED = "abandoned"


class ProcessName(str, Enum):
    CONSOLIDATOR = "consolidator"
    REFLECTOR = "reflector"
    PROMOTER = "promoter"
    COMPOSER = "composer"
    DECAYER = "decayer"
    CHALLENGER = "challenger"


# ── Episodic Memory ────────────────────────────────────────────


class Episode(BaseModel):
    id: str = Field(default_factory=_new_id)
    agent_id: str
    session_id: str
    timestamp: str = Field(default_factory=_now_iso)
    action: str
    action_type: ActionType
    input_context: dict[str, Any] | None = None
    output_result: dict[str, Any] | None = None
    success: bool = True
    content_text: str
    embedding: list[float] | None = None
    access_count: int = 1
    access_times: list[float] = Field(default_factory=lambda: [time.time()])
    last_accessed: str = Field(default_factory=_now_iso)
    consolidated: bool = False
    cluster_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    domain: str | None = None
    created_at: str = Field(default_factory=_now_iso)


# ── Semantic Memory ────────────────────────────────────────────


class SemanticNode(BaseModel):
    id: str = Field(default_factory=_new_id)
    node_type: NodeType
    content: str
    embedding: list[float] | None = None
    source_type: SourceType
    source_ids: list[str] = Field(default_factory=list)
    access_count: int = 1
    access_times: list[float] = Field(default_factory=lambda: [time.time()])
    last_accessed: str = Field(default_factory=_now_iso)
    confidence: float = 0.5
    valid_from: str | None = None
    valid_until: str | None = None
    superseded_by: str | None = None
    domain: str | None = None
    tags: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=_now_iso)
    updated_at: str = Field(default_factory=_now_iso)


# ── Procedural Memory ──────────────────────────────────────────


class ProcedureStep(BaseModel):
    order: int
    description: str
    step_type: StepType = StepType.CORE
    variants: list[str] = Field(default_factory=list)
    condition: str | None = None


class Procedure(BaseModel):
    id: str = Field(default_factory=_new_id)
    name: str
    description: str | None = None
    trigger_pattern: str
    trigger_embedding: list[float] | None = None
    steps: list[ProcedureStep]
    preconditions: list[str] = Field(default_factory=list)
    postconditions: list[str] = Field(default_factory=list)
    confidence: float = 0.5
    predicted_success_rate: float | None = None
    actual_success_rate: float | None = None
    calibration_offset: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    modify_count: int = 0
    activation_score: float = 0.0
    access_times: list[float] = Field(default_factory=list)
    last_executed: str | None = None
    source_agent: str
    source_episodes: list[str] = Field(default_factory=list)
    promotion_method: PromotionMethod = PromotionMethod.AUTO
    is_composite: bool = False
    component_procedures: list[str] = Field(default_factory=list)
    parent_procedures: list[str] = Field(default_factory=list)
    transferred_to: list[str] = Field(default_factory=list)
    transfer_success_rate: float = 0.0
    status: ProcedureStatus = ProcedureStatus.DRAFT
    version: int = 1
    domain: str | None = None
    tags: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=_now_iso)
    updated_at: str = Field(default_factory=_now_iso)


# ── Metacognition ──────────────────────────────────────────────


class DomainConfidence(BaseModel):
    id: str = Field(default_factory=_new_id)
    domain: str
    confidence: float = 0.0
    episode_count: int = 0
    procedure_count: int = 0
    last_activity: str | None = None
    trend: str = "stable"
    trend_delta: float = 0.0
    created_at: str = Field(default_factory=_now_iso)
    updated_at: str = Field(default_factory=_now_iso)


class LearningGoal(BaseModel):
    id: str = Field(default_factory=_new_id)
    domain: str
    goal: str
    strategy: str | None = None
    priority: float = 0.5
    status: GoalStatus = GoalStatus.ACTIVE
    episodes_needed: int = 3
    episodes_collected: int = 0
    created_at: str = Field(default_factory=_now_iso)
    resolved_at: str | None = None


class SelfEvaluation(BaseModel):
    id: str = Field(default_factory=_new_id)
    timestamp: str = Field(default_factory=_now_iso)
    top_domains: list[str]
    weak_domains: list[str]
    improving: list[str]
    declining: list[str]
    insights: str | None = None


# ── Transfer ───────────────────────────────────────────────────


class AgentProfile(BaseModel):
    agent_id: str
    agent_name: str | None = None
    tools: list[str] = Field(default_factory=list)
    context_format: str | None = None
    model_family: str | None = None
    max_context: int | None = None
    supports_images: bool = False
    capabilities: dict[str, Any] = Field(default_factory=dict)
    first_seen: str = Field(default_factory=_now_iso)
    last_seen: str = Field(default_factory=_now_iso)


class TransferRecord(BaseModel):
    id: str = Field(default_factory=_new_id)
    procedure_id: str
    source_agent: str
    target_agent: str
    similarity_score: float
    transfer_confidence: float
    adapted: bool = False
    adaptation_details: dict[str, Any] | None = None
    outcome: str = "pending"
    timestamp: str = Field(default_factory=_now_iso)


# ── Cognitive Process Tracking ─────────────────────────────────


class ProcessRun(BaseModel):
    id: str = Field(default_factory=_new_id)
    process_name: ProcessName
    started_at: str = Field(default_factory=_now_iso)
    completed_at: str | None = None
    items_processed: int = 0
    items_created: int = 0
    items_modified: int = 0
    status: str = "running"
    error: str | None = None
    details: dict[str, Any] | None = None
