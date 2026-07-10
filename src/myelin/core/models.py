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
    DREAM = "dream"


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


class TrustState(str, Enum):
    SEED = "seed"
    CANDIDATE = "candidate"
    TRUSTED = "trusted"
    VALIDATED = "validated"
    STALE = "stale"


class EvidenceSource(str, Enum):
    EXECUTION = "execution"
    FEEDBACK = "feedback"
    APPROVAL = "approval"


class EvidenceOutcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"


class StepType(str, Enum):
    CORE = "core"
    VARIANT = "variant"
    OPTIONAL = "optional"


class GoalStatus(str, Enum):
    ACTIVE = "active"
    ACHIEVED = "achieved"
    ABANDONED = "abandoned"


class UpdateMode(str, Enum):
    CONFIRMED = "confirmed"
    SELECTIVE_EDIT = "selective_edit"
    INTEGRATION = "integration"
    NEW_EPISODE = "new_episode"


class ProcessName(str, Enum):
    CONSOLIDATOR = "consolidator"
    REFLECTOR = "reflector"
    PROMOTER = "promoter"
    COMPOSER = "composer"
    DECAYER = "decayer"
    CHALLENGER = "challenger"
    SLEEP = "sleep"
    NREM_SLEEP = "nrem_sleep"
    REM_SLEEP = "rem_sleep"
    SELF_MODEL = "self_model"
    CURIOSITY = "curiosity"
    PREDICTION_LEARNER = "prediction_learner"
    SCHEMA_LEARNER = "schema_learner"
    CURIOUS_EXPLORER = "curious_explorer"
    RECONSOLIDATOR = "reconsolidator"
    PRIORITIZED_REPLAY = "prioritized_replay"


class EntityType(str, Enum):
    TOOL = "tool"
    SERVICE = "service"
    CONCEPT = "concept"
    FILE = "file"
    PERSON = "person"
    CONFIG = "config"
    ERROR = "error"
    COMMAND = "command"
    PATTERN = "pattern"


class RelationType(str, Enum):
    USES = "uses"
    REQUIRES = "requires"
    PRODUCES = "produces"
    CAUSES = "causes"
    CONTRADICTS = "contradicts"
    SUPERSEDES = "supersedes"
    RELATED_TO = "related_to"
    PART_OF = "part_of"
    TRIGGERS = "triggers"
    DREAMED_CONNECTION = "dreamed_connection"


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


# ── Procedure Evidence / Trust Lifecycle ───────────────────────


class ProcedureEvidence(BaseModel):
    id: str = Field(default_factory=_new_id)
    procedure_id: str
    source: EvidenceSource
    outcome: EvidenceOutcome
    confidence_delta: float = 0.0
    episode_id: str | None = None
    timestamp: str = Field(default_factory=_now_iso)


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


class Entity(BaseModel):
    id: str = Field(default_factory=_new_id)
    name: str
    entity_type: EntityType
    canonical_name: str
    description: str | None = None
    embedding: list[float] | None = None
    mention_count: int = 1
    access_times: list[float] = Field(default_factory=lambda: [time.time()])
    first_seen: str = Field(default_factory=_now_iso)
    last_seen: str = Field(default_factory=_now_iso)
    domain: str | None = None
    source_episodes: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=_now_iso)


class EntityMention(BaseModel):
    id: str = Field(default_factory=_new_id)
    entity_id: str
    source_type: str
    source_id: str
    context_snippet: str | None = None
    role: str = "subject"
    timestamp: str = Field(default_factory=_now_iso)


class Relationship(BaseModel):
    id: str = Field(default_factory=_new_id)
    source_entity_id: str
    target_entity_id: str
    relation_type: RelationType
    strength: float = 1.0
    evidence_count: int = 1
    evidence_episodes: list[str] = Field(default_factory=list)
    domain: str | None = None
    first_observed: str = Field(default_factory=_now_iso)
    last_observed: str = Field(default_factory=_now_iso)


class CuriosityTopic(BaseModel):
    """A knowledge gap detected by the curiosity engine.

    Represents a specific target (entity, domain, procedure, or relationship)
    that the system knows little about and could benefit from exploring.
    """

    gap_type: str  # 'entity_undermentions', 'domain_low_procedures', etc.
    target_id: str
    target_name: str
    domain: str | None = None
    raw_score: float = 0.0
    novelty_score: float = 0.0
    uncertainty_score: float = 0.0
    infogain_potential: float = 0.0
    curiosity_score: float = 0.0
    exploration_attempts: int = 0
    fatigue_factor: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_now_iso)


class CuriousGoalModel(BaseModel):
    """Extended learning goal generated from a curiosity topic."""

    id: str = Field(default_factory=_new_id)
    domain: str | None = None
    goal: str
    strategy: str | None = None
    priority: float = 0.5
    status: GoalStatus = GoalStatus.ACTIVE
    episodes_needed: int = 3
    episodes_collected: int = 0
    gap_type: str | None = None
    target_id: str | None = None
    created_at: str = Field(default_factory=_now_iso)
    resolved_at: str | None = None


class TemporalState(BaseModel):
    id: str = Field(default_factory=_new_id)
    entity_id: str | None = None
    semantic_node_id: str | None = None
    state_description: str
    valid_from: str = Field(default_factory=_now_iso)
    valid_until: str | None = None
    confidence: float = 0.5
    source_episode_id: str | None = None
    domain: str | None = None
    created_at: str = Field(default_factory=_now_iso)


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


class SchemaType(str, Enum):
    BEHAVIORAL = "behavioral"
    PREFERENCE = "preference"
    DOMAIN_MODEL = "domain_model"


class SchemaStatus(str, Enum):
    HYPOTHESIS = "hypothesis"
    ACTIVE = "active"
    REFUTED = "refuted"
    ARCHIVED = "archived"


class SchemaModel(BaseModel):
    id: str = Field(default_factory=_new_id)
    name: str
    description: str | None = None
    behavioral_pattern: str
    schema_type: SchemaType = SchemaType.BEHAVIORAL
    semantic_source_ids: list[str] = Field(default_factory=list)
    episode_source_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.5
    induction_count: int = 1
    domain: str | None = None
    conditions: list[str] = Field(default_factory=list)
    exceptions: list[str] = Field(default_factory=list)
    version: int = 1
    status: SchemaStatus = SchemaStatus.HYPOTHESIS
    created_at: str = Field(default_factory=_now_iso)
    updated_at: str = Field(default_factory=_now_iso)


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


# ── Provenance ─────────────────────────────────────────────────


class RetrievalProvenance(BaseModel):
    """Durable provenance metadata attached to every retrieved result.

    Carries the lineage and retrieval-time signals so consumers can audit
    where a result came from, how it was scored, and when it was fetched.
    All fields are JSON-serialisable by design.
    """

    source_id: str
    """The primary ID of the source memory (episode, semantic node, procedure)."""

    source_type: str
    """'episode', 'semantic', or 'procedure'."""

    source_agent: str
    """Agent that created the source memory, or 'unknown'."""

    domain: str | None = None
    """Optional domain the source memory is scoped to."""

    timestamp: str | None = None
    """ISO-8601 timestamp of when the source memory was created / recorded."""

    retrieved_at: str = Field(default_factory=_now_iso)
    """ISO-8601 timestamp of when this retrieval occurred."""

    retrieval_signals: dict[str, float] = Field(default_factory=dict)
    """Per-signal scores used during retrieval (text, vector, entity, temporal, activation, importance)."""

    composite_score: float = 0.0
    """Fused multi-signal composite score at retrieval time."""

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict safe for JSON serialisation and dict storage."""
        return self.model_dump()

    @classmethod
    def from_result(
        cls, result: dict[str, Any], retrieved_at: str | None = None
    ) -> RetrievalProvenance:
        """Build provenance from a raw retriever result dict (non-destructive)."""
        import datetime

        return cls(
            source_id=result.get("id", ""),
            source_type=str(result.get("_source_type", "unknown")),
            source_agent=str(result.get("source_agent", "unknown")),
            domain=result.get("domain"),
            timestamp=result.get("timestamp") or result.get("created_at"),
            retrieved_at=retrieved_at or datetime.datetime.utcnow().isoformat(),
            retrieval_signals=result.get("_scores", {}),
            composite_score=float(result.get("_composite_score", 0.0)),
        )
