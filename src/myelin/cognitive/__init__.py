"""Six cognitive background processes inspired by ACT-R, SOAR, and Generative Agents.

Sleep cycle now has two-phase NREM + REM architecture.
Also includes PrioritizedReplay, SchemaLearner, and ConsolidationScheduler.
"""

from .nrem_sleep import NREMPhase as NREMPhase
from .rem_sleep import REMPhase as REMPhase
from .scheduler import ConsolidationScheduler as ConsolidationScheduler

__all__ = ["NREMPhase", "REMPhase", "ConsolidationScheduler"]
