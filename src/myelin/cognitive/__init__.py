"""Six cognitive background processes inspired by ACT-R, SOAR, and Generative Agents.

Sleep cycle now has two-phase NREM + REM architecture.
Also includes PrioritizedReplay and SchemaLearner.
"""

from .nrem_sleep import NREMPhase as NREMPhase
from .rem_sleep import REMPhase as REMPhase

__all__ = ["NREMPhase", "REMPhase"]
