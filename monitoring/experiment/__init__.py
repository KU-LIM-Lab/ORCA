"""
Experiment tracking system for ORCA user studies.

This module provides comprehensive experiment tracking including:
- Run context management with automatic directory creation
- Structured event logging (JSONL format)
- Artifact management with integrity checks
- Integration with existing monitoring infrastructure
"""

from .run_context import RunContext
from .event_logger import EventLogger
from .artifact_manager import ArtifactManager
from .log_config import configure_experiment_logging
from .utils import get_event_logger, get_artifact_manager

__all__ = [
    "RunContext",
    "EventLogger",
    "ArtifactManager",
    "configure_experiment_logging",
    "get_event_logger",
    "get_artifact_manager",
]

