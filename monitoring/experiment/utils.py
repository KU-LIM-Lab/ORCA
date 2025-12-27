"""
Utility functions for experiment tracking.

Provides convenient access to EventLogger and ArtifactManager from anywhere in the codebase.
"""

from typing import Optional, Any


def get_event_logger() -> Optional[Any]:
    """
    Get the active EventLogger instance if experiment mode is enabled.
    
    Returns:
        EventLogger instance or None
    """
    try:
        from monitoring.metrics.collector import get_metrics_collector
        collector = get_metrics_collector()
        if collector and hasattr(collector, 'event_logger'):
            return collector.event_logger
    except Exception:
        pass
    return None


def get_artifact_manager() -> Optional[Any]:
    """
    Get the active ArtifactManager instance if experiment mode is enabled.
    
    Returns:
        ArtifactManager instance or None
    """
    try:
        from monitoring.metrics.collector import get_metrics_collector
        collector = get_metrics_collector()
        if collector and hasattr(collector, 'artifact_manager'):
            return collector.artifact_manager
    except Exception:
        pass
    return None

