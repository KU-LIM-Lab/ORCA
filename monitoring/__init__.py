# monitoring/__init__.py
"""
Monitoring and metrics collection for ORCA system.

Key Components:
- MetricsCollector: Core metrics collection (execution time, memory, errors)
- Experiment tracking: Run context and artifact management
"""

from .metrics import MetricsCollector, set_metrics_collector, get_metrics_collector

__all__ = [
    "MetricsCollector",
    "set_metrics_collector",
    "get_metrics_collector",
]
