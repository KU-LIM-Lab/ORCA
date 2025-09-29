# monitoring/__init__.py
"""
Unified monitoring and metrics collection for ORCA system.

This package provides comprehensive monitoring capabilities including:
- Metrics collection (execution time, memory, tokens, errors)
- LLM-specific tracking and monitoring
- Execution tracing and logging
- Visualization and dashboard creation
- Unified monitoring interface

Key Components:
- MetricsCollector: Core metrics collection
- UnifiedMonitor: Integrated tracing and metrics
- LLM tracking: Language model specific monitoring
- Visualization: Dashboard and reporting tools
"""

from .metrics import MetricsCollector, set_metrics_collector, get_metrics_collector
from .llm import track_llm_call, track_llm_generation, create_llm_tracker, record_llm_tokens
from .visualization import create_dashboard, MetricsDashboard
from .tracing import (
    TraceEvent, TraceCollector, AgentTracer, get_tracer, 
    trace_execution, trace_context, UnifiedMonitor, get_unified_monitor
)

__all__ = [
    # Core metrics
    "MetricsCollector",
    "set_metrics_collector", 
    "get_metrics_collector",
    
    # LLM tracking
    "track_llm_call",
    "track_llm_generation", 
    "create_llm_tracker",
    "record_llm_tokens",
    
    # Visualization
    "create_dashboard",
    "MetricsDashboard",
    
    # Tracing and unified monitoring
    "TraceEvent",
    "TraceCollector", 
    "AgentTracer",
    "get_tracer",
    "trace_execution",
    "trace_context",
    "UnifiedMonitor",
    "get_unified_monitor"
]
