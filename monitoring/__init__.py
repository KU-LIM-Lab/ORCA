# monitoring/__init__.py
"""
Unified monitoring and metrics collection for ORCA system.

This package provides comprehensive monitoring capabilities including:
- Metrics collection (execution time, memory, tokens, errors)
- LLM-specific tracking and monitoring
- Execution tracing and logging
- Experiment tracking and artifact management

Key Components:
- MetricsCollector: Core metrics collection
- LLM tracking: Language model specific monitoring
- Experiment tracking: Run context and event logging
"""

from .metrics import MetricsCollector, set_metrics_collector, get_metrics_collector
from .llm import track_llm_call, track_llm_generation, create_llm_tracker, record_llm_tokens
from .tracing import (
    TraceEvent, TraceCollector, AgentTracer, get_tracer, 
    trace_execution, trace_context
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
    
    # Tracing
    "TraceEvent",
    "TraceCollector", 
    "AgentTracer",
    "get_tracer",
    "trace_execution",
    "trace_context",
]
