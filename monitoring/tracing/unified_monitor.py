# monitoring/tracing/unified_monitor.py
from typing import Any, Dict, List, Optional
from datetime import datetime
import time
import psutil
import threading
from contextlib import contextmanager

from .tracer import TraceCollector, TraceLevel, get_tracer
from ..metrics.collector import MetricsCollector, MetricType, track_execution_time, track_memory_usage

class UnifiedMonitor:
    """Unified monitoring system combining tracing and metrics"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.trace_collector = TraceCollector()
        self.metrics_collector = MetricsCollector(session_id)
        self._lock = threading.Lock()
    
    def get_tracer(self, agent_name: str):
        """Get tracer for agent"""
        return get_tracer(agent_name)
    
    def start_monitoring(self, interval: float = 1.0):
        """Start background monitoring"""
        self.metrics_collector.start_monitoring(interval)
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.metrics_collector.stop_monitoring()
    
    def record_metric(self, metric_type: MetricType, value: float, 
                     agent_name: str, metadata: Optional[Dict[str, Any]] = None):
        """Record a metric"""
        self.metrics_collector.record_metric(metric_type, value, agent_name, metadata)
    
    def record_execution_time(self, agent_name: str, duration: float, 
                            metadata: Optional[Dict[str, Any]] = None):
        """Record execution time"""
        self.metrics_collector.record_execution_time(agent_name, duration, metadata)
    
    def record_memory_usage(self, agent_name: str, memory_mb: float, 
                          metadata: Optional[Dict[str, Any]] = None):
        """Record memory usage"""
        self.metrics_collector.record_memory_usage(agent_name, memory_mb, metadata)
    
    def record_error(self, agent_name: str, error_type: str, 
                    metadata: Optional[Dict[str, Any]] = None):
        """Record error"""
        self.metrics_collector.record_error(agent_name, error_type, metadata)
        
        # Also log as trace event
        tracer = self.get_tracer(agent_name)
        tracer.error(f"Error occurred: {error_type}", metadata)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        return self.metrics_collector.get_metrics_summary()
    
    def get_trace_summary(self, agent: Optional[str] = None) -> Dict[str, Any]:
        """Get trace summary"""
        events = self.trace_collector.get_events(agent=agent)
        stats = self.trace_collector.get_stats()
        
        return {
            "stats": stats,
            "recent_events": [event.to_dict() for event in events[-10:]]
        }
    
    def get_unified_summary(self) -> Dict[str, Any]:
        """Get unified summary of both metrics and traces"""
        return {
            "session_id": self.session_id,
            "metrics": self.get_metrics_summary(),
            "traces": self.get_trace_summary(),
            "timestamp": datetime.now().isoformat()
        }
    
    @contextmanager
    def track_execution(self, agent_name: str, operation: str, 
                       metadata: Optional[Dict[str, Any]] = None):
        """Context manager for tracking execution with both metrics and traces"""
        tracer = self.get_tracer(agent_name)
        trace_id = tracer.start_trace("execution", f"Executing {operation}", metadata)
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            yield tracer
        except Exception as e:
            # Record error in both systems
            self.record_error(agent_name, type(e).__name__, 
                            {"error": str(e), "operation": operation})
            tracer.error(f"Error in {operation}: {str(e)}", 
                       {"error": str(e), "operation": operation})
            raise
        finally:
            # Record metrics
            duration = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_delta = end_memory - start_memory
            
            self.record_execution_time(agent_name, duration, metadata)
            self.record_memory_usage(agent_name, memory_delta, metadata)
            
            # End trace
            tracer.end_trace(trace_id, f"Completed {operation}", 
                           {"duration": duration, "memory_delta": memory_delta})
    
    def export_all_data(self, filename: str) -> None:
        """Export both metrics and traces to file"""
        import json
        
        data = {
            "session_id": self.session_id,
            "metrics": self.metrics_collector.export_metrics("json"),
            "traces": {
                "events": [event.to_dict() for event in self.trace_collector.events],
                "stats": self.trace_collector.get_stats()
            },
            "exported_at": datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

# Global unified monitor
_unified_monitor: Optional[UnifiedMonitor] = None

def get_unified_monitor(session_id: str = "default") -> UnifiedMonitor:
    """Get or create global unified monitor"""
    global _unified_monitor
    if _unified_monitor is None or _unified_monitor.session_id != session_id:
        _unified_monitor = UnifiedMonitor(session_id)
    return _unified_monitor

def set_unified_monitor(monitor: UnifiedMonitor):
    """Set global unified monitor"""
    global _unified_monitor
    _unified_monitor = monitor
