# core/metrics.py
import time
import psutil
import threading
import json
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from contextlib import contextmanager
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of metrics to track"""
    EXECUTION_TIME = "execution_time"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    TOKEN_COUNT = "token_count"
    API_CALLS = "api_calls"
    ERROR_COUNT = "error_count"
    CUSTOM = "custom"

@dataclass
class MetricPoint:
    """Single metric measurement"""
    metric_type: MetricType
    value: float
    timestamp: datetime
    agent_name: str
    session_id: str
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat(),
            'metric_type': self.metric_type.value
        }

@dataclass
class PerformanceSnapshot:
    """Snapshot of system performance at a point in time"""
    timestamp: datetime
    memory_mb: float
    cpu_percent: float
    process_memory_mb: float
    thread_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat()
        }

class MetricsCollector:
    """Central metrics collection and management"""
    
    def __init__(self, session_id: str, event_logger: Any = None, artifact_manager: Any = None, 
                 max_metrics: int = 10000, max_snapshots: int = 3600):
        self.session_id = session_id
        self.metrics: List[MetricPoint] = []
        self.performance_snapshots: List[PerformanceSnapshot] = []
        self._lock = threading.Lock()
        self._start_time = time.time()
        self.event_logger = event_logger  # EventLogger for experiment tracking
        self.artifact_manager = artifact_manager  # ArtifactManager for experiment tracking
        
        # Memory limits to prevent unbounded growth
        self.max_metrics = max_metrics  # Maximum number of metrics to keep
        self.max_snapshots = max_snapshots  # Maximum number of snapshots to keep (e.g., 1 hour at 1s interval)
        
        # Performance tracking
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        
    def start_monitoring(self, interval: float = 1.0):
        """Start background performance monitoring"""
        if self._monitoring_thread is None or not self._monitoring_thread.is_alive():
            self._stop_monitoring.clear()
            self._monitoring_thread = threading.Thread(
                target=self._monitor_performance,
                args=(interval,),
                daemon=True
            )
            self._monitoring_thread.start()
            logger.info(f"Started performance monitoring for session {self.session_id}")
    
    def stop_monitoring(self):
        """Stop background performance monitoring"""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._stop_monitoring.set()
            self._monitoring_thread.join(timeout=2.0)
            logger.info(f"Stopped performance monitoring for session {self.session_id}")
    
    def _monitor_performance(self, interval: float):
        """Background thread for continuous performance monitoring"""
        while not self._stop_monitoring.wait(interval):
            try:
                snapshot = self._capture_performance_snapshot()
                with self._lock:
                    self.performance_snapshots.append(snapshot)
                    # Limit snapshot list size to prevent memory growth
                    if len(self.performance_snapshots) > self.max_snapshots:
                        # Remove oldest snapshots (keep most recent)
                        excess = len(self.performance_snapshots) - self.max_snapshots
                        self.performance_snapshots = self.performance_snapshots[excess:]
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
    
    def _capture_performance_snapshot(self) -> PerformanceSnapshot:
        """Capture current system performance"""
        process = psutil.Process()
        return PerformanceSnapshot(
            timestamp=datetime.now(),
            memory_mb=psutil.virtual_memory().used / 1024 / 1024,
            cpu_percent=psutil.cpu_percent(),
            process_memory_mb=process.memory_info().rss / 1024 / 1024,
            thread_count=process.num_threads()
        )
    
    def record_metric(self, 
                     metric_type: MetricType,
                     value: float,
                     agent_name: str,
                     metadata: Optional[Dict[str, Any]] = None):
        """Record a single metric"""
        metric = MetricPoint(
            metric_type=metric_type,
            value=value,
            timestamp=datetime.now(),
            agent_name=agent_name,
            session_id=self.session_id,
            metadata=metadata or {}
        )
        
        with self._lock:
            self.metrics.append(metric)
            # Limit metrics list size to prevent memory growth
            if len(self.metrics) > self.max_metrics:
                # Remove oldest metrics (keep most recent)
                excess = len(self.metrics) - self.max_metrics
                self.metrics = self.metrics[excess:]
        
        logger.debug(f"Recorded metric: {metric_type.value}={value} for {agent_name}")
    
    def record_execution_time(self, agent_name: str, duration: float, metadata: Optional[Dict[str, Any]] = None):
        """Record execution time for an agent"""
        self.record_metric(MetricType.EXECUTION_TIME, duration, agent_name, metadata)
    
    def record_memory_usage(self, agent_name: str, memory_mb: float, metadata: Optional[Dict[str, Any]] = None):
        """Record memory usage for an agent"""
        self.record_metric(MetricType.MEMORY_USAGE, memory_mb, agent_name, metadata)
    
    def record_token_count(self, agent_name: str, token_count: int, metadata: Optional[Dict[str, Any]] = None):
        """Record token count for LLM operations"""
        self.record_metric(MetricType.TOKEN_COUNT, float(token_count), agent_name, metadata)
    
    def record_api_call(self, agent_name: str, api_name: str, duration: float, metadata: Optional[Dict[str, Any]] = None):
        """Record API call metrics"""
        api_metadata = {"api_name": api_name, **(metadata or {})}
        self.record_metric(MetricType.API_CALLS, duration, agent_name, api_metadata)
    
    def record_error(self, agent_name: str, error_type: str, metadata: Optional[Dict[str, Any]] = None):
        """Record error occurrence"""
        error_metadata = {"error_type": error_type, **(metadata or {})}
        self.record_metric(MetricType.ERROR_COUNT, 1.0, agent_name, error_metadata)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        with self._lock:
            if not self.metrics:
                return {"total_metrics": 0}
            
            # Group by metric type
            by_type = {}
            for metric in self.metrics:
                metric_type = metric.metric_type.value
                if metric_type not in by_type:
                    by_type[metric_type] = []
                by_type[metric_type].append(metric.value)
            
            # Calculate statistics
            summary = {
                "session_id": self.session_id,
                "total_metrics": len(self.metrics),
                "session_duration": time.time() - self._start_time,
                "by_type": {}
            }
            
            for metric_type, values in by_type.items():
                summary["by_type"][metric_type] = {
                    "count": len(values),
                    "total": sum(values),
                    "average": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values)
                }
            
            return summary
    
    def get_agent_metrics(self, agent_name: str) -> Dict[str, Any]:
        """Get metrics for a specific agent"""
        with self._lock:
            agent_metrics = [m for m in self.metrics if m.agent_name == agent_name]
            
            if not agent_metrics:
                return {"agent_name": agent_name, "total_metrics": 0}
            
            # Group by metric type
            by_type = {}
            for metric in agent_metrics:
                metric_type = metric.metric_type.value
                if metric_type not in by_type:
                    by_type[metric_type] = []
                by_type[metric_type].append(metric.value)
            
            summary = {
                "agent_name": agent_name,
                "total_metrics": len(agent_metrics),
                "by_type": {}
            }
            
            for metric_type, values in by_type.items():
                summary["by_type"][metric_type] = {
                    "count": len(values),
                    "total": sum(values),
                    "average": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values)
                }
            
            return summary
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format"""
        with self._lock:
            data = {
                "session_id": self.session_id,
                "metrics": [m.to_dict() for m in self.metrics],
                "performance_snapshots": [s.to_dict() for s in self.performance_snapshots],
                "summary": self.get_metrics_summary()
            }
            
            if format.lower() == "json":
                return json.dumps(data, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
    
    def clear_metrics(self):
        """Clear all metrics (useful for testing)"""
        with self._lock:
            self.metrics.clear()
            self.performance_snapshots.clear()

# Global metrics collector instance
_global_collector: Optional[MetricsCollector] = None

def get_metrics_collector() -> Optional[MetricsCollector]:
    """Get the global metrics collector"""
    return _global_collector

def set_metrics_collector(collector: MetricsCollector):
    """Set the global metrics collector"""
    global _global_collector
    _global_collector = collector

def record_metric(metric_type: MetricType, value: float, agent_name: str, metadata: Optional[Dict[str, Any]] = None):
    """Record a metric using the global collector"""
    if _global_collector:
        _global_collector.record_metric(metric_type, value, agent_name, metadata)

@contextmanager
def track_execution_time(agent_name: str, metadata: Optional[Dict[str, Any]] = None):
    """Context manager to track execution time"""
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024
    
    try:
        yield
    finally:
        duration = time.time() - start_time
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_delta = end_memory - start_memory
        
        if _global_collector:
            _global_collector.record_execution_time(agent_name, duration, metadata)
            _global_collector.record_memory_usage(agent_name, memory_delta, metadata)

@contextmanager
def track_memory_usage(agent_name: str, metadata: Optional[Dict[str, Any]] = None):
    """Context manager to track memory usage"""
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024
    
    try:
        yield
    finally:
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_used = end_memory - start_memory
        
        if _global_collector:
            _global_collector.record_memory_usage(agent_name, memory_used, metadata)
