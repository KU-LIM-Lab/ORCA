# monitoring/tracing/tracer.py
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
import json
import uuid
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from contextlib import contextmanager

class TraceLevel(Enum):
    """Trace level enumeration"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class TraceEvent:
    """Individual trace event"""
    id: str
    timestamp: datetime
    level: TraceLevel
    agent: str
    event_type: str
    message: str
    data: Optional[Dict[str, Any]] = None
    duration: Optional[float] = None
    parent_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['level'] = self.level.value
        return result

class TraceCollector:
    """Collects and manages trace events"""
    
    def __init__(self, max_events: int = 10000):
        self.max_events = max_events
        self.events: List[TraceEvent] = []
        self.lock = threading.Lock()
        self.active_traces: Dict[str, str] = {}  # trace_id -> parent_id mapping
    
    def add_event(self, event: TraceEvent) -> None:
        """Add trace event"""
        with self.lock:
            # Remove oldest events if at capacity
            while len(self.events) >= self.max_events:
                self.events.pop(0)
            
            self.events.append(event)
    
    def create_event(self, level: TraceLevel, agent: str, event_type: str, 
                    message: str, data: Optional[Dict[str, Any]] = None,
                    parent_id: Optional[str] = None) -> TraceEvent:
        """Create trace event"""
        event_id = str(uuid.uuid4())
        return TraceEvent(
            id=event_id,
            timestamp=datetime.now(),
            level=level,
            agent=agent,
            event_type=event_type,
            message=message,
            data=data,
            parent_id=parent_id
        )
    
    def log(self, level: TraceLevel, agent: str, event_type: str, 
            message: str, data: Optional[Dict[str, Any]] = None,
            parent_id: Optional[str] = None) -> str:
        """Log trace event and return event ID"""
        event = self.create_event(level, agent, event_type, message, data, parent_id)
        self.add_event(event)
        return event.id
    
    def get_events(self, agent: Optional[str] = None, 
                  level: Optional[TraceLevel] = None,
                  event_type: Optional[str] = None,
                  limit: Optional[int] = None) -> List[TraceEvent]:
        """Get filtered trace events"""
        with self.lock:
            events = self.events.copy()
        
        # Apply filters
        if agent:
            events = [e for e in events if e.agent == agent]
        if level:
            events = [e for e in events if e.level == level]
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        # Apply limit
        if limit:
            events = events[-limit:]
        
        return events
    
    def get_events_by_trace(self, trace_id: str) -> List[TraceEvent]:
        """Get all events in a trace"""
        with self.lock:
            events = self.events.copy()
        
        # Find all events with this trace_id or parent_id
        trace_events = []
        for event in events:
            if event.id == trace_id or event.parent_id == trace_id:
                trace_events.append(event)
        
        return trace_events
    
    def clear(self) -> None:
        """Clear all events"""
        with self.lock:
            self.events.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get trace statistics"""
        with self.lock:
            events = self.events.copy()
        
        if not events:
            return {"total_events": 0}
        
        # Count by level
        level_counts = {}
        for event in events:
            level_counts[event.level.value] = level_counts.get(event.level.value, 0) + 1
        
        # Count by agent
        agent_counts = {}
        for event in events:
            agent_counts[event.agent] = agent_counts.get(event.agent, 0) + 1
        
        # Count by event type
        type_counts = {}
        for event in events:
            type_counts[event.event_type] = type_counts.get(event.event_type, 0) + 1
        
        return {
            "total_events": len(events),
            "level_counts": level_counts,
            "agent_counts": agent_counts,
            "type_counts": type_counts,
            "oldest_event": events[0].timestamp.isoformat() if events else None,
            "newest_event": events[-1].timestamp.isoformat() if events else None
        }

class AgentTracer:
    """Tracer for individual agents"""
    
    def __init__(self, agent_name: str, collector: TraceCollector):
        self.agent_name = agent_name
        self.collector = collector
        self.current_trace_id: Optional[str] = None
    
    def start_trace(self, trace_type: str, message: str, 
                   data: Optional[Dict[str, Any]] = None) -> str:
        """Start a new trace"""
        self.current_trace_id = self.collector.log(
            TraceLevel.INFO, self.agent_name, trace_type, message, data
        )
        return self.current_trace_id
    
    def end_trace(self, trace_id: str, message: str = "Trace completed",
                 data: Optional[Dict[str, Any]] = None) -> None:
        """End a trace"""
        self.collector.log(
            TraceLevel.INFO, self.agent_name, "trace_end", message, data, trace_id
        )
        if self.current_trace_id == trace_id:
            self.current_trace_id = None
    
    def log(self, level: TraceLevel, event_type: str, message: str,
           data: Optional[Dict[str, Any]] = None) -> str:
        """Log event"""
        return self.collector.log(
            level, self.agent_name, event_type, message, data, self.current_trace_id
        )
    
    def debug(self, message: str, data: Optional[Dict[str, Any]] = None) -> str:
        """Log debug message"""
        return self.log(TraceLevel.DEBUG, "debug", message, data)
    
    def info(self, message: str, data: Optional[Dict[str, Any]] = None) -> str:
        """Log info message"""
        return self.log(TraceLevel.INFO, "info", message, data)
    
    def warning(self, message: str, data: Optional[Dict[str, Any]] = None) -> str:
        """Log warning message"""
        return self.log(TraceLevel.WARNING, "warning", message, data)
    
    def error(self, message: str, data: Optional[Dict[str, Any]] = None) -> str:
        """Log error message"""
        return self.log(TraceLevel.ERROR, "error", message, data)
    
    def critical(self, message: str, data: Optional[Dict[str, Any]] = None) -> str:
        """Log critical message"""
        return self.log(TraceLevel.CRITICAL, "critical", message, data)

# Global trace collector
trace_collector = TraceCollector()

def get_tracer(agent_name: str) -> AgentTracer:
    """Get tracer for agent"""
    return AgentTracer(agent_name, trace_collector)

# Decorators for automatic tracing
def trace_execution(agent_name: str, trace_type: str = "execution"):
    """Decorator to trace function execution"""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            tracer = get_tracer(agent_name)
            trace_id = tracer.start_trace(trace_type, f"Executing {func.__name__}")
            
            try:
                start_time = datetime.now()
                result = func(*args, **kwargs)
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                tracer.end_trace(trace_id, f"Completed {func.__name__}", 
                               {"duration": duration, "success": True})
                return result
            except Exception as e:
                tracer.error(f"Error in {func.__name__}: {str(e)}", 
                           {"error": str(e), "args": str(args), "kwargs": str(kwargs)})
                raise
        
        return wrapper
    return decorator

@contextmanager
def trace_context(agent_name: str, trace_type: str, message: str, 
                 data: Optional[Dict[str, Any]] = None):
    """Context manager for tracing"""
    tracer = get_tracer(agent_name)
    trace_id = tracer.start_trace(trace_type, message, data)
    
    try:
        yield tracer
    finally:
        tracer.end_trace(trace_id, f"Completed {trace_type}")

# Utility functions
def export_traces(filename: str, agent: Optional[str] = None) -> None:
    """Export traces to file"""
    events = trace_collector.get_events(agent=agent)
    events_dict = [event.to_dict() for event in events]
    
    with open(filename, 'w') as f:
        json.dump(events_dict, f, indent=2)

def get_trace_summary(agent: Optional[str] = None) -> Dict[str, Any]:
    """Get trace summary"""
    events = trace_collector.get_events(agent=agent)
    stats = trace_collector.get_stats()
    
    return {
        "stats": stats,
        "recent_events": [event.to_dict() for event in events[-10:]]
    }
