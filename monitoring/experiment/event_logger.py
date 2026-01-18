"""
Event logger for structured experiment tracking.

Provides append-only JSONL logging with schema validation.
All events include common fields: timestamp, run_id, participant_id, condition, etc.
"""

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class EventLogger:
    """Thread-safe JSONL event logger for experiment tracking."""
    
    def __init__(
        self,
        events_file: Path,
        run_id: str,
        participant_id: str,
        condition: str,
        task_id: str
    ):
        """
        Initialize event logger.
        
        Args:
            events_file: Path to events.jsonl file
            run_id: Unique run identifier (UUID)
            participant_id: Participant identifier
            condition: Study condition (orca, baseline)
            task_id: Task identifier
        """
        self.events_file = events_file
        self.run_id = run_id
        self.participant_id = participant_id
        self.condition = condition
        self.task_id = task_id
        
        self._lock = threading.Lock()
        self._event_count = 0
        
        # Ensure parent directory exists
        self.events_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"EventLogger initialized: {self.events_file}")
    
    def log_event(
        self,
        event_type: str,
        data: Optional[Dict[str, Any]] = None,
        step_id: Optional[str] = None
    ) -> None:
        """
        Log a structured event to JSONL file.
        
        Args:
            event_type: Type of event (session_start, step_enter, tool_call, etc.)
            data: Event-specific data dictionary
            step_id: Optional step identifier (1, 2, 3 for pipeline steps)
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "run_id": self.run_id,
            "participant_id": self.participant_id,
            "condition": self.condition,
            "event_type": event_type,
        }
        
        # Add optional fields
        event["task_id"] = self.task_id
        if step_id:
            event["step_id"] = step_id
        if data:
            event["data"] = data
        
        # Thread-safe write
        with self._lock:
            try:
                with open(self.events_file, "a") as f:
                    f.write(json.dumps(event, default=str) + "\n")
                self._event_count += 1
            except Exception as e:
                logger.error(f"Failed to log event {event_type}: {e}")
    
    def log_session_start(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log session start event."""
        data = {"metadata": metadata or {}}
        self.log_event("session_start", data)
    
    def log_session_end(
        self,
        termination_reason: str = "completed",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log session end event.
        
        Args:
            termination_reason: Reason for termination (completed, timeout, error, user_abort)
            metadata: Additional metadata
        """
        data = {
            "termination_reason": termination_reason,
            "total_events": self._event_count,
            "metadata": metadata or {}
        }
        self.log_event("session_end", data)
    
    def log_step_enter(self, step_id: str, substep: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log step entry."""
        data = {"substep": substep, "metadata": metadata or {}}
        self.log_event("step_enter", data, step_id=step_id)
    
    def log_step_exit(
        self,
        step_id: str,
        substep: str,
        success: bool = True,
        duration: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log step exit."""
        data = {
            "substep": substep,
            "success": success,
            "metadata": metadata or {}
        }
        if duration is not None:
            data["duration"] = duration
        self.log_event("step_exit", data, step_id=step_id)
    
    def log_tool_call_start(
        self,
        tool_name: str,
        step_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log tool call start."""
        data = {"tool_name": tool_name, "metadata": metadata or {}}
        self.log_event("tool_call_start", data, step_id=step_id)
    
    def log_tool_call_end(
        self,
        tool_name: str,
        duration: float,
        success: bool = True,
        error: Optional[str] = None,
        step_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log tool call end."""
        data = {
            "tool_name": tool_name,
            "duration": duration,
            "success": success,
            "metadata": metadata or {}
        }
        if error:
            data["error"] = error
        self.log_event("tool_call_end", data, step_id=step_id)
    
    def log_hitl_prompt_shown(
        self,
        step_id: str,
        phase: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log HITL prompt shown to user."""
        data = {
            "phase": phase,
            "description": description,
            "metadata": metadata or {}
        }
        self.log_event("hitl_prompt_shown", data, step_id=step_id)
    
    def log_hitl_decision(
        self,
        step_id: str,
        decision: str,
        edits: Optional[Dict[str, Any]] = None,
        feedback: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log HITL decision.
        
        Args:
            step_id: Step identifier
            decision: User decision (approve, edit, rerun, abort)
            edits: User edits if decision is 'edit'
            feedback: User feedback if decision is 'rerun'
            metadata: Additional metadata
        """
        data = {
            "decision": decision,
            "metadata": metadata or {}
        }
        if edits:
            data["edits"] = edits
        if feedback:
            data["feedback"] = feedback
        self.log_event("hitl_decision", data, step_id=step_id)
    
    def log_hitl_applied(
        self,
        step_id: str,
        applied: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log HITL decision applied."""
        data = {"applied": applied, "metadata": metadata or {}}
        self.log_event("hitl_applied", data, step_id=step_id)
    
    def log_artifact_saved(
        self,
        artifact_type: str,
        path: str,
        sha256: Optional[str] = None,
        step_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log artifact saved."""
        data = {
            "artifact_type": artifact_type,
            "path": path,
            "metadata": metadata or {}
        }
        if sha256:
            data["sha256"] = sha256
        self.log_event("artifact_saved", data, step_id=step_id)
    
    def log_llm_call_start(
        self,
        model: str,
        agent_name: str,
        operation: str,
        step_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log LLM call start."""
        data = {
            "model": model,
            "agent_name": agent_name,
            "operation": operation,
            "metadata": metadata or {}
        }
        self.log_event("llm_call_start", data, step_id=step_id)
    
    def log_llm_call_end(
        self,
        model: str,
        agent_name: str,
        operation: str,
        duration: float,
        token_count: Optional[int] = None,
        success: bool = True,
        error: Optional[str] = None,
        step_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log LLM call end."""
        data = {
            "model": model,
            "agent_name": agent_name,
            "operation": operation,
            "duration": duration,
            "success": success,
            "metadata": metadata or {}
        }
        if token_count:
            data["token_count"] = token_count
        if error:
            data["error"] = error
        self.log_event("llm_call_end", data, step_id=step_id)
    
    def get_event_count(self) -> int:
        """Get total number of events logged."""
        return self._event_count

