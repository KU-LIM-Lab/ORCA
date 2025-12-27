"""
Run context manager for experiment sessions.

Provides centralized management of:
- Run directory creation and cleanup
- EventLogger and ArtifactManager initialization
- Metrics and tracing integration
- Automatic zip generation for submission
"""

import logging
import uuid
import zipfile
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from .event_logger import EventLogger
from .artifact_manager import ArtifactManager
from .log_config import configure_experiment_logging

logger = logging.getLogger(__name__)


class RunContext:
    """Context manager for experiment run sessions."""
    
    def __init__(
        self,
        participant_id: str,
        condition: str,
        task_id: str,
        base_dir: Optional[Path] = None,
        run_id: Optional[str] = None
    ):
        """
        Initialize run context.
        
        Args:
            participant_id: Participant identifier (e.g., P001)
            condition: Study condition (orca, baseline)
            task_id: Task identifier (e.g., marketing, operations)
            base_dir: Base directory for runs (default: ./runs)
            run_id: Optional run ID (will generate UUID if not provided)
        """
        self.participant_id = participant_id
        self.condition = condition
        self.task_id = task_id
        self.run_id = run_id or str(uuid.uuid4())
        
        # Directory structure: runs/{participant_id}/{condition}/{task_id}/{run_id}/
        self.base_dir = Path(base_dir) if base_dir else Path("runs")
        self.run_dir = self.base_dir / participant_id / condition / task_id / self.run_id
        
        self.artifacts_dir = self.run_dir / "artifacts"
        self.events_file = self.run_dir / "events.jsonl"
        
        # Managers (initialized in __enter__)
        self.event_logger: Optional[EventLogger] = None
        self.artifact_manager: Optional[ArtifactManager] = None
        self.metrics_collector: Optional[Any] = None
        
        # Session metadata
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.environment_info: Dict[str, Any] = {}
        
        logger.info(f"RunContext created: {self.run_dir}")
    
    def __enter__(self) -> "RunContext":
        """Enter context: create directories and initialize managers."""
        self.start_time = datetime.now()
        
        # Create directory structure
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize event logger
        self.event_logger = EventLogger(
            events_file=self.events_file,
            run_id=self.run_id,
            participant_id=self.participant_id,
            condition=self.condition,
            task_id=self.task_id
        )
        
        # Initialize artifact manager
        self.artifact_manager = ArtifactManager(
            artifacts_dir=self.artifacts_dir,
            event_logger=self.event_logger
        )
        
        # Configure logging
        configure_experiment_logging(
            run_dir=self.run_dir,
            level="INFO",
            suppress_terminal=True
        )
        
        # Capture environment info
        self.environment_info = self._capture_environment_info()
        
        # Log session start
        self.event_logger.log_session_start(metadata={
            "environment": self.environment_info,
            "run_dir": str(self.run_dir)
        })
        
        logger.info(f"Run context entered: {self.run_dir}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context: finalize and create submission package."""
        self.end_time = datetime.now()
        
        # Determine termination reason
        if exc_type is None:
            termination_reason = "completed"
        elif exc_type == KeyboardInterrupt:
            termination_reason = "user_abort"
        else:
            termination_reason = "error"
        
        # Log session end
        if self.event_logger:
            duration = (self.end_time - self.start_time).total_seconds() if self.start_time else None
            self.event_logger.log_session_end(
                termination_reason=termination_reason,
                metadata={
                    "duration_seconds": duration,
                    "error": str(exc_val) if exc_val else None
                }
            )
        
        # Finalize
        try:
            self.finalize()
        except Exception as e:
            logger.error(f"Error during finalization: {e}")
        
        logger.info(f"Run context exited: {self.run_dir}")
        
        # Don't suppress exceptions
        return False
    
    def finalize(self) -> None:
        """Finalize run: save metadata, export metrics/traces, create zip."""
        logger.info("Finalizing run context...")
        
        # Save artifact manifest
        if self.artifact_manager:
            try:
                self.artifact_manager.save_manifest()
            except Exception as e:
                logger.error(f"Failed to save artifact manifest: {e}")
        
        # Save run metadata
        try:
            self._save_run_metadata()
        except Exception as e:
            logger.error(f"Failed to save run metadata: {e}")
        
        # Export metrics if collector is available
        if self.metrics_collector:
            try:
                metrics_json = self.metrics_collector.export_metrics(format="json")
                metrics_file = self.run_dir / "metrics.json"
                with open(metrics_file, "w") as f:
                    f.write(metrics_json)
                logger.info(f"Metrics exported: {metrics_file}")
            except Exception as e:
                logger.error(f"Failed to export metrics: {e}")
        
        # Export traces
        try:
            self._export_traces()
        except Exception as e:
            logger.error(f"Failed to export traces: {e}")
        
        # Create submission zip
        try:
            zip_path = self._create_submission_zip()
            print(f"\n{'='*60}")
            print(f"✅ Submission file created: {zip_path}")
            print(f"{'='*60}")
            print(f"📦 Please submit this file for your experiment.")
            print(f"   File: {zip_path.name}")
            print(f"   Size: {zip_path.stat().st_size / 1024:.2f} KB")
            print(f"{'='*60}\n")
        except Exception as e:
            logger.error(f"Failed to create submission zip: {e}")
    
    def _capture_environment_info(self) -> Dict[str, Any]:
        """Capture environment information."""
        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "platform_system": platform.system(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        }
    
    def _save_run_metadata(self) -> None:
        """Save run metadata to JSON file."""
        import json
        
        metadata = {
            "run_id": self.run_id,
            "participant_id": self.participant_id,
            "condition": self.condition,
            "task_id": self.task_id,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else None,
            "environment": self.environment_info,
            "run_dir": str(self.run_dir),
            "events_file": str(self.events_file),
            "artifacts_dir": str(self.artifacts_dir),
        }
        
        metadata_file = self.run_dir / "run_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Run metadata saved: {metadata_file}")
    
    def _export_traces(self) -> None:
        """Export traces from global trace collector."""
        try:
            from monitoring.tracing.tracer import trace_collector
            
            if trace_collector.events:
                traces_file = self.run_dir / "traces.json"
                import json
                traces_data = {
                    "total_events": len(trace_collector.events),
                    "stats": trace_collector.get_stats(),
                    "events": [event.to_dict() for event in trace_collector.events]
                }
                with open(traces_file, "w") as f:
                    json.dump(traces_data, f, indent=2)
                logger.info(f"Traces exported: {traces_file}")
        except Exception as e:
            logger.warning(f"Could not export traces: {e}")
    
    def _create_submission_zip(self) -> Path:
        """Create submission zip file."""
        # Zip filename: {participant_id}_{condition}_{task_id}_{run_id}.zip
        zip_name = f"{self.participant_id}_{self.condition}_{self.task_id}_{self.run_id}.zip"
        
        zip_path = self.run_dir / zip_name
        
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            # Add all files in run directory
            for file_path in self.run_dir.rglob("*"):
                if file_path.is_file() and file_path != zip_path:
                    arcname = file_path.relative_to(self.run_dir)
                    zipf.write(file_path, arcname=arcname)
        
        logger.info(f"Submission zip created: {zip_path}")
        return zip_path
    
    def set_metrics_collector(self, collector: Any) -> None:
        """Set metrics collector for export."""
        self.metrics_collector = collector
    
    def get_event_logger(self) -> Optional[EventLogger]:
        """Get event logger instance."""
        return self.event_logger
    
    def get_artifact_manager(self) -> Optional[ArtifactManager]:
        """Get artifact manager instance."""
        return self.artifact_manager

