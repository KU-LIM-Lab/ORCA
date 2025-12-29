"""
Integration tests for experiment tracking system.

Tests the complete experiment tracking workflow including:
- RunContext creation and cleanup
- Event logging
- Artifact saving
- ZIP generation
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import json
from monitoring.experiment import RunContext, EventLogger, ArtifactManager
from monitoring.metrics.collector import MetricsCollector, set_metrics_collector


class TestEventLogger:
    """Test EventLogger functionality."""
    
    def test_event_logger_creation(self, tmp_path):
        """Test that EventLogger creates events file correctly."""
        events_file = tmp_path / "events.jsonl"
        
        logger = EventLogger(
            events_file=events_file,
            run_id="test_run_123",
            participant_id="P001",
            condition="orca",
            task_id="test_task"
        )
        
        assert events_file.parent.exists()
    
    def test_log_session_events(self, tmp_path):
        """Test logging session start and end events."""
        events_file = tmp_path / "events.jsonl"
        
        logger = EventLogger(
            events_file=events_file,
            run_id="test_run_123",
            participant_id="P001",
            condition="orca"
        )
        
        logger.log_session_start(metadata={"test": "data"})
        logger.log_session_end(termination_reason="completed")
        
        # Read events
        events = []
        with open(events_file) as f:
            for line in f:
                events.append(json.loads(line))
        
        assert len(events) == 2
        assert events[0]["event_type"] == "session_start"
        assert events[1]["event_type"] == "session_end"
        assert events[1]["data"]["termination_reason"] == "completed"
    
    def test_log_step_events(self, tmp_path):
        """Test logging step enter/exit events."""
        events_file = tmp_path / "events.jsonl"
        
        logger = EventLogger(
            events_file=events_file,
            run_id="test_run_123",
            participant_id="P001",
            condition="orca"
        )
        
        logger.log_step_enter(step_id="1", substep="data_preprocessing")
        logger.log_step_exit(step_id="1", substep="data_preprocessing", success=True, duration=10.5)
        
        # Read events
        events = []
        with open(events_file) as f:
            for line in f:
                events.append(json.loads(line))
        
        assert len(events) == 2
        assert events[0]["event_type"] == "step_enter"
        assert events[0]["step_id"] == "1"
        assert events[1]["event_type"] == "step_exit"
        assert events[1]["data"]["duration"] == 10.5
    
    def test_log_hitl_events(self, tmp_path):
        """Test logging HITL events."""
        events_file = tmp_path / "events.jsonl"
        
        logger = EventLogger(
            events_file=events_file,
            run_id="test_run_123",
            participant_id="P001",
            condition="orca"
        )
        
        logger.log_hitl_prompt_shown(step_id="2", phase="graph_evaluation")
        logger.log_hitl_decision(step_id="2", decision="approve")
        logger.log_hitl_applied(step_id="2", applied=True)
        
        # Read events
        events = []
        with open(events_file) as f:
            for line in f:
                events.append(json.loads(line))
        
        assert len(events) == 3
        assert events[0]["event_type"] == "hitl_prompt_shown"
        assert events[1]["event_type"] == "hitl_decision"
        assert events[1]["data"]["decision"] == "approve"
        assert events[2]["event_type"] == "hitl_applied"


class TestArtifactManager:
    """Test ArtifactManager functionality."""
    
    def test_artifact_manager_creation(self, tmp_path):
        """Test that ArtifactManager creates artifacts directory."""
        artifacts_dir = tmp_path / "artifacts"
        
        manager = ArtifactManager(artifacts_dir=artifacts_dir)
        
        assert artifacts_dir.exists()
    
    def test_save_sql_artifact(self, tmp_path):
        """Test saving SQL artifact."""
        artifacts_dir = tmp_path / "artifacts"
        
        manager = ArtifactManager(artifacts_dir=artifacts_dir)
        
        sql_query = "SELECT * FROM users WHERE age > 18"
        path = manager.save_artifact(
            artifact_type="sql",
            data=sql_query,
            filename="test_query.sql",
            step_id="1"
        )
        
        assert path.exists()
        assert path.read_text() == sql_query
        assert len(manager.saved_artifacts) == 1
    
    def test_save_json_artifact(self, tmp_path):
        """Test saving JSON artifact."""
        artifacts_dir = tmp_path / "artifacts"
        
        manager = ArtifactManager(artifacts_dir=artifacts_dir)
        
        data = {"test": "data", "value": 123}
        path = manager.save_artifact(
            artifact_type="schema",
            data=data,
            filename="test_schema.json",
            step_id="1"
        )
        
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded == data
    
    def test_artifact_integrity(self, tmp_path):
        """Test that SHA256 hash is computed for artifacts."""
        artifacts_dir = tmp_path / "artifacts"
        
        manager = ArtifactManager(artifacts_dir=artifacts_dir)
        
        data = "test content"
        manager.save_artifact(
            artifact_type="sql",
            data=data,
            filename="test.sql"
        )
        
        artifact = manager.saved_artifacts[0]
        assert "sha256" in artifact
        assert len(artifact["sha256"]) == 64  # SHA256 hash length


class TestRunContext:
    """Test RunContext functionality."""
    
    def test_run_context_creation(self, tmp_path):
        """Test that RunContext creates directory structure."""
        with RunContext(
            participant_id="P001",
            condition="orca",
            task_id="test_task",
            base_dir=tmp_path
        ) as ctx:
            assert ctx.run_dir.exists()
            assert ctx.artifacts_dir.exists()
            assert ctx.events_file.exists()
            assert ctx.event_logger is not None
            assert ctx.artifact_manager is not None
    
    def test_run_context_logging(self, tmp_path):
        """Test that RunContext logs session events."""
        with RunContext(
            participant_id="P001",
            condition="orca",
            task_id="test_task",
            base_dir=tmp_path
        ) as ctx:
            # Session start should be logged automatically
            pass
        
        # Check events file
        events = []
        for events_file in tmp_path.rglob("events.jsonl"):
            with open(events_file) as f:
                for line in f:
                    events.append(json.loads(line))
        
        assert len(events) >= 2  # At least start and end
        assert events[0]["event_type"] == "session_start"
        assert events[-1]["event_type"] == "session_end"
    
    def test_run_context_zip_creation(self, tmp_path):
        """Test that RunContext creates submission ZIP."""
        with RunContext(
            participant_id="P001",
            condition="orca",
            task_id="test_task",
            base_dir=tmp_path
        ) as ctx:
            # Add some artifacts
            ctx.artifact_manager.save_artifact(
                artifact_type="sql",
                data="SELECT * FROM test",
                step_id="1"
            )
        
        # Check for ZIP file
        zip_files = list(tmp_path.rglob("*.zip"))
        assert len(zip_files) == 1
        assert zip_files[0].name.startswith("P001_orca_test_task_")
    
    def test_run_context_with_metrics(self, tmp_path):
        """Test RunContext with MetricsCollector integration."""
        with RunContext(
            participant_id="P001",
            condition="orca",
            task_id="test_task",
            base_dir=tmp_path
        ) as ctx:
            # Set up metrics collector
            collector = MetricsCollector(
                session_id="test_session",
                event_logger=ctx.get_event_logger(),
                artifact_manager=ctx.get_artifact_manager()
            )
            set_metrics_collector(collector)
            ctx.set_metrics_collector(collector)
            
            # Record some metrics
            collector.record_execution_time("test_agent", 1.5)
            collector.record_token_count("test_agent", 100)
        
        # Check metrics file
        metrics_files = list(tmp_path.rglob("metrics.json"))
        assert len(metrics_files) == 1


class TestIntegration:
    """Integration tests for the complete workflow."""
    
    def test_complete_workflow(self, tmp_path):
        """Test complete experiment tracking workflow."""
        with RunContext(
            participant_id="P001",
            condition="orca",
            task_id="test_task",
            base_dir=tmp_path
        ) as ctx:
            event_logger = ctx.get_event_logger()
            artifact_manager = ctx.get_artifact_manager()
            
            # Simulate Step 1: Data Wrangling
            event_logger.log_step_enter(step_id="1", substep="data_preprocessing")
            
            # Simulate SQL execution
            event_logger.log_tool_call_start(tool_name="sql_query", step_id="1")
            artifact_manager.save_artifact(
                artifact_type="sql",
                data="SELECT * FROM users",
                step_id="1"
            )
            event_logger.log_tool_call_end(
                tool_name="sql_query",
                duration=0.5,
                success=True,
                step_id="1"
            )
            
            event_logger.log_step_exit(step_id="1", substep="data_preprocessing", success=True, duration=2.0)
            
            # Simulate Step 2: Causal Discovery
            event_logger.log_step_enter(step_id="2", substep="run_algorithms_portfolio")
            
            # Simulate algorithm execution
            event_logger.log_tool_call_start(tool_name="algorithm_pc", step_id="2")
            event_logger.log_tool_call_end(
                tool_name="algorithm_pc",
                duration=5.0,
                success=True,
                step_id="2"
            )
            
            # Simulate HITL
            event_logger.log_hitl_prompt_shown(step_id="2", phase="graph_evaluation")
            event_logger.log_hitl_decision(step_id="2", decision="approve")
            
            # Save graph artifact
            artifact_manager.save_artifact(
                artifact_type="graph",
                data={"nodes": ["A", "B"], "edges": [("A", "B")]},
                step_id="2"
            )
            
            event_logger.log_step_exit(step_id="2", substep="run_algorithms_portfolio", success=True, duration=10.0)
        
        # Verify ZIP file was created
        zip_files = list(tmp_path.rglob("*.zip"))
        assert len(zip_files) == 1
        
        # Verify events were logged
        events_files = list(tmp_path.rglob("events.jsonl"))
        assert len(events_files) == 1
        
        with open(events_files[0]) as f:
            events = [json.loads(line) for line in f]
        
        # Check key events
        event_types = [e["event_type"] for e in events]
        assert "session_start" in event_types
        assert "step_enter" in event_types
        assert "step_exit" in event_types
        assert "tool_call_start" in event_types
        assert "tool_call_end" in event_types
        assert "hitl_prompt_shown" in event_types
        assert "hitl_decision" in event_types
        assert "artifact_saved" in event_types
        assert "session_end" in event_types


@pytest.fixture
def tmp_path():
    """Create temporary directory for tests."""
    path = Path(tempfile.mkdtemp())
    yield path
    shutil.rmtree(path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

