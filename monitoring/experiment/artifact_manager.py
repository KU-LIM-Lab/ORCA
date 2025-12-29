"""
Artifact manager for experiment tracking.

Handles standardized artifact storage with automatic path management,
type-specific handlers, and integrity verification.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
import pandas as pd

logger = logging.getLogger(__name__)


class ArtifactManager:
    """Manages artifact storage and integrity tracking."""
    
    def __init__(self, artifacts_dir: Path, event_logger: Optional[Any] = None):
        """
        Initialize artifact manager.
        
        Args:
            artifacts_dir: Directory for artifact storage
            event_logger: Optional EventLogger instance for logging artifact_saved events
        """
        self.artifacts_dir = artifacts_dir
        self.event_logger = event_logger
        
        # Create artifacts directory
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Track saved artifacts
        self.saved_artifacts = []
        
        logger.info(f"ArtifactManager initialized: {self.artifacts_dir}")
    
    def save_artifact(
        self,
        artifact_type: str,
        data: Any,
        filename: Optional[str] = None,
        step_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save artifact with type-specific handling.
        
        Args:
            artifact_type: Type of artifact (sql, dataset, graph, ate, schema, etc.)
            data: Artifact data (format depends on type)
            filename: Optional custom filename (will auto-generate if not provided)
            step_id: Optional step identifier
            metadata: Additional metadata
        
        Returns:
            Path to saved artifact
        """
        # Generate filename if not provided
        if not filename:
            filename = self._generate_filename(artifact_type, step_id)
        
        filepath = self.artifacts_dir / filename
        
        # Type-specific saving
        try:
            if artifact_type == "sql":
                self._save_sql(filepath, data)
            elif artifact_type == "dataset":
                self._save_dataset(filepath, data)
            elif artifact_type == "schema":
                self._save_json(filepath, data)
            elif artifact_type == "graph":
                self._save_json(filepath, data)
            elif artifact_type == "graph_adj":
                self._save_csv(filepath, data)
            elif artifact_type == "ate":
                self._save_json(filepath, data)
            elif artifact_type == "estimation_spec":
                self._save_json(filepath, data)
            elif artifact_type == "hitl_edits":
                self._save_jsonl(filepath, data)
            else:
                # Default: try JSON
                self._save_json(filepath, data)
            
            # Compute SHA256 hash
            sha256 = self._compute_sha256(filepath)
            
            # Track artifact
            artifact_info = {
                "type": artifact_type,
                "path": str(filepath.relative_to(self.artifacts_dir.parent)),
                "filename": filename,
                "sha256": sha256,
                "step_id": step_id,
                "metadata": metadata or {}
            }
            self.saved_artifacts.append(artifact_info)
            
            # Log to event logger if available
            if self.event_logger:
                self.event_logger.log_artifact_saved(
                    artifact_type=artifact_type,
                    path=str(filepath.relative_to(self.artifacts_dir.parent)),
                    sha256=sha256,
                    step_id=step_id,
                    metadata=metadata
                )
            
            logger.info(f"Artifact saved: {filepath.name} (type={artifact_type}, sha256={sha256[:8]}...)")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to save artifact {artifact_type}: {e}")
            raise
    
    def _generate_filename(self, artifact_type: str, step_id: Optional[str] = None) -> str:
        """Generate filename based on artifact type and step."""
        if artifact_type == "sql":
            return f"step{step_id}_final.sql" if step_id else "final.sql"
        elif artifact_type == "dataset":
            return f"step{step_id}_dataset.parquet" if step_id else "dataset.parquet"
        elif artifact_type == "schema":
            return f"step{step_id}_schema.json" if step_id else "schema.json"
        elif artifact_type == "graph":
            return "graph_final.json"
        elif artifact_type == "graph_adj":
            return "graph_final_adj.csv"
        elif artifact_type == "ate":
            return "ate_result.json"
        elif artifact_type == "estimation_spec":
            return "estimation_spec.json"
        elif artifact_type == "hitl_edits":
            return f"step{step_id}_hitl_edits.jsonl" if step_id else "hitl_edits.jsonl"
        else:
            return f"{artifact_type}.json"
    
    def _save_sql(self, filepath: Path, data: str) -> None:
        """Save SQL query."""
        with open(filepath, "w") as f:
            f.write(data)
    
    def _save_dataset(self, filepath: Path, data: Union[pd.DataFrame, Dict]) -> None:
        """Save dataset as Parquet."""
        if isinstance(data, pd.DataFrame):
            data.to_parquet(filepath, index=False)
        elif isinstance(data, dict):
            # If dict, try to convert to DataFrame
            df = pd.DataFrame(data)
            df.to_parquet(filepath, index=False)
        else:
            raise ValueError(f"Unsupported dataset type: {type(data)}")
    
    def _save_json(self, filepath: Path, data: Any) -> None:
        """Save JSON data."""
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
    
    def _save_csv(self, filepath: Path, data: Union[pd.DataFrame, list]) -> None:
        """Save CSV data."""
        if isinstance(data, pd.DataFrame):
            data.to_csv(filepath, index=False)
        elif isinstance(data, list):
            # Assume list of lists (adjacency matrix)
            import csv
            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(data)
        else:
            raise ValueError(f"Unsupported CSV data type: {type(data)}")
    
    def _save_jsonl(self, filepath: Path, data: Union[list, dict]) -> None:
        """Save JSONL data (list of dicts or single dict)."""
        with open(filepath, "w") as f:
            if isinstance(data, list):
                for item in data:
                    f.write(json.dumps(item, default=str) + "\n")
            else:
                f.write(json.dumps(data, default=str) + "\n")
    
    def _compute_sha256(self, filepath: Path) -> str:
        """Compute SHA256 hash of file."""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def get_artifact_manifest(self) -> Dict[str, Any]:
        """Get manifest of all saved artifacts."""
        return {
            "total_artifacts": len(self.saved_artifacts),
            "artifacts": self.saved_artifacts
        }
    
    def save_manifest(self) -> None:
        """Save artifact manifest to file."""
        manifest_path = self.artifacts_dir / "manifest.json"
        manifest = self.get_artifact_manifest()
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Artifact manifest saved: {manifest_path}")

