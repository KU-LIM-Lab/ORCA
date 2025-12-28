"""
Baseline Tools

Functional tools for the baseline single-GPT agent.
These tools provide access to database schema, SQL execution, Python code execution,
and artifact management.
"""

import logging
import time
import json
import hashlib
from typing import Dict, Any, List, Optional
import pandas as pd

from utils.redis_client import redis_client
from utils.database import Database
from shared.code_runner import CodeRunner

logger = logging.getLogger(__name__)

# Global context for sharing data between tools
_tool_context = {
    "dataframes": {},  # Store DataFrames by name for Python code execution
    "last_sql_result": None,
    "event_logger": None,
    "artifact_manager": None,
    "current_step": "1"
}


def set_tool_context(event_logger=None, artifact_manager=None):
    """Set global tool context for event logging and artifact management."""
    _tool_context["event_logger"] = event_logger
    _tool_context["artifact_manager"] = artifact_manager


def get_schema(db_id: str) -> Dict[str, Any]:
    """
    Retrieve database schema including tables, columns, types, and relationships.
    
    Args:
        db_id: Database identifier
        
    Returns:
        Dictionary with schema information including:
        - tables: Dict of table names to column info
        - relationships: Foreign key relationships
        - table_count: Number of tables
    """
    event_logger = _tool_context.get("event_logger")
    start_time = time.time()
    
    # Log tool call start
    if event_logger:
        event_logger.log_tool_call_start(
            tool_name="get_schema",
            step_id=_tool_context.get("current_step"),
            metadata={"db_id": db_id}
        )
    
    try:
        # Try to get schema from Redis metadata
        metadata_key = f"{db_id}:metadata"
        metadata_raw = redis_client.get(metadata_key)
        
        if metadata_raw:
            metadata = json.loads(metadata_raw)
            schema_info = metadata.get("schema_info", {})
            table_relations = metadata.get("table_relations", {})
            
            result = {
                "success": True,
                "tables": schema_info.get("tables", {}),
                "relationships": table_relations.get("edges", {}),
                "table_count": len(schema_info.get("tables", {})),
                "db_id": db_id
            }
        else:
            # Fallback: try to get table_relations directly
            relations_key = f"{db_id}:table_relations"
            relations_raw = redis_client.get(relations_key)
            
            if relations_raw:
                relations_info = json.loads(relations_raw)
                source_schema = relations_info.get("source_schema", {})
                
                result = {
                    "success": True,
                    "tables": source_schema,
                    "relationships": relations_info.get("edges", {}),
                    "table_count": len(source_schema),
                    "db_id": db_id
                }
            else:
                result = {
                    "success": False,
                    "error": f"No metadata found for database '{db_id}'. Run metadata generation first.",
                    "db_id": db_id
                }
        
        # Log tool call end
        duration = time.time() - start_time
        if event_logger:
            event_logger.log_tool_call_end(
                tool_name="get_schema",
                duration=duration,
                success=result["success"],
                error=result.get("error"),
                step_id=_tool_context.get("current_step"),
                metadata={
                    "db_id": db_id,
                    "table_count": result.get("table_count", 0)
                }
            )
        
        return result
        
    except Exception as e:
        duration = time.time() - start_time
        error_msg = f"Failed to retrieve schema: {str(e)}"
        logger.exception(error_msg)
        
        if event_logger:
            event_logger.log_tool_call_end(
                tool_name="get_schema",
                duration=duration,
                success=False,
                error=error_msg,
                step_id=_tool_context.get("current_step"),
                metadata={"db_id": db_id}
            )
        
        return {
            "success": False,
            "error": error_msg,
            "db_id": db_id
        }


def run_sql(db_id: str, sql: str) -> Dict[str, Any]:
    """
    Execute SQL query and return results.
    
    Args:
        db_id: Database identifier
        sql: SQL query to execute
        
    Returns:
        Dictionary with:
        - success: Whether query succeeded
        - rows: List of result rows (limited to 100 for preview)
        - columns: List of column names
        - row_count: Total number of rows returned
        - preview: String preview of first few rows
        - error: Error message if failed
    """
    event_logger = _tool_context.get("event_logger")
    artifact_manager = _tool_context.get("artifact_manager")
    start_time = time.time()
    
    sql_hash = hashlib.md5(sql.encode()).hexdigest()[:8]
    
    # Log tool call start
    if event_logger:
        event_logger.log_tool_call_start(
            tool_name="run_sql",
            step_id=_tool_context.get("current_step"),
            metadata={
                "sql_hash": sql_hash,
                "sql_preview": sql[:200] if len(sql) > 200 else sql,
                "db_id": db_id
            }
        )
    
    try:
        # Execute SQL
        database = Database()
        rows, column_names = database.run_query(sql, db_id)
        
        # Create DataFrame and store in context
        if rows and column_names:
            df = pd.DataFrame(rows, columns=column_names)
            _tool_context["dataframes"]["last_result"] = df
            _tool_context["last_sql_result"] = {
                "df": df,
                "sql": sql,
                "columns": column_names,
                "row_count": len(rows)
            }
            
            # Generate preview
            preview = df.head(10).to_string()
        else:
            df = None
            preview = "No rows returned"
        
        result = {
            "success": True,
            "rows": rows[:100],  # Limit for response size
            "columns": column_names,
            "row_count": len(rows),
            "preview": preview,
            "sql_hash": sql_hash
        }
        
        # Log tool call end
        duration = time.time() - start_time
        if event_logger:
            event_logger.log_tool_call_end(
                tool_name="run_sql",
                duration=duration,
                success=True,
                step_id=_tool_context.get("current_step"),
                metadata={
                    "sql_hash": sql_hash,
                    "row_count": len(rows),
                    "column_count": len(column_names),
                    "db_id": db_id
                }
            )
                
        return result
        
    except Exception as e:
        duration = time.time() - start_time
        error_msg = f"SQL execution failed: {str(e)}"
        logger.exception(error_msg)
        
        if event_logger:
            event_logger.log_tool_call_end(
                tool_name="run_sql",
                duration=duration,
                success=False,
                error=error_msg,
                step_id=_tool_context.get("current_step"),
                metadata={"sql_hash": sql_hash, "db_id": db_id}
            )
        
        return {
            "success": False,
            "error": error_msg,
            "sql": sql[:100] + "..." if len(sql) > 100 else sql
        }


def run_python(code: str, context_vars: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Execute Python code safely with access to dataframes from previous SQL queries.
    
    Args:
        code: Python code to execute
        context_vars: Optional additional variables to provide (not typically needed)
        
    Returns:
        Dictionary with:
        - success: Whether execution succeeded
        - outputs: Dictionary of output variables
        - stdout: Printed output
        - error: Error message if failed
    """
    event_logger = _tool_context.get("event_logger")
    start_time = time.time()
    
    code_hash = hashlib.md5(code.encode()).hexdigest()[:8]
    
    # Log tool call start
    if event_logger:
        event_logger.log_tool_call_start(
            tool_name="run_python",
            step_id=_tool_context.get("current_step"),
            metadata={
                "code_hash": code_hash,
                "code_preview": code[:200] if len(code) > 200 else code
            }
        )
    
    try:
        # Prepare execution context
        exec_context = {}
        
        # Add dataframes from tool context
        if _tool_context.get("dataframes"):
            exec_context.update(_tool_context["dataframes"])
        
        # Add last SQL result as 'df' for convenience
        if _tool_context.get("last_sql_result"):
            exec_context["df"] = _tool_context["last_sql_result"]["df"]
        
        # Add any additional context vars
        if context_vars:
            exec_context.update(context_vars)
        
        # Execute code
        runner = CodeRunner()
        globals_after, locals_after, stdout = runner.execute(
            code, 
            globals_dict=exec_context,
            locals_dict={}
        )
        
        # Extract output variables (exclude built-ins and modules)
        outputs = {}
        for key, value in locals_after.items():
            if not key.startswith('_'):
                # Store DataFrames in context for future use
                if isinstance(value, pd.DataFrame):
                    _tool_context["dataframes"][key] = value
                    outputs[key] = f"<DataFrame: shape={value.shape}>"
                elif isinstance(value, (int, float, str, bool, list, dict)):
                    outputs[key] = value
        
        result = {
            "success": True,
            "outputs": outputs,
            "stdout": stdout,
            "code_hash": code_hash
        }
        
        # Log tool call end
        duration = time.time() - start_time
        if event_logger:
            event_logger.log_tool_call_end(
                tool_name="run_python",
                duration=duration,
                success=True,
                step_id=_tool_context.get("current_step"),
                metadata={
                    "code_hash": code_hash,
                    "output_vars": list(outputs.keys())
                }
            )
        
        return result
        
    except Exception as e:
        duration = time.time() - start_time
        error_msg = f"Python execution failed: {str(e)}"
        logger.exception(error_msg)
        
        if event_logger:
            event_logger.log_tool_call_end(
                tool_name="run_python",
                duration=duration,
                success=False,
                error=error_msg,
                step_id=_tool_context.get("current_step"),
                metadata={"code_hash": code_hash}
            )
        
        return {
            "success": False,
            "error": error_msg,
            "stdout": "",
            "outputs": {}
        }


def save_artifact(
    artifact_type: str,
    data_ref: Optional[str] = None,
    step_id: Optional[str] = None,
    filename: Optional[str] = None
) -> Dict[str, Any]:
    """
    Save an artifact (SQL, dataset, graph, ATE result) to the artifacts directory.
    
    Args:
        artifact_type: Type of artifact ("sql", "dataset", "graph", "graph_adj", "ate", "estimation_spec")
        data_ref: Reference to data:
            - For "sql": the SQL query string itself, or "last" to use last executed SQL
            - For "dataset": name of dataframe in context, or "last" to use last SQL result
            - For "graph"/"graph_adj"/"ate": name of variable from Python execution
        step_id: Step identifier ("1", "2", "3")
        filename: Optional custom filename (will auto-generate if not provided)
        
    Returns:
        Dictionary with:
        - success: Whether save succeeded
        - path: Path to saved artifact
        - sha256: SHA256 hash of artifact
        - artifact_type: Type of artifact saved
    """
    artifact_manager = _tool_context.get("artifact_manager")
    
    if not artifact_manager:
        return {
            "success": False,
            "error": "Artifact manager not available. This tool requires experiment tracking context."
        }
    
    try:
        # Determine data to save based on artifact_type and data_ref
        if artifact_type == "sql":
            if data_ref == "last" or not data_ref:
                if _tool_context.get("last_sql_result"):
                    data = _tool_context["last_sql_result"]["sql"]
                else:
                    return {"success": False, "error": "No SQL query available to save"}
            else:
                data = data_ref  # Assume it's the SQL string itself
        
        elif artifact_type == "dataset":
            if data_ref == "last" or not data_ref:
                if _tool_context.get("last_sql_result"):
                    data = _tool_context["last_sql_result"]["df"]
                else:
                    return {"success": False, "error": "No dataset available to save"}
            else:
                # Look up dataframe by name
                if data_ref in _tool_context.get("dataframes", {}):
                    data = _tool_context["dataframes"][data_ref]
                else:
                    return {"success": False, "error": f"DataFrame '{data_ref}' not found in context"}
        
        elif artifact_type in ["graph", "graph_adj", "ate", "estimation_spec"]:
            # For these types, data_ref should be a variable name or the data itself
            if isinstance(data_ref, str) and data_ref in _tool_context.get("dataframes", {}):
                # If it's a dataframe reference (for adjacency matrix)
                data = _tool_context["dataframes"][data_ref]
            elif isinstance(data_ref, (dict, list)):
                # Direct data structure
                data = data_ref
            elif isinstance(data_ref, str):
                # Try to parse as JSON
                try:
                    data = json.loads(data_ref)
                except:
                    return {"success": False, "error": f"Could not parse data for {artifact_type}"}
            else:
                return {"success": False, "error": f"Invalid data reference for {artifact_type}"}
        
        elif artifact_type == "schema":
            # Schema data
            if isinstance(data_ref, dict):
                data = data_ref
            elif isinstance(data_ref, str):
                try:
                    data = json.loads(data_ref)
                except:
                    return {"success": False, "error": "Could not parse schema data"}
            else:
                return {"success": False, "error": "Invalid schema data"}
        
        else:
            return {"success": False, "error": f"Unknown artifact type: {artifact_type}"}
        
        # Save artifact
        filepath = artifact_manager.save_artifact(
            artifact_type=artifact_type,
            data=data,
            filename=filename,
            step_id=step_id,
            metadata={"data_ref": str(data_ref)[:100] if data_ref else None}
        )
        
        # Get SHA256 from saved artifacts list
        saved_artifacts = artifact_manager.saved_artifacts
        sha256 = saved_artifacts[-1]["sha256"] if saved_artifacts else "unknown"
        
        return {
            "success": True,
            "path": str(filepath),
            "sha256": sha256[:8] + "...",  # Abbreviated for readability
            "artifact_type": artifact_type,
            "step_id": step_id
        }
        
    except Exception as e:
        error_msg = f"Failed to save artifact: {str(e)}"
        logger.exception(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "artifact_type": artifact_type
        }


def final_answer(summary: str, all_steps_complete: bool = False) -> Dict[str, Any]:
    """
    Signal completion of the causal analysis.
    
    Args:
        summary: Summary of the analysis results
        all_steps_complete: Whether all 3 steps have been completed
        
    Returns:
        Dictionary with:
        - success: Whether analysis is complete
        - message: Completion message
        - summary: Analysis summary
    """
    return {
        "success": True,
        "message": "Analysis complete" if all_steps_complete else "Analysis incomplete",
        "summary": summary,
        "all_steps_complete": all_steps_complete
    }


# Tool registry for easy lookup
BASELINE_TOOLS = {
    "get_schema": get_schema,
    "run_sql": run_sql,
    "run_python": run_python,
    "save_artifact": save_artifact,
    "final_answer": final_answer
}
