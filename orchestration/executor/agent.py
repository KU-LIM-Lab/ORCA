# orchestration/executor/agent.py
from typing import Dict, Any, Optional, Tuple
from core.base import OrchestratorAgent, AgentResult
from core.state import AgentState, ExecutionStatus
from monitoring.metrics.collector import MetricsCollector
import asyncio
import time
import logging
from datetime import datetime
from langgraph.types import interrupt
from utils.llm import get_llm
from utils.settings import CONFIG

logger = logging.getLogger(__name__)

class ExecutorAgent(OrchestratorAgent):
    """Executor agent responsible for executing plans and managing agent execution"""
    
    def __init__(self, name: str = "executor", config: Optional[Dict[str, Any]] = None,
                 metrics_collector: Optional[MetricsCollector] = None):
        super().__init__(name, config, metrics_collector)
        
        # Execution state
        self.current_step = 0
        self.execution_log = []
        self.results = {}

    @staticmethod
    def create_llm_from_config():
        """Create LLM instance from _config.yml settings."""
        llm_config = CONFIG.get("llm", {})
        provider = llm_config.get("provider", "openai")
        model = llm_config.get("model", "gpt-4o-mini")
        temperature = llm_config.get("temperature", 0.3)
        
        try:
            return get_llm(model=model, temperature=temperature, provider=provider)
        except Exception as e:
            print(f"[TEST] Failed to create LLM from config: {e}")
            return None
        
    def execute_plan(self, state: AgentState) -> AgentResult:
        """Execute the CURRENT step from execution plan (single step per call).
        
        This method executes ONE step at a time, allowing LangGraph to properly
        persist state between steps and handle HITL interrupts correctly.
        """
        plan = state.get("execution_plan", [])
        if not plan:
            return AgentResult(
                success=False,
                error="No execution plan provided",
                metadata={"executor": self.name}
            )
        
        # Initialize execution state
        self.execution_log = state.get("execution_log", [])
        self.results = state.get("results", {})
        
        # Execute ONLY the current step, not a while loop
        idx = state.get("current_execute_step", 0)
        
        
        # Check if all steps completed
        if idx >= len(plan):
            state["executor_completed"] = True
            state["execution_log"] = self.execution_log
            state["results"] = self.results
            return AgentResult(
                success=True,
                data=state,
                metadata={"executor": self.name, "all_completed": True}
            )
        
        step = plan[idx]
        
        # Skip step if in skip list
        skip_list = state.get("skip_steps", []) or []
        if step.get("substep") in skip_list:
            state.setdefault("completed_substeps", []).append(step["substep"])
            state["current_execute_step"] = idx + 1
            state["current_state_executed"] = False
            # Update phase status
            phase = step.get("phase")
            phase_status_map = {
                "data_exploration": "data_exploration_status",
                "causal_discovery": "causal_discovery_status",
                "causal_analysis": "causal_analysis_status"
            }
            if phase in phase_status_map:
                state[phase_status_map[phase]] = state.get(phase_status_map[phase], "skipped")
            return AgentResult(success=True, data=state)
        
        # Update current context
        state["current_substep"] = step["substep"]
        state["current_phase"] = step.get("phase", "")
        
        # Execute ONLY if not already executed
        if not state.get("current_state_executed"):
            llm = self.create_llm_from_config()
            if llm is None:
                return AgentResult(
                    success=False,
                    error=f"Failed to create LLM from config"
                )
            
            result = self._execute_step(step, state, llm)
            
            self.execution_log.append({
                "phase": step["phase"],
                "substep": step["substep"],
                "timestamp": datetime.now().isoformat(),
                "success": result.success,
                "duration": result.execution_time
            })
            
            if not result.success:
                state["error"] = f"Step {step['substep']} failed: {result.error}"
                state["execution_log"] = self.execution_log
                state["results"] = self.results
                return AgentResult(
                    success=False,
                    error=result.error,
                    data=state,
                    metadata={"execution_log": self.execution_log}
                )
            
            # Merge results (convert numpy types to avoid msgpack errors)
            safe_data = self._convert_numpy_types(result.data or {})
            state.update(safe_data)
            self.results[step["substep"]] = safe_data
            state["current_state_executed"] = True
            
            # ✅ Advance step counter BEFORE returning
            # This ensures it's persisted when the node returns
            state.setdefault("completed_substeps", []).append(step["substep"])
            state["current_execute_step"] = idx + 1
            state["current_state_executed"] = False  # Reset for next step
            
            # Store logs and results
            state["execution_log"] = self.execution_log
            state["results"] = self.results
            
            # ✅ Return immediately - if HITL requested, interrupt will fire in _executor_node
            # The state with advanced counter will be persisted when node returns
            return AgentResult(
                success=True,
                data=state,
                metadata={"executor": self.name, "hitl_requested": state.get("__hitl_requested__")}
            )
        
        # If already executed, just advance
        state.setdefault("completed_substeps", []).append(step["substep"])
        state["current_execute_step"] = idx + 1
        state["current_state_executed"] = False
        state["execution_log"] = self.execution_log
        state["results"] = self.results
        return AgentResult(success=True, data=state)
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python natives to avoid msgpack serialization errors"""
        try:
            import numpy as _np
        except Exception:
            _np = None
        if _np is not None:
            if isinstance(obj, _np.bool_):
                return bool(obj)
            if isinstance(obj, _np.integer):
                return int(obj)
            if isinstance(obj, _np.floating):
                return float(obj)
            if isinstance(obj, _np.ndarray):
                return obj.tolist()
        if isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            t = type(obj)
            return t(self._convert_numpy_types(v) for v in obj)
        return obj
    
    def _execute_step(self, step: Dict[str, Any], state: AgentState, llm) -> AgentResult:
        """Execute a single step in the plan"""
        substep = step["substep"]
        agent_name = step["agent"]
        action = step["action"]
        timeout = step.get("timeout", 300)
        
        self.current_step += 1
        
        try:
            # Dispatch to specialist agents (minimal integration for causal_discovery)
            start_time = time.time()
            if agent_name == "causal_discovery":
                from agents.causal_discovery.agent import CausalDiscoveryAgent
                from utils.settings import CONFIG
                cd_config = {}
                try:
                    cd_config = (CONFIG.get("agents", {}) or {}).get("causal_discovery", {})
                except Exception:
                    cd_config = {}

                cd_agent = CausalDiscoveryAgent(config=cd_config)
                # Map substep to agent step
                state_copy = dict(state)
                state_copy["current_substep"] = substep
                new_state = cd_agent.step(state_copy)
                execution_time = time.time() - start_time
                # Report
                success = not bool(new_state.get("error"))
                return AgentResult(
                    success=success,
                    data=new_state,
                    error=new_state.get("error"),
                    execution_time=execution_time,
                    metadata={"substep": substep, "agent": agent_name, "action": action}
                )

            if agent_name == "data_explorer":
                from agents.data_explorer.agent import DataExplorerAgent
                de_agent = DataExplorerAgent(llm=llm)
                state_copy = dict(state)
                state_copy["current_substep"] = substep
                new_state = de_agent.step(state_copy)
                execution_time = time.time() - start_time
                success = not bool(new_state.get("error"))
                return AgentResult(
                    success=success,
                    data=new_state,
                    error=new_state.get("error"),
                    execution_time=execution_time,
                    metadata={"substep": substep, "agent": agent_name, "action": action}
                )

            if agent_name == "causal_analysis":
                try:
                    # Attempt to run the causal analysis LangGraph subgraph end-to-end
                    from agents.causal_analysis import CausalAnalysisAgent
                    ca_agent = CausalAnalysisAgent(llm=llm)
                    state_copy = dict(state)
                    state_copy["current_substep"] = substep
                    new_state = ca_agent.step(state_copy)
                    execution_time = time.time() - start_time
                    success = not bool(new_state.get("error"))
                    return AgentResult(
                        success=success,
                        data=new_state,
                        error=new_state.get("error"),
                        execution_time=execution_time,
                        metadata={"substep": substep, "agent": agent_name, "action": action}
                    )
                except Exception as e:
                    execution_time = time.time() - start_time
                    return AgentResult(
                        success=False,
                        error=f"Causal analysis execution failed: {str(e)}",
                        execution_time=execution_time,
                        metadata={"substep": substep, "agent": agent_name, "action": action}
                    )

            if agent_name == "report_generator":
                from agents.report_generation.agent import ReportGenerationAgent
                rg_agent = ReportGenerationAgent()
                state_copy = dict(state)
                state_copy["current_substep"] = substep
                new_state = rg_agent.step(state_copy)
                execution_time = time.time() - start_time
                success = not bool(new_state.get("error"))
                return AgentResult(
                    success=success,
                    data=new_state,
                    error=new_state.get("error"),
                    execution_time=execution_time,
                    metadata={"substep": substep, "agent": agent_name, "action": action}
                )

            # Fallback for unimplemented agents
            return AgentResult(
                success=False,
                error=f"Agent {agent_name} not implemented in current architecture",
                metadata={"substep": substep, "agent": agent_name}
            )
            
        except asyncio.TimeoutError:
            return AgentResult(
                success=False,
                error=f"Step {substep} timed out after {timeout}s",
                metadata={"substep": substep, "timeout": timeout}
            )
        except Exception as e:
            return AgentResult(
                success=False,
                error=f"Step {substep} failed with exception: {str(e)}",
                metadata={"substep": substep, "exception": str(e)}
            )
    
    async def _execute_with_timeout(self, coro, timeout: int):
        """Execute coroutine with timeout"""
        return await asyncio.wait_for(coro, timeout=timeout)
    
    def _execute_with_timeout_sync(self, func, timeout: int):
        """Execute synchronous function with timeout"""
        return asyncio.run(asyncio.wait_for(asyncio.to_thread(func), timeout=timeout))
    
    def _check_dependencies(self, step: Dict[str, Any], state: AgentState) -> bool:
        """Check if step requirements are met by verifying state values are present/non-empty."""
        def _is_present(value: Any) -> bool:
            if value is None:
                return False
            if isinstance(value, (list, dict, set, tuple)):
                return len(value) > 0
            if isinstance(value, str):
                return value.strip() != ""
            return True

        for key in step.get("required_state_keys", []):
            if not _is_present(state.get(key)):
                return False
        return True


    @staticmethod
    def _get_editable_fields() -> Dict[str, Dict[str, Any]]:
        """Get editable fields definition for HITL edit operations."""
        return {
            "selected_tables": {
                "type": "list[str]",
                "description": "List of table names selected for analysis",
                "example": ["users", "orders", "products"]
            },
            "sql_query": {
                "type": "str",
                "description": "SQL query string to fetch data",
                "example": "SELECT * FROM users WHERE age > 18"
            },
            "selected_algorithms": {
                "type": "list[str]",
                "description": "List of algorithm names to run (e.g., ['PC', 'GES', 'LiNGAM'])",
                "example": ["PC", "GES"]
            },
            "selected_graph": {
                "type": "dict",
                "description": "Final causal graph structure. You can modify edges by providing a graph dict with 'graph' key containing 'variables' (list) and 'edges' (list of dicts with 'from', 'to', 'type'). Format: {'graph': {'variables': [...], 'edges': [{'from': 'X', 'to': 'Y', 'type': '->'}]}, 'metadata': {...}}",
                "example": {
                    "graph": {
                        "variables": ["X", "Y", "Z"],
                        "edges": [
                            {"from": "X", "to": "Y", "type": "->"},
                            {"from": "Y", "to": "Z", "type": "->"}
                        ]
                    },
                    "metadata": {"graph_type": "DAG"}
                }
            },
            "treatment_variable": {
                "type": "str",
                "description": "Name of the treatment/intervention variable",
                "example": "gender"
            },
            "outcome_variable": {
                "type": "str",
                "description": "Name of the outcome variable",
                "example": "used_coupon"
            },
            "confounders": {
                "type": "list[str]",
                "description": "List of confounder variable names",
                "example": ["age", "income"]
            },
            "instrumental_variables": {
                "type": "list[str]",
                "description": "List of instrumental variable names",
                "example": ["instrument"]
            },
            "target_columns": {
                "type": "list[str]",
                "description": "Specific columns to include in analysis (optional - if not set, all columns will be used)",
                "example": ["age", "gender", "purchase_amount"]
            },
            "clean_nulls_ratio": {
                "type": "float",
                "description": "Drop columns with null ratio above this threshold (0.0-1.0)",
                "default": 0.95
            },
            "one_hot_threshold": {
                "type": "int",
                "description": "Maximum unique values for one-hot encoding categorical columns",
                "default": 20
            },
            "high_cardinality_threshold": {
                "type": "int",
                "description": "Threshold for detecting high cardinality categorical variables",
                "default": 50
            },
            "execution_plan": {
                "type": "list[dict]",
                "description": "Algorithm execution plan with selected algorithms and configurations",
                "example": [{"algorithm": "PC", "params": {}}, {"algorithm": "GES", "params": {}}]
            },
            "mediators": {
                "type": "list[str]",
                "description": "Mediator variables (on causal path between treatment and outcome)",
                "example": ["mediator_var"]
            },
            "strategy": {
                "type": "dict",
                "description": "Causal analysis strategy configuration with task, identification_method, estimator, and optional refuter. Tasks: estimating_causal_effect (ATE), mediation_analysis, causal_prediction, what_if, root_cause. Identification: backdoor, frontdoor, iv, mediation, id_algorithm. Estimators: backdoor.linear_regression, backdoor.propensity_score_matching, backdoor.generalized_linear_model, iv.instrumental_variable, etc.",
                "example": [
                    {"task": "estimating_causal_effect", "identification_method": "backdoor", "estimator": "backdoor.propensity_score_matching", "refuter": "placebo_treatment_refuter"},
                    {"task": "estimating_causal_effect", "identification_method": "backdoor", "estimator": "backdoor.linear_regression", "refuter": None},
                ]
            }
        }

    def _hitl_gate(self, step: Dict[str, Any], state: AgentState) -> None:
        """HITL interrupt: triggers interrupt with simplified payload.
        User input will be handled interactively in graph.py and stored in state.
        """
        from langgraph.types import interrupt
        
        payload = {
            "step": step["substep"],
            "phase": step["phase"],
            "description": step.get("description", ""),
            "decisions": ["approve", "edit", "rerun", "abort"]
        }
        
        interrupt(payload)

    def _apply_edits(self, state: AgentState, edits: Dict[str, Any]) -> None:
        """Apply only safe, whitelisted edits to state."""
        allowed = {
            "selected_tables",
            "sql_query",
            "final_sql",
            "selected_algorithms",
            "selected_graph",
            "treatment_variable",
            "outcome_variable",
            "confounders",
            "instrumental_variables",
            "target_columns",
            "clean_nulls_ratio",
            "one_hot_threshold",
            "high_cardinality_threshold",
            "execution_plan",
            "mediators",
            "strategy"
        }
        for k, v in (edits or {}).items():
            if k in allowed:
                state[k] = v

    def _rerun_step(self, step: Dict[str, Any], state: AgentState, feedback: str, llm) -> AgentResult:
        """Re-run the same substep; store feedback for agent consumption."""
        if feedback:
            state["user_feedback"] = feedback
        return self._execute_step(step, state, llm)

    def get_execution_status(self) -> Dict[str, Any]:
        """Get current execution status"""
        return {
            "current_step": self.current_step,
            "total_steps": len(self.execution_log),
            "completed_steps": list(self.results.keys()),
            "execution_log": self.execution_log,
            "status": "running" if self.current_step < len(self.execution_log) else "completed"
        }
    
    def step(self, state: AgentState) -> AgentState:
        """Execute executor step - returns the complete updated state"""
        if state.get("execution_paused"):
            result = self.resume_execution(state)
        else:
            result = self.execute_plan(state)
        
        if result.success:
            updated_state = result.data or state
                        
            # Preserve executor action for graph.py to detect HITL needs
            if result.metadata and result.metadata.get("action"):
                updated_state["__executor_action__"] = result.metadata.get("action")
            
            # Remove DataFrame keys to avoid serialization issues
            for key in ("df_preprocessed", "df_raw", "df", "gt_df"):
                updated_state.pop(key, None)
            
            return updated_state
        else:
            # On failure, update state with error info
            state["execution_status"] = ExecutionStatus.FAILED
            state["error"] = result.error or "Execution failed"
            state["error_log"] = state.get("error_log", [])
            state["error_log"].append({
                "timestamp": datetime.now().isoformat(),
                "error": result.error,
                "metadata": result.metadata
            })
            return state
