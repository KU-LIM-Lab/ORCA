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
        """Execute the plan end-to-end. HITL is handled inside via interrupt gate or auto-approved when non-interactive."""
        plan = state.get("execution_plan", [])
        if not plan:
            return AgentResult(
                success=False,
                error="No execution plan provided",
                metadata={"executor": self.name}
            )
        
        # Initialize execution state
        self.current_step = state.get("current_execute_step", 0)
        self.execution_log = []
        self.results = {}
        max_rerun = self.config.get("max_rerun_per_substep", 1) if self.config else 1
        interactive = bool(state.get("interactive", False))
        
        llm = self.create_llm_from_config()
        if llm is None:
            return AgentResult(
                    success=False,
                    error=f"Failed to create LLM from config"
                )

        def _convert_numpy_types(obj):
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
                return {k: _convert_numpy_types(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                t = type(obj)
                return t(_convert_numpy_types(v) for v in obj)
            return obj

        # Execute steps based on current_execute_step pointer
        # This loop handles step execution and HITL gates:
        # 1. Execute step (if not already executed)
        # 2. If HITL required, trigger interrupt and return (execution pauses)
        # 3. On resume, check HITL decision and process accordingly
        while True:
            idx = state.get("current_execute_step", 0)
            if idx >= len(plan)-1:
                state["executor_completed"] = True
                break
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
                continue

            # Execute substep
            if not state.get("current_state_executed"):
                result = self._execute_step(step, state, llm)
                self.execution_log.append({
                    "phase": step["phase"],
                    "substep": step["substep"],
                    "timestamp": datetime.now().isoformat(),
                    "success": result.success,
                    "duration": result.execution_time
                })
                if not result.success:
                    return AgentResult(
                        success=False,
                        error=f"Step {step['substep']} failed: {result.error}",
                        metadata={"execution_log": self.execution_log}
                    )

                # Merge results (convert numpy types to Python natives to avoid msgpack errors)
                safe_data = _convert_numpy_types(result.data or {})
                state.update(safe_data)
                self.results[step["substep"]] = safe_data
                state["current_state_executed"] = True

                # If no HITL required or non-interactive mode, advance to next step
                if not step.get("hitl_required", False) or not interactive:
                    state.setdefault("completed_substeps", []).append(step["substep"])
                    state["current_execute_step"] = idx + 1
                    state["current_state_executed"] = False
                    continue

            # HITL not required - advance to next step
            if not step.get("hitl_required", False):
                state.setdefault("completed_substeps", []).append(step["substep"])
                state["current_execute_step"] = idx + 1
                state["current_state_executed"] = False
                continue

            # HITL gate: Process user decision
            if state.get("hitl_executed", False):
                decision = state.get("hitl_decision", "approve")
                edits = state.get("edits", {})
                feedback = state.get("feedback", "")
                
                if decision == "abort":
                    return AgentResult(
                        success=False,
                        error="Aborted by user",
                        metadata={"substep": step["substep"], "execution_log": self.execution_log}
                    )

                if decision == "edit":
                    self._apply_edits(state, edits)

                if decision == "rerun":
                    rerun_ok = False
                    for _ in range(max_rerun):
                        re = self._rerun_step(step, state, feedback, llm)
                        self.execution_log.append({
                            "phase": step["phase"],
                            "substep": f"{step['substep']}_rerun",
                            "timestamp": datetime.now().isoformat(),
                            "success": re.success,
                            "duration": re.execution_time
                        })
                        if re.success:
                            state.update(re.data or {})
                            self.results[step["substep"]] = re.data or {}
                            rerun_ok = True
                            state["current_state_executed"] = False
                            break
                    if not rerun_ok:
                        return AgentResult(
                            success=False,
                            error=f"Rerun failed at {step['substep']}",
                            metadata={"execution_log": self.execution_log}
                        )
                    
                # Decision processed - advance to next step and clear HITL flags
                state.setdefault("completed_substeps", []).append(step["substep"])
                next_step_idx = idx + 1
                state["current_execute_step"] = next_step_idx
                state["current_state_executed"] = False
                for key in ["hitl_executed", "hitl_decision", "edits", "feedback"]:
                    state.pop(key, None)
                
                # Return to persist state changes (current_execute_step increment)
                return AgentResult(
                    success=True,
                    data={
                        "current_execute_step": next_step_idx,
                        "current_state_executed": False
                    },
                    metadata={"substep": step["substep"], "hitl_processed": True}
                )
            
            # Trigger interrupt - this pauses execution and waits for user input in graph.py
            # Flow:
            # 1. interrupt() called → graph.py stream loop catches __interrupt__ event
            # 2. User provides decision → state updated with hitl_executed=True, hitl_decision, etc.
            # 3. Executor resumes → current_state_executed=True so we skip step execution
            # 4. Enter HITL gate above → process decision and advance to next step
            self._hitl_gate(step, state)
            return AgentResult(
                success=True,
                data={},
                metadata={"substep": step["substep"], "waiting_for_hitl": True}
            )
            
        return AgentResult(
            success=True,
            data={
                "execution_completed": True,
                "total_steps": len(plan),
                "execution_log": self.execution_log,
                "results": self.results
            },
            metadata={"executor": self.name}
        )
    
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

            # # Execute the agent with timeout
            # start_time = time.time()
            
            # if asyncio.iscoroutinefunction(agent.execute_async):
            #     result = asyncio.run(self._execute_with_timeout(
            #         agent.execute_async(state), timeout
            #     ))
            # else:
            #     result = self._execute_with_timeout_sync(
            #         lambda: agent.execute(state), timeout
            #     )
            
            # execution_time = time.time() - start_time
            
            # # Record metrics
            # if self.metrics_collector:
            #     self.metrics_collector.record_execution_time(
            #         f"{self.name}.{substep}", execution_time,
            #         {"agent": agent_name, "action": action}
            #     )
            
            # return AgentResult(
            #     success=result.success,
            #     data=result.data,
            #     error=result.error,
            #     execution_time=execution_time,
            #     metadata={
            #         "substep": substep,
            #         "agent": agent_name,
            #         "action": action
            #     }
            # )
            
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
                "description": "List of table names to use for analysis",
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
                "description": "Causal graph structure with 'nodes' and 'edges'",
                "example": {"nodes": ["X", "Y"], "edges": [{"from": "X", "to": "Y"}]}
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
                "description": "List of column names to include in analysis",
                "example": ["col1", "col2", "col3"]
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
            "selected_algorithms",
            "selected_graph",
            "treatment_variable",
            "outcome_variable",
            "confounders",
            "instrumental_variables",
            "target_columns",
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
        """Execute executor step"""
        if state.get("execution_paused"):
            result = self.resume_execution(state)
        else:
            result = self.execute_plan(state)
        
        # Update state with execution results
        if result.success:
            state.update(result.data or {})
            for key in ("df_preprocessed", "df_raw", "df", "gt_df"):
                state.pop(key, None)
        else:
            state["execution_status"] = ExecutionStatus.FAILED
            state["error"] = result.error or "Execution failed"
            state["error_log"] = state.get("error_log", [])
            state["error_log"].append({
                "timestamp": datetime.now().isoformat(),
                "error": result.error,
                "metadata": result.metadata
            })
        
        return state
