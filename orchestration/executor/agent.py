# orchestration/executor/agent.py
from typing import Dict, Any, Optional, Tuple
from core.base import OrchestratorAgent, AgentResult
from core.state import AgentState, ExecutionStatus
from monitoring.metrics.collector import MetricsCollector
import asyncio
import time
from datetime import datetime
from langgraph.types import interrupt

class ExecutorAgent(OrchestratorAgent):
    """Executor agent responsible for executing plans and managing agent execution"""
    
    def __init__(self, name: str = "executor", config: Optional[Dict[str, Any]] = None,
                 metrics_collector: Optional[MetricsCollector] = None):
        super().__init__(name, config, metrics_collector)
        
        # Execution state
        self.current_step = 0
        self.execution_log = []
        self.results = {}
        
        # Initialize system agents
        self._initialize_system_agents()
    
    def _initialize_system_agents(self):
        """Initialize and register system agents"""
        try:
            from utils.system_agents import create_system_agents
            
            # Create system agents
            db_agent, metadata_agent = create_system_agents(
                db_id="reef_db", 
                db_type="postgresql"
            )
            
            # Keep system agents locally (they are not BaseAgent instances)
            self.system_agents: Dict[str, Any] = {
                "system_database": db_agent,
                "system_metadata": metadata_agent,
            }
            
        except Exception as e:
            print(f"Warning: Could not initialize system agents: {e}")
        
    def execute_plan(self, state: AgentState) -> AgentResult:
        """Execute the plan. HITL is handled inside via interrupt gate."""
        plan = state.get("execution_plan", [])
        if not plan:
            return AgentResult(
                success=False,
                error="No execution plan provided",
                metadata={"executor": self.name}
            )
        
        try:
            # Initialize execution state
            self.current_step = 0
            self.execution_log = []
            self.results = {}
            max_rerun = self.config.get("max_rerun_per_substep", 1) if self.config else 1
            
            # Execute each step in the plan
            for step in plan:
                # Check dependencies
                if not self._check_dependencies(step, state):
                    return AgentResult(
                        success=False,
                        error=f"Dependencies not met for {step['substep']}",
                        metadata={"executor": self.name}
                    )

                # Execute substep
                result = self._execute_step(step, state)
                self.execution_log.append({
                    "phase": step["phase"].value,
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
                
                # Merge results
                state.update(result.data or {})
                self.results[step["substep"]] = result.data or {}

                # If no HITL, mark complete
                if not step.get("hitl_required", False):
                    state.setdefault("completed_substeps", []).append(step["substep"])
                    continue

                # HITL gate
                decision, edits, feedback = self._hitl_gate(step, state, self.results[step["substep"]])

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
                        re = self._rerun_step(step, state, feedback)
                        self.execution_log.append({
                            "phase": step["phase"].value,
                            "substep": f"{step['substep']}_rerun",
                            "timestamp": datetime.now().isoformat(),
                            "success": re.success,
                            "duration": re.execution_time
                        })
                        if re.success:
                            state.update(re.data or {})
                            self.results[step["substep"]] = re.data or {}
                            rerun_ok = True
                            break
                    if not rerun_ok:
                        return AgentResult(
                            success=False,
                            error=f"Rerun failed at {step['substep']}",
                            metadata={"execution_log": self.execution_log}
                        )

                # After approve/edit success
                state.setdefault("completed_substeps", []).append(step["substep"])
            
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
            
        except Exception as e:
            return AgentResult(
                success=False,
                error=f"Execution failed: {str(e)}",
                metadata={
                    "execution_log": self.execution_log,
                    "current_step": self.current_step
                }
            )
    
    def _execute_step(self, step: Dict[str, Any], state: AgentState) -> AgentResult:
        """Execute a single step in the plan"""
        substep = step["substep"]
        agent_name = step["agent"]
        action = step["action"]
        timeout = step.get("timeout", 300)
        
        self.current_step += 1
        
        try:
            # Check dependencies
            if not self._check_dependencies(step, state):
                return AgentResult(
                    success=False,
                    error=f"Dependencies not met for step {substep}",
                    metadata={"substep": substep}
                )
            
            # Get the agent
            agent = None
            if step.get("is_system_component", False):
                agent = getattr(self, "system_agents", {}).get(agent_name)
            else:
                agent = self.sub_agents.get(agent_name)
            if not agent:
                return AgentResult(
                    success=False,
                    error=f"Agent {agent_name} not found",
                    metadata={"substep": substep, "agent": agent_name}
                )
            
            # Handle system components explicitly (allow pre-initialized state to skip)
            if step.get("is_system_component", False):
                if substep == "connect_database":
                    # Skip if already connected (pre-initialized outside the graph)
                    if state.get("database_connection"):
                        return AgentResult(success=True, data={})
                    ok = getattr(agent, "connect", lambda: False)()
                    if not ok:
                        return AgentResult(success=False, error="Database connection failed")
                    conn = getattr(agent, "get_connection", lambda: None)()
                    return AgentResult(success=True, data={"database_connection": conn})
                elif substep == "create_metadata":
                    # Skip if metadata already present (pre-initialized outside the graph)
                    if state.get("schema_info") and state.get("table_metadata"):
                        return AgentResult(success=True, data={})
                    ok = getattr(agent, "initialize", lambda: False)()
                    if not ok:
                        return AgentResult(success=False, error="Metadata initialization failed")
                    data = {
                        "schema_info": getattr(agent, "get_schema_info", lambda: {})(),
                        "table_metadata": getattr(agent, "get_table_metadata", lambda: {})(),
                        "table_relations": getattr(agent, "get_table_relations", lambda: {})(),
                    }
                    return AgentResult(success=True, data=data)

            # Execute the agent with timeout
            start_time = time.time()
            
            if asyncio.iscoroutinefunction(agent.execute_async):
                result = asyncio.run(self._execute_with_timeout(
                    agent.execute_async(state), timeout
                ))
            else:
                result = self._execute_with_timeout_sync(
                    lambda: agent.execute(state), timeout
                )
            
            execution_time = time.time() - start_time
            
            # Record metrics
            if self.metrics_collector:
                self.metrics_collector.record_execution_time(
                    f"{self.name}.{substep}", execution_time,
                    {"agent": agent_name, "action": action}
                )
            
            return AgentResult(
                success=result.success,
                data=result.data,
                error=result.error,
                execution_time=execution_time,
                metadata={
                    "substep": substep,
                    "agent": agent_name,
                    "action": action
                }
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
        return asyncio.run(asyncio.wait_for(
            asyncio.to_thread(func), timeout=timeout
        ))
    
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

    def _hitl_gate(self, step: Dict[str, Any], state: AgentState, result_data: Dict[str, Any]) -> Tuple[str, Dict[str, Any], str]:
        """HITL interrupt: returns (decision, edits, feedback)."""
        payload = {
            "question": "Choose: approve / edit / rerun / abort",
            "phase": step["phase"].value,
            "substep": step["substep"],
            "agent": step["agent"],
            "action": step["action"],
            "description": step.get("description", ""),
            "step_result_keys": list((result_data or {}).keys())[:50],
            "choices": ["approve", "edit", "rerun", "abort"]
        }
        decision_payload = interrupt(payload)
        d = decision_payload.get("decision", "approve")
        edits = decision_payload.get("edits", {}) if d == "edit" else {}
        feedback = decision_payload.get("feedback", "") if d == "rerun" else ""
        return d, edits, feedback

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
            "filters",
            "time_window",
            "target_columns",
        }
        for k, v in (edits or {}).items():
            if k in allowed:
                state[k] = v

    def _rerun_step(self, step: Dict[str, Any], state: AgentState, feedback: str) -> AgentResult:
        """Re-run the same substep; store feedback for agent consumption."""
        if feedback:
            state["user_feedback"] = feedback
        return self._execute_step(step, state)

    def _create_hitl_context(self, step: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
        """Create context for HITL interaction"""
        return {
            "phase": step["phase"].value,
            "substep": step["substep"],
            "agent": step["agent"],
            "action": step["action"],
            "description": step.get("description", ""),
            "current_state": state,
            "timestamp": datetime.now().isoformat()
        }
    
    def pause_execution(self) -> Dict[str, Any]:
        """Pause current execution"""
        return {
            "execution_paused": True,
            "current_step": self.current_step,
            "completed_steps": list(self.results.keys()),
            "execution_log": self.execution_log
        }
    
    def resume_execution(self, state: AgentState) -> AgentResult:
        """Resume paused execution"""
        plan = state.get("execution_plan", [])
        remaining_steps = plan[self.current_step:]
        
        if not remaining_steps:
            return AgentResult(
                success=True,
                data={"execution_completed": True, "resumed": True}
            )
        
        # Continue with remaining steps
        for step in remaining_steps:
            result = self._execute_step(step, state)
            self.execution_log.append({
                "step_id": step["step_id"],
                "timestamp": datetime.now().isoformat(),
                "success": result.success,
                "duration": result.execution_time
            })
            
            if not result.success:
                return AgentResult(
                    success=False,
                    error=f"Resumed execution failed at step {step['step_id']}: {result.error}"
                )
            
            state.update(result.data or {})
            self.results[step["step_id"]] = result.data
        
        return AgentResult(
            success=True,
            data={
                "execution_completed": True,
                "resumed": True,
                "execution_log": self.execution_log,
                "results": self.results
            }
        )
    
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
        else:
            state["execution_status"] = ExecutionStatus.FAILED
            state["error_log"] = state.get("error_log", [])
            state["error_log"].append({
                "timestamp": datetime.now().isoformat(),
                "error": result.error,
                "metadata": result.metadata
            })
        
        return state
