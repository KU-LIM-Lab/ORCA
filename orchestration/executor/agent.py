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

        # Execute steps based on current_execute_step pointer
        while True:
            idx = state.get("current_execute_step", 0)
            if idx >= len(plan):
                # Completed all steps
                state["executor_completed"] = True
                break
            step = plan[idx]

            # Check dependencies
            # if not self._check_dependencies(step, state):
            #     return AgentResult(
            #         success=False,
            #         error=f"Dependencies not met for {step['substep']}",
            #         metadata={"executor": self.name}
            #     )

            # Execute substep (first-time execution for this step)
            if not state.get("current_state_executed"):
                state["hitl_executed"] = False
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

                # Merge results
                state.update(result.data or {})
                self.results[step["substep"]] = result.data or {}
                state["current_state_executed"] = True

                # If the step does not require HITL or we're in non-interactive mode, mark complete and advance
                if not step.get("hitl_required", False) or not interactive:
                    state.setdefault("completed_substeps", []).append(step["substep"])
                    state["current_execute_step"] = idx + 1
                    state["current_state_executed"] = False
                    continue
                # Otherwise, fall through to HITL gate logic below

            # If no HITL, mark complete and advance pointer
            if not step.get("hitl_required", False):
                state.setdefault("completed_substeps", []).append(step["substep"])
                state["current_execute_step"] = idx + 1
                state["current_state_executed"] = False
                continue

            # HITL gate
            if state.get("hitl_executed", False):
                if state.get("hitl_decision") == "abort":
                    return AgentResult(
                        success=False,
                        error="Aborted by user",
                        metadata={"substep": step["substep"], "execution_log": self.execution_log}
                    )

                if state.get("hitl_decision") == "edit":
                    self._apply_edits(state, edits)

                if state.get("hitl_decision") == "rerun":
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
                    

                # If decision is approve or unset, wait for explicit advancement via HITL
                state.setdefault("completed_substeps", []).append(step["substep"])
                state["current_execute_step"] = idx + 1
                state["current_state_executed"] = False
                continue
            
            # First time HITL trigger for this step
            decision, edits, feedback = self._hitl_gate(step, state)
            # After HITL decision, loop will re-evaluate this step and advance accordingly
            
        # Finalize result after completing or erroring out earlier
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
            # Check dependencies
            # if not self._check_dependencies(step, state):
            #     return AgentResult(
            #         success=False,
            #         error=f"Dependencies not met for step {substep}",
            #         metadata={"substep": substep}
            #     )

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
                    from agents.causal_analysis import generate_causal_analysis_graph
                    # Use a simple LLM callable if available, otherwise None
                    try:
                        # 우선 놔두기는 했는데 이렇게 해서 안되면 위의 data analysis agent 참고
                        from utils.llm import call_llm as llm
                    except Exception:
                        llm = None
                    compiled = generate_causal_analysis_graph(llm)
                    # Invoke subgraph with current state
                    new_state = compiled.invoke(state)
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


    def _hitl_gate(self, step: Dict[str, Any], state: AgentState) -> Tuple[str, Dict[str, Any], str]:
        """HITL interrupt: returns (decision, edits, feedback)."""
        payload = {
            "question": "Choose: approve / edit / rerun / abort",
            "phase": step["phase"],
            "substep": step["substep"],
            "agent": step["agent"],
            "action": step["action"],
            "description": step.get("description", ""),
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
