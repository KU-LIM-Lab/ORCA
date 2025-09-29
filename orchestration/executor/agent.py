# orchestration/executor/agent.py
from typing import Dict, Any, Optional
from core.base import OrchestratorAgent, AgentResult
from core.state import AgentState, ExecutionStatus, HITLType, add_interrupt_point, resolve_interrupt
from monitoring.metrics.collector import MetricsCollector
import asyncio
import time
from datetime import datetime

class ExecutorAgent(OrchestratorAgent):
    """Executor agent responsible for executing plans and managing agent execution"""
    
    def __init__(self, name: str = "executor", config: Optional[Dict[str, Any]] = None,
                 metrics_collector: Optional[MetricsCollector] = None):
        super().__init__(name, config, metrics_collector)
        
        # Execution state
        self.current_step = 0
        self.execution_log = []
        self.results = {}
        
    def execute_plan(self, state: AgentState) -> AgentResult:
        """Execute the complete plan with HITL support"""
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
            
            # Execute each step in the plan
            for step in plan:
                # Check if HITL is required for this step
                if step.get("hitl_required", False):
                    # Create interrupt point
                    hitl_context = self._create_hitl_context(step, state)
                    state = add_interrupt_point(
                        state, 
                        step["phase"], 
                        step["substep"], 
                        step.get("hitl_type", HITLType.APPROVAL),
                        hitl_context
                    )
                    
                    # Return state for HITL interaction
                    return AgentResult(
                        success=True,
                        data=state,
                        metadata={"hitl_required": True, "step": step}
                    )
                
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
                        metadata={
                            "failed_step": step["substep"],
                            "execution_log": self.execution_log
                        }
                    )
                
                # Update state with step results
                state.update(result.data or {})
                self.results[step["substep"]] = result.data
                
                # Mark substep as completed
                if "completed_substeps" not in state:
                    state["completed_substeps"] = []
                state["completed_substeps"].append(step["substep"])
            
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
            agent = self.sub_agents.get(agent_name)
            if not agent:
                return AgentResult(
                    success=False,
                    error=f"Agent {agent_name} not found",
                    metadata={"substep": substep, "agent": agent_name}
                )
            
            # Prepare step context
            step_context = {
                "substep": substep,
                "action": action,
                "description": step.get("description", ""),
                "timeout": timeout,
                "previous_results": self.results
            }
            
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
        """Check if step dependencies are met"""
        dependencies = step.get("dependencies", [])
        
        for dep in dependencies:
            if dep not in state.get("completed_substeps", []):
                return False
        
        return True

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

    def handle_hitl_response(self, state: AgentState, user_decision: str, 
                           user_edits: Dict[str, Any] = None, user_feedback: str = None) -> AgentState:
        """Handle HITL response and continue execution"""
        # Resolve the interrupt
        state = resolve_interrupt(state, user_decision, user_edits, user_feedback)
        
        # Continue execution
        return self.execute_plan(state)
    
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
