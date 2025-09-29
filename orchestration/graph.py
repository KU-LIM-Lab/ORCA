# orchestration/graph.py
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END
from core.state import AgentState, ExecutionStatus, HITLType, create_initial_state
from orchestration.planner.agent import PlannerAgent
from orchestration.executor.agent import ExecutorAgent
from monitoring.metrics.collector import MetricsCollector

class OrchestrationGraph:
    """Main orchestration graph that coordinates planner and executor"""
    
    def __init__(self, 
                 planner_config: Optional[Dict[str, Any]] = None,
                 executor_config: Optional[Dict[str, Any]] = None,
                 metrics_collector: Optional[MetricsCollector] = None):
        self.metrics_collector = metrics_collector
        
        # Initialize agents
        self.planner = PlannerAgent(
            name="planner",
            config=planner_config,
            metrics_collector=metrics_collector
        )
        
        self.executor = ExecutorAgent(
            name="executor", 
            config=executor_config,
            metrics_collector=metrics_collector
        )
        
        # Add executor as sub-agent to planner
        self.planner.add_sub_agent(self.executor)
        
        # Build the graph
        self.graph = self._build_graph()
        self.compiled_graph = None
    
    def _build_graph(self) -> StateGraph:
        """Build the orchestration graph with HITL support"""
        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node("planner", self._planner_node)
        graph.add_node("executor", self._executor_node)
        graph.add_node("hitl_handler", self._hitl_handler_node)
        graph.add_node("error_handler", self._error_handler_node)
        graph.add_node("finalizer", self._finalizer_node)
        
        # Set entry point
        graph.set_entry_point("planner")
        
        # Add edges
        graph.add_edge("planner", "executor")
        
        # Add HITL interrupts
        graph.add_interrupt_after("planner", "hitl_handler")
        graph.add_interrupt_before("executor", "hitl_handler")
        graph.add_interrupt_after("executor", "hitl_handler")
        
        # Add conditional edges
        graph.add_conditional_edges(
            "hitl_handler",
            self._route_hitl_decision,
            {
                "approve": "executor",
                "edit": "hitl_handler",
                "rerun": "planner",
                "abort": END
            }
        )
        
        graph.add_conditional_edges(
            "executor",
            self._route_after_execution,
            {
                "success": "finalizer",
                "error": "error_handler",
                "replan": "planner",
                "hitl": "hitl_handler"
            }
        )
        
        graph.add_edge("error_handler", "planner")
        graph.add_edge("finalizer", END)
        
        return graph
    
    def _planner_node(self, state: AgentState) -> AgentState:
        """Planner node execution"""
        try:
            result = self.planner.step(state)
            state.update(result)
            state["planner_completed"] = True
            return state
        except Exception as e:
            state["error"] = str(e)
            state["error_type"] = "planner_error"
            return state

    def _hitl_handler_node(self, state: AgentState) -> AgentState:
        """HITL handler node"""
        hitl_type = state.get("hitl_type")
        hitl_context = state.get("hitl_context", {})
        
        # Handle different HITL types
        if hitl_type == HITLType.APPROVAL:
            return self._handle_approval_hitl(state, hitl_context)
        elif hitl_type == HITLType.EDIT:
            return self._handle_edit_hitl(state, hitl_context)
        elif hitl_type == HITLType.FEEDBACK_LOOPBACK:
            return self._handle_feedback_hitl(state, hitl_context)
        elif hitl_type == HITLType.ABORT:
            return self._handle_abort_hitl(state, hitl_context)
        else:
            # Default to approval
            return self._handle_approval_hitl(state, hitl_context)

    def _handle_approval_hitl(self, state: AgentState, context: Dict[str, Any]) -> AgentState:
        """Handle approval HITL"""
        user_decision = state.get("user_decision", "approve")
        
        if user_decision == "approve":
            state["hitl_required"] = False
            state["execution_status"] = ExecutionStatus.RUNNING
        elif user_decision == "abort":
            state["hitl_required"] = False
            state["execution_status"] = ExecutionStatus.FAILED
        
        return state

    def _handle_edit_hitl(self, state: AgentState, context: Dict[str, Any]) -> AgentState:
        """Handle edit HITL"""
        user_edits = state.get("user_edits", {})
        
        # Apply user edits to state
        for key, value in user_edits.items():
            state[key] = value
        
        state["hitl_required"] = False
        state["execution_status"] = ExecutionStatus.RUNNING
        
        return state

    def _handle_feedback_hitl(self, state: AgentState, context: Dict[str, Any]) -> AgentState:
        """Handle feedback HITL"""
        user_feedback = state.get("user_feedback", "")
        
        # Add feedback to history
        if "feedback_history" not in state:
            state["feedback_history"] = []
        
        state["feedback_history"].append({
            "timestamp": context.get("timestamp", ""),
            "phase": context.get("phase", ""),
            "substep": context.get("substep", ""),
            "feedback": user_feedback
        })
        
        state["hitl_required"] = False
        state["execution_status"] = ExecutionStatus.RUNNING
        
        return state

    def _handle_abort_hitl(self, state: AgentState, context: Dict[str, Any]) -> AgentState:
        """Handle abort HITL"""
        state["hitl_required"] = False
        state["execution_status"] = ExecutionStatus.FAILED
        state["abort_reason"] = "User requested abort"
        
        return state

    def _route_hitl_decision(self, state: AgentState) -> str:
        """Route HITL decision"""
        user_decision = state.get("user_decision", "approve")
        
        if user_decision == "approve":
            return "approve"
        elif user_decision == "edit":
            return "edit"
        elif user_decision == "rerun":
            return "rerun"
        elif user_decision == "abort":
            return "abort"
        else:
            return "approve"  # Default
    
    def _executor_node(self, state: AgentState) -> AgentState:
        """Executor node execution"""
        try:
            result = self.executor.step(state)
            state.update(result)
            state["executor_completed"] = True
            return state
        except Exception as e:
            state["error"] = str(e)
            state["error_type"] = "executor_error"
            return state
    
    def _error_handler_node(self, state: AgentState) -> AgentState:
        """Error handling node"""
        error_type = state.get("error_type", "unknown")
        error_message = state.get("error", "Unknown error")
        
        # Log error
        if self.metrics_collector:
            self.metrics_collector.record_error(
                "orchestration", error_type, 
                {"error_message": error_message, "state": state}
            )
        
        # Determine recovery strategy
        if error_type == "planner_error":
            state["recovery_strategy"] = "retry_planner"
        elif error_type == "executor_error":
            state["recovery_strategy"] = "retry_executor"
        else:
            state["recovery_strategy"] = "replan"
        
        # Clear error state for retry
        state.pop("error", None)
        state.pop("error_type", None)
        
        return state
    
    def _finalizer_node(self, state: AgentState) -> AgentState:
        """Finalization node"""
        # Generate final report
        final_report = self._generate_final_report(state)
        state["final_report"] = final_report
        state["execution_status"] = ExecutionStatus.COMPLETED.value
        
        # Record completion metrics
        if self.metrics_collector:
            total_time = state.get("total_execution_time", 0)
            self.metrics_collector.record_execution_time(
                "orchestration", total_time,
                {"status": "completed", "steps": len(state.get("execution_plan", []))}
            )
        
        return state
    
    def _route_after_execution(self, state: AgentState) -> str:
        """Route after executor execution"""
        if state.get("error"):
            return "error"
        
        if state.get("hitl_required", False):
            return "hitl"
        
        if state.get("execution_result", {}).get("success", False):
            return "success"
        
        if state.get("replan_required", False):
            return "replan"
        
        return "error"  # Default to error if unclear
    
    def _generate_final_report(self, state: AgentState) -> Dict[str, Any]:
        """Generate final analysis report"""
        execution_log = state.get("execution_log", [])
        results = state.get("results", {})
        
        report = {
            "summary": {
                "query": state.get("initial_query", ""),
                "status": "completed",
                "total_steps": len(execution_log),
                "successful_steps": len([log for log in execution_log if log.get("success", False)]),
                "execution_time": sum(log.get("duration", 0) for log in execution_log)
            },
            "results": {
                "data_exploration": results.get("data_exploration", {}),
                "causal_discovery": results.get("causal_discovery", {}),
                "causal_inference": results.get("causal_inference", {})
            },
            "execution_log": execution_log,
            "recommendations": self._generate_recommendations(state)
        }
        
        return report
    
    def _generate_recommendations(self, state: AgentState) -> list:
        """Generate recommendations based on analysis results"""
        recommendations = []
        
        # Data quality recommendations
        if state.get("data_quality_issues"):
            recommendations.append("데이터 품질 개선을 위해 추가적인 전처리가 필요합니다.")
        
        # Causal discovery recommendations
        if state.get("causal_graph_confidence", 0) < 0.7:
            recommendations.append("인과 그래프의 신뢰도가 낮습니다. 추가 데이터나 다른 알고리즘을 고려해보세요.")
        
        # Causal inference recommendations
        if state.get("causal_estimates_confidence", 0) < 0.8:
            recommendations.append("인과 효과 추정의 신뢰도가 낮습니다. 더 많은 데이터나 민감도 분석을 수행해보세요.")
        
        return recommendations
    
    def compile(self):
        """Compile the graph"""
        self.compiled_graph = self.graph.compile()
        return self.compiled_graph
    
    def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> AgentState:
        """Execute the orchestration with a user query"""
        if not self.compiled_graph:
            self.compile()
        
        # Create initial state
        initial_state = create_initial_state(query)
        if context:
            initial_state.update(context)
        
        # Execute the graph
        result = self.compiled_graph.invoke(initial_state)
        
        return result

    def handle_hitl_response(self, state: AgentState, user_decision: str, 
                           user_edits: Dict[str, Any] = None, user_feedback: str = None) -> AgentState:
        """Handle HITL response and continue execution"""
        # Update state with user input
        state["user_decision"] = user_decision
        if user_edits:
            state["user_edits"] = user_edits
        if user_feedback:
            state["user_feedback"] = user_feedback
        
        # Continue execution
        if not self.compiled_graph:
            self.compile()
        
        return self.compiled_graph.invoke(state)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current orchestration status"""
        return {
            "planner_status": self.planner.get_capabilities(),
            "executor_status": self.executor.get_execution_status(),
            "graph_compiled": self.compiled_graph is not None
        }

def create_orchestration_graph(
    planner_config: Optional[Dict[str, Any]] = None,
    executor_config: Optional[Dict[str, Any]] = None,
    metrics_collector: Optional[MetricsCollector] = None
) -> OrchestrationGraph:
    """Create and return an orchestration graph"""
    return OrchestrationGraph(planner_config, executor_config, metrics_collector)
