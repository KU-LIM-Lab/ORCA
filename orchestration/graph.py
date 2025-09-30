# orchestration/graph.py
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from core.state import AgentState, ExecutionStatus, create_initial_state
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
        self.planner.add_sub_agent(self.executor) # Q. executor를 planner의 sub-agent로 추가하는 게 어떤 의미인지
        
        # Build the graph
        self.graph = self._build_graph()
        self.compiled_graph = None # Q. compiled_graph는 무엇인가?
    
    def _build_graph(self) -> StateGraph:
        """Build the orchestration graph with HITL support"""
        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node("planner", self._planner_node)
        graph.add_node("executor", self._executor_node)
        graph.add_node("error_handler", self._error_handler_node)
        graph.add_node("finalizer", self._finalizer_node)
        
        # Set entry point
        graph.set_entry_point("planner")
        
        # Add edges
        graph.add_edge("planner", "executor")
        graph.add_conditional_edges(
            "executor",
            self._route_after_execution,
            {
                "success": "finalizer",
                "error": "error_handler"
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
        
        # Determine recovery strategy -> 수정
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
        return "success"
    
    def _generate_final_report(self, state: AgentState) -> Dict[str, Any]:
        """Generate final analysis report"""
        """ llm으로 결과 작성하는 specialist agent로 구성 예정 """
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
            "results": results,
            "execution_log": execution_log
        }
        
        return report
    
    def compile(self):
        """Compile the graph with in-memory checkpointer"""
        self.compiled_graph = self.graph.compile(checkpointer=InMemorySaver())
        return self.compiled_graph
    
    def execute(self, query: str, context: Optional[Dict[str, Any]] = None, session_id: Optional[str] = None) -> AgentState:
        """Execute the orchestration with a user query. Uses thread_id for resumable interrupts."""
        if not self.compiled_graph:
            self.compile()
        
        # Create initial state
        initial_state = create_initial_state(query)
        if context:
            initial_state.update(context)
        
        # Execute the graph
        config = {"configurable": {"thread_id": session_id or "default_session"}}
        result = self.compiled_graph.invoke(initial_state, config=config)
        
        return result
    
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
