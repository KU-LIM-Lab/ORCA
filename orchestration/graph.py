# orchestration/graph.py
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from core.state import AgentState, ExecutionStatus, create_initial_state
from orchestration.planner.agent import PlannerAgent
from orchestration.executor.agent import ExecutorAgent
from monitoring.metrics.collector import MetricsCollector
import json

class OrchestrationGraph:
    """Main orchestration graph that coordinates planner and executor"""
    
    def __init__(self, 
                 planner_config: Optional[Dict[str, Any]] = None,
                 executor_config: Optional[Dict[str, Any]] = None,
                 metrics_collector: Optional[MetricsCollector] = None,
                 orchestration_config: Optional[Dict[str, Any]] = None):
        self.metrics_collector = metrics_collector
        self.orchestration_config = orchestration_config or {}
        self.interactive = bool(self.orchestration_config.get("interactive", False))
        
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
        
        # Build the graph
        self.graph = self._build_graph()
        self.compiled_graph = None 

    
    def _build_graph(self) -> StateGraph:
        """Build the orchestration graph with HITL support"""
        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node("planner", self._planner_node)
        graph.add_node("executor", self._executor_node)
        
        # Set entry point
        graph.set_entry_point("planner")
        
        # Add edges
        graph.add_conditional_edges(
            "executor",
            self._route_after_execution,
            {
                # terminate graph on success or error
                "success": END,
                "error": END,
                "continue": "executor"
            }
        )

        graph.add_conditional_edges(
            "planner",  # planner 노드의 결과에 따라
            lambda x: x["plan_created"], # state의 'next_node' 값을 보고 판단
            {
                False: "planner",   # 'ask_user'이면 다시 planner로
                True: "executor"  # 'success'이면 executor로
            }
        )

        
        return graph
    
    def _planner_node(self, state: AgentState) -> AgentState:
        """Planner node execution"""
        result = self.planner.step(state)
        state.update(result)
        state["planner_completed"] = True
        return state

    
    def _executor_node(self, state: AgentState) -> AgentState:
        """Executor node execution"""
        from langgraph.types import interrupt
        
        result = self.executor.step(state)
        state.update(result)
        
        # Check if any agent requested HITL via state flag
        # This happens when agents like DataPreprocessorAgent are called from within
        # regular Python methods (not LangGraph nodes), so they can't call interrupt() directly
        if state.get("__hitl_requested__"):
            payload = state.get("__hitl_payload__", {})
            hitl_type = state.get("__hitl_type__", "unknown")
            
            # Clear the flags
            state.pop("__hitl_requested__", None)
            state.pop("__hitl_payload__", None)
            state.pop("__hitl_type__", None)
            
            # Now we're in a LangGraph node, so interrupt() will work properly
            # This will be caught by the stream() loop as __interrupt__ event
            user_input = interrupt(payload)
            
            if user_input and isinstance(user_input, dict):
                # Apply user input to state
                state.update(user_input)
                # If schema was provided for schema_review, it's already in state
                if "variable_schema" in user_input and hitl_type == "schema_review":
                    state["variable_info"] = user_input["variable_schema"]
        
        # state["executor_completed"] = True
        return state
    
    
    def _route_after_execution(self, state: AgentState) -> str:
        """Generate final analysis report"""
        """ llm으로 결과 작성하는 specialist agent로 구성 예정 """
        if state.get("error"):
            return "error"
        if state.get("executor_completed"):
            return "success"
        return "continue"
    
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
        self.compiled_graph = self.graph.compile(checkpointer=InMemorySaver()) #checkpointer=InMemorySaver()
        return self.compiled_graph
    
    def execute(self, query: str, context: Optional[Dict[str, Any]] = None, session_id: Optional[str] = None) -> AgentState:
        """Execute the orchestration with a user query.
        If interactive is False, run non-interactively and return final state via invoke.
        If interactive is True, stream with HITL prompts and return the last known state.
        Uses thread_id for resumable interrupts.
        """
        if not self.compiled_graph:
            self.compile()
        
        # Create initial state
        initial_state = create_initial_state(query)
        if context:
            initial_state.update(context)
            skip_steps = list(context.get("skip", []) or [])
            if context.get("gt_df") is not None:
                try:
                    from utils.redis_df import save_df_parquet
                    df = context.get("gt_df")
                    # Create a session-specific key
                    sid = session_id or "default_session"
                    key = f"{initial_state.get('db_id','default')}:df:{sid}"
                    save_df_parquet(key, df)
                    initial_state["df_redis_key"] = key
                    try:
                        initial_state["df_shape"] = tuple(df.shape) if hasattr(df, "shape") else None
                        initial_state["columns"] = list(df.columns) if hasattr(df, "columns") else None
                    except Exception:
                        pass
                except Exception:
                    initial_state["df_preprocessed"] = context.get("gt_df")
                if "gt_df" in initial_state:
                    try:
                        del initial_state["gt_df"]
                    except Exception:
                        initial_state["gt_df"] = None
                initial_state["data_exploration_status"] = "skipped"
                for s in ["table_selection", "table_retrieval", "data_preprocessing"]:
                    if s not in skip_steps:
                        skip_steps.append(s)

            if context.get("gt_graph") is not None:
                initial_state["selected_graph"] = context.get("gt_graph")
                initial_state["causal_discovery_status"] = "skipped"
                for s in [
                    "data_profiling",
                    "algorithm_configuration",
                    "run_algorithms_portfolio",
                    "graph_scoring",
                    "graph_evaluation",
                    "ensemble_synthesis",
                ]:
                    if s not in skip_steps:
                        skip_steps.append(s)
            if context.get("treatment"):
                initial_state["treatment_variable"] = context.get("treatment")
            if context.get("outcome"):
                initial_state["outcome_variable"] = context.get("outcome")
            if skip_steps:
                initial_state["skip_steps"] = skip_steps
        
        initial_state["interactive"] = self.interactive
        for key in ("df_preprocessed", "df_raw", "df", "gt_df"):
            initial_state.pop(key, None)
        
        # Execute the graph
        config = {"configurable": {"thread_id": session_id or "default_session"}}
        
        if not self.interactive:
            result = self.compiled_graph.invoke(initial_state, config=config)
            return result

        print("\n--- Graph Execution Stream Starts ---")
        final_state = None
        current_input = initial_state
        interrupt_count = 0

        # Main execution loop: handles multiple interrupts
        # Flow:
        # 1. Stream graph execution until interrupt or completion
        # 2. If interrupt: get user input, update state, resume from checkpoint
        # 3. If completed: exit loop
        while True:
            print(f"\n{'='*60}")
            print(f"🔄 Stream iteration (interrupt count: {interrupt_count})")
            print(f"{'='*60}")
            
            # Stream graph execution - naturally terminates on interrupt or completion
            found_interrupt = False
            completed = False
            
            for step in self.compiled_graph.stream(current_input, config=config):
                step_name = list(step.keys())[0]
                
                if step_name == '__interrupt__':
                    state_data = step[step_name]
                    interrupt_obj = state_data[0] if isinstance(state_data, tuple) else state_data
                    payload = interrupt_obj.value if hasattr(interrupt_obj, 'value') else (interrupt_obj if isinstance(interrupt_obj, dict) else {})
                    
                    # Ensure payload is a dict
                    if not isinstance(payload, dict):
                        payload = {}
                    
                    interrupt_count += 1
                    step_name_from_payload = payload.get('step', 'unknown')
                    phase_name = payload.get('phase', 'unknown')
                    description = payload.get('description', 'N/A')
                    
                    # User-friendly message instead of technical "INTERRUPT"
                    print(f"\n{'='*60}")
                    print(f"⏸️  Please check the following information (#{interrupt_count})")
                    print(f"{'='*60}")
                    print(f"📋  Step: {step_name_from_payload} ({phase_name})")
                    if description and description != 'N/A':
                        print(f"   {description}")
                    print(f"\nPlease check the following information before proceeding to the next step.")
                    result_fields = {"df_shape": "Data shape", "variable_schema": "Variable schema"}
                    
                    # Show current step results and editable fields before decision
                    try:
                        current_state = self.compiled_graph.get_state(config).values
                        step_name = payload.get('step', '')
                        
                        result_fields = {}
                        if step_name == "table_selection":
                            result_fields = {"selected_tables": "Selected tables"}
                        elif step_name == "table_retrieval":
                            result_fields = {"sql_query": "Generated SQL query"}
                        elif step_name == "data_preprocessing":
                            result_fields = {"df_shape": "Data shape", "variable_schema": "Variable schema"}
                        elif step_name == "algorithm_configuration":
                            result_fields = {"selected_algorithms": "Selected algorithms"}
                        elif step_name == "ensemble_synthesis":
                            result_fields = {"selected_graph": "Selected causal graph"}
                        elif step_name == "select_configuration":
                            result_fields = {
                                "treatment_variable": "Treatment variable",
                                "outcome_variable": "Outcome variable",
                                "confounders": "Confounders"
                            }
                        
                        if result_fields:
                            print("\n📊 Current step results:")
                            for field, label in result_fields.items():
                                value = current_state.get(field)
                                if value:
                                    if isinstance(value, (list, dict)) and len(str(value)) > 150:
                                        print(f"   - {label}: {str(value)[:150]}...")
                                    else:
                                        print(f"   - {label}: {value}")
                                else:
                                    print(f"   - {label}: (not set)")
                        
                        # Show editable fields information before decision
                        from orchestration.executor.agent import ExecutorAgent
                        all_editable_fields = ExecutorAgent._get_editable_fields()
                        
                        relevant_fields = []
                        if step_name == "table_selection":
                            relevant_fields = ["selected_tables"]
                        elif step_name == "table_retrieval":
                            relevant_fields = ["sql_query"]
                        elif step_name == "data_preprocessing":
                            relevant_fields = ["target_columns"]
                        elif step_name == "algorithm_configuration":
                            relevant_fields = ["selected_algorithms"]
                        elif step_name == "ensemble_synthesis":
                            relevant_fields = ["selected_graph"]
                        elif step_name == "select_configuration":
                            relevant_fields = ["treatment_variable", "outcome_variable", "confounders", "instrumental_variables"]
                        else:
                            relevant_fields = list(all_editable_fields.keys())
                        
                        if relevant_fields:
                            print("\n📝 Editable fields (if you choose 'edit'):")
                            for field_name in relevant_fields:
                                if field_name not in all_editable_fields:
                                    continue
                                field_info = all_editable_fields[field_name]
                                current_value = current_state.get(field_name)
                                
                                print(f"   - {field_name} ({field_info['type']}): {field_info['description']}")
                                if current_value:
                                    if isinstance(current_value, (list, dict)) and len(str(current_value)) > 100:
                                        print(f"     Current: {str(current_value)[:100]}...")
                                    else:
                                        print(f"     Current: {current_value}")
                                else:
                                    print(f"     Current: (not set)")
                                    if 'example' in field_info:
                                        print(f"     Example: {field_info['example']}")
                    except Exception as e:
                        print(f"\n⚠️  Could not retrieve step information: {e}")
                    
                    # Step 1: Get decision
                    while True:
                        decision = input("\n💬 Choose decision (approve/edit/rerun/abort): ").strip().lower()
                        if decision in ["approve", "edit", "rerun", "abort"]:
                            break
                        print("❌ Invalid decision. Please choose: approve, edit, rerun, or abort")
                    
                    # Initialize user_data with decision
                    user_data = {
                        "decision": decision,
                        "hitl_decision": decision,  # Also store as hitl_decision for executor
                        "hitl_executed": True
                    }
                    
                    # Step 2: Get additional info based on decision
                    if decision == "edit":
                        # Editable fields were already shown above, just get edits
                        while True:
                            try:
                                edits_input = input("\n💬 Enter edits (JSON format, or '{}' for no edits): ").strip()
                                if not edits_input:
                                    edits_input = "{}"
                                edits = json.loads(edits_input)
                                if isinstance(edits, dict):
                                    user_data["edits"] = edits
                                    break
                                else:
                                    print("❌ Edits must be a JSON object (dictionary)")
                            except json.JSONDecodeError:
                                print("❌ Invalid JSON format. Please try again.")
                    
                    elif decision == "rerun":
                        # Get feedback
                        feedback = input("\n💬 Enter feedback (optional): ").strip()
                        if feedback:
                            user_data["feedback"] = feedback
                    
                    # For approve and abort, no additional input needed
                    print(f"\n✅ Decision: {decision}")
                    if decision == "edit" and "edits" in user_data:
                        print(f"   Edits: {json.dumps(user_data['edits'], indent=2)}")
                    elif decision == "rerun" and "feedback" in user_data:
                        print(f"   Feedback: {user_data['feedback']}")
                
                    # Update state with user decision - this allows executor to resume and process HITL
                    current_state = self.compiled_graph.get_state(config).values.copy()
                    current_state.update(user_data)
                    self.compiled_graph.update_state(config, current_state)
                    
                    # Verify state was updated correctly
                    verify_state = self.compiled_graph.get_state(config).values
                    if not verify_state.get("hitl_executed"):
                        verify_state.update(user_data)
                        self.compiled_graph.update_state(config, verify_state)
                    
                    found_interrupt = True
                    # Stream terminates after interrupt, break to exit for loop
                    # Resume from checkpoint in next while loop iteration
                    break
                else:
                    # 일반 노드 실행 완료
                    node_state = step[step_name]
                    
                    if isinstance(node_state, dict):
                        print(f"\n✅ Node '{step_name}' completed")
                        print(f"📊 Current state:")
                        completed_key = f'{step_name}_completed'
                        # Check both node_state and get_state for completed flag
                        completed_value = node_state.get(completed_key)
                        if completed_value is None:
                            try:
                                current_state = self.compiled_graph.get_state(config).values
                                completed_value = current_state.get(completed_key, 'N/A')
                            except Exception:
                                completed_value = 'N/A'
                        print(f"   - {completed_key}: {completed_value}")
                        final_state = node_state
            
            # Check if stream completed without interrupts
            if not found_interrupt:
                print("\n✅ Stream completed without interrupts!")
                completed = True
                break

            # Interrupt occurred - resume from checkpoint in next iteration
            # Setting current_input=None makes LangGraph resume from checkpoint
            current_input = None
            print(f"\n🔄 Resuming from checkpoint after interrupt #{interrupt_count}...")
            
    
        print("\n--- Graph Execution Stream 종료 ---\n")
        return final_state
    
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
    metrics_collector: Optional[MetricsCollector] = None,
    orchestration_config: Optional[Dict[str, Any]] = None
) -> OrchestrationGraph:
    """Create and return an orchestration graph"""
    return OrchestrationGraph(
        planner_config=planner_config,
        executor_config=executor_config,
        metrics_collector=metrics_collector,
        orchestration_config=orchestration_config,
    )
