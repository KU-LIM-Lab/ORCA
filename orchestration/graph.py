# orchestration/graph.py
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from core.state import AgentState, ExecutionStatus, create_initial_state
from orchestration.planner.agent import PlannerAgent
from orchestration.executor.agent import ExecutorAgent
from monitoring.metrics.collector import MetricsCollector
import json
import time

class OrchestrationGraph:
    """Main orchestration graph that coordinates planner and executor"""
    
    def __init__(self, 
                 planner_config: Optional[Dict[str, Any]] = None,
                 executor_config: Optional[Dict[str, Any]] = None,
                 metrics_collector: Optional[MetricsCollector] = None,
                 orchestration_config: Optional[Dict[str, Any]] = None,
                 event_logger: Optional[Any] = None):
        self.metrics_collector = metrics_collector
        self.orchestration_config = orchestration_config or {}
        self.interactive = bool(self.orchestration_config.get("interactive", False))
        self.event_logger = event_logger  # EventLogger for experiment tracking
        
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
        """Build the orchestration graph with HITL support via separate gate node"""
        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node("planner", self._planner_node)
        graph.add_node("executor", self._executor_node)
        graph.add_node("hitl_gate", self._hitl_gate_node)
        
        # Set entry point
        graph.set_entry_point("planner")
        
        # Planner edges
        graph.add_conditional_edges(
            "planner",
            lambda x: x["plan_created"],
            {
                False: "planner",   # Loop back if plan not created
                True: "executor"    # Start execution when plan ready
            }
        )
        
        # Executor always goes to HITL gate (state persisted before gate)
        graph.add_edge("executor", "hitl_gate")
        
        # HITL gate routes to executor (continue) or END (done/error)
        graph.add_conditional_edges(
            "hitl_gate",
            self._route_after_hitl,
            {
                "executor": "executor",  # Loop back for next step
                "end": END               # Done or error
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
        """Execute ONE step from the plan.
        
        State is persisted when this node returns.
        """
        # Map substep to step_id for logging
        plan = state.get("execution_plan", []) or []
        idx = state.get("current_execute_step", 0)
        current_substep = plan[idx]["substep"] if idx < len(plan) else state.get("current_substep", "")
        step_id = self._map_substep_to_step_id(current_substep)
        
        # Log step enter if event logger available
        if self.event_logger and step_id:
            self.event_logger.log_step_enter(
                step_id=step_id,
                substep=current_substep,
                metadata={"timestamp": time.time()}
            )
        
        step_start_time = time.time()
        
        # Execute single step - returns state with advanced step counter
        updated_state = self.executor.step(state)
        
        print(f"🔍 DEBUG [_executor_node]: Received from executor.step(): current_execute_step={updated_state.get('current_execute_step', 'N/A')}")
        
        state = updated_state
        
        # Log step exit if event logger available
        if self.event_logger and step_id:
            duration = time.time() - step_start_time
            self.event_logger.log_step_exit(
                step_id=step_id,
                substep=current_substep,
                success=not bool(state.get("error")),
                duration=duration,
                metadata={"timestamp": time.time()}
            )
        
        # Return state immediately so LangGraph persists it to checkpoint
        print(f"🔍 DEBUG [_executor_node]: Returning state with current_execute_step={state.get('current_execute_step', 'N/A')}")
        return state
    
    def _hitl_gate_node(self, state: AgentState) -> AgentState:
        """HITL gate node - handles interrupts and user decisions.
        
        This node checks if HITL was requested, triggers interrupt if needed,
        and handles user decisions (approve, edit, rerun, abort).
        """
        from langgraph.types import interrupt
        
        # Check if HITL was requested by the step execution
        if not state.get("__hitl_requested__"):
            # No HITL needed, just pass through
            print(f"🔍 DEBUG [HITL Gate]: No HITL requested, passing through")
            return state
        
        # HITL needed - get payload and trigger interrupt
        payload = state.get("__hitl_payload__", {})
        hitl_type = state.get("__hitl_type__", "unknown")
        
        # Get the substep that just executed (step counter was already advanced)
        plan = state.get("execution_plan", [])
        current_idx = state.get("current_execute_step", 1) - 1  # Step already advanced
        substep = plan[current_idx]["substep"] if 0 <= current_idx < len(plan) else "unknown"
        
        print(f"🔍 DEBUG [HITL Gate]: HITL requested for substep '{substep}', checkpoint has current_execute_step={state.get('current_execute_step', 'N/A')}")
        
        
        # Log HITL prompt
        step_id = self._map_substep_to_step_id(substep)
        if self.event_logger and step_id:
            self.event_logger.log_hitl_prompt_shown(
                step_id=step_id,
                phase=payload.get("phase", "unknown"),
                description=payload.get("description"),
                metadata={"hitl_type": hitl_type}
            )
        
        # Trigger interrupt - this will suspend execution
        # After resume, the node re-executes from the beginning
        # So we check if flags were cleared by update_state
        user_input = interrupt(payload)
        
        
        if user_input and isinstance(user_input, dict):
            decision = user_input.get("decision", "approve")
            print(f"🔍 DEBUG [HITL Gate]: User decision: {decision}")
            
            # Log decision
            if self.event_logger and step_id:
                self.event_logger.log_hitl_decision(
                    step_id=step_id,
                    decision=decision,
                    edits=user_input.get("edits"),
                    feedback=user_input.get("feedback"),
                    metadata={}
                )
            
            # Handle different decisions
            if decision == "approve":
                # Step already advanced in executor, just continue
                print(f"🔍 DEBUG [HITL Gate]: Approved, continuing with current_execute_step={state.get('current_execute_step')}")
                pass
                
            elif decision == "edit":
                # Apply edits and re-execute current step
                edits = user_input.get("edits", {})
                state.update(edits)
                # Decrement step counter to re-execute the step that was just completed
                current_step = state.get("current_execute_step", 0)
                state["current_execute_step"] = max(0, current_step - 1)
                state["current_state_executed"] = False
                # Remove from completed substeps so it can be re-executed
                completed = state.get("completed_substeps", [])
                if substep in completed:
                    completed.remove(substep)
                print(f"🔍 DEBUG [HITL Gate]: Edits applied, re-executing step {substep} (idx={state.get('current_execute_step')})")
                
            elif decision == "rerun":
                # Store feedback and re-execute current step
                feedback = user_input.get("feedback", "")
                if feedback:
                    state["user_feedback"] = feedback
                # Decrement step counter to re-execute
                current_step = state.get("current_execute_step", 0)
                state["current_execute_step"] = max(0, current_step - 1)
                state["current_state_executed"] = False
                # Remove from completed substeps
                completed = state.get("completed_substeps", [])
                if substep in completed:
                    completed.remove(substep)
                print(f"🔍 DEBUG [HITL Gate]: Rerun requested, re-executing step {substep} (idx={state.get('current_execute_step')})")
                
            elif decision == "abort":
                # Set error to stop execution
                state["error"] = "User aborted execution"
                state["executor_completed"] = True
                print(f"🔍 DEBUG [HITL Gate]: Execution aborted by user")
            
            # Log HITL applied
            if self.event_logger and step_id:
                self.event_logger.log_hitl_applied(
                    step_id=step_id,
                    applied=True,
                    metadata={"decision": decision}
                )
        
        return state
    
    
    def _map_substep_to_step_id(self, substep: str) -> Optional[str]:
        """Map substep name to step_id (1, 2, 3) for logging."""
        # Step 1: Data Wrangling/Exploration
        step1_substeps = [
            "table_selection",
            "table_retrieval",
            "data_preprocessing",
            "table_recommendation",
            "text2sql_generation",
        ]
        
        # Step 2: Causal Discovery
        step2_substeps = [
            "data_profiling",
            "algorithm_configuration",
            "run_algorithms_portfolio",
            "graph_scoring",
            "graph_evaluation",
            "ensemble_synthesis",
        ]
        
        # Step 3: Causal Inference
        step3_substeps = [
            "parse_question",
            "select_configuration",
            "dowhy_analysis",
            "generate_answer",
        ]
        
        if substep in step1_substeps:
            return "1"
        elif substep in step2_substeps:
            return "2"
        elif substep in step3_substeps:
            return "3"
        return None
    
    def _route_after_execution(self, state: AgentState) -> str:
        """Route after execution - DEPRECATED, kept for compatibility"""
        """ llm으로 결과 작성하는 specialist agent로 구성 예정 """
        if state.get("error"):
            return "error"
        if state.get("executor_completed"):
            return "success"
        return "continue"
    
    def _route_after_hitl(self, state: AgentState) -> str:
        """Route after HITL gate node.
        
        Returns:
            "end" - execution completed or error occurred
            "executor" - continue to next step
        """
        if state.get("error"):
            print(f"🔍 DEBUG [Router]: Error detected, routing to END")
            return "end"
        if state.get("executor_completed"):
            print(f"🔍 DEBUG [Router]: Execution completed, routing to END")
            return "end"
        print(f"🔍 DEBUG [Router]: Continuing to executor for next step")
        return "executor"
    
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

        final_state = None
        current_input = initial_state
        interrupt_count = 0

        # Main execution loop: handles multiple interrupts
        # Flow:
        # 1. Stream graph execution until interrupt or completion
        # 2. If interrupt: get user input, update state, resume from checkpoint
        # 3. If completed: exit loop
        while True:
            
            # Stream graph execution - naturally terminates on interrupt or completion
            found_interrupt = False
            completed = False
            
            for step in self.compiled_graph.stream(current_input, config=config):
                step_name = list[str](step.keys())[0]
                
                # Debug: Log which node is being executed
                if step_name not in ['__interrupt__']:
                    print(f"🔍 Debug: Executing node: {step_name}")
                
                if step_name == '__interrupt__':
                    print(f"🔍 Debug: Interrupted by: {step_name}")
                    state_data = step[step_name]
                    interrupt_obj = state_data[0] if isinstance(state_data, tuple) else state_data
                    payload = interrupt_obj.value if hasattr(interrupt_obj, 'value') else (interrupt_obj if isinstance(interrupt_obj, dict) else {})
                    
                    # Ensure payload is a dict
                    if not isinstance(payload, dict):
                        payload = {}
                    
                    # ✅ DEBUG: Check checkpoint state after interrupt
                    try:
                        checkpoint_state = self.compiled_graph.get_state(config).values
                        current_step_idx = checkpoint_state.get("current_execute_step", "N/A")
                        print(f"🔍 DEBUG [Graph]: Checkpoint state has current_execute_step={current_step_idx}")
                    except Exception as e:
                        print(f"🔍 DEBUG [Graph]: Could not read checkpoint: {e}")
                    
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
                        # The executor node has returned with advanced step counter
                        # So the checkpoint now has the correct state with current results
                        # This works because we moved step advancement BEFORE HITL check
                        current_state = self.compiled_graph.get_state(config).values
                        
                        step_name = payload.get('step', '')
                        
                        result_fields = {}
                        if step_name == "table_selection":
                            result_fields = {
                                "selected_tables": "Selected tables"
                            }
                        elif step_name == "table_retrieval":
                            result_fields = {"final_sql": "Generated SQL query", "sql_query": "SQL query"}
                        elif step_name == "data_preprocessing":
                            result_fields = {
                                "df_shape": "Data shape",
                                "columns": "Columns",
                                "variable_schema": "Variable schema"
                            }
                        elif step_name == "data_profiling":
                            result_fields = {
                                "data_profile": "Data profile summary"
                            }
                        elif step_name == "algorithm_configuration":
                            result_fields = {"cd_execution_plan": "Algorithm execution plan"}
                        elif step_name == "run_algorithms_portfolio":
                            result_fields = {
                                "executed_algorithms": "Executed algorithms",
                                "algorithm_results": "Algorithm results",
                                "algorithm_graph_visualization_paths": "Graph visualizations"
                            }
                        elif step_name == "graph_scoring":
                            result_fields = {
                                "scored_graphs": "Algorithm graphs with scores",
                                "ranked_graphs": "Ranked graphs (if evaluated)",
                                "top_candidates": "Top candidate graphs (if evaluated)"
                            }
                        elif step_name == "graph_evaluation":
                            result_fields = {
                                "ranked_graphs": "Ranked graphs by composite score",
                                "top_candidates": "Top 3 candidate graphs"
                            }
                        elif step_name == "ensemble_synthesis":
                            result_fields = {
                                "graph_visualization_path": "Final graph visualization",
                                "selected_graph": "Final ensemble DAG"
                            }
                        elif step_name == "parse_question":
                            result_fields = {
                                "parsed_query": "Identified variable roles"
                            }
                        elif step_name == "select_configuration":
                            result_fields = {
                                "treatment_variable": "Treatment",
                                "outcome_variable": "Outcome",
                                "confounders": "Confounders",
                                "strategy": "Selected strategy"
                            }
                        
                        if result_fields:
                            print("\n📊 Current step results:")
                            for field, label in result_fields.items():
                                value = current_state.get(field)
                                                                
                                if value is not None: 
                                    if field == "selected_tables" and isinstance(value, list):
                                        if value:
                                            print(f"   ✓ {', '.join(value)}")
                                    elif field == "data_profile" and isinstance(value, dict):
                                        basic_checks = value.get("basic_checks", {})
                                        global_scores = value.get("global_scores", {})
                                        pairwise_scores = value.get("pairwise_scores", {})
                                        print(f"   - {label}:")
                                        
                                        # Show basic checks details
                                        if basic_checks:
                                            print(f"     • Basic checks ({len(basic_checks)} performed):")
                                            for key, val in list(basic_checks.items())[:5]:  # Show first 5
                                                print(f"       - {key}: {val}")
                                            if len(basic_checks) > 5:
                                                print(f"       ... and {len(basic_checks) - 5} more")
                                        
                                        # Show global scores summary
                                        if global_scores:
                                            print(f"     • Global scores ({len(global_scores)} scores computed):")
                                            for key, val in list(global_scores.items())[:3]:  # Show first 3
                                                if isinstance(val, (int, float)):
                                                    print(f"       - {key}: {val:.4f}" if isinstance(val, float) else f"       - {key}: {val}")
                                                else:
                                                    print(f"       - {key}: {val}")
                                            if len(global_scores) > 3:
                                                print(f"       ... and {len(global_scores) - 3} more")
                                        
                                        # Show pairwise scores count
                                        if pairwise_scores:
                                            print(f"     • Pairwise scores: {len(pairwise_scores)} pairs analyzed")
                                        
                                        if not basic_checks and not global_scores and not pairwise_scores:
                                            print(f"     (empty profile)")
                                    # Special handling for columns - show count and sample
                                    elif field == "columns" and isinstance(value, list):
                                        col_count = len(value)
                                        if col_count > 10:
                                            sample_cols = ", ".join(value[:10])
                                            print(f"   - {label}: {col_count} columns")
                                            print(f"     Sample (first 10): {sample_cols}, ...")
                                        else:
                                            print(f"   - {label}: {', '.join(value)}")
                                    # Special handling for cd_execution_plan - show algorithm list
                                    elif field == "cd_execution_plan" and isinstance(value, list):
                                        print(f"   - {label}: {len(value)} algorithm(s)")
                                        for i, alg_config in enumerate(value[:10], 1):  # Show first 10
                                            alg_name = alg_config.get("alg", "Unknown")
                                            config_str = ", ".join([f"{k}={v}" for k, v in alg_config.items() if k != "alg"])
                                            if config_str:
                                                print(f"     {i}. {alg_name} ({config_str})")
                                            else:
                                                print(f"     {i}. {alg_name}")
                                        if len(value) > 10:
                                            print(f"     ... and {len(value) - 10} more algorithm(s)")
                                    # Special handling for executed_algorithms - show algorithm list
                                    elif field == "executed_algorithms" and isinstance(value, list):
                                        print(f"   - {label}: {len(value)} algorithm(s)")
                                        if value:
                                            print(f"     • {', '.join(value)}")
                                    # Special handling for algorithm_graph_visualization_paths - show visualization paths
                                    elif field == "algorithm_graph_visualization_paths" and isinstance(value, dict):
                                        print(f"   - {label}:")
                                        print(f"     📊 Graph visualizations saved for {len(value)} algorithm(s):")
                                        for alg_name, paths in value.items():
                                            if isinstance(paths, dict) and "error" not in paths:
                                                png_path = paths.get("png", "N/A")
                                                svg_path = paths.get("svg", "N/A")
                                                print(f"       • {alg_name}:")
                                                if png_path != "N/A":
                                                    print(f"         PNG: {png_path}")
                                                if svg_path != "N/A":
                                                    print(f"         SVG: {svg_path}")
                                            elif isinstance(paths, dict) and "error" in paths:
                                                print(f"       • {alg_name}: (visualization failed - {paths.get('error', 'Unknown error')})")
                                            else:
                                                print(f"       • {alg_name}: (no visualization available)")
                                        print(f"     💡 You can view these graph visualizations to see the causal structures discovered by each algorithm.")
                                    # Special handling for ranked_graphs - show detailed scores
                                    elif field == "ranked_graphs" and isinstance(value, list):
                                        print(f"   - {label}: {len(value)} graph(s)")
                                        for i, graph_item in enumerate(value, 1):
                                            alg_name = graph_item.get("algorithm", "Unknown")
                                            mc = graph_item.get("markov_consistency")
                                            ss = graph_item.get("sampling_stability")
                                            sts = graph_item.get("structural_stability")
                                            comp_score = graph_item.get("composite_score")
                                            
                                            score_parts = []
                                            if mc is not None:
                                                score_parts.append(f"markov_consistency={mc:.3f}")
                                            if ss is not None:
                                                score_parts.append(f"sampling_stability={ss:.3f}")
                                            if sts is not None:
                                                score_parts.append(f"structural_stability={sts:.3f}")
                                            if comp_score is not None:
                                                score_parts.append(f"composite_score={comp_score:.3f}")
                                            
                                            score_str = ", ".join(score_parts) if score_parts else "N/A"
                                            print(f"     {i}. {alg_name}: {score_str}")
                                    # Special handling for scored_graphs - show algorithm names and scores only
                                    elif field == "scored_graphs" and isinstance(value, list):
                                        print(f"   - {label}: {len(value)} graph(s)")
                                        for i, scored_graph in enumerate(value, 1):
                                            alg_name = scored_graph.get("algorithm", "Unknown")
                                            
                                            score_parts = []
                                            mc = scored_graph.get("markov_consistency")
                                            ss = scored_graph.get("sampling_stability")
                                            sts = scored_graph.get("structural_stability")
                                            
                                            if mc is not None:
                                                score_parts.append(f"markov_consistency={mc:.3f}")
                                            if ss is not None:
                                                score_parts.append(f"sampling_stability={ss:.3f}")
                                            if sts is not None:
                                                score_parts.append(f"structural_stability={sts:.3f}")
                                            
                                            score_str = ", ".join(score_parts) if score_parts else "N/A"
                                            print(f"     {i}. {alg_name}: {score_str}")
                                    # Special handling for top_candidates - show algorithm names and scores only
                                    elif field == "top_candidates" and isinstance(value, list):
                                        print(f"   - {label}: {len(value)} graph(s)")
                                        for i, candidate in enumerate(value, 1):
                                            alg_name = candidate.get("algorithm", "Unknown")
                                            comp_score = candidate.get("composite_score")
                                            
                                            score_parts = []
                                            mc = candidate.get("markov_consistency")
                                            ss = candidate.get("sampling_stability")
                                            sts = candidate.get("structural_stability")
                                            
                                            if mc is not None:
                                                score_parts.append(f"markov_consistency={mc:.3f}")
                                            if ss is not None:
                                                score_parts.append(f"sampling_stability={ss:.3f}")
                                            if sts is not None:
                                                score_parts.append(f"structural_stability={sts:.3f}")
                                            if comp_score is not None:
                                                score_parts.append(f"composite_score={comp_score:.3f}")
                                            
                                            score_str = ", ".join(score_parts) if score_parts else "N/A"
                                            print(f"     {i}. {alg_name}: {score_str}")
                                    # Special handling for graph_visualization_path - show visualization paths
                                    elif field == "graph_visualization_path" and isinstance(value, dict):
                                        print(f"   - {label}:")
                                        png_path = value.get("png", "N/A")
                                        svg_path = value.get("svg", "N/A")
                                        if png_path != "N/A":
                                            print(f"     📊 PNG: {png_path}")
                                        if svg_path != "N/A":
                                            print(f"     📊 SVG: {svg_path}")
                                        if png_path == "N/A" and svg_path == "N/A":
                                            print(f"     (No visualization available)")
                                    # Special handling for selected_graph - show nodes and edges
                                    elif field == "selected_graph" and isinstance(value, dict):
                                        print(f"   - {label}:")
                                        try:
                                            from agents.causal_discovery.tools import get_variables, get_edges
                                            nodes = get_variables(value)
                                            edges = get_edges(value)
                                            
                                            print(f"     📊 Nodes ({len(nodes)}): {', '.join(nodes)}")
                                            print(f"     📊 Edges ({len(edges)}):")
                                            for i, edge in enumerate(edges[:20], 1):  # Show first 20 edges
                                                from_node = edge.get("from", "?")
                                                to_node = edge.get("to", "?")
                                                edge_type = edge.get("type", "->")
                                                print(f"       {i}. {from_node} --{edge_type}--> {to_node}")
                                            if len(edges) > 20:
                                                print(f"       ... and {len(edges) - 20} more edge(s)")
                                        except Exception as e:
                                            print(f"     (Could not parse graph structure: {e})")
                                            print(f"     Graph type: {value.get('metadata', {}).get('graph_type', 'Unknown')}")

                                else:
                                    print(f"   - {label}: (not set)")
                        
                        # Show editable fields information before decision
                        from orchestration.executor.agent import ExecutorAgent
                        all_editable_fields = ExecutorAgent._get_editable_fields()
                        
                        relevant_fields = []
                        if step_name == "table_selection":
                            relevant_fields = ["selected_tables"]
                        elif step_name == "table_retrieval":
                            relevant_fields = ["sql_query", "final_sql"]
                        elif step_name == "data_preprocessing":
                            relevant_fields = ["target_columns", "clean_nulls_ratio", "one_hot_threshold", "high_cardinality_threshold"]
                        elif step_name == "data_profiling":
                            relevant_fields = []  # Approval only
                        elif step_name == "algorithm_configuration":
                            relevant_fields = [] # Approval only
                        elif step_name == "run_algorithms_portfolio":
                            relevant_fields = []  # Approval/rerun only
                        elif step_name == "graph_scoring":
                            relevant_fields = []  # Approval/rerun only
                        elif step_name == "graph_evaluation":
                            relevant_fields = []  # Approval/rerun only
                        elif step_name == "ensemble_synthesis":
                            relevant_fields = ["selected_graph"]
                        elif step_name == "parse_question":
                            relevant_fields = ["treatment_variable", "outcome_variable", "confounders", "mediators", "instrumental_variables"]
                        elif step_name == "select_configuration":
                            relevant_fields = ["treatment_variable", "outcome_variable", "confounders", "instrumental_variables", "strategy"]
                        else:
                            relevant_fields = list(all_editable_fields.keys())
                        
                        if relevant_fields:
                            print("\n📝 Editable fields (if you choose 'edit'):")
                            for field_name in relevant_fields:
                                if field_name not in all_editable_fields:
                                    continue
                                field_info = all_editable_fields[field_name]
                                
                                print(f"   - {field_name} ({field_info['type']}): {field_info['description']}")
                                if 'example' in field_info:
                                    print(f"     Example: {field_info['example']}")
                        else:
                            print("\n📝 This step requires approval only (no editable fields)")
                            print("   Available decisions: approve, rerun, abort")
                    except Exception as e:
                        print(f"\n⚠️  Could not retrieve step information: {e}")
                    
                    # Step 1: Get decision
                    while True:
                        if relevant_fields:  # Has editable fields
                            decision = input("\n💬 Choose decision (approve/edit/rerun/abort): ").strip().lower()
                            valid_decisions = ["approve", "edit", "rerun", "abort"]
                        else:  # Approval-only
                            decision = input("\n💬 Choose decision (approve/rerun/abort): ").strip().lower()
                            valid_decisions = ["approve", "rerun", "abort"]
                        
                        if decision in valid_decisions:
                            break
                        print(f"❌ Invalid decision. Please choose: {', '.join(valid_decisions)}")
                    
                    # Initialize user_data with decision
                    user_data = {
                        "decision": decision,
                        "hitl_decision": decision,  # Also store as hitl_decision for executor
                        "hitl_executed": True,
                        "__hitl_requested__": None,
                        "__hitl_payload__": None,
                        "__hitl_type__": None
                    }
                    
                    # Step 2: Get additional info based on decision
                    if decision == "edit":
                        # Editable fields were already shown above, just get edits
                        print("\n" + "="*60)
                        print("📝 Edit Instructions")
                        print("="*60)
                        print("You can use one of these input formats:")
                        print()
                        
                        # Show specific example based on step
                        if step_name == "data_preprocessing":
                            # Show available columns if they exist
                            available_cols = current_state.get("columns")
                            if available_cols:
                                print("💡 Note: All columns will be used by default.")
                                print("   You can filter specific columns using formats below.")
                                print()
                                print("Available columns:")
                                print(f"  {', '.join(available_cols[:30])}")
                                if len(available_cols) > 30:
                                    print(f"  ... and {len(available_cols) - 30} more")
                                print(f"  (Total: {len(available_cols)} columns)")
                                print()
                            print("Input formats:")
                            print("  1️⃣  Simple column list (recommended):")
                            print("     age, gender, purchase_amount")
                            print()
                            print("  2️⃣  Keep all settings:")
                            print("     {} or just press Enter")
                            print()
                            print("  3️⃣  Full JSON to edit specific settings:")
                            print('     {"target_columns": ["age", "gender"], "clean_nulls_ratio": 0.9}')
                            print()
                            print("Or enter {} to keep all current settings (use all columns)")
                        elif step_name == "table_selection":
                            print('  {"selected_tables": ["users", "orders", "products"]}')
                        elif step_name == "table_retrieval":
                            print('  {"sql_query": "SELECT * FROM users WHERE age > 18"}')
                        
                        print("="*60)
                        
                        while True:
                            try:
                                edits_input = input("\n💬 Enter edits (column list, JSON, or '{}' for no edits): ").strip()
                                
                                if not edits_input:
                                    edits_input = "{}"
                                
                                # Try to parse as JSON first
                                try:
                                    edits = json.loads(edits_input)
                                    if isinstance(edits, dict):
                                        user_data["edits"] = edits
                                        break
                                    else:
                                        print("❌ Edits must be a JSON object (dictionary)")
                                        continue
                                except json.JSONDecodeError:                                    
                                    # Check if it's a simple column list (comma-separated)
                                    if "," in edits_input and "{" not in edits_input:
                                        # Parse as comma-separated column names
                                        columns = [col.strip() for col in edits_input.split(",") if col.strip()]
                                        if columns:
                                            edits = {"target_columns": columns}
                                            user_data["edits"] = edits
                                            print(f"✓ Parsed as column list: {len(columns)} columns")
                                            break
                                    
                                    # Otherwise, show JSON error
                                    raise
                                    
                            except json.JSONDecodeError as e:
                                print(f"❌ Invalid input format: {e}")
                                print(f"\n💡 You can use these formats:")
                                print(f"   1) Column list: col1, col2, col3")
                                print(f"   2) Keep all: {{}} or press Enter")
                                print(f"   3) Full JSON: {{\"target_columns\": [\"col1\", \"col2\"]}}")
                    
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
                
                    # ✅ DEBUG: Show state update
                    print(f"🔍 DEBUG [Graph]: Updating state with user decision: {decision}")
                    self.compiled_graph.update_state(config, user_data)
                    
                    # ✅ DEBUG: Verify checkpoint state after update
                    updated_checkpoint = self.compiled_graph.get_state(config).values
                    print(f"🔍 DEBUG [Graph]: After update, checkpoint has current_execute_step={updated_checkpoint.get('current_execute_step', 'N/A')}")
                    
                    found_interrupt = True

                    break
                else:
                    # 일반 노드 실행 완료
                    node_state = step[step_name]
                    
                    if isinstance(node_state, dict):
                        print(f"\n✅ Node '{step_name}' completed")
                        print(f"📊 Current state:")
                        completed_key = f'{step_name}_completed'
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
                completed = True
                print(f"🔍 DEBUG [Graph]: Stream completed without interrupts")
                break

            # Interrupt occurred - resume from checkpoint in next iteration
            # Setting current_input=None makes LangGraph resume from checkpoint
            print(f"🔍 DEBUG [Graph]: Interrupt handled, resuming from checkpoint...")
            print(f"🔍 DEBUG [Graph]: Checkpoint state should have current_execute_step advanced")
            current_input = None
            
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
    orchestration_config: Optional[Dict[str, Any]] = None,
    event_logger: Optional[Any] = None
) -> OrchestrationGraph:
    """Create and return an orchestration graph"""
    return OrchestrationGraph(
        planner_config=planner_config,
        executor_config=executor_config,
        metrics_collector=metrics_collector,
        orchestration_config=orchestration_config,
        event_logger=event_logger,
    )
