# orchestration/planner/agent.py
from typing import Dict, Any, List, Optional
from core.base import OrchestratorAgent
from core.state import AgentState
from langgraph.types import interrupt
from monitoring.metrics.collector import MetricsCollector

class PlannerAgent(OrchestratorAgent):
    """Planner agent responsible for analyzing user queries and creating execution plans"""
    
    def __init__(self, name: str = "planner", config: Optional[Dict[str, Any]] = None, 
                 metrics_collector: Optional[MetricsCollector] = None):
        super().__init__(name, config, metrics_collector)
        
        # Standard analysis pipeline steps
        self.standard_pipeline = [
            "data_exploration",
            "causal_discovery", 
            "causal_analysis",
            "report_generation"
        ]
        
        # Step configurations
        self.step_configs = {
            "data_exploration": {
                "agent": "data_explorer",
                "action": "explore_data",
                "description": "데이터베이스 탐색 및 스키마 분석",
                "timeout": 300,
                "required_outputs": ["data_explored", "schema_metadata", "candidate_tables"]
            },
            "causal_discovery": {
                "agent": "causal_discovery",
                "action": "discover_causal_graph", 
                "description": "인과 발견 알고리즘 선택 및 그래프 생성",
                "timeout": 600,
                "required_outputs": ["causal_graph", "algorithm_scores"]
            },
            "causal_analysis": {
                "agent": "causal_analysis",
                "action": "estimate_causal_effects",
                "description": "인과 효과 추정", 
                "timeout": 900,
                "required_outputs": ["causal_estimates", "confidence_intervals"]
            },
            "report_generation": {
                "agent": "orchestrator",
                "action": "generate_final_report",
                "description": "최종 결과 리포트 생성",
                "timeout": 300,
                "required_outputs": ["final_report"]
            }
        }
    
    def create_execution_plan(self, goal: str, context: AgentState) -> List[Dict[str, Any]]:
        """Create execution plan based on state and user-selected mode (no query analysis)"""
        analysis_mode = context.get("analysis_mode")  # 'full_pipeline' | 'data_exploration'
        
        # If mode not selected, ask user via HITL (handled in step())
        if not analysis_mode:
            return []
        
        if analysis_mode == "full_pipeline":
            # Return the unfiltered full pipeline (system + all phases)
            return self.get_full_pipeline_plan()
        
        if analysis_mode == "data_exploration":
            # Minimal plan: only data exploration steps
            plan = []
            for step in self.get_full_pipeline_plan():
                if step["phase"].value == "data_exploration":
                    plan.append(step)
            return plan
        
        return []
    
    def _is_step_completed(self, step: Dict[str, Any], current_state: AgentState) -> bool:
        """Check if a step is already completed based on current state"""
        phase = step["phase"]
        substep = step["substep"]
        
        # Check phase completion
        if phase in current_state.get("completed_phases", []):
            return True
            
        # Check substep completion
        if substep in current_state.get("completed_substeps", []):
            return True
            
        return False
    
    def _are_dependencies_met(self, step: Dict[str, Any], current_state: AgentState) -> bool:
        """Check if step dependencies are met"""
        dependencies = step.get("dependencies", [])
        
        # System components don't have dependencies
        if step.get("is_system_component", False):
            return True
        
        # For data exploration steps, allow them to be planned initially
        if step["phase"].value == "data_exploration":
            # Data exploration steps can be planned initially
            # They will be executed after system components complete
            return True
        
        # For other steps, check explicit dependencies
        for dep in dependencies:
            if dep not in current_state.get("completed_substeps", []):
                return False
        return True
    
    
    def step(self, state: AgentState) -> AgentState:
        """Execute planning step with HITL for prerequisites and mode selection."""
        # If analysis_mode already provided, skip HITL prompts
        if state.get("analysis_mode") is None:
            state["plan_created"] = False

            state["analysis_mode"] = "full_pipeline"
            # decision = interrupt({
            #     "question": "Select analysis mode",
            #     "options": ["full_pipeline", "data_exploration"],
            #     "edit_hint": "Set analysis_mode to 'full_pipeline' or 'data_exploration'"
            # })

        else:
            if state.get("allow_start_without_ground_truth") is None:
                # a) Check required ground-truth inputs at the very first planning step
                df_keys = [
                    "ground_truth_dataframe_path",
                    "ground_truth_dataframe",
                    "gt_df",
                ]
                graph_keys = [
                    "ground_truth_causal_graph_path",
                    "ground_truth_causal_graph",
                    "gt_graph",
                ]
                has_gt_df = any(state.get(k) is not None for k in df_keys)
                has_gt_graph = any(state.get(k) is not None for k in graph_keys)

                if not (has_gt_df and has_gt_graph):
                    state["allow_start_without_ground_truth"] = True

                # If GT df provided, mark exploration as skippable
                if has_gt_df:
                    state["data_exploration_status"] = state.get("data_exploration_status", "skipped")
                    for s in ["table_selection", "table_retrieval", "data_preprocessing"]:
                        state.setdefault("skip_steps", [])
                        if s not in state["skip_steps"]:
                            state["skip_steps"].append(s)
                            
                # If GT graph provided, mark discovery as skippable
                if has_gt_graph:
                    state["causal_discovery_status"] = state.get("causal_discovery_status", "skipped")
                    for s in [
                        "data_profiling",
                        "algorithm_configuration",
                        "run_algorithms_portfolio",
                        "graph_scoring",
                        "graph_evaluation",
                        "ensemble_synthesis",
                    ]:
                        state.setdefault("skip_steps", [])
                        if s not in state["skip_steps"]:
                            state["skip_steps"].append(s)
                
                # Original interrupt-based approach (commented out)
                # if not (has_gt_df and has_gt_graph):
                #     payload = {
                #         "question": "Provide ground-truth inputs or skip",
                #         "required_fields": [
                #             {
                #                 "key": "ground_truth_dataframe_path",
                #                 "desc": "CSV/Parquet path or set allow_start_without_ground_truth=True"
                #             },
                #             {
                #                 "key": "ground_truth_causal_graph_path",
                #                 "desc": "Graph JSON path (e.g., DoWhy graph) or skip"
                #             }
                #         ],
                #         "examples": {
                #             "ground_truth_dataframe_path": "/path/to/data.csv",
                #             "ground_truth_causal_graph_path": "/path/to/graph.json"
                #         }
                #     }
                #     state["check_gt"]=True
                #     decision = interrupt(payload)
                        
        
        # c) Build plan according to the chosen mode (no query analysis)
        plan = self.create_execution_plan(state.get("initial_query", ""), state)
        
        # Update state with new plan
        state["execution_plan"] = plan
        state["plan_created"] = True
        state["total_steps"] = len(plan)
        state["estimated_duration"] = sum(step.get("timeout", 0) for step in plan)
        
        return state
