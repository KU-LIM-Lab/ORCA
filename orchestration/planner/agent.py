# orchestration/planner/agent.py
from typing import Dict, Any, List, Optional
from core.base import OrchestratorAgent
from core.state import AgentState
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
            "causal_inference",
            "report_generation"
        ]
        
        # Step dependencies (what needs to be completed before each step)
        self.step_dependencies = {
            "data_exploration": [],
            "causal_discovery": ["data_exploration"],
            "causal_inference": ["causal_discovery"],
            "report_generation": ["causal_inference"]
        }
        
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
            "causal_inference": {
                "agent": "causal_inference",
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
        """Create execution plan - defaults to full pipeline with state-based skipping"""
        # Get the full pipeline plan
        full_plan = self.get_full_pipeline_plan()
        
        # Determine entry point based on current state
        entry_point = self.determine_entry_point(context)
        
        # Filter plan to start from entry point
        filtered_plan = []
        start_adding = False
        
        for step in full_plan:
            if step["phase"] == entry_point:
                start_adding = True
            
            if start_adding:
                # Check if step is already completed
                if self._is_step_completed(step, context):
                    continue
                    
                # Check if dependencies are met
                if not self._are_dependencies_met(step, context):
                    continue
                    
                filtered_plan.append(step)
        
        return filtered_plan
    
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
        
        for dep in dependencies:
            # Check if dependency substep is completed
            if dep not in current_state.get("completed_substeps", []):
                return False
        return True
    
    
    def step(self, state: AgentState) -> AgentState:
        """Execute planning step"""
        query = state.get("initial_query", "")
        
        # Create execution plan
        plan = self.create_execution_plan(query, state)
        
        # Update state with new plan
        state["execution_plan"] = plan
        state["plan_created"] = True
        state["total_steps"] = len(plan)
        state["estimated_duration"] = sum(step.get("timeout", 0) for step in plan)
        
        return state
