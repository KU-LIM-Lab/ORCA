# agents/causal_analysis/agent.py
from typing import Any, Dict, Optional
from core.base import SpecialistAgent, AgentType, AgentState
from monitoring.metrics.collector import MetricsCollector
from utils.redis_df import load_df_parquet
from agents.causal_analysis.nodes.dowhy_analysis import build_dowhy_analysis_node
from agents.causal_analysis.nodes.config_selection import build_config_selection_node
from agents.causal_analysis.nodes.generate_answer import build_generate_answer_node
from agents.causal_analysis.nodes.parse_question import build_parse_question_node


class CausalAnalysisAgent(SpecialistAgent):
    """Specialist agent for causal analysis and inference."""
    
    def __init__(self, llm: Optional[Any] = None, name: str = "causal_analysis", 
                 config: Optional[Dict[str, Any]] = None, 
                 metrics_collector: Optional[MetricsCollector] = None):
        super().__init__(name, AgentType.SPECIALIST, config, metrics_collector)
        self.llm = llm
        
        # 1. 도메인 전문성 설정
        self.set_domain_expertise([
            "causal_analysis",
            "doWhy",
            "treatment_effect_estimation",
            "confounder_identification",
            "causal_graph_construction"
        ])
        
        # Set input/output schemas
        self.input_schema = {
            "df_preprocessed": "pandas.DataFrame",
            "initial_query": "str",
            "db_id": "str",
            "expression_dict": "Dict[str, str]"
        }
        
        self.output_schema = {
            "causal_estimate": "CausalEstimate",
            "ate": "float",
            "confidence_interval": "Tuple[float, float]",
            "refutation_result": "str",
            "final_answer": "str"
        }
    
    def get_required_state_keys(self):
        """Return required state keys for causal analysis."""
        return ["df_preprocessed", "initial_query"]
    
    def _register_specialist_tools(self) -> None:
        """Register causal analysis specific tools"""
        self.register_tool(
            "parse_question",
            self._parse_question_tool,
            "Parse natural language question to identify causal variables"
        )
        
        self.register_tool(
            "config_selection",
            self._config_selection_tool,
            "Select causal analysis configuration and strategy"
        )
        
        self.register_tool(
            "dowhy_analysis",
            self._dowhy_analysis_tool,
            "Perform causal analysis using DoWhy"
        )
        
        self.register_tool(
            "generate_answer",
            self._generate_answer_tool,
            "Generate human-readable explanation of results"
        )
    
    def step(self, state: AgentState) -> AgentState:
        """Execute one step of the causal analysis process"""
        current_substep = state.get("current_substep", "full_pipeline")
        
        # Map execution plan substep names to internal method names
        if current_substep == "parse_question":
            return self._execute_parse_question(state)
        elif current_substep == "select_configuration":
            return self._execute_config_selection(state)
        elif current_substep == "effect_estimation":
            return self._execute_dowhy_analysis(state)
        elif current_substep == "interpretation":
            return self._execute_generate_answer(state)
        elif current_substep == "full_pipeline":
            return self._execute_full_pipeline(state)
        else:
            raise ValueError(f"Unknown substep: {current_substep}")
    
    def _parse_question_tool(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Parse natural language question to identify causal variables"""
        try:
            if not self.llm:
                raise ValueError("LLM is required for question parsing")
            
            # Ensure required fields for parsing
            if not state.get("db_id"):
                state["db_id"] = "reef_db"  # Default database
            if not state.get("expression_dict"):
                state["expression_dict"] = {}
            
            # Build and invoke parse_question node
            parse_question_node = build_parse_question_node(self.llm)
            result_state = parse_question_node.invoke(state)
            
            return {
                "parsed_query": result_state.get("parsed_query"),
                "table_schema_str": result_state.get("table_schema_str"),
                "success": True
            }
            
        except Exception as e:
            self.on_event("parse_question_error", error=str(e))
            return {"error": str(e), "success": False}
    
    def _config_selection_tool(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Select causal analysis configuration and strategy"""
        try:
            if not self.llm:
                raise ValueError("LLM is required for config selection")
            
            # Ensure required fields
            if not state.get("parsed_query"):
                raise ValueError("Parsed query is required for config selection")
            
            df_redis_key = state.get("df_redis_key")
            df_preprocessed = state.get("df_preprocessed")
            if df_preprocessed is None and df_redis_key:
                df_preprocessed = load_df_parquet(df_redis_key)
            
            if df_preprocessed is None:
                raise ValueError("Preprocessed data is required for config selection")
                        
            # Build and invoke config_selection node
            config_selection_node = build_config_selection_node(self.llm)
            result_state = config_selection_node.invoke(state)
            
            return {
                "strategy": result_state.get("strategy"),
                "success": True
            }
            
        except Exception as e:
            self.on_event("config_selection_error", error=str(e))
            return {"error": str(e), "success": False}
    
    def _dowhy_analysis_tool(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Perform causal analysis using DoWhy"""
        try:
            # Ensure required fields
            if not state.get("strategy"):
                raise ValueError("Strategy is required for DoWhy analysis")
            if not state.get("parsed_query"):
                raise ValueError("Parsed query is required for DoWhy analysis")
            
            redis_key = state.get("df_redis_key")
            if redis_key:
                try:
                    df = load_df_parquet(redis_key)
                    
                except Exception as e:
                    print(f"⚠️ Failed to load DataFrame from Redis key {redis_key}: {e}")
                
            # Build and invoke dowhy_analysis node
            dowhy_analysis_node = build_dowhy_analysis_node()
            result_state = dowhy_analysis_node.invoke(state)
            
            return {
                "causal_model": result_state.get("causal_model"),
                "causal_estimand": result_state.get("causal_estimand"),
                "causal_estimate": result_state.get("causal_estimate"),
                "causal_effect_ate": result_state.get("causal_effect_ate"),
                "causal_effect_ci": result_state.get("causal_effect_ci"),
                "refutation_result": result_state.get("refutation_result"),
                "success": True
            }
            
        except Exception as e:
            self.on_event("dowhy_analysis_error", error=str(e))
            return {"error": str(e), "success": False}
    
    def _generate_answer_tool(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate human-readable explanation of results"""
        try:
            if not self.llm:
                raise ValueError("LLM is required for answer generation")
            
            # Ensure required fields
            if not state.get("strategy"):
                raise ValueError("Strategy is required for answer generation")
            if not state.get("causal_estimate"):
                raise ValueError("Causal estimate is required for answer generation")
            
            # Build and invoke generate_answer node
            generate_answer_node = build_generate_answer_node(self.llm)
            result_state = generate_answer_node.invoke(state)
            
            return {
                "final_answer": result_state.get("final_answer"),
                "success": True
            }
            
        except Exception as e:
            self.on_event("generate_answer_error", error=str(e))
            return {"error": str(e), "success": False}
    
    # 5. 단계별 실행 메서드
    def _execute_parse_question(self, state: AgentState) -> AgentState:
        """Execute parse question step"""
        print("[CAUSAL] Step: Parsing question...")
        
        # 커스텀 도구 사용
        result = self.use_tool("parse_question", state)
        
        if result.get("success"):
            # 상태 업데이트
            state["parsed_query"] = result.get("parsed_query")
            state["table_schema_str"] = result.get("table_schema_str")
            state["parse_question_completed"] = True
            
            # Request HITL for parse question review if interactive mode
            if state.get("interactive", False):
                payload = {
                    "step": "parse_question",
                    "phase": "causal_analysis",
                    "description": "Question parsed. Review the identified variables before configuration.",
                    "decisions": ["approve", "edit", "rerun", "abort"]
                }
                state = self.request_hitl(state, payload=payload, hitl_type="parse_question_review")
                return state
        else:
            state["error"] = result.get("error", "Parse question failed")
        
        return state
    
    def _execute_config_selection(self, state: AgentState) -> AgentState:
        """Execute config selection step"""
        print("[CAUSAL] Step: Selecting configuration...")
        
        # 커스텀 도구 사용
        result = self.use_tool("config_selection", state)
        
        if result.get("success"):
            # 상태 업데이트
            state["strategy"] = result.get("strategy")
            state["config_selection_completed"] = True
            
            # Request HITL for strategy review if interactive mode
            if state.get("interactive", False):
                payload = {
                    "step": "select_configuration",
                    "phase": "causal_analysis",
                    "description": "Causal analysis strategy configured. Review the treatment, outcome, confounders, and estimation method.",
                    "decisions": ["approve", "edit", "rerun", "abort"]
                }
                state = self.request_hitl(state, payload=payload, hitl_type="strategy_review")
                return state
        else:
            state["error"] = result.get("error", "Config selection failed")
        
        return state
    
    def _execute_dowhy_analysis(self, state: AgentState) -> AgentState:
        """Execute DoWhy analysis step"""
        print("[CAUSAL] Step: Running DoWhy analysis...")
        
        # 커스텀 도구 사용
        result = self.use_tool("dowhy_analysis", state)
        
        if result.get("success"):
            # 상태 업데이트
            state.update({
                "causal_model": result.get("causal_model"),
                "causal_estimand": result.get("causal_estimand"),
                "causal_estimate": result.get("causal_estimate"),
                "causal_effect_ate": result.get("causal_effect_ate"),
                "causal_effect_ci": result.get("causal_effect_ci"),
                "refutation_result": result.get("refutation_result")
            })
            state["dowhy_analysis_completed"] = True
        else:
            state["error"] = result.get("error", "DoWhy analysis failed")
        
        return state
    
    def _execute_generate_answer(self, state: AgentState) -> AgentState:
        """Execute generate answer step"""
        print("[CAUSAL] Step: Generating answer...")
        
        # 커스텀 도구 사용
        result = self.use_tool("generate_answer", state)
        
        if result.get("success"):
            # 상태 업데이트
            state["final_answer"] = result.get("final_answer")
            state["generate_answer_completed"] = True
        else:
            # Fallback to simple answer generation
            state["final_answer"] = self._generate_simple_answer(state)
            state["generate_answer_completed"] = True
        
        return state
    
    def _execute_full_pipeline(self, state: AgentState) -> AgentState:
        """Execute full causal analysis pipeline"""
        print("[CAUSAL] Executing full pipeline...")
        
        try:
            # Validate input
            self.validate_state(state)
            
            # Check if we have preprocessed data
            df_preprocessed = state.get("df_preprocessed")
            if df_preprocessed is None:
                raise ValueError("df_preprocessed is required for causal analysis")
            
            # Check if we have input question (initial_query)
            input_question = state.get("initial_query")
            if not input_question:
                raise ValueError("initial_query is required for causal analysis")
            
            # Ensure LLM is available
            if not self.llm:
                raise ValueError("LLM is required for causal analysis")
            
            # Execute pipeline steps
            state = self._execute_parse_question(state)
            if state.get("error"):
                return state
            
            state = self._execute_config_selection(state)
            if state.get("error"):
                return state
            
            state = self._execute_dowhy_analysis(state)
            if state.get("error"):
                return state
            
            state = self._execute_generate_answer(state)
            if state.get("error"):
                return state
            
            print("[CAUSAL] Pipeline completed successfully!")
            return state
            
        except Exception as e:
            self.on_event("causal_analysis_pipeline_error", error=str(e))
            state["error"] = str(e)
            return state
    
    def _generate_simple_answer(self, state: Dict[str, Any]) -> str:
        """Generate simple human-readable final answer as fallback."""
        ate = state.get("causal_effect_ate", 0)
        treatment = state.get("treatment_variable", "treatment")
        outcome = state.get("outcome_variable", "outcome")
        
        answer = f"""
Causal Analysis Results:

Treatment Variable: {treatment}
Outcome Variable: {outcome}
Average Treatment Effect (ATE): {ate:.4f}

"""
        
        if state.get("refutation_result"):
            answer += f"Refutation Test Result: {state['refutation_result']}\n"
        
        return answer.strip()
    
    def get_capabilities(self) -> list:
        """Return causal analysis capabilities."""
        return [
            "causal_effect_estimation",
            "confounder_identification", 
            "treatment_effect_analysis",
            "refutation_testing",
            "dowhy_integration"
        ]
