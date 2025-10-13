# agents/data_explorer/agent.py
"""
Data Explorer Agent implementation using SpecialistAgent pattern.

This agent orchestrates data exploration tasks by leveraging
subgraph agents for table recommendation, text2sql generation, and table exploration.
"""

import json
from typing import Dict, Any, Optional
from core.base import SpecialistAgent, AgentType, AgentState
from monitoring.metrics.collector import MetricsCollector
from agents.data_explorer.table_recommender.agent import TableRecommenderAgent
from agents.data_explorer.text2sql_generator.agent import Text2SQLGeneratorAgent
from agents.data_explorer.table_explorer.agent import TableExplorerAgent
from agents.data_explorer.data_preprocessor.agent import DataPreprocessorAgent


class DataExplorerAgent(SpecialistAgent):
    """Specialist agent for data exploration and analysis."""
    
    def __init__(self, llm: Optional[Any] = None, name: str = "data_explorer", 
                 config: Optional[Dict[str, Any]] = None, 
                 metrics_collector: Optional[MetricsCollector] = None):
        super().__init__(name, AgentType.SPECIALIST, config, metrics_collector)
        self.llm = llm
        
        # 1. 도메인 전문성 설정
        self.set_domain_expertise([
            "table_selection",
            "data_retrieval", 
            "schema_analysis",
            "text2sql_generation",
            "data_preprocessing",
            "sql_query_generation"
        ])
        
        # Set input/output schemas
        self.input_schema = {
            "input": "str",
            "db_id": "str",
            "query_type": "str"
        }
        
        self.output_schema = {
            "selected_tables": "List[str]",
            "sql_query": "str",
            "df_raw": "pandas.DataFrame",
            "df_preprocessed": "pandas.DataFrame",
            "schema_analysis": "Dict[str, Any]",
            "final_output": "Dict[str, Any]"
        }
    
    def get_required_state_keys(self):
        """Return required state keys for data exploration."""
        return ["input", "db_id"]
    
    def _normalize_table_list(self, tables: Any) -> list:
        """Normalize recommended table objects to a list of table name strings."""
        normalized: list = []
        if not tables:
            return normalized
        for item in tables:
            if isinstance(item, str):
                normalized.append(item)
                continue
            if isinstance(item, dict):
                candidate = (
                    item.get("table")
                    or item.get("table_name")
                    or item.get("name")
                    or item.get("id")
                )
                if isinstance(candidate, str):
                    normalized.append(candidate)
                    continue
            for attr in ("table", "table_name", "name", "id"):
                if hasattr(item, attr):
                    value = getattr(item, attr)
                    if isinstance(value, str):
                        normalized.append(value)
                        break
            else:
                normalized.append(str(item))
        return normalized
    
    def _register_specialist_tools(self) -> None:
        """Register data explorer specific tools"""
        # 2. 커스텀 도구들 등록
        self.register_tool(
            "table_selection",
            self._table_recommendation_tool,
            "Recommend relevant tables for analysis"
        )
        
        self.register_tool(
            "table_retrieval",
            self._text2sql_generation_tool,
            "Generate SQL query from natural language"
        )
        
        self.register_tool(
            "table_exploration",
            self._table_exploration_tool,
            "Explore and analyze table schema"
        )
        
        self.register_tool(
            "data_preprocessing",
            self._data_preprocessing_tool,
            "Preprocess raw data for analysis"
        )
    
    def step(self, state: AgentState) -> AgentState:
        """Execute one step of the data exploration process"""
        # 3. 메인 실행 로직
        current_substep = state.get("current_substep", "full_pipeline")
        
        if current_substep == "table_selection":
            return self._execute_table_recommendation(state)
        elif current_substep == "table_retrieval":
            return self._execute_text2sql_generation(state)
        elif current_substep == "table_exploration":
            return self._execute_table_exploration(state)
        elif current_substep == "data_preprocessing":
            return self._execute_data_preprocessing(state)
        elif current_substep == "full_pipeline":
            return self._execute_full_pipeline(state)
        else:
            raise ValueError(f"Unknown substep: {current_substep}")
    
    # 4. 커스텀 도구 구현
    def _table_recommendation_tool(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend relevant tables for analysis"""
        try:
            if not self.llm:
                raise ValueError("LLM is required for table recommendation")
            
            # Ensure required fields
            if not state.get("initial_query"):
                raise ValueError("Input query is required for table recommendation")
            if not state.get("db_id"):
                raise ValueError("Database ID is required for table recommendation")
            
            # Initialize TableRecommenderAgent
            recommender = TableRecommenderAgent(llm=self.llm, name="table_recommender_internal")
            
            # Prepare state for table recommender
            recommender_state = {
                "db_id": state.get("db_id"),
                "input": state.get("initial_query"),
                "input_type": "text"
            }
            
            # Execute table recommendation
            result = recommender.execute(recommender_state)
            
            if result.success and result.data:
                recommended_raw = result.data.get("recommended_tables", [])
                recommended_tables = self._normalize_table_list(recommended_raw)
                return {
                    # "recommended_tables": result.data.get("recommended_tables", []),
                    "recommended_tables": recommended_tables,
                    "objective_summary": result.data.get("objective_summary", ""),
                    "erd_image_path": result.data.get("erd_image_path", ""),
                    "success": True
                }
            else:
                return {"error": result.error or "Table recommendation failed", "success": False}
            
        except Exception as e:
            self.on_event("table_recommendation_error", error=str(e))
            return {"error": str(e), "success": False}
    
    def _text2sql_generation_tool(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate SQL query from natural language"""
        try:
            if not self.llm:
                raise ValueError("LLM is required for text2sql generation")
            
            # Ensure required fields
            if not state.get("initial_query"):
                raise ValueError("Input query are required for text2sql generation")
            if not state.get("db_id"):
                raise ValueError("Database ID is required for text2sql generation")
            
            # Initialize Text2SQLGeneratorAgent
            text2sql_agent = Text2SQLGeneratorAgent(llm=self.llm, name="text2sql_internal")
            
            # Prepare state for text2sql generator
            selected_tables = self._normalize_table_list(state.get("selected_tables", []))
            text2sql_state = {
                "db_id": state.get("db_id"),
                "query": state.get("initial_query"),
                "messages": [],
                "evidence": f"Selected tables: {', '.join(selected_tables)}"
            }
            
            # Execute SQL generation
            result = text2sql_agent.execute(text2sql_state)
            
            if result.success and result.data:
                # Try to propagate columns for proper DataFrame construction downstream
                out = {
                    "final_sql": result.data.get("final_sql", ""),
                    "result": result.data.get("result", []),
                    "llm_review": result.data.get("llm_review", ""),
                    "success": True
                }
                if result.data.get("columns"):
                    out["columns"] = result.data.get("columns")
                return out
            else:
                return {"error": result.error or "Text2SQL generation failed", "success": False}
            
        except Exception as e:
            self.on_event("text2sql_generation_error", error=str(e))
            return {"error": str(e), "success": False}
    
    def _table_exploration_tool(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Explore and analyze table schema"""
        try:
            if not self.llm:
                raise ValueError("LLM is required for table exploration")
            
            # Ensure required fields
            selected_tables = state.get("selected_tables", [])
            if not selected_tables:
                raise ValueError("Selected tables are required for table exploration")
            
            # Initialize TableExplorerAgent
            explorer = TableExplorerAgent(llm=self.llm, name="table_explorer_internal")
            
            # Analyze each selected table
            table_analyses = {}
            
            for table_name in selected_tables:
                try:
                    # Prepare state for table explorer
                    explorer_state = {
                        "db_id": state.get("db_id"),
                        "input": table_name
                    }
                    
                    # Execute table exploration
                    result = explorer.execute(explorer_state)
                    
                    if result.success and result.data:
                        table_analyses[table_name] = {
                            "table_analysis": result.data.get("table_analysis", {}),
                            "related_tables": result.data.get("related_tables", {}),
                            "recommended_analysis": result.data.get("recommended_analysis", []),
                            "final_output": result.data.get("final_output", {})
                        }
                    else:
                        table_analyses[table_name] = {
                            "error": f"Table exploration failed: {result.error}"
                        }
                        
                except Exception as table_error:
                    table_analyses[table_name] = {
                        "error": f"Failed to analyze table {table_name}: {str(table_error)}"
                    }
            
            return {
                "schema_analysis": table_analyses,
                "success": True
            }
            
        except Exception as e:
            self.on_event("table_exploration_error", error=str(e))
            return {"error": str(e), "success": False}
    
    def _data_preprocessing_tool(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess raw data for analysis"""
        try:
            # Ensure required fields
            df_raw = state.get("df_raw")
            if df_raw is None and not state.get("sql_query"):
                raise ValueError("Raw data or sql_query is required for preprocessing")
            
            # Initialize DataPreprocessorAgent
            preprocessor = DataPreprocessorAgent(name="data_preprocessor_internal")
            
            # Prepare state for data preprocessor
            preprocessor_state = {
                "db_id": state.get("db_id"),
                "final_sql": state.get("sql_query", ""),
                "df_raw": df_raw,
                "persist_to_redis": state.get("persist_to_redis", True),
                "fetch_only": state.get("fetch_only", True),
                "steps": ["fetch", "clean_nulls", "type_cast", "derive_features", "impute", "encode", "scale", "split", "report"]
            }
            
            # Execute data preprocessing
            result = preprocessor.execute(preprocessor_state)
            
            if result.success and result.data:
                response = {
                    "preprocess_report": result.data.get("preprocess_report", {}),
                    "column_stats": result.data.get("column_stats", {}),
                    "feature_map": result.data.get("feature_map", {}),
                    "warnings": result.data.get("warnings", []),
                    "df_redis_key": result.data.get("df_redis_key"),
                    "df_shape": result.data.get("df_shape"),
                    "success": True
                }
                # Only include df_preprocessed if it exists and fetch_only is False
                if not state.get("fetch_only", True) and result.data.get("df_preprocessed") is not None:
                    response["df_preprocessed"] = result.data.get("df_preprocessed")
                return response
            else:
                return {"error": result.error or "Data preprocessing failed", "success": False}
            
        except Exception as e:
            self.on_event("data_preprocessing_error", error=str(e))
            return {"error": str(e), "success": False}
    
    # 5. 단계별 실행 메서드
    def _execute_table_recommendation(self, state: AgentState) -> AgentState:
        """Execute table recommendation step"""
        print("[DATA_EXPLORER] Step: Table recommendation...")
        
        # 커스텀 도구 사용
        result = self.use_tool("table_selection", state)
        
        if result.get("success"):
            # 상태 업데이트
            state["selected_tables"] = result.get("recommended_tables", "")
            state["objective_summary"] = result.get("objective_summary", "")
            state["erd_image_path"] = result.get("erd_image_path", "")
            state["table_recommendation_completed"] = True
        else:
            state["error"] = result.get("error", "Table recommendation failed")
        
        return state
    
    def _execute_text2sql_generation(self, state: AgentState) -> AgentState:
        """Execute text2sql generation step"""
        print("[DATA_EXPLORER] Step: Text2SQL generation...")
        
        # 커스텀 도구 사용
        result = self.use_tool("table_retrieval", state)
        
        if result.get("success"):
            # 상태 업데이트
            state["sql_query"] = result.get("final_sql", "")
            state["df_raw"] = result.get("result", [])
            # Keep columns if provided for DataFrame coercion in fetch_node
            if result.get("columns"):
                state["columns"] = result.get("columns")
            state["llm_review"] = result.get("llm_review", "")
            state["text2sql_generation_completed"] = True
        else:
            state["error"] = result.get("error", "Text2SQL generation failed")
        
        return state
    
    def _execute_table_exploration(self, state: AgentState) -> AgentState:
        """Execute table exploration step"""
        print("[DATA_EXPLORER] Step: Table exploration...")
        
        # 커스텀 도구 사용
        result = self.use_tool("table_exploration", state)
        
        if result.get("success"):
            # 상태 업데이트
            state["schema_analysis"] = result.get("schema_analysis", {})
            state["table_exploration_completed"] = True
            
            # Extract summary information
            all_related_tables = set()
            analysis_recommendations = []
            
            for table_name, analysis in state["schema_analysis"].items():
                if "related_tables" in analysis:
                    all_related_tables.update(analysis["related_tables"].keys())
                if "recommended_analysis" in analysis:
                    analysis_recommendations.extend(analysis["recommended_analysis"])
            
            state["all_related_tables"] = list(all_related_tables)
            state["analysis_recommendations"] = analysis_recommendations
        else:
            state["error"] = result.get("error", "Table exploration failed")
        
        return state
    
    def _execute_data_preprocessing(self, state: AgentState) -> AgentState:
        """Execute data preprocessing step"""
        print("[DATA_EXPLORER] Step: Data preprocessing...")
        
        # 커스텀 도구 사용
        result = self.use_tool("data_preprocessing", state)
        
        if result.get("success"):
            # 상태 업데이트 - only set non-DataFrame fields to avoid serialization issues
            if result.get("df_preprocessed") is not None:
                state["df_preprocessed"] = result.get("df_preprocessed")
            state["preprocess_report"] = result.get("preprocess_report", "")
            state["column_stats"] = result.get("column_stats", {})
            state["feature_map"] = result.get("feature_map", {})
            state["warnings"] = result.get("warnings", [])
            # New Redis-based fields
            if result.get("df_redis_key"):
                state["df_redis_key"] = result.get("df_redis_key")
            if result.get("df_shape"):
                state["df_shape"] = result.get("df_shape")
            state["data_preprocessing_completed"] = True
        else:
            state["error"] = result.get("error", "Data preprocessing failed")
        
        return state
    
    def _execute_full_pipeline(self, state: AgentState) -> AgentState:
        """Execute full data exploration pipeline"""
        print("[DATA_EXPLORER] Executing full pipeline...")
        
        try:
            # Validate input
            self.validate_state(state)
            
            # Check if we have required inputs
            input_query = state.get("input")
            if not input_query:
                raise ValueError("Input query is required for data exploration")
            
            db_id = state.get("db_id")
            if not db_id:
                raise ValueError("Database ID is required for data exploration")
            
            # Ensure LLM is available
            if not self.llm:
                raise ValueError("LLM is required for data exploration")
            
            # Execute pipeline steps
            state = self._execute_table_recommendation(state)
            if state.get("error"):
                return state
            
            state = self._execute_text2sql_generation(state)
            if state.get("error"):
                return state
            
            state = self._execute_table_exploration(state)
            if state.get("error"):
                return state
            
            state = self._execute_data_preprocessing(state)
            if state.get("error"):
                return state
            
            print("[DATA_EXPLORER] Pipeline completed successfully!")
            return state
            
        except Exception as e:
            self.on_event("data_exploration_pipeline_error", error=str(e))
            state["error"] = str(e)
            return state
    
    def get_capabilities(self) -> list:
        """Return data exploration capabilities."""
        return [
            "table_selection",
            "table_retrieval",
            "table_exploration",
            "data_preprocessing",
            "schema_analysis",
            "sql_query_generation"
        ]