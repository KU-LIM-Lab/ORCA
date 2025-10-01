# agents/data_explorer/agent.py
"""
Data Explorer Agent implementation using tool registry pattern.

This agent demonstrates how to use the tool registry system
while maintaining the benefits of direct tool access.
"""

from typing import Dict, Any, Optional
from core.base import SpecialistAgent, AgentType, AgentResult
from core.state import AgentState
from monitoring.metrics.collector import MetricsCollector

class DataExplorerAgent(SpecialistAgent):
    """Data Explorer Agent using tool registry pattern"""
    
    def __init__(self, name: str = "data_explorer", config: Optional[Dict[str, Any]] = None, 
                 metrics_collector: Optional[MetricsCollector] = None):
        super().__init__(name, AgentType.SPECIALIST, config, metrics_collector)
        
        # Set domain expertise
        self.set_domain_expertise([
            "table_selection",
            "data_retrieval", 
            "schema_analysis",
            "text2sql_generation"
        ])
    
    def _register_specialist_tools(self) -> None:
        """Register data explorer specific tools"""
        # Register custom tools for data exploration
        self.register_tool(
            "get_table_metadata",
            self._get_table_metadata_tool,
            "Get metadata for a specific table"
        )
        
        self.register_tool(
            "generate_sql_query",
            self._generate_sql_query_tool,
            "Generate SQL query from natural language"
        )
        
        self.register_tool(
            "analyze_table_schema",
            self._analyze_table_schema_tool,
            "Analyze table schema and relationships"
        )
    
    def step(self, state: AgentState) -> AgentState:
        """Execute data exploration step"""
        try:
            # Get the current substep from state
            substep = state.get("current_substep", "table_selection")
            
            if substep == "table_selection":
                return self._select_tables(state)
            elif substep == "table_retrieval":
                return self._retrieve_data(state)
            elif substep == "schema_analysis":
                return self._analyze_schema(state)
            else:
                raise ValueError(f"Unknown substep: {substep}")
                
        except Exception as e:
            state["error"] = str(e)
            state["error_type"] = "data_explorer_error"
            return state
    
    def _select_tables(self, state: AgentState) -> AgentState:
        """Select relevant tables for analysis"""
        try:
            # Use tool registry to get database tool
            db_tool = self.use_tool("database")
            
            # Get available tables
            tables = db_tool.list_tables()
            
            # Use LLM tool to select relevant tables
            llm_tool = self.use_tool("llm")
            query = state.get("initial_query", "")
            
            selection_prompt = f"""
            Based on the user query: "{query}"
            Select the most relevant tables from: {tables}
            
            Return only the table names as a list.
            """
            
            selected_tables = llm_tool.execute(selection_prompt)
            
            # Update state
            state["selected_tables"] = selected_tables
            state["table_selection_completed"] = True
            
            return state
            
        except Exception as e:
            state["error"] = f"Table selection failed: {str(e)}"
            return state
    
    def _retrieve_data(self, state: AgentState) -> AgentState:
        """Retrieve data using text2sql"""
        try:
            # Use custom tool for SQL generation
            sql_query = self.use_tool("generate_sql_query", 
                                    state.get("initial_query", ""),
                                    state.get("selected_tables", []))
            
            # Execute query using database tool
            db_tool = self.use_tool("database")
            results = db_tool.execute_query(sql_query)
            
            # Update state
            state["sql_query"] = sql_query
            state["df_raw"] = results
            state["data_retrieval_completed"] = True
            
            return state
            
        except Exception as e:
            state["error"] = f"Data retrieval failed: {str(e)}"
            return state
    
    def _analyze_schema(self, state: AgentState) -> AgentState:
        """Analyze table schema and relationships"""
        try:
            # Use custom tool for schema analysis
            schema_analysis = self.use_tool("analyze_table_schema",
                                          state.get("selected_tables", []))
            
            # Update state
            state["schema_analysis"] = schema_analysis
            state["schema_analysis_completed"] = True
            
            return state
            
        except Exception as e:
            state["error"] = f"Schema analysis failed: {str(e)}"
            return state
    
    # Custom tool implementations
    def _get_table_metadata_tool(self, table_name: str) -> Dict[str, Any]:
        """Tool: Get table metadata"""
        try:
            from utils.system_init import get_table_metadata
            return get_table_metadata("reef_db", table_name) or {}
        except Exception as e:
            return {"error": str(e)}
    
    def _generate_sql_query_tool(self, query: str, tables: list) -> str:
        """Tool: Generate SQL query from natural language"""
        try:
            # Use existing text2sql logic
            from utils.llm import call_llm
            # ... existing text2sql implementation
            return "SELECT * FROM users LIMIT 10"  # Placeholder
        except Exception as e:
            raise Exception(f"SQL generation failed: {str(e)}")
    
    def _analyze_table_schema_tool(self, tables: list) -> Dict[str, Any]:
        """Tool: Analyze table schema and relationships"""
        try:
            # Use existing schema analysis logic
            from utils.data_prep.metadata import extract_schema
            schema = extract_schema("reef_db")
            return schema
        except Exception as e:
            return {"error": str(e)}
