# core/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Callable, Optional, List, Union, Type
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import asyncio
import json
from monitoring.metrics.collector import MetricsCollector, track_execution_time, track_memory_usage, record_metric, MetricType
from core.state import AgentState, PipelinePhase, ExecutionStatus

class AgentType(Enum):
    """Agent type definitions"""
    ORCHESTRATOR = "orchestrator"
    SPECIALIST = "specialist"
    SUBGRAPH = "subgraph"
    TOOL = "tool"

class AgentStatus(Enum):
    """Agent status definitions"""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"

@dataclass
class AgentResult:
    """Standardized agent execution result"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    execution_time: float = 0.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}

class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(
        self, 
        name: str, 
        agent_type: AgentType,
        config: Optional[Dict[str, Any]] = None,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        self.name = name
        self.agent_type = agent_type
        self.config = config or {}
        self.memory = None
        self.logger = print
        self.status = AgentStatus.IDLE
        self.execution_history: List[AgentResult] = []
        self.metrics_collector = metrics_collector

    def reset(self, **kwargs) -> None:
        """Initialize session (memory/cache/local state)"""
        self.status = AgentStatus.IDLE
        self.execution_history.clear()
        self.on_event("reset", kwargs=kwargs)

    def validate_state(self, state: Dict[str, Any]) -> None:
        """Validate required keys and types - runtime protection"""
        if not isinstance(state, dict):
            raise TypeError("state must be a dict")
        
        # Validate agent-specific required keys
        required_keys = self.get_required_state_keys()
        for key in required_keys:
            if key not in state:
                raise KeyError(f"Required state key '{key}' not found")

    def get_required_state_keys(self) -> List[str]:
        """Return required state keys for this agent (can be overridden)"""
        return []

    def on_event(self, event: str, **kwargs) -> None:
        """before/after/error/retry hooks. Default is logger call"""
        if self.logger:
            self.logger({
                "agent": self.name, 
                "type": self.agent_type.value,
                "status": self.status.value,
                "event": event, 
                "timestamp": datetime.now().isoformat(),
                **kwargs
            })


    async def execute_async(self, state: AgentState) -> AgentResult:
        """Asynchronous execution (can be overridden)"""
        with track_execution_time(self.name, {"agent_type": self.agent_type.value}):
            start_time = datetime.now()
            self.status = AgentStatus.RUNNING
            
            try:
                self.validate_state(state)
                self.on_event("execution_started", state=state)
                
                result_data = await self.step_async(state)
                
                execution_time = (datetime.now() - start_time).total_seconds()
                result = AgentResult(
                    success=True,
                    data=result_data,
                    execution_time=execution_time
                )
                
                self.status = AgentStatus.COMPLETED
                self.on_event("execution_completed", result=result)
                
                # Record success metrics
                if self.metrics_collector:
                    self.metrics_collector.record_execution_time(self.name, execution_time)
                
            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                result = AgentResult(
                    success=False,
                    error=str(e),
                    execution_time=execution_time
                )
                
                self.status = AgentStatus.FAILED
                self.on_event("execution_failed", error=str(e), result=result)
                
                # Record error metrics
                if self.metrics_collector:
                    self.metrics_collector.record_error(self.name, type(e).__name__, {"error": str(e)})
                
            finally:
                self.execution_history.append(result)
                
            return result

    def execute(self, state: AgentState) -> AgentResult:
        """Synchronous execution"""
        return asyncio.run(self.execute_async(state))

    @abstractmethod
    def step(self, state: AgentState) -> AgentState:
        """Execute one agent step: state -> state (synchronous)"""
        ...

    async def step_async(self, state: AgentState) -> AgentState:
        """Execute one agent step: state -> state (asynchronous)"""
        return self.step(state)

    def get_capabilities(self) -> List[str]:
        """Return list of agent capabilities"""
        return []

    def use_tool(self, tool_name: str, *args, **kwargs) -> Any:
        """
        Execute a tool by name with monitoring.
        
        This provides a simple interface for agents to access tools
        while maintaining monitoring and error handling.
        """
        try:
            from utils.tools import tool_registry
            
            # Record tool usage for monitoring
            self.on_event("tool_used", tool_name=tool_name, args=args, kwargs=kwargs)
            
            # Execute tool
            result = tool_registry.execute(tool_name, *args, **kwargs)
            
            # Record success
            self.on_event("tool_success", tool_name=tool_name, result_type=type(result).__name__)
            
            return result
            
        except Exception as e:
            # Record error
            self.on_event("tool_error", tool_name=tool_name, error=str(e))
            raise

    def register_tool(self, tool_name: str, tool_func: Callable, description: str = "") -> None:
        """
        Register a custom tool for this agent.
        
        Args:
            tool_name: Name of the tool
            tool_func: Function to execute
            description: Description of what the tool does
        """
        try:
            from utils.tools import tool_registry, BaseTool
            
            # Create a simple tool wrapper
            class SimpleTool(BaseTool):
                def __init__(self, name, func, description):
                    super().__init__(name, description)
                    self.func = func
                
                def execute(self, *args, **kwargs):
                    return self.func(*args, **kwargs)
            
            tool = SimpleTool(tool_name, tool_func, description)
            tool_registry.register(tool)
            
            self.on_event("tool_registered", tool_name=tool_name)
            
        except Exception as e:
            self.on_event("tool_registration_error", tool_name=tool_name, error=str(e))
            raise

    def to_dict(self) -> Dict[str, Any]:
        """Serialize agent state to dictionary"""
        return {
            "name": self.name,
            "type": self.agent_type.value,
            "status": self.status.value,
            "capabilities": self.get_capabilities(),
            "config": self.config,
            "execution_count": len(self.execution_history)
        }

    def requires_hitl(self, state: AgentState, phase: PipelinePhase, substep: str) -> bool:
        """Check if this agent requires HITL at the given phase/substep"""
        # Override in subclasses to define HITL requirements
        return False


class OrchestratorAgent(BaseAgent):
    """Orchestrator agent (Supervisor)"""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None, metrics_collector: Optional[MetricsCollector] = None):
        super().__init__(name, AgentType.ORCHESTRATOR, config, metrics_collector)
        self.execution_plan: List[Dict[str, Any]] = []
        
        
    def create_execution_plan(self, goal: str, context: AgentState) -> List[Dict[str, Any]]:
        """Create execution plan (must be overridden)"""
        raise NotImplementedError("Subclasses must implement create_execution_plan")
        

    def get_full_pipeline_plan(self) -> List[Dict[str, Any]]:
        """
        Get the full pipeline plan structure.
        
        This defines the complete pipeline with all phases and sub-steps.

        """
        return [
            # Data Exploration Phase
            {
                "phase": PipelinePhase.DATA_EXPLORATION,
                "substep": "table_selection",
                "agent": "data_explorer",
                "action": "select_tables",
                "description": "Select relevant tables for analysis",
                "required_state_keys": ["schema_info", "table_metadata"],
                "timeout": 120,
                "hitl_required": True
            },
            {
                "phase": PipelinePhase.DATA_EXPLORATION,
                "substep": "table_retrieval",
                "agent": "data_explorer",
                "action": "retrieve_data",
                "description": "Retrieve data via text2sql",
                "required_state_keys": ["selected_tables"],
                "timeout": 180,
                "hitl_required": True
            },
            {
                "phase": PipelinePhase.DATA_EXPLORATION,
                "substep": "schema_analysis",
                "agent": "data_explorer",
                "action": "analyze_schema",
                "description": "Analyze schema and relationships",
                "required_state_keys": ["selected_tables"],
                "timeout": 120,
                "hitl_required": True
            },
            
            # Causal Discovery Phase
            {
                "phase": PipelinePhase.CAUSAL_DISCOVERY,
                "substep": "data_profiling",
                "agent": "causal_discovery",
                "action": "profile_data",
                "description": "Profile data characteristics and generate qualitative summary",
                "required_state_keys": ["df_preprocessed"],
                "timeout": 300,
                "hitl_required": True
            },
            {
                "phase": PipelinePhase.CAUSAL_DISCOVERY,
                "substep": "algorithm_tiering",
                "agent": "causal_discovery",
                "action": "tier_algorithms",
                "description": "Tier algorithms based on data profile compatibility",
                "required_state_keys": ["data_profiling_completed", "data_profile"],
                "timeout": 60,
                "hitl_required": True
            },
            {
                "phase": PipelinePhase.CAUSAL_DISCOVERY,
                "substep": "run_algorithms_portfolio",
                "agent": "causal_discovery",
                "action": "run_algorithm_portfolio",
                "description": "Run algorithms from all tiers in parallel",
                "required_state_keys": ["algorithm_tiering_completed", "algorithm_tiers"],
                "timeout": 600,
                "hitl_required": True
            },
            {
                "phase": PipelinePhase.CAUSAL_DISCOVERY,
                "substep": "candidate_pruning",
                "agent": "causal_discovery",
                "action": "prune_candidates",
                "description": "Prune candidates using CI testing and structural consistency",
                "required_state_keys": ["run_algorithms_portfolio_completed", "algorithm_results"],
                "timeout": 180,
                "hitl_required": True
            },
            {
                "phase": PipelinePhase.CAUSAL_DISCOVERY,
                "substep": "scorecard_evaluation",
                "agent": "causal_discovery",
                "action": "evaluate_scorecard",
                "description": "Evaluate candidates using composite scorecard",
                "required_state_keys": ["candidate_pruning_completed", "pruned_candidates"],
                "timeout": 120,
                "hitl_required": True
            },
            {
                "phase": PipelinePhase.CAUSAL_DISCOVERY,
                "substep": "ensemble_synthesis",
                "agent": "causal_discovery",
                "action": "synthesize_ensemble",
                "description": "Synthesize ensemble with PAG-like and DAG outputs",
                "required_state_keys": ["scorecard_evaluation_completed", "top_candidates"],
                "timeout": 180,
                "hitl_required": True
            },
            
            # Causal Inference Phase
            {
                "phase": PipelinePhase.CAUSAL_INFERENCE,
                "substep": "select_configuration",
                "agent": "causal_inference",
                "action": "select_config",
                "description": "Select inference configuration",
                "required_state_keys": ["selected_graph", "df_preprocessed"],
                "timeout": 120,
                "hitl_required": True
            },
            {
                "phase": PipelinePhase.CAUSAL_INFERENCE,
                "substep": "effect_estimation",
                "agent": "causal_inference",
                "action": "estimate_effects",
                "description": "Estimate causal effects",
                "required_state_keys": ["treatment_variable", "outcome_variable", "selected_graph"],
                "timeout": 300,
                "hitl_required": False
            },
            {
                "phase": PipelinePhase.CAUSAL_INFERENCE,
                "substep": "interpretation",
                "agent": "causal_inference",
                "action": "interpret_results",
                "description": "Interpret and validate results",
                "required_state_keys": ["causal_estimates"],
                "timeout": 180,
                "hitl_required": True
            },
            
            # Report Generation
            {
                "phase": PipelinePhase.REPORT_GENERATION,
                "substep": "generate_report",
                "agent": "report_generator",
                "action": "generate_report",
                "description": "Generate final analysis report",
                "required_state_keys": [],
                "timeout": 240,
                "hitl_required": True
            }
        ]

    def determine_entry_point(self, state: AgentState) -> PipelinePhase:
        """Determine the entry point based on current state"""
        # Check if database connection is already established
        if not state.get("database_connection"):
            return PipelinePhase.DB_CONNECTION
        
        # Check if metadata is already created
        if not state.get("schema_info") or not state.get("table_metadata"):
            return PipelinePhase.METADATA_CREATION
        
        # Check if data exploration is already completed
        if not state.get("data_exploration_status") == "completed":
            return PipelinePhase.DATA_EXPLORATION
        
        # Check if causal discovery is already completed
        if not state.get("causal_discovery_status") == "completed":
            return PipelinePhase.CAUSAL_DISCOVERY
        
        # Check if causal inference is already completed
        if not state.get("causal_inference_status") == "completed":
            return PipelinePhase.CAUSAL_INFERENCE
        
        # Default to report generation
        return PipelinePhase.REPORT_GENERATION

class SpecialistAgent(BaseAgent):
    """Specialist agent (Data Explorer, Causal Discovery, Causal Inference)"""
    
    def __init__(self, name: str, agent_type: AgentType, config: Optional[Dict[str, Any]] = None, metrics_collector: Optional[MetricsCollector] = None):
        super().__init__(name, agent_type, config, metrics_collector)
        self.domain_expertise: List[str] = []
        self.input_schema: Dict[str, Any] = {}
        self.output_schema: Dict[str, Any] = {}
        self._initialize_tools()
    
    def _initialize_tools(self) -> None:
        """Initialize tools specific to this specialist agent"""
        try:
            from utils.tools import tool_registry, DatabaseTool, LLMTool
            
            # Register common tools if not already registered
            if not tool_registry.get("database"):
                db_tool = DatabaseTool("reef_db", "postgresql")
                tool_registry.register(db_tool)
            
            if not tool_registry.get("llm"):
                llm_tool = LLMTool()
                tool_registry.register(llm_tool)
            
            # Register specialist-specific tools
            self._register_specialist_tools()
            
        except Exception as e:
            self.on_event("tool_initialization_error", error=str(e))
    
    def _register_specialist_tools(self) -> None:
        """Register tools specific to this specialist agent (override in subclasses)"""
        pass
        
    def set_domain_expertise(self, expertise: List[str]) -> None:
        """Set domain expertise"""
        self.domain_expertise = expertise
        
    def validate_input(self, data: Any) -> bool:
        """Validate input data"""
        # Implement schema-based validation logic
        return True
        
    def validate_output(self, data: Any) -> bool:
        """Validate output data"""
        # Implement schema-based validation logic
        return True

    def requires_hitl(self, state: AgentState, phase: PipelinePhase, substep: str) -> bool:
        """Check if this specialist agent requires HITL at the given phase/substep"""
        # Define HITL requirements for each specialist agent
        hitl_requirements = {
            "data_explorer": {
                PipelinePhase.DATA_EXPLORATION: ["table_selection", "table_retrieval"]
            },
            "causal_discovery": {
                PipelinePhase.CAUSAL_DISCOVERY: ["assumption_method_matrix", "algorithm_scoring", "final_graph_selection"]
            },
            "causal_inference": {
                PipelinePhase.CAUSAL_INFERENCE: ["select_configuration", "interpretation"]
            }
        }
        
        agent_name = self.name.lower()
        if agent_name in hitl_requirements:
            phase_requirements = hitl_requirements[agent_name]
            if phase in phase_requirements:
                return substep in phase_requirements[phase]
        
        return False

class SubgraphAgent(BaseAgent):
    """Subgraph agent (LangGraph based)"""
    
    def __init__(self, name: str, graph, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, AgentType.SUBGRAPH, config)
        self.graph = graph
        self.compiled_graph = None
        
    def compile_graph(self) -> None:
        """Compile graph"""
        if self.graph:
            self.compiled_graph = self.graph.compile()
            self.on_event("graph_compiled")
            
    def step(self, state: AgentState) -> AgentState:
        """Execute graph"""
        if not self.compiled_graph:
            self.compile_graph()
            
        if not self.compiled_graph:
            raise RuntimeError("Graph not compiled")
            
        return self.compiled_graph.invoke(state)