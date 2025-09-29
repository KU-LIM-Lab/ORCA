# core/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Callable, Optional, List, Union, Type
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import asyncio
import json
from monitoring.metrics.collector import MetricsCollector, track_execution_time, track_memory_usage, record_metric, MetricType
from core.state import AgentState, PipelinePhase, HITLType, ExecutionStatus

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
        parent_agent: Optional['BaseAgent'] = None,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        self.name = name
        self.agent_type = agent_type
        self.config = config or {}
        self.parent_agent = parent_agent
        self.tools: Dict[str, Callable] = {}
        self.memory = None
        self.logger = print
        self.status = AgentStatus.IDLE
        self.execution_history: List[AgentResult] = []
        self.dependencies: List[str] = []  # List of dependency agents
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

    def add_tool(self, name: str, tool: Callable) -> None:
        """Add tool"""
        self.tools[name] = tool
        self.on_event("tool_added", tool_name=name)

    def get_tool(self, name: str) -> Optional[Callable]:
        """Get tool"""
        return self.tools.get(name)

    def execute_tool(self, tool_name: str, *args, **kwargs) -> Any:
        """Execute tool"""
        tool = self.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        self.on_event("tool_executed", tool_name=tool_name, args=args, kwargs=kwargs)
        return tool(*args, **kwargs)

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

    def can_handle(self, state: AgentState) -> bool:
        """Check if this agent can handle the given state"""
        try:
            self.validate_state(state)
            return True
        except (TypeError, KeyError):
            return False

    def get_capabilities(self) -> List[str]:
        """Return list of agent capabilities"""
        return list(self.tools.keys())

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

    def create_hitl_context(self, state: AgentState, phase: PipelinePhase, substep: str) -> Dict[str, Any]:
        """Create context for HITL interaction"""
        return {
            "phase": phase.value,
            "substep": substep,
            "agent": self.name,
            "current_state": state,
            "timestamp": datetime.now().isoformat()
        }

    def handle_hitl_approval(self, state: AgentState, user_decision: str) -> AgentState:
        """Handle HITL approval decision"""
        if user_decision == "approve":
            state["user_decision"] = "approve"
            state["hitl_required"] = False
            state["execution_status"] = ExecutionStatus.RUNNING
        elif user_decision == "abort":
            state["user_decision"] = "abort"
            state["hitl_required"] = False
            state["execution_status"] = ExecutionStatus.FAILED
        
        return state

    def handle_hitl_edit(self, state: AgentState, user_edits: Dict[str, Any]) -> AgentState:
        """Handle HITL edit decision"""
        state["user_edits"] = user_edits
        state["hitl_required"] = False
        state["execution_status"] = ExecutionStatus.RUNNING
        
        # Apply user edits to state
        for key, value in user_edits.items():
            state[key] = value
        
        return state

    def handle_hitl_feedback(self, state: AgentState, user_feedback: str) -> AgentState:
        """Handle HITL feedback for loopback"""
        state["user_feedback"] = user_feedback
        state["hitl_required"] = False
        state["execution_status"] = ExecutionStatus.RUNNING
        
        # Add feedback to history
        if "feedback_history" not in state:
            state["feedback_history"] = []
        
        state["feedback_history"].append({
            "timestamp": datetime.now().isoformat(),
            "agent": self.name,
            "feedback": user_feedback
        })
        
        return state

class OrchestratorAgent(BaseAgent):
    """Orchestrator agent (Supervisor)"""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, AgentType.ORCHESTRATOR, config)
        self.sub_agents: Dict[str, BaseAgent] = {}
        self.execution_plan: List[Dict[str, Any]] = []
        
    def add_sub_agent(self, agent: BaseAgent) -> None:
        """Add sub-agent"""
        self.sub_agents[agent.name] = agent
        agent.parent_agent = self
        self.on_event("sub_agent_added", agent_name=agent.name)
        
    def create_execution_plan(self, goal: str, context: AgentState) -> List[Dict[str, Any]]:
        """Create execution plan (must be overridden)"""
        raise NotImplementedError("Subclasses must implement create_execution_plan")
        

    def get_full_pipeline_plan(self) -> List[Dict[str, Any]]:
        """
        Get the full pipeline plan structure.
        
        This defines the complete pipeline with all phases and sub-steps.
        Database and metadata agents are system components that are initialized
        during agent setup, not traditional agents.
        """
        return [
            # Pre-processing phases (System initialization)
            {
                "phase": PipelinePhase.DB_CONNECTION,
                "substep": "connect_database",
                "agent": "system_database",  # System component, not agent
                "action": "connect",
                "description": "Connect to database using utils.system_agents.DatabaseAgent",
                "dependencies": [],
                "timeout": 30,
                "hitl_required": False,
                "is_system_component": True
            },
            {
                "phase": PipelinePhase.METADATA_CREATION,
                "substep": "create_metadata",
                "agent": "system_metadata",  # System component, not agent
                "action": "create_metadata",
                "description": "Create database metadata using utils.system_agents.MetadataAgent",
                "dependencies": ["connect_database"],
                "timeout": 60,
                "hitl_required": False,
                "is_system_component": True
            },
            
            # Data Exploration Phase
            {
                "phase": PipelinePhase.DATA_EXPLORATION,
                "substep": "table_selection",
                "agent": "data_explorer",
                "action": "select_tables",
                "description": "Select relevant tables for analysis",
                "dependencies": ["create_metadata"],
                "timeout": 120,
                "hitl_required": True,
                "hitl_type": HITLType.APPROVAL
            },
            {
                "phase": PipelinePhase.DATA_EXPLORATION,
                "substep": "table_retrieval",
                "agent": "data_explorer",
                "action": "retrieve_data",
                "description": "Retrieve data via text2sql",
                "dependencies": ["table_selection"],
                "timeout": 180,
                "hitl_required": True,
                "hitl_type": HITLType.APPROVAL
            },
            
            # Causal Discovery Phase
            {
                "phase": PipelinePhase.CAUSAL_DISCOVERY,
                "substep": "assumption_method_matrix",
                "agent": "causal_discovery",
                "action": "create_assumption_method_matrix",
                "description": "Create assumption-method compatibility matrix",
                "dependencies": ["table_retrieval"],
                "timeout": 300,
                "hitl_required": True,
                "hitl_type": HITLType.APPROVAL
            },
            {
                "phase": PipelinePhase.CAUSAL_DISCOVERY,
                "substep": "algorithm_scoring",
                "agent": "causal_discovery",
                "action": "score_algorithms",
                "description": "Score algorithms based on assumption-method matrix",
                "dependencies": ["assumption_method_matrix"],
                "timeout": 60,
                "hitl_required": True,
                "hitl_type": HITLType.EDIT
            },
            {
                "phase": PipelinePhase.CAUSAL_DISCOVERY,
                "substep": "run_algorithms",
                "agent": "causal_discovery",
                "action": "run_algorithms",
                "description": "Run selected algorithms in parallel",
                "dependencies": ["algorithm_scoring"],
                "timeout": 600,
                "hitl_required": False
            },
            {
                "phase": PipelinePhase.CAUSAL_DISCOVERY,
                "substep": "intermediate_scoring",
                "agent": "causal_discovery",
                "action": "calculate_intermediate_scores",
                "description": "Calculate intermediate scores for each algorithm",
                "dependencies": ["run_algorithms"],
                "timeout": 120,
                "hitl_required": False
            },
            {
                "phase": PipelinePhase.CAUSAL_DISCOVERY,
                "substep": "final_graph_selection",
                "agent": "causal_discovery",
                "action": "select_final_graph",
                "description": "Select final causal graph based on scores",
                "dependencies": ["intermediate_scoring"],
                "timeout": 180,
                "hitl_required": True,
                "hitl_type": HITLType.EDIT
            },
            
            # Causal Inference Phase
            {
                "phase": PipelinePhase.CAUSAL_INFERENCE,
                "substep": "select_configuration",
                "agent": "causal_inference",
                "action": "select_config",
                "description": "Select inference configuration",
                "dependencies": ["final_graph_selection"],
                "timeout": 120,
                "hitl_required": True,
                "hitl_type": HITLType.EDIT
            },
            {
                "phase": PipelinePhase.CAUSAL_INFERENCE,
                "substep": "effect_estimation",
                "agent": "causal_inference",
                "action": "estimate_effects",
                "description": "Estimate causal effects",
                "dependencies": ["select_configuration"],
                "timeout": 300,
                "hitl_required": False
            },
            {
                "phase": PipelinePhase.CAUSAL_INFERENCE,
                "substep": "interpretation",
                "agent": "causal_inference",
                "action": "interpret_results",
                "description": "Interpret and validate results",
                "dependencies": ["effect_estimation"],
                "timeout": 180,
                "hitl_required": True,
                "hitl_type": HITLType.APPROVAL
            },
            
            # Report Generation
            {
                "phase": PipelinePhase.REPORT_GENERATION,
                "substep": "generate_report",
                "agent": "report_generator",
                "action": "generate_report",
                "description": "Generate final analysis report",
                "dependencies": ["interpretation"],
                "timeout": 240,
                "hitl_required": True,
                "hitl_type": HITLType.APPROVAL
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
    
    def __init__(self, name: str, agent_type: AgentType, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, agent_type, config)
        self.domain_expertise: List[str] = []
        self.input_schema: Dict[str, Any] = {}
        self.output_schema: Dict[str, Any] = {}
        
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