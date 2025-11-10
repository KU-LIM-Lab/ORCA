# core/state.py
from typing import TypedDict, Optional, Any, Dict, List, Annotated
from datetime import datetime
from enum import Enum

class ExecutionStatus(Enum):
    """Execution status for different phases"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    INTERRUPTED = "interrupted"

class PipelinePhase(Enum):
    """Pipeline phases for the full causal analysis workflow"""
    # Pre-processing phases
    DB_CONNECTION = "db_connection"
    METADATA_CREATION = "metadata_creation"
    
    # Main analysis phases
    DATA_EXPLORATION = "data_exploration"
    CAUSAL_DISCOVERY = "causal_discovery"
    CAUSAL_INFERENCE = "causal_inference"
    
    # Post-processing phases
    REPORT_GENERATION = "report_generation"

class AgentState(TypedDict, total=False):
    """Global state shared across all agents in the ORCA system"""
    
    # === Core Execution State ===
    initial_query: str
    execution_status: ExecutionStatus
    current_phase: PipelinePhase
    current_substep: str
    completed_phases: List[PipelinePhase]
    completed_substeps: List[str]
    error_log: List[Dict[str, Any]]

    # === Optional Input ===
    ground_truth_dataframe_path: str
    ground_truth_causal_graph_path: str
    ground_truth_dataframe: Dict[str, Any]
    ground_truth_causal_graph: Dict[str, Any]
    
    # === HITL State ===
    user_edits: Dict[str, Any]
    user_feedback: Optional[str]
    
    # === User Interaction ===
    user_constraints: Dict[str, Any]  # User-provided constraints for causal discovery
    user_preferences: Dict[str, Any]  # User preferences for analysis
    hitl_decision: str
    hitl_executed: bool
    
    
    # === Database Connection Phase Outputs ===
    db_id: str  # Database identifier
    database_connection: Optional[Dict[str, Any]]
    connection_status: str  # "connected", "failed", "pending"
    
    # === Metadata Creation Phase Outputs (utils.data_prep 기반) ===
    schema_info: Dict[str, Any]  # extract_schema() result
    table_metadata: Dict[str, Any]  # generate_metadata() result
    table_relations: Dict[str, Any]  # update_table_relations() result
    metadata_creation_status: str  # "completed", "failed", "pending"

    # === Planning & execution Phase Outputs ===
    error: str
    error_type: str
    execution_plan: List[Dict[str, Any]]
    # Runtime orchestration controls
    skip_steps: List[str]

    allow_start_without_ground_truth: bool
    analysis_mode: str
    plan_created: bool
    total_steps: int
    estimated_duration: float
    current_state_executed: bool

    recovery_strategy: str
    planner_completed: bool
    executor_completed: bool
    error_handler_completed: bool
    finalizer_completed: bool

    current_execute_step: int

    # === Data Exploration Phase Outputs ===
    # A.1 Table Selection
    candidate_tables: List[str]
    selected_tables: List[str]

    table_recommendation_completed: bool
    
    objective_summary: Annotated[str, None]
    recommended_tables: Annotated[list, None]
    recommended_method: Annotated[str, None]
    erd_image_path: Annotated[str, None]
    final_output: Annotated[str, None]
    
    # Some modules refer to schema-level analysis as 'schema_analysis'
    schema_analysis: Dict[str, Any]
    # For compatibility with modules using 'table_analysis' and 'related_tables'
    table_analysis: Annotated[dict, None]
    related_tables: Annotated[dict, None]

    table_exploration_completed: bool
    
    # Aggregated related tables and analysis hints
    all_related_tables: Dict[str, Any]
    analysis_recommendations: Dict[str, Any]
    
    final_sql: Optional[str]
    columns: Optional[List[str]]
    result: Optional[List]
    error: Optional[str]
    llm_review: Optional[str]
    output: Optional[dict]
    
    # A.2 Table Retrieval (via text2sql)
    sql_query: str
    df_raw: Optional[Any]  # pandas DataFrame
    df_preprocessed: Optional[Any]  # pandas DataFrame
    # Redis persistence for dataframes
    df_redis_key: Optional[str]
    df_shape: Optional[Any]
    variable_info: Dict[str, Any]
    variable_schema: Dict[str, Any]  # Comprehensive schema with types and cardinality
    text2sql_generation_completed: bool
    
    # Data Exploration Phase Status
    data_exploration_status: str  # "completed", "failed", "pending"
    
    # Data preprocessing artifacts
    preprocess_report: str
    column_stats: Dict[str, Any]
    feature_map: Dict[str, Any]
    warnings: List[str]
    data_preprocessing_completed: bool
    
    # Data preprocessor configuration options
    steps: Optional[List[str]]  # List of preprocessing steps to execute
    impute_strategy: Optional[str]  # Strategy for imputing missing values
    scaling: Optional[str]  # Scaling method (standard, minmax, none)
    one_hot_threshold: Optional[int]  # Threshold for one-hot encoding
    skip_one_hot_encoding: Optional[bool]  # Skip one-hot encoding for causal discovery
    persist_to_redis: Optional[bool]  # Whether to persist dataframes to Redis
    fetch_only: Optional[bool]  # Whether to only fetch data without full preprocessing
    
    # === Causal Discovery Phase Outputs ===
    # 1. Assumption-method compatibility matrix
    data_assumptions: Dict[str, bool]  # Validated data assumptions
    assumption_method_scores: Dict[str, Dict[str, float]]  # Assumption-method compatibility matrix
    
    # 2. Algorithm scores
    algorithm_scores: Dict[str, float]  # Algorithm scores
    selected_algorithms: List[str]  # Selected algorithms
    
    # Profiling & tiering
    data_profile: Dict[str, Any]
    data_profiling_completed: bool
    algorithm_tiers: Dict[str, Any]
    tiering_reasoning: str
    algorithm_tiering_completed: bool
    algorithm_configuration_completed: bool  # New: algorithm configuration completion flag
    
    # 3. Algorithm execution
    algorithm_results: Dict[str, Any]  # Algorithm execution results
    candidate_graphs: List[Dict[str, Any]]  # Generated candidate graphs
    algorithm_results_key: str
    run_algorithms_portfolio_completed: bool
    
    # 4. Intermediate scores
    intermediate_scores: Dict[str, Dict[str, float]]  # Intermediate scores
    
    # Pruning & evaluation & ensemble
    pruned_candidates: List[Dict[str, Any]]
    pruning_log: List[Dict[str, Any]]
    candidate_pruning_completed: bool
    scorecard: Dict[str, Any]
    ranked_graphs: List[Dict[str, Any]]  # Ranked graphs from scorecard evaluation
    top_candidates: List[Dict[str, Any]]
    scorecard_evaluation_completed: bool
    user_params: Dict[str, Any]  # User parameters (ranking_mode, pruning_thresholds, etc.)
    consensus_pag: Dict[str, Any]
    synthesis_reasoning: str
    ensemble_synthesis_completed: bool
    
    # 5. Final graph decision
    selected_graph: Dict[str, Any]  # Final selected causal graph
    graph_selection_reasoning: str  # Graph selection reasoning
    
    # Causal Discovery Phase Status
    causal_discovery_status: str  # "completed", "failed", "pending"
    
    # === Causal Inference Phase Outputs ===
    # C.1 Select Configuration
    treatment_variable: str
    outcome_variable: str
    confounders: List[str]
    instrumental_variables: List[str]
    
    # C.2 Effect Estimation
    inference_method: str
    causal_estimates: Dict[str, Any]
    confidence_intervals: Dict[str, List[float]]
    
    # C.3 Interpretation
    interpretation_results: Dict[str, Any]
    sensitivity_analysis: Dict[str, Any]
    
    # Causal Inference Phase Status
    causal_inference_status: str  # "completed", "failed", "pending"
    
    # === Causal Analysis (pipeline variant) ===
    parsed_query: Dict[str, Any]
    table_schema_str: str
    strategy: Any
    final_answer: str
    parse_question_completed: bool
    config_selection_completed: bool
    dowhy_analysis_completed: bool
    generate_answer_completed: bool
    
    # === Final Outputs ===
    final_report: Dict[str, Any]
    recommendations: List[str]
    
    # === System State ===
    session_id: str
    timestamp: datetime
    agent_execution_log: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]
    
    # Orchestration execution artifacts
    execution_log: List[Dict[str, Any]]
    results: Dict[str, Any]

class DataExplorerState(TypedDict, total=False):
    """State specific to Data Explorer Agent"""
    database_url: str
    schema_info: Dict[str, Any]
    target_tables: List[str]
    data_sample: Dict[str, Any]
    preprocessing_steps: List[Dict[str, Any]]
    data_summary: Dict[str, Any]

class CausalDiscoveryState(TypedDict, total=False):
    """State specific to Causal Discovery Agent"""
    data_assumptions: Dict[str, bool]  # Validated data assumptions
    algorithm_configs: Dict[str, Dict[str, Any]]  # Algorithm configurations
    bootstrap_results: List[Dict[str, Any]]  # Bootstrap sampling results
    graph_ensemble: List[Dict[str, Any]]  # Ensemble of discovered graphs
    edge_frequencies: Dict[str, float]  # Edge occurrence frequencies
    graph_metrics: Dict[str, float]  # Graph quality metrics

class CausalInferenceState(TypedDict, total=False):
    """State specific to Causal Inference Agent"""
    treatment_variable: str
    outcome_variable: str
    confounders: List[str]
    instrumental_variables: List[str]
    estimation_method: str
    identification_strategy: str
    results_summary: Dict[str, Any]

class OrchestratorState(TypedDict, total=False):
    """State specific to Orchestrator Agent"""
    execution_plan: List[Dict[str, Any]]
    agent_status: Dict[str, ExecutionStatus]
    resource_usage: Dict[str, Any]
    error_recovery_actions: List[Dict[str, Any]]

def create_initial_state(query: str, db_id: str = "reef_db", session_id: str = None) -> AgentState:
    """Create initial state for a new analysis session"""
    if session_id is None:
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    return AgentState(
        initial_query=query,
        execution_status=ExecutionStatus.PENDING,
        current_phase=PipelinePhase.DB_CONNECTION,
        current_substep="",
        completed_phases=[],
        completed_substeps=[],
        error_log=[],
        user_edits={},
        user_feedback=None,
        user_constraints={},
        user_preferences={},
        db_id=db_id,
        database_connection=None,
        connection_status="pending",
        schema_info={},
        table_metadata={},
        table_relations={},
        metadata_creation_status="pending",
        candidate_tables=[],
        selected_tables=[],
        sql_query="",
        df_raw=None,
        df_preprocessed=None,
        variable_info={},
        variable_schema={},
        data_exploration_status="pending",
        data_assumptions={},
        assumption_method_scores={},
        algorithm_scores={},
        selected_algorithms=[],
        algorithm_results={},
        candidate_graphs=[],
        intermediate_scores={},
        selected_graph={},
        graph_selection_reasoning="",
        causal_discovery_status="pending",
        treatment_variable="",
        outcome_variable="",
        confounders=[],
        instrumental_variables=[],
        inference_method="",
        causal_estimates={},
        confidence_intervals={},
        interpretation_results={},
        sensitivity_analysis={},
        causal_inference_status="pending",
        final_report={},
        recommendations=[],
        session_id=session_id,
        timestamp=datetime.now(),
        agent_execution_log=[],
        performance_metrics={}
    )

def validate_state(state: AgentState) -> bool:
    """Validate state structure and required fields"""
    required_fields = ["initial_query", "session_id", "execution_status", "current_phase"]
    
    for field in required_fields:
        if field not in state:
            return False
    
    # Validate execution status
    if not isinstance(state["execution_status"], ExecutionStatus):
        return False
    
    # Validate current phase
    if not isinstance(state["current_phase"], PipelinePhase):
        return False
    
    return True

def get_agent_specific_state(state: AgentState, agent_type: str) -> Dict[str, Any]:
    """
    Extract agent-specific state from global state
    agent 내부 수정에 맞추어 변경 필요
    """
    agent_states = {
        "data_explorer": ["db_id", "schema_info", "table_metadata", "candidate_tables", 
                         "selected_tables", "sql_query", "df_raw", "df_preprocessed", "variable_schema"],
        "causal_discovery": ["data_assumptions", "assumption_method_scores", "algorithm_scores", 
                            "selected_algorithms", "algorithm_results", "candidate_graphs",
                            "intermediate_scores", "selected_graph", "variable_schema"],
        "causal_inference": ["treatment_variable", "outcome_variable", "confounders",
                            "inference_method", "causal_estimates", "confidence_intervals",
                            "interpretation_results", "sensitivity_analysis"],
        "orchestrator": ["current_phase", "completed_phases", "agent_execution_log"]
    }
    
    if agent_type not in agent_states:
        return {}
    
    return {key: state.get(key) for key in agent_states[agent_type] if key in state}

def get_phase_status(state: AgentState, phase: PipelinePhase) -> str:
    """Get the status of a specific phase"""
    phase_status_map = {
        PipelinePhase.DB_CONNECTION: state.get("connection_status", "pending"),
        PipelinePhase.METADATA_CREATION: state.get("metadata_creation_status", "pending"),
        PipelinePhase.DATA_EXPLORATION: state.get("data_exploration_status", "pending"),
        PipelinePhase.CAUSAL_DISCOVERY: state.get("causal_discovery_status", "pending"),
        PipelinePhase.CAUSAL_INFERENCE: state.get("causal_inference_status", "pending"),
        PipelinePhase.REPORT_GENERATION: state.get("final_report", {}).get("status", "pending")
    }
    
    return phase_status_map.get(phase, "pending")

def is_phase_completed(state: AgentState, phase: PipelinePhase) -> bool:
    """Check if a phase is completed"""
    return phase in state.get("completed_phases", [])

def is_substep_completed(state: AgentState, substep: str) -> bool:
    """Check if a substep is completed"""
    return substep in state.get("completed_substeps", [])

