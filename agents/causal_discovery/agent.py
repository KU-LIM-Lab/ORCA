# agents/causal_discovery/agent.py
"""
Causal Discovery Agent implementation using tool registry pattern.

This agent implements the complete causal discovery pipeline:
1. data_profiling - Profile data characteristics and generate qualitative summary
2. algorithm_configuration - Configure algorithms based on data profile (replaces algorithm_tiering)
3. run_algorithms_portfolio - Execute algorithms from execution_plan in parallel
4. graph_scoring - Calculate 3 scores (markov_consistency, sampling_stability, structural_stability) for all graphs
5. graph_evaluation - Evaluate and rank graphs using composite scorecard
6. ensemble_synthesis - Synthesize ensemble with PAG-like and DAG outputs
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from core.base import SpecialistAgent, AgentType
from core.state import AgentState
from monitoring.metrics.collector import MetricsCollector
from agents.causal_discovery.tools import get_edges, get_variables, get_graph_type, validate_graph_schema


logger = logging.getLogger(__name__)


def _convert_numpy_types(obj):
    """Convert numpy types to Python native types for msgpack serialization."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: _convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_convert_numpy_types(item) for item in obj)
    else:
        return obj

"""Algorithm cheatsheet (assumptions → brief)  [근거/References in brackets]

LiNGAM  — Required: linear relations + non‑Gaussian noise.
          Identifiable causal ordering via ICA/independent noise.  
          [Shimizu et al., JMLR 2006]
ANM     — Required: additive noise model + noise ⟂ cause (residual independence).
          Nonlinear, identifies direction by testing residual–input independence.  
          [Hoyer et al., NIPS 2009]
GES     — Required: (typically) linear‑Gaussian likelihood; equal‑variance often assumed.
          Score‑based DAG search optimizing BIC/MDL over equivalence classes.  
          [Chickering, JMLR 2002]
PC      — Required: CI tests valid for the data‑generating process (often Gaussian‑CI).
          Constraint‑based: removes/extends edges via conditional independence tests.  
          [Spirtes, Glymour, Scheines, 2000]
CAM     — Required: additive models (nonparametric) per node.
          Hybrid (score + pruning) using GAMs to recover DAG under ANM‑like assumptions.  
          [Bühlmann et al., Biometrika 2014]
FCI     — Required: CI tests valid; allows latent confounders (outputs PAG).
          Constraint‑based generalization of PC robust to hidden variables.  
          [Spirtes et al., 1995/2000]"""

# === Standardized algorithm catalog & assumption-role mapping ===
ALGORITHM_LIST = [
    "LiNGAM",     # linear + non-Gaussian (fully identifiable)
    "ANM",        # additive noise (fully identifiable)
    "GES",        # score-based (BIC/generalized)
    "PC",         # constraint-based
    # "CAM",      # additive models (temporarily disabled)
    "FCI",        # allows latent confounders (PAG)
    "LiM",        # Linear Mixed model for mixed data
]

# === Algorithm Family Classification ===
# Representative algorithms per family (default mode)
ALGORITHM_FAMILIES_REP = {
    "Linear": ["LiNGAM"],
    "Nonlinear-FCM": ["ANM"],
    "Constraint-based": ["PC"],
    "Score-based": ["GES"],
    "Latent-robust": ["FCI"]
}

# Full algorithm sets per family (when run_all_tier_algorithms=True)
ALGORITHM_FAMILIES_ALL = {
    "Linear": ["LiNGAM"],
    "Nonlinear-FCM": ["ANM"],
    "Constraint-based": ["PC"],
    "Score-based": ["GES"],
    "Latent-robust": ["FCI"]
}

# Roles → scoring: Required: S^beta (conservative), Preferred: 0.5+0.5S, Irrelevant: 0.5.
# A soft‑AND across required assumptions is applied via geometric mean.
ASSUMPTION_METHODS = {
    # LiNGAM: linear relations + non‑Gaussian noise (identifiable ordering)
    "LiNGAM": {
        "required": ["S_lin", "S_nG"],
        "preferred": ["S_ANM"],
        "irrelevant": ["S_Gauss", "S_EqVar"]
    },
    # ANM: additive noise model + noise ⟂ cause (residual independence)
    "ANM": {
        "required": ["S_ANM"],
        "preferred": ["S_lin"],  # prefer nonlinearity: (1 - S_lin) in scoring
        "irrelevant": ["S_nG", "S_Gauss", "S_EqVar"]
    },
    # PNL: post‑nonlinear transform around additive noise
    "PNL": {
        "required": ["S_ANM"],
        "preferred": ["S_lin"],
        "irrelevant": ["S_nG", "S_Gauss", "S_EqVar"]
    },
    # GES: linear‑Gaussian likelihood + equal variance (score-based)
    "GES": {
        "required": ["S_lin", "S_Gauss", "S_EqVar"],
        "preferred": [],
        "irrelevant": ["S_nG", "S_ANM"]
    },
    # PC: CI tests valid for data‑generating process (often Gaussian‑CI)
    "PC": {
        "required": ["S_lin"], 
        "preferred": ["S_Gauss"],
        "irrelevant": ["S_EqVar", "S_nG", "S_ANM"]
    },
    # CAM: additive models (nonparametric) per node
    "CAM": {
        "required": ["S_ANM"],
        "preferred": ["S_lin"],
        "irrelevant": ["S_nG", "S_Gauss", "S_EqVar"]
    },
    # FCI: CI tests valid; allows latent confounders (outputs PAG)
    "FCI": {
        "required": ["S_lin"],
        "preferred": ["S_Gauss"],
        "irrelevant": ["S_EqVar", "S_nG", "S_ANM"]
    }
}

class CausalDiscoveryAgent(SpecialistAgent):
    """Causal Discovery Agent using tool registry pattern"""
    
    def __init__(self, name: str = "causal_discovery", config: Optional[Dict[str, Any]] = None, 
                 metrics_collector: Optional[MetricsCollector] = None):
        super().__init__(name, AgentType.SPECIALIST, config, metrics_collector)
        
        # Set domain expertise
        self.set_domain_expertise([
            "assumption_validation", "algorithm_selection", "causal_graph_generation",
            "graph_evaluation", "statistical_testing"
        ])
        
        # Load configuration
        self._load_configuration(config)
    
    def on_event(self, event: str, **kwargs) -> None:
        """Override to reduce verbose logging for causal discovery"""
        if self.logger:
            # Skip verbose tool events for cleaner output
            if event in ["tool_used", "tool_success", "tool_registered"]:
                return
                
            self.logger({
                "agent": self.name, 
                "type": self.agent_type.value,
                "status": self.status.value,
                "event": event, 
                "timestamp": datetime.now().isoformat(),
                **kwargs
            })
    
    def _load_configuration(self, config: Optional[Dict[str, Any]]) -> None:
        """Load and validate configuration parameters"""
        self.config = config or {}
        
        # Core parameters
        self.bootstrap_iterations = self.config.get("bootstrap_iterations", 50)
        self.cv_folds = self.config.get("cv_folds", 5)
        
        # Pipeline parameters
        self.ci_alpha = self.config.get("ci_alpha", 0.05)
        self.violation_threshold = self.config.get("violation_threshold", 0.1)
        self.n_subsets = self.config.get("n_subsets", 3)
        self.profiling_max_pairs = self.config.get("profiling_max_pairs", 300)
        self.profiling_parallelism = self.config.get("profiling_parallelism")
        self.anm_rf_estimators = self.config.get("anm_rf_estimators", 100)
        self.run_all_tier_algorithms = self.config.get("run_all_tier_algorithms", False)
        self.max_pa_size = self.config.get("max_pa_size", 3) 
        
        # Composite weights 
        self.composite_weights = self.config.get("composite_weights", {
            "markov_consistency": 0.4,
            "sampling_stability": 0.3,
            "structural_stability": 0.3
        })
        
        # Scoring configuration
        scoring = self.config.get("scoring", {})
        self.use_quantile_thresholds = scoring.get("use_quantile_thresholds", True)
        self.min_pairs_for_quantile = scoring.get("min_pairs_for_quantile", 30)
        self.penalty_weight = scoring.get("penalty_weight", 0.5)
        
        # Thresholds
        thresholds = scoring.get("simple_thresholds", {})
        self.threshold_high = thresholds.get("high", 0.66)
        self.threshold_low = thresholds.get("low", 0.33)
        self.min_confidence = thresholds.get("min_confidence", 0.05)
        
        # ANM thresholds
        anm_config = scoring.get("anm_compatibility", {})
        self.anm_mean_threshold = anm_config.get("mean_threshold", 0.5)
        self.anm_pair_ratio_threshold = anm_config.get("pair_ratio_threshold", 0.6)
        
        # Store constants
        self.config.setdefault("ASSUMPTION_METHODS", ASSUMPTION_METHODS)
        self.config.setdefault("algorithm_list", ALGORITHM_LIST)
    
    def _register_specialist_tools(self) -> None:
        """Register causal discovery specific tools"""
        # skip tools that are already present in the global registry
        from utils.tools import tool_registry
        tools = [
            # Statistical & Independence Testing
            ("stats_tool", self._stats_tool, "Statistical testing: GLM/GAM, LRT/AIC/BIC, normality tests"),
            ("independence_tool", self._independence_tool, "Independence testing: HSIC/KCI, non-parametric regression"),
            
            # Causal Discovery Algorithms
            ("lingam_tool", self._lingam_tool, "LiNGAM algorithm: DirectLiNGAM order/weight estimation"),
            ("anm_tool", self._anm_tool, "ANM algorithm: Additive Noise Model directionality testing"),
            ("pc_tool", self._pc_tool, "PC algorithm via backend (default: causal-learn)"),
            ("ges_tool", self._ges_tool, "GES algorithm via backend (default: causal-learn)"),
            ("fci_tool", self._fci_tool, "FCI algorithm via backend (default: causal-learn)"),
            ("cam_tool", self._cam_tool, "CAM algorithm (via CDT)"),
            ("lim_tool", self._lim_tool, "LiM algorithm: Linear Mixed model for mixed data"),
            ("tscm_tool", self._tscm_tool, "TSCM algorithm: Tree-Structured Causal Model for mixed data"),
            ("notears_linear_tool", self._notears_linear_tool, "NOTEARS-linear algorithm: DAG learning via continuous optimization (linear)"),
            ("notears_nonlinear_tool", self._notears_nonlinear_tool, "NOTEARS-nonlinear algorithm: DAG learning via continuous optimization (nonlinear MLP)"),
            
            # Evaluation & Analysis
            ("bootstrapper", self._bootstrapper, "Bootstrap resampling and frequency aggregation"),
            ("graph_evaluator", self._graph_evaluator, "Graph evaluation: BIC/MDL, CV, robustness, assumption scores"),
            ("graph_ops", self._graph_ops, "Graph operations: DAG/PAG/AG conversion, merge, voting"),
            
            # Pipeline Tools
            ("pruning_tool", self._pruning_tool, "CI testing and structural consistency"),
            ("ensemble_tool", self._ensemble_tool, "Consensus skeleton and PAG construction"),
        ]
        
        for tool_name, tool_method, description in tools:
            # If tool already exists, skip registration (idempotent)
            if tool_registry.get(tool_name):
                continue
            self.register_tool(tool_name, tool_method, description)
    
    def step(self, state: AgentState) -> AgentState:
        """Execute causal discovery step"""
        substep = state.get("current_substep", "data_profiling")
        
        logger.info(f"CausalDiscoveryAgent executing substep: {substep}")
        
        try:
            if substep == "data_profiling":
                return self._data_profiling(state)
            elif substep == "algorithm_configuration":
                return self._algorithm_configuration(state)
            elif substep == "run_algorithms_portfolio":
                return self._run_algorithms_portfolio(state)
            elif substep == "graph_scoring":
                return self._graph_scoring(state)
            elif substep == "graph_evaluation":
                return self._graph_evaluation(state)
            elif substep == "ensemble_synthesis":
                return self._ensemble_synthesis(state)
            else:
                raise ValueError(f"Unknown substep: {substep}")
                
        except Exception as e:
            logger.error(f"CausalDiscoveryAgent substep {substep} failed: {str(e)}")
            state["error"] = f"Causal discovery {substep} failed: {str(e)}"
            state["causal_discovery_status"] = "failed"
            return state
    
    def _data_profiling(self, state: AgentState) -> AgentState:
        """Stage 1: Data profiling with three-stage approach (basic checks + global tests + pairwise profiles)"""
        logger.info("Performing data profiling...")
        
        try:
            # Get preprocessed data (load from reference if necessary)
            df = self._load_dataframe_from_state(state)
            if df is None:
                raise ValueError("No preprocessed data available")
            
            # Get variable information
            variables = list[Any](df.columns)
            logger.info(f"Profiling data for variables: {variables}")
            
            # Check for variable schema from preprocessing
            variable_schema = state.get("variable_schema", {})
            
            # 1. 기본 검사 (p > n, 데이터 유형 등)
            basic_profile = self._run_basic_checks(variable_schema, df)
            logger.info(f"Basic checks profile: {basic_profile}")
            
            # 2. 전역 테스트 (GES 및 Mixed Data)
            global_profile = self._run_global_tests(df, variable_schema, basic_profile)
            logger.info(f"Global tests profile: {global_profile}")
            
            # 3. 쌍별 테스트 (LiNGAM, ANM)
            pairwise_profile = self._run_pairwise_profiles(df, variable_schema, basic_profile)
            logger.info(f"Pairwise profiles: {pairwise_profile}")
            
            # 4. 최종 프로파일 취합
            data_profile = {
                "basic_checks": basic_profile,
                "global_scores": global_profile,
                "pairwise_scores": pairwise_profile
            }
            
            # Update state
            state["data_profile"] = data_profile
            state["data_profiling_completed"] = True
            
            # Store variable_schema if available (for use in algorithm selection)
            if variable_schema:
                state["variable_schema"] = variable_schema
            
            logger.info("Data profiling completed")
            
            # Request HITL for data profile review if interactive mode
            if state.get("interactive", False):
                state = self.request_hitl(
                    state,
                    payload={
                        "step": "data_profiling",
                        "phase": "causal_discovery",
                        "description": "Please review the data profiling results",
                        "decisions": ["approve", "edit", "rerun", "abort"]
                    },
                    hitl_type="data_profile_review"
                )
            
            return state
            
        except Exception as e:
            logger.error(f"Data profiling failed: {str(e)}")
            state["error"] = f"Data profiling failed: {str(e)}"
            return state
    
    def _run_basic_checks(self, variable_schema: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """Basic checks: Classify data type and assess CI reliability
                
        Args:
            variable_schema: Variable schema from preprocessing
            df: DataFrame to profile
            
        Returns:
            Dictionary with:
            - data_type_profile: "Pure Continuous", "Mixed", or "Pure Categorical"
            - ci_reliability: "High", "Low (High-Dimensional)", or "Critically Low (p > n)"
            - high_cardinality: Boolean indicating high cardinality variables
        """
        stats = variable_schema.get("statistics", {}) if variable_schema else {}
        n_cont = stats.get("n_continuous", 0)
        n_cat = stats.get("n_categorical", 0) + stats.get("n_binary", 0)
        
        # 1. 데이터 유형(Data Type) 분류
        data_type_profile = "Mixed"
        if n_cont > 0 and n_cat == 0:
            data_type_profile = "Pure Continuous"
        elif n_cont == 0 and n_cat > 0:
            data_type_profile = "Pure Categorical"
        
        # 2. 차원성(Dimensionality)에 기반한 CI 신뢰도 판단 
        n_variables = len(df.columns)
        n_samples = len(df)
        p_gt_n = n_variables > n_samples
        
        ci_reliability = "High"
        if p_gt_n:
            ci_reliability = "Critically Low (p > n)"  # p > n 이면 CI 테스트는 통계적으로 신뢰 불가
        elif n_variables > 20:  # 변수가 많아도 CI 테스트 검정력 저하
            ci_reliability = "Low (High-Dimensional)"
        
        # 3. High cardinality check
        high_cardinality = len(variable_schema.get("high_cardinality_vars", [])) > 0 if variable_schema else False
        
        return {
            "data_type_profile": data_type_profile,
            "ci_reliability": ci_reliability,
            "high_cardinality": high_cardinality,
            "n_variables": n_variables,
            "n_samples": n_samples
        }
    
    def _run_global_tests(self, df: pd.DataFrame, variable_schema: Dict[str, Any], basic_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Global tests: GES 및 Mixed Data 전역 가정 검증
        
        Args:
            df: DataFrame to profile
            variable_schema: Variable schema from preprocessing
            basic_profile: Results from basic checks
            
        Returns:
            Dictionary with global test scores:
            - s_global_normality_pvalue: Multivariate normality p-value (for Pure Continuous)
            - s_global_linearity_pvalue: Global linearity p-value (for Pure Continuous or Mixed)
        """
        from .tools import StatsTool
        
        data_type = basic_profile.get("data_type_profile", "Mixed")
        global_scores = {}
        
        # 변수 분류
        schema_vars = variable_schema.get("variables", {}) if variable_schema else {}
        cont_vars = [k for k, v in schema_vars.items() if v.get("data_type") == "Continuous"]
        cat_vars = [k for k, v in schema_vars.items() if v.get("data_type") in ["Nominal", "Binary", "Ordinal"]]
        
        # 다변량 정규성: Pure Continuous만
        if data_type == "Pure Continuous" and cont_vars:
            try:
                # Henze-Zirkler(HZ) 테스트
                # H0: 데이터가 다변량 정규분포를 따름
                hz_pvalue = StatsTool.global_normality_test(df[cont_vars])
                global_scores["s_global_normality_pvalue"] = hz_pvalue
            except Exception as e:
                logger.warning(f"Global normality test failed: {e}")
                global_scores["s_global_normality_pvalue"] = 0.0  # 실패 시 비정규성으로 간주
        else:
            global_scores["s_global_normality_pvalue"] = np.nan
        
        # 전역 선형성: Pure Continuous 또는 Mixed
        if data_type in ["Pure Continuous", "Mixed"]:
            try:
                if data_type == "Pure Continuous" and cont_vars:
                    test_df = df[cont_vars].copy()  # Explicit copy to avoid modifying original
                elif data_type == "Mixed":
                    test_df = df.copy()  # Deep copy to ensure original df is not modified
                    for cat_var in cat_vars:
                        if cat_var in test_df.columns:
                            dummies = pd.get_dummies(test_df[cat_var], prefix=cat_var, drop_first=True)
                            test_df = pd.concat([test_df.drop(columns=[cat_var]), dummies], axis=1)
                    # 연속형 변수와 더미 변수 모두 포함
                    selected_cols = []
                    if cont_vars:
                        selected_cols.extend([c for c in test_df.columns if c in cont_vars])
                    if cat_vars:
                        selected_cols.extend([c for c in test_df.columns if any(c.startswith(cat_var + "_") for cat_var in cat_vars)])
                    if selected_cols:
                        test_df = test_df[selected_cols]
                else:
                    # Fallback: use original df (but make copy for safety)
                    test_df = df.copy()
                
                if test_df.shape[0] >= 10 and test_df.shape[1] >= 2:
                    # Ramsey RESET test 
                    reset_pvalue = StatsTool.global_linearity_test(test_df)
                    global_scores["s_global_linearity_pvalue"] = reset_pvalue
                else:
                    global_scores["s_global_linearity_pvalue"] = np.nan
            except Exception as e:
                logger.warning(f"Global linearity test failed: {e}")
                global_scores["s_global_linearity_pvalue"] = np.nan
        else:
            global_scores["s_global_linearity_pvalue"] = np.nan
        
        return global_scores
    
    def _run_pairwise_profiles(self, df: pd.DataFrame, variable_schema: Dict[str, Any], basic_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Pairwise profiles: LiNGAM / ANM용 쌍별 가정 검증
        
        Args:
            df: DataFrame to profile
            variable_schema: Variable schema from preprocessing
            basic_profile: Results from basic checks
            
        Returns:
            Dictionary with pairwise scores:
            - s_pairwise_linearity_median: Median linearity score
            - s_pairwise_non_gaussianity_median: Median non-Gaussianity score
            - s_pairwise_anm_median: Median ANM score
        """
        from .tools import StatsTool, IndependenceTool
        
        if basic_profile.get("data_type_profile") != "Pure Continuous":
            return {}
        
        scores_lin_cont = []
        scores_ng_cont = []
        scores_anm_cont = []
        
        # Get continuous variables
        schema_vars = variable_schema.get("variables", {}) if variable_schema else {}
        cont_vars = [k for k, v in schema_vars.items() if v.get("data_type") == "Continuous"]
        
        if not cont_vars:
            return {}
        
        cont_cont_pairs = [(cont_vars[i], cont_vars[j]) for i in range(len(cont_vars)) for j in range(i + 1, len(cont_vars))]
        
        for var1, var2 in cont_cont_pairs:
            try:
                # 1. 선형성 (Linearity): GLM vs GAM MSE 비교
                lin_result = StatsTool.linearity_test(df[var1], df[var2])
                scores_lin_cont.append(lin_result.get("linearity_score", np.nan))
            except Exception as e:
                logger.warning(f"Linearity test failed for {var1}-{var2}: {e}")
                scores_lin_cont.append(np.nan)
            
            try:
                # 2. 비정규성 (Non-Gaussianity): 잔차의 정규성 p-value
                ng_result = StatsTool.gaussian_eqvar_test(df[var1], df[var2])  # 잔차 정규성 테스트
                gaussian_p_value = ng_result.get("gaussian_score", 0.5)  # p-value (높을수록 정규성)
                scores_ng_cont.append(1.0 - gaussian_p_value)  # 비정규성 점수로 변환 (높을수록 비정규성)
            except Exception as e:
                logger.warning(f"Non-Gaussianity test failed for {var1}-{var2}: {e}")
                scores_ng_cont.append(np.nan)
            
            try:
                # 3. ANM 적합성: HSIC p-value
                anm_result = IndependenceTool.anm_test(df[var1], df[var2])
                scores_anm_cont.append(anm_result.get("anm_score", np.nan))
            except Exception as e:
                logger.warning(f"ANM test failed for {var1}-{var2}: {e}")
                scores_anm_cont.append(np.nan)


        pairwise_scores = {
            "s_pairwise_linearity_score": float(np.nanmean(scores_lin_cont)) if scores_lin_cont else np.nan,
            "s_pairwise_non_gaussianity_score": float(np.nanmedian(scores_ng_cont)) if scores_ng_cont else np.nan,
            "s_pairwise_anm_score": float(np.nanmedian(scores_anm_cont)) if scores_anm_cont else np.nan
        }
        
        return pairwise_scores
    
    def _algorithm_configuration(self, state: AgentState) -> AgentState:
        """Stage 2: Algorithm configuration based on data profile - generates execution_plan"""
        logger.info("Performing algorithm configuration...")
        
        try:
            data_profile = state.get("data_profile", {})
            variable_schema = state.get("variable_schema", {})
            
            if not data_profile:
                raise ValueError("No data profile available")
            
            execution_plan = self._generate_execution_plan(data_profile, variable_schema)
            
            state["execution_plan"] = execution_plan
            state["algorithm_configuration_completed"] = True
                        
            logger.info(f"Algorithm configuration completed. Execution plan: {len(execution_plan)} algorithms")
            
            # Request HITL for algorithm configuration review if interactive mode
            if state.get("interactive", False):
                state = self.request_hitl(
                    state,
                    payload={
                        "step": "algorithm_configuration",
                        "phase": "causal_discovery",
                        "description": "Please review the algorithm execution plan",
                        "decisions": ["approve", "edit", "rerun", "abort"]
                    },
                    hitl_type="algorithm_config_review"
                )
            
            return state
            
        except Exception as e:
            logger.error(f"Algorithm configuration failed: {str(e)}")
            state["error"] = f"Algorithm configuration failed: {str(e)}"
            return state
    
    def _generate_execution_plan(self, data_profile: Dict[str, Any], variable_schema: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Generate execution plan based on data profile"""
        execution_plan = []
        
        # Extract basic checks information
        basic_checks = data_profile.get("basic_checks", {})
        n_samples = data_profile.get("n_samples", 1000)
        data_type_profile = basic_checks.get("data_type_profile", "Mixed")
        ci_reliability = basic_checks.get("ci_reliability", "High")
        high_cardinality = basic_checks.get("high_cardinality", False)
        
        global_scores = data_profile.get("global_scores", {})
        pairwise_scores = data_profile.get("pairwise_scores", {})
        
        # Default CI test 
        default_ci_test = "fisherz"
        if data_type_profile == "Pure Categorical":
            default_ci_test = "gsq"
        elif ci_reliability != "High":  # p > n 또는 High-Dimensional
            if n_samples <= 1000 :
                default_ci_test = "kernel_kcit"
        
        # Linearity 
        global_linearity_pvalue = global_scores.get("s_global_linearity_pvalue", np.nan)
        pairwise_linearity_score = pairwise_scores.get("s_pairwise_linearity_score", np.nan)
        
        is_mostly_linear = None  # 기본 가정
        
        if data_type_profile == "Pure Continuous" and not np.isnan(pairwise_linearity_score):
            # 1st priority: Pure Continuous uses Pairwise score
            is_mostly_linear = True if pairwise_linearity_score >= 0.5 else False
        elif not np.isnan(global_linearity_pvalue):
            # 2nd priority: Mixed data or failed Pairwise uses Global (Ramsey RESET) score
            is_mostly_linear = True if global_linearity_pvalue >= 0.05 else False   
        
        
        # Scenario 6: High Cardinality
        if high_cardinality:
            logger.info("Scenario 6: High Cardinality")
            execution_plan.extend([
                {"alg": "PC", "ci_test": "kernel_kcit"},
                {"alg": "FCI", "ci_test": "kernel_kcit"},
            ])
        
        # Scenario 5: Pure Categorical
        elif data_type_profile == "Pure Categorical":
            logger.info("Scenario 5: Pure Categorical")
            execution_plan.extend([
                {"alg": "PC", "ci_test": "gsq"},
                {"alg": "FCI", "ci_test": "gsq"},
                {"alg": "GES", "score": "bic-d"}, 
            ])
        
        # Scenario 1: Pure Continuous, Linear
        elif data_type_profile == "Pure Continuous" and is_mostly_linear:
            logger.info("Scenario 1: Pure Continuous, Linear")
            
            # 정규성(Gaussianity) 확인 (GES, PC용)
            is_gaussian = global_scores.get("s_global_normality_pvalue", 0.0) >= 0.05
            
            pc_ci_test = "fisherz" if (is_gaussian and ci_reliability == "High") else "kernel_kcit"
            fci_ci_test = "fisherz" if ci_reliability == "High" else "kernel_kcit"
            
            execution_plan.extend([
                {"alg": "PC", "ci_test": pc_ci_test},
                {"alg": "GES", "score": "bic-g" if is_gaussian else "generalized_rkhs"},  # 정규성이면 bic-g, 아니면 rkhs
                {"alg": "FCI", "ci_test": fci_ci_test},
                {"alg": "NOTEARS-linear"}
            ])
            
            # LiNGAM: Linear + Non-Gaussianity
            non_gaussian_score = pairwise_scores.get("s_pairwise_non_gaussianity_score", 0.0)
            if non_gaussian_score >= 0.5:  # 비정규성 점수가 0.5 이상일 때만
                execution_plan.append({"alg": "LiNGAM"})
        
        # Scenario 2: Pure Continuous, Nonlinear
        elif data_type_profile == "Pure Continuous" and not is_mostly_linear:
            logger.info("Scenario 2: Pure Continuous, Nonlinear")
          
            execution_plan.extend([
                {"alg": "PC", "ci_test": "kernel_kcit"},
                {"alg": "GES", "score": "generalized_rkhs"},
                {"alg": "FCI", "ci_test": "kernel_kcit"},
                {"alg": "NOTEARS-nonlinear"}
            ])
            
            # ANM: 비선형 + ANM 적합성
            anm_score = pairwise_scores.get("s_pairwise_anm_score", 0.0)
            if anm_score >= 0.5:  # ANM 적합성 점수가 0.5 이상일 때만
                execution_plan.append({"alg": "ANM"})
        
        # Scenario 3: Mixed Data, Linear (High Cardinality가 아닐 때)
        elif data_type_profile == "Mixed" and is_mostly_linear:
            logger.info("Scenario 3: Mixed Data, Linear")
            execution_plan.extend([
                {"alg": "PC", "ci_test": "lrt"},  # Linear Mixed
                {"alg": "GES", "score": "bic-cg"},  # Linear Mixed (CG-BIC)
                {"alg": "LiM"},
                {"alg": "FCI", "ci_test": "kernel_kcit"}
            ])
        
        # Scenario 4: Mixed Data, Nonlinear 
        elif data_type_profile == "Mixed" and not is_mostly_linear:
            logger.info("Scenario 4: Mixed Data, Nonlinear")
            execution_plan.extend([
                {"alg": "PC", "ci_test": "kernel_kcit"},  # Nonlinear Mixed
                {"alg": "FCI", "ci_test": "kernel_kcit"}
            ])
        
        # Default (이론상 도달 불가능해야)
        if not execution_plan:
            logger.warning("No scenario matched, using default execution plan")
            default_ci_test = "kernel_kcit" if ci_reliability != "High" else "fisherz"
            if data_type_profile == "Pure Categorical":
                default_ci_test = "gsq"
            
            execution_plan.extend([
                {"alg": "PC", "ci_test": default_ci_test},
                {"alg": "FCI", "ci_test": default_ci_test}
            ])
        
        return execution_plan

    # def _generate_execution_plan(
    #     self,
    #     data_profile: Dict[str, Any],
    #     variable_schema: Optional[Dict[str, Any]] = None
    # ) -> List[Dict[str, Any]]:
    #     """Generate execution plan based on data profile (n-sample aware, kernel-gated)."""
    #     execution_plan: List[Dict[str, Any]] = []

    #     # ---------------------------
    #     # Config knobs (tune as needed)
    #     # ---------------------------
    #     N_KERNEL_SOFT_CUTOFF = 2000   # >= soft: kernel only with subsampling
    #     N_KERNEL_HARD_CUTOFF = 5000   # >= hard: kernel forbidden
    #     KCI_SUBSAMPLE_SIZE   = 1500   # subsample size when kernel_sampled_only
    #     KCI_MAX_PERMUTATIONS = 200    # limit permutations/bootstrapping (if supported)

    #     D_PCFI_CUTOFF = 30            # if d > cutoff => exclude PC/FCI
    #     # (optional) you can set a tighter cutoff for FCI only if you want:
    #     # D_FCI_CUTOFF = 20

    #     # ---------------------------
    #     # Extract profile
    #     # ---------------------------
    #     basic_checks = data_profile.get("basic_checks", {})
    #     n_samples = int(data_profile.get("n_samples", 1000))
    #     data_type_profile = basic_checks.get("data_type_profile", "Mixed")
    #     ci_reliability = basic_checks.get("ci_reliability", "High")
    #     high_cardinality = bool(basic_checks.get("high_cardinality", False))

    #     global_scores = data_profile.get("global_scores", {})
    #     pairwise_scores = data_profile.get("pairwise_scores", {})

    #     # ---------------------------
    #     # Determine d (num variables)
    #     # ---------------------------
    #     d = None
    #     if variable_schema:
    #         # try common layouts
    #         if isinstance(variable_schema.get("variables"), list):
    #             d = len(variable_schema["variables"])
    #         elif isinstance(variable_schema.get("variables"), dict):
    #             d = len(variable_schema["variables"])
    #         elif isinstance(variable_schema.get("columns"), list):
    #             d = len(variable_schema["columns"])
    #     if d is None:
    #         d = int(basic_checks.get("n_variables", 0)) or None  # optional if you store it
    #     # if still None, we just won't apply d cutoff

    #     # ---------------------------
    #     # Kernel policy (n-sample gated)
    #     # ---------------------------
    #     kernel_forbidden = (n_samples >= N_KERNEL_HARD_CUTOFF)
    #     kernel_sampled_only = (N_KERNEL_SOFT_CUTOFF <= n_samples < N_KERNEL_HARD_CUTOFF)

    #     def kci_params() -> Dict[str, Any]:
    #         """Parameters to keep kernel CI tests tractable (if runner supports them)."""
    #         params = {"max_permutations": KCI_MAX_PERMUTATIONS}
    #         if kernel_sampled_only:
    #             params["subsample"] = KCI_SUBSAMPLE_SIZE
    #         return params

    #     # ---------------------------
    #     # Linearity 판단 (기존 로직 유지)
    #     # ---------------------------
    #     global_linearity_pvalue = global_scores.get("s_global_linearity_pvalue", np.nan)
    #     pairwise_linearity_score = pairwise_scores.get("s_pairwise_linearity_score", np.nan)

    #     is_mostly_linear = None
    #     if data_type_profile == "Pure Continuous" and not np.isnan(pairwise_linearity_score):
    #         is_mostly_linear = True if pairwise_linearity_score >= 0.5 else False
    #     elif not np.isnan(global_linearity_pvalue):
    #         is_mostly_linear = True if global_linearity_pvalue >= 0.05 else False

    #     # ---------------------------
    #     # d cutoff: exclude PC/FCI when too many variables
    #     # ---------------------------
    #     exclude_pc_fci = (d is not None and d > D_PCFI_CUTOFF)

    #     # ---------------------------
    #     # Fast CI test defaults (by data type)
    #     # ---------------------------
    #     def fast_ci_test_for_pc_fci() -> str:
    #         if data_type_profile == "Pure Categorical":
    #             return "gsq"
    #         if data_type_profile == "Mixed":
    #             # linear mixed는 lrt가 1순위
    #             return "lrt"
    #         # Pure Continuous
    #         return "fisherz"

    #     # ---------------------------
    #     # Decide CI tests for PC/FCI with new rules
    #     # ---------------------------
    #     def choose_pc_ci() -> Dict[str, Any]:
    #         """
    #         Prefer fast tests; kernel only if allowed AND really needed.
    #         - Pure Continuous + Linear: fisherz default (do not flip to kernel by gaussianity pvalue)
    #         - Nonlinear cases: kernel allowed only under n cutoff
    #         - ci_reliability 낮아도, 큰 n에서는 kernel 금지 → fast test + (optionally) limit depth elsewhere
    #         """
    #         ci = fast_ci_test_for_pc_fci()

    #         # Pure Continuous + Nonlinear or Mixed + Nonlinear: kernel could help, but only if allowed
    #         nonlinear = (is_mostly_linear is False)
    #         if nonlinear and (not kernel_forbidden):
    #             # allow kernel (sampled_only handled via ci_params)
    #             return {"ci_test": "kernel_kcit", "ci_params": kci_params()}

    #         # ci_reliability가 낮아서 kernel을 쓰고 싶어도, n 크면 금지
    #         if (ci_reliability != "High") and (not kernel_forbidden) and (n_samples < N_KERNEL_SOFT_CUTOFF):
    #             return {"ci_test": "kernel_kcit", "ci_params": kci_params()}

    #         return {"ci_test": ci}

    #     def choose_fci_ci() -> Dict[str, Any]:
    #         """
    #         FCI는 kernel이 가장 잘 터지므로 더 보수적으로:
    #         - Mixed Linear: lrt 우선 (kernel은 n 작을 때만)
    #         - Pure Continuous Linear: fisherz 우선
    #         - Nonlinear: kernel은 n 작을 때만 (soft/hard 정책 적용)
    #         """
    #         ci = fast_ci_test_for_pc_fci()
    #         nonlinear = (is_mostly_linear is False)

    #         if nonlinear and (not kernel_forbidden):
    #             return {"ci_test": "kernel_kcit", "ci_params": kci_params()}

    #         if (ci_reliability != "High") and (not kernel_forbidden) and (n_samples < N_KERNEL_SOFT_CUTOFF):
    #             return {"ci_test": "kernel_kcit", "ci_params": kci_params()}

    #         return {"ci_test": ci}

    #     # ---------------------------
    #     # Scenario handling (updated)
    #     # ---------------------------

    #     # Scenario 6: High Cardinality
    #     if high_cardinality:
    #         logger.info("Scenario 6: High Cardinality (kernel forbidden + prefer fast tests)")
    #         # high-cardinality + kernel is worst -> avoid kernel; also FCI is often not worth it.
    #         if not exclude_pc_fci:
    #             pc_ci = fast_ci_test_for_pc_fci()
    #             execution_plan.append({"alg": "PC", "ci_test": pc_ci})
    #             # FCI 제외(기본). 꼭 필요하면 아래 주석 해제 + fast ci로만
    #             # execution_plan.append({"alg": "FCI", "ci_test": pc_ci})
    #         # 대안들
    #         if data_type_profile == "Pure Categorical":
    #             execution_plan.append({"alg": "GES", "score": "bic-d"})
    #         elif data_type_profile == "Mixed":
    #             execution_plan.append({"alg": "GES", "score": "bic-cg"})
    #             execution_plan.append({"alg": "LiM"})
    #         else:  # Pure Continuous
    #             execution_plan.append({"alg": "GES", "score": "bic-g"})
    #             execution_plan.append({"alg": "NOTEARS-linear" if (is_mostly_linear is True) else "NOTEARS-nonlinear"})

    #     # Scenario 5: Pure Categorical
    #     elif data_type_profile == "Pure Categorical":
    #         logger.info("Scenario 5: Pure Categorical")
    #         execution_plan.append({"alg": "GES", "score": "bic-d"})
    #         if not exclude_pc_fci:
    #             execution_plan.append({"alg": "PC", "ci_test": "gsq"})
    #             execution_plan.append({"alg": "FCI", "ci_test": "gsq"})

    #     # Scenario 1: Pure Continuous, Linear
    #     elif data_type_profile == "Pure Continuous" and is_mostly_linear:
    #         logger.info("Scenario 1: Pure Continuous, Linear (prefer fisherz; no gaussianity flip)")
    #         # GES scoring만 gaussianity로 분기 유지(이건 CI test 비용이랑 별개라 OK)
    #         is_gaussian = global_scores.get("s_global_normality_pvalue", 0.0) >= 0.05

    #         execution_plan.extend([
    #             {"alg": "GES", "score": "bic-g" if is_gaussian else "generalized_rkhs"},
    #             {"alg": "NOTEARS-linear"},
    #         ])

    #         if not exclude_pc_fci:
    #             pc_sel = choose_pc_ci()   # 대부분 fisherz로 떨어짐
    #             fci_sel = choose_fci_ci()
    #             execution_plan.insert(0, {"alg": "PC", **pc_sel})
    #             execution_plan.insert(2, {"alg": "FCI", **fci_sel})

    #         # LiNGAM: Linear + Non-Gaussianity
    #         non_gaussian_score = pairwise_scores.get("s_pairwise_non_gaussianity_score", 0.0)
    #         if non_gaussian_score >= 0.5:
    #             execution_plan.append({"alg": "LiNGAM"})

    #     # Scenario 2: Pure Continuous, Nonlinear
    #     elif data_type_profile == "Pure Continuous" and is_mostly_linear is False:
    #         logger.info("Scenario 2: Pure Continuous, Nonlinear (kernel only if allowed)")
    #         execution_plan.extend([
    #             {"alg": "GES", "score": "generalized_rkhs"},
    #             {"alg": "NOTEARS-nonlinear"},
    #         ])

    #         if not exclude_pc_fci:
    #             pc_sel = choose_pc_ci()
    #             fci_sel = choose_fci_ci()
    #             execution_plan.insert(0, {"alg": "PC", **pc_sel})
    #             execution_plan.insert(2, {"alg": "FCI", **fci_sel})

    #         anm_score = pairwise_scores.get("s_pairwise_anm_score", 0.0)
    #         if anm_score >= 0.5:
    #             execution_plan.append({"alg": "ANM"})

    #     # Scenario 3: Mixed Data, Linear
    #     elif data_type_profile == "Mixed" and is_mostly_linear:
    #         logger.info("Scenario 3: Mixed Data, Linear (FCI not forced to kernel)")
    #         execution_plan.extend([
    #             {"alg": "GES", "score": "bic-cg"},
    #             {"alg": "LiM"},
    #         ])

    #         if not exclude_pc_fci:
    #             # PC: lrt, FCI: lrt 우선 (kernel은 n 작을 때만)
    #             pc_sel = {"ci_test": "lrt"}
    #             fci_sel = choose_fci_ci()
    #             execution_plan.insert(0, {"alg": "PC", **pc_sel})
    #             execution_plan.insert(2, {"alg": "FCI", **fci_sel})

    #     # Scenario 4: Mixed Data, Nonlinear
    #     elif data_type_profile == "Mixed" and is_mostly_linear is False:
    #         logger.info("Scenario 4: Mixed Data, Nonlinear (kernel only if allowed; else skip PC/FCI)")
    #         # Mixed nonlinear은 kernel이 없으면 PC/FCI로 얻는 이득이 제한적일 수 있음
    #         if (not exclude_pc_fci) and (not kernel_forbidden):
    #             pc_sel = choose_pc_ci()
    #             fci_sel = choose_fci_ci()
    #             execution_plan.extend([
    #                 {"alg": "PC", **pc_sel},
    #                 {"alg": "FCI", **fci_sel},
    #             ])
    #         else:
    #             # kernel 금지/또는 d 큼: PC/FCI 제외하고 대안 위주
    #             execution_plan.extend([
    #                 {"alg": "GES", "score": "bic-cg"},  # mixed에서도 baseline으로 유용
    #                 {"alg": "LiM"},
    #             ])

    #     # Default fallback
    #     if not execution_plan:
    #         logger.warning("No scenario matched, using safe default execution plan")
    #         # safe fast default
    #         if data_type_profile == "Pure Categorical":
    #             fast_ci = "gsq"
    #         elif data_type_profile == "Mixed":
    #             fast_ci = "lrt"
    #         else:
    #             fast_ci = "fisherz"

    #         if not exclude_pc_fci:
    #             execution_plan.extend([
    #                 {"alg": "PC", "ci_test": fast_ci},
    #                 {"alg": "FCI", "ci_test": fast_ci},
    #             ])
    #         else:
    #             execution_plan.append({"alg": "GES", "score": "bic-cg" if data_type_profile == "Mixed" else "bic-g"})

    #     return execution_plan
            
    def _run_algorithms_portfolio(self, state: AgentState) -> AgentState:
        """Stage 3: Run algorithms from execution_plan in parallel"""
        logger.info("Running algorithm portfolio in parallel...")
        
        # Get event logger if available
        event_logger = None
        try:
            from monitoring.metrics.collector import get_metrics_collector
            collector = get_metrics_collector()
            if collector and hasattr(collector, 'event_logger'):
                event_logger = collector.event_logger
        except Exception:
            pass
        
        try:
            execution_plan = state.get("execution_plan", [])
            df = self._load_dataframe_from_state(state)
            data_profile = state.get("data_profile", {})
            variable_schema = state.get("variable_schema", {})
            
            if df is None:
                raise ValueError("No preprocessed data available")
            
            algorithm_configs = execution_plan
            
            if not algorithm_configs:
                raise ValueError("No algorithms to execute")
            
            logger.info(f"Executing {len(algorithm_configs)} algorithms from execution plan")
            
            # Run algorithms in parallel
            algorithm_results = {}
            executor = ThreadPoolExecutor(max_workers=len(algorithm_configs))
            try:
                futures = {}
                algorithm_timeout = 300
                for config in algorithm_configs:
                    alg_name = config["alg"]
                    
                    # Log algorithm start
                    if event_logger:
                        event_logger.log_tool_call_start(
                            tool_name=f"algorithm_{alg_name}",
                            step_id="2",
                            metadata={"algorithm": alg_name, "config": config}
                        )
                    
                    future = executor.submit(self._dispatch_algorithm, config, df, data_profile, variable_schema)
                    futures[alg_name] = future

                for alg_name, future in futures.items():
                    import time
                    alg_start = time.time()
                    try:
                        result = future.result(timeout=algorithm_timeout)
                        algorithm_results[alg_name] = result
                        logger.info(f"Algorithm {alg_name} completed successfully")
                        
                        # Log algorithm success
                        if event_logger:
                            duration = time.time() - alg_start
                            event_logger.log_tool_call_end(
                                tool_name=f"algorithm_{alg_name}",
                                duration=duration,
                                success=True,
                                step_id="2",
                                metadata={
                                    "algorithm": alg_name,
                                    "nodes": result.get("num_nodes", 0) if isinstance(result, dict) else 0,
                                    "edges": result.get("num_edges", 0) if isinstance(result, dict) else 0
                                }
                            )
                    except TimeoutError:
                        cancelled = future.cancel()
                        logger.error(f"Algorithm {alg_name} timeout after {algorithm_timeout}s. cancel()={cancelled} (thread may still be running)")
                        algorithm_results[alg_name] = {"error": "timeout"}
                        
                        # Log algorithm timeout
                        if event_logger:
                            duration = time.time() - alg_start
                            event_logger.log_tool_call_end(
                                tool_name=f"algorithm_{alg_name}",
                                duration=duration,
                                success=False,
                                error="timeout",
                                step_id="2",
                                metadata={"algorithm": alg_name}
                            )
                    except Exception as e:
                        logger.error(f"Algorithm {alg_name} failed: {e}")
                        algorithm_results[alg_name] = {"error": str(e)}
                        
                        # Log algorithm failure
                        if event_logger:
                            duration = time.time() - alg_start
                            event_logger.log_tool_call_end(
                                tool_name=f"algorithm_{alg_name}",
                                duration=duration,
                                success=False,
                                error=str(e),
                                step_id="2",
                                metadata={"algorithm": alg_name}
                            )
            finally:
                executor.shutdown(wait=False, cancel_futures=True)  
            
            algorithm_results = _convert_numpy_types(algorithm_results)
            
            # Store results in Redis if large
            if len(algorithm_results) > 5:  # threshold
                try:
                    from utils.redis_client import redis_client
                    import json
                    session_id = state.get("session_id", "default")
                    redis_key = f"{state.get('db_id', 'default')}:algorithm_results:{session_id}"
                    redis_client.set(redis_key, json.dumps(algorithm_results))
                    state["algorithm_results_key"] = redis_key
                    logger.info(f"Algorithm results stored in Redis: {redis_key}")
                except Exception as e:
                    logger.warning(f"Failed to store results in Redis: {e}")
            
            # Update state
            state["algorithm_results"] = algorithm_results
            state["executed_algorithms"] = list(algorithm_results.keys())
            state["run_algorithms_portfolio_completed"] = True
            
            logger.info("Algorithm portfolio execution completed")
            
            # Request HITL for algorithm results review if interactive mode
            if state.get("interactive", False):
                state = self.request_hitl(
                    state,
                    payload={
                        "step": "run_algorithms_portfolio",
                        "phase": "causal_discovery",
                        "description": "Please review the algorithm execution results",
                        "decisions": ["approve", "edit", "rerun", "abort"]
                    },
                    hitl_type="algorithm_results_review"
                )
            
            return state
            
        except Exception as e:
            logger.error(f"Algorithm portfolio execution failed: {str(e)}")
            state["error"] = f"Algorithm portfolio execution failed: {str(e)}"
            return state
    
    def _graph_scoring(self, state: AgentState) -> AgentState:
        """Stage 4: Graph scoring"""
        logger.info("Performing graph scoring...")
        
        try:
            algorithm_results = self._load_algorithm_results_from_state(state)
            df = self._load_dataframe_from_state(state)
            execution_plan = state.get("execution_plan", [])
            
            if not algorithm_results:
                raise ValueError("No algorithm results available")
            if df is None:
                raise ValueError("No preprocessed data available")
            
            scored_graphs = []
            
            for alg_name, result in algorithm_results.items():
                if "error" in result:
                    logger.warning(f"Skipping {alg_name} due to execution error: {result['error']}")
                    continue
                
                logger.info(f"Scoring graph from {alg_name}...")
                
                # Calculate 3 scores in parallel for this graph
                scores = self._calculate_graph_scores(result, alg_name, df, execution_plan)
                
                scored_graphs.append({
                    "algorithm": alg_name,
                    "graph": result,
                    "markov_consistency": scores["markov_consistency"],
                    "sampling_stability": scores["sampling_stability"],
                    "structural_stability": scores["structural_stability"]
                })
            
            # Convert numpy types to Python native types for msgpack serialization
            scored_graphs = _convert_numpy_types(scored_graphs)
            
            # Update state
            state["scored_graphs"] = scored_graphs
            state["graph_scoring_completed"] = True
            
            logger.info(f"Graph scoring completed. Scored {len(scored_graphs)} graphs")
            
            # Request HITL for graph scoring review if interactive mode
            if state.get("interactive", False):
                state = self.request_hitl(
                    state,
                    payload={
                        "step": "graph_scoring",
                        "phase": "causal_discovery",
                        "description": "Please review the graph scoring results",
                        "decisions": ["approve", "edit", "rerun", "abort"]
                    },
                    hitl_type="graph_scoring_review"
                )
            
            return state
            
        except Exception as e:
            logger.error(f"Graph scoring failed: {str(e)}")
            state["error"] = f"Graph scoring failed: {str(e)}"
            return state
    
    def _graph_evaluation(self, state: AgentState) -> AgentState:
        """Stage 5: Graph evaluation - Evaluate and rank graphs using composite scoring or sequential pruning"""
        logger.info("Performing scorecard evaluation...")
        
        try:
            scored_graphs = state.get("scored_graphs", [])
            user_params = state.get("user_params", {})
            ranking_mode = user_params.get("ranking_mode", "composite_score")
            
            if not scored_graphs:
                raise ValueError("No scored graphs available")
            
            # Build scorecard from pre-calculated scores
            scorecard = []
            
            for scored_graph in scored_graphs:
                alg_name = scored_graph["algorithm"]
                graph = scored_graph["graph"]
                
                logger.info(f"Evaluating {alg_name} for scorecard...")
                

                scorecard.append({
                    "algorithm": alg_name,
                    "graph_id": f"{alg_name}_{id(graph)}",
                    "markov_consistency": scored_graph["markov_consistency"],
                    "sampling_stability": scored_graph["sampling_stability"],
                    "structural_stability": scored_graph["structural_stability"],
                    "graph": graph
                })
            
            # Apply ranking mode
            if ranking_mode == "sequential_pruning":
                ranked_graphs = self._sequential_pruning(scorecard, user_params)
            else:
                # Default: composite_score
                ranked_graphs = self._composite_score_ranking(scorecard)
            
            # Select top candidates
            top_candidates = ranked_graphs[:3]  # Top 3 graphs
            
            # Convert numpy types to Python native types for msgpack serialization
            scorecard = _convert_numpy_types(scorecard)
            ranked_graphs = _convert_numpy_types(ranked_graphs)
            top_candidates = _convert_numpy_types(top_candidates)
            
            # Update state
            state["scorecard"] = scorecard
            state["ranked_graphs"] = ranked_graphs
            state["top_candidates"] = top_candidates
            state["graph_evaluation_completed"] = True
            
            logger.info(f"Graph evaluation completed. Top candidate: {top_candidates[0]['algorithm'] if top_candidates else 'None'}")
            
            # Request HITL for graph evaluation review if interactive mode
            if state.get("interactive", False):
                state = self.request_hitl(
                    state,
                    payload={
                        "step": "graph_evaluation",
                        "phase": "causal_discovery",
                        "description": "Please review the graph evaluation and ranking results",
                        "decisions": ["approve", "edit", "rerun", "abort"]
                    },
                    hitl_type="graph_evaluation_review"
                )
            
            return state
            
        except Exception as e:
            logger.error(f"Graph evaluation failed: {str(e)}")
            state["error"] = f"Graph evaluation failed: {str(e)}"
            return state
    
    def _get_dynamic_metric_functions(self, execution_plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get appropriate metric functions for each algorithm in execution_plan"""
        metric_functions = {}
                
        # Find PC/FCI config to determine markov_consistency function
        pc_config = next((c for c in execution_plan if c["alg"] in ["PC", "FCI"]), None)
        if pc_config and pc_config.get("ci_test") == "lrt":
            def lrt_consistency(graph, candidate, df):
                # Use LRT-based consistency (1 - violation_ratio)
                return 1.0 - candidate.get("violation_ratio", 1.0)
            metric_functions["markov_consistency"] = lrt_consistency
        
        return metric_functions
    
    def _composite_score_ranking(self, scorecard: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank graphs using composite score
        
        For DAG: Uses 3 scores (markov_consistency, sampling_stability, structural_stability)
        For CPDAG: Uses 2 scores (sampling_stability, structural_stability) with weight renormalization
        For PAG: Uses only sampling_stability (composite_score = sampling_stability)
        """
        for item in scorecard:
            graph = item["graph"]
            graph_type = get_graph_type(graph)
            is_pag = graph_type == "PAG"
            is_cpdag = graph_type == "CPDAG"
            
            markov_consistency = item.get("markov_consistency")
            sampling_stability = item.get("sampling_stability")
            structural_stability = item.get("structural_stability")
            
            if is_pag:
                if sampling_stability is None:
                    composite_score = 0.0
                else:
                    composite_score = sampling_stability
            elif is_cpdag:
                # Collect available scores and their weights
                available_scores = {}
                available_weights = {}
                
                if sampling_stability is not None:
                    available_scores["sampling_stability"] = sampling_stability
                    available_weights["sampling_stability"] = self.composite_weights["sampling_stability"]
                
                if structural_stability is not None:
                    available_scores["structural_stability"] = structural_stability
                    available_weights["structural_stability"] = self.composite_weights["structural_stability"]
                
                # Normalize weights to sum to 1.0 for available scores
                total_weight = sum(available_weights.values())
                if total_weight > 0 and len(available_scores) > 0:
                    composite_score = sum(
                        (available_weights[key] / total_weight) * score
                        for key, score in available_scores.items()
                    )
                else:
                    composite_score = 0.0
            else:
                available_scores = {}
                available_weights = {}
                
                if markov_consistency is not None:
                    available_scores["markov_consistency"] = markov_consistency
                    available_weights["markov_consistency"] = self.composite_weights["markov_consistency"]
                
                if sampling_stability is not None:
                    available_scores["sampling_stability"] = sampling_stability
                    available_weights["sampling_stability"] = self.composite_weights["sampling_stability"]
                
                if structural_stability is not None:
                    available_scores["structural_stability"] = structural_stability
                    available_weights["structural_stability"] = self.composite_weights["structural_stability"]
                
                # Normalize weights to sum to 1.0 for available scores
                total_weight = sum(available_weights.values())
                if total_weight > 0 and len(available_scores) > 0:
                    composite_score = sum(
                        (available_weights[key] / total_weight) * score
                        for key, score in available_scores.items()
                    )
                else:
                    composite_score = 0.0
            
            # Penalty for low edge count
            n_edges = len(get_edges(graph))
            if n_edges == 0:
                composite_score *= 0.1

            item["composite_score"] = composite_score
        
        # Sort by composite_score, with tie-breaking: DAG algorithms first, then by sampling_stability
        ranked = sorted(scorecard, key=lambda x: (
            x["composite_score"],
            get_graph_type(x["graph"]) == "DAG",
            x.get("sampling_stability", 0.0)  
        ), reverse=True)
        return ranked
    
    def _sequential_pruning(self, scorecard: List[Dict[str, Any]], 
                           user_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        pruning_thresholds = user_params.get("pruning_thresholds", {
            "consistency": 0.05,
            "sampling": 0.2
        })
        
        # Step 1: Filter by markov_consistency (skip for PAG/CPDAG where it's None)
        step1 = []
        for g in scorecard:
            graph = g["graph"]
            graph_type = get_graph_type(graph)
            is_pag = graph_type == "PAG"
            is_cpdag = graph_type == "CPDAG"
            markov_consistency = g.get("markov_consistency")
            
            # Skip filtering for PAG/CPDAG (markov_consistency is None)
            if is_pag or is_cpdag or (markov_consistency is not None and markov_consistency >= pruning_thresholds["consistency"]):
                step1.append(g)
        
        # Step 2: Filter by sampling_stability
        step2 = [g for g in step1 if g["sampling_stability"] >= pruning_thresholds["sampling"]]
        
        # Step 3: Sort by structural_stability (None values go to end)
        final_ranked = sorted(step2, key=lambda x: (x["structural_stability"] is None, -(x.get("structural_stability") or 0.0))
)
        
        return final_ranked
    
    def _calculate_graph_scores(self, graph: Dict[str, Any], alg_name: str, df: pd.DataFrame, 
                                execution_plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate 3 scores for a single graph with parallelization
        
        Args:
            graph: Graph result dictionary
            alg_name: Algorithm name
            df: DataFrame for evaluation
            execution_plan: Execution plan for dynamic metric functions
            
        Returns:
            Dictionary with 3 scores: markov_consistency, sampling_stability, structural_stability
        """
        def calculate_markov_consistency():
            """Calculate Markov consistency score"""
            try:
                graph_type = get_graph_type(graph)
                # CPDAG is MEC (equivalence class), not a single DAG, so skip global_markov_test
                # PAG also skipped due to latent confounders
                if graph_type in ["PAG", "CPDAG"]:
                    return None
                
                metric_functions = self._get_dynamic_metric_functions(execution_plan)
                markov_consistency_func = metric_functions.get("markov_consistency")
                
                markov_result = self.use_tool("pruning_tool", "global_markov_test", graph, df, 
                                             alpha=self.ci_alpha, max_pa_size=self.max_pa_size)
                violation_ratio = markov_result.get("violation_ratio")
                
                # If violation_ratio is None (no successful tests), return None
                if violation_ratio is None:
                    return None
                
                if markov_consistency_func:
                    candidate_dict = {"violation_ratio": violation_ratio}
                    return markov_consistency_func(graph, candidate_dict, df)
                else:
                    # Default: use 1.0 - violation_ratio
                    return 1.0 - violation_ratio 
            except Exception as e:
                logger.warning(f"Markov consistency calculation failed for {alg_name}: {e}")
                return None
        
        def calculate_sampling_stability():
            """Calculate sampling stability score
            
            Uses robustness_score which focuses on edge reproducibility (based on successful iterations only).
            Alternative: effective_score = robustness_score * success_rate (penalizes algorithm/data reliability issues)
            """
            try:
                bootstrap_result = self.use_tool("bootstrapper", "bootstrap_evaluation", df, graph, alg_name, self.bootstrap_iterations)
                # Use robustness_score: mean confidence across original edges (based on successful iterations)
                # For failure-aware scoring, use: bootstrap_result.get("effective_score", 0.5)
                return bootstrap_result.get("robustness_score", 0.5)
            except Exception as e:
                logger.warning(f"Sampling stability calculation failed for {alg_name}: {e}")
                return 0.5
        
        def calculate_structural_stability():
            """Calculate structural stability score"""
            try:
                consistency_result = self.use_tool("pruning_tool", "structural_consistency_test", graph, df, alg_name, n_subsets=self.n_subsets)
                instability_score = consistency_result.get("instability_score")
                
                # PAG graphs return None (test skipped)
                if instability_score is None:
                    return None
                
                return 1.0 - instability_score
            except Exception as e:
                logger.warning(f"Structural stability calculation failed for {alg_name}: {e}")
                return 0.0
        
        # Calculate 3 scores in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_consistency = executor.submit(calculate_markov_consistency)
            future_sampling = executor.submit(calculate_sampling_stability)
            future_structural = executor.submit(calculate_structural_stability)
            
            # Collect results
            markov_consistency = future_consistency.result()
            sampling_stability = future_sampling.result()
            structural_stability = future_structural.result()
        
        return {
            "markov_consistency": markov_consistency,
            "sampling_stability": sampling_stability,
            "structural_stability": structural_stability
        }

    # === Helpers and dispatch ===

    def _load_dataframe_from_state(self, state: AgentState) -> Optional[pd.DataFrame]:
        # First, try to get DataFrame directly from df_preprocessed
        df = state.get("df_preprocessed")
        if isinstance(df, pd.DataFrame):
            return df
        
        # If no DataFrame, load from df_redis_key
        redis_key = state.get("df_redis_key")
        if redis_key:
            try:
                from utils.redis_df import load_df_parquet
                df = load_df_parquet(redis_key)
                if df is not None:
                    return df
            except Exception as e:
                logger.warning(f"Failed to load DataFrame from Redis key {redis_key}: {e}")
                return None

        return None

    def _load_algorithm_results_from_state(self, state: AgentState) -> Dict[str, Any]:
        results = state.get("algorithm_results", {})
        if results:
            return results
        if state.get("algorithm_results_key"):
            try:
                from utils.redis_client import redis_client
                import json
                redis_key = state["algorithm_results_key"]
                data = redis_client.get(redis_key)
                if data:
                    return json.loads(data)
            except Exception:
                return {}
        return {}

    def _validate_graph_structure(self, graph: Dict[str, Any]) -> bool:
        """Validate graph structure using unified schema validator"""
        try:
            validate_graph_schema(graph)
            return True
        except ValueError:
            return False

    def _dispatch_algorithm(self, config: Dict[str, Any], df: pd.DataFrame, 
                                       data_profile: Dict = None, variable_schema: Dict = None) -> Dict[str, Any]:
        """Dispatch algorithm with configuration from execution_plan"""
        alg_name = config["alg"]
        ci_test = config.get("ci_test")
        score = config.get("score")
        
        if alg_name == "LiNGAM":
            return self._run_lingam(df)
        if alg_name == "ANM":
            return self._run_anm(df)
        if alg_name == "CAM":
            return self._run_cam(df)
        if alg_name == "PC":
            return self._run_pc(df, data_profile=data_profile, variable_schema=variable_schema, 
                              indep_test=ci_test)
        if alg_name == "GES":
            return self._run_ges(df, score_func=score)
        if alg_name == "FCI":
            return self._run_fci(df, data_profile=data_profile, variable_schema=variable_schema,
                              indep_test=ci_test)
        if alg_name == "LiM":
            return self._run_lim(df, variable_schema=variable_schema)
        if alg_name == "NOTEARS-linear":
            return self._run_notears_linear(df)
        if alg_name == "NOTEARS-nonlinear":
            return self._run_notears_nonlinear(df)
        return {"error": f"Unknown algorithm: {alg_name}"}
    
    
    def _ensemble_synthesis(self, state: AgentState) -> AgentState:
        """Stage 6: Ensemble synthesis with PAG-like and DAG outputs"""
        logger.info("Performing ensemble synthesis...")
        
        try:
            top_candidates = state.get("top_candidates", [])
            data_profile = state.get("data_profile", {})
            
            if not top_candidates:
                raise ValueError("No top candidates available")
            
            # Extract graphs and weights from top candidates
            graphs = []
            weights = []
            
            for candidate in top_candidates:
                graphs.append(candidate["graph"])
                weights.append(candidate["composite_score"])
            
            # 1. Build consensus skeleton
            skeleton_result = self.use_tool("ensemble_tool", "build_consensus_skeleton", graphs, weights=weights)
            
            # 2. Resolve directions
            directions_result = self.use_tool("ensemble_tool", "resolve_directions", skeleton_result, graphs, data_profile, weights=weights)
            
            # 3. Construct PAG-like graph
            pag_result = self.use_tool("ensemble_tool", "construct_pag", directions_result)
            
            # 4. Construct single DAG with tie-breaking
            top_algorithm = top_candidates[0]["algorithm"]
            execution_plan = state.get("execution_plan", [])
            algorithm_results = self._load_algorithm_results_from_state(state)
            dag_result = self.use_tool("ensemble_tool", "construct_dag", 
                                      pag_result, data_profile, top_algorithm,
                                      execution_plan=execution_plan,
                                      algorithm_results=algorithm_results,
                                      top_candidates=top_candidates)
            
            # Generate synthesis reasoning
            reasoning = self._generate_synthesis_reasoning(top_candidates, pag_result, dag_result, data_profile)
            
            # Convert numpy types to Python native types for msgpack serialization
            pag_result = _convert_numpy_types(pag_result)
            dag_result = _convert_numpy_types(dag_result)
            reasoning = _convert_numpy_types(reasoning)
            
            # Update state
            state["consensus_pag"] = pag_result
            state["selected_graph"] = dag_result
            state["synthesis_reasoning"] = reasoning
            state["causal_discovery_status"] = "completed"
            state["ensemble_synthesis_completed"] = True
            
            # Visualize and save DAG
            try:
                from .tools import GraphVisualizer
                visualization_result = GraphVisualizer.save_graph(
                    dag_result, 
                    output_dir="outputs/images/causal_graphs",
                    formats=["png", "svg"]
                )
                if "error" not in visualization_result:
                    state["graph_visualization_path"] = visualization_result.get("saved_paths", {})
                    logger.info(f"DAG visualization saved: {visualization_result.get('saved_paths', {})}")
                else:
                    logger.warning(f"DAG visualization failed: {visualization_result.get('error', 'Unknown error')}")
            except Exception as e:
                logger.warning(f"Failed to visualize DAG: {e}")
            
            # Log final selected graph with details and reasoning
            n_edges = len(get_edges(dag_result))
            n_nodes = len(get_variables(dag_result))
            graph_type = get_graph_type(dag_result)
            
            # Get top candidate scores for logging
            top_candidate_scores = {}
            if top_candidates:
                top_candidate = top_candidates[0]
                top_candidate_scores = {
                    "composite_score": top_candidate.get("composite_score", "N/A"),
                    "markov_consistency": top_candidate.get("markov_consistency", "N/A"),
                    "sampling_stability": top_candidate.get("sampling_stability", "N/A"),
                    "structural_stability": top_candidate.get("structural_stability", "N/A")
                }
            
            # 실험 돌릴 때만 주석처리
            logger.info( 
                f"Final selected graph:\n"
                f"  Algorithm: {top_algorithm}\n"
                f"  Graph Type: {graph_type}\n"
                f"  Nodes: {n_nodes}, Edges: {n_edges}\n"
                f"  Scores: {top_candidate_scores}\n"
                f"  Reasoning: {reasoning}"
            )
            
            logger.info(f"Ensemble synthesis completed. Top algorithm: {top_algorithm}")
            
            # Save graph artifacts if artifact manager is available
            try:
                from monitoring.experiment.utils import get_artifact_manager
                artifact_manager = get_artifact_manager()
                
                if artifact_manager and state.get("selected_graph"):
                    selected_graph = state["selected_graph"]
                    
                    # Save final graph as JSON
                    artifact_manager.save_artifact(
                        artifact_type="graph",
                        data=selected_graph,
                        filename="graph_final.json",
                        step_id="2",
                        metadata={
                            "algorithm": selected_graph.get("algorithm"),
                            "graph_type": selected_graph.get("graph_type"),
                            "num_nodes": selected_graph.get("num_nodes"),
                            "num_edges": selected_graph.get("num_edges")
                        }
                    )
                    
                    # Save adjacency matrix as CSV
                    if "adjacency_matrix" in selected_graph:
                        artifact_manager.save_artifact(
                            artifact_type="graph_adj",
                            data=selected_graph["adjacency_matrix"],
                            filename="graph_final_adj.csv",
                            step_id="2",
                            metadata={}
                        )
            except Exception as e:
                import logging as log
                log.getLogger(__name__).warning(f"Failed to save graph artifacts: {e}")
            
            # Request HITL for ensemble synthesis review if interactive mode
            if state.get("interactive", False):
                state = self.request_hitl(
                    state,
                    payload={
                        "step": "ensemble_synthesis",
                        "phase": "causal_discovery",
                        "description": "Please review the final synthesized causal graph",
                        "decisions": ["approve", "edit", "rerun", "abort"]
                    },
                    hitl_type="ensemble_synthesis_review"
                )
            
            return state
            
        except Exception as e:
            logger.error(f"Ensemble synthesis failed: {str(e)}")
            state["error"] = f"Ensemble synthesis failed: {str(e)}"
            return state
    
    def _generate_synthesis_reasoning(self, top_candidates: List[Dict[str, Any]], pag_result: Dict[str, Any], 
                                    dag_result: Dict[str, Any], data_profile: Dict[str, Any]) -> str:
        """Generate reasoning for ensemble synthesis
        
        Updated to work with new profile structure
        """
        try:
            top_algorithm = top_candidates[0]["algorithm"] if top_candidates else "Unknown"
            
            # Generate summary from new profile structure
            basic_checks = data_profile.get("basic_checks", {})
            global_scores = data_profile.get("global_scores", {})
            
            data_type = basic_checks.get("data_type_profile", "Unknown")
            ci_reliability = basic_checks.get("ci_reliability", "Unknown")
            global_linearity_pvalue = global_scores.get("s_global_linearity_pvalue", 0.5)
            is_mostly_linear = global_linearity_pvalue >= 0.05 if global_linearity_pvalue > 0 else True
            
            profile_summary = f"{data_type}, CI reliability: {ci_reliability}, Mostly linear: {is_mostly_linear}"
            
            # Build candidate scores summary
            candidate_details = []
            for i, candidate in enumerate(top_candidates, 1):
                alg = candidate.get("algorithm", "Unknown")
                comp_score = candidate.get("composite_score", "N/A")
                mc = candidate.get("markov_consistency", "N/A")
                ss = candidate.get("sampling_stability", "N/A")
                sts = candidate.get("structural_stability", "N/A")
                
                score_str = f"composite={comp_score:.3f}" if isinstance(comp_score, (int, float)) else f"composite={comp_score}"
                if isinstance(mc, (int, float)):
                    score_str += f", markov_consistency={mc:.3f}"
                if isinstance(ss, (int, float)):
                    score_str += f", sampling={ss:.3f}"
                if isinstance(sts, (int, float)):
                    score_str += f", structural={sts:.3f}"
                
                candidate_details.append(f"  {i}. {alg}: {score_str}")
            
            candidates_summary = "\n".join(candidate_details) if candidate_details else "  None"
            
            # Get top candidate scores for rationale
            top_candidate = top_candidates[0] if top_candidates else {}
            top_comp_score = top_candidate.get("composite_score", "N/A")
            top_comp_str = f"{top_comp_score:.3f}" if isinstance(top_comp_score, (int, float)) else str(top_comp_score)
            
            reasoning = f"""
Ensemble Synthesis Results:

Top Candidates (ranked by composite score):
{candidates_summary}

Leading Algorithm: {top_algorithm} (composite_score: {top_comp_str})

Data Profile: {profile_summary}

PAG Construction:
- Graph Type: {get_graph_type(pag_result)}
- Edge Count: {len(get_edges(pag_result))}
- Uncertainty Markers: {pag_result.get('metadata', {}).get('uncertainty_markers', False)}

DAG Construction:
- Graph Type: {get_graph_type(dag_result)}
- Edge Count: {len(get_edges(dag_result))}
- Tie-breaking Method: {dag_result.get('metadata', {}).get('construction_method', 'Unknown')}

Selection Rationale:
The ensemble synthesis combines the top {len(top_candidates)} candidates using consensus skeleton building and direction resolution. 
{top_algorithm} was selected as the leading algorithm based on its highest composite score ({top_comp_str}), which balances 
markov consistency, sampling stability, and structural stability according to the data characteristics ({profile_summary}).
The PAG preserves uncertainty information for reporting, while the DAG applies assumption-based tie-breaking for downstream inference tasks.
"""
            
            return reasoning.strip()
            
        except Exception as e:
            logger.warning(f"Synthesis reasoning generation failed: {e}")
            return f"Ensemble synthesis completed with top algorithm: {top_algorithm}"
    
    def _final_graph_selection(self, state: AgentState) -> AgentState:
        """Legacy method for backward compatibility"""
        return self._ensemble_synthesis(state)
    
    # === Assumption Testing Methods ===
    
    def _test_linearity(self, x: pd.Series, y: pd.Series) -> float:
        """Test linearity using GLM vs GAM comparison"""
        try:
            # Use stats tool for GLM vs GAM comparison
            result = self.use_tool("stats_tool", "linearity_test", x, y)
            return result.get("linearity_score", 0.5)
        except Exception as e:
            logger.warning(f"Linearity test failed: {e}")
            return 0.5
    
    def _test_anm(self, x: pd.Series, y: pd.Series) -> float:
        """Test ANM assumption using independence testing"""
        try:
            # Use independence tool for ANM testing
            result = self.use_tool("independence_tool", "anm_test", x, y, n_estimators=self.anm_rf_estimators)
            return result.get("anm_score", 0.5)
        except Exception as e:
            logger.warning(f"ANM test failed: {e}")
            return 0.5
    
    def _test_gaussian_eqvar(self, x: pd.Series, y: pd.Series) -> Tuple[float, float]:
        """Test Gaussian and equal variance assumptions"""
        try:
            # Use stats tool for Gaussian and equal variance tests
            result = self.use_tool("stats_tool", "gaussian_eqvar_test", x, y)
            return result.get("gaussian_score", 0.5), result.get("eqvar_score", 0.5)
        except Exception as e:
            logger.warning(f"Gaussian/EqVar test failed: {e}")
            return 0.5, 0.5
    
    # === Algorithm Execution Methods ===
    
    def _run_lingam(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run LiNGAM algorithm"""
        try:
            result = self.use_tool("lingam_tool", "direct_lingam", df)
            return result
        except Exception as e:
            logger.error(f"LiNGAM execution failed: {e}")
            return {"error": str(e)}
    
    def _run_anm(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Run ANM algorithm"""
        try:
            # Use configurable thresholds with sensible defaults
            delta = kwargs.get('delta', 0.02)  # More lenient than original 0.05
            tau = kwargs.get('tau', 0.05)      # More lenient than original 0.10
            
            result = self.use_tool("anm_tool", "anm_discovery", df, delta=delta, tau=tau)
            return result
        except Exception as e:
            logger.error(f"ANM execution failed: {e}")
            return {"error": str(e)}
    

    def _run_cam(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run CAM algorithm"""
        try:
            result = self.use_tool("cam_tool", "cam_discovery", df)
            return result
        except Exception as e:
            logger.error(f"CAM execution failed: {e}")
            return {"error": str(e)}

    def _run_pc(self, df: pd.DataFrame, data_profile: Dict = None, variable_schema: Dict = None, 
               indep_test: str = None, **kwargs) -> Dict[str, Any]:
        """Run PC algorithm with CI test from execution_plan"""
        try:
            # indep_test should always be provided from execution_plan
            if indep_test is None:
                logger.warning("indep_test not provided from execution_plan, using default fisherz")
                indep_test = "fisherz"
            
            # Pass alpha, indep_test, and variable_schema
            kwargs['alpha'] = kwargs.get('alpha', self.ci_alpha)
            kwargs['indep_test'] = indep_test
            if variable_schema:
                kwargs['variable_schema'] = variable_schema
            
            result = self.use_tool("pc_tool", "pc_discovery", df, **kwargs)
            return result
        except Exception as e:
            logger.error(f"PC execution failed: {e}")
            return {"error": str(e)}

    def _run_ges(self, df: pd.DataFrame, score_func: str = "bic-g") -> Dict[str, Any]:
        """Run GES algorithm with dynamic score function selection
        
        Supported score functions:
        - bic-g: pgmpy bic-g (continuous)
        - bic-d: pgmpy bic-d (categorical)
        - bic-cg: pgmpy bic-cg (mixed)
        - generalized_rkhs: causal-learn generalized RKHS (nonlinear)
        """
        try:
            result = self.use_tool("ges_tool", "ges_discovery", df, score_func=score_func)
            return result
        except Exception as e:
            logger.error(f"GES execution failed: {e}")
            return {"error": str(e)}
        
    def _run_fci(self, df: pd.DataFrame, data_profile: Dict = None, variable_schema: Dict = None,
                indep_test: str = None, **kwargs) -> Dict[str, Any]:
        """Run FCI algorithm with CI test from execution_plan"""
        try:
            # indep_test should always be provided from execution_plan
            if indep_test is None:
                logger.warning("indep_test not provided from execution_plan, using default fisherz")
                indep_test = "fisherz"
            
            # Pass alpha, indep_test, and variable_schema
            kwargs['alpha'] = kwargs.get('alpha', self.ci_alpha)
            kwargs['indep_test'] = indep_test
            if variable_schema:
                kwargs['variable_schema'] = variable_schema
            
            result = self.use_tool("fci_tool", "fci_discovery", df, **kwargs)
            return result
        except Exception as e:
            logger.error(f"FCI execution failed: {e}")
            return {"error": str(e)}
    
    def _run_lim(self, df: pd.DataFrame, variable_schema: Dict = None) -> Dict[str, Any]:
        """Run LiM algorithm for mixed data functional model"""
        try:
            result = self.use_tool("lim_tool", "lim_discovery", df, variable_schema=variable_schema)
            return result
        except Exception as e:
            logger.error(f"LiM execution failed: {e}")
            return {"error": str(e)}
    
    def _run_notears_linear(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Run NOTEARS-linear algorithm"""
        try:
            result = self.use_tool("notears_linear_tool", "discover", df, **kwargs)
            return result
        except Exception as e:
            logger.error(f"NOTEARS-linear execution failed: {e}")
            return {"error": str(e)}
    
    def _run_notears_nonlinear(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Run NOTEARS-nonlinear algorithm"""
        try:
            result = self.use_tool("notears_nonlinear_tool", "discover", df, **kwargs)
            return result
        except Exception as e:
            logger.error(f"NOTEARS-nonlinear execution failed: {e}")
            return {"error": str(e)}
    
    # === Evaluation Methods ===
    
    def _evaluate_robustness(self, df: pd.DataFrame, result: Dict[str, Any], alg_name: str) -> float:
        """Evaluate robustness using bootstrap
        
        Uses robustness_score which focuses on edge reproducibility (based on successful iterations only).
        Alternative: effective_score = robustness_score * success_rate (penalizes algorithm/data reliability issues)
        """
        try:
            bootstrap_result = self.use_tool("bootstrapper", "bootstrap_evaluation", df, result, alg_name, self.bootstrap_iterations)
            # Use robustness_score: mean confidence across original edges (based on successful iterations)
            # This focuses on reproducibility when algorithm succeeds, ignoring failures
            # For failure-aware scoring, use: bootstrap_result.get("effective_score", 0.5)
            return bootstrap_result.get("robustness_score", 0.5)
        except Exception as e:
            logger.warning(f"Robustness evaluation failed: {e}")
            return 0.5
    
    # === Tool Implementations ===
    
    def _safe_execute(self, func, *args, default_return=None, error_msg="Operation failed", **kwargs):
        """Common error handling pattern for tool execution"""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"{error_msg}: {e}")
            return default_return or {"error": str(e)}
    
    def _stats_tool(self, test_type: str, *args, **kwargs) -> Dict[str, Any]:
        """Statistical testing tool implementation"""
        from .tools import StatsTool
        
        if test_type == "linearity_test":
            return StatsTool.linearity_test(args[0], args[1])
        elif test_type == "gaussian_eqvar_test":
            return StatsTool.gaussian_eqvar_test(args[0], args[1])
        else:
            return {"error": f"Unknown test type: {test_type}"}
    
    def _independence_tool(self, test_type: str, *args, **kwargs) -> Dict[str, Any]:
        """Independence testing tool implementation"""
        from .tools import IndependenceTool
        
        if test_type == "anm_test":
            return IndependenceTool.anm_test(args[0], args[1], **kwargs)
        else:
            return {"error": f"Unknown test type: {test_type}"}
    
    def _lingam_tool(self, method: str, *args, **kwargs) -> Dict[str, Any]:
        """LiNGAM algorithm tool implementation"""
        from .tools import LiNGAMTool
        
        if method == "direct_lingam":
            return LiNGAMTool.direct_lingam(args[0])
        else:
            return {"error": f"Unknown method: {method}"}
    
    def _anm_tool(self, method: str, *args, **kwargs) -> Dict[str, Any]:
        """ANM algorithm tool implementation"""
        from .tools import ANMTool
        
        if method == "anm_discovery":
            return ANMTool.anm_discovery(args[0])
        else:
            return {"error": f"Unknown method: {method}"}
    
    def _bootstrapper(self, method: str, *args, **kwargs) -> Dict[str, Any]:
        """Bootstrap tool implementation"""
        from .tools import Bootstrapper
        
        if method == "bootstrap_evaluation":
            return Bootstrapper.bootstrap_evaluation(args[0], args[1], args[2], args[3])
        else:
            return {"error": f"Unknown method: {method}"}
    
    def _graph_evaluator(self, method: str, *args, **kwargs) -> Dict[str, Any]:
        """Graph evaluation tool implementation"""
        from .tools import GraphEvaluator
        
        if method == "cv_evaluation":
            return GraphEvaluator.cv_evaluation(args[0], args[1], args[2])
        else:
            return {"error": f"Unknown method: {method}"}
    
    def _graph_ops(self, method: str, *args, **kwargs) -> Dict[str, Any]:
        """Graph operations tool implementation"""
        from .tools import GraphOps
        
        if method == "convert_dag_to_pag":
            return GraphOps.convert_dag_to_pag(args[0])
        elif method == "merge_graphs":
            return GraphOps.merge_graphs(args[0])
        else:
            return {"error": f"Unknown method: {method}"}
    
    def _pruning_tool(self, method: str, *args, **kwargs) -> Dict[str, Any]:
        """Pruning tool implementation"""
        from .tools import PruningTool
        
        if method == "global_markov_test":
            return PruningTool.global_markov_test(
                args[0], args[1], 
                alpha=kwargs.get("alpha", self.ci_alpha),
                max_pa_size=kwargs.get("max_pa_size", self.max_pa_size)
            )
        elif method == "structural_consistency_test":
            return PruningTool.structural_consistency_test(args[0], args[1], args[2], kwargs.get("n_subsets", self.n_subsets))
        else:
            return {"error": f"Unknown method: {method}"}
    
    def _ensemble_tool(self, method: str, *args, **kwargs) -> Dict[str, Any]:
        """Ensemble tool implementation"""
        from .tools import EnsembleTool
        
        if method == "build_consensus_skeleton":
            return EnsembleTool.build_consensus_skeleton(args[0], kwargs.get("weights"))
        elif method == "resolve_directions":
            return EnsembleTool.resolve_directions(args[0], args[1], args[2], weights=kwargs.get("weights"))
        elif method == "construct_pag":
            return EnsembleTool.construct_pag(args[0])
        elif method == "construct_dag":
            return EnsembleTool.construct_dag(
                args[0], args[1], args[2],
                execution_plan=kwargs.get("execution_plan"),
                algorithm_results=kwargs.get("algorithm_results")
            )
        else:
            return {"error": f"Unknown method: {method}"}
    
    def _pc_tool(self, method: str, *args, **kwargs) -> Dict[str, Any]:
        """PC algorithm tool implementation"""
        from .tools import PCTool
        
        if method == "pc_discovery":
            df = args[0]
            # Pass variable_schema if available
            variable_schema = kwargs.get("variable_schema") or (args[1] if len(args) > 1 else None)
            if variable_schema:
                kwargs["variable_schema"] = variable_schema
            return PCTool.discover(df, **kwargs)
        else:
            return {"error": f"Unknown method: {method}"}
    
    def _ges_tool(self, method: str, *args, **kwargs) -> Dict[str, Any]:
        """GES algorithm tool implementation"""
        from .tools import GESTool
        
        if method == "ges_discovery":
            df = args[0]
            return GESTool.discover(df, **kwargs)
        else:
            return {"error": f"Unknown method: {method}"}
    
    def _fci_tool(self, method: str, *args, **kwargs) -> Dict[str, Any]:
        """FCI algorithm tool implementation"""
        from .tools import FCITool
        
        if method == "fci_discovery":
            df = args[0]
            return FCITool.discover(df, **kwargs)
        else:
            return {"error": f"Unknown method: {method}"}
    
    def _cam_tool(self, method: str, *args, **kwargs) -> Dict[str, Any]:
        """CAM algorithm tool implementation"""
        from .tools import CAMTool
        
        if method == "cam_discovery":
            df = args[0]
            return CAMTool.discover(df, **kwargs)
        else:
            return {"error": f"Unknown method: {method}"}
    
    def _lim_tool(self, method: str, *args, **kwargs) -> Dict[str, Any]:
        """LiM algorithm tool implementation"""
        from .tools import LiMTool
        
        if method == "lim_discovery":
            return LiMTool.discover(args[0], **kwargs)
        else:
            return {"error": f"Unknown method: {method}"}
    
    def _tscm_tool(self, method: str, *args, **kwargs) -> Dict[str, Any]:
        """TSCM algorithm tool implementation"""
        from .tools import TSCMTool
        
        if method == "tscm_discovery":
            return TSCMTool.discover(args[0], **kwargs)
        else:
            return {"error": f"Unknown method: {method}"}
    
    def _notears_linear_tool(self, method: str, *args, **kwargs) -> Dict[str, Any]:
        """NOTEARS-linear algorithm tool implementation"""
        from .tools import NOTEARSLinearTool
        
        if method == "discover":
            df = args[0]
            return NOTEARSLinearTool.discover(df, **kwargs)
        else:
            return {"error": f"Unknown method: {method}"}
    
    def _notears_nonlinear_tool(self, method: str, *args, **kwargs) -> Dict[str, Any]:
        """NOTEARS-nonlinear algorithm tool implementation"""
        from .tools import NOTEARSNonlinearTool
        
        if method == "discover":
            df = args[0]
            return NOTEARSNonlinearTool.discover(df, **kwargs)
        else:
            return {"error": f"Unknown method: {method}"}
