# agents/causal_discovery/agent.py
"""
Causal Discovery Agent implementation using tool registry pattern.

This agent implements the complete causal discovery pipeline:
1. data_profiling - Profile data characteristics and generate qualitative summary
2. algorithm_tiering - Tier algorithms based on data profile compatibility
3. run_algorithms_portfolio - Execute algorithms from all tiers in parallel
4. candidate_pruning - Prune candidates using CI testing and structural consistency
5. scorecard_evaluation - Evaluate candidates using composite scorecard
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
    "CAM",        # additive models
    "FCI"         # allows latent confounders (PAG)
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
    "Nonlinear-FCM": ["ANM", "CAM"],
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
        "required": ["S_lin"],  # constraint-based; minimal linearity assumed
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
        self.bootstrap_iterations = self.config.get("bootstrap_iterations", 100)
        self.cv_folds = self.config.get("cv_folds", 5)
        
        # Pipeline parameters
        self.ci_alpha = self.config.get("ci_alpha", 0.05)
        self.violation_threshold = self.config.get("violation_threshold", 0.1)
        self.n_subsets = self.config.get("n_subsets", 3)
        self.profiling_max_pairs = self.config.get("profiling_max_pairs", 300)
        self.profiling_parallelism = self.config.get("profiling_parallelism")
        self.anm_rf_estimators = self.config.get("anm_rf_estimators", 100)
        self.run_all_tier_algorithms = self.config.get("run_all_tier_algorithms", False)
        
        # Composite weights
        self.composite_weights = self.config.get("composite_weights", {
            "statistical_fit": 0.3, "global_consistency": 0.25,
            "sampling_stability": 0.25, "structural_stability": 0.2
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
            elif substep == "algorithm_tiering":
                return self._algorithm_tiering(state)
            elif substep == "run_algorithms_portfolio":
                return self._run_algorithms_portfolio(state)
            elif substep == "candidate_pruning":
                return self._candidate_pruning(state)
            elif substep == "scorecard_evaluation":
                return self._scorecard_evaluation(state)
            elif substep == "ensemble_synthesis":
                return self._ensemble_synthesis(state)
            # Legacy substeps for backward compatibility
            elif substep == "assumption_method_matrix":
                return self._data_profiling(state)
            elif substep == "algorithm_scoring":
                return self._algorithm_tiering(state)
            elif substep == "run_algorithms":
                return self._run_algorithms_portfolio(state)
            elif substep == "intermediate_scoring":
                return self._scorecard_evaluation(state)
            elif substep == "final_graph_selection":
                return self._ensemble_synthesis(state)
            else:
                raise ValueError(f"Unknown substep: {substep}")
                
        except Exception as e:
            logger.error(f"CausalDiscoveryAgent substep {substep} failed: {str(e)}")
            state["error"] = f"Causal discovery {substep} failed: {str(e)}"
            state["causal_discovery_status"] = "failed"
            return state
    
    def _data_profiling(self, state: AgentState) -> AgentState:
        """Stage 1: Data profiling with qualitative summary generation"""
        logger.info("Performing data profiling...")
        
        try:
            # Get preprocessed data (load from reference if necessary)
            df = self._load_dataframe_from_state(state)
            if df is None:
                raise ValueError("No preprocessed data available")
            
            # Get variable information
            variables = list(df.columns)
            logger.info(f"Profiling data for variables: {variables}")
            
            # Initialize assumption scores
            assumption_scores = {
                "S_lin": {},      # Linearity scores
                "S_nG": {},       # Non-Gaussian scores  
                "S_ANM": {},      # ANM scores
                "S_Gauss": {},    # Gaussian scores
                "S_EqVar": {}     # Equal variance scores
            }
            
            # Build all lower-triangle pairs
            pairs = [(variables[i], variables[j]) for i in range(len(variables)) for j in range(i+1, len(variables))]
            # Sampling for very large pair sets
            if self.profiling_max_pairs and len(pairs) > self.profiling_max_pairs:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(pairs), size=self.profiling_max_pairs, replace=False)
                pairs = [pairs[k] for k in idx]
                logger.info(f"Sampling pairwise tests: using {len(pairs)} pairs")

            # Determine parallelism
            max_workers = self.profiling_parallelism or min(8, max(1, len(pairs)))

            def _eval_pair(var1: str, var2: str):
                pair_key = f"{var1}_{var2}"
                # 1. Linearity test
                linearity_score = self._test_linearity(df[var1], df[var2])
                # 2. Non-Gaussian test
                non_gaussian_score = self._test_non_gaussian(df[var1], df[var2])
                # 3. ANM test
                anm_score = self._test_anm(df[var1], df[var2])
                # 4. Gaussian + Equal Variance
                gauss_score, eqvar_score = self._test_gaussian_eqvar(df[var1], df[var2])
                return pair_key, linearity_score, non_gaussian_score, anm_score, gauss_score, eqvar_score

            # Parallel execution
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_eval_pair, v1, v2) for v1, v2 in pairs]
                for fut in futures:
                    pair_key, lin_s, ng_s, anm_s, g_s, ev_s = fut.result()
                    assumption_scores["S_lin"][pair_key] = lin_s
                    assumption_scores["S_nG"][pair_key] = ng_s
                    assumption_scores["S_ANM"][pair_key] = anm_s
                    assumption_scores["S_Gauss"][pair_key] = g_s
                    assumption_scores["S_EqVar"][pair_key] = ev_s
            
            # Generate qualitative data profile
            data_profile = self._generate_qualitative_profile(assumption_scores, variables)
            
            # Update state
            state["assumption_method_scores"] = assumption_scores
            state["data_profile"] = data_profile
            state["data_profiling_completed"] = True
            
            logger.info("Data profiling completed")
            return state
            
        except Exception as e:
            logger.error(f"Data profiling failed: {str(e)}")
            state["error"] = f"Data profiling failed: {str(e)}"
            return state
    
    def _classify_assumption_strength(self, scores_dict: Dict[str, float], 
                                       n_pairs: int, 
                                       use_quantile: bool = None) -> str:
        """Classify assumption strength using adaptive thresholds"""
        if use_quantile is None:
            use_quantile = self.use_quantile_thresholds
            
        scores = np.array(list(scores_dict.values()))
        mean = scores_dict["mean"]
        std = scores_dict["std"]
        
        # Use quantile approach if enough pairs
        if use_quantile and n_pairs >= self.min_pairs_for_quantile:
            q1, q3 = np.quantile(scores, [0.25, 0.75])
            if mean >= q3:
                return "strong"
            elif mean <= q1:
                return "weak"
            else:
                return "moderate"
        
        # Fallback: simple approach with confidence
        hi, lo, min_conf = self.threshold_high, self.threshold_low, self.min_confidence
        
        if std < min_conf:
            return "strong" if mean >= hi else "weak" if mean <= lo else "moderate"
        
        if mean >= hi and std <= 0.15:
            return "strong"
        elif mean <= lo and std <= 0.15:
            return "weak"
        else:
            return "moderate"

    def _generate_qualitative_profile(self, assumption_scores: Dict[str, Dict[str, float]], variables: List[str]) -> Dict[str, Any]:
        """Generate qualitative data profile from assumption scores"""
        try:
            # Aggregate scores across all pairs
            aggregated_scores = {}
            for assumption_type, pair_scores in assumption_scores.items():
                if pair_scores:
                    scores = list(pair_scores.values())
                    aggregated_scores[assumption_type] = {
                        "mean": np.mean(scores),
                        "median": np.median(scores),
                        "std": np.std(scores),
                        "min": np.min(scores),
                        "max": np.max(scores)
                    }
                else:
                    aggregated_scores[assumption_type] = {"mean": 0.5, "median": 0.5, "std": 0.0, "min": 0.5, "max": 0.5}
            
            # Generate qualitative descriptions using adaptive thresholds
            n_pairs = len(variables) * (len(variables) - 1) // 2
            
            linearity_desc = self._classify_assumption_strength(
                aggregated_scores["S_lin"], n_pairs)
            non_gaussian_desc = self._classify_assumption_strength(
                aggregated_scores["S_nG"], n_pairs)
            gaussian_desc = self._classify_assumption_strength(
                aggregated_scores["S_Gauss"], n_pairs)
            eqvar_desc = self._classify_assumption_strength(
                aggregated_scores["S_EqVar"], n_pairs)
            
            # ANM compatibility with dual check
            anm_scores = np.array(list(assumption_scores["S_ANM"].values()))
            anm_mean = aggregated_scores["S_ANM"]["mean"]
            anm_compatible = (anm_mean >= self.anm_mean_threshold) and (np.mean(anm_scores >= self.anm_mean_threshold) >= self.anm_pair_ratio_threshold)
            
            data_profile = {
                "linearity": linearity_desc,
                "non_gaussian": non_gaussian_desc,
                "anm_compatible": anm_compatible,
                "gaussian": gaussian_desc,
                "equal_variance": eqvar_desc,
                "n_variables": len(variables),
                "n_pairs": len(variables) * (len(variables) - 1) // 2,
                "aggregated_scores": aggregated_scores,
                "summary": f"Data shows {linearity_desc} linearity, {non_gaussian_desc} non-Gaussianity, "
                          f"{'ANM-compatible' if anm_compatible else 'ANM-incompatible'} patterns"
            }
            
            return data_profile
            
        except Exception as e:
            logger.warning(f"Qualitative profile generation failed: {e}")
            return {
                "linearity": "moderate",
                "non_gaussian": "moderate", 
                "anm_compatible": False,
                "gaussian": "moderate",
                "equal_variance": "moderate",
                "n_variables": len(variables),
                "n_pairs": len(variables) * (len(variables) - 1) // 2,
                "summary": "Default profile due to generation error"
            }
    
    def _calculate_family_score(self, data_profile: Dict[str, Any], 
                                requirements: Dict[str, List[str]]) -> float:
        """Calculate family score with penalties for opposite assumptions"""
        
        def desc_to_value(desc: str) -> float:
            return {"strong": 1.0, "moderate": 0.6, "weak": 0.0}.get(desc, 0.5)
        
        def get_desc(assumption: str) -> str:
            mapping = {
                "linearity": data_profile["linearity"],
                "non_gaussian": data_profile["non_gaussian"],
                "anm_compatible": "strong" if data_profile["anm_compatible"] else "weak",
                "gaussian": data_profile["gaussian"],
                "equal_variance": data_profile["equal_variance"]
            }
            return mapping.get(assumption, "moderate")
        
        score, total = 0.0, 0.0
        
        # Required assumptions (weight = 1.0)
        for req in requirements.get("required", []):
            s = desc_to_value(get_desc(req))
            score += s * 1.0
            total += 1.0
        
        # Preferred assumptions (weight = 0.5)
        for pref in requirements.get("preferred", []):
            s = desc_to_value(get_desc(pref))
            score += s * 0.5
            total += 0.5
        
        # Opposite assumptions (penalty = configurable weight)
        pen, pen_w = 0.0, 0.0
        for opp in requirements.get("opposite", []):
            s = desc_to_value(get_desc(opp))
            pen += s * self.penalty_weight
            pen_w += self.penalty_weight
        
        # Normalize with penalty and clip to [0, 1]
        if total + pen_w > 0:
            family_score = (score - pen + pen_w) / (total + pen_w)
        else:
            family_score = 0.5
        
        return np.clip(family_score, 0.0, 1.0)

    def _algorithm_tiering(self, state: AgentState) -> AgentState:
        """Stage 1: Algorithm tiering based on data profile"""
        logger.info("Performing algorithm tiering...")
        
        try:
            data_profile = state.get("data_profile", {})
            if not data_profile:
                raise ValueError("No data profile available")
                        
            # Match data profile to family requirements
            tier1_algorithms = []  # Best match algorithms
            tier2_algorithms = []  # Partial match algorithms  
            tier3_algorithms = []  # Opposite assumption algorithms (for exploration)
            
            # Define family requirements based on data profile
            family_requirements = {
                "Linear": {
                    "required": ["linearity"],
                    "preferred": ["non_gaussian"],
                    "opposite": ["anm_compatible"]
                },
                "Nonlinear-FCM": {
                    "required": ["anm_compatible"],
                    "preferred": ["linearity"],  # prefer nonlinearity
                    "opposite": ["gaussian"]
                },
                "Constraint-based": {
                    "required": ["linearity"],
                    "preferred": ["gaussian"],
                    "opposite": ["non_gaussian"]
                },
                "Score-based": {
                    "required": ["linearity", "gaussian", "equal_variance"],
                    "preferred": [],
                    "opposite": ["non_gaussian", "anm_compatible"]
                },
                "Latent-robust": {
                    "required": ["linearity"],
                    "preferred": ["gaussian"],
                    "opposite": ["non_gaussian"]
                }
            }
            
            # Score each family
            family_scores = {}
            for family_name in ALGORITHM_FAMILIES_REP.keys():
                if family_name not in family_requirements:
                    continue
                
                requirements = family_requirements[family_name]
                family_scores[family_name] = self._calculate_family_score(
                    data_profile, requirements)
            
            # Build tier algorithms based on mode: representatives vs all family algorithms
            run_all = state.get("run_all_tier_algorithms", self.run_all_tier_algorithms)
            family_source = ALGORITHM_FAMILIES_ALL if run_all else ALGORITHM_FAMILIES_REP
            for family_name, alg_list in family_source.items():
                if family_name not in family_scores:
                    continue
                score = family_scores[family_name]
                target = tier1_algorithms if score >= 0.7 else tier2_algorithms if score >= 0.4 else tier3_algorithms
                target.extend(alg_list)
            
            # FCI always included (latent-robust)
            if "FCI" not in tier1_algorithms and "FCI" not in tier2_algorithms and "FCI" not in tier3_algorithms:
                tier1_algorithms.append("FCI")
            
            # Remove duplicates
            tier1_algorithms = list(set(tier1_algorithms))
            tier2_algorithms = list(set(tier2_algorithms))
            tier3_algorithms = list(set(tier3_algorithms))
            
            # Generate tiering reasoning
            reasoning = self._generate_tiering_reasoning(data_profile, family_scores, tier1_algorithms, tier2_algorithms, tier3_algorithms)
            
            # Update state
            state["algorithm_tiers"] = {
                "tier1": tier1_algorithms,
                "tier2": tier2_algorithms, 
                "tier3": tier3_algorithms
            }
            state["tiering_reasoning"] = reasoning
            state["algorithm_tiering_completed"] = True
            
            logger.info(f"Algorithm tiering completed. Tier1: {tier1_algorithms}, Tier2: {tier2_algorithms}, Tier3: {tier3_algorithms}")
            return state
            
        except Exception as e:
            logger.error(f"Algorithm tiering failed: {str(e)}")
            state["error"] = f"Algorithm tiering failed: {str(e)}"
            return state
    
    def _generate_tiering_reasoning(self, data_profile: Dict[str, Any], family_scores: Dict[str, float], 
                                   tier1: List[str], tier2: List[str], tier3: List[str]) -> str:
        """Generate reasoning for algorithm tiering"""
        reasoning = f"""
        Data Profile Analysis:
        - Linearity: {data_profile['linearity']}
        - Non-Gaussianity: {data_profile['non_gaussian']}
        - ANM Compatibility: {data_profile['anm_compatible']}
        - Gaussian: {data_profile['gaussian']}
        - Equal Variance: {data_profile['equal_variance']}
        
        Family Scores: {family_scores}
        
        Algorithm Tiering:
        - Tier 1 (Best Match): {tier1}
        - Tier 2 (Partial Match): {tier2}
        - Tier 3 (Exploration): {tier3}
        
        Rationale: Each family contributes one representative algorithm by default. Extended algorithms
        (PNL, CAM) can be included when explicitly requested. FCI is always included for latent robustness.
        Tier 1 algorithms have the best assumption compatibility, Tier 2 have partial compatibility,
        and Tier 3 algorithms have opposite assumptions for exploratory analysis.
        """
        return reasoning.strip()
    
    
    def _run_algorithms_portfolio(self, state: AgentState) -> AgentState:
        """Stage 2: Run algorithms from all tiers in parallel"""
        logger.info("Running algorithm portfolio in parallel...")
        
        try:
            algorithm_tiers = state.get("algorithm_tiers", {})
            df = self._load_dataframe_from_state(state)
            data_profile = state.get("data_profile", {})
            
            # Allow explicit user override of algorithms via executor edits
            user_selected = state.get("selected_algorithms") or []
            if not isinstance(user_selected, list):
                user_selected = []
            user_selected = [str(a) for a in user_selected]
            if df is None:
                raise ValueError("No preprocessed data available")
            
            # Collect all algorithms from all tiers
            all_algorithms = []
            if user_selected:
                # Filter to known algorithms only
                all_algorithms = [a for a in user_selected if a in ALGORITHM_LIST]
                logger.info(f"Using user-selected algorithms override: {all_algorithms}")
            else:
                if not algorithm_tiers:
                    raise ValueError("No algorithm tiers available")
                all_algorithms.extend(algorithm_tiers.get("tier1", []))
                all_algorithms.extend(algorithm_tiers.get("tier2", []))
                all_algorithms.extend(algorithm_tiers.get("tier3", []))
            
            # Remove duplicates
            all_algorithms = list(set(all_algorithms))
            
            if not all_algorithms:
                raise ValueError("No algorithms to execute")
            
            logger.info(f"Executing algorithms: {all_algorithms}")
            
            # Run algorithms in parallel
            algorithm_results = {}
            
            with ThreadPoolExecutor(max_workers=len(all_algorithms)) as executor:
                futures = {}
                # Per-algorithm timeouts (seconds)
                timeout_map = {"LiNGAM": 120, "ANM": 180, "PC": 180, "GES": 300, "FCI": 300, "CAM": 300}
                for alg_name in all_algorithms:
                    future = executor.submit(self._dispatch_algorithm, alg_name, df, data_profile)
                    futures[alg_name] = future
                
                # Collect results
                for alg_name, future in futures.items():
                    try:
                        result = future.result(timeout=timeout_map.get(alg_name, 300))
                        algorithm_results[alg_name] = result
                        logger.info(f"Algorithm {alg_name} completed successfully")
                    except Exception as e:
                        logger.error(f"Algorithm {alg_name} failed: {str(e)}")
                        algorithm_results[alg_name] = {"error": str(e)}
            
            # Convert numpy types to Python native types for msgpack serialization
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
            # Echo which algorithms were executed for verification in tests
            state["executed_algorithms"] = list(algorithm_results.keys())
            state["run_algorithms_portfolio_completed"] = True
            
            logger.info("Algorithm portfolio execution completed")
            return state
            
        except Exception as e:
            logger.error(f"Algorithm portfolio execution failed: {str(e)}")
            state["error"] = f"Algorithm portfolio execution failed: {str(e)}"
            return state
    
    def _candidate_pruning(self, state: AgentState) -> AgentState:
        """Stage 3: Candidate pruning using CI testing and structural consistency"""
        logger.info("Performing candidate pruning...")
        
        try:
            algorithm_results = self._load_algorithm_results_from_state(state)
            df = self._load_dataframe_from_state(state)
            
            if not algorithm_results:
                raise ValueError("No algorithm results available")
            if df is None:
                raise ValueError("No preprocessed data available")
            
            pruned_candidates = []
            pruning_log = []
            
            for alg_name, result in algorithm_results.items():
                if "error" in result:
                    pruning_log.append({
                        "algorithm": alg_name,
                        "reason": "execution_error",
                        "error": result["error"]
                    })
                    continue
                
                logger.info(f"Pruning candidate from {alg_name}...")
                
                # 1. Global Markov test
                markov_result = self.use_tool("pruning_tool", "global_markov_test", result, df, alpha=self.ci_alpha)
                violation_ratio = markov_result.get("violation_ratio", 1.0)
                
                # Filter if violation ratio too high
                if violation_ratio > self.violation_threshold:
                    pruning_log.append({
                        "algorithm": alg_name,
                        "reason": "high_ci_violations",
                        "violation_ratio": violation_ratio,
                        "threshold": self.violation_threshold
                    })
                    continue
                
                # 2. Structural consistency test
                consistency_result = self.use_tool("pruning_tool", "structural_consistency_test", result, df, alg_name)
                instability_score = consistency_result.get("instability_score", 1.0)
                
                # Store pruned candidate
                pruned_candidates.append({
                    "algorithm": alg_name,
                    "graph": result,
                    "violation_ratio": violation_ratio,
                    "instability_score": instability_score,
                    "markov_result": markov_result,
                    "consistency_result": consistency_result
                })
            
            # Convert numpy types to Python native types for msgpack serialization
            pruned_candidates = _convert_numpy_types(pruned_candidates)
            pruning_log = _convert_numpy_types(pruning_log)
            
            # Update state
            state["pruned_candidates"] = pruned_candidates
            state["pruning_log"] = pruning_log
            state["candidate_pruning_completed"] = True
            
            logger.info(f"Candidate pruning completed. {len(pruned_candidates)} candidates retained, {len(pruning_log)} rejected")
            return state
            
        except Exception as e:
            logger.error(f"Candidate pruning failed: {str(e)}")
            state["error"] = f"Candidate pruning failed: {str(e)}"
            return state
    
    def _scorecard_evaluation(self, state: AgentState) -> AgentState:
        """Stage 4: Scorecard evaluation with composite scoring"""
        logger.info("Performing scorecard evaluation...")
        
        try:
            pruned_candidates = state.get("pruned_candidates", [])
            df = self._load_dataframe_from_state(state)
            
            if not pruned_candidates:
                raise ValueError("No pruned candidates available")
            if df is None:
                raise ValueError("No preprocessed data available")
            
            scorecard = []
            
            for candidate in pruned_candidates:
                alg_name = candidate["algorithm"]
                graph = candidate["graph"]
                
                logger.info(f"Evaluating {alg_name} for scorecard...")
                
                # 1. Statistical fit (BIC/AIC) - use algorithm-native score if available
                statistical_fit = self._compute_algorithm_native_score(graph, alg_name)
                if statistical_fit is None:
                    # Fallback to GraphEvaluator
                    fidelity_result = self.use_tool("graph_evaluator", "fidelity_evaluation", df, graph)
                    statistical_fit = fidelity_result.get("fidelity_score", 0.5)
                
                # 2. Global consistency - use CI violation ratio from pruning
                # For PAG (FCI), skip global consistency as d-separation doesn't apply
                is_pag = graph.get("metadata", {}).get("graph_type") == "PAG"
                
                if is_pag:
                    global_consistency = 0.0  # Not applicable for PAG
                    # Normalize remaining weights to sum to 1.0
                    w_stat = self.composite_weights["statistical_fit"]
                    w_samp = self.composite_weights["sampling_stability"]
                    w_struct = self.composite_weights["structural_stability"]
                    weight_sum = w_stat + w_samp + w_struct
                    
                    # 3. Sampling stability - use existing Bootstrapper
                    sampling_stability = self._evaluate_robustness(df, graph, alg_name)
                    
                    # 4. Structural stability - use instability score from pruning
                    structural_stability = 1.0 - candidate["instability_score"]
                    
                    composite_score = (
                        (w_stat / weight_sum) * statistical_fit +
                        (w_samp / weight_sum) * sampling_stability +
                        (w_struct / weight_sum) * structural_stability
                    )
                else:
                    global_consistency = 1.0 - candidate["violation_ratio"]
                    
                    # 3. Sampling stability - use existing Bootstrapper
                    sampling_stability = self._evaluate_robustness(df, graph, alg_name)
                    
                    # 4. Structural stability - use instability score from pruning
                    structural_stability = 1.0 - candidate["instability_score"]
                    
                    # Use standard weights
                    composite_score = (
                        self.composite_weights["statistical_fit"] * statistical_fit +
                        self.composite_weights["global_consistency"] * global_consistency +
                        self.composite_weights["sampling_stability"] * sampling_stability +
                        self.composite_weights["structural_stability"] * structural_stability
                    )
                
                # Apply edge count penalty to discourage empty graphs
                n_edges = len(graph.get("graph", {}).get("edges", []))
                if n_edges == 0:
                    composite_score *= 0.1  # Heavy penalty for empty graphs
                elif n_edges < 3:
                    composite_score *= 0.5  # Moderate penalty for very sparse graphs
                
                scorecard.append({
                    "algorithm": alg_name,
                    "graph_id": f"{alg_name}_{id(graph)}",
                    "statistical_fit": statistical_fit,
                    "global_consistency": global_consistency,
                    "sampling_stability": sampling_stability,
                    "structural_stability": structural_stability,
                    "composite_score": composite_score,
                    "graph": graph
                })
            
            # Sort by composite score and select top candidates
            scorecard.sort(key=lambda x: x["composite_score"], reverse=True)
            top_candidates = scorecard[:3]  # Top 3 graphs
            
            # Convert numpy types to Python native types for msgpack serialization
            scorecard = _convert_numpy_types(scorecard)
            top_candidates = _convert_numpy_types(top_candidates)
            
            # Update state
            state["scorecard"] = scorecard
            state["top_candidates"] = top_candidates
            state["scorecard_evaluation_completed"] = True
            
            logger.info(f"Scorecard evaluation completed. Top candidate: {top_candidates[0]['algorithm'] if top_candidates else 'None'}")
            return state
            
        except Exception as e:
            logger.error(f"Scorecard evaluation failed: {str(e)}")
            state["error"] = f"Scorecard evaluation failed: {str(e)}"
            return state
    
    def _compute_algorithm_native_score(self, graph: Dict[str, Any], algorithm_name: str) -> Optional[float]:
        """Extract algorithm-native BIC/AIC score from metadata if available"""
        try:
            if "metadata" in graph:
                metadata = graph["metadata"]
                
                # Check for common score fields
                if "bic" in metadata:
                    # Convert BIC to score (lower BIC = higher score)
                    bic = metadata["bic"]
                    return max(0.0, min(1.0, 1.0 - (bic / 1000)))  # Simple normalization
                elif "aic" in metadata:
                    # Convert AIC to score
                    aic = metadata["aic"]
                    return max(0.0, min(1.0, 1.0 - (aic / 1000)))
                elif "score" in metadata:
                    return float(metadata["score"])
                elif "log_likelihood" in metadata:
                    # Convert log-likelihood to score
                    ll = metadata["log_likelihood"]
                    return max(0.0, min(1.0, (ll + 100) / 200))  # Simple normalization
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to extract native score for {algorithm_name}: {e}")
            return None

    # === Helpers and dispatch ===

    def _select_ci_test(self, data_profile: Dict) -> str:
        """Select appropriate CI test based on data characteristics"""
        linearity = data_profile.get("linearity", "moderate")
        
        # Non-linear data needs kernel-based test
        if linearity == "weak":
            logger.info("Non-linear data detected, using KCI test")
            return "kci"
        
        # For now, default to fisherz for linear data
        # Future: add checks for categorical data → "gsq"
        logger.info("Linear data detected, using Fisher-Z test")
        return "fisherz"

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
        try:
            return bool(graph and "graph" in graph and "edges" in graph["graph"])
        except Exception:
            return False

    def _dispatch_algorithm(self, alg_name: str, df: pd.DataFrame, data_profile: Dict = None) -> Dict[str, Any]:
        if alg_name == "LiNGAM":
            return self._run_lingam(df)
        if alg_name == "ANM":
            return self._run_anm(df)
        if alg_name == "PC":
            return self._run_pc(df, data_profile=data_profile)
        if alg_name == "GES":
            return self._run_ges(df)
        if alg_name == "FCI":
            return self._run_fci(df, data_profile=data_profile)
        if alg_name == "CAM":
            return self._run_cam(df)
        return {"error": f"Unknown algorithm: {alg_name}"}
    
    def _intermediate_scoring(self, state: AgentState) -> AgentState:
        """Legacy method for backward compatibility"""
        return self._scorecard_evaluation(state)
    
    def _ensemble_synthesis(self, state: AgentState) -> AgentState:
        """Stage 5: Ensemble synthesis with PAG-like and DAG outputs"""
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
            skeleton_result = self.use_tool("ensemble_tool", "build_consensus_skeleton", graphs, weights)
            
            # 2. Resolve directions
            directions_result = self.use_tool("ensemble_tool", "resolve_directions", skeleton_result, graphs, data_profile)
            
            # 3. Construct PAG-like graph
            pag_result = self.use_tool("ensemble_tool", "construct_pag", directions_result, {})
            
            # 4. Construct single DAG with tie-breaking
            top_algorithm = top_candidates[0]["algorithm"]
            dag_result = self.use_tool("ensemble_tool", "construct_dag", pag_result, data_profile, top_algorithm)
            
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
            
            logger.info(f"Ensemble synthesis completed. Top algorithm: {top_algorithm}")
            return state
            
        except Exception as e:
            logger.error(f"Ensemble synthesis failed: {str(e)}")
            state["error"] = f"Ensemble synthesis failed: {str(e)}"
            return state
    
    def _generate_synthesis_reasoning(self, top_candidates: List[Dict[str, Any]], pag_result: Dict[str, Any], 
                                    dag_result: Dict[str, Any], data_profile: Dict[str, Any]) -> str:
        """Generate reasoning for ensemble synthesis"""
        try:
            top_algorithm = top_candidates[0]["algorithm"] if top_candidates else "Unknown"
            
            reasoning = f"""
            Ensemble Synthesis Results:
            
            Top Candidates: {[c['algorithm'] for c in top_candidates]}
            Leading Algorithm: {top_algorithm}
            
            Data Profile: {data_profile.get('summary', 'N/A')}
            
            PAG Construction:
            - Graph Type: {pag_result.get('graph_type', 'Unknown')}
            - Edge Count: {len(pag_result.get('edges', []))}
            - Uncertainty Markers: {pag_result.get('metadata', {}).get('uncertainty_markers', False)}
            
            DAG Construction:
            - Graph Type: {dag_result.get('graph_type', 'Unknown')}
            - Edge Count: {len(dag_result.get('edges', []))}
            - Tie-breaking Method: {dag_result.get('metadata', {}).get('construction_method', 'Unknown')}
            
            Rationale: The ensemble synthesis combines the top {len(top_candidates)} candidates using consensus
            skeleton building and direction resolution. The PAG preserves uncertainty information for reporting,
            while the DAG applies assumption-based tie-breaking for downstream inference tasks.
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
    
    def _test_non_gaussian(self, x: pd.Series, y: pd.Series) -> float:
        """Test non-Gaussianity using normality tests"""
        try:
            # Use stats tool for normality tests
            result = self.use_tool("stats_tool", "normality_test", x, y)
            return result.get("non_gaussian_score", 0.5)
        except Exception as e:
            logger.warning(f"Non-Gaussian test failed: {e}")
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

    def _cam_tool(self, method: str, *args, **kwargs) -> Dict[str, Any]:
        from .tools import CAMTool
        if method == "cam_discovery":
            df = args[0]
            return CAMTool.discover(df, **kwargs)
        return {"error": f"Unknown method: {method}"}

    def _run_pc(self, df: pd.DataFrame, data_profile: Dict = None, **kwargs) -> Dict[str, Any]:
        """Run PC algorithm (backend default: causal-learn)"""
        try:
            # Select appropriate CI test based on data profile
            if data_profile is None:
                data_profile = {}
            indep_test = self._select_ci_test(data_profile)
            
            # Pass alpha and indep_test
            kwargs['alpha'] = kwargs.get('alpha', self.ci_alpha)
            kwargs['indep_test'] = kwargs.get('indep_test', indep_test)
            
            result = self.use_tool("pc_tool", "pc_discovery", df, **kwargs)
            return result
        except Exception as e:
            logger.error(f"PC execution failed: {e}")
            return {"error": str(e)}

    def _run_ges(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run GES algorithm (backend default: causal-learn, score=BIC)"""
        try:
            result = self.use_tool("ges_tool", "ges_discovery", df, score_func="bic")
            return result
        except Exception as e:
            logger.error(f"GES execution failed: {e}")
            return {"error": str(e)}
    def _run_fci(self, df: pd.DataFrame, data_profile: Dict = None, **kwargs) -> Dict[str, Any]:
        """Run FCI algorithm (backend default: causal-learn)"""
        try:
            # Select appropriate CI test based on data profile
            if data_profile is None:
                data_profile = {}
            indep_test = self._select_ci_test(data_profile)
            
            # Pass alpha and indep_test
            kwargs['alpha'] = kwargs.get('alpha', self.ci_alpha)
            kwargs['indep_test'] = kwargs.get('indep_test', indep_test)
            
            result = self.use_tool("fci_tool", "fci_discovery", df, **kwargs)
            return result
        except Exception as e:
            logger.error(f"FCI execution failed: {e}")
            return {"error": str(e)}
    def _pc_tool(self, method: str, *args, **kwargs) -> Dict[str, Any]:
        from .tools import PCTool
        if method == "pc_discovery":
            df = args[0]
            return PCTool.discover(df, **kwargs)
        return {"error": f"Unknown method: {method}"}

    def _ges_tool(self, method: str, *args, **kwargs) -> Dict[str, Any]:
        from .tools import GESTool
        if method == "ges_discovery":
            df = args[0]
            return GESTool.discover(df, **kwargs)
        return {"error": f"Unknown method: {method}"}
    def _fci_tool(self, method: str, *args, **kwargs) -> Dict[str, Any]:
        from .tools import FCITool
        if method == "fci_discovery":
            df = args[0]
            return FCITool.discover(df, **kwargs)
        return {"error": f"Unknown method: {method}"}
    
    # === Evaluation Methods ===
    
    def _evaluate_robustness(self, df: pd.DataFrame, result: Dict[str, Any], alg_name: str) -> float:
        """Evaluate robustness using bootstrap"""
        try:
            bootstrap_result = self.use_tool("bootstrapper", "bootstrap_evaluation", df, result, alg_name, self.bootstrap_iterations)
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
        elif test_type == "normality_test":
            return StatsTool.normality_test(args[0], args[1])
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
        
        if method == "fidelity_evaluation":
            return GraphEvaluator.fidelity_evaluation(args[0], args[1])
        elif method == "cv_evaluation":
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
            return PruningTool.global_markov_test(args[0], args[1], kwargs.get("alpha", self.ci_alpha))
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
            return EnsembleTool.resolve_directions(args[0], args[1], args[2])
        elif method == "construct_pag":
            return EnsembleTool.construct_pag(args[0], args[1])
        elif method == "construct_dag":
            return EnsembleTool.construct_dag(args[0], args[1], args[2])
        else:
            return {"error": f"Unknown method: {method}"}
    
    def _pc_tool(self, method: str, *args, **kwargs) -> Dict[str, Any]:
        """PC algorithm tool implementation"""
        from .tools import PCTool
        
        if method == "pc_discovery":
            return PCTool.discover(args[0], **kwargs)
        else:
            return {"error": f"Unknown method: {method}"}
    
    def _ges_tool(self, method: str, *args, **kwargs) -> Dict[str, Any]:
        """GES algorithm tool implementation"""
        from .tools import GESTool
        
        if method == "ges_discovery":
            return GESTool.discover(args[0], **kwargs)
        else:
            return {"error": f"Unknown method: {method}"}
    
    def _fci_tool(self, method: str, *args, **kwargs) -> Dict[str, Any]:
        """FCI algorithm tool implementation"""
        from .tools import FCITool
        
        if method == "fci_discovery":
            return FCITool.discover(args[0], **kwargs)
        else:
            return {"error": f"Unknown method: {method}"}
    
    def _cam_tool(self, method: str, *args, **kwargs) -> Dict[str, Any]:
        """CAM algorithm tool implementation"""
        from .tools import CAMTool
        
        if method == "cam_discovery":
            return CAMTool.discover(args[0], **kwargs)
        else:
            return {"error": f"Unknown method: {method}"}
