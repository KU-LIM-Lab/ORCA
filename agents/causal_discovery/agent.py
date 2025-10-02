# agents/causal_discovery/agent.py
"""
Causal Discovery Agent implementation using tool registry pattern.

This agent implements the complete causal discovery pipeline:
1. assumption_method_matrix - Test assumptions and generate scores
2. algorithm_scoring - Score algorithms based on assumption compatibility
3. run_algorithms - Execute selected algorithms in parallel
4. intermediate_scoring - Evaluate robustness, fidelity, generalizability
5. final_graph_selection - Select final causal graph
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
import asyncio
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from core.base import SpecialistAgent, AgentType, AgentResult
from core.state import AgentState, PipelinePhase
from monitoring.metrics.collector import MetricsCollector

logger = logging.getLogger(__name__)

class CausalDiscoveryAgent(SpecialistAgent):
    """Causal Discovery Agent using tool registry pattern"""
    
    def __init__(self, name: str = "causal_discovery", config: Optional[Dict[str, Any]] = None, 
                 metrics_collector: Optional[MetricsCollector] = None):
        super().__init__(name, AgentType.SPECIALIST, config, metrics_collector)
        
        # Set domain expertise
        self.set_domain_expertise([
            "assumption_validation",
            "algorithm_selection", 
            "causal_graph_generation",
            "graph_evaluation",
            "statistical_testing"
        ])
        
        # Configuration
        self.config = config or {}
        self.bootstrap_iterations = self.config.get("bootstrap_iterations", 100)
        self.cv_folds = self.config.get("cv_folds", 5)
        self.top_k_algorithms = self.config.get("top_k_algorithms", 3)
        self.lambda_soft_and = self.config.get("lambda_soft_and", 0.7)
        self.beta_conservative = self.config.get("beta_conservative", 2.0)
    
    def _register_specialist_tools(self) -> None:
        """Register causal discovery specific tools"""
        # Statistical testing tools
        self.register_tool(
            "stats_tool",
            self._stats_tool,
            "Statistical testing: GLM/GAM, LRT/AIC/BIC, normality tests"
        )
        
        # Independence testing tools
        self.register_tool(
            "independence_tool", 
            self._independence_tool,
            "Independence testing: HSIC/KCI, non-parametric regression"
        )
        
        # Causal discovery algorithm tools
        self.register_tool(
            "lingam_tool",
            self._lingam_tool,
            "LiNGAM algorithm: DirectLiNGAM order/weight estimation"
        )
        
        self.register_tool(
            "anm_tool",
            self._anm_tool,
            "ANM algorithm: Additive Noise Model directionality testing"
        )
        
        self.register_tool(
            "eqvar_linear_tool",
            self._eqvar_linear_tool,
            "EqVar Linear: Linear+Gaussian+equal variance assumption"
        )
        
        # Evaluation and analysis tools
        self.register_tool(
            "bootstrapper",
            self._bootstrapper,
            "Bootstrap resampling and frequency aggregation"
        )
        
        self.register_tool(
            "graph_evaluator",
            self._graph_evaluator,
            "Graph evaluation: BIC/MDL, CV, robustness, assumption scores"
        )
        
        self.register_tool(
            "graph_ops",
            self._graph_ops,
            "Graph operations: DAG/PAG/AG conversion, merge, voting"
        )
    
    def step(self, state: AgentState) -> AgentState:
        """Execute causal discovery step"""
        substep = state.get("current_substep", "assumption_method_matrix")
        
        logger.info(f"CausalDiscoveryAgent executing substep: {substep}")
        
        try:
            if substep == "assumption_method_matrix":
                return self._assumption_method_matrix(state)
            elif substep == "algorithm_scoring":
                return self._algorithm_scoring(state)
            elif substep == "run_algorithms":
                return self._run_algorithms(state)
            elif substep == "intermediate_scoring":
                return self._intermediate_scoring(state)
            elif substep == "final_graph_selection":
                return self._final_graph_selection(state)
            else:
                raise ValueError(f"Unknown substep: {substep}")
                
        except Exception as e:
            logger.error(f"CausalDiscoveryAgent substep {substep} failed: {str(e)}")
            state["error"] = f"Causal discovery {substep} failed: {str(e)}"
            state["causal_discovery_status"] = "failed"
            return state
    
    def _assumption_method_matrix(self, state: AgentState) -> AgentState:
        """Step 1: Generate assumption-method compatibility matrix"""
        logger.info("Generating assumption-method matrix...")
        
        try:
            # Get preprocessed data
            df = state.get("df_preprocessed")
            if df is None:
                raise ValueError("No preprocessed data available")
            
            # Get variable information
            variables = list(df.columns)
            logger.info(f"Testing assumptions for variables: {variables}")
            
            # Initialize assumption scores
            assumption_scores = {
                "S_lin": {},      # Linearity scores
                "S_nG": {},       # Non-Gaussian scores  
                "S_ANM": {},      # ANM scores
                "S_Gauss": {},    # Gaussian scores
                "S_EqVar": {}     # Equal variance scores
            }
            
            # Test each variable pair for assumptions
            for i, var1 in enumerate(variables):
                for j, var2 in enumerate(variables):
                    if i >= j:  # Skip diagonal and upper triangle
                        continue
                    
                    pair_key = f"{var1}_{var2}"
                    logger.info(f"Testing assumptions for pair: {pair_key}")
                    
                    # 1. Linearity test (GLM vs GAM)
                    linearity_score = self._test_linearity(df[var1], df[var2])
                    assumption_scores["S_lin"][pair_key] = linearity_score
                    
                    # 2. Non-Gaussian test
                    non_gaussian_score = self._test_non_gaussian(df[var1], df[var2])
                    assumption_scores["S_nG"][pair_key] = non_gaussian_score
                    
                    # 3. ANM test (bidirectional independence)
                    anm_score = self._test_anm(df[var1], df[var2])
                    assumption_scores["S_ANM"][pair_key] = anm_score
                    
                    # 4. Gaussian + Equal Variance test
                    gauss_score, eqvar_score = self._test_gaussian_eqvar(df[var1], df[var2])
                    assumption_scores["S_Gauss"][pair_key] = gauss_score
                    assumption_scores["S_EqVar"][pair_key] = eqvar_score
            
            # Update state
            state["assumption_method_scores"] = assumption_scores
            state["assumption_method_matrix_completed"] = True
            
            logger.info("Assumption-method matrix generation completed")
            return state
            
        except Exception as e:
            logger.error(f"Assumption method matrix generation failed: {str(e)}")
            state["error"] = f"Assumption method matrix failed: {str(e)}"
            return state
    
    def _algorithm_scoring(self, state: AgentState) -> AgentState:
        """Step 2: Score algorithms based on assumption compatibility"""
        logger.info("Scoring algorithms based on assumption compatibility...")
        
        try:
            assumption_scores = state.get("assumption_method_scores", {})
            if not assumption_scores:
                raise ValueError("No assumption scores available")
            
            # Algorithm requirements and preferences
            algorithm_configs = {
                "LiNGAM": {
                    "required": ["S_lin", "S_nG"],
                    "preferred": ["S_ANM"],
                    "irrelevant": ["S_Gauss", "S_EqVar"]
                },
                "ANM": {
                    "required": ["S_ANM"],
                    "preferred": ["S_lin"],  # 1 - S_lin
                    "irrelevant": ["S_nG", "S_Gauss", "S_EqVar"]
                },
                "EqVar": {
                    "required": ["S_lin", "S_Gauss", "S_EqVar"],
                    "preferred": [],
                    "irrelevant": ["S_nG", "S_ANM"]
                }
            }
            
            algorithm_scores = {}
            selected_algorithms = []
            
            for alg_name, config in algorithm_configs.items():
                logger.info(f"Scoring algorithm: {alg_name}")
                
                # Calculate scores for each assumption type
                scores = {}
                weights = {}
                
                for assumption_type in ["S_lin", "S_nG", "S_ANM", "S_Gauss", "S_EqVar"]:
                    if assumption_type in assumption_scores:
                        # Aggregate scores across all variable pairs
                        pair_scores = list(assumption_scores[assumption_type].values())
                        if pair_scores:
                            scores[assumption_type] = np.mean(pair_scores)
                            weights[assumption_type] = 1.0
                        else:
                            scores[assumption_type] = 0.5
                            weights[assumption_type] = 0.0
                    else:
                        scores[assumption_type] = 0.5
                        weights[assumption_type] = 0.0
                
                # Calculate role-based scores
                req_scores = []
                pref_scores = []
                irr_scores = []
                
                for assumption_type in config["required"]:
                    if assumption_type in scores:
                        # Conservative transformation for required
                        req_score = scores[assumption_type] ** self.beta_conservative
                        req_scores.append(req_score)
                
                for assumption_type in config["preferred"]:
                    if assumption_type in scores:
                        # Preferred transformation
                        if assumption_type == "S_lin" and alg_name == "ANM":
                            # ANM prefers non-linearity
                            pref_score = 0.5 + 0.5 * (1 - scores[assumption_type])
                        else:
                            pref_score = 0.5 + 0.5 * scores[assumption_type]
                        pref_scores.append(pref_score)
                
                for assumption_type in config["irrelevant"]:
                    if assumption_type in scores:
                        # Irrelevant gets neutral score
                        irr_scores.append(0.5)
                
                # Calculate weighted average score
                total_weight = len(req_scores) + len(pref_scores) + len(irr_scores)
                if total_weight > 0:
                    weighted_score = (sum(req_scores) + sum(pref_scores) + sum(irr_scores)) / total_weight
                else:
                    weighted_score = 0.5
                
                # Calculate soft-AND for required assumptions
                if req_scores:
                    soft_and_score = np.prod(req_scores) ** (1/len(req_scores))
                else:
                    soft_and_score = 1.0
                
                # Final score: combination of weighted average and soft-AND
                final_score = (self.lambda_soft_and * soft_and_score + 
                             (1 - self.lambda_soft_and) * weighted_score)
                
                algorithm_scores[alg_name] = {
                    "final_score": final_score,
                    "weighted_score": weighted_score,
                    "soft_and_score": soft_and_score,
                    "individual_scores": scores
                }
            
            # Select top-k algorithms
            sorted_algorithms = sorted(algorithm_scores.items(), 
                                     key=lambda x: x[1]["final_score"], 
                                     reverse=True)
            selected_algorithms = [alg[0] for alg in sorted_algorithms[:self.top_k_algorithms]]
            
            # Update state
            state["algorithm_scores"] = algorithm_scores
            state["selected_algorithms"] = selected_algorithms
            state["algorithm_scoring_completed"] = True
            
            logger.info(f"Algorithm scoring completed. Selected: {selected_algorithms}")
            return state
            
        except Exception as e:
            logger.error(f"Algorithm scoring failed: {str(e)}")
            state["error"] = f"Algorithm scoring failed: {str(e)}"
            return state
    
    def _run_algorithms(self, state: AgentState) -> AgentState:
        """Step 3: Run selected algorithms in parallel"""
        logger.info("Running selected algorithms in parallel...")
        
        try:
            selected_algorithms = state.get("selected_algorithms", [])
            df = state.get("df_preprocessed")
            
            if not selected_algorithms:
                raise ValueError("No algorithms selected")
            if df is None:
                raise ValueError("No preprocessed data available")
            
            # Run algorithms in parallel
            algorithm_results = {}
            
            with ThreadPoolExecutor(max_workers=len(selected_algorithms)) as executor:
                futures = {}
                
                for alg_name in selected_algorithms:
                    if alg_name == "LiNGAM":
                        future = executor.submit(self._run_lingam, df)
                    elif alg_name == "ANM":
                        future = executor.submit(self._run_anm, df)
                    elif alg_name == "EqVar":
                        future = executor.submit(self._run_eqvar_linear, df)
                    else:
                        logger.warning(f"Unknown algorithm: {alg_name}")
                        continue
                    
                    futures[alg_name] = future
                
                # Collect results
                for alg_name, future in futures.items():
                    try:
                        result = future.result(timeout=300)  # 5 minute timeout
                        algorithm_results[alg_name] = result
                        logger.info(f"Algorithm {alg_name} completed successfully")
                    except Exception as e:
                        logger.error(f"Algorithm {alg_name} failed: {str(e)}")
                        algorithm_results[alg_name] = {"error": str(e)}
            
            # Update state
            state["algorithm_results"] = algorithm_results
            state["run_algorithms_completed"] = True
            
            logger.info("Algorithm execution completed")
            return state
            
        except Exception as e:
            logger.error(f"Algorithm execution failed: {str(e)}")
            state["error"] = f"Algorithm execution failed: {str(e)}"
            return state
    
    def _intermediate_scoring(self, state: AgentState) -> AgentState:
        """Step 4: Calculate intermediate scores (robustness, fidelity, generalizability)"""
        logger.info("Calculating intermediate scores...")
        
        try:
            algorithm_results = state.get("algorithm_results", {})
            df = state.get("df_preprocessed")
            
            if not algorithm_results:
                raise ValueError("No algorithm results available")
            if df is None:
                raise ValueError("No preprocessed data available")
            
            intermediate_scores = {}
            graph_candidates = []
            
            for alg_name, result in algorithm_results.items():
                if "error" in result:
                    continue
                
                logger.info(f"Evaluating {alg_name} results...")
                
                # 1. Robustness: Bootstrap evaluation
                robustness_score = self._evaluate_robustness(df, result, alg_name)
                
                # 2. Fidelity: BIC/MDL score
                fidelity_score = self._evaluate_fidelity(df, result)
                
                # 3. Generalizability: Cross-validation
                generalizability_score = self._evaluate_generalizability(df, result)
                
                # 4. Assumption-Method: Reuse previous scores
                assumption_score = self._get_assumption_score(alg_name, state)
                
                intermediate_scores[alg_name] = {
                    "robustness": robustness_score,
                    "fidelity": fidelity_score,
                    "generalizability": generalizability_score,
                    "assumption_method": assumption_score
                }
                
                # Store graph candidate
                graph_candidates.append({
                    "algorithm": alg_name,
                    "graph": result.get("graph", {}),
                    "scores": intermediate_scores[alg_name],
                    "metadata": result.get("metadata", {})
                })
            
            # Update state
            state["intermediate_scores"] = intermediate_scores
            state["candidate_graphs"] = graph_candidates
            state["intermediate_scoring_completed"] = True
            
            logger.info("Intermediate scoring completed")
            return state
            
        except Exception as e:
            logger.error(f"Intermediate scoring failed: {str(e)}")
            state["error"] = f"Intermediate scoring failed: {str(e)}"
            return state
    
    def _final_graph_selection(self, state: AgentState) -> AgentState:
        """Step 5: Select final causal graph"""
        logger.info("Selecting final causal graph...")
        
        try:
            candidate_graphs = state.get("candidate_graphs", [])
            intermediate_scores = state.get("intermediate_scores", {})
            
            if not candidate_graphs:
                raise ValueError("No candidate graphs available")
            
            # Calculate composite scores for each candidate
            final_scores = {}
            for candidate in candidate_graphs:
                alg_name = candidate["algorithm"]
                scores = intermediate_scores.get(alg_name, {})
                
                # Weighted combination of all scores
                composite_score = (
                    0.3 * scores.get("robustness", 0.5) +
                    0.3 * scores.get("fidelity", 0.5) +
                    0.2 * scores.get("generalizability", 0.5) +
                    0.2 * scores.get("assumption_method", 0.5)
                )
                
                final_scores[alg_name] = composite_score
            
            # Select best graph
            best_algorithm = max(final_scores.items(), key=lambda x: x[1])[0]
            best_candidate = next(c for c in candidate_graphs if c["algorithm"] == best_algorithm)
            
            # Generate selection reasoning
            reasoning = self._generate_selection_reasoning(best_candidate, final_scores, intermediate_scores)
            
            # Update state
            state["selected_graph"] = best_candidate["graph"]
            state["graph_selection_reasoning"] = reasoning
            state["causal_discovery_status"] = "completed"
            state["final_graph_selection_completed"] = True
            
            logger.info(f"Final graph selected: {best_algorithm}")
            return state
            
        except Exception as e:
            logger.error(f"Final graph selection failed: {str(e)}")
            state["error"] = f"Final graph selection failed: {str(e)}"
            return state
    
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
            result = self.use_tool("independence_tool", "anm_test", x, y)
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
    
    def _run_anm(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run ANM algorithm"""
        try:
            result = self.use_tool("anm_tool", "anm_discovery", df)
            return result
        except Exception as e:
            logger.error(f"ANM execution failed: {e}")
            return {"error": str(e)}
    
    def _run_eqvar_linear(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run EqVar Linear algorithm"""
        try:
            result = self.use_tool("eqvar_linear_tool", "eqvar_linear", df)
            return result
        except Exception as e:
            logger.error(f"EqVar Linear execution failed: {e}")
            return {"error": str(e)}
    
    # === Evaluation Methods ===
    
    def _evaluate_robustness(self, df: pd.DataFrame, result: Dict[str, Any], alg_name: str) -> float:
        """Evaluate robustness using bootstrap"""
        try:
            bootstrap_result = self.use_tool("bootstrapper", "bootstrap_evaluation", df, result, alg_name, self.bootstrap_iterations)
            return bootstrap_result.get("robustness_score", 0.5)
        except Exception as e:
            logger.warning(f"Robustness evaluation failed: {e}")
            return 0.5
    
    def _evaluate_fidelity(self, df: pd.DataFrame, result: Dict[str, Any]) -> float:
        """Evaluate fidelity using BIC/MDL"""
        try:
            fidelity_result = self.use_tool("graph_evaluator", "fidelity_evaluation", df, result)
            return fidelity_result.get("fidelity_score", 0.5)
        except Exception as e:
            logger.warning(f"Fidelity evaluation failed: {e}")
            return 0.5
    
    def _evaluate_generalizability(self, df: pd.DataFrame, result: Dict[str, Any]) -> float:
        """Evaluate generalizability using cross-validation"""
        try:
            cv_result = self.use_tool("graph_evaluator", "cv_evaluation", df, result, self.cv_folds)
            return cv_result.get("generalizability_score", 0.5)
        except Exception as e:
            logger.warning(f"Generalizability evaluation failed: {e}")
            return 0.5
    
    def _get_assumption_score(self, alg_name: str, state: AgentState) -> float:
        """Get assumption-method score for algorithm"""
        algorithm_scores = state.get("algorithm_scores", {})
        if alg_name in algorithm_scores:
            return algorithm_scores[alg_name].get("final_score", 0.5)
        return 0.5
    
    def _generate_selection_reasoning(self, candidate: Dict[str, Any], final_scores: Dict[str, float], 
                                    intermediate_scores: Dict[str, Dict[str, float]]) -> str:
        """Generate reasoning for graph selection"""
        alg_name = candidate["algorithm"]
        scores = intermediate_scores.get(alg_name, {})
        
        reasoning = f"""
        Selected Algorithm: {alg_name}
        Final Score: {final_scores[alg_name]:.3f}
        
        Score Breakdown:
        - Robustness: {scores.get('robustness', 0.5):.3f}
        - Fidelity: {scores.get('fidelity', 0.5):.3f}
        - Generalizability: {scores.get('generalizability', 0.5):.3f}
        - Assumption-Method: {scores.get('assumption_method', 0.5):.3f}
        
        Selection Criteria: Weighted combination of robustness (30%), fidelity (30%), 
        generalizability (20%), and assumption-method compatibility (20%).
        """
        
        return reasoning.strip()
    
    # === Tool Implementations ===
    
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
            return IndependenceTool.anm_test(args[0], args[1])
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
    
    def _eqvar_linear_tool(self, method: str, *args, **kwargs) -> Dict[str, Any]:
        """EqVar Linear algorithm tool implementation"""
        from .tools import EqVarLinearTool
        
        if method == "eqvar_linear":
            return EqVarLinearTool.eqvar_linear(args[0])
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
