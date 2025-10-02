# agents/causal_discovery/tools.py
"""
Causal Discovery Agent Tools Implementation

This module implements the specialized tools for causal discovery:
- Statistical testing tools
- Independence testing tools  
- Causal discovery algorithm tools
- Evaluation and analysis tools
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore') 

logger = logging.getLogger(__name__)

class StatsTool:
    """Statistical testing tools for assumption validation"""
    
    @staticmethod
    def linearity_test(x: pd.Series, y: pd.Series) -> Dict[str, Any]:
        """Test linearity using GLM vs GAM comparison"""
        try:
            # Simple linear regression
            lr = LinearRegression()
            lr.fit(x.values.reshape(-1, 1), y.values)
            linear_pred = lr.predict(x.values.reshape(-1, 1))
            linear_mse = mean_squared_error(y.values, linear_pred)
            
            # Gaussian Process (non-parametric) as GAM proxy
            gp = GaussianProcessRegressor(random_state=42)
            gp.fit(x.values.reshape(-1, 1), y.values)
            gam_pred = gp.predict(x.values.reshape(-1, 1))
            gam_mse = mean_squared_error(y.values, gam_pred)
            
            # Calculate linearity score (lower MSE ratio = more linear)
            if gam_mse > 0:
                linearity_ratio = linear_mse / gam_mse
                linearity_score = min(1.0, max(0.0, 1.0 - linearity_ratio + 0.5))
            else:
                linearity_score = 0.5
            
            return {
                "linearity_score": linearity_score,
                "linear_mse": linear_mse,
                "gam_mse": gam_mse,
                "ratio": linearity_ratio if gam_mse > 0 else 1.0
            }
            
        except Exception as e:
            logger.warning(f"Linearity test failed: {e}")
            return {"linearity_score": 0.5, "error": str(e)}
    
    @staticmethod
    def normality_test(x: pd.Series, y: pd.Series) -> Dict[str, Any]:
        """Test normality using Jarque-Bera and Shapiro-Wilk tests"""
        try:
            # Test both variables
            x_jb_stat, x_jb_p = stats.jarque_bera(x.dropna())
            y_jb_stat, y_jb_p = stats.jarque_bera(y.dropna())
            
            x_sw_stat, x_sw_p = stats.shapiro(x.dropna())
            y_sw_stat, y_sw_p = stats.shapiro(y.dropna())
            
            # Combine p-values (lower p-value = more non-Gaussian)
            combined_p = np.mean([x_jb_p, y_jb_p, x_sw_p, y_sw_p])
            
            # Convert to non-Gaussian score (lower p-value = higher score)
            non_gaussian_score = 1.0 - min(1.0, combined_p)
            
            return {
                "non_gaussian_score": non_gaussian_score,
                "x_jb_p": x_jb_p,
                "y_jb_p": y_jb_p,
                "x_sw_p": x_sw_p,
                "y_sw_p": y_sw_p,
                "combined_p": combined_p
            }
            
        except Exception as e:
            logger.warning(f"Normality test failed: {e}")
            return {"non_gaussian_score": 0.5, "error": str(e)}
    
    @staticmethod
    def gaussian_eqvar_test(x: pd.Series, y: pd.Series) -> Dict[str, Any]:
        """Test Gaussian and equal variance assumptions"""
        try:
            # Test normality of residuals
            lr = LinearRegression()
            lr.fit(x.values.reshape(-1, 1), y.values)
            residuals = y.values - lr.predict(x.values.reshape(-1, 1))
            
            # Normality test on residuals
            jb_stat, jb_p = stats.jarque_bera(residuals)
            sw_stat, sw_p = stats.shapiro(residuals)
            
            gaussian_score = np.mean([jb_p, sw_p])
            
            # Equal variance test (Levene's test)
            # Split data into groups based on x values
            x_median = x.median()
            group1 = y[x <= x_median].dropna()
            group2 = y[x > x_median].dropna()
            
            if len(group1) > 1 and len(group2) > 1:
                levene_stat, levene_p = stats.levene(group1, group2)
                eqvar_score = levene_p
            else:
                eqvar_score = 0.5
            
            return {
                "gaussian_score": gaussian_score,
                "eqvar_score": eqvar_score,
                "jb_p": jb_p,
                "sw_p": sw_p,
                "levene_p": levene_p if 'levene_p' in locals() else 0.5
            }
            
        except Exception as e:
            logger.warning(f"Gaussian/EqVar test failed: {e}")
            return {"gaussian_score": 0.5, "eqvar_score": 0.5, "error": str(e)}

class IndependenceTool:
    """Independence testing tools for ANM and other tests"""
    
    @staticmethod
    def anm_test(x: pd.Series, y: pd.Series) -> Dict[str, Any]:
        """Test ANM assumption using independence testing"""
        try:
            # Simple independence test using correlation
            correlation = x.corr(y)
            
            # Test independence of residuals
            lr = LinearRegression()
            lr.fit(x.values.reshape(-1, 1), y.values)
            residuals = y.values - lr.predict(x.values.reshape(-1, 1))
            
            # Test independence between x and residuals
            x_residual_corr = x.corr(pd.Series(residuals, index=x.index))
            
            # ANM score based on independence of residuals
            anm_score = 1.0 - abs(x_residual_corr)
            
            return {
                "anm_score": max(0.0, min(1.0, anm_score)),
                "correlation": correlation,
                "residual_correlation": x_residual_corr
            }
            
        except Exception as e:
            logger.warning(f"ANM test failed: {e}")
            return {"anm_score": 0.5, "error": str(e)}

class LiNGAMTool:
    """LiNGAM algorithm implementation"""
    
    @staticmethod
    def direct_lingam(df: pd.DataFrame) -> Dict[str, Any]:
        """DirectLiNGAM algorithm implementation"""
        try:
            # Simplified LiNGAM implementation
            n_vars = len(df.columns)
            variables = list(df.columns)
            
            # Initialize adjacency matrix
            adjacency_matrix = np.zeros((n_vars, n_vars))
            
            # Simple causal ordering based on variance
            var_order = df.var().sort_values(ascending=True).index.tolist()
            
            # Build causal graph
            for i, var1 in enumerate(var_order):
                for j, var2 in enumerate(var_order):
                    if i < j:  # Only consider forward relationships
                        # Simple correlation-based edge detection
                        corr = df[var1].corr(df[var2])
                        if abs(corr) > 0.3:  # Threshold
                            adjacency_matrix[i][j] = corr
            
            # Convert to edge list
            edges = []
            for i in range(n_vars):
                for j in range(n_vars):
                    if adjacency_matrix[i][j] != 0:
                        edges.append({
                            "from": variables[i],
                            "to": variables[j],
                            "weight": adjacency_matrix[i][j]
                        })
            
            return {
                "graph": {
                    "edges": edges,
                    "adjacency_matrix": adjacency_matrix.tolist(),
                    "variables": variables
                },
                "metadata": {
                    "method": "DirectLiNGAM",
                    "n_variables": n_vars,
                    "n_edges": len(edges)
                }
            }
            
        except Exception as e:
            logger.error(f"LiNGAM execution failed: {e}")
            return {"error": str(e)}

class ANMTool:
    """Additive Noise Model algorithm implementation"""
    
    @staticmethod
    def anm_discovery(df: pd.DataFrame) -> Dict[str, Any]:
        """ANM algorithm implementation"""
        try:
            n_vars = len(df.columns)
            variables = list(df.columns)
            edges = []
            
            # Test all variable pairs for ANM directionality
            for i, var1 in enumerate(variables):
                for j, var2 in enumerate(variables):
                    if i != j:
                        # Test X -> Y direction
                        x_to_y_score = IndependenceTool.anm_test(df[var1], df[var2])["anm_score"]
                        # Test Y -> X direction  
                        y_to_x_score = IndependenceTool.anm_test(df[var2], df[var1])["anm_score"]
                        
                        # Determine direction based on ANM scores
                        if x_to_y_score > y_to_x_score and x_to_y_score > 0.6:
                            edges.append({
                                "from": var1,
                                "to": var2,
                                "weight": x_to_y_score,
                                "method": "ANM"
                            })
                        elif y_to_x_score > x_to_y_score and y_to_x_score > 0.6:
                            edges.append({
                                "from": var2,
                                "to": var1,
                                "weight": y_to_x_score,
                                "method": "ANM"
                            })
            
            return {
                "graph": {
                    "edges": edges,
                    "variables": variables
                },
                "metadata": {
                    "method": "ANM",
                    "n_variables": n_vars,
                    "n_edges": len(edges)
                }
            }
            
        except Exception as e:
            logger.error(f"ANM execution failed: {e}")
            return {"error": str(e)}

class EqVarLinearTool:
    """Equal Variance Linear algorithm implementation"""
    
    @staticmethod
    def eqvar_linear(df: pd.DataFrame) -> Dict[str, Any]:
        """EqVar Linear algorithm implementation"""
        try:
            n_vars = len(df.columns)
            variables = list(df.columns)
            edges = []
            
            # Test all variable pairs for linear relationships with equal variance
            for i, var1 in enumerate(variables):
                for j, var2 in enumerate(variables):
                    if i != j:
                        # Test linearity and equal variance
                        linearity_result = StatsTool.linearity_test(df[var1], df[var2])
                        gauss_eqvar_result = StatsTool.gaussian_eqvar_test(df[var1], df[var2])
                        
                        # Combine scores
                        combined_score = (
                            0.5 * linearity_result["linearity_score"] +
                            0.3 * gauss_eqvar_result["gaussian_score"] +
                            0.2 * gauss_eqvar_result["eqvar_score"]
                        )
                        
                        if combined_score > 0.6:
                            edges.append({
                                "from": var1,
                                "to": var2,
                                "weight": combined_score,
                                "method": "EqVarLinear"
                            })
            
            return {
                "graph": {
                    "edges": edges,
                    "variables": variables
                },
                "metadata": {
                    "method": "EqVarLinear",
                    "n_variables": n_vars,
                    "n_edges": len(edges)
                }
            }
            
        except Exception as e:
            logger.error(f"EqVar Linear execution failed: {e}")
            return {"error": str(e)}

class Bootstrapper:
    """Bootstrap evaluation tools"""
    
    @staticmethod
    def bootstrap_evaluation(df: pd.DataFrame, result: Dict[str, Any], 
                           algorithm: str, n_iterations: int = 100) -> Dict[str, Any]:
        """Bootstrap evaluation for robustness"""
        try:
            n_samples = len(df)
            edge_frequencies = {}
            
            # Bootstrap sampling
            for i in range(n_iterations):
                # Sample with replacement
                bootstrap_df = df.sample(n=n_samples, replace=True, random_state=i)
                
                # Run algorithm on bootstrap sample
                if algorithm == "LiNGAM":
                    bootstrap_result = LiNGAMTool.direct_lingam(bootstrap_df)
                elif algorithm == "ANM":
                    bootstrap_result = ANMTool.anm_discovery(bootstrap_df)
                elif algorithm == "EqVar":
                    bootstrap_result = EqVarLinearTool.eqvar_linear(bootstrap_df)
                else:
                    continue
                
                # Count edges
                if "graph" in bootstrap_result and "edges" in bootstrap_result["graph"]:
                    for edge in bootstrap_result["graph"]["edges"]:
                        edge_key = f"{edge['from']}->{edge['to']}"
                        edge_frequencies[edge_key] = edge_frequencies.get(edge_key, 0) + 1
            
            # Calculate robustness score
            if edge_frequencies:
                max_frequency = max(edge_frequencies.values())
                robustness_score = max_frequency / n_iterations
            else:
                robustness_score = 0.0
            
            return {
                "robustness_score": robustness_score,
                "edge_frequencies": edge_frequencies,
                "n_iterations": n_iterations
            }
            
        except Exception as e:
            logger.warning(f"Bootstrap evaluation failed: {e}")
            return {"robustness_score": 0.5, "error": str(e)}

class GraphEvaluator:
    """Graph evaluation tools"""
    
    @staticmethod
    def fidelity_evaluation(df: pd.DataFrame, result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate fidelity using BIC/MDL"""
        try:
            if "graph" not in result or "edges" not in result["graph"]:
                return {"fidelity_score": 0.5}
            
            edges = result["graph"]["edges"]
            n_edges = len(edges)
            n_vars = len(df.columns)
            n_samples = len(df)
            
            # Simple BIC calculation
            # BIC = -2 * log_likelihood + k * log(n)
            # For simplicity, use number of edges as complexity
            complexity_penalty = n_edges * np.log(n_samples)
            
            # Calculate likelihood (simplified)
            log_likelihood = 0
            for edge in edges:
                from_var = edge["from"]
                to_var = edge["to"]
                if from_var in df.columns and to_var in df.columns:
                    # Simple correlation-based likelihood
                    corr = abs(df[from_var].corr(df[to_var]))
                    log_likelihood += np.log(corr + 1e-10)
            
            bic = -2 * log_likelihood + complexity_penalty
            
            # Convert BIC to fidelity score (lower BIC = higher fidelity)
            # Normalize to 0-1 range
            max_bic = n_vars * n_vars * np.log(n_samples)  # Maximum possible BIC
            fidelity_score = max(0.0, min(1.0, 1.0 - (bic / max_bic)))
            
            return {
                "fidelity_score": fidelity_score,
                "bic": bic,
                "n_edges": n_edges,
                "log_likelihood": log_likelihood
            }
            
        except Exception as e:
            logger.warning(f"Fidelity evaluation failed: {e}")
            return {"fidelity_score": 0.5, "error": str(e)}
    
    @staticmethod
    def cv_evaluation(df: pd.DataFrame, result: Dict[str, Any], cv_folds: int = 5) -> Dict[str, Any]:
        """Cross-validation evaluation for generalizability"""
        try:
            if "graph" not in result or "edges" not in result["graph"]:
                return {"generalizability_score": 0.5}
            
            edges = result["graph"]["edges"]
            if not edges:
                return {"generalizability_score": 0.5}
            
            # Use cross-validation on linear models for each edge
            cv_scores = []
            
            for edge in edges:
                from_var = edge["from"]
                to_var = edge["to"]
                
                if from_var in df.columns and to_var in df.columns:
                    X = df[from_var].values.reshape(-1, 1)
                    y = df[to_var].values
                    
                    # Cross-validation
                    lr = LinearRegression()
                    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                    scores = cross_val_score(lr, X, y, cv=kf, scoring='neg_mean_squared_error')
                    
                    if len(scores) > 0:
                        cv_scores.append(np.mean(scores))
            
            if cv_scores:
                # Convert MSE to generalizability score
                mean_cv_score = np.mean(cv_scores)
                generalizability_score = max(0.0, min(1.0, -mean_cv_score / 1000 + 0.5))
            else:
                generalizability_score = 0.5
            
            return {
                "generalizability_score": generalizability_score,
                "cv_scores": cv_scores,
                "n_edges_tested": len(cv_scores)
            }
            
        except Exception as e:
            logger.warning(f"CV evaluation failed: {e}")
            return {"generalizability_score": 0.5, "error": str(e)}

class GraphOps:
    """Graph operations tools"""
    
    @staticmethod
    def convert_dag_to_pag(graph: Dict[str, Any]) -> Dict[str, Any]:
        """Convert DAG to PAG (Partially Ancestral Graph)"""
        # Placeholder implementation
        return graph
    
    @staticmethod
    def merge_graphs(graphs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple graphs using voting"""
        if not graphs:
            return {"edges": [], "variables": []}
        
        # Collect all edges
        all_edges = []
        all_variables = set()
        
        for graph in graphs:
            if "edges" in graph:
                all_edges.extend(graph["edges"])
            if "variables" in graph:
                all_variables.update(graph["variables"])
        
        # Count edge frequencies
        edge_counts = {}
        for edge in all_edges:
            edge_key = f"{edge['from']}->{edge['to']}"
            if edge_key not in edge_counts:
                edge_counts[edge_key] = {"count": 0, "weight_sum": 0, "edge": edge}
            edge_counts[edge_key]["count"] += 1
            edge_counts[edge_key]["weight_sum"] += edge.get("weight", 0)
        
        # Select edges that appear in majority of graphs
        n_graphs = len(graphs)
        threshold = n_graphs // 2 + 1
        
        merged_edges = []
        for edge_key, data in edge_counts.items():
            if data["count"] >= threshold:
                edge = data["edge"].copy()
                edge["weight"] = data["weight_sum"] / data["count"]  # Average weight
                edge["frequency"] = data["count"] / n_graphs
                merged_edges.append(edge)
        
        return {
            "edges": merged_edges,
            "variables": list(all_variables),
            "n_input_graphs": n_graphs,
            "n_merged_edges": len(merged_edges)
        }
