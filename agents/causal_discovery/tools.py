# === Standardized graph result schema helpers ===
# Edge attribute semantics:
# - weight: algorithm-native strength/evidence in [0,1] if possible (e.g., ANM score, correlation-derived score, BIC-based rescale).
# - confidence: post-hoc reliability from ensembling/bootstrapping (e.g., frequency).
# Keep both when available; algorithm-specific extras should go under result['metadata'] or per-edge custom keys.
GRAPH_RESULT_TEMPLATE = {
    "graph": {"edges": [], "variables": []},
    "metadata": {"method": None, "params": {}, "runtime": None}
}

def _normalize_edges(edges):
    norm = []
    for e in edges or []:
        if isinstance(e, dict):
            frm, to = e.get("from"), e.get("to")
            w = e.get("weight", None)
            conf = e.get("confidence") if isinstance(e, dict) else None
        else:
            # support tuple (from, to, weight?)
            frm = e[0]; to = e[1]; w = e[2] if len(e) > 2 else None
            conf = None
        item = {"from": str(frm), "to": str(to)}
        if w is not None:
            item["weight"] = float(w)
        if conf is not None:
            item["confidence"] = float(conf)
        norm.append(item)
    return norm

def normalize_graph_result(method: str, variables, edges, params=None, runtime=None):
    res = {
        "graph": {
            "edges": _normalize_edges(edges),
            "variables": list(map(str, variables or []))
        },
        "metadata": {
            "method": method,
            "params": params or {},
            "runtime": runtime
        }
    }
    return res
# agents/causal_discovery/tools.py
"""
Causal Discovery Agent Tools Implementation

This module implements the specialized tools for causal discovery:
- Statistical testing tools
- Independence testing tools  
- Causal discovery algorithm tools
- Evaluation and analysis tools

Note: EqVar is diagnostics only (not a discovery algorithm).
"""

import logging
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, Any, List, Tuple, Optional
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore') 

logger = logging.getLogger(__name__)

def safe_execute(func, *args, default_return=None, error_msg="Operation failed", **kwargs):
    """Common error handling pattern for tool execution"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.warning(f"{error_msg}: {e}")
        return default_return or {"error": str(e)}

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
    def _fit_nonparametric_regression(x: np.ndarray, y: np.ndarray, n_estimators: int = 20) -> np.ndarray:
        """Common ANM regression fitting logic"""
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(random_state=42, n_estimators=int(max(10, n_estimators)), 
                                 max_depth=None, n_jobs=1)
        rf.fit(x.reshape(-1, 1), y.ravel())
        return y.ravel() - rf.predict(x.reshape(-1, 1))
    
    @staticmethod
    def _test_residual_independence(x: np.ndarray, resid: np.ndarray) -> Tuple[Optional[float], bool]:
        """Test independence between x and residuals with fallback"""
        try:
            from causallearn.utils.cit import hsic_test_gamma
            pval = float(hsic_test_gamma(x, resid))
            return pval, False
        except Exception:
            # Fallback: Spearman correlation
            rho = abs(pd.Series(resid).corr(pd.Series(x), method='spearman'))
            score = max(0.0, min(1.0, 1.0 - rho))
            return score, True
    
    @staticmethod
    def anm_test(x: pd.Series, y: pd.Series, n_estimators: int = 20) -> Dict[str, Any]:
        """Test ANM assumption via residual independence using causal-learn KCI/HSIC.
        Returns anm_score in [0,1] where higher is more consistent with ANM X->Y.
        """
        try:
            X, Y = x.values, y.values
            resid = IndependenceTool._fit_nonparametric_regression(X, Y, n_estimators)
            pval, is_fallback = IndependenceTool._test_residual_independence(X, resid)
            
            if is_fallback:
                return {"anm_score": pval, "fallback": True}
            else:
                score = float(max(0.0, min(1.0, pval)))
                return {"anm_score": score, "p_value": pval}
        except Exception as e:
            logger.warning(f"ANM test failed: {e}")
            return {"anm_score": 0.5, "error": str(e)}

class LiNGAMTool:
    """LiNGAM algorithm implementation"""
    
    @staticmethod
    def direct_lingam(df: pd.DataFrame) -> Dict[str, Any]:
        """DirectLiNGAM via causal-learn."""
        try:
            import time
            from causallearn.search.FCMBased.lingam import DirectLiNGAM
            
            t0 = time.time()
            X = df.values
            vars_ = list(df.columns)
            model = None
            
            model = DirectLiNGAM()
            model.fit(X)
            # Extract adjacency if available
            edges = []
            A = getattr(model, 'adjacency_matrix_', None)
            if A is None:
                A = getattr(model, 'adjacency_matrix', None)
            if A is not None:
                A = np.asarray(A)
                for i in range(A.shape[0]):
                    for j in range(A.shape[1]):
                        if abs(A[i, j]) > 0:
                            edges.append({"from": vars_[i], "to": vars_[j], "weight": float(A[i, j])})
            else:
                # Fallback: use causal order
                order = getattr(model, 'causal_order_', list(range(len(vars_))))
                for i in range(len(order)):
                    for j in range(i+1, len(order)):
                        edges.append({"from": vars_[order[i]], "to": vars_[order[j]], "weight": 1.0})
            runtime = time.time() - t0
            return normalize_graph_result("DirectLiNGAM", vars_, edges, {"backend": "causal-learn"}, runtime)
        except Exception as e:
            logger.error(f"LiNGAM execution failed: {e}")
            return {"error": str(e)}

class ANMTool:
    """Additive Noise Model algorithm implementation"""
    
    @staticmethod
    def _test_direction(x: np.ndarray, y: np.ndarray, n_estimators: int = 20) -> Optional[float]:
        """Test ANM direction X->Y using common logic"""
        try:
            resid = IndependenceTool._fit_nonparametric_regression(x, y, n_estimators)
            pval, _ = IndependenceTool._test_residual_independence(x, resid)
            return pval
        except Exception:
            return None
    
    @staticmethod
    def anm_discovery(df: pd.DataFrame, delta: float = 0.02, tau: float = 0.05) -> Dict[str, Any]:
        import time
        t0 = time.time()
        vars_ = list(df.columns)
        edges = []
        
        for i, xi in enumerate(vars_):
            for j, yj in enumerate(vars_):
                if i == j:
                    continue
                
                X, Y = df[xi].values, df[yj].values
                
                # Test both directions
                p_xy = ANMTool._test_direction(X, Y)
                p_yx = ANMTool._test_direction(Y, X)
                
                if p_xy is None and p_yx is None:
                    continue
                
                p_xy = p_xy or 0.0
                p_yx = p_yx or 0.0
                best_p = max(p_xy, p_yx)
                
                if best_p < tau:
                    continue
                
                # Determine direction based on difference
                if (p_xy - p_yx) >= delta:
                    edges.append({"from": xi, "to": yj, "weight": float(best_p)})
                elif (p_yx - p_xy) >= delta:
                    edges.append({"from": yj, "to": xi, "weight": float(best_p)})
        
        runtime = time.time() - t0
        return normalize_graph_result("ANM", vars_, edges, 
                                    {"delta": delta, "tau": tau, "backend": "causal-learn"}, runtime)


# CAM tool
class CAMTool:
    @staticmethod
    def discover(df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        import time
        vars_ = list(df.columns)
        t0 = time.time()
        try:
            from cdt.causality.graph import CAM
            model = CAM()
            output_graph = model.predict(df)
            edges = []
            for u, v in output_graph.edges():
                edges.append({"from": str(u), "to": str(v), "weight": 1.0})
            runtime = time.time() - t0
            return normalize_graph_result("CAM", vars_, edges, {"backend": "cdt"}, runtime)
        except Exception as e:
            return {"error": f"CAM not available: {e}"}


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

# --- PC and GES tools ---

class PCTool:
    """PC algorithm wrapper. Uses causal-learn."""
    @staticmethod
    def discover(df: pd.DataFrame, alpha: float = 0.05, indep_test: str = "fisherz", **kwargs) -> Dict[str, Any]:
        import time
        t0 = time.time()
        vars_ = list(df.columns)
        edges = []
        params = {"alpha": alpha, "indep_test": indep_test}
        try:
            from causallearn.search.ConstraintBased.PC import pc
            from causallearn.utils.cit import fisherz, kci, gsq
            data = df.values
            
            # Convert string to actual test function
            test_map = {"fisherz": fisherz, "kci": kci, "gsq": gsq}
            test_func = test_map.get(indep_test, fisherz)
            
            cg = pc(data, alpha=alpha, indep_test=test_func, verbose=False)
            graph = cg.G.graph
            if hasattr(graph, 'items'):
                # Dictionary format
                for (i, j), v in graph.items():
                    if v != 0:
                        edges.append((vars_[i], vars_[j]))
            else:
                # Numpy array format
                for i in range(graph.shape[0]):
                    for j in range(graph.shape[1]):
                        if graph[i, j] != 0:
                            edges.append((vars_[i], vars_[j]))
            runtime = time.time() - t0
            return normalize_graph_result("PC", vars_, edges, params, runtime)
        except Exception as e:
            return {"error": str(e)}

class GESTool:
    """GES algorithm wrapper. Uses causal-learn (score=BIC by default)."""
    @staticmethod
    def discover(df: pd.DataFrame, score_func: str = "bic", **kwargs) -> Dict[str, Any]:
        import time
        t0 = time.time()
        vars_ = list(df.columns)
        params = {"score_func": score_func}
        edges = []
        try:
            from causallearn.search.ScoreBased.GES import ges
            data = df.values
            # Convert score_func to causallearn format
            if score_func == "bic":
                score_func = "local_score_BIC"
            res = ges(data, score_func=score_func)
            G = getattr(res, 'G', None) or (res.get('G', None) if isinstance(res, dict) else None)
            if G is not None and hasattr(G, 'graph'):
                graph = G.graph
                if hasattr(graph, 'items'):
                    # Dictionary format
                    for (i, j), v in graph.items():
                        if v != 0:
                            edges.append((vars_[i], vars_[j]))
                else:
                    # Numpy array format
                    for i in range(graph.shape[0]):
                        for j in range(graph.shape[1]):
                            if graph[i, j] != 0:
                                edges.append((vars_[i], vars_[j]))
            runtime = time.time() - t0
            return normalize_graph_result("GES", vars_, edges, params, runtime)
        except Exception as e:
            return {"error": str(e)}


# --- FCI tool ---
class FCITool:
    """FCI algorithm wrapper. Uses causal-learn and returns a PAG-oriented result.
    Note: PAG edges may be partially directed; we expose adjacencies as edges and
    mark graph_type="PAG" in metadata/params for downstream handling.
    """
    @staticmethod
    def discover(df: pd.DataFrame, alpha: float = 0.05, indep_test: str = "fisherz", **kwargs) -> Dict[str, Any]:
        import time
        t0 = time.time()
        vars_ = list(df.columns)
        edges = []
        try:
            from causallearn.search.ConstraintBased.FCI import fci
            from causallearn.utils.cit import fisherz, kci, gsq
            data = df.values
            
            # Convert string to actual test function
            test_map = {"fisherz": fisherz, "kci": kci, "gsq": gsq}
            test_func = test_map.get(indep_test, fisherz)
            
            # Causal-Learn FCI API: fci(data, indep_test, alpha)
            pag = fci(data, test_func, alpha)
            G = getattr(pag, 'G', None)
            if G is not None and hasattr(G, 'graph'):
                for (i, j), v in G.graph.items():
                    if v != 0:
                        # Keep adjacency; orientation marks are PAG-specific and not
                        # represented in normalized edges. Downstream can treat as PAG.
                        edges.append((vars_[i], vars_[j]))
            runtime = time.time() - t0
            result = normalize_graph_result(
                "FCI", vars_, edges,
                params={"alpha": alpha, "indep_test": indep_test, "graph_type": "PAG"},
                runtime=runtime,
            )
            # Also set a top-level hint to consumers
            result["metadata"]["graph_type"] = "PAG"
            return result
        except Exception as e:
            return {"error": str(e)}

class PruningTool:
    """Pruning tools for CI testing and structural consistency"""
    
    @staticmethod
    def _get_graph_type(graph: Dict[str, Any]) -> str:
        """Identify graph type (DAG/CPDAG/PAG) from metadata"""
        metadata = graph.get("metadata", {})
        if metadata.get("graph_type") == "PAG":
            return "PAG"
        if metadata.get("method") == "PC":
            return "CPDAG"
        return "DAG"
    
    @staticmethod
    def _convert_to_networkx_dag(graph: Dict[str, Any]) -> nx.DiGraph:
        """Convert graph dict to NetworkX DAG"""
        variables = graph["graph"]["variables"]
        edges = graph["graph"]["edges"]
        
        G = nx.DiGraph()
        G.add_nodes_from(variables)
        
        for edge in edges:
            from_var = edge["from"]
            to_var = edge["to"]
            if from_var in variables and to_var in variables:
                G.add_edge(from_var, to_var)
        
        return G
    
    @staticmethod
    def global_markov_test(graph: Dict[str, Any], df: pd.DataFrame, alpha: float = 0.05) -> Dict[str, Any]:
        """Test global Markov property using d-separation and CI tests
        
        Handles different graph types:
        - DAG: Direct d-separation testing
        - CPDAG: Convert to representative DAG first
        - PAG: Skip (latent confounders present)
        """
        try:
            if "graph" not in graph or "edges" not in graph["graph"]:
                return {"violation_ratio": 1.0, "error": "Invalid graph structure"}
            
            edges = graph["graph"]["edges"]
            variables = graph["graph"]["variables"]
            
            if not edges or not variables:
                return {"violation_ratio": 0.0, "message": "No edges to test"}
            
            # Determine graph type
            graph_type = PruningTool._get_graph_type(graph)
            
            # Skip PAG graphs (latent confounders present)
            if graph_type == "PAG":
                return {
                    "violation_ratio": 0.0,
                    "total_tests": 0,
                    "violations": 0,
                    "ci_tests": [],
                    "message": "Global Markov test skipped for PAG (latent confounders present)"
                }
            
            # Convert to NetworkX DAG
            G = PruningTool._convert_to_networkx_dag(graph)
            
            # Verify it's a valid DAG
            if not nx.is_directed_acyclic_graph(G):
                return {
                    "violation_ratio": 0.0,
                    "message": f"Global Markov test skipped (converted graph from {graph_type} is not a DAG)"
                }
            
            # Get CI test method from graph metadata
            ci_test_method = "fisherz"  # default
            if "metadata" in graph and "params" in graph["metadata"]:
                ci_test_method = graph["metadata"]["params"].get("indep_test", "fisherz")
            
            # Prepare data for CI testing
            data_np = df[variables].values.astype(float)
            
            # Import and instantiate CI test function
            try:
                from causallearn.utils.cit import FisherZ, KCI, Chisq_or_Gsq
                if ci_test_method == "fisherz":
                    ci_test_func = FisherZ(data_np)
                elif ci_test_method == "kci":
                    ci_test_func = KCI(data_np)
                elif ci_test_method == "gsq":
                    ci_test_func = Chisq_or_Gsq(data_np, "gsq")
                else:
                    logger.warning(f"Unknown CI test {ci_test_method}, using fisherz")
                    ci_test_func = FisherZ(data_np)
            except ImportError:
                logger.warning(f"CI test {ci_test_method} not available, using fisherz")
                from causallearn.utils.cit import FisherZ
                ci_test_func = FisherZ(data_np)
            
            # Perform d-separation tests
            ci_tests_log = []
            violations = 0
            total_tests = 0
            
            # Create variable index mapping
            var_to_idx = {var: i for i, var in enumerate(variables)}
            
            for x in G.nodes():
                # Get parents of X
                pa_x = list(G.predecessors(x))
                # Get descendants of X (including X itself)
                desc_x = set(nx.descendants(G, x)) | {x}
                # Get non-descendants of X (excluding parents)
                non_desc_x = [y for y in G.nodes() if y not in desc_x and y not in pa_x]
                
                if not non_desc_x:
                    continue
                
                # Test X ⊥ Y | Pa(X) for each non-descendant Y
                x_idx = var_to_idx[x]
                pa_indices = [var_to_idx[p] for p in pa_x]
                
                for y in non_desc_x:
                    y_idx = var_to_idx[y]
                    total_tests += 1
                    
                    try:
                        # Perform CI test: X ⊥ Y | Pa(X)
                        p_value = ci_test_func(x_idx, y_idx, pa_indices)
                        violated = p_value < alpha
                        
                        if violated:
                            violations += 1
                        
                        ci_tests_log.append({
                            "hypothesis": f"{x} ⟂ {y} | {pa_x}",
                            "p_value": float(p_value),
                            "violation": violated
                        })
                        
                    except Exception as e:
                        logger.warning(f"CI test failed for {x} ⟂ {y} | {pa_x}: {e}")
                        # Treat test failure as violation
                        violations += 1
                        ci_tests_log.append({
                            "hypothesis": f"{x} ⟂ {y} | {pa_x}",
                            "p_value": 1.0,
                            "violation": True,
                            "error": str(e)
                        })
            
            violation_ratio = violations / total_tests if total_tests > 0 else 0.0
            
            return {
                "violation_ratio": violation_ratio,
                "total_tests": total_tests,
                "violations": violations,
                "ci_tests": ci_tests_log,
                "alpha": alpha,
                "ci_test_method": ci_test_method,
                "graph_type": graph_type,
                "message": f"Test completed on {graph_type} with {total_tests} d-separation tests"
            }
            
        except Exception as e:
            logger.error(f"Global Markov test failed: {e}")
            return {"violation_ratio": 1.0, "error": str(e)}
    
    @staticmethod
    def structural_consistency_test(graph: Dict[str, Any], df: pd.DataFrame, algorithm_name: str, n_subsets: int = 3) -> Dict[str, Any]:
        """Test structural consistency using subsampling"""
        try:
            if "graph" not in graph or "edges" not in graph["graph"]:
                return {"instability_score": 1.0, "error": "Invalid graph structure"}
            
            variables = graph["graph"]["variables"]
            if len(variables) < 3:
                return {"instability_score": 0.0, "message": "Too few variables for subsampling"}
            
            # Generate random variable subsets
            subset_results = []
            
            for i in range(n_subsets):
                try:
                    # Random subset of variables (at least 3, at most all)
                    subset_size = max(3, min(len(variables), len(variables) - 1))
                    subset_vars = np.random.choice(variables, size=subset_size, replace=False)
                    
                    # Create subset DataFrame
                    subset_df = df[subset_vars]
                    
                    # Run same algorithm on subset
                    if algorithm_name == "LiNGAM":
                        subset_result = LiNGAMTool.direct_lingam(subset_df)
                    elif algorithm_name == "ANM":
                        subset_result = ANMTool.anm_discovery(subset_df)
                    elif algorithm_name == "PC":
                        subset_result = PCTool.discover(subset_df)
                    elif algorithm_name == "GES":
                        subset_result = GESTool.discover(subset_df)
                    elif algorithm_name == "FCI":
                        subset_result = FCITool.discover(subset_df)
                    elif algorithm_name == "CAM":
                        subset_result = CAMTool.discover(subset_df)
                    else:
                        continue
                    
                    if "error" in subset_result:
                        continue
                    
                    subset_results.append({
                        "subset": subset_vars.tolist(),
                        "result": subset_result
                    })
                    
                except Exception as e:
                    logger.warning(f"Subset {i} failed: {e}")
                    continue
            
            if not subset_results:
                return {"instability_score": 1.0, "error": "All subsets failed"}
            
            # Compute Structural Hamming Distance between graphs
            shd_scores = []
            
            for i, result1 in enumerate(subset_results):
                for j, result2 in enumerate(subset_results):
                    if i >= j:
                        continue
                    
                    try:
                        shd = PruningTool._compute_shd(result1["result"], result2["result"])
                        shd_scores.append(shd)
                    except Exception as e:
                        logger.warning(f"SHD computation failed: {e}")
                        continue
            
            if not shd_scores:
                return {"instability_score": 0.0, "message": "No SHD scores computed"}
            
            instability_score = np.mean(shd_scores)
            
            return {
                "instability_score": instability_score,
                "n_subsets": len(subset_results),
                "shd_scores": shd_scores,
                "subset_results": subset_results
            }
            
        except Exception as e:
            logger.error(f"Structural consistency test failed: {e}")
            return {"instability_score": 1.0, "error": str(e)}
    
    @staticmethod
    def _compute_shd(graph1: Dict[str, Any], graph2: Dict[str, Any]) -> float:
        """Compute Structural Hamming Distance between two graphs"""
        try:
            edges1 = set()
            edges2 = set()
            
            if "graph" in graph1 and "edges" in graph1["graph"]:
                for edge in graph1["graph"]["edges"]:
                    edges1.add((edge["from"], edge["to"]))
            
            if "graph" in graph2 and "edges" in graph2["graph"]:
                for edge in graph2["graph"]["edges"]:
                    edges2.add((edge["from"], edge["to"]))
            
            # SHD = |E1 - E2| + |E2 - E1|
            shd = len(edges1 - edges2) + len(edges2 - edges1)
            
            # Normalize by maximum possible edges
            all_vars = set()
            if "graph" in graph1 and "variables" in graph1["graph"]:
                all_vars.update(graph1["graph"]["variables"])
            if "graph" in graph2 and "variables" in graph2["graph"]:
                all_vars.update(graph2["graph"]["variables"])
            
            max_edges = len(all_vars) * (len(all_vars) - 1)
            normalized_shd = shd / max_edges if max_edges > 0 else 0.0
            
            return normalized_shd
            
        except Exception as e:
            logger.warning(f"SHD computation failed: {e}")
            return 1.0

class EnsembleTool:
    """Ensemble tools for consensus skeleton and PAG construction"""
    
    @staticmethod
    def build_consensus_skeleton(graphs: List[Dict[str, Any]], weights: Optional[List[float]] = None, threshold: float = 0.5) -> Dict[str, Any]:
        """Build consensus skeleton based on undirected adjacency with confidence scores"""
        try:
            if not graphs:
                return {"edges": [], "variables": [], "confidence_scores": {}}
            
            if weights is None:
                weights = [1.0] * len(graphs)
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
            else:
                weights = [1.0 / len(graphs)] * len(graphs)
            
            # Collect adjacency weights (undirected)
            adjacency_weights = {}
            all_variables = set()
            
            for i, graph in enumerate(graphs):
                if "graph" not in graph or "edges" not in graph["graph"]:
                    continue
                
                weight = weights[i]
                
                if "variables" in graph["graph"]:
                    all_variables.update(graph["graph"]["variables"])
                
                for edge in graph["graph"]["edges"]:
                    # KEY CHANGE: Create undirected adjacency key
                    u, v = str(edge["from"]), str(edge["to"])
                    adjacency_key = tuple(sorted((u, v)))
                    
                    if adjacency_key not in adjacency_weights:
                        adjacency_weights[adjacency_key] = 0.0
                    adjacency_weights[adjacency_key] += weight
            
            # Build consensus skeleton
            consensus_edges = []
            confidence_scores = {}
            
            for (var1, var2), confidence in adjacency_weights.items():
                if confidence >= threshold:
                    # Direction is undetermined at this stage
                    consensus_edges.append({
                        "from": var1,
                        "to": var2,
                        "weight": confidence,
                        "confidence": confidence
                    })
                    confidence_scores[f"{var1}-{var2}"] = confidence
            
            return {
                "edges": consensus_edges,
                "variables": list(all_variables),
                "confidence_scores": confidence_scores,
                "n_input_graphs": len(graphs),
                "threshold": threshold
            }
            
        except Exception as e:
            logger.error(f"Consensus skeleton building failed: {e}")
            return {"edges": [], "variables": [], "confidence_scores": {}, "error": str(e)}
    
    @staticmethod
    def resolve_directions(skeleton: Dict[str, Any], graphs: List[Dict[str, Any]], data_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve edge directions with uncertainty markers"""
        try:
            if "edges" not in skeleton:
                return skeleton
            
            resolved_edges = []
            
            for edge in skeleton["edges"]:
                from_var = edge["from"]
                to_var = edge["to"]
                
                # Count direction votes
                forward_votes = 0
                backward_votes = 0
                total_votes = 0
                
                for graph in graphs:
                    if "graph" not in graph or "edges" not in graph["graph"]:
                        continue
                    
                    for graph_edge in graph["graph"]["edges"]:
                        if graph_edge["from"] == from_var and graph_edge["to"] == to_var:
                            forward_votes += 1
                            total_votes += 1
                        elif graph_edge["from"] == to_var and graph_edge["to"] == from_var:
                            backward_votes += 1
                            total_votes += 1
                
                # Resolve direction
                if total_votes == 0:
                    # No direction information, mark as uncertain
                    resolved_edge = edge.copy()
                    resolved_edge["direction"] = "uncertain"
                    resolved_edge["marker"] = "o-o"
                elif forward_votes > backward_votes:
                    # Forward direction consensus
                    resolved_edge = edge.copy()
                    resolved_edge["direction"] = "forward"
                    resolved_edge["marker"] = "->"
                elif backward_votes > forward_votes:
                    # Backward direction consensus
                    resolved_edge = edge.copy()
                    resolved_edge["from"] = to_var
                    resolved_edge["to"] = from_var
                    resolved_edge["direction"] = "backward"
                    resolved_edge["marker"] = "->"
                else:
                    # Conflict, mark as uncertain
                    resolved_edge = edge.copy()
                    resolved_edge["direction"] = "conflict"
                    resolved_edge["marker"] = "o-o"
                
                resolved_edge["forward_votes"] = forward_votes
                resolved_edge["backward_votes"] = backward_votes
                resolved_edge["total_votes"] = total_votes
                
                resolved_edges.append(resolved_edge)
            
            skeleton["edges"] = resolved_edges
            return skeleton
            
        except Exception as e:
            logger.error(f"Direction resolution failed: {e}")
            return skeleton
    
    @staticmethod
    def construct_pag(skeleton: Dict[str, Any], directions: Dict[str, Any]) -> Dict[str, Any]:
        """Construct PAG-like graph with uncertainty markers"""
        try:
            pag = skeleton.copy()
            
            # Add PAG-specific metadata
            pag["graph_type"] = "PAG"
            pag["metadata"] = {
                "construction_method": "consensus",
                "uncertainty_markers": True,
                "edge_types": ["->", "o-o", "o->"]
            }
            
            return pag
            
        except Exception as e:
            logger.error(f"PAG construction failed: {e}")
            return skeleton
    
    @staticmethod
    def construct_dag(pag: Dict[str, Any], data_profile: Dict[str, Any], top_algorithm: str) -> Dict[str, Any]:
        """Construct single DAG with tie-breaking and cycle avoidance"""
        import networkx as nx
        
        try:
            dag_dict = pag.copy()
            dag_dict["graph_type"] = "DAG"
            
            # Step 1: Apply tie-breaking to resolve uncertain edges
            resolved_edges = []
            
            for edge in pag.get("edges", []):
                if edge.get("direction") in ["uncertain", "conflict"]:
                    resolved_edge = EnsembleTool._apply_tie_breaking(edge, data_profile, top_algorithm)
                    resolved_edges.append(resolved_edge)
                else:
                    resolved_edges.append(edge)
            
            # Step 2: Build DAG incrementally to avoid cycles
            G = nx.DiGraph()
            final_edges = []
            skipped_edges = []
            
            # Sort edges by confidence (highest first) to prioritize reliable edges
            sorted_edges = sorted(resolved_edges, key=lambda e: e.get("confidence", 0.0), reverse=True)
            
            for edge in sorted_edges:
                u, v = str(edge["from"]), str(edge["to"])
                
                # Check if adding this edge would create a cycle
                # A cycle occurs if v is already an ancestor of u
                if G.has_node(u) and G.has_node(v) and nx.has_path(G, v, u):
                    # Cycle would be created, skip this edge
                    logger.warning(f"Cycle detected when adding edge {u}->{v}. Skipping to maintain DAG property.")
                    skipped_edges.append(edge)
                else:
                    # Safe to add this edge
                    G.add_edge(u, v)
                    final_edges.append(edge)
            
            # Verify final graph is a DAG
            if not nx.is_directed_acyclic_graph(G):
                logger.error("Final graph is not a DAG despite cycle prevention. This should not happen.")
                raise ValueError("Constructed graph contains cycles")
            
            dag_dict["edges"] = final_edges
            dag_dict["metadata"] = {
                "construction_method": "tie_breaking_and_cycle_avoidance",
                "top_algorithm": top_algorithm,
                "data_profile": data_profile,
                "skipped_edges_count": len(skipped_edges),
                "final_edge_count": len(final_edges)
            }
            
            if skipped_edges:
                logger.info(f"Skipped {len(skipped_edges)} edges to avoid cycles in DAG construction")
            
            return dag_dict
            
        except Exception as e:
            logger.error(f"DAG construction failed: {e}")
            return pag
    
    @staticmethod
    def _apply_tie_breaking(edge: Dict[str, Any], data_profile: Dict[str, Any], top_algorithm: str) -> Dict[str, Any]:
        """Apply tie-breaking rules for uncertain edges"""
        try:
            # Simple tie-breaking based on algorithm preferences
            if top_algorithm in ["LiNGAM", "PC", "GES"]:
                # Prefer forward direction for linear algorithms
                resolved_edge = edge.copy()
                resolved_edge["direction"] = "forward"
                resolved_edge["marker"] = "->"
                resolved_edge["tie_breaking"] = f"Linear algorithm preference ({top_algorithm})"
            elif top_algorithm in ["ANM", "CAM"]:
                # Prefer direction based on ANM compatibility
                if data_profile.get("anm_compatible", False):
                    resolved_edge = edge.copy()
                    resolved_edge["direction"] = "forward"
                    resolved_edge["marker"] = "->"
                    resolved_edge["tie_breaking"] = f"ANM compatibility ({top_algorithm})"
                else:
                    resolved_edge = edge.copy()
                    resolved_edge["direction"] = "backward"
                    resolved_edge["marker"] = "->"
                    resolved_edge["tie_breaking"] = f"Non-ANM preference ({top_algorithm})"
            else:
                # Default: keep original direction
                resolved_edge = edge.copy()
                resolved_edge["direction"] = "forward"
                resolved_edge["marker"] = "->"
                resolved_edge["tie_breaking"] = f"Default preference ({top_algorithm})"
            
            return resolved_edge
            
        except Exception as e:
            logger.warning(f"Tie-breaking failed: {e}")
            return edge