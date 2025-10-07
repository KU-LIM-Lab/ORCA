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
        """Test ANM assumption via residual independence using causal-learn KCI/HSIC.
        Returns anm_score in [0,1] where higher is more consistent with ANM X->Y.
        """
        try:
            import numpy as np
            from sklearn.ensemble import RandomForestRegressor
            # Fit nonparametric regression y ~ f(x)
            X = x.values.reshape(-1, 1)
            Y = y.values.reshape(-1, 1)
            rf = RandomForestRegressor(random_state=42, n_estimators=200, max_depth=None)
            rf.fit(X, Y.ravel())
            resid = Y.ravel() - rf.predict(X)
            # Independence test between x and residual
            pval = None
            try:
                # Try HSIC gamma approximation
                from causallearn.utils.cit import hsic_test_gamma
                pval = float(hsic_test_gamma(X.ravel(), resid))
            except Exception:
                try:
                    # Fallback: KCI (may require bandwidth selection)
                    from causallearn.utils.cit import kci
                    pval = float(kci(X.reshape(-1,1), resid.reshape(-1,1)))
                except Exception:
                    pass
            if pval is None:
                # Final fallback: absolute Spearman correlation
                rho = abs(pd.Series(resid).corr(pd.Series(X.ravel()), method='spearman'))
                score = max(0.0, min(1.0, 1.0 - rho))
                return {"anm_score": score, "fallback": True}
            # High p-value ⇒ fail to reject independence ⇒ ANM satisfied
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
    def anm_discovery(df: pd.DataFrame) -> Dict[str, Any]:
        import time
        from sklearn.ensemble import RandomForestRegressor
        t0 = time.time()
        vars_ = list(df.columns)
        edges = []
        delta = 0.05
        tau = 0.10
        for i, xi in enumerate(vars_):
            for j, yj in enumerate(vars_):
                if i == j:
                    continue
                X = df[xi].values.reshape(-1,1)
                Y = df[yj].values.reshape(-1,1)
                # X->Y
                p_xy = None
                try:
                    rf = RandomForestRegressor(random_state=42, n_estimators=200)
                    rf.fit(X, Y.ravel())
                    resid_xy = Y.ravel() - rf.predict(X)
                    try:
                        from causallearn.utils.cit import hsic_test_gamma
                        p_xy = float(hsic_test_gamma(X.ravel(), resid_xy))
                    except Exception:
                        from causallearn.utils.cit import kci
                        p_xy = float(kci(X, resid_xy.reshape(-1,1)))
                except Exception:
                    p_xy = None
                # Y->X
                p_yx = None
                try:
                    rf = RandomForestRegressor(random_state=42, n_estimators=200)
                    rf.fit(Y, X.ravel())
                    resid_yx = X.ravel() - rf.predict(Y)
                    try:
                        from causallearn.utils.cit import hsic_test_gamma
                        p_yx = float(hsic_test_gamma(Y.ravel(), resid_yx))
                    except Exception:
                        from causallearn.utils.cit import kci
                        p_yx = float(kci(Y, resid_yx.reshape(-1,1)))
                except Exception:
                    p_yx = None
                if p_xy is None and p_yx is None:
                    continue
                if p_xy is None: p_xy = 0.0
                if p_yx is None: p_yx = 0.0
                best_p = max(p_xy, p_yx)
                if best_p < tau:
                    continue
                if (p_xy - p_yx) >= delta:
                    edges.append({"from": xi, "to": yj, "weight": float(best_p)})
                elif (p_yx - p_xy) >= delta:
                    edges.append({"from": yj, "to": xi, "weight": float(best_p)})
        runtime = time.time() - t0
        return normalize_graph_result("ANM", vars_, edges, {"delta": delta, "tau": tau, "backend": "causal-learn"}, runtime)





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


    # Dataset-level EqVar scoring method
    @staticmethod
    def score_eqvar_assumptions(
        df: pd.DataFrame,
        pair_agg: str = "mean",
        transform: Optional[str] = None,
        a: float = 50.0,
        tau: float = 0.05
    ) -> Dict[str, Any]:
        import numpy as np, math
        def _t(x):
            if transform == "softlogistic":
                return 1.0 / (1.0 + math.exp(-a * (x - tau)))
            return x
        vars_ = list(df.columns)
        lin_vals, gau_vals, eqv_vals = [], [], []
        for i, vi in enumerate(vars_):
            for j, vj in enumerate(vars_):
                if i >= j:
                    continue
                lin = StatsTool.linearity_test(df[vi], df[vj]).get("linearity_score", 0.5)
                ge = StatsTool.gaussian_eqvar_test(df[vi], df[vj])
                gau = ge.get("gaussian_score", 0.5)
                eqv = ge.get("eqvar_score", 0.5)
                lin_vals.append(_t(lin)); gau_vals.append(_t(gau)); eqv_vals.append(_t(eqv))
        agg = np.median if pair_agg == "median" else np.mean
        S_lin = float(agg(lin_vals)) if lin_vals else 0.5
        S_Gauss = float(agg(gau_vals)) if gau_vals else 0.5
        S_EqVar = float(agg(eqv_vals)) if eqv_vals else 0.5
        def smry(vals):
            if not vals: return {"mean":0.5,"median":0.5,"q25":0.5,"q75":0.5}
            q25, med, q75 = np.percentile(vals, [25,50,75])
            return {"mean": float(np.mean(vals)), "median": float(med), "q25": float(q25), "q75": float(q75), "n_pairs": len(vals)}
        return {"S_lin": S_lin, "S_Gauss": S_Gauss, "S_EqVar": S_EqVar,
                "summary": {"S_lin": smry(lin_vals), "S_Gauss": smry(gau_vals), "S_EqVar": smry(eqv_vals),
                             "transform": transform, "pair_agg": pair_agg}}

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
    def discover(df: pd.DataFrame, alpha: float = 0.05, **kwargs) -> Dict[str, Any]:
        import time
        t0 = time.time()
        vars_ = list(df.columns)
        edges = []
        params = {"alpha": alpha}
        try:
            from causallearn.search.ConstraintBased.PC import pc
            from causallearn.utils.cit import fisherz
            data = df.values
            cg = pc(data, alpha=alpha, indep_test=fisherz, verbose=False)
            for (i, j), v in cg.G.graph.items():
                if v != 0:
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
            res = ges(data, score_func=score_func)
            G = getattr(res, 'G', None) or (res.get('G', None) if isinstance(res, dict) else None)
            if G is not None and hasattr(G, 'graph'):
                for (i, j), v in G.graph.items():
                    if v != 0:
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
    def discover(df: pd.DataFrame, alpha: float = 0.05, **kwargs) -> Dict[str, Any]:
        import time
        t0 = time.time()
        vars_ = list(df.columns)
        edges = []
        try:
            from causallearn.search.ConstraintBased.FCI import fci
            from causallearn.utils.cit import fisherz
            data = df.values
            # Causal-Learn FCI API: fci(data, indep_test, alpha)
            pag = fci(data, fisherz, alpha)
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
                params={"alpha": alpha, "graph_type": "PAG"},
                runtime=runtime,
            )
            # Also set a top-level hint to consumers
            result["metadata"]["graph_type"] = "PAG"
            return result
        except Exception as e:
            return {"error": str(e)}