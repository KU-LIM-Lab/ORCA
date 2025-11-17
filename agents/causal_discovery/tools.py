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
            conf = e.get("confidence", None)
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
    
    @staticmethod
    def global_normality_test(df: pd.DataFrame) -> float:
        """
        Global multivariate normality test using Henze–Zirkler test
        via pingouin.multivariate_normality.

        Returns:
            p-value (float): High = normal, Low = non-normal
        """
        try:
            import pingouin as pg

            data = df.dropna().values
            if data.shape[0] < 5 or data.shape[1] < 2:
                return 0.0

            # Pingouin HZ test
            hz_res = pg.multivariate_normality(data, alpha=0.05)
            # hz_res = HZResults(hz=..., pval=..., normal=...)

            return float(hz_res.pval)

        except Exception as e:
            logger.warning(f"Global normality test (pingouin) failed: {e}")
            return 0.0
    
    @staticmethod
    def global_linearity_test(df: pd.DataFrame) -> float:
        """Global linearity test using Ramsey RESET via statsmodels.

        Tests H0: Linear specification is adequate for multiple regressions
        via statsmodels.stats.diagnostic.linear_reset.

        It then aggregates the resulting p-values into a single global p-value using a Holm step-down adjustment (less conservative than simple Bonferroni).

        Returns:
            p-value in [0, 1]; high values suggest no strong evidence against linearity.
        """
        try:
            import statsmodels.api as sm
            from statsmodels.stats.diagnostic import linear_reset

            data = df.dropna()
            if data.shape[0] < 10 or data.shape[1] < 2:
                # Not enough data to run a sensible global test
                return 0.5

            pvalues = []

            # Test each variable as dependent variable
            cols = list(data.columns)
            max_vars = len(cols) # can limit max_vars if needed

            for i in range(max_vars):
                y_col = cols[i]
                y = data[y_col].values
                X = data.drop(columns=[y_col]).values

                if X.ndim == 1:
                    X = X.reshape(-1, 1)

                # Add constant term
                X_const = sm.add_constant(X, has_constant="add")

                # Fit OLS model
                model = sm.OLS(y, X_const)
                res = model.fit()

                # Ramsey RESET test
                reset_res = linear_reset(res, power=3, test_type="fitted", use_f=True)

                pval = reset_res.pvalue
                if hasattr(pval, "__len__") and not np.isscalar(pval):
                    pval = float(np.min(pval))
                else:
                    pval = float(pval)

                pvalues.append(pval)

            if not pvalues:
                logger.warning("Global linearity test produced no valid p-values.")
                return np.nan

            # --- Holm step-down 기반 global p-value 집계 ---
            pvals = np.asarray(pvalues, dtype=float)
            m = pvals.size

            order = np.argsort(pvals)
            sorted_p = pvals[order]

            # Holm 조정 p-value 계산: p_(i) * (m - i + 1)
            holm_adj = np.empty_like(sorted_p)
            for i, p in enumerate(sorted_p):
                k = m - i  # remaining hypotheses
                holm_adj[i] = min(1.0, p * k)

            for i in range(1, m):
                if holm_adj[i] < holm_adj[i - 1]:
                    holm_adj[i] = holm_adj[i - 1]

            # Global p-value is the minimum of the Holm-adjusted p-values
            global_p = float(np.min(holm_adj))

            return global_p

        except Exception as e:
            logger.warning(f"Global linearity test (statsmodels RESET) failed: {e}")
            return 0.0

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

class LRTTest:
    """Likelihood-Ratio Test for conditional independence with mixed data support"""
    def __init__(self, data: np.ndarray, variable_schema: Dict[str, Any] = None):
        self.data = data
        self.variable_schema = variable_schema or {}
        self.vars = variable_schema.get("variables", {}) if variable_schema else {}
        # Convert data to DataFrame for easier dummy encoding
        var_names = list(self.vars.keys()) if self.vars else [f"var_{i}" for i in range(self.data.shape[1])]
        self.df = pd.DataFrame(self.data, columns=var_names[:self.data.shape[1]])
    
    def _dummy_encode_categorical(self, var_idx: int, data: np.ndarray) -> Tuple[np.ndarray, int]:
        """Convert categorical variable to dummy variables
        
        Returns:
            encoded_data: Dummy-encoded array (n_samples, n_dummies)
            n_dummies: Number of dummy variables created (for df calculation)
        """
        if var_idx >= len(self.vars):
            # No schema info, treat as continuous
            return data.reshape(-1, 1), 1
        
        var_name = list(self.vars.keys())[var_idx] if self.vars else f"var_{var_idx}"
        var_info = self.vars.get(var_name, {})
        var_type = var_info.get("data_type", "Continuous")
        
        if var_type in ["Nominal", "Binary", "Ordinal"]:
            # Convert to categorical and create dummy variables
            var_series = pd.Series(data, name=var_name)
            # Get unique values (excluding NaN)
            unique_vals = var_series.dropna().unique()
            n_levels = len(unique_vals)
            
            if n_levels <= 1:
                # Only one level, return single column
                return np.ones((len(data), 1)), 1
            
            # Create dummy variables (drop_first=True to avoid multicollinearity)
            dummies = pd.get_dummies(var_series, drop_first=True, dummy_na=False)
            # Fill NaN with 0 (reference category)
            dummies = dummies.fillna(0)
            n_dummies = dummies.shape[1]
            return dummies.values, n_dummies
        else:
            # Continuous variable, return as-is
            return data.reshape(-1, 1), 1
    
    def _prepare_features(self, x_idx: int, z_indices: List[int], 
                         x_data: np.ndarray, z_data: np.ndarray = None) -> Tuple[np.ndarray, int]:
        """Prepare feature matrix with dummy encoding for categorical variables
        
        Returns:
            features: Feature matrix (n_samples, n_features)
            df_diff: Difference in degrees of freedom between full and reduced model
        """
        # Encode X
        x_encoded, x_df = self._dummy_encode_categorical(x_idx, x_data)
        
        # Encode Z variables
        z_encoded_list = []
        z_df_total = 0
        if z_data is not None and len(z_data.shape) > 1 and z_data.shape[1] > 0:
            for i, z_idx in enumerate(z_indices):
                z_col = z_data[:, i] if z_data.shape[1] > i else z_data.flatten()
                z_enc, z_df = self._dummy_encode_categorical(z_idx, z_col)
                z_encoded_list.append(z_enc)
                z_df_total += z_df
        
        # Combine features
        if z_encoded_list:
            features = np.column_stack([x_encoded] + z_encoded_list)
        else:
            features = x_encoded
        
        # df_diff is the number of parameters added by X (after dummy encoding)
        df_diff = x_df
        
        return features, df_diff
    
    def __call__(self, x: int, y: int, z: List[int] = None) -> float:
        """Perform LRT test: X ⟂ Y | Z"""
        if z is None:
            z = []
        
        try:
            # Get variable names
            var_names = list(self.vars.keys()) if self.vars else [f"var_{i}" for i in range(self.data.shape[1])]
            if x >= len(var_names) or y >= len(var_names):
                return 1.0  # Default: independent
            
            x_var = var_names[x]
            y_var = var_names[y]
            
            # Get variable types from schema
            x_type = self.vars.get(x_var, {}).get("data_type", "Continuous") if self.vars else "Continuous"
            y_type = self.vars.get(y_var, {}).get("data_type", "Continuous") if self.vars else "Continuous"
            
            # Extract data
            X = self.data[:, x]
            Y = self.data[:, y]
            Z = self.data[:, z] if z else None
            
            # Remove NaN
            valid_idx = ~(np.isnan(X) | np.isnan(Y))
            if Z is not None:
                valid_idx = valid_idx & ~np.isnan(Z).any(axis=1)
            X_clean = X[valid_idx]
            Y_clean = Y[valid_idx]
            Z_clean = Z[valid_idx] if Z is not None else None
            
            if len(X_clean) < 3:
                return 1.0
            
            from scipy.stats import chi2
            
            # Prepare features with dummy encoding
            XZ_features, df_diff = self._prepare_features(x, z, X_clean, Z_clean)
            has_z = Z_clean is not None and len(Z_clean.shape) > 1 and Z_clean.shape[1] > 0
            
            # Prepare Z features for reduced model
            if has_z:
                Z_features_list = []
                for i, z_idx in enumerate(z):
                    z_col = Z_clean[:, i] if Z_clean.shape[1] > i else Z_clean.flatten()
                    z_enc, _ = self._dummy_encode_categorical(z_idx, z_col)
                    Z_features_list.append(z_enc)
                Z_features = np.column_stack(Z_features_list) if Z_features_list else None
            else:
                Z_features = None
            
            # Perform LRT based on Y type
            if y_type == "Continuous":
                # Linear regression
                from sklearn.linear_model import LinearRegression
                
                # Full model: Y ~ X + Z
                lr_full = LinearRegression()
                lr_full.fit(XZ_features, Y_clean)
                y_pred_full = lr_full.predict(XZ_features)
                ssr_full = np.sum((Y_clean - y_pred_full) ** 2)
                
                # Reduced model: Y ~ Z (or intercept only)
                if Z_features is not None:
                    lr_reduced = LinearRegression()
                    lr_reduced.fit(Z_features, Y_clean)
                    y_pred_reduced = lr_reduced.predict(Z_features)
                    ssr_reduced = np.sum((Y_clean - y_pred_reduced) ** 2)
                else:
                    ssr_reduced = np.sum((Y_clean - np.mean(Y_clean)) ** 2)
                
                # LRT statistic
                if ssr_full > 0:
                    lr_stat = len(Y_clean) * np.log(ssr_reduced / ssr_full)
                    p_value = 1.0 - chi2.cdf(max(0, lr_stat), df=df_diff)
                else:
                    p_value = 1.0
            
            elif y_type == "Binary":
                # Binary logistic regression
                from sklearn.linear_model import LogisticRegression
                
                # Full model: Y ~ X + Z
                lr_full = LogisticRegression(max_iter=1000, random_state=42)
                lr_full.fit(XZ_features, Y_clean)
                
                # Reduced model: Y ~ Z (or intercept only)
                if Z_features is not None:
                    lr_reduced = LogisticRegression(max_iter=1000, random_state=42)
                    lr_reduced.fit(Z_features, Y_clean)
                    prob_reduced = lr_reduced.predict_proba(Z_features)
                else:
                    lr_reduced = LogisticRegression(max_iter=1000, random_state=42)
                    lr_reduced.fit(np.ones((len(Y_clean), 1)), Y_clean)
                    prob_reduced = lr_reduced.predict_proba(np.ones((len(Y_clean), 1)))
                
                # Calculate log-likelihoods
                prob_full = lr_full.predict_proba(XZ_features)
                y_int = Y_clean.astype(int)
                ll_full = np.sum(np.log(prob_full[np.arange(len(Y_clean)), y_int] + 1e-10))
                ll_reduced = np.sum(np.log(prob_reduced[np.arange(len(Y_clean)), y_int] + 1e-10))
                
                # LRT statistic
                lr_stat = -2 * (ll_reduced - ll_full)
                p_value = 1.0 - chi2.cdf(max(0, lr_stat), df=df_diff)
            
            elif y_type == "Nominal":
                # Multinomial logistic regression
                from sklearn.linear_model import LogisticRegression
                
                # Full model: Y ~ X + Z
                lr_full = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42, solver='lbfgs')
                lr_full.fit(XZ_features, Y_clean)
                
                # Reduced model: Y ~ Z (or intercept only)
                if Z_features is not None:
                    lr_reduced = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42, solver='lbfgs')
                    lr_reduced.fit(Z_features, Y_clean)
                    prob_reduced = lr_reduced.predict_proba(Z_features)
                else:
                    lr_reduced = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42, solver='lbfgs')
                    lr_reduced.fit(np.ones((len(Y_clean), 1)), Y_clean)
                    prob_reduced = lr_reduced.predict_proba(np.ones((len(Y_clean), 1)))
                
                # Calculate log-likelihoods
                prob_full = lr_full.predict_proba(XZ_features)
                y_int = Y_clean.astype(int)
                ll_full = np.sum(np.log(prob_full[np.arange(len(Y_clean)), y_int] + 1e-10))
                ll_reduced = np.sum(np.log(prob_reduced[np.arange(len(Y_clean)), y_int] + 1e-10))
                
                # LRT statistic
                lr_stat = -2 * (ll_reduced - ll_full)
                p_value = 1.0 - chi2.cdf(max(0, lr_stat), df=df_diff)
            
            elif y_type == "Ordinal":
                # Ordinal logistic regression (proportional odds model)
                try:
                    import mord
                    # Full model: Y ~ X + Z
                    lr_full = mord.LogisticAT()
                    lr_full.fit(XZ_features, Y_clean)
                    
                    # Reduced model: Y ~ Z (or intercept only)
                    if Z_features is not None:
                        lr_reduced = mord.LogisticAT()
                        lr_reduced.fit(Z_features, Y_clean)
                    else:
                        lr_reduced = mord.LogisticAT()
                        lr_reduced.fit(np.ones((len(Y_clean), 1)), Y_clean)
                    
                    # Calculate log-likelihoods (mord doesn't provide predict_proba easily)
                    # Use score as approximation or calculate manually
                    # For now, use a simplified approach
                    ll_full = lr_full.score(XZ_features, Y_clean) * len(Y_clean)
                    ll_reduced = lr_reduced.score(Z_features if Z_features is not None else np.ones((len(Y_clean), 1)), Y_clean) * len(Y_clean)
                    
                    lr_stat = -2 * (ll_reduced - ll_full)
                    p_value = 1.0 - chi2.cdf(max(0, lr_stat), df=df_diff)
                except ImportError:
                    # Fallback: treat as multinomial if mord not available
                    logger.warning("mord library not available for ordinal regression, using multinomial as fallback")
                    from sklearn.linear_model import LogisticRegression
                    lr_full = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42, solver='lbfgs')
                    lr_full.fit(XZ_features, Y_clean)
                    if Z_features is not None:
                        lr_reduced = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42, solver='lbfgs')
                        lr_reduced.fit(Z_features, Y_clean)
                        prob_reduced = lr_reduced.predict_proba(Z_features)
                    else:
                        lr_reduced = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42, solver='lbfgs')
                        lr_reduced.fit(np.ones((len(Y_clean), 1)), Y_clean)
                        prob_reduced = lr_reduced.predict_proba(np.ones((len(Y_clean), 1)))
                    
                    prob_full = lr_full.predict_proba(XZ_features)
                    y_int = Y_clean.astype(int)
                    ll_full = np.sum(np.log(prob_full[np.arange(len(Y_clean)), y_int] + 1e-10))
                    ll_reduced = np.sum(np.log(prob_reduced[np.arange(len(Y_clean)), y_int] + 1e-10))
                    lr_stat = -2 * (ll_reduced - ll_full)
                    p_value = 1.0 - chi2.cdf(max(0, lr_stat), df=df_diff)
            
            else:
                # Unknown type, fallback
                logger.warning(f"Unknown y_type {y_type}, using correlation-based fallback")
                if Z_clean is None or (len(Z_clean.shape) == 1 and Z_clean.shape[0] == 0):
                    corr, p_val = stats.pearsonr(X_clean, Y_clean)
                    p_value = float(p_val)
                else:
                    p_value = 0.5
            
            # Symmetric test: also test Y ⟂ X | Z (swap roles)
            # For symmetric test, we swap X and Y and test again
            try:
                # Prepare features for Y->X test (swap roles)
                YX_features, df_diff_yx = self._prepare_features(y, z, Y_clean, Z_clean)
                
                if x_type == "Continuous":
                    # Linear regression for Y->X
                    from sklearn.linear_model import LinearRegression
                    
                    # Full model: X ~ Y + Z
                    lr_full_yx = LinearRegression()
                    lr_full_yx.fit(YX_features, X_clean)
                    x_pred_full = lr_full_yx.predict(YX_features)
                    ssr_full_yx = np.sum((X_clean - x_pred_full) ** 2)
                    
                    # Reduced model: X ~ Z (or intercept only)
                    if Z_features is not None:
                        lr_reduced_yx = LinearRegression()
                        lr_reduced_yx.fit(Z_features, X_clean)
                        x_pred_reduced = lr_reduced_yx.predict(Z_features)
                        ssr_reduced_yx = np.sum((X_clean - x_pred_reduced) ** 2)
                    else:
                        ssr_reduced_yx = np.sum((X_clean - np.mean(X_clean)) ** 2)
                    
                    if ssr_full_yx > 0:
                        lr_stat_yx = len(X_clean) * np.log(ssr_reduced_yx / ssr_full_yx)
                        p_value_yx = 1.0 - chi2.cdf(max(0, lr_stat_yx), df=df_diff_yx)
                    else:
                        p_value_yx = 1.0
                elif x_type == "Binary":
                    # Binary logistic regression for Y->X
                    from sklearn.linear_model import LogisticRegression
                    
                    # Full model: X ~ Y + Z
                    lr_full_yx = LogisticRegression(max_iter=1000, random_state=42)
                    lr_full_yx.fit(YX_features, X_clean)
                    
                    # Reduced model: X ~ Z (or intercept only)
                    if Z_features is not None:
                        lr_reduced_yx = LogisticRegression(max_iter=1000, random_state=42)
                        lr_reduced_yx.fit(Z_features, X_clean)
                        prob_reduced_yx = lr_reduced_yx.predict_proba(Z_features)
                    else:
                        lr_reduced_yx = LogisticRegression(max_iter=1000, random_state=42)
                        lr_reduced_yx.fit(np.ones((len(X_clean), 1)), X_clean)
                        prob_reduced_yx = lr_reduced_yx.predict_proba(np.ones((len(X_clean), 1)))
                    
                    prob_full_yx = lr_full_yx.predict_proba(YX_features)
                    x_int = X_clean.astype(int)
                    ll_full_yx = np.sum(np.log(prob_full_yx[np.arange(len(X_clean)), x_int] + 1e-10))
                    ll_reduced_yx = np.sum(np.log(prob_reduced_yx[np.arange(len(X_clean)), x_int] + 1e-10))
                    lr_stat_yx = -2 * (ll_reduced_yx - ll_full_yx)
                    p_value_yx = 1.0 - chi2.cdf(max(0, lr_stat_yx), df=df_diff_yx)
                else:
                    # For non-continuous/non-binary X, use same p-value
                    p_value_yx = p_value
            except Exception as e:
                logger.warning(f"Symmetric test failed: {e}")
                p_value_yx = p_value
            
            # Combine p-values (max for conservative test)
            combined_p = max(float(p_value), float(p_value_yx))
            return combined_p
            
        except Exception as e:
            logger.warning(f"LRT test failed: {e}")
            return 1.0  # Default: independent

class PCTool:
    """PC algorithm wrapper. Uses causal-learn with support for LRT and other CI tests."""
    @staticmethod
    def discover(df: pd.DataFrame, alpha: float = 0.05, indep_test: str = "fisherz", 
                 variable_schema: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        import time
        t0 = time.time()
        vars_ = list(df.columns)
        edges = []
        params = {"alpha": alpha, "indep_test": indep_test}
        if variable_schema:
            params["variable_schema"] = variable_schema
        try:
            from causallearn.search.ConstraintBased.PC import pc
            from causallearn.utils.cit import fisherz, kci, gsq
            data = df.values.astype(float)
            
            # Convert string to actual test function
            if indep_test == "lrt":
                # Use LRT test for mixed data
                test_func = LRTTest(data, variable_schema)
            elif indep_test == "cg":
                # PLACEHOLDER: Conditional Gaussian test
                logger.warning("CG test not yet implemented, using fisherz")
                test_func = fisherz
            elif indep_test == "pillai":
                # PLACEHOLDER: Pillai test
                logger.warning("Pillai test not yet implemented, using fisherz")
                test_func = fisherz
            elif indep_test == "kernel_kcit":
                # Use KCI test (kernel-based)
                test_func = kci
            elif indep_test == "cmi":
                # PLACEHOLDER: Conditional Mutual Information
                logger.warning("CMI test not yet implemented, using kci")
                test_func = kci
            else:
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
                causallearn_score = "local_score_BIC"
            elif score_func == "bic-cg":
                # PLACEHOLDER: BIC for Conditional Gaussian (mixed data)
                logger.warning("bic-cg score not yet implemented, using standard BIC")
                causallearn_score = "local_score_BIC"
            elif score_func == "generalized_rkhs":
                # PLACEHOLDER: Generalized RKHS regression-based non-parametric scoring
                logger.warning("generalized_rkhs score not yet implemented, using standard BIC")
                causallearn_score = "local_score_BIC"
            else:
                # Default: try to use as-is or fallback to BIC
                causallearn_score = score_func if hasattr(ges, score_func) else "local_score_BIC"
            
            res = ges(data, score_func=causallearn_score)
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
class LiMTool:
    """LiM (Linear Mixed) algorithm for mixed data functional model"""
    
    @staticmethod
    def _build_dis_con(variables: List[str], variable_schema: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Build dis_con vector for LiM algorithm
        
        Args:
            variables: List of variable names
            variable_schema: Variable schema dictionary
            
        Returns:
            dis_con: (1, n_features) array where 1=Continuous, 0=Discrete
        """
        n_features = len(variables)
        dis_con = np.ones((1, n_features), dtype=int)
        
        if variable_schema:
            schema_vars = variable_schema.get("variables", {})
            for j, name in enumerate(variables):
                meta = schema_vars.get(name, {})
                dtype = meta.get("data_type")
                if dtype in ["Binary", "Nominal", "Ordinal"]:
                    dis_con[0, j] = 0  # Discrete
                elif dtype == "Continuous":
                    dis_con[0, j] = 1  # Continuous
                else:
                    # Unknown type defaults to discrete
                    dis_con[0, j] = 0
                    logger.warning(f"Unknown data_type '{dtype}' for variable '{name}', treating as discrete")
        
        return dis_con
    
    @staticmethod
    def discover(df: pd.DataFrame, variable_schema: Dict[str, Any] = None, algo_config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """LiM algorithm implementation using lingam library
        
        Args:
            df: DataFrame with mixed data (continuous and categorical)
            variable_schema: Variable schema dictionary with data types
            algo_config: Optional algorithm configuration dictionary
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with graph result in ORCA standard format
        """
        import time
        t0 = time.time()
        
        try:
            import lingam
            
            # 1. 입력 데이터 정리 (NaN 제거)
            df_li = df.dropna()
            if df_li.empty:
                raise ValueError("DataFrame is empty after dropping NaN values")
            
            variables = list(df_li.columns)
            X = df_li.values.astype(float)
            
            # 2. dis_con 벡터 생성
            dis_con = LiMTool._build_dis_con(variables, variable_schema)
            
            # 3. LiM 모델 생성 및 학습
            lim_params = {}
            if algo_config and "LiM" in algo_config:
                lim_params = algo_config["LiM"]
            
            lim_params.update(kwargs)
            
            model = lingam.LiM(**lim_params)
            model.fit(X, dis_con)
            
            # 4. Adjacency matrix에서 edge list 변환
            edges = []
            A = getattr(model, 'adjacency_matrix_', None)
            if A is None:
                A = getattr(model, 'adjacency_matrix', None)
            
            if A is not None:
                A = np.asarray(A)
                for i in range(A.shape[0]):
                    for j in range(A.shape[1]):
                        if abs(A[i, j]) > 0:
                            edges.append({"from": variables[i], "to": variables[j], "weight": float(A[i, j])})
            else:
                # Fallback: use causal order if available
                order = getattr(model, 'causal_order_', None)
                if order is None:
                    order = getattr(model, 'causal_order', None)
                
                if order is not None:
                    order = np.asarray(order)
                    for i in range(len(order)):
                        for j in range(i + 1, len(order)):
                            edges.append({"from": variables[order[i]], "to": variables[order[j]], "weight": 1.0})
                else:
                    logger.warning("No adjacency matrix or causal order found in LiM model")
            
            runtime = time.time() - t0
            
            params = {"variable_schema": variable_schema, "backend": "lingam"}
            if lim_params:
                params["algo_config"] = lim_params
            
            return normalize_graph_result("LiM", variables, edges, params, runtime)
            
        except ImportError as e:
            logger.error(f"LiM algorithm requires lingam library: {e}")
            return {"error": f"LiM not available: lingam library not installed"}
        except Exception as e:
            logger.error(f"LiM execution failed: {e}")
            return {"error": f"LiM not available: {e}"}

class FCITool:
    """FCI algorithm wrapper. Uses causal-learn and returns a PAG-oriented result.
    Note: PAG edges may be partially directed; we expose adjacencies as edges and
    mark graph_type="PAG" in metadata/params for downstream handling.
    """
    @staticmethod
    def discover(df: pd.DataFrame, alpha: float = 0.05, indep_test: str = "fisherz", 
                 variable_schema: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        import time
        t0 = time.time()
        vars_ = list(df.columns)
        edges = []
        try:
            from causallearn.search.ConstraintBased.FCI import fci
            from causallearn.utils.cit import fisherz, kci, gsq
            data = df.values.astype(float)
            
            # Convert string to actual test function
            if indep_test == "lrt":
                # Use LRT test for mixed data
                test_func = LRTTest(data, variable_schema)
            elif indep_test == "cg":
                # PLACEHOLDER: Conditional Gaussian test
                logger.warning("CG test not yet implemented for FCI, using fisherz")
                test_func = fisherz
            elif indep_test == "pillai":
                # PLACEHOLDER: Pillai test
                logger.warning("Pillai test not yet implemented for FCI, using fisherz")
                test_func = fisherz
            elif indep_test == "kernel_kcit":
                test_func = kci
            elif indep_test == "cmi":
                # PLACEHOLDER: Conditional Mutual Information
                logger.warning("CMI test not yet implemented for FCI, using kci")
                test_func = kci
            else:
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
            params_dict = {"alpha": alpha, "indep_test": indep_test, "graph_type": "PAG"}
            if variable_schema:
                params_dict["variable_schema"] = variable_schema
            result = normalize_graph_result(
                "FCI", vars_, edges,
                params=params_dict,
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
            
            # Compute normalized SHD: compare G|Vi with Gi for each subset
            normalized_shd_scores = []
            
            for subset_info in subset_results:
                try:
                    subset_vars = subset_info["subset"]
                    subset_graph = subset_info["result"]
                    
                    # Restrict original graph to this subset
                    original_restricted = PruningTool._restrict_graph_to_subset(graph, subset_vars)
                    
                    # Compute normalized SHD between G|Vi and Gi
                    normalized_shd = PruningTool._compute_shd(
                        original_restricted, 
                        subset_graph, 
                        normalize=True
                    )
                    
                    normalized_shd_scores.append(normalized_shd)
                    
                except Exception as e:
                    logger.warning(f"SHD computation failed for subset: {e}")
                    continue
            
            if not normalized_shd_scores:
                return {"instability_score": 0.0, "message": "No SHD scores computed"}
            
            # Average normalized SHD is the final instability score [0, 1]
            instability_score = np.mean(normalized_shd_scores)
            
            return {
                "instability_score": instability_score,
                "n_subsets": len(subset_results),
                "normalized_shd_scores": normalized_shd_scores,
                "subset_results": subset_results
            }
            
        except Exception as e:
            logger.error(f"Structural consistency test failed: {e}")
            return {"instability_score": 1.0, "error": str(e)}
    
    @staticmethod
    def _compute_shd(graph1: Dict[str, Any], graph2: Dict[str, Any], normalize: bool = True) -> float:
        """Compute raw or normalized Structural Hamming Distance based on common variables."""
        try:
            vars1 = set(graph1.get("graph", {}).get("variables", []))
            vars2 = set(graph2.get("graph", {}).get("variables", []))
            common_vars = vars1.intersection(vars2)
            
            if not common_vars:
                return 1.0 if normalize else 0.0
            
            # Extract edges within common variables only
            edges1 = {
                (e["from"], e["to"]) for e in graph1.get("graph", {}).get("edges", [])
                if e["from"] in common_vars and e["to"] in common_vars
            }
            edges2 = {
                (e["from"], e["to"]) for e in graph2.get("graph", {}).get("edges", [])
                if e["from"] in common_vars and e["to"] in common_vars
            }
            
            raw_shd = len(edges1.symmetric_difference(edges2))
            
            if not normalize:
                return float(raw_shd)
            
            n = len(common_vars)
            if n < 2:
                return 0.0
            
            max_edges = n * (n - 1)  # For directed graphs
            normalized_shd = raw_shd / max_edges if max_edges > 0 else 0.0
            return normalized_shd
            
        except Exception:
            return 1.0  # Return max instability on failure

    @staticmethod
    def _restrict_graph_to_subset(graph: Dict[str, Any], subset_vars: List[str]) -> Dict[str, Any]:
        """Restrict graph to a subset of variables.
        
        Args:
            graph: Original graph dictionary
            subset_vars: List of variable names to keep
        
        Returns:
            New graph containing only edges between subset_vars
        """
        try:
            subset_set = set(subset_vars)
            
            restricted_edges = []
            if "graph" in graph and "edges" in graph["graph"]:
                for edge in graph["graph"]["edges"]:
                    if edge["from"] in subset_set and edge["to"] in subset_set:
                        restricted_edges.append(edge.copy())
            
            return {
                "graph": {
                    "variables": list(subset_vars),
                    "edges": restricted_edges
                }
            }
        except Exception as e:
            logger.warning(f"Graph restriction failed: {e}")
            return {"graph": {"variables": list(subset_vars), "edges": []}}

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
    def construct_dag(pag: Dict[str, Any], data_profile: Dict[str, Any], top_algorithm: str,
                      execution_plan: List[Dict[str, Any]] = None,
                      algorithm_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """Construct single DAG with tie-breaking and cycle avoidance"""
        import networkx as nx
        
        try:
            dag_dict = pag.copy()
            dag_dict["graph_type"] = "DAG"
            
            # Step 1: Apply tie-breaking to resolve uncertain edges
            resolved_edges = []
            
            for edge in pag.get("edges", []):
                if edge.get("direction") in ["uncertain", "conflict"]:
                    resolved_edge = EnsembleTool._apply_tie_breaking(
                        edge, data_profile, top_algorithm, 
                        execution_plan=execution_plan,
                        algorithm_results=algorithm_results
                    )
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
    def _apply_tie_breaking(edge: Dict[str, Any], data_profile: Dict[str, Any], 
                           top_algorithm: str, execution_plan: List[Dict[str, Any]] = None,
                           algorithm_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """Apply tie-breaking rules for uncertain edges with mixed data support"""
        try:
            from_var = edge.get("from")
            to_var = edge.get("to")
            global_props = data_profile.get("global", {})
            is_mixed = global_props.get("is_mixed", False)
            pairwise = data_profile.get("pairwise", {})
            
            # New Rule 1: Nonlinear Mixed Data - use TSCM direction
            if is_mixed and execution_plan:
                # Check if TSCM is in execution plan
                tscm_config = next((c for c in execution_plan if c["alg"] == "TSCM"), None)
                if tscm_config:
                    # Check pairwise properties for nonlinearity
                    pair_key = f"{from_var}_{to_var}"
                    pair_key_rev = f"{to_var}_{from_var}"
                    
                    # Check cont_cont or cont_cat pairs for nonlinearity
                    cont_cont_pair = pairwise.get("cont_cont", {}).get(pair_key) or pairwise.get("cont_cont", {}).get(pair_key_rev)
                    if cont_cont_pair and cont_cont_pair.get("nonlinearity", 0) > 0.5:
                        # Use TSCM direction if available
                        tscm_result = algorithm_results.get("TSCM", {}) if algorithm_results else {}
                        tscm_direction = EnsembleTool._extract_tscm_direction(tscm_result, from_var, to_var)
                        if tscm_direction:
                            resolved_edge = edge.copy()
                            resolved_edge["from"] = tscm_direction["from"]
                            resolved_edge["to"] = tscm_direction["to"]
                            resolved_edge["direction"] = "forward"
                            resolved_edge["marker"] = "->"
                            resolved_edge["tie_breaking"] = "TSCM direction (nonlinear mixed data)"
                            return resolved_edge
            
            # New Rule 2: Linear Mixed Data - use LiM direction
            if is_mixed and execution_plan:
                # Check if LiM is in execution plan
                lim_config = next((c for c in execution_plan if c["alg"] == "LiM"), None)
                if lim_config:
                    # Check pairwise properties for linearity
                    pair_key = f"{from_var}_{to_var}"
                    pair_key_rev = f"{to_var}_{from_var}"
                    
                    cont_cont_pair = pairwise.get("cont_cont", {}).get(pair_key) or pairwise.get("cont_cont", {}).get(pair_key_rev)
                    if cont_cont_pair and cont_cont_pair.get("linearity", 0) > 0.6:
                        # Use LiM direction if available
                        lim_result = algorithm_results.get("LiM", {}) if algorithm_results else {}
                        lim_direction = EnsembleTool._extract_lim_direction(lim_result, from_var, to_var)
                        if lim_direction:
                            resolved_edge = edge.copy()
                            resolved_edge["from"] = lim_direction["from"]
                            resolved_edge["to"] = lim_direction["to"]
                            resolved_edge["direction"] = "forward"
                            resolved_edge["marker"] = "->"
                            resolved_edge["tie_breaking"] = "LiM direction (linear mixed data)"
                            return resolved_edge
            
            # Fallback: Original rules
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
    
    @staticmethod
    def _extract_tscm_direction(tscm_result: Dict[str, Any], var1: str, var2: str) -> Optional[Dict[str, str]]:
        """Extract direction from TSCM result for a specific edge (placeholder)"""
        # PLACEHOLDER: Extract direction from TSCM result
        # This should parse TSCM output to find edge direction
        if "graph" in tscm_result and "edges" in tscm_result["graph"]:
            edges = tscm_result["graph"]["edges"]
            for e in edges:
                if (e.get("from") == var1 and e.get("to") == var2) or \
                   (e.get("from") == var2 and e.get("to") == var1):
                    return {"from": e["from"], "to": e["to"]}
        return None
    
    @staticmethod
    def _extract_lim_direction(lim_result: Dict[str, Any], var1: str, var2: str) -> Optional[Dict[str, str]]:
        """Extract direction from LiM result for a specific edge (placeholder)"""
        # PLACEHOLDER: Extract direction from LiM result
        # This should parse LiM output to find edge direction
        if "graph" in lim_result and "edges" in lim_result["graph"]:
            edges = lim_result["graph"]["edges"]
            for e in edges:
                if (e.get("from") == var1 and e.get("to") == var2) or \
                   (e.get("from") == var2 and e.get("to") == var1):
                    return {"from": e["from"], "to": e["to"]}
        return None