# === Imports ===
import logging
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, Any, List, Tuple, Optional, Union
from scipy import stats
from sklearn.linear_model import LinearRegression
from pygam import LinearGAM, s, l
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
import warnings

logger = logging.getLogger(__name__)

# === Standardized graph result schema helpers ===
# Edge attribute semantics:
# - weight: algorithm-native strength/evidence in [0,1] if possible (e.g., ANM score, correlation-derived score, BIC-based rescale).
# - confidence: post-hoc reliability from ensembling/bootstrapping (e.g., frequency).
# - type: edge type marker ("->", "--", "o->", "o-o", "<->", etc.) - preserves PAG endpoint information
# Keep both when available; algorithm-specific extras should go under result['metadata'] or per-edge custom keys.
GRAPH_RESULT_TEMPLATE = {
    "graph": {"edges": [], "variables": []},
    "metadata": {"method": None, "params": {}, "runtime": None, "graph_type": None}
}

# === Schema validation and helper functions ===

def validate_graph_schema(graph: Dict[str, Any]) -> bool:
    """Validate that graph follows unified schema structure.
    
    Args:
        graph: Graph dictionary to validate
        
    Returns:
        True if valid, raises ValueError if invalid
        
    Raises:
        ValueError: If graph schema is invalid
    """
    if not isinstance(graph, dict):
        raise ValueError(f"Graph must be a dict, got {type(graph)}")
    
    # Check graph.graph structure
    if "graph" not in graph:
        raise ValueError("Graph missing 'graph' key")
    
    graph_data = graph["graph"]
    if not isinstance(graph_data, dict):
        raise ValueError(f"graph['graph'] must be a dict, got {type(graph_data)}")
    
    # Check variables
    if "variables" not in graph_data:
        raise ValueError("Graph missing 'graph.variables' key")
    if not isinstance(graph_data["variables"], list):
        raise ValueError(f"graph['graph']['variables'] must be a list, got {type(graph_data['variables'])}")
    
    # Check edges
    if "edges" not in graph_data:
        raise ValueError("Graph missing 'graph.edges' key")
    if not isinstance(graph_data["edges"], list):
        raise ValueError(f"graph['graph']['edges'] must be a list, got {type(graph_data['edges'])}")
    
    # Check metadata
    if "metadata" not in graph:
        raise ValueError("Graph missing 'metadata' key")
    
    metadata = graph["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError(f"graph['metadata'] must be a dict, got {type(metadata)}")
    
    # Check graph_type in metadata
    if "graph_type" not in metadata:
        raise ValueError("Graph missing 'metadata.graph_type' key")
    
    graph_type = metadata["graph_type"]
    if graph_type not in ["DAG", "CPDAG", "PAG", "skeleton"]:
        logger.warning(f"Unknown graph_type '{graph_type}', expected DAG|CPDAG|PAG|skeleton")
    
    # Validate edges structure
    for i, edge in enumerate(graph_data["edges"]):
        if not isinstance(edge, dict):
            raise ValueError(f"Edge at index {i} must be a dict, got {type(edge)}")
        
        if "from" not in edge:
            raise ValueError(f"Edge at index {i} missing 'from' field")
        if "to" not in edge:
            raise ValueError(f"Edge at index {i} missing 'to' field")
        if "type" not in edge:
            raise ValueError(f"Edge at index {i} missing 'type' field")
    
    return True

def get_edges(graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "graph" in graph and "edges" in graph["graph"]:
        return graph["graph"]["edges"]
    raise ValueError("Graph does not follow unified schema: missing graph.edges")

def get_variables(graph: Dict[str, Any]) -> List[str]:
    if "graph" in graph and "variables" in graph["graph"]:
        return graph["graph"]["variables"]
    raise ValueError("Graph does not follow unified schema: missing graph.variables")

def get_graph_type(graph: Dict[str, Any]) -> str:
    if "metadata" in graph and "graph_type" in graph["metadata"]:
        return graph["metadata"]["graph_type"]
    raise ValueError("Graph missing 'metadata.graph_type' - schema validation required")

def _normalize_edges(edges, graph_type: Optional[str] = None):
    """Normalize edges to unified schema format.
    
    Args:
        edges: List of edge dictionaries or tuples
        graph_type: Optional graph type (DAG|CPDAG|PAG|skeleton) for default type assignment
        
    Returns:
        List of normalized edge dictionaries with 'from', 'to', 'type' fields
    """
    norm = []
    for e in edges or []:
        if isinstance(e, dict):
            frm, to = e.get("from"), e.get("to")
            w = e.get("weight", None)
            conf = e.get("confidence", None)
            marker = e.get("marker", None)
            edge_type = e.get("type", None)
        else:
            # support tuple (from, to, weight?)
            frm = e[0]; to = e[1]; w = e[2] if len(e) > 2 else None
            conf = None
            marker = None
            edge_type = None
        
        item = {"from": str(frm), "to": str(to)}
        
        # Convert marker to type if type not already present
        if edge_type is None and marker is not None:
            edge_type = marker
        elif edge_type is None:
            # Set default type based on graph_type
            if graph_type == "DAG":
                edge_type = "->"
            elif graph_type == "CPDAG":
                # CPDAG: default to "--" (undirected), can be "->" if direction is determined
                edge_type = "--"
            elif graph_type == "PAG":
                # PAG: default to "o-o" (uncertain) if no marker
                edge_type = "o-o"
            elif graph_type == "skeleton":
                edge_type = "--"
            else:
                # Default fallback
                edge_type = "->"
        
        item["type"] = str(edge_type)
        
        if w is not None:
            item["weight"] = float(w)
        if conf is not None:
            item["confidence"] = float(conf)
        
        # Preserve endpoints information if available
        if isinstance(e, dict):
            if "endpoints" in e:
                item["endpoints"] = e["endpoints"]
        
        norm.append(item)
    return norm

def normalize_graph_result(method: str, variables, edges, params=None, runtime=None, graph_type: Optional[str] = None):
    """Normalize graph result to unified schema.
    
    Args:
        method: Algorithm method name
        variables: List of variable names
        edges: List of edges (dicts or tuples)
        params: Optional parameters dictionary
        runtime: Optional runtime in seconds
        graph_type: Optional graph type (DAG|CPDAG|PAG|skeleton). If None, inferred from method.
        
    Returns:
        Normalized graph dictionary following unified schema
    """
    # Determine graph_type from method if not provided
    if graph_type is None:
        if method == "PC":
            graph_type = "CPDAG"
        elif method == "FCI":
            graph_type = "PAG"
        elif method in ["LiNGAM", "ANM", "GES", "NOTEARS-linear", "NOTEARS-nonlinear", "LiM"]:
            graph_type = "DAG"
        else:
            # Default fallback
            graph_type = "DAG"
    
    # Normalize edges with graph_type for default type assignment
    normalized_edges = _normalize_edges(edges, graph_type=graph_type)
    
    res = {
        "graph": {
            "edges": normalized_edges,
            "variables": list(map(str, variables or []))
        },
        "metadata": {
            "method": method,
            "params": params or {},
            "runtime": runtime,
            "graph_type": graph_type
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
        """
        Nested test for nonlinearity:
        H0: LinearGAM(l(0))  (linear term only)
        H1: LinearGAM(s(0))  (smooth term)
        Return:
        linearity_score = p_value (higher => more linear / no evidence of nonlinearity)
        """
        try:
            df = pd.DataFrame({"x": x.values, "y": y.values}).dropna()
            if len(df) < 10:
                return {"linearity_score": np.nan, "error": "too_few_samples"}

            X = df["x"].values.reshape(-1, 1)
            Y = df["y"].values

            # --- 1) Fit null (linear) model ---
            gam_lin = LinearGAM(l(0), fit_intercept=True, max_iter=200, tol=1e-4, verbose=False)
            gam_lin.fit(X, Y)

            # --- 2) Fit alternative (smooth) model ---
            gam_smooth = LinearGAM(s(0, n_splines=15, spline_order=3), fit_intercept=True,
                                max_iter=200, tol=1e-4, verbose=False)
            lam_grid = np.logspace(-2, 2, 8)
            gam_smooth.gridsearch(X, Y, lam=lam_grid, progress=False)

            yhat_lin = gam_lin.predict(X)
            yhat_smooth = gam_smooth.predict(X)

            rss_lin = float(np.sum((Y - yhat_lin) ** 2))
            rss_smooth = float(np.sum((Y - yhat_smooth) ** 2))

            delta = max(0.0, rss_lin - rss_smooth)

            # edof(=effective DoF) 차이로 자유도 근사
            edof_lin = float(getattr(gam_lin, "statistics_", {}).get("edof", 2.0))   # intercept+linear ≈ 2
            edof_smooth = float(getattr(gam_smooth, "statistics_", {}).get("edof", 4.0))

            df_diff = max(1.0, edof_smooth - edof_lin) 

            p_nonlin = 1.0 - stats.chi2.cdf(delta, df=df_diff)

            alpha = 0.05
            linearity_score = 1.0 if p_nonlin > alpha else 0.0 
            
            mse_lin = float(mean_squared_error(Y, yhat_lin))
            mse_smooth = float(mean_squared_error(Y, yhat_smooth))

            return {
                "linearity_score": linearity_score, 
                "p_nonlin": float(p_nonlin),
                "rss_lin": rss_lin,
                "rss_smooth": rss_smooth,
                "delta_rss": float(delta),
                "edof_lin": edof_lin,
                "edof_smooth": edof_smooth,
                "df_diff": float(df_diff),
                "mse_lin": mse_lin,
                "mse_smooth": mse_smooth,
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

class Bootstrapper:
    """Bootstrap evaluation tools"""
    
    @staticmethod
    def _create_edge_key(edge: Dict[str, Any], graph_type: str) -> Optional[Union[str, Tuple[str, str]]]:
        """Create edge key based on graph type.
        
        Args:
            edge: Edge dictionary with 'from' and 'to' keys
            graph_type: Graph type ('DAG', 'CPDAG', 'PAG', 'skeleton')
            
        Returns:
            For DAG: directed key string like "u->v"
            For CPDAG/PAG/skeleton: undirected adjacency tuple (sorted(u, v))
            Returns None if u == v (self-loop, invalid edge)
        """
        u, v = str(edge["from"]), str(edge["to"])
        
        # Prevent self-loops
        if u == v:
            return None
        
        if graph_type == "DAG":
            return f"{u}->{v}"  # Directed edge key
        else:  # CPDAG, PAG, skeleton
            return tuple(sorted([u, v]))  # Undirected adjacency key
    
    @staticmethod
    def bootstrap_evaluation(df: pd.DataFrame, result: Dict[str, Any], 
                           algorithm: str, n_iterations: int = 100) -> Dict[str, Any]:
        """Bootstrap evaluation for robustness (sampling stability).
        
        Evaluates how consistently edges from the original graph G are reproduced
        in bootstrap graphs. Only counts edges that exist in the original graph.
        
        Args:
            df: DataFrame for bootstrap sampling
            result: Original graph (evaluation reference graph G) - must follow unified schema
            algorithm: Algorithm name to run on bootstrap samples
            n_iterations: Number of bootstrap iterations (default: 100)
            
        Returns:
            Dictionary with:
            - robustness_score: Mean confidence across all original edges (based on successful iterations)
            - effective_score: robustness_score * success_rate (penalized by failure rate)
            - edge_confidences: Dict mapping edge_key -> confidence
            - edge_frequencies: Dict mapping edge_key -> raw count (for debugging)
            - n_original_edges: Number of edges in original graph
            - n_iterations: Total iterations attempted
            - n_success_iterations: Number of successful bootstrap runs
            - success_rate: n_success_iterations / n_iterations
            - failure_iterations: List of iteration indices that failed (for debugging)
        """
        try:
            # Extract original graph information
            try:
                original_edges = get_edges(result)
                graph_type = get_graph_type(result)
            except (ValueError, KeyError) as e:
                logger.warning(f"Original graph schema validation failed: {e}")
                return {
                    "robustness_score": 0.0,
                    "effective_score": 0.0,
                    "edge_confidences": {},
                    "edge_frequencies": {},
                    "n_original_edges": 0,
                    "n_iterations": n_iterations,
                    "n_success_iterations": 0,
                    "success_rate": 0.0,
                    "failure_iterations": [],
                    "error": f"Invalid original graph schema: {e}"
                }
            
            # Handle empty original edges case
            n_original_edges = len(original_edges)
            if n_original_edges == 0:
                logger.warning("Original graph has no edges, cannot calculate sampling stability")
                return {
                    "robustness_score": 0.0,
                    "effective_score": 0.0,
                    "edge_confidences": {},
                    "edge_frequencies": {},
                    "n_original_edges": 0,
                    "n_iterations": n_iterations,
                    "n_success_iterations": n_iterations,  # All iterations "succeeded" (no edges to evaluate)
                    "success_rate": 1.0,  # no failures, just no edges
                    "failure_iterations": []
                }
            
            # Create original edge keys set (reference set for counting)
            original_edge_keys = set()
            for edge in original_edges:
                key = Bootstrapper._create_edge_key(edge, graph_type)
                if key is not None:  # Skip self-loops
                    original_edge_keys.add(key)
            
            # Initialize counts only for original edges
            counts = {key: 0 for key in original_edge_keys}
            n_success = 0
            failures = []
            n_samples = len(df)
            
            # Bootstrap sampling loop
            for i in range(n_iterations):
                try:
                    # Sample with replacement
                    bootstrap_df = df.sample(n=n_samples, replace=True, random_state=i)
                    
                    # Run algorithm on bootstrap sample
                    if algorithm == "LiNGAM":
                        bootstrap_result = LiNGAMTool.direct_lingam(bootstrap_df)
                    elif algorithm == "ANM":
                        bootstrap_result = ANMTool.anm_discovery(bootstrap_df)
                    elif algorithm == "PC":
                        bootstrap_result = PCTool.discover(bootstrap_df)
                    elif algorithm == "GES":
                        bootstrap_result = GESTool.discover(bootstrap_df)
                    elif algorithm == "FCI":
                        bootstrap_result = FCITool.discover(bootstrap_df)
                    elif algorithm == "LiM":
                        # LiM requires variable_schema, but we'll try without it for bootstrap
                        bootstrap_result = LiMTool.discover(bootstrap_df)
                    elif algorithm == "NOTEARS-linear":
                        bootstrap_result = NOTEARSLinearTool.discover(bootstrap_df)
                    elif algorithm == "NOTEARS-nonlinear":
                        bootstrap_result = NOTEARSNonlinearTool.discover(bootstrap_df)
                    else:
                        failures.append(i)
                        logger.debug(f"Unknown algorithm '{algorithm}' in bootstrap iteration {i}")
                        continue
                    
                    # Check for errors in bootstrap result
                    if not isinstance(bootstrap_result, dict):
                        failures.append(i)
                        logger.debug(f"Bootstrap iteration {i} returned non-dict result: {type(bootstrap_result)}")
                        continue
                    
                    if "error" in bootstrap_result:
                        failures.append(i)
                        logger.debug(f"Algorithm error in bootstrap iteration {i}: {bootstrap_result.get('error')}")
                        continue
                    
                    # Normalize bootstrap result to ensure schema consistency
                    try:
                        # Try to use as-is if already normalized
                        bootstrap_edges = get_edges(bootstrap_result)
                        get_graph_type(bootstrap_result)  # Verify metadata exists too
                    except (ValueError, KeyError):
                        variables = None
                        edges = []
                        
                        if "graph" in bootstrap_result:
                            variables = bootstrap_result["graph"].get("variables")
                            edges = bootstrap_result["graph"].get("edges", [])
                        elif "variables" in bootstrap_result:
                            variables = bootstrap_result["variables"]
                            edges = bootstrap_result.get("edges", [])
                        else:
                            edges = bootstrap_result.get("edges", [])
                        
                        if variables is None or not variables:
                            variables = list(bootstrap_df.columns)
                        
                        # Extract metadata if available
                        params = bootstrap_result.get("metadata", {}).get("params", {}) if "metadata" in bootstrap_result else {}
                        runtime = bootstrap_result.get("metadata", {}).get("runtime") if "metadata" in bootstrap_result else None
                        
                        
                        bootstrap_result = normalize_graph_result(
                            method=algorithm,
                            variables=variables,
                            edges=edges,
                            params=params,
                            runtime=runtime,
                            graph_type=graph_type  # Use original graph_type to ensure consistency
                        )
                        bootstrap_edges = get_edges(bootstrap_result)
                    
                    # Count only original edges that appear in bootstrap result
                    present_keys = set()
                    for edge in bootstrap_edges:
                        key = Bootstrapper._create_edge_key(edge, graph_type)
                        if key is not None and key in original_edge_keys:  # Skip self-loops and only count original edges
                            present_keys.add(key)
                    
                    for key in present_keys:
                        counts[key] += 1
                    
                    n_success += 1
                    
                except Exception as e:
                    failures.append(i)
                    logger.debug(f"Bootstrap iteration {i} failed: {e}")
                    continue
            
            # Calculate confidence scores
            if n_success == 0:
                # All iterations failed
                return {
                    "robustness_score": 0.0,
                    "effective_score": 0.0,
                    "edge_confidences": {key: 0.0 for key in original_edge_keys},
                    "edge_frequencies": {key: 0 for key in original_edge_keys},
                    "n_original_edges": n_original_edges,
                    "n_iterations": n_iterations,
                    "n_success_iterations": 0,
                    "success_rate": 0.0,
                    "failure_iterations": failures,
                    "error": "All bootstrap iterations failed"
                }
            
            # Calculate per-edge confidence
            edge_confidences = {
                key: counts[key] / n_success
                for key in original_edge_keys
            }
            
            # Graph-level scores
            robustness_score = float(np.mean(list(edge_confidences.values()))) if edge_confidences else 0.0
            success_rate = n_success / n_iterations if n_iterations > 0 else 0.0
            effective_score = robustness_score * success_rate  # Penalized by failure rate
            
            # Convert edge keys to strings for JSON serialization (tuples need conversion)
            # Use "|" delimiter for tuple keys to make parsing easier later
            edge_confidences_str = {
                "|".join(k) if isinstance(k, tuple) else k: v
                for k, v in edge_confidences.items()
            }
            edge_frequencies_str = {
                "|".join(k) if isinstance(k, tuple) else k: v
                for k, v in counts.items()
            }
            
            return {
                "robustness_score": robustness_score,
                "effective_score": effective_score,
                "edge_confidences": edge_confidences_str,
                "edge_frequencies": edge_frequencies_str,
                "n_original_edges": n_original_edges,
                "n_iterations": n_iterations,
                "n_success_iterations": n_success,
                "success_rate": success_rate,
                "failure_iterations": failures if failures else []
            }
            
        except Exception as e:
            logger.warning(f"Bootstrap evaluation failed: {e}")
            return {
                "robustness_score": 0.0,
                "effective_score": 0.0,
                "edge_confidences": {},
                "edge_frequencies": {},
                "n_original_edges": 0,
                "n_iterations": n_iterations,
                "n_success_iterations": 0,
                "success_rate": 0.0,
                "failure_iterations": [],
                "error": str(e)
            }

class GraphEvaluator:
    """Graph evaluation tools"""
    
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
                 variable_schema: Dict[str, Any] = None, max_k: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        import time
        t0 = time.time()
        vars_ = list(df.columns)
        edges = []
        p = len(vars_)  # number of variables
        
        # Set max_k: min(3, p-2) if not provided
        if max_k is None:
            max_k = min(3, max(0, p - 2))
        
        params = {"alpha": alpha, "indep_test": indep_test, "max_k": max_k}
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
            
            cg = pc(data, alpha=alpha, indep_test=test_func, max_k=max_k, verbose=False)
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
            return normalize_graph_result("PC", vars_, edges, params, runtime, graph_type="CPDAG")
        except Exception as e:
            return {"error": str(e)}

class GESTool:
    """GES algorithm wrapper. Supports multiple scoring methods
    """
    @staticmethod
    def discover(df: pd.DataFrame, score_func: str = "bic-g", max_indegree: int = 3, **kwargs) -> Dict[str, Any]:
        import time
        t0 = time.time()
        vars_ = list(df.columns)
        params = {"score_func": score_func, "max_indegree": max_indegree}
        edges = []
        
        try:
            if score_func in {"bic-g", "bic-cg", "bic-d", "bdeu"}:
                # --- pgmpy ---
                try:
                    from pgmpy.estimators import GES
                except ImportError:
                    logger.error("pgmpy not available for GES scoring. Install with: pip install pgmpy")
                    return {"error": "pgmpy not available"}
                
                est = GES(df)
                
                scoring_method = score_func
                

                dag = est.estimate(scoring_method=scoring_method)
                edges = [(str(u), str(v)) for u, v in dag.edges()]
                
                params["backend"] = "pgmpy"
                params["scoring_method"] = scoring_method
                
            elif score_func == "generalized_rkhs":
                try:
                    from causallearn.search.ScoreBased.GES import ges
                except ImportError:
                    logger.error("causal-learn not available for GES scoring")
                    return {"error": "causal-learn not available"}
                
                data = df.values
                res = ges(data, score_func="local_score_CV_general", maxP=max_indegree)
                G = getattr(res, 'G', None) or (res.get('G', None) if isinstance(res, dict) else None)
                
                if G is not None and hasattr(G, 'graph'):
                    graph = G.graph
                    if hasattr(graph, 'items'):
                        # Dictionary format
                        for (i, j), v in graph.items():
                            if v != 0:
                                edges.append((vars_[i], vars_[j]))
                    else:
                        for i in range(graph.shape[0]):
                            for j in range(graph.shape[1]):
                                if graph[i, j] != 0:
                                    edges.append((vars_[i], vars_[j]))
                
                params["backend"] = "causallearn"
                if hasattr(res, "score"):
                    params["raw_score"] = float(res.score)
            
            else:
                # Unsupported score function
                supported_funcs = ["bic-g", "bic-cg", "bic-d", "bdeu", "generalized_rkhs"]
                error_msg = f"Unsupported score_func '{score_func}'. Supported functions: {supported_funcs}"
                logger.error(error_msg)
                return {"error": error_msg}
            
            runtime = time.time() - t0
            return normalize_graph_result("GES", vars_, edges, params, runtime)
            
        except Exception as e:
            logger.error(f"GES execution failed: {e}")
            return {"error": str(e)}


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

class NOTEARSLinearTool:
    """NOTEARS linear algorithm implementation"""
    
    @staticmethod
    def _prepare_notears_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Common data preprocessing for NOTEARS algorithms"""
        # Remove NaN values
        df_clean = df.dropna()
        if df_clean.empty:
            raise ValueError("DataFrame is empty after dropping NaN values")
        vars_ = list(df_clean.columns)
        return df_clean, vars_
    
    @staticmethod
    def _extract_notears_edges(W_est: np.ndarray, vars_: List[str]) -> List[Dict[str, Any]]:
        """Common edge extraction logic from NOTEARS adjacency matrix"""
        edges = []
        d = W_est.shape[0]
        for i in range(d):
            for j in range(d):
                w = float(W_est[i, j])
                if w != 0.0:
                    edges.append({"from": vars_[i], "to": vars_[j], "weight": w})
        return edges
    
    @staticmethod
    def discover(df: pd.DataFrame, lambda1: float = 0.1, loss_type: str = "l2",
                 max_iter: int = 100, h_tol: float = 1e-8, rho_max: float = 1e16,
                 w_threshold: float = 0.3, **kwargs) -> Dict[str, Any]:
        """NOTEARS linear algorithm discovery"""
        import time
        t0 = time.time()
        
        try:
            from algorithms.notears.linear import notears_linear
            
            # Prepare data
            df_clean, vars_ = NOTEARSLinearTool._prepare_notears_data(df)
            d = df_clean.shape[1]
            
            # Run NOTEARS linear
            W_est = notears_linear(
                df_clean.values,
                lambda1=lambda1,
                loss_type=loss_type,
                max_iter=max_iter,
                h_tol=h_tol,
                rho_max=rho_max,
                w_threshold=w_threshold,
            )
            
            # Extract edges
            edges = NOTEARSLinearTool._extract_notears_edges(W_est, vars_)
            
            runtime = time.time() - t0
            
            params = {
                "backend": "notears",
                "lambda1": lambda1,
                "loss_type": loss_type,
                "max_iter": max_iter,
                "h_tol": h_tol,
                "rho_max": rho_max,
                "w_threshold": w_threshold,
            }
            
            return normalize_graph_result("NOTEARS-linear", vars_, edges, params=params, runtime=runtime)
            
        except ImportError as e:
            logger.error(f"NOTEARS-linear requires orca.algorithms.notears.linear: {e}")
            return {"error": f"NOTEARS-linear not available: orca.algorithms.notears.linear not installed"}
        except Exception as e:
            logger.error(f"NOTEARS-linear execution failed: {e}")
            return {"error": f"NOTEARS-linear execution failed: {str(e)}"}


class NOTEARSNonlinearTool:
    """NOTEARS nonlinear algorithm implementation"""
    
    @staticmethod
    def discover(df: pd.DataFrame, dims: List[int] = None, lambda1: float = 0.01,
                 lambda2: float = 0.01, max_iter: int = 100, h_tol: float = 1e-8,
                 rho_max: float = 1e16, w_threshold: float = 0.3, **kwargs) -> Dict[str, Any]:
        """NOTEARS nonlinear algorithm discovery"""
        import time
        import torch
        t0 = time.time()
        
        try:
            import sys
            from pathlib import Path
            
            # Get algorithms directory path
            this_file = Path(__file__).resolve()
            algorithms_dir = this_file.parent.parent.parent / "algorithms"
            algorithms_dir_str = str(algorithms_dir)
            
            # Add to sys.path if not already there
            if algorithms_dir_str not in sys.path:
                sys.path.insert(0, algorithms_dir_str)
            
            # Now import with the path in sys.path (nonlinear.py expects "notears" module)
            from notears.nonlinear import NotearsMLP, notears_nonlinear  # type: ignore
            
            # Prepare data
            df_clean, vars_ = NOTEARSLinearTool._prepare_notears_data(df)
            d = df_clean.shape[1]
            
            # Set default dims if not provided
            if dims is None:
                dims = [d, 10, 1]
            
            # Set torch default dtype
            torch.set_default_dtype(torch.double)
            
            # Create model
            model = NotearsMLP(dims=dims, bias=True)
            
            # Run NOTEARS nonlinear
            W_est = notears_nonlinear(
                model,
                df_clean.values,
                lambda1=lambda1,
                lambda2=lambda2,
                max_iter=max_iter,
                h_tol=h_tol,
                rho_max=rho_max,
                w_threshold=w_threshold,
            )
            
            # Extract edges
            edges = NOTEARSLinearTool._extract_notears_edges(W_est, vars_)
            
            runtime = time.time() - t0
            
            params = {
                "backend": "notears-nonlinear",
                "dims": dims,
                "lambda1": lambda1,
                "lambda2": lambda2,
                "max_iter": max_iter,
                "h_tol": h_tol,
                "rho_max": rho_max,
                "w_threshold": w_threshold,
            }
            
            return normalize_graph_result("NOTEARS-nonlinear", vars_, edges, params=params, runtime=runtime)
            
        except ImportError as e:
            logger.error(f"NOTEARS-nonlinear requires orca.algorithms.notears.nonlinear: {e}")
            return {"error": f"NOTEARS-nonlinear not available: orca.algorithms.notears.nonlinear not installed"}
        except Exception as e:
            logger.error(f"NOTEARS-nonlinear execution failed: {e}")
            return {"error": f"NOTEARS-nonlinear execution failed: {str(e)}"}


class FCITool:
    """FCI algorithm wrapper. Uses causal-learn and returns a PAG-oriented result.
    Note: PAG edges may be partially directed; we expose adjacencies as edges and
    mark graph_type="PAG" in metadata/params for downstream handling.
    """
    @staticmethod
    def discover(df: pd.DataFrame, alpha: float = 0.05, indep_test: str = "fisherz", 
                 variable_schema: Dict[str, Any] = None, max_k: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        import time
        t0 = time.time()
        vars_ = list(df.columns)
        edges = []
        p = len(vars_)  # number of variables
        
        # Set max_k: min(3, p-2) if not provided
        if max_k is None:
            max_k = min(3, max(0, p - 2))
        
        params = {"alpha": alpha, "indep_test": indep_test, "max_k": max_k}
        if variable_schema:
            params["variable_schema"] = variable_schema
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
                logger.warning("CMI test not yet implemented for FCI, using kci")
                test_func = kci
            else:
                test_map = {"fisherz": fisherz, "kci": kci, "gsq": gsq}
                test_func = test_map.get(indep_test, fisherz)
            
            # Causal-Learn FCI API: fci(data, indep_test, alpha, max_k)
            # Returns tuple: (G: GeneralGraph, edge_list: List[Edge])
            pag = fci(data, test_func, alpha, max_k=max_k)
            G, edge_list = pag
            
            # Extract node names from GeneralGraph
            nodes = G.nodes
            node_names = [n.get_name() for n in nodes]
            
            # Map node names to variable indices for edge construction
            name_to_idx = {name: idx for idx, name in enumerate(node_names)}
            
            # Extract edges from edge_list
            for e in edge_list:
                u_name = e.get_node1().get_name()
                v_name = e.get_node2().get_name()
                ep1 = e.get_endpoint1()  # Endpoint.TAIL, Endpoint.ARROW, Endpoint.CIRCLE
                ep2 = e.get_endpoint2()
                
                # Convert endpoint enums to PAG edge type markers
                # Endpoint.TAIL (-1) -> "-"
                # Endpoint.ARROW (1) -> ">"
                # Endpoint.CIRCLE (2) -> "o"
                ep1_str = "-" if ep1.value == -1 else (">" if ep1.value == 1 else "o")
                ep2_str = "-" if ep2.value == -1 else (">" if ep2.value == 1 else "o")
                
                # Construct PAG edge type: e.g., "o-o", "o->", "->", etc.
                if ep1_str == "o" and ep2_str == "o":
                    edge_type = "o-o"
                elif ep1_str == "o" and ep2_str == ">":
                    edge_type = "o->"
                elif ep1_str == ">" and ep2_str == "o":
                    edge_type = "<-o"
                elif ep1_str == "-" and ep2_str == ">":
                    edge_type = "->"
                elif ep1_str == ">" and ep2_str == "-":
                    edge_type = "<-"
                elif ep1_str == "-" and ep2_str == "-":
                    edge_type = "--"
                else:
                    edge_type = f"{ep1_str}{ep2_str}"
                
                # Map node names to our variable names
                u_idx = name_to_idx.get(u_name)
                v_idx = name_to_idx.get(v_name)
                
                if u_idx is not None and v_idx is not None:
                    edges.append({
                        "from": vars_[u_idx],
                        "to": vars_[v_idx],
                        "type": edge_type
                    })
            runtime = time.time() - t0
            params_dict = {"alpha": alpha, "indep_test": indep_test, "graph_type": "PAG", "max_k": max_k}
            if variable_schema:
                params_dict["variable_schema"] = variable_schema
            result = normalize_graph_result(
                "FCI", vars_, edges,
                params=params_dict,
                runtime=runtime,
                graph_type="PAG"
            )
            return result
        except Exception as e:
            return {"error": str(e)}

class PruningTool:
    """Pruning tools for CI testing and structural consistency"""
        
    @staticmethod
    def _convert_to_networkx_dag(graph: Dict[str, Any]) -> nx.DiGraph:
        """Convert graph dict to NetworkX DAG"""
        variables = get_variables(graph)
        edges = get_edges(graph)
        
        G = nx.DiGraph()
        G.add_nodes_from(variables)
        
        for edge in edges:
            from_var = edge["from"]
            to_var = edge["to"]
            if from_var in variables and to_var in variables:
                G.add_edge(from_var, to_var)
        
        return G
    
    @staticmethod
    def global_markov_test(graph: Dict[str, Any], df: pd.DataFrame, alpha: float = 0.05, max_pa_size: Optional[int] = None) -> Dict[str, Any]:
        """Test global Markov property using d-separation and CI tests
        
        Args:
            graph: Graph dictionary
            df: DataFrame for CI testing
            alpha: Significance level for CI tests
            max_pa_size: Maximum parent set size. Tests with |Pa(X)| > max_pa_size are skipped.
                        If None, no limit is applied.
        """
        try:
            try:
                validate_graph_schema(graph)
            except ValueError as e:
                return {"violation_ratio": 1.0, "error": f"Invalid graph structure: {e}"}
            
            edges = get_edges(graph)
            variables = get_variables(graph)
            
            if not edges or not variables:
                return {"violation_ratio": 0.0, "message": "No edges to test"}
            
            # Determine graph type
            graph_type = get_graph_type(graph)
            
            # Convert to NetworkX DAG
            G = PruningTool._convert_to_networkx_dag(graph)
            
            # Verify it's a valid DAG
            if not nx.is_directed_acyclic_graph(G):
                return {
                    "violation_ratio": None,
                    "total_tests": 0,
                    "violations": 0,
                    "n_success_tests": 0,
                    "n_failed_tests": 0,
                    "n_skipped_tests": 0,
                    "failure_rate": None,
                    "message": f"Global Markov test skipped (converted graph from {graph_type} is not a DAG)"
                }
            
            # Get CI test method from graph metadata
            ci_test_method = "fisherz"  # default
            variable_schema = None
            if "metadata" in graph and "params" in graph["metadata"]:
                params = graph["metadata"]["params"]
                ci_test_method = params.get("indep_test", "fisherz")
                variable_schema = params.get("variable_schema")
            
            # Prepare data for CI testing
            data_np = df[variables].values.astype(float)
            
            # Import and instantiate CI test function
            try:
                if ci_test_method == "lrt":
                    # Use LRT test for mixed data (requires variable_schema)
                    if variable_schema is None:
                        logger.warning("LRT test requires variable_schema, falling back to fisherz")
                        from causallearn.utils.cit import FisherZ
                        ci_test_func = FisherZ(data_np)
                    else:
                        ci_test_func = LRTTest(data_np, variable_schema)
                else:
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
            n_success_tests = 0
            n_failed_tests = 0
            n_skipped_tests = 0
            n_no_testable_pairs = 0  # Track nodes with no testable non-descendants
            
            # Create variable index mapping
            var_to_idx = {var: i for i, var in enumerate(variables)}
            
            for x in G.nodes():
                pa_x = list(G.predecessors(x))
                
                desc_x = set(nx.descendants(G, x)) | {x}
                non_desc_x = [y for y in G.nodes() if y not in desc_x and y not in pa_x]
                
                if not non_desc_x:
                    # Log when there are no testable pairs for a node
                    n_no_testable_pairs += 1
                    ci_tests_log.append({
                        "hypothesis": f"{x} ⟂ Y | {pa_x}",
                        "status": "no_testable_pairs",
                        "reason": f"No non-descendants to test (all nodes are descendants or parents of {x})"
                    })
                    continue
                
                # Test X ⊥ Y | Pa(X) for each non-descendant Y
                x_idx = var_to_idx[x]
                pa_indices = [var_to_idx[p] for p in pa_x]
                
                for y in non_desc_x:
                    y_idx = var_to_idx[y]
                    
                    # Skip tests if parent set size exceeds max_pa_size (policy-based skip)
                    if max_pa_size is not None and len(pa_x) > max_pa_size:
                        n_skipped_tests += 1
                        ci_tests_log.append({
                            "hypothesis": f"{x} ⟂ {y} | {pa_x}",
                            "status": "skipped",
                            "reason": f"max_pa_size={max_pa_size} exceeded (|Pa({x})|={len(pa_x)})"
                        })
                        continue
                    
                    try:
                        # Perform CI test: X ⊥ Y | Pa(X)
                        p_value = ci_test_func(x_idx, y_idx, pa_indices)
                        violated = p_value < alpha
                        
                        if violated:
                            violations += 1
                        
                        n_success_tests += 1
                        ci_tests_log.append({
                            "hypothesis": f"{x} ⟂ {y} | {pa_x}",
                            "status": "success",
                            "p_value": float(p_value),
                            "violation": violated
                        })
                        
                    except Exception as e:
                        # Test failure: numerical issues, library problems, missing data, impossible conditioning set, etc.
                        n_failed_tests += 1
                        logger.warning(f"CI test failed for {x} ⟂ {y} | {pa_x}: {e}")
                        ci_tests_log.append({
                            "hypothesis": f"{x} ⟂ {y} | {pa_x}",
                            "status": "failed",
                            "error": str(e),
                            "error_type": type(e).__name__
                        })
            
            # violation_ratio is calculated only from successful tests
            violation_ratio = violations / n_success_tests if n_success_tests > 0 else None
            
            # failure_rate = n_failed / (n_success + n_failed)
            total_attempted = n_success_tests + n_failed_tests
            failure_rate = n_failed_tests / total_attempted if total_attempted > 0 else None
            
            message = f"Test completed on {graph_type} with {n_success_tests} successful d-separation tests"
            if n_failed_tests > 0 or n_skipped_tests > 0:
                message += f", {n_failed_tests} failed, {n_skipped_tests} skipped"
            if n_no_testable_pairs > 0:
                message += f", {n_no_testable_pairs} nodes with no testable pairs"
            
            # Log the message
            logger.info(message)
            
            return {
                "violation_ratio": violation_ratio,
                "total_tests": n_success_tests + n_failed_tests + n_skipped_tests + n_no_testable_pairs,
                "total_attempted": total_attempted,  # Explicitly include total_attempted in return
                "violations": violations,
                "n_success_tests": n_success_tests,
                "n_failed_tests": n_failed_tests,
                "n_skipped_tests": n_skipped_tests,
                "n_no_testable_pairs": n_no_testable_pairs,
                "failure_rate": failure_rate,
                "ci_tests": ci_tests_log,
                "alpha": alpha,
                "max_pa_size": max_pa_size,
                "ci_test_method": ci_test_method,
                "graph_type": graph_type,
                "message": message
            }
            
        except Exception as e:
            logger.error(f"Global Markov test failed: {e}")
            return {"violation_ratio": 1.0, "error": str(e)}
    
    @staticmethod
    def structural_consistency_test(graph: Dict[str, Any], df: pd.DataFrame, algorithm_name: str, n_subsets: int = 3) -> Dict[str, Any]:
        """Test structural consistency using subsampling
        
        Graph type-specific handling:
        - DAG: Fully oriented SHD 
        - CPDAG: Edge-wise adjacency differences
        - PAG: Skip test, return None/NaN
        """
        try:
            if "graph" not in graph or "edges" not in graph["graph"]:
                return {"instability_score": 1.0, "error": "Invalid graph structure"}
            
            # Determine graph type
            graph_type = get_graph_type(graph)
            
            # Skip PAG graphs
            if graph_type == "PAG":
                return {
                    "instability_score": None,
                    "message": "Structural consistency test skipped for PAG (latent confounders present)",
                    "graph_type": graph_type
                }
            
            variables = get_variables(graph)
            if len(variables) < 3:
                return {"instability_score": 0.0, "message": "Too few variables for subsampling"}
            
            # Generate random variable subsets
            subset_results = []
            subset_fracs = [0.6, 0.7, 0.8]  # Fraction options for subset sizes
            
            for i in range(n_subsets):
                try:
                    # Randomly select a fraction and calculate subset size
                    frac = np.random.choice(subset_fracs)
                    subset_size = max(3, int(frac * len(variables)))
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
                    elif algorithm_name == "NOTEARS-linear":
                        subset_result = NOTEARSLinearTool.discover(subset_df)
                    elif algorithm_name == "NOTEARS-nonlinear":
                        subset_result = NOTEARSNonlinearTool.discover(subset_df)
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
                    
                    # Compute SHD based on graph type
                    if graph_type == "CPDAG":
                        # CPDAG: adjacency differences
                        normalized_shd = PruningTool._compute_adjacency_difference(
                            original_restricted,
                            subset_graph,
                            normalize=True,
                            original_graph=graph
                        )
                    else:
                        # DAG: fully oriented SHD
                        normalized_shd = PruningTool._compute_shd(
                            original_restricted, 
                            subset_graph, 
                            normalize=True,
                            original_graph=graph
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
                "subset_results": subset_results,
                "graph_type": graph_type
            }
            
        except Exception as e:
            logger.error(f"Structural consistency test failed: {e}")
            return {"instability_score": 1.0, "error": str(e)}
    
    @staticmethod
    def _compute_shd(graph1: Dict[str, Any], graph2: Dict[str, Any], normalize: bool = True, 
                     original_graph: Optional[Dict[str, Any]] = None) -> float:
        """Compute raw or normalized Structural Hamming Distance based on common variables.
        
        Args:
            graph1: First graph (restricted graph) 
            graph2: Second graph (subset graph)
            normalize: Whether to normalize SHD by max possible edges
            original_graph: Original full graph. If provided, edges in graph2 that exist due to
                          directed paths in original_graph are excluded from penalty (induced edges).
        """
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
            
            if original_graph is not None:
                try:
                    original_G = PruningTool._convert_to_networkx_dag(original_graph)
                    
                    filtered_edges2 = set()
                    for edge in edges2:
                        x, z = edge
                        
                        if edge in edges1:
                            filtered_edges2.add(edge)
                        else:
                            # Check if this edge is induced (exists due to path in original_graph)
                            if nx.has_path(original_G, x, z) or nx.has_path(original_G, z, x):
                                continue
                            else:
                                filtered_edges2.add(edge)
                    
                    edges2 = filtered_edges2
                except Exception as e:
                    logger.warning(f"Failed to filter induced edges: {e}, using original edges2")
            
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
    def _compute_adjacency_difference(graph1: Dict[str, Any], graph2: Dict[str, Any], normalize: bool = True,
                              original_graph: Optional[Dict[str, Any]] = None) -> float:
        """Compute edge-wise adjacency differences, ignore orientation).
        
        Adjacency difference = |E1 Δ E2| where E1, E2 are undirected edge sets.
        
        Args:
            graph1: First graph (restricted graph)
            graph2: Second graph (subset graph)
            normalize: Whether to normalize adjacency difference by max possible edges
            original_graph: Original full graph. Used to exclude induced edges.
        """
        try:
            def get_adjacency_edges(g: Dict[str, Any]) -> set:
                """Extract undirected skeleton edges from graph"""
                edges = g.get("graph", {}).get("edges", [])
                skel = set()
                for e in edges:
                    u, v = e["from"], e["to"]
                    if u == v:
                        continue
                    skel.add(tuple(sorted((u, v))))  # undirected edge
                return skel
            
            vars1 = set(graph1.get("graph", {}).get("variables", []))
            vars2 = set(graph2.get("graph", {}).get("variables", []))
            common_vars = vars1.intersection(vars2)
            
            if not common_vars:
                return 1.0 if normalize else 0.0
            
            # Get skeleton edges within common variables
            E1 = get_adjacency_edges(graph1)
            E2 = get_adjacency_edges(graph2)
            
            # Filter to common variables only
            E1_filtered = {e for e in E1 if e[0] in common_vars and e[1] in common_vars}
            E2_filtered = {e for e in E2 if e[0] in common_vars and e[1] in common_vars}
            
            # If original_graph is provided, exclude induced edges from E2
            if original_graph is not None:
                try:
                    original_G = PruningTool._convert_to_networkx_dag(original_graph)
                    
                    filtered_E2 = set()
                    for edge in E2_filtered:
                        x, z = edge
                        
                        # If edge exists in E1, keep it
                        if edge in E1_filtered:
                            filtered_E2.add(edge)
                        else:
                            # Check if this edge is induced (exists due to path in original_graph)
                            if nx.has_path(original_G, x, z) or nx.has_path(original_G, z, x):
                                # Induced edge, do not count as a penalty
                                continue
                            else:
                                # This is a real difference, count it
                                filtered_E2.add(edge)
                    
                    E2_filtered = filtered_E2
                except Exception as e:
                    logger.warning(f"Failed to filter induced edges in adjacency difference: {e}")
            
            # Compute symmetric difference
            diff = E1_filtered.symmetric_difference(E2_filtered)
            raw_shd = len(diff)
            
            if not normalize:
                return float(raw_shd)
            
            n = len(common_vars)
            if n < 2:
                return 0.0
            
            max_edges = n * (n - 1) // 2  # For undirected skeleton
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
            try:
                graph_edges = get_edges(graph)
                for edge in graph_edges:
                    if edge["from"] in subset_set and edge["to"] in subset_set:
                        restricted_edges.append(edge.copy())
            except ValueError:
                # Fallback for old schema
                if "graph" in graph and "edges" in graph["graph"]:
                    for edge in graph["graph"]["edges"]:
                        if edge["from"] in subset_set and edge["to"] in subset_set:
                            restricted_edges.append(edge.copy())
            
            # Preserve metadata from original graph if available
            metadata = {}
            if "metadata" in graph:
                metadata = graph["metadata"].copy()
            
            return {
                "graph": {
                    "variables": list(subset_vars),
                    "edges": restricted_edges
                },
                "metadata": metadata
            }
        except Exception as e:
            logger.warning(f"Graph restriction failed: {e}")
            metadata = {}
            if "metadata" in graph:
                metadata = graph["metadata"].copy()
            return {
                "graph": {
                    "variables": list(subset_vars),
                    "edges": []
                },
                "metadata": metadata
            }

class EnsembleTool:
    """Ensemble tools for consensus skeleton and PAG construction"""
    
    @staticmethod
    def build_consensus_skeleton(graphs: List[Dict[str, Any]], weights: Optional[List[float]] = None, threshold: float = 0.5) -> Dict[str, Any]:
        """Build consensus skeleton based on undirected adjacency with confidence scores"""
        try:
            if not graphs:
                return {
                    "graph": {
                        "edges": [],
                        "variables": []
                    },
                    "metadata": {
                        "graph_type": "skeleton",
                        "method": "consensus_skeleton",
                        "params": {"threshold": threshold}
                    }
                }
            
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
                # Use helper functions for unified schema access
                try:
                    graph_edges = get_edges(graph)
                    graph_vars = get_variables(graph)
                except ValueError:
                    # Fallback for old schema
                    if "graph" in graph and "edges" in graph["graph"]:
                        graph_edges = graph["graph"]["edges"]
                        graph_vars = graph["graph"].get("variables", [])
                    else:
                        continue
                
                weight = weights[i]
                all_variables.update(graph_vars)
                
                for edge in graph_edges:
                    # Create undirected adjacency key
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
                    # Direction is undetermined at skeleton stage, use "--" for undirected
                    consensus_edges.append({
                        "from": var1,
                        "to": var2,
                        "type": "--",
                        "weight": confidence,
                        "confidence": confidence
                    })
                    confidence_scores[f"{var1}-{var2}"] = confidence
            
            return {
                "graph": {
                    "edges": consensus_edges,
                    "variables": list(all_variables)
                },
                "metadata": {
                    "graph_type": "skeleton",
                    "method": "consensus_skeleton",
                    "params": {
                        "threshold": threshold,
                        "n_input_graphs": len(graphs),
                        "confidence_scores": confidence_scores
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Consensus skeleton building failed: {e}")
            return {
                "graph": {
                    "edges": [],
                    "variables": []
                },
                "metadata": {
                    "graph_type": "skeleton",
                    "method": "consensus_skeleton",
                    "params": {},
                    "error": str(e)
                }
            }
    
    @staticmethod
    def resolve_directions(skeleton: Dict[str, Any], graphs: List[Dict[str, Any]], data_profile: Dict[str, Any], weights: Optional[List[float]] = None) -> Dict[str, Any]:
        """Resolve edge directions with uncertainty markers using weight-based voting.
        
        Maintains unified schema structure. Uses 'type' field instead of 'marker'.
        
        Args:
            skeleton: Consensus skeleton graph
            graphs: List of input graphs for direction voting
            data_profile: Data profile dictionary
            weights: Optional list of weights for each graph (defaults to equal weights)
        """
        try:
            # Ensure skeleton follows unified schema
            if "graph" not in skeleton or "edges" not in skeleton["graph"]:
                # Try to convert old schema
                if "edges" in skeleton:
                    skeleton = {
                        "graph": {
                            "edges": skeleton.get("edges", []),
                            "variables": skeleton.get("variables", [])
                        },
                        "metadata": skeleton.get("metadata", {"graph_type": "skeleton"})
                    }
                else:
                    return skeleton
            
            # Normalize weights if provided
            if weights is None:
                weights = [1.0] * len(graphs)
            
            # Normalize weights to sum to 1.0
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
            else:
                weights = [1.0 / len(graphs)] * len(graphs)
            
            # Constants for tie/uncertain judgment
            eps = 1e-6  # Threshold for conflict detection
            min_conf = 0.1  # Minimum confidence threshold for uncertain detection
            
            resolved_edges = []
            skeleton_edges = skeleton["graph"]["edges"]
            
            for edge in skeleton_edges:
                from_var = edge["from"]
                to_var = edge["to"]
                
                # Weight-based direction votes
                forward_votes = 0.0
                backward_votes = 0.0
                total_votes = 0.0
                
                for i, graph in enumerate(graphs):
                    weight = weights[i]
                    try:
                        graph_edges = get_edges(graph)
                    except ValueError:
                        if "graph" in graph and "edges" in graph["graph"]:
                            graph_edges = graph["graph"]["edges"]
                        else:
                            continue
                    
                    for graph_edge in graph_edges:
                        # Only include directed edges (type == "->") in voting
                        if graph_edge.get("type") != "->":
                            continue
                        
                        if graph_edge["from"] == from_var and graph_edge["to"] == to_var:
                            forward_votes += weight
                            total_votes += weight
                        elif graph_edge["from"] == to_var and graph_edge["to"] == from_var:
                            backward_votes += weight
                            total_votes += weight
                
                # Resolve direction and set type field
                resolved_edge = edge.copy()
                
                if total_votes == 0:
                    # No direction information, mark as uncertain
                    resolved_edge["direction"] = "uncertain"
                    resolved_edge["type"] = "o-o"
                elif abs(forward_votes - backward_votes) < eps:
                    # Conflict: votes are too close (within eps)
                    resolved_edge["direction"] = "conflict"
                    resolved_edge["type"] = "o-o"
                elif max(forward_votes, backward_votes) < min_conf:
                    # Uncertain: maximum vote is below minimum confidence threshold
                    resolved_edge["direction"] = "uncertain"
                    resolved_edge["type"] = "o-o"
                elif forward_votes > backward_votes:
                    # Forward direction consensus
                    resolved_edge["direction"] = "forward"
                    resolved_edge["type"] = "->"
                else:
                    # Backward direction consensus
                    resolved_edge["from"] = to_var
                    resolved_edge["to"] = from_var
                    resolved_edge["direction"] = "backward"
                    resolved_edge["type"] = "->"
                
                resolved_edge["forward_votes"] = forward_votes
                resolved_edge["backward_votes"] = backward_votes
                resolved_edge["total_votes"] = total_votes
                
                resolved_edges.append(resolved_edge)
            
            # Return unified schema
            result = skeleton.copy()
            result["graph"]["edges"] = resolved_edges
            return result
            
        except Exception as e:
            logger.error(f"Direction resolution failed: {e}")
            return skeleton
    
    @staticmethod
    def construct_pag(directions_result: Dict[str, Any]) -> Dict[str, Any]:
        """Construct PAG-like graph with uncertainty markers.
        
        Args:
            directions_result: Resolved directions result (already contains unified schema structure)
            
        Returns:
            PAG graph following unified schema with graph.graph.edges and metadata.graph_type
        """
        try:
            # directions_result should already be unified schema from resolve_directions
            pag = directions_result.copy()
            
            # Ensure unified schema structure
            if "graph" not in pag:
                # Convert old schema if needed
                if "edges" in pag:
                    pag = {
                        "graph": {
                            "edges": pag.get("edges", []),
                            "variables": pag.get("variables", [])
                        },
                        "metadata": pag.get("metadata", {})
                    }
            
            # Update metadata with PAG-specific information
            if "metadata" not in pag:
                pag["metadata"] = {}
            
            pag["metadata"]["graph_type"] = "PAG"
            pag["metadata"]["construction_method"] = "consensus"
            pag["metadata"]["uncertainty_markers"] = True
            pag["metadata"]["edge_types"] = ["->", "o-o", "o->", "<->", "--"]
            
            # Ensure all edges have type field (convert marker if present)
            if "graph" in pag and "edges" in pag["graph"]:
                for edge in pag["graph"]["edges"]:
                    if "type" not in edge and "marker" in edge:
                        edge["type"] = edge["marker"]
                    elif "type" not in edge:
                        # Default PAG uncertain edge
                        edge["type"] = "o-o"
            
            return pag
            
        except Exception as e:
            logger.error(f"PAG construction failed: {e}")
            return directions_result
    
    @staticmethod
    def construct_dag(pag: Dict[str, Any], data_profile: Dict[str, Any], top_algorithm: str,
                      execution_plan: List[Dict[str, Any]] = None,
                      algorithm_results: Dict[str, Any] = None,
                      top_candidates: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Construct single DAG with tie-breaking and cycle avoidance.
        
        Uses incremental cycle-aware construction:
        - Rule 2: If resolved direction creates cycle, try opposite direction, drop if both cycle
        - Rule 3: If no direction evidence, try cycle-safe directions, drop if both cycle
        
        Returns unified schema with graph.graph.edges and metadata.graph_type.
        """
        import networkx as nx
        
        try:
            # Ensure pag follows unified schema
            if "graph" not in pag:
                # Convert old schema if needed
                if "edges" in pag:
                    pag = {
                        "graph": {
                            "edges": pag.get("edges", []),
                            "variables": pag.get("variables", [])
                        },
                        "metadata": pag.get("metadata", {})
                    }
                else:
                    raise ValueError("Invalid pag structure for DAG construction")
            
            dag_dict = pag.copy()
            pag_edges = dag_dict["graph"]["edges"]
            
            # Sort edges by confidence (highest first) to prioritize reliable edges
            sorted_edges = sorted(pag_edges, key=lambda e: e.get("confidence", 0.0), reverse=True)
            
            # Build DAG incrementally with cycle checking
            G = nx.DiGraph()
            final_edges = []
            skipped_edges = []
            
            def would_create_cycle(partial_dag, u, v):
                """Check if adding edge u->v would create a cycle"""
                if partial_dag.has_node(u) and partial_dag.has_node(v):
                    return nx.has_path(partial_dag, v, u)
                return False
            
            for edge in sorted_edges:
                u, v = str(edge["from"]), str(edge["to"])
                direction = edge.get("direction")
                
                # Handle uncertain/conflict edges
                if direction in ["uncertain", "conflict"]:
                    resolved_edge = EnsembleTool._apply_tie_breaking(
                        edge, top_candidates=top_candidates,
                        execution_plan=execution_plan,
                        algorithm_results=algorithm_results
                    )
                    
                    if resolved_edge:
                        # Rule 2: Try resolved direction, then opposite if cycle
                        resolved_u, resolved_v = resolved_edge["from"], resolved_edge["to"]
                        
                        if not would_create_cycle(G, resolved_u, resolved_v):
                            G.add_edge(resolved_u, resolved_v)
                            final_edge = resolved_edge.copy()
                            final_edge["type"] = "->"
                            final_edges.append(final_edge)
                        elif not would_create_cycle(G, resolved_v, resolved_u):
                            # Try opposite direction
                            G.add_edge(resolved_v, resolved_u)
                            final_edge = resolved_edge.copy()
                            final_edge["from"], final_edge["to"] = resolved_v, resolved_u
                            final_edge["type"] = "->"
                            final_edges.append(final_edge)
                        else:
                            # Both create cycles → drop
                            skipped_edges.append(edge)
                    else:
                        # Rule 3: No direction evidence → try cycle-safe directions
                        if not would_create_cycle(G, u, v):
                            G.add_edge(u, v)
                            final_edge = edge.copy()
                            final_edge["type"] = "->"
                            final_edges.append(final_edge)
                        elif not would_create_cycle(G, v, u):
                            G.add_edge(v, u)
                            final_edge = edge.copy()
                            final_edge["from"], final_edge["to"] = v, u
                            final_edge["type"] = "->"
                            final_edges.append(final_edge)
                        else:
                            # Both create cycles → drop
                            skipped_edges.append(edge)
                else:
                    # Edge already has direction
                    if not would_create_cycle(G, u, v):
                        G.add_edge(u, v)
                        final_edge = edge.copy()
                        final_edge["type"] = "->"
                        final_edges.append(final_edge)
                    else:
                        # Cycle would be created, skip this edge
                        skipped_edges.append(edge)
            
            # Verify final graph is a DAG
            if not nx.is_directed_acyclic_graph(G):
                logger.error("Final graph is not a DAG despite cycle prevention. This should not happen.")
                raise ValueError("Constructed graph contains cycles")
            
            # Update unified schema structure
            dag_dict["graph"]["edges"] = final_edges
            if "metadata" not in dag_dict:
                dag_dict["metadata"] = {}
            
            dag_dict["metadata"]["graph_type"] = "DAG"
            dag_dict["metadata"]["construction_method"] = "tie_breaking_and_cycle_avoidance"
            dag_dict["metadata"]["top_algorithm"] = top_algorithm
            dag_dict["metadata"]["params"] = {
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
    def _apply_tie_breaking(edge: Dict[str, Any], top_candidates: List[Dict[str, Any]] = None,
                           execution_plan: List[Dict[str, Any]] = None,
                           algorithm_results: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Apply tie-breaking by iterating through top_candidates sorted by composite_score.
        
        Returns the first algorithm that has a directed edge (->) for the uncertain edge,
        or None if no directed edge is found in any candidate.
        """
        try:
            from_var = edge.get("from")
            to_var = edge.get("to")
            
            if not top_candidates or not algorithm_results:
                return None
            
            # Sort top_candidates by composite_score descending
            sorted_candidates = sorted(
                top_candidates, 
                key=lambda c: c.get("composite_score", 0.0), 
                reverse=True
            )
            
            for rank, candidate in enumerate(sorted_candidates, 1):
                alg_name = candidate.get("algorithm")
                
                # Get algorithm result
                alg_result = algorithm_results.get(alg_name)
                if not alg_result:
                    continue
                
                # Get edges from algorithm graph
                try:
                    graph_edges = get_edges(alg_result)
                except (ValueError, KeyError):
                    continue
                
                # Find matching edge
                for graph_edge in graph_edges:
                    edge_from = graph_edge.get("from")
                    edge_to = graph_edge.get("to")
                    edge_type = graph_edge.get("type")
                    
                    # Check if edge matches (forward or backward)
                    if (edge_from == from_var and edge_to == to_var) or \
                       (edge_from == to_var and edge_to == from_var):
                        # Only use directed edges (->)
                        if edge_type == "->":
                            resolved_edge = edge.copy()
                            resolved_edge["from"] = graph_edge["from"]
                            resolved_edge["to"] = graph_edge["to"]
                            resolved_edge["direction"] = "forward" if (edge_from == from_var) else "backward"
                            resolved_edge["type"] = "->"
                            resolved_edge["tie_breaking"] = f"Rank-{rank} algorithm ({alg_name})"
                            return resolved_edge
                        # Skip undirected edges (--, o-o, etc.) and continue to next candidate
                        break
            
            # No directed edge found in any candidate
            return None
            
        except Exception as e:
            logger.warning(f"Tie-breaking failed: {e}")
            return None

    
    @staticmethod
    def _extract_lim_direction(lim_result: Dict[str, Any], var1: str, var2: str) -> Optional[Dict[str, str]]:
        """Extract direction from LiM result for a specific edge (placeholder)"""
        # This should parse LiM output to find edge direction
        if "graph" in lim_result and "edges" in lim_result["graph"]:
            edges = lim_result["graph"]["edges"]
            for e in edges:
                if (e.get("from") == var1 and e.get("to") == var2) or \
                   (e.get("from") == var2 and e.get("to") == var1):
                    return {"from": e["from"], "to": e["to"]}
        return None

class GraphVisualizer:
    """Graph visualization tools for causal DAGs"""
    
    @staticmethod
    def visualize_dag(dag_dict: Dict[str, Any]) -> Optional[nx.DiGraph]:
        """Convert DAG dictionary to NetworkX DiGraph
        
        Args:
            dag_dict: DAG dictionary with 'graph' key containing 'variables' and 'edges',
                     or with 'variables' and 'edges' at top level
            
        Returns:
            NetworkX DiGraph or None if conversion fails
        """
        try:
            # Support both structures: {"graph": {...}} and direct structure
            if "graph" in dag_dict:
                graph_data = dag_dict["graph"]
                variables = graph_data.get("variables", [])
                edges = graph_data.get("edges", [])
            else:
                # Direct structure: variables and edges at top level
                variables = dag_dict.get("variables", [])
                edges = dag_dict.get("edges", [])
            
            # If variables not found, extract from edges
            if not variables and edges:
                variables_set = set()
                for edge in edges:
                    from_var = str(edge.get("from", ""))
                    to_var = str(edge.get("to", ""))
                    if from_var:
                        variables_set.add(from_var)
                    if to_var:
                        variables_set.add(to_var)
                variables = list(variables_set)
            
            if not variables:
                logger.warning("No variables found in DAG")
                return None
            
            G = nx.DiGraph()
            G.add_nodes_from(variables)
            
            # Add edges
            for edge in edges:
                from_var = str(edge.get("from", ""))
                to_var = str(edge.get("to", ""))
                
                if from_var and to_var and from_var in variables and to_var in variables:
                    edge_attrs = {}
                    if "weight" in edge:
                        edge_attrs["weight"] = float(edge["weight"])
                    if "confidence" in edge:
                        edge_attrs["confidence"] = float(edge["confidence"])
                    
                    G.add_edge(from_var, to_var, **edge_attrs)
            
            return G
            
        except Exception as e:
            logger.error(f"Failed to convert DAG to NetworkX graph: {e}")
            return None
    
    @staticmethod
    def save_graph(dag_dict: Dict[str, Any], output_dir: str = "outputs/images/causal_graphs", 
                   formats: List[str] = ["png", "svg"]) -> Dict[str, Any]:
        """Save DAG visualization to file(s)
        
        Args:
            dag_dict: DAG dictionary to visualize
            output_dir: Output directory path
            formats: List of formats to save (e.g., ["png", "svg"])
            
        Returns:
            Dictionary with saved file paths or error information
        """
        try:
            import os
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            from datetime import datetime
            
            os.makedirs(output_dir, exist_ok=True)
            
            G = GraphVisualizer.visualize_dag(dag_dict)
            if G is None:
                return {"error": "Failed to convert DAG to NetworkX graph"}
            
            if len(G.nodes()) == 0:
                logger.warning("Empty graph, skipping visualization")
                return {"error": "Empty graph"}
            
            # Generate timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"causal_graph_{timestamp}"
            
            saved_paths = {}
            
            # Try different layout algorithms
            try:
                # Try hierarchical layout first (good for DAGs)
                pos = GraphVisualizer._hierarchical_layout(G)
            except Exception:
                try:
                    # Fallback to spring layout
                    pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
                except Exception:
                    # Final fallback to circular layout
                    pos = nx.circular_layout(G)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Draw nodes
            node_colors = ['#4A90E2' for _ in G.nodes()]
            node_sizes = [2000 for _ in G.nodes()]
            
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                 node_size=node_sizes, alpha=0.9, ax=ax)
            
            # Draw edges
            edge_colors = ['#666666' for _ in G.edges()]
            edge_widths = [2.0 for _ in G.edges()]
            
            # Adjust edge width based on weight/confidence if available
            for i, (u, v, data) in enumerate(G.edges(data=True)):
                if "weight" in data:
                    edge_widths[i] = 1.0 + abs(float(data["weight"])) * 2.0
                elif "confidence" in data:
                    edge_widths[i] = 1.0 + float(data["confidence"]) * 2.0
            
            nx.draw_networkx_edges(G, pos, edge_color=edge_colors, 
                                 width=edge_widths, alpha=0.6, arrows=True, 
                                 arrowsize=20, arrowstyle='->', ax=ax)
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
            
            # Set title
            metadata = dag_dict.get("metadata", {})
            graph_type = metadata.get("graph_type", "DAG")
            construction_method = metadata.get("construction_method", "unknown")
            title = f"Causal {graph_type} - {construction_method}"
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            
            ax.axis('off')
            plt.tight_layout()
            
            # Save in requested formats
            for fmt in formats:
                if fmt.lower() == "png":
                    filepath = os.path.join(output_dir, f"{base_filename}.png")
                    plt.savefig(filepath, format='png', dpi=300, bbox_inches='tight')
                    saved_paths["png"] = filepath
                    logger.info(f"Saved DAG visualization to {filepath}")
                elif fmt.lower() == "svg":
                    filepath = os.path.join(output_dir, f"{base_filename}.svg")
                    plt.savefig(filepath, format='svg', bbox_inches='tight')
                    saved_paths["svg"] = filepath
                    logger.info(f"Saved DAG visualization to {filepath}")
                elif fmt.lower() == "pdf":
                    filepath = os.path.join(output_dir, f"{base_filename}.pdf")
                    plt.savefig(filepath, format='pdf', bbox_inches='tight')
                    saved_paths["pdf"] = filepath
                    logger.info(f"Saved DAG visualization to {filepath}")
            
            plt.close(fig)
            
            return {
                "success": True,
                "saved_paths": saved_paths,
                "n_nodes": len(G.nodes()),
                "n_edges": len(G.edges())
            }
            
        except ImportError as e:
            error_msg = f"Visualization dependencies not available: {e}"
            logger.warning(error_msg)
            return {"error": error_msg}
        except Exception as e:
            error_msg = f"Failed to save graph visualization: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    @staticmethod
    def _hierarchical_layout(G: nx.DiGraph) -> Dict[str, Tuple[float, float]]:
        """Generate hierarchical layout for DAG using topological sort
        
        Args:
            G: NetworkX DiGraph (assumed to be a DAG, verified in construct_dag)
            
        Returns:
            Dictionary mapping nodes to (x, y) positions
        """
        try:
            # Get topological levels
            levels = {}
            for node in nx.topological_sort(G):
                # Find the maximum level of predecessors
                pred_levels = [levels.get(pred, -1) for pred in G.predecessors(node)]
                levels[node] = max(pred_levels, default=-1) + 1
            
            # Assign positions based on levels
            pos = {}
            level_nodes = {}
            for node, level in levels.items():
                if level not in level_nodes:
                    level_nodes[level] = []
                level_nodes[level].append(node)
            
            # Position nodes
            max_level = max(levels.values()) if levels else 0
            for level, nodes in level_nodes.items():
                y = max_level - level  # Reverse so top level is at top
                n_nodes = len(nodes)
                for i, node in enumerate(sorted(nodes)):  # Sort for consistency
                    x = i - (n_nodes - 1) / 2.0  # Center nodes in level
                    pos[node] = (x, y)
            
            # Scale positions
            if pos:
                xs = [p[0] for p in pos.values()]
                ys = [p[1] for p in pos.values()]
                x_range = max(xs) - min(xs) if max(xs) != min(xs) else 1.0
                y_range = max(ys) - min(ys) if max(ys) != min(ys) else 1.0
                
                # Normalize to reasonable range
                scale = 3.0
                for node in pos:
                    pos[node] = (
                        (pos[node][0] - min(xs)) / x_range * scale - scale/2,
                        (pos[node][1] - min(ys)) / y_range * scale - scale/2
                    )
            
            return pos
            
        except Exception as e:
            logger.warning(f"Hierarchical layout failed: {e}, using spring layout")
            return nx.spring_layout(G, k=1.5, iterations=50, seed=42)