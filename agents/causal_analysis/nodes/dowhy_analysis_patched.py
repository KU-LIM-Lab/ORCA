# nodes/dowhy_analysis.py

from typing import Dict, List
from langchain_core.runnables import RunnableLambda

import pandas as pd
import numpy as np
from dowhy import CausalModel
from tabpfn import TabPFNClassifier

import statsmodels.api as sm
from utils.redis_df import load_df_parquet

def clean_var_names(vars: List[str]) -> List[str]:
    return [v.split(".")[-1] if isinstance(v, str) and "." in v else v for v in vars]

def _convert_to_dot_graph(causal_graph: Dict) -> str:
    """
    Convert causal discovery graph format to DOT format for DoWhy.
    
    Expected input format:
    {
        "graph": {
            "edges": [{"from": "A", "to": "B", "type": "->", ...}, ...],
            "variables": ["A", "B", "C", ...]
        },
        "metadata": {...}
    }
    
    Returns DOT format string like:
    "digraph { A -> B; B -> C; }"
    """
    # Handle both unified schema and legacy format
    if "graph" in causal_graph:
        edges = causal_graph["graph"].get("edges", [])
        variables = causal_graph["graph"].get("variables", [])
    else:
        edges = causal_graph.get("edges", [])
        variables = causal_graph.get("variables", []) or causal_graph.get("nodes", [])
    
    # Check if we have dot_graph key (pre-existing DOT format)
    if "dot_graph" in causal_graph:
        return causal_graph["dot_graph"]
    
    if not edges:
        return None
    
    # Build DOT format string
    dot_lines = ["digraph {"]
    
    for edge in edges:
        from_node = str(edge.get("from", "")).replace(".", "_")
        to_node = str(edge.get("to", "")).replace(".", "_")
        edge_type = edge.get("type", "->")
        
        # Only include directed edges (->)
        if edge_type == "->":
            dot_lines.append(f"    {from_node} -> {to_node};")
        # For undirected edges (--), add both directions (or skip)
        # For now, we'll convert them to directed edges
        elif edge_type in ["--", "o-o"]:
            dot_lines.append(f"    {from_node} -> {to_node};")
    
    dot_lines.append("}")
    
    return "\n".join(dot_lines)

def _coerce_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert object columns to numeric or category codes to avoid statsmodels dtype errors."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            coerced = pd.to_numeric(df[col], errors="ignore")
            if coerced.dtype == object:
                df[col] = df[col].astype("category").cat.codes
            else:
                df[col] = coerced
    return df

def _extract_edges_and_vars(causal_graph: Dict):
    """Return (edges, variables) from either unified schema or legacy schema."""
    if causal_graph is None:
        return [], []
    if "graph" in causal_graph:
        edges = causal_graph["graph"].get("edges", []) or []
        variables = causal_graph["graph"].get("variables", []) or []
    else:
        edges = causal_graph.get("edges", []) or []
        variables = causal_graph.get("variables", []) or causal_graph.get("nodes", []) or []
    return edges, variables


def _has_directed_path(edges: List[Dict], src: str, dst: str) -> bool:
    """Directed reachability src -> ... -> dst following the same edge inclusion rules as _convert_to_dot_graph."""
    if src is None or dst is None:
        return False
    src = str(src)
    dst = str(dst)
    if src == dst:
        return True

    adj: Dict[str, List[str]] = {}
    for e in edges or []:
        f = str(e.get("from", ""))
        t = str(e.get("to", ""))
        et = e.get("type", "->")
        if not f or not t:
            continue
        if et == "->" or et in ["--", "o-o"]:
            adj.setdefault(f, []).append(t)

    seen = {src}
    stack = [src]
    while stack:
        u = stack.pop()
        for v in adj.get(u, []):
            if v == dst:
                return True
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return False


def build_dowhy_analysis_node() -> RunnableLambda:
    """
    Performs causal analysis using the selected strategy and preprocessed data.
    """

    def invoke(state: Dict) -> Dict:
        strategy = state["strategy"]
        parsed_info = state["parsed_query"]
        df = state.get("df_preprocessed")
        if df is None:
            redis_key = state.get("df_redis_key")
            if redis_key:
                df = load_df_parquet(redis_key)
        
        if df is None or df.shape[0] < 10:
            raise ValueError(f"Insufficient data for analysis: {df.shape[0]} rows found, at least 10 required.")

        if not strategy or not parsed_info:
            raise ValueError("Missing strategy or parsed query.")

        # Extract variables
        treatment = clean_var_names([parsed_info.get("treatment")])[0]
        outcome = clean_var_names([parsed_info.get("outcome")])[0]
        confounders = clean_var_names(parsed_info.get("confounders", []))
        mediators = clean_var_names(parsed_info.get("mediators", []))
        ivs = clean_var_names(parsed_info.get("instrumental_variables", []))


        # Create CausalModel
        causal_graph = state.get("selected_graph") or state.get("causal_graph")
        if not causal_graph:
            raise ValueError("The causal graph generated is required for DoWhy analysis")
        
        # Convert graph to DOT format for DoWhy
        dot_graph = _convert_to_dot_graph(causal_graph)

        # --- Pre-check: if there is no directed causal path, skip estimation ---
        edges, _ = _extract_edges_and_vars(causal_graph)

        # 1) Treatment -> Outcome must exist (otherwise effect is structurally zero under the learned DAG)
        if not _has_directed_path(edges, treatment, outcome):
            state["causal_effect_ate"] = 0.0
            state["causal_effect_ci"] = None
            state["causal_effect_note"] = (
                f"No directed path from [{treatment}] to [{outcome}] in the discovered graph; "
                "returned ATE=0.0 by fallback policy."
            )
            state["causal_model"] = None
            state["causal_estimand"] = None
            state["causal_estimate"] = {
                "value":  0.0,
                "method": "NOT_ESTIMATED",
            }
            return state

        # 2) If using IV, each instrument must have a directed path to the treatment
        if strategy.identification_method == "iv" and ivs:
            invalid_ivs = [iv for iv in ivs if not _has_directed_path(edges, iv, treatment)]
            if invalid_ivs:
                state["causal_effect_ate"] = 0.0
                state["causal_effect_ci"] = None
                state["causal_effect_note"] = (
                    f"Invalid IVs (no directed path to treatment [{treatment}]): {invalid_ivs}; "
                    "returned ATE=0.0 by fallback policy."
                )
                state["causal_model"] = None
                state["causal_estimand"] = None
                state["causal_estimate"] = {
                    "value":  0.0,
                    "method": "NOT_ESTIMATED",
                }
                return state

        df = _coerce_object_columns(df)

        
        # Identification
        try:
            if dot_graph:
                model = CausalModel(
                    data=df,
                    treatment=treatment,
                    outcome=outcome,
                    common_causes=confounders if strategy.identification_method == "backdoor" else None,
                    instruments=ivs if strategy.identification_method == "iv" else None,
                    mediators=mediators if strategy.identification_method == "mediation" else None,
                    graph=dot_graph
                )
            else:
                model = CausalModel(
                    data=df,
                    treatment=treatment,
                    outcome=outcome,
                    common_causes=confounders if strategy.identification_method == "backdoor" else None,
                    instruments=ivs if strategy.identification_method == "iv" else None,
                    mediators=mediators if strategy.identification_method == "mediation" else None,
                )
        except Exception as e:
            raise RuntimeError(f"Failed to construct CausalModel: {e}")
        
        identified_estimand = model.identify_effect()
        
        method_params = {}

        # Additional configuration for estimators
        if strategy.estimator == "backdoor.generalized_linear_model":
                y = df[outcome].dropna()
                is_binary = (
                    y.nunique() == 2 and
                    set(y.unique()).issubset({0, 1, True, False})
                )

                if is_binary:
                    method_params["glm_family"] = sm.families.Binomial()
                else:
                    method_params["glm_family"] = sm.families.Gaussian()

        classification_estimators = [
            "backdoor.propensity_score_matching",
            "backdoor.propensity_score_stratification",
            "backdoor.propensity_score_weighting",
            "backdoor.distance_matching"
        ]
        if strategy.estimator in classification_estimators:
            method_params["propensity_score_model"] = TabPFNClassifier()

            # Encode categorical columns
            cat_cols = df.select_dtypes(include="category").columns
            if len(cat_cols) > 0:
                label_maps = {}
                for col in cat_cols:
                    df[col], uniques = df[col].factorize()
                    label_maps[col] = dict(enumerate(uniques))
                state["label_maps"] = label_maps
        
        # Estimate causal effect
        try:
            estimate = model.estimate_effect(
                identified_estimand,
                method_name=strategy.estimator,
                method_params=method_params if method_params else None
            )
        except Exception as e:
            raise RuntimeError(f"Failed to estimate causal effect: {e}")

        # (Optional) refutation
        if strategy.refuter:
            refute_result = model.refute_estimate(
                identified_estimand,
                estimate,
                method_name=strategy.refuter[0],
            )
            state["refutation_result"] = refute_result.summary()

        # Save to state
        state["causal_model"] = str(model)
        state["causal_estimand"] = str(identified_estimand) if identified_estimand is not None else "NOT_IDENTIFIED"
        state["causal_estimate"] = {
            "value": float(estimate.value) if estimate is not None and getattr(estimate, "value", None) is not None else 0.0,
            "method": str(getattr(estimate, "method_name", None)) if estimate is not None and getattr(estimate, "method_name", None) is not None else "NOT_ESTIMATED",
        }
        
        # Save useful scalar values separately
        try:
            ci_raw = estimate.get_confidence_intervals()
            ci_arr = np.asarray(ci_raw, dtype=float)
            
            if ci_arr.ndim == 2 and ci_arr.shape[0] == 1:
                ci_arr = ci_arr[0]
                
            ci = ci_arr.tolist()
        except Exception:
            ci = None
        state["causal_effect_ate"] = float(getattr(estimate, "value", None)) if getattr(estimate, "value", None) is not None else None
        state["causal_effect_ci"] = ci

        # state.pop("df_preprocessed", None)
        return state

    return RunnableLambda(invoke)
