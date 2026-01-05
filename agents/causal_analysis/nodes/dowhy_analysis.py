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
        state["causal_estimand"] = str(identified_estimand)
        state["causal_estimate"] = {
            "value": float(getattr(estimate, "value", None)) if getattr(estimate, "value", None) is not None else None,
            "method": getattr(estimate, "method_name", None),
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

        state.pop("df_preprocessed", None)
        return state

    return RunnableLambda(invoke)
