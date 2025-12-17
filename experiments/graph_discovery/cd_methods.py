from __future__ import annotations

"""
Registry and thin wrappers for causal discovery baselines.

Each method follows a unified signature:

    fn(X: np.ndarray, context: dict | None = None) -> dict

Inputs
------
X : np.ndarray
    Data matrix of shape (n_samples, n_variables).
context : dict | None
    Optional extra information; for most classic algorithms only the
    original DataFrame (with column names) is required.

Outputs
-------
A standardized graph result dictionary that is compatible with the
ORCA causal discovery tools, i.e. the structure returned by
`agents.causal_discovery.tools.normalize_graph_result`:

    {
        "graph": {
            "variables": List[str],
            "edges": List[{"from": str, "to": str, "weight"?: float, "confidence"?: float}],
        },
        "metadata": {
            "method": str,
            "params": dict,
            "runtime": float | None,
            ...
        },
    }
"""

from typing import Callable, Dict, Any

import numpy as np
import pandas as pd

from agents.causal_discovery.tools import (
    normalize_graph_result,
    LiNGAMTool,
    ANMTool,
)

# --- Common registry ---------------------------------------------------------

METHOD_REGISTRY: Dict[str, Callable[[np.ndarray, dict | None], dict]] = {}


def register_method(name: str):
    """Decorator to register graph discovery methods by name."""

    def decorator(fn: Callable[[np.ndarray, dict | None], dict]):
        METHOD_REGISTRY[name] = fn
        return fn

    return decorator


def get_method(name: str) -> Callable[[np.ndarray, dict | None], dict]:
    if name not in METHOD_REGISTRY:
        raise ValueError(f"Unknown graph discovery method: {name}. Available: {list(METHOD_REGISTRY.keys())}")
    return METHOD_REGISTRY[name]


def _context_df(X: np.ndarray, context: dict | None) -> pd.DataFrame:
    """Helper: build a DataFrame with column names if not supplied."""
    context = context or {}
    df = context.get("df")
    if isinstance(df, pd.DataFrame) and df.shape == X.shape:
        return df
    # Fallback: create simple V0..V{d-1} columns
    n, d = X.shape
    cols = [f"V{i}" for i in range(d)]
    return pd.DataFrame(X, columns=cols)


# === 1. NOTEARS (linear) =====================================================

@register_method("notears_linear")
def notears_linear_method(
    X: np.ndarray,
    context: dict | None = None,
) -> dict:
    """
    NOTEARS linear model (Zheng et al., 2018) wrapper.

    Uses the original reference implementation bundled under
    `experiments/graph_discovery/methods/notears`.
    """
    from experiments.graph_discovery.methods.notears.notears.linear import notears_linear

    df = _context_df(X, context)
    lambda1 = (context or {}).get("lambda1", 0.1)
    loss_type = (context or {}).get("loss_type", "l2")
    max_iter = (context or {}).get("max_iter", 100)
    h_tol = (context or {}).get("h_tol", 1e-8)
    rho_max = (context or {}).get("rho_max", 1e16)
    w_threshold = (context or {}).get("w_threshold", 0.3)

    W_est = notears_linear(
        df.values,
        lambda1=lambda1,
        loss_type=loss_type,
        max_iter=max_iter,
        h_tol=h_tol,
        rho_max=rho_max,
        w_threshold=w_threshold,
    )

    vars_ = list(df.columns)
    edges = []
    d = W_est.shape[0]
    for i in range(d):
        for j in range(d):
            w = float(W_est[i, j])
            if w != 0.0:
                edges.append({"from": vars_[i], "to": vars_[j], "weight": w})

    params = {
        "backend": "notears",
        "lambda1": lambda1,
        "loss_type": loss_type,
        "max_iter": max_iter,
        "h_tol": h_tol,
        "rho_max": rho_max,
        "w_threshold": w_threshold,
    }
    # notears_linear does not expose runtime; keep as None
    return normalize_graph_result("NOTEARS-linear", vars_, edges, params=params, runtime=None)


# === 2. NOTEARS (nonlinear MLP / Sobolev) ====================================


@register_method("notears_nonlinear")
def notears_nonlinear_method(
    X: np.ndarray,
    context: dict | None = None,
) -> dict:
    """
    NOTEARS nonlinear (MLP-based) wrapper.

    This uses the MLP formulation from `notears.nonlinear` with default
    architecture dims=[d, 10, 1] unless overridden via context.
    """
    import torch
    from pathlib import Path
    import sys

    this_file = Path(__file__).resolve()
    methods_dir = (this_file.parent / "methods/notears").resolve()
    if str(methods_dir) not in sys.path:
        sys.path.insert(0, str(methods_dir))

    from notears.nonlinear import NotearsMLP, notears_nonlinear

    df = _context_df(X, context)
    d = df.shape[1]
    ctx = context or {}

    dims = ctx.get("dims", [d, 10, 1])
    lambda1 = ctx.get("lambda1", 0.01)
    lambda2 = ctx.get("lambda2", 0.01)
    max_iter = ctx.get("max_iter", 100)
    h_tol = ctx.get("h_tol", 1e-8)
    rho_max = ctx.get("rho_max", 1e16)
    w_threshold = ctx.get("w_threshold", 0.3)

    torch.set_default_dtype(torch.double)
    model = NotearsMLP(dims=dims, bias=True)

    W_est = notears_nonlinear(
        model,
        df.values,
        lambda1=lambda1,
        lambda2=lambda2,
        max_iter=max_iter,
        h_tol=h_tol,
        rho_max=rho_max,
        w_threshold=w_threshold,
    )

    vars_ = list(df.columns)
    edges = []
    for i in range(d):
        for j in range(d):
            w = float(W_est[i, j])
            if w != 0.0:
                edges.append({"from": vars_[i], "to": vars_[j], "weight": w})

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
    return normalize_graph_result("NOTEARS-nonlinear", vars_, edges, params=params, runtime=None)


# === 3. AutoCD ===============================================================


@register_method("autocd")
def autocd_method(
    X: np.ndarray,
    context: dict | None = None,
) -> dict:
    """
    AutoCD causal discovery wrapper.

    AutoCD is a multi-stage pipeline; for this benchmarking method we
    rely on its core causal discovery function and return a DAG over
    the observed variables.

    NOTE: This wrapper assumes the AutoCD package within
    `experiments/graph_discovery/methods/AutoCD` is importable as
    `AutoCD.*`. If imports fail, the method returns an error dict.
    """
    ctx = context or {}
    df = _context_df(X, ctx)
    vars_ = list(df.columns)

    try:
        # Use ORCA-specific AutoCD entrypoint that wraps the original
        # AutoCD pipeline (data_object → CausalConfigurator → OCT).
        from experiments.graph_discovery.methods.autocd_entry import autocd_entry
    except Exception as e:  # pragma: no cover - defensive import
        return {
            "error": f"AutoCD not available or import failed: {type(e).__name__}: {e}"
        }

    try:
        # The ORCA entrypoint already handles both DataFrame and ndarray.
        cd_result = autocd_entry(df)

        edges: list[dict[str, Any]] = []
        if isinstance(cd_result, dict):
            # e.g., {"adjacency": np.ndarray} or {"edges": [...]} style
            if "edges" in cd_result:
                for e in cd_result["edges"]:
                    if isinstance(e, dict) and "from" in e and "to" in e:
                        edges.append(
                            {
                                "from": str(e["from"]),
                                "to": str(e["to"]),
                                **{k: v for k, v in e.items() if k not in {"from", "to"}},
                            }
                        )
                    elif isinstance(e, (list, tuple)) and len(e) >= 2:
                        edges.append({"from": str(e[0]), "to": str(e[1])})
            elif "adjacency" in cd_result:
                A = np.asarray(cd_result["adjacency"])
                d = A.shape[0]
                for i in range(d):
                    for j in range(d):
                        if A[i, j] != 0:
                            edges.append(
                                {
                                    "from": vars_[i],
                                    "to": vars_[j],
                                    "weight": float(A[i, j]),
                                }
                            )
        elif isinstance(cd_result, np.ndarray):
            A = np.asarray(cd_result)
            d = A.shape[0]
            for i in range(d):
                for j in range(d):
                    if A[i, j] != 0:
                        edges.append(
                            {
                                "from": vars_[i],
                                "to": vars_[j],
                                "weight": float(A[i, j]),
                            }
                        )
        else:
            return {
                "error": f"Unsupported AutoCD result type: {type(cd_result)}"
            }

        params = {
            "backend": "AutoCD",
        }
        return normalize_graph_result("AutoCD", vars_, edges, params=params, runtime=None)
    except Exception as e:  # pragma: no cover - robustness
        return {
            "error": f"AutoCD causal discovery failed: {type(e).__name__}: {e}"
        }


# === 4. LiNGAM / ANM (from ORCA tools) =======================================


@register_method("lingam")
def lingam_method(
    X: np.ndarray,
    context: dict | None = None,
) -> dict:
    """
    DirectLiNGAM wrapper using ORCA's `LiNGAMTool`.
    """
    df = _context_df(X, context)
    return LiNGAMTool.direct_lingam(df)


@register_method("anm")
def anm_method(
    X: np.ndarray,
    context: dict | None = None,
) -> dict:
    """
    ANM discovery wrapper using ORCA's `ANMTool`.
    """
    df = _context_df(X, context)
    delta = (context or {}).get("delta", 0.02)
    tau = (context or {}).get("tau", 0.05)
    return ANMTool.anm_discovery(df, delta=delta, tau=tau)


# === 5. ORCA Causal Discovery Agent ==========================================

@register_method("orca")
def orca_method(
    X: np.ndarray,
    context: dict | None = None,
) -> dict:
    """
    Run the full ORCA CausalDiscoveryAgent pipeline and return the final DAG.

    This mirrors the logic of `tests/run_causal_discovery.py`:
    - run data preprocessing (schema detection, clean_nulls)
    - run discovery substeps up to ensemble_synthesis
    - return `selected_graph` as a standard graph result dict
    """
    from agents.causal_discovery.agent import CausalDiscoveryAgent
    from agents.data_explorer.data_preprocessor.agent import DataPreprocessorAgent
    from utils.settings import CONFIG

    ctx = context or {}
    df_input = _context_df(X, ctx)

    # 1. Data preprocessing (schema detection + clean_nulls)
    preprocessor = DataPreprocessorAgent()
    preprocessor.df = df_input.copy()
    preprocessor._data_fetched = True  # type: ignore[attr-defined]

    prep_state: Dict[str, Any] = {
        "df_preprocessed": df_input.copy(),
        "db_id": ctx.get("db_id", "graph_discovery"),
        "skip_one_hot_encoding": True,
    }

    prep_state["current_substep"] = "schema_detection"
    prep_state = preprocessor.step(prep_state)
    if prep_state.get("error"):
        return {
            "error": f"Schema detection failed: {prep_state.get('error')}"
        }

    prep_state["current_substep"] = "clean_nulls"
    prep_state = preprocessor.step(prep_state)
    if prep_state.get("error"):
        return {
            "error": f"Clean nulls failed: {prep_state.get('error')}"
        }

    variable_schema = prep_state.get("variable_schema", {})
    df_proc = preprocessor.df if preprocessor.df is not None else df_input

    cd_config = (CONFIG.get("agents", {}) or {}).get("causal_discovery", {}) or {}
    cd_config = cd_config.copy()
    if "bootstrap_iterations" not in cd_config:
        cd_config["bootstrap_iterations"] = ctx.get("bootstrap_iterations", 10)

    agent = CausalDiscoveryAgent(config=cd_config)

    state: Dict[str, Any] = {
        "df_preprocessed": df_proc,
        "variable_schema": variable_schema,
        "db_id": ctx.get("db_id", "graph_discovery"),
    }

    substeps = [
        "data_profiling",
        "algorithm_configuration",
        "run_algorithms_portfolio",
        "graph_scoring",
        "graph_evaluation",
        "ensemble_synthesis",
    ]

    for sub in substeps:
        state["current_substep"] = sub
        state = agent.step(state)
        if state.get("error"):
            return {"error": state["error"]}

    dag = state.get("selected_graph")
    if not isinstance(dag, dict):
        return {"error": "CausalDiscoveryAgent did not produce a DAG in 'selected_graph'"}
    return dag

