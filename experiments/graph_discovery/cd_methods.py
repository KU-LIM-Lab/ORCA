from __future__ import annotations

import json
import time
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, Any, List, Optional, Tuple

from agents.causal_discovery.tools import normalize_graph_result


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

from typing import Callable, Dict, Any, List, Tuple, Optional
import json
import re
import time

import numpy as np
import pandas as pd
import networkx as nx

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


# === 6. GPT-4o-mini (LLM-direct baseline) ===================================

def _safe_corr_topk(df: pd.DataFrame, top_k_pairs: int) -> List[Dict[str, Any]]:
    """Return top-K (undirected) correlation pairs by |corr|. O(d^2), OK for d<=100."""
    cols = list(df.columns)
    d = len(cols)
    C = df.corr(numeric_only=True).to_numpy()
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)

    pairs: List[Tuple[float, str, str, float]] = []
    for i in range(d):
        for j in range(i + 1, d):
            c = float(C[i, j])
            pairs.append((abs(c), cols[i], cols[j], c))
    pairs.sort(key=lambda x: x[0], reverse=True)
    pairs = pairs[: max(0, min(top_k_pairs, len(pairs)))]

    return [{"u": u, "v": v, "corr": c, "abs_corr": a} for (a, u, v, c) in pairs]


def _build_prompt(
    variables: List[str],
    top_pairs: List[Dict[str, Any]],
    max_candidates: int,
    max_edges: int,
) -> str:
    """
    LLM returns ONLY:
      - order: permutation of variables
      - candidates: undirected pairs (u, v) (subset of top_pairs + optional additions)
    We will orient u->v using the order (earlier -> later) and keep up to max_edges.
    """
    payload = {
        "role": "You are a causal discovery assistant.",
        "goal": "Propose (1) a plausible causal order over variables and (2) a small set of candidate dependency pairs.",
        "inputs": {
            "variables": variables,
            "top_correlation_pairs": top_pairs,
        },
        "hard_constraints": [
            "Return ONLY valid JSON (no markdown, no extra text).",
            "order must be a permutation of variables (same names, no missing, no duplicates).",
            f"candidates must contain at most {max_candidates} undirected pairs.",
            "Each candidate pair must be an object: {\"u\": <var>, \"v\": <var>} with u!=v and both in variables.",
            "No duplicate candidate pairs ignoring order (u,v) == (v,u).",
        ],
        "what_candidates_mean": [
            "candidates are NOT directed edges.",
            "They mean 'these two variables likely have a direct connection (in some direction)'.",
            "We will direct them later using the order (earlier causes later).",
        ],
        "selection_guidance": [
            "Prefer sparsity and precision: choose pairs that look most likely to be direct dependencies.",
            "Use top_correlation_pairs as evidence, but do NOT blindly include all highly correlated pairs.",
            "Avoid adding many redundant pairs that all connect to one variable unless strongly justified.",
            "If direction is ambiguous, that is fine: order will resolve direction later.",
            "If you are uncertain, choose fewer candidates rather than more.",
        ],
        "output_schema": {
            "order": ["V0", "V2", "..."],
            "candidates": [{"u": "V0", "v": "V3"}, {"u": "V2", "v": "V5"}],
        },
        "notes_on_limits": {
            "max_edges_final_dag": max_edges,
            "how_edges_are_formed": "We will orient each candidate pair from earlier->later in order, then keep up to max_edges using abs_corr from top_correlation_pairs as tie-breaker if needed.",
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def _parse_json(text: str) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    s, e = text.find("{"), text.rfind("}")
    if s >= 0 and e > s:
        try:
            obj = json.loads(text[s : e + 1])
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def _sanitize_order(order: Any, variables: List[str]) -> Optional[List[str]]:
    if not isinstance(order, list):
        return None
    order = [str(x).strip() for x in order]
    if len(order) != len(variables):
        return None
    if set(order) != set(variables):
        return None
    # keep as given (already a permutation)
    return order


def _sanitize_candidates(cands: Any, variables: List[str], max_candidates: int) -> List[Tuple[str, str]]:
    if not isinstance(cands, list):
        return []
    var_set = set(variables)
    seen = set()
    out: List[Tuple[str, str]] = []

    for item in cands:
        if not isinstance(item, dict):
            continue
        u = str(item.get("u", "")).strip()
        v = str(item.get("v", "")).strip()
        if u not in var_set or v not in var_set or u == v:
            continue
        key = tuple(sorted([u, v]))
        if key in seen:
            continue
        seen.add(key)
        out.append((key[0], key[1]))
        if len(out) >= max_candidates:
            break
    return out


def _orient_by_order(
    order: List[str],
    candidates: List[Tuple[str, str]],
    top_pairs: List[Dict[str, Any]],
    max_edges: int,
) -> List[Dict[str, Any]]:
    """
    Deterministic DAG construction:
      - orient each undirected pair by order index (earlier -> later)
      - rank edges by abs_corr if available (otherwise 0)
      - keep top max_edges
    """
    pos = {v: i for i, v in enumerate(order)}

    # map undirected pair -> abs_corr from sketch (if present)
    score = {}
    for p in top_pairs:
        u, v = str(p["u"]), str(p["v"])
        score[tuple(sorted([u, v]))] = float(p.get("abs_corr", 0.0))

    oriented = []
    for a, b in candidates:
        ua, ub = a, b
        if pos[ua] < pos[ub]:
            frm, to = ua, ub
        else:
            frm, to = ub, ua
        key = tuple(sorted([ua, ub]))
        oriented.append({"from": frm, "to": to, "_score": score.get(key, 0.0)})

    # dedup directed edges just in case
    seen = set()
    uniq = []
    for e in oriented:
        k = (e["from"], e["to"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(e)

    # keep strongest first
    uniq.sort(key=lambda x: x.get("_score", 0.0), reverse=True)
    uniq = uniq[: max_edges]

    # remove private field
    return [{"from": e["from"], "to": e["to"]} for e in uniq]


@register_method("gpt4o_mini")
def gpt4o_mini_method(X: np.ndarray, context: dict | None = None) -> dict:
    """
    LLM-direct baseline:
      - compute top correlation pairs (compact evidence)
      - ask LLM for (order, candidates)
      - orient candidates by order to get a DAG (acyclic by construction)
    """
    from utils.llm import call_llm 

    t0 = time.time()
    ctx = context or {}
    df = _context_df(X, ctx)
    variables = list(df.columns)
    d = len(variables)

    top_k_pairs = int(ctx.get("top_k_pairs", 2 * d))
    max_candidates = int(ctx.get("max_candidates", 3 * d))
    max_edges = int(ctx.get("max_edges", 2 * d))
    n_retries = int(ctx.get("n_retries", 1))
    model = ctx.get("model", "gpt-4o-mini")
    temperature = float(ctx.get("temperature", 0.2))
    llm_client = ctx.get("llm_client", None)

    top_pairs = _safe_corr_topk(df, top_k_pairs=top_k_pairs)
    prompt = _build_prompt(
        variables=variables,
        top_pairs=top_pairs,
        max_candidates=max_candidates,
        max_edges=max_edges,
    )

    exec_meta = {
        "top_k_pairs": top_k_pairs,
        "max_candidates": max_candidates,
        "max_edges": max_edges,
        "attempts": 0,
        "candidates_raw": 0,
        "candidates_kept": 0,
        "edges_final": 0,
    }

    last_text = None
    obj = None
    for attempt in range(n_retries + 1):
        exec_meta["attempts"] = attempt + 1
        try:
            if llm_client is not None:
                last_text = call_llm(prompt, llm=llm_client)
            else:
                last_text = call_llm(prompt, model=model, temperature=temperature)
            obj = _parse_json(last_text)
            if obj is not None:
                break
            obj = None
        except Exception as e:
            if attempt == n_retries:
                return {
                    "error": f"LLM call failed after {n_retries+1} attempts: {type(e).__name__}: {e}"
                }
            # Continue to next retry
            continue

    if obj is None:
        return {"error": f"Failed to parse JSON after {n_retries+1} attempts. preview={str(last_text)[:300]}"}

    order = _sanitize_order(obj.get("order"), variables)
    if order is None:
        return {"error": "Invalid 'order': must be a permutation of variables."}

    cands_raw = obj.get("candidates", [])
    exec_meta["candidates_raw"] = len(cands_raw) if isinstance(cands_raw, list) else 0

    cands = _sanitize_candidates(cands_raw, variables=variables, max_candidates=max_candidates)
    exec_meta["candidates_kept"] = len(cands)

    edges = _orient_by_order(order, cands, top_pairs=top_pairs, max_edges=max_edges)
    exec_meta["edges_final"] = len(edges)

    runtime = time.time() - t0
    params = {
        "backend": "gpt-4o-mini",
        "model": model,
        "temperature": temperature,
        "top_k_pairs": top_k_pairs,
        "max_candidates": max_candidates,
        "max_edges": max_edges,
    }

    result = normalize_graph_result(
        "GPT-4o-mini(order+candidates)",
        variables,
        edges,
        params=params,
        runtime=runtime,
        graph_type="DAG",
    )
    result["metadata"]["execution_metadata"] = exec_meta
    return result