# experiments/run_experiment.py

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, Any, Tuple, List
import time

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from experiments.effect_estimation.methods import get_method, METHOD_REGISTRY
from experiments.graph_discovery import cd_methods

# ORCA LLM / graph
from agents.causal_analysis.graph import generate_causal_analysis_graph
from utils.llm import get_llm
import networkx as nx
import re


# === 1. Data loaders ===

def fetch_df_from_db(sql_query: str, db_url: str) -> pd.DataFrame:
    import psycopg2
    df = None
    conn = psycopg2.connect(db_url)
    try:
        df = pd.read_sql(sql_query, conn)
    finally:
        conn.close()
    return df

def load_ihdp_dataset(
    data_path: str,
    replicate_idx: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, Dict[str, Any]]:
    """
    IHDP NP-CI data loader (train set).
    
    Data structure:
    - x: (672, 25, 1000) - 672 samples, 25 features, 1000 replications
    - t: (672, 1000) - treatment assignments
    - yf: (672, 1000) - factual outcomes
    - mu0, mu1: (672, 1000) - potential outcomes
    
    replicate_idx selects which replication (0-999) to use.
    """
    npz = np.load(data_path, allow_pickle=True)
    
    # Extract data for the specified replication
    # x: (672, 25, 1000) -> (672, 25) for replicate_idx
    X = npz["x"][:, :, replicate_idx]  # (n, d)
    T = npz["t"][:, replicate_idx].astype(int)  # (n,)
    yf = npz["yf"][:, replicate_idx]  # (n,)
    mu0 = npz["mu0"][:, replicate_idx]  # (n,)
    mu1 = npz["mu1"][:, replicate_idx]  # (n,)
    
    # Calculate ITE: mu1 - mu0
    ite = mu1 - mu0  # (n,)
    
    tau_true_cate = ite
    tau_true_ate = float(ite.mean())
    Y = yf  # 여기서는 factual outcome으로 학습 (실험 설계에 따라 조정 가능)

    meta = {
        "dataset": "IHDP",
        "replicate_idx": replicate_idx,
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "n_replications": npz["x"].shape[2] if len(npz["x"].shape) == 3 else 1,
        "tau_true_cate": tau_true_cate,
        "tau_true_ate": tau_true_ate,
        "tau_true_ate_global": float(npz["ate"]) if "ate" in npz.files else tau_true_ate,
    }
    return X, T, Y, tau_true_cate, tau_true_ate, meta


def get_available_runs(scenario_dir: Path) -> List[int]:
    """
    Get list of available run indices for a scenario directory.
    Returns sorted list of run indices (e.g., [0, 1, 2, ...]).
    """
    available_runs = []
    for run_dir in scenario_dir.iterdir():
        if run_dir.is_dir() and run_dir.name.startswith("run_"):
            try:
                run_idx = int(run_dir.name.split("_")[1])
                data_path = run_dir / "data.csv"
                if data_path.exists():
                    available_runs.append(run_idx)
            except (ValueError, IndexError):
                continue
    return sorted(available_runs)


def load_synthetic_ci_dataset(
    scenario_dir: Path,
    run_idx: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, Dict[str, Any]]:
    """
    Synthetic CI scenario data loader.

    data.csv: [W*, T, Y, propensity, tau_true, mu0, mu1]
    stats.json: {"ATE_true": ..., ...}
    """
    run_dir = scenario_dir / f"run_{run_idx:03d}"
    data_path = run_dir / "data.csv"
    stats_path = run_dir / "stats.json"
    config_path = run_dir / "config.json"

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)

    # covariates are columns starting with W*
    w_cols = [c for c in df.columns if c.startswith("W")]
    X = df[w_cols].values
    T = df["T"].values.astype(int)
    Y = df["Y"].values
    tau_true_cate = df["tau_true"].values

    if stats_path.exists():
        with open(stats_path, "r") as f:
            stats = json.load(f)
        tau_true_ate = float(stats.get("ATE_true", float(tau_true_cate.mean())))
    else:
        tau_true_ate = float(tau_true_cate.mean())
        stats = {}

    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        config = {}

    meta = {
        "dataset": "synthetic_ci",
        "scenario": scenario_dir.name,
        "run_idx": run_idx,
        "df": df,
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "stats": stats,
        "config": config,
        "w_cols": w_cols,
    }
    return X, T, Y, tau_true_cate, tau_true_ate, meta


# === 2. ORCA용 causal_graph 생성 (Oracle vs Agent) ===

def create_oracle_causal_graph(df: pd.DataFrame, treatment: str, outcome: str) -> Dict[str, Any]:
    """
    Synthetic CI oracle graph example:
    구조: W -> T, W -> Y, T -> Y
    """
    confounders = [c for c in df.columns if c.startswith("W")]

    edges = []
    for w in confounders:
        edges.append({"from": w, "to": treatment})
        edges.append({"from": w, "to": outcome})
    edges.append({"from": treatment, "to": outcome})

    G = nx.DiGraph()
    G.add_nodes_from(confounders + [treatment, outcome])
    for e in edges:
        G.add_edge(e["from"], e["to"])

    return {
        "nodes": confounders + [treatment, outcome],
        "edges": edges,
        "variables": confounders + [treatment, outcome],
        "graph": {"variables": confounders + [treatment, outcome], "edges": edges},
        "nx_graph": G,
    }


# === 3. Metric calculator ===

def compute_metrics(
    tau_hat_ate: float | None,
    tau_hat_cate: np.ndarray | None,
    tau_true_ate: float,
    tau_true_cate: np.ndarray | None,
    ate_ci: Tuple[float, float] | None = None,
) -> Dict[str, Any]:
    """Calculate ATE / CATE accuracy and CI-related metrics."""
    metrics: Dict[str, Any] = {}

    # ATE metrics
    if tau_hat_ate is not None:
        bias = float(tau_hat_ate - tau_true_ate)
        abs_err = float(abs(bias))
        sq_err = float(bias**2)
        metrics.update(
            {
                "ate_hat": float(tau_hat_ate),
                "ate_true": float(tau_true_ate),
                "ate_bias": bias,
                "ate_abs_error": abs_err,
                "ate_sq_error": sq_err,
            }
        )
    else:
        metrics.update(
            {
                "ate_hat": None,
                "ate_true": float(tau_true_ate),
                "ate_bias": None,
                "ate_abs_error": None,
                "ate_sq_error": None,
            }
        )

    # CI metrics
    if ate_ci is not None and tau_hat_ate is not None:
        l, u = ate_ci
        covered = (l <= tau_true_ate <= u)
        width = float(u - l)
        metrics.update(
            {
                "ate_ci_lower": float(l),
                "ate_ci_upper": float(u),
                "ate_ci_covered": bool(covered),
                "ate_ci_width": width,
            }
        )
    else:
        metrics.update(
            {
                "ate_ci_lower": None,
                "ate_ci_upper": None,
                "ate_ci_covered": None,
                "ate_ci_width": None,
            }
        )

    # CATE metrics
    if tau_true_cate is not None and tau_hat_cate is not None:
        tau_true_cate = np.asarray(tau_true_cate)
        tau_hat_cate = np.asarray(tau_hat_cate)
        diff = tau_hat_cate - tau_true_cate
        mse = float(np.mean(diff**2))
        pehe = float(np.sqrt(mse))
        metrics.update(
            {
                "cate_mse": mse,
                "cate_pehe": pehe,
            }
        )
    else:
        metrics.update(
            {
                "cate_mse": None,
                "cate_pehe": None,
            }
        )

    # TODO: AUUC, refutation 등은 여기 추가
    metrics["auuc"] = None

    return metrics


# === 4. Main experiment loop ===

def _sanitize_name(name: str | int | None) -> str:
    if name is None:
        return "none"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name))


def _normalize_discovery_graph(
    discovery_result: Dict[str, Any],
    fallback_nodes: List[str],
) -> Dict[str, Any]:
    """
    Normalize discovery output into causal_graph format expected by ORCA causal analysis.
    """
    if discovery_result is None:
        raise ValueError("Discovery result is None.")

    if "graph" in discovery_result and isinstance(discovery_result["graph"], dict):
        graph = discovery_result["graph"]
        variables = graph.get("variables") or discovery_result.get("variables") or discovery_result.get("nodes") or fallback_nodes
        edges = graph.get("edges") or discovery_result.get("edges") or []
    else:
        variables = discovery_result.get("variables") or discovery_result.get("nodes") or fallback_nodes
        edges = discovery_result.get("edges") or []

    nodes = discovery_result.get("nodes") or variables

    nx_graph = discovery_result.get("nx_graph")
    if nx_graph is None:
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        for edge in edges:
            if isinstance(edge, dict):
                from_node = str(edge.get("from", ""))
                to_node = str(edge.get("to", ""))
            elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
                from_node = str(edge[0])
                to_node = str(edge[1])
            else:
                continue
            if from_node in nodes and to_node in nodes:
                G.add_edge(from_node, to_node)
        nx_graph = G

    causal_graph = {
        "nodes": list(nodes),
        "edges": edges,
        "variables": list(variables),
        "graph": {"variables": list(variables), "edges": edges},
        "nx_graph": nx_graph,
    }

    if "dot_graph" in discovery_result:
        causal_graph["dot_graph"] = discovery_result["dot_graph"]
    if "metadata" in discovery_result:
        causal_graph["metadata"] = discovery_result["metadata"]

    return causal_graph


def _discovery_cache_path(
    discovery_result_dir: Path,
    dataset: str,
    scenario: str | int | None,
    run_id: int,
    discovery_method: str,
) -> Path:
    safe_dataset = _sanitize_name(dataset)
    safe_scenario = _sanitize_name(scenario)
    safe_method = _sanitize_name(discovery_method)
    return (
        discovery_result_dir
        / safe_dataset
        / safe_scenario
        / f"run_{run_id:03d}"
        / f"{safe_method}.json"
    )


def _serialize_causal_graph(causal_graph: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "nodes": causal_graph.get("nodes"),
        "edges": causal_graph.get("edges"),
        "variables": causal_graph.get("variables"),
        "graph": causal_graph.get("graph"),
    }
    if "metadata" in causal_graph:
        payload["metadata"] = causal_graph["metadata"]
    if "dot_graph" in causal_graph:
        payload["dot_graph"] = causal_graph["dot_graph"]
    return payload


def _run_causal_discovery(
    df: pd.DataFrame,
    discovery_method: str,
    discovery_context: Dict[str, Any] | None = None,
    discovery_result_dir: Path | None = None,
    dataset: str | None = None,
    scenario: str | int | None = None,
    run_id: int | None = None,
) -> Tuple[Dict[str, Any] | None, str | None]:
    """
    Run causal discovery (or load cached result) and return a causal_graph dict.
    """
    discovery_context = discovery_context or {}
    cache_path = None

    if discovery_result_dir is not None and dataset is not None and run_id is not None:
        cache_path = _discovery_cache_path(
            discovery_result_dir,
            dataset=dataset,
            scenario=scenario,
            run_id=run_id,
            discovery_method=discovery_method,
        )
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text())
                causal_graph = _normalize_discovery_graph(cached, list(df.columns))
                return causal_graph, None
            except Exception as e:
                cache_error = f"Failed to load cached discovery graph: {e}"
                tqdm.write(f"[WARNING] {cache_error}")

    try:
        method_fn = cd_methods.get_method(discovery_method)
    except Exception as e:
        return None, f"Unknown discovery method '{discovery_method}': {e}"

    try:
        discovery_result = method_fn(df.values, context={"df": df, **discovery_context})
    except Exception as e:
        return None, f"Discovery method '{discovery_method}' failed: {e}"

    if isinstance(discovery_result, dict) and discovery_result.get("error"):
        return None, str(discovery_result.get("error"))

    try:
        causal_graph = _normalize_discovery_graph(discovery_result, list(df.columns))
    except Exception as e:
        return None, f"Failed to normalize discovery graph: {e}"

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = _serialize_causal_graph(causal_graph)
        try:
            with open(cache_path, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            pass

    return causal_graph, None


def _save_summary_intermediate(all_records: List[Dict[str, Any]], results_dir: Path, 
                               dataset: str, setting: str) -> None:
    """Save intermediate summary from current all_records."""
    if not all_records:
        return
    
    try:
        df = pd.DataFrame(all_records)
        group_cols = ["dataset", "scenario", "setting", "method"]
        agg_dict = {
            "ate_abs_error": ["mean", "std"],
            "ate_sq_error": ["mean", "std"],
            "ate_bias": ["mean", "std"],
            "ate_ci_covered": ["mean"],
            "cate_pehe": ["mean", "std"],
            "cate_mse": ["mean", "std"],
        }
        
        # Filter columns that exist in the dataframe
        available_cols = df.columns.tolist()
        filtered_agg_dict = {
            col: funcs for col, funcs in agg_dict.items()
            if col in available_cols
        }
        
        if not filtered_agg_dict:
            return
        
        summary = df.groupby(group_cols).agg(filtered_agg_dict)
        summary.columns = ["_".join([c for c in col if c]) for col in summary.columns.values]
        summary = summary.reset_index()
        
        # Add additional performance metrics
        # 1. Coverage rate as percentage
        if "ate_ci_covered_mean" in summary.columns:
            summary["ate_ci_covered_percent"] = summary["ate_ci_covered_mean"] * 100
        
        # 2. RMSE (Root Mean Squared Error) = sqrt(MSE)
        if "ate_sq_error_mean" in summary.columns:
            summary["ate_rmse"] = np.sqrt(summary["ate_sq_error_mean"])
        
        # 3. Absolute bias (mean of absolute bias values)
        if "ate_bias_mean" in summary.columns:
            summary["ate_abs_bias"] = np.abs(summary["ate_bias_mean"])
        
        # 4. CI width statistics (if available)
        if "ate_ci_width" in df.columns:
            ci_width_agg = df.groupby(group_cols)["ate_ci_width"].agg(["mean", "std"])
            summary = summary.merge(ci_width_agg, left_on=group_cols, right_index=True, how="left")
            summary = summary.rename(columns={"mean": "ate_ci_width_mean", "std": "ate_ci_width_std"})
        
        summary_dir = results_dir / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summary_dir / f"summary_{dataset}_{setting}.csv"
        
        summary.to_csv(summary_path, index=False)
        print(f"[INFO] Intermediate summary saved: {summary_path} ({len(all_records)} records)")
    except Exception as e:
        print(f"[WARNING] Failed to save intermediate summary: {e}")


def run_experiments(
    dataset: str,
    methods: List[str],
    setting: str,
    runs: int | None = None,
    run_indices: List[int] | None = None,
    results_dir: Path | None = None,
    ihdp_path: str | None = None,
    synthetic_base_dir: str | None = None,
    scenarios: List[str] | None = None,
    discovery_method: str | None = None,
    discovery_context: Dict[str, Any] | None = None,
    discovery_result_dir: Path | None = None,
    seed: int = 0,
    save_interval: int = 50,
):
    if results_dir is None:
        results_dir = Path("experiments/results/effect_estimation")
    results_dir.mkdir(parents=True, exist_ok=True)
    all_records: List[Dict[str, Any]] = []
    
    orca_app = None
    
    if "orca" in methods:
        llm = get_llm(model="gpt-4o-mini", temperature=0.7, provider="openai")
        orca_app = generate_causal_analysis_graph(llm=llm)

    # Random seed
    np.random.seed(seed)

    if dataset.lower() == "ihdp":
        if ihdp_path is None:
            raise ValueError("IHDP dataset path must be provided for dataset='ihdp'.")

        # Check number of replications available
        npz = np.load(ihdp_path, allow_pickle=True)
        n_replications = npz["x"].shape[2] if len(npz["x"].shape) == 3 else 1
        npz.close()
        
        # Determine which replication indices to use
        if run_indices is not None:
            # Use specified run indices
            replication_indices = [idx for idx in run_indices if 0 <= idx < n_replications]
            if len(replication_indices) != len(run_indices):
                missing = [idx for idx in run_indices if idx not in replication_indices]
                print(f"Warning: Some replication indices are out of range (0-{n_replications-1}): {missing}")
        elif runs is not None and runs > 0:
            # Use first N runs
            replication_indices = list(range(min(runs, n_replications)))
        else:
            # Use all available replications
            replication_indices = list(range(n_replications))
        
        if not replication_indices:
            raise ValueError(f"No valid replication indices found. Available: 0-{n_replications-1}")
        
        print(f"Running experiments on IHDP with methods={methods}, setting={setting}, replications={replication_indices} (out of {n_replications} available)")
        
        # Outer progress bar for replications
        replication_pbar = tqdm(
            replication_indices,
            desc="Replications",
            unit="rep",
            position=0,
            leave=True,
            ncols=100
        )
        
        for run_id in replication_pbar:
            replication_pbar.set_description(f"Replication {run_id}")
            
            try:
                X, T, Y, tau_true_cate, tau_true_ate, meta = load_ihdp_dataset(ihdp_path, replicate_idx=run_id)
            except Exception as e:
                tqdm.write(f"[ERROR] Failed to load IHDP dataset for replication {run_id}: {e}")
                continue

            # ORCA df / causal_graph (Oracle setting uses all confounders)
            df = pd.DataFrame(X, columns=[f"W{i}" for i in range(X.shape[1])])
            df["T"] = T
            df["Y"] = Y

            if setting == "oracle_graph":
                causal_graph = create_oracle_causal_graph(df, treatment="T", outcome="Y")
            elif setting == "agent_graph":
                causal_graph = create_oracle_causal_graph(df, treatment="T", outcome="Y")
            else:
                raise ValueError(f"Unknown setting: {setting}")

            # Inner progress bar for methods
            methods_pbar = tqdm(
                methods,
                desc=f"  Methods (rep {run_id})",
                unit="method",
                position=1,
                leave=False,
                ncols=100
            )
            
            for method_name in methods_pbar:
                methods_pbar.set_description(f"  {method_name} (rep {run_id})")
                
                if method_name not in METHOD_REGISTRY:
                    tqdm.write(f"Skipping unknown method: {method_name}")
                    continue
                method_fn = get_method(method_name)

                context: Dict[str, Any] = {
                    "seed": seed + run_id,
                }
                if method_name == "orca":
                    if setting == "agent_graph" and causal_graph is None:
                        tqdm.write(
                            f"  [ERROR] Missing discovery graph for {method_name} (rep {run_id}). Skipping."
                        )
                        continue
                    context.update(
                        {
                            "df": df,
                            "causal_graph": causal_graph,
                            "app": orca_app,
                            "treatment_name": "T",
                            "outcome_name": "Y",
                        }
                    )

                try:
                    result = method_fn(X, T, Y, context=context)
                except Exception as e:
                    tqdm.write(f"  [ERROR] Method {method_name} failed for replication {run_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Continue with next method
                    continue
                metrics = compute_metrics(
                    tau_hat_ate=result.get("tau_hat_ate"),
                    tau_hat_cate=result.get("tau_hat_cate"),
                    tau_true_ate=tau_true_ate,
                    tau_true_cate=tau_true_cate,
                    ate_ci=result.get("ate_ci"),
                )

                run_record = {
                    "dataset": "IHDP",
                    "scenario": None,
                    "setting": setting,
                    "method": method_name,
                    "run_id(replication_idx)": run_id,
                    "n_samples": meta["n_samples"],
                    "n_features": meta["n_features"],
                    "metrics": metrics,
                }

                # Organize by dataset / setting / run_id / method_name.json
                out_dir = results_dir / "IHDP" / setting / f"run_{run_id:03d}"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{method_name}.json"
                try:
                    with open(out_path, "w") as f:
                        json.dump(run_record, f, indent=2)
                except Exception as e:
                    tqdm.write(f"  [ERROR] Failed to save results for {method_name} replication {run_id}: {e}")

                # Create flat record for summary
                flat = {
                    "dataset": "IHDP",
                    "scenario": None,
                    "setting": setting,
                    "method": method_name,
                    "run_id(replication_idx)": run_id,
                    "replicate_idx": meta.get("replicate_idx", run_id),
                    **metrics,
                }
                all_records.append(flat)
                
                # Save intermediate summary periodically
                if save_interval > 0 and len(all_records) % save_interval == 0:
                    _save_summary_intermediate(all_records, results_dir, dataset.lower(), setting)

    elif dataset.lower() == "synthetic_ci":
        if synthetic_base_dir is None:
            raise ValueError("synthetic_base_dir must be provided for dataset='synthetic_ci'.")

        base_dir = Path(synthetic_base_dir)
        if scenarios is None or len(scenarios) == 0:
            scenario_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()])
        else:
            scenario_dirs = [base_dir / s for s in scenarios]

        print(f"Running experiments on Synthetic CI scenarios={ [d.name for d in scenario_dirs] }, methods={methods}, setting={setting}")

        # Outer progress bar for scenarios
        scenario_pbar = tqdm(
            scenario_dirs,
            desc="Scenarios",
            unit="scenario",
            position=0,
            leave=True,
            ncols=100
        )
        
        for scenario_dir in scenario_pbar:
            scenario_pbar.set_description(f"Scenario: {scenario_dir.name}")
            
            # Get available runs for this scenario
            available_runs = get_available_runs(scenario_dir)
            
            if not available_runs:
                tqdm.write(f"Warning: No valid run directories found in {scenario_dir}. Skipping.")
                continue
            
            # Determine which run indices to use
            if run_indices is not None:
                # Use specified run indices (filter to only those that exist)
                scenario_run_indices = [idx for idx in run_indices if idx in available_runs]
                if len(scenario_run_indices) != len(run_indices):
                    missing = [idx for idx in run_indices if idx not in scenario_run_indices]
                    tqdm.write(f"Warning: Some run indices not found in {scenario_dir.name}: {missing}")
            elif runs is not None and runs > 0:
                # Use first N runs
                scenario_run_indices = available_runs[:min(runs, len(available_runs))]
            else:
                # Use all available runs
                scenario_run_indices = available_runs
            
            if not scenario_run_indices:
                tqdm.write(f"Warning: No valid run indices for {scenario_dir.name}. Skipping.")
                continue
            
            # Middle progress bar for runs
            runs_pbar = tqdm(
                scenario_run_indices,
                desc=f"  Runs ({scenario_dir.name})",
                unit="run",
                position=1,
                leave=False,
                ncols=100
            )
            
            for run_id in runs_pbar:
                runs_pbar.set_description(f"  Run {run_id} ({scenario_dir.name})")
                
                try:
                    X, T, Y, tau_true_cate, tau_true_ate, meta = load_synthetic_ci_dataset(
                        scenario_dir, run_idx=run_id
                    )
                except Exception as e:
                    tqdm.write(f"[ERROR] Failed to load synthetic CI dataset for scenario {scenario_dir.name}, run {run_id}: {e}")
                    continue
                    
                df = meta["df"]
                
                if setting == "oracle_graph":
                    causal_graph = create_oracle_causal_graph(df, treatment="T", outcome="Y")
                elif setting == "agent_graph":
                    causal_graph = create_oracle_causal_graph(df, treatment="T", outcome="Y")
                else:
                    raise ValueError(f"Unknown setting: {setting}")

                # Inner progress bar for methods
                methods_pbar = tqdm(
                    methods,
                    desc=f"    Methods (run {run_id})",
                    unit="method",
                    position=2,
                    leave=False,
                    ncols=100
                )
                
                for method_name in methods_pbar:
                    methods_pbar.set_description(f"    {method_name} (run {run_id})")
                    
                    if method_name not in METHOD_REGISTRY:
                        tqdm.write(f"Skipping unknown method: {method_name}")
                        continue
                    method_fn = get_method(method_name)

                    context: Dict[str, Any] = {
                        "seed": seed + run_id,
                    }
                    if method_name == "orca":
                        if setting == "agent_graph" and causal_graph is None:
                            tqdm.write(
                                f"    [ERROR] Missing discovery graph for {method_name} (run {run_id}). Skipping."
                            )
                            continue
                        context.update(
                            {
                                "df": df,
                                "causal_graph": causal_graph,
                                "app": orca_app,
                                "treatment_name": "T",
                                "outcome_name": "Y",
                            }
                        )

                    try:
                        result = method_fn(X, T, Y, context=context)
                    except Exception as e:
                        tqdm.write(f"    [ERROR] Method {method_name} failed for scenario {scenario_dir.name}, run {run_id}: {e}")
                        import traceback
                        traceback.print_exc()
                        # Continue with next method
                        continue
                        
                    metrics = compute_metrics(
                        tau_hat_ate=result.get("tau_hat_ate"),
                        tau_hat_cate=result.get("tau_hat_cate"),
                        tau_true_ate=tau_true_ate,
                        tau_true_cate=tau_true_cate,
                        ate_ci=result.get("ate_ci"),
                    )

                    run_record = {
                        "dataset": "synthetic_ci",
                        "scenario": scenario_dir.name,
                        "setting": setting,
                        "method": method_name,
                        "run_id(replication_idx)": run_id,
                        "n_samples": meta["n_samples"],
                        "n_features": meta["n_features"],
                        "stats": meta.get("stats"),
                        "config": meta.get("config"),
                        "metrics": metrics,
                    }

                    # Organize by dataset / scenario / setting / run_id / method_name.json
                    out_dir = results_dir / "synthetic_ci" / scenario_dir.name / setting / f"run_{run_id:03d}"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"{method_name}.json"
                    try:
                        with open(out_path, "w") as f:
                            json.dump(run_record, f, indent=2)
                    except Exception as e:
                        tqdm.write(f"    [ERROR] Failed to save results for {method_name}, scenario {scenario_dir.name}, run {run_id}: {e}")

                    # Create flat record for summary
                    flat = {
                        "dataset": "synthetic_ci",
                        "scenario": scenario_dir.name,
                        "setting": setting,
                        "method": method_name,
                        "run_id(replication_idx)": run_id,
                        **metrics,
                    }
                    all_records.append(flat)
                    
                    # Save intermediate summary periodically
                    if save_interval > 0 and len(all_records) % save_interval == 0:
                        _save_summary_intermediate(all_records, results_dir, dataset.lower(), setting)

    elif dataset.lower() == "reef":
        
        ##### Reef DB에 연결 #####
        from REEF_v2.src.reef_data_loader import REEFDataLoader
        from REEF_v2.src.ate_calculator import coerce_df_to_numeric
        
        loader = REEFDataLoader(db_name="reef_db")
        
        ##### Load Scenarios #####
        # Scenarios는 experiments/questions/reef/causal_analysis.json에 들어 있음 
        scenarios_path = Path("experiments/questions/reef/causal_analysis.json")
        if not scenarios_path.exists():
            raise FileNotFoundError(f"Scenarios file not found: {scenarios_path}")
        
        scenarios = json.loads(scenarios_path.read_text())
        
        print(f"Running experiments on REEF dataset: {len(scenarios)} scenarios, methods={methods}, setting={setting}")

        # method 별 실행 
        for method_name in methods:
            
            if method_name not in METHOD_REGISTRY:
                tqdm.write(f"Skipping unknown method: {method_name}")
                continue
            
            method_fn = get_method(method_name)
            
            # 질문 하나씩 (scenario 하나를 의미)
            scenario_pbar = tqdm(
                enumerate(scenarios),
                desc=f"Scenarios ({method_name})",
                total=len(scenarios),
                unit="scenario",
                position=0,
                leave=True,
                ncols=100
            )
            for scenario_idx, scenario in scenario_pbar:
                scenario_pbar.set_description(f"Scenario {scenario_idx+1}/{len(scenarios)} ({method_name})")
                
                # Scenario 정보 추출
                treatment = scenario.get("treatment")
                outcome = scenario.get("outcome")
                confounders = scenario.get("confounders", [])
                mediators = scenario.get("mediators", [])
                instrumental_variables = scenario.get("instrumental_variables", [])
                sql_query = scenario.get("sql_query")
                ground_truth_ate = scenario.get("ground_truth_ate")
                question = scenario.get("question", f"What is the causal effect of {treatment} on {outcome}?")
                
                if not sql_query:
                    tqdm.write(f"  [ERROR] Scenario {scenario_idx+1} missing sql_query. Skipping.")
                    continue
                
                # 데이터 로드
                try:
                    df = loader.load_custom_query(sql_query)
                    if len(df) == 0:
                        tqdm.write(f"  [ERROR] Scenario {scenario_idx+1}: Empty dataframe. Skipping.")
                        continue
                    
                    # 데이터 타입 변환 (numeric으로)
                    df = coerce_df_to_numeric(df, dropna=True, verbose=False)
                    
                    if len(df) < 10:
                        tqdm.write(f"  [ERROR] Scenario {scenario_idx+1}: Insufficient data ({len(df)} rows). Skipping.")
                        continue
                    
                except Exception as e:
                    tqdm.write(f"  [ERROR] Scenario {scenario_idx+1}: Failed to load data: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                
                # 변수명 해석 (테이블 prefix 제거)
                def resolve_var_name(var_name: str, df: pd.DataFrame) -> str:
                    """변수명을 데이터프레임의 실제 컬럼명으로 해석"""
                    if var_name in df.columns:
                        return var_name
                    # 부분 매칭 시도
                    for col in df.columns:
                        if col.endswith(f".{var_name}") or col.split(".")[-1] == var_name:
                            return col
                    return var_name  # 못 찾으면 원래 이름 반환
                
                treatment_resolved = resolve_var_name(treatment, df)
                outcome_resolved = resolve_var_name(outcome, df)
                confounders_resolved = [resolve_var_name(c, df) for c in confounders if resolve_var_name(c, df) in df.columns]
                
                # X, T, Y 추출
                if treatment_resolved not in df.columns:
                    tqdm.write(f"  [ERROR] Scenario {scenario_idx+1}: Treatment '{treatment_resolved}' not found. Skipping.")
                    continue
                if outcome_resolved not in df.columns:
                    tqdm.write(f"  [ERROR] Scenario {scenario_idx+1}: Outcome '{outcome_resolved}' not found. Skipping.")
                    continue
                
                # Confounders를 X로 사용
                X_cols = confounders_resolved if confounders_resolved else []
                # X가 비어있으면 더미 변수 하나 추가 (일부 method가 X를 요구할 수 있음)
                if len(X_cols) == 0:
                    # 더미 변수 생성 (상수 1)
                    df["_dummy"] = 1.0
                    X_cols = ["_dummy"]
                
                X = df[X_cols].values if X_cols else np.ones((len(df), 1))
                T = df[treatment_resolved].values
                Y = df[outcome_resolved].values
                
                # Ground truth ATE
                tau_true_ate = float(ground_truth_ate) if ground_truth_ate is not None else None
                tau_true_cate = None  # REEF 데이터에는 CATE ground truth가 없음
                
                # Causal graph 생성 (ORCA용)
                if setting == "oracle_graph":
                    # Confounders -> T, Y / T -> Y 구조
                    edges = []
                    for conf in confounders_resolved:
                        edges.append({"from": conf, "to": treatment_resolved})
                        edges.append({"from": conf, "to": outcome_resolved})
                    edges.append({"from": treatment_resolved, "to": outcome_resolved})
                    
                    causal_graph = {
                        "nodes": confounders_resolved + [treatment_resolved, outcome_resolved],
                        "edges": edges,
                        "variables": confounders_resolved + [treatment_resolved, outcome_resolved],
                        "graph": {
                            "variables": confounders_resolved + [treatment_resolved, outcome_resolved],
                            "edges": edges
                        },
                    }
                elif setting == "agent_graph":
                    discovery_method_name = discovery_method or "orca"
                    causal_graph, discovery_error = _run_causal_discovery(
                        df=df,
                        discovery_method=discovery_method_name,
                        discovery_context=discovery_context,
                        discovery_result_dir=discovery_result_dir,
                        dataset="reef",
                        scenario=scenario_idx,
                        run_id=scenario_idx,
                    )
                    if discovery_error:
                        tqdm.write(
                            f"  [ERROR] Discovery failed for scenario {scenario_idx+1}: {discovery_error}"
                        )
                        causal_graph = None
                else:
                    raise ValueError(f"Unknown setting: {setting}")
                
                # Method 실행
                context: Dict[str, Any] = {
                    "seed": 0,
                }
                
                if method_name == "orca":
                    if setting == "agent_graph" and causal_graph is None:
                        tqdm.write(
                            f"  [ERROR] Missing discovery graph for {method_name} (scenario {scenario_idx+1}). Skipping."
                        )
                        continue
                    context.update({
                        "df": df,
                        "causal_graph": causal_graph,
                        "app": orca_app,
                        "treatment_name": treatment_resolved,
                        "outcome_name": outcome_resolved,
                    })
                
                try:
                    result = method_fn(X, T, Y, context=context)
                except Exception as e:
                    tqdm.write(f"  [ERROR] Method {method_name} failed for scenario {scenario_idx+1}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                
                # Metrics 계산
                # tau_true_ate가 None이면 metrics 계산을 건너뛰거나 경고만 표시
                if tau_true_ate is None:
                    tqdm.write(f"  [WARNING] Scenario {scenario_idx+1}: No ground_truth_ate. Skipping metrics calculation.")
                    metrics = {
                        "ate_hat": result.get("tau_hat_ate"),
                        "ate_true": None,
                        "ate_bias": None,
                        "ate_abs_error": None,
                        "ate_sq_error": None,
                        "ate_ci_covered": None,
                        "ate_ci_width": None,
                        "cate_pehe": None,
                        "cate_mse": None,
                    }
                else:
                    metrics = compute_metrics(
                        tau_hat_ate=result.get("tau_hat_ate"),
                        tau_hat_cate=result.get("tau_hat_cate"),
                        tau_true_ate=tau_true_ate,
                        tau_true_cate=tau_true_cate,
                        ate_ci=result.get("ate_ci"),
                    )
                
                # Run record 생성
                run_record = {
                    "dataset": "REEF",
                    "scenario": question,  # scenario 식별자로 question 사용
                    "setting": setting,
                    "method": method_name,
                    "run_id(replication_idx)": scenario_idx,
                    "treatment": treatment,
                    "outcome": outcome,
                    "confounders": confounders,
                    "n_samples": len(df),
                    "n_features": X.shape[1] if X.ndim > 1 else 1,
                    "ground_truth_ate": tau_true_ate,
                    "metrics": metrics,
                    "sql_query": sql_query,
                }
                
                # 결과 저장
                # REEF의 경우: 한 파일에 모든 scenario 결과를 리스트로 저장
                # Organize by dataset / setting / method_name.json
                out_dir = results_dir / "REEF" / setting
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{method_name}.json"
                
                try:
                    # 기존 파일이 있으면 읽어서 리스트에 추가, 없으면 새 리스트 생성
                    if out_path.exists():
                        with open(out_path, "r") as f:
                            all_results = json.load(f)
                        if not isinstance(all_results, list):
                            # 기존 파일이 리스트가 아니면 리스트로 변환
                            all_results = [all_results]
                    else:
                        all_results = []
                    
                    # 새 결과 추가
                    all_results.append(run_record)
                    
                    # 파일에 저장
                    with open(out_path, "w") as f:
                        json.dump(all_results, f, indent=2)
                except Exception as e:
                    tqdm.write(f"  [ERROR] Failed to save results for {method_name}, scenario {scenario_idx+1}: {e}")
                
                # Flat record for summary
                flat = {
                    "dataset": "REEF",
                    "scenario": question,
                    "setting": setting,
                    "method": method_name,
                    "run_id(replication_idx)": scenario_idx,
                    "treatment": treatment,
                    "outcome": outcome,
                    "n_samples": len(df),
                    "ground_truth_ate": tau_true_ate,
                    **metrics,
                }
                all_records.append(flat)
                
                # Save intermediate summary periodically
                if save_interval > 0 and len(all_records) % save_interval == 0:
                    _save_summary_intermediate(all_records, results_dir, dataset.lower(), setting)
            
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Generate summary from all_records
    if all_records:
        df = pd.DataFrame(all_records)
        group_cols = ["dataset", "scenario", "setting", "method"]
        agg_dict = {
            "ate_abs_error": ["mean", "std"],
            "ate_sq_error": ["mean", "std"],
            "ate_bias": ["mean", "std"],
            "ate_ci_covered": ["mean"],
            "cate_pehe": ["mean", "std"],
            "cate_mse": ["mean", "std"],
        }
        summary = df.groupby(group_cols).agg(agg_dict)
        summary.columns = ["_".join([c for c in col if c]) for col in summary.columns.values]
        summary = summary.reset_index()
        
        # Add additional performance metrics
        # 1. Coverage rate as percentage
        if "ate_ci_covered_mean" in summary.columns:
            summary["ate_ci_covered_percent"] = summary["ate_ci_covered_mean"] * 100
        
        # 2. RMSE (Root Mean Squared Error) = sqrt(MSE)
        if "ate_sq_error_mean" in summary.columns:
            summary["ate_rmse"] = np.sqrt(summary["ate_sq_error_mean"])
        
        # 3. Absolute bias (mean of absolute bias values)
        if "ate_bias_mean" in summary.columns:
            summary["ate_abs_bias"] = np.abs(summary["ate_bias_mean"])
        
        # 4. CI width statistics (if available)
        if "ate_ci_width" in df.columns:
            ci_width_agg = df.groupby(group_cols)["ate_ci_width"].agg(["mean", "std"])
            summary = summary.merge(ci_width_agg, left_on=group_cols, right_index=True, how="left")
            summary = summary.rename(columns={"mean": "ate_ci_width_mean", "std": "ate_ci_width_std"})

        summary_dir = results_dir / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_dir / f"summary_{dataset}_{setting}.csv", index=False)
        
        return summary
    else:
        return None


# === 5. YAML Config Loader ===

def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load experiment configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


# === 6. Main entry point ===

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run effect estimation experiments from YAML config.")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/effect_estimation/effect_experiments.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Run only a specific experiment by name. If omitted, run all experiments.",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=50,
        help="Save intermediate summary every N records (default: 50). Set to 0 to disable intermediate saving.",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    
    # Get global settings
    global_settings = config.get("global", {})
    global_results_dir = global_settings.get("results_dir", "experiments/results/effect_estimation")
    global_seed = global_settings.get("seed", 0)
    
    # Get experiments
    experiments = config.get("experiments", [])
    
    if args.experiment:
        # Filter to specific experiment
        experiments = [exp for exp in experiments if exp.get("name") == args.experiment]
        if not experiments:
            raise ValueError(f"Experiment '{args.experiment}' not found in config.")
    
    # Run each experiment
    for exp_config in experiments:
        exp_name = exp_config.get("name", "unnamed")
        print(f"\n{'='*60}")
        print(f"Running experiment: {exp_name}")
        print(f"{'='*60}")
        
        dataset = exp_config.get("dataset")
        if dataset is None:
            print(f"Skipping experiment '{exp_name}': missing 'dataset' field")
            continue
        
        setting = exp_config.get("setting", "oracle_graph")
        runs = exp_config.get("runs", None)  # None means use all available
        run_indices = exp_config.get("run_indices", None)  # Optional: specify exact run indices
        methods = exp_config.get("methods", [])
        discovery_method = exp_config.get("discovery_method")
        discovery_context = exp_config.get("discovery_context")
        discovery_result_dir = exp_config.get("discovery_result_dir")
        discovery_result_dir = Path(discovery_result_dir) if discovery_result_dir else None
        
        # Use experiment-specific results_dir if provided, otherwise use global
        results_dir = Path(exp_config.get("results_dir", global_results_dir))
        seed = exp_config.get("seed", global_seed)
        
        # Dataset-specific parameters
        if dataset.lower() == "ihdp":
            ihdp_path = exp_config.get("data_path")
            if ihdp_path is None:
                print(f"Skipping experiment '{exp_name}': missing 'data_path' for IHDP dataset")
                continue
            
            run_experiments(
                dataset=dataset,
                methods=methods,
                setting=setting,
                runs=runs,
                run_indices=run_indices,
                results_dir=results_dir,
                ihdp_path=ihdp_path,
                synthetic_base_dir=None,
                scenarios=None,
                discovery_method=discovery_method,
                discovery_context=discovery_context,
                discovery_result_dir=discovery_result_dir,
                seed=seed,
                save_interval=args.save_interval,
            )
        
        elif dataset.lower() == "synthetic_ci":
            base_dir = exp_config.get("base_dir")
            if base_dir is None:
                print(f"Skipping experiment '{exp_name}': missing 'base_dir' for synthetic_ci dataset")
                continue
            
            scenarios = exp_config.get("scenarios")
            # If scenarios is None or empty list, run_experiments will use all scenarios
            
            run_experiments(
                dataset=dataset,
                methods=methods,
                setting=setting,
                runs=runs,
                run_indices=run_indices,
                results_dir=results_dir,
                ihdp_path=None,
                synthetic_base_dir=base_dir,
                scenarios=scenarios,
                discovery_method=discovery_method,
                discovery_context=discovery_context,
                discovery_result_dir=discovery_result_dir,
                seed=seed,
                save_interval=args.save_interval,
            )
        
        elif dataset.lower() == "reef" :

            run_experiments(
                dataset=dataset,
                methods=methods,
                setting=setting,
                runs=runs,
                run_indices=run_indices,
                results_dir=results_dir,
                ihdp_path=None,
                synthetic_base_dir=None,
                scenarios=None,
                discovery_method=discovery_method,
                discovery_context=discovery_context,
                discovery_result_dir=discovery_result_dir,
                seed=None,
                save_interval=args.save_interval,
            )

        else:
            print(f"Skipping experiment '{exp_name}': unknown dataset '{dataset}'")
            continue
        
        print(f"Completed experiment: {exp_name}")
    
    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
