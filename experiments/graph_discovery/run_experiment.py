# experiments/run_experiment.py

from __future__ import annotations
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Iterator, Tuple, Optional
import itertools

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file (project root)
ROOT_DIR = Path(__file__).parent.parent.parent
load_dotenv(dotenv_path=ROOT_DIR / ".env")

from experiments.graph_discovery import cd_methods
METHOD_REGISTRY = cd_methods.METHOD_REGISTRY
get_method = cd_methods.get_method
from experiments.graph_discovery.metrics import compute_graph_metrics, compute_node_level_records


# === Data loader implementations ==========================================


def iter_synthetic_cd_runs(
    scenarios: List[str] | None = None,
    d_list: List[int] | None = None,
    run_indices: List[int] | None = None,
) -> Iterator[Tuple[str, int, int, pd.DataFrame, List[Dict[str, Any]], List[str], Dict[str, Any]]]:
    """
    Iterate over synthetic CD benchmark datasets.

    Yields
    ------
    scenario : str
    d : int
    run_id : int
    df : pd.DataFrame
    edges_true : List[{"from": str, "to": str}]
    nodes : List[str]
    meta : Dict[str, Any]
        Parsed from config.json (n, d, avg_degree, etc.).
    """
    base_dir = Path("data/synthetic_cd")

    if scenarios is None or len(scenarios) == 0:
        scenario_dirs = [p for p in base_dir.iterdir() if p.is_dir()]
    else:
        scenario_dirs = [base_dir / s for s in scenarios]

    for scen_dir in scenario_dirs:
        if not scen_dir.exists():
            continue
        scenario_name = scen_dir.name

        # Dimension subdirectories: d_3, d_5, ...
        d_dirs = [p for p in scen_dir.iterdir() if p.is_dir() and p.name.startswith("d_")]
        if d_list is not None:
            d_filter = set(d_list)
            d_dirs = [p for p in d_dirs if int(p.name.split("_")[1]) in d_filter]

        for d_dir in sorted(d_dirs, key=lambda p: int(p.name.split("_")[1])):
            d_val = int(d_dir.name.split("_")[1])

            run_dirs = [p for p in d_dir.iterdir() if p.is_dir() and p.name.startswith("run_")]
            if run_indices is not None:
                idx_set = set(run_indices)
                run_dirs = [p for p in run_dirs if int(p.name.split("_")[1]) in idx_set]

            for run_dir in sorted(run_dirs, key=lambda p: int(p.name.split("_")[1])):
                run_id = int(run_dir.name.split("_")[1])

                data_path = run_dir / "data.csv"
                edges_path = run_dir / "edges.csv"
                config_path = run_dir / "config.json"

                if not data_path.exists() or not edges_path.exists():
                    continue

                df = pd.read_csv(data_path)
                # Ground-truth edges: u,v are integer node indices
                edges_df = pd.read_csv(edges_path)
                nodes = [f"V{i}" for i in range(df.shape[1])]
                edges_true: List[Dict[str, Any]] = []
                for _, row in edges_df.iterrows():
                    u_idx = int(row["u"])
                    v_idx = int(row["v"])
                    edges_true.append({"from": nodes[u_idx], "to": nodes[v_idx]})

                meta: Dict[str, Any] = {
                    "scenario": scenario_name,
                    "d": d_val,
                    "run_id": run_id,
                }
                if config_path.exists():
                    try:
                        with open(config_path, "r") as f:
                            cfg = json.load(f)
                        meta.update(cfg)
                    except Exception:
                        pass

                yield scenario_name, d_val, run_id, df, edges_true, nodes, meta


def iter_bnlearn_datasets(
    dataset_names: List[str] | None = None,
) -> Iterator[Tuple[str, pd.DataFrame, List[Dict[str, Any]], List[str], Dict[str, Any]]]:
    """
    Iterate over pre-exported bnlearn datasets.

    Assumes directories like:
      data/raw/bnlearn/asia/
        - data.csv
        - model         (bnlearn save file)
        - graph.png     (optional)

    Yields
    ------
    name : str
    df : pd.DataFrame
    edges_true : List[{'from': str, 'to': str}]
    nodes : List[str]
    meta : Dict[str, Any]
    """
    import bnlearn as bn  # type: ignore[import]

    base_dir = Path("data/raw/bnlearn")

    if dataset_names is None or len(dataset_names) == 0:
        ds_dirs = [p for p in base_dir.iterdir() if p.is_dir()]
    else:
        ds_dirs = [base_dir / name for name in dataset_names]

    for ds_dir in ds_dirs:
        if not ds_dir.exists():
            continue
        name = ds_dir.name
        data_path = ds_dir / "data.csv"
        model_path = ds_dir / "model.pkl"
        if not data_path.exists() or not model_path.exists():
            continue

        df = pd.read_csv(data_path)
        try:
            model = bn.load(filepath=str(model_path))
        except Exception:
            continue

        # bnlearn models typically expose edges under model['model'].edges or similar
        edges_true: List[Dict[str, Any]] = []
        nodes = list(df.columns)
        try:
            G = model["model"]
            for u, v in G.edges():
                edges_true.append({"from": str(u), "to": str(v)})
        except Exception:
            # Fallback: use structure_learning.edges if present
            try:
                for u, v in model["structure_learning"]["model_edges"]:
                    edges_true.append({"from": str(u), "to": str(v)})
            except Exception:
                # If we cannot extract edges, skip this dataset
                continue

        meta: Dict[str, Any] = {
            "dataset": name,
            "n_samples": df.shape[0],
            "n_features": df.shape[1],
        }
        yield name, df, edges_true, nodes, meta


# === 4. Summary helpers ======================================================


def _save_summary_intermediate(
    all_records: List[Dict[str, Any]],
    results_dir: Path,
    dataset: str,
) -> None:
    """Save intermediate summary from current all_records."""
    if not all_records:
        return

    try:
        df = pd.DataFrame(all_records)
        group_cols = [c for c in ["dataset", "scenario", "d", "method", "bnlearn_dataset"] if c in df.columns]
        agg_dict = {
            "shd": ["mean", "std"],
            "sid": ["mean", "std"],
            "precision": ["mean", "std"],
            "recall": ["mean", "std"],
            "f1": ["mean", "std"],
        }
        available_cols = df.columns.tolist()
        filtered_agg = {k: v for k, v in agg_dict.items() if k in available_cols}
        if not filtered_agg:
            return

        summary = df.groupby(group_cols).agg(filtered_agg)
        summary.columns = ["_".join([c for c in col if c]) for col in summary.columns.values]
        summary = summary.reset_index()

        summary_dir = results_dir / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = summary_dir / f"summary_{dataset}_{timestamp}.csv"
        summary.to_csv(summary_path, index=False)
        print(f"[INFO] Intermediate summary saved: {summary_path} ({len(all_records)} records)")
    except Exception as e:
        print(f"[WARNING] Failed to save intermediate summary: {e}")


# === 5. Main experiment loop ================================================


def run_experiments(
    dataset: str,
    methods: List[str],
    results_dir: Optional[Path] = None,
    scenarios: Optional[List[str]] = None,
    d_list: Optional[List[int]] = None,
    runs: Optional[int] = None,
    run_indices: Optional[List[int]] = None,
    bnlearn_datasets: Optional[List[str]] = None,
    seed: int = 0,
    save_interval: int = 50,
    method_context: Optional[Dict[str, Any]] = None,
) -> Optional[pd.DataFrame]:
    """
    Run graph discovery experiments for a given dataset type.

    Parameters
    ----------
    dataset : {'synthetic_cd', 'bnlearn'}
    methods : list of method names registered in METHOD_REGISTRY
    """
    if results_dir is None:
        results_dir = Path("experiments/results/graph_discovery")
    results_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)
    all_records: List[Dict[str, Any]] = []
    node_records: List[Dict[str, Any]] = []

    if dataset.lower() == "synthetic_cd":
        base = Path("data/synthetic_cd")
        print(f"Running graph discovery on synthetic_cd (base={base}), "
              f"scenarios={scenarios}, d_list={d_list}, methods={methods}")

        # Collect all run descriptors
        run_iter = iter_synthetic_cd_runs(
            scenarios=scenarios or [],
            d_list=d_list,
            run_indices=run_indices,
        )

        for scenario, d_val, run_id, df, edges_true, nodes, meta in tqdm(
            run_iter, desc="Synthetic CD runs", unit="run"
        ):
            # Optional limit on number of runs per (scenario, d)
            if runs is not None and run_id >= runs:
                continue

            tqdm.write(f"[INFO] Starting experiment: scenario={scenario}, d={d_val}, run={run_id}")

            for method_name in methods:
                if method_name not in METHOD_REGISTRY:
                    tqdm.write(f"Skipping unknown method: {method_name}")
                    continue

                method_fn = get_method(method_name)

                base_context: Dict[str, Any] = {
                    "df": df,
                    "seed": seed + run_id,
                }
                if method_context:
                    shared_ctx = method_context.get("_shared", {})
                    per_method = method_context.get(method_name, {})
                    ctx = {**shared_ctx, **per_method}
                    base_context.update(ctx)

                result = method_fn(df.values, context=base_context)

                # Check for error in result (some methods return {"error": "..."} dict)
                if isinstance(result, dict) and "error" in result:
                    error_msg = result.get("error", "Unknown error")
                    tqdm.write(
                        f"[ERROR] Method {method_name} returned error "
                        f"for scenario={scenario}, d={d_val}, run={run_id}: {error_msg}"
                    )
                    continue

                # Extract predicted edges / nodes from result
                # ORCA and other methods use unified schema: {"graph": {"edges": [...], "variables": [...]}, "metadata": {...}}
                if isinstance(result, dict):
                    if "graph" in result:
                        # Unified nested format: result["graph"]["variables"] and result["graph"]["edges"] (e.g., ORCA, normalize_graph_result)
                        g = result["graph"]
                        nodes_hat = g.get("variables", nodes)
                        edges_hat = g.get("edges", [])
                    else:
                        # Fallback: assume same nodes and empty edges
                        nodes_hat = nodes
                        edges_hat = []
                else:
                    # Fallback: assume same nodes and empty edges
                    nodes_hat = nodes
                    edges_hat = []

                metrics = compute_graph_metrics(nodes_hat, edges_hat, edges_true)

                run_record = {
                    "dataset": "synthetic_cd",
                    "scenario": scenario,
                    "d": d_val,
                    "method": method_name,
                    "run_id": run_id,
                    "n_samples": df.shape[0],
                    "n_features": df.shape[1],
                    "metrics": metrics,
                    "meta": meta,
                }

                out_dir = results_dir / "synthetic_cd" / scenario / f"d_{d_val}" / f"run_{run_id:03d}"
                out_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = out_dir / f"{method_name}_{timestamp}.json"
                try:
                    with open(out_path, "w") as f:
                        json.dump(run_record, f, indent=2)
                except Exception as e:
                    tqdm.write(
                        f"[ERROR] Failed to save results for method={method_name}, "
                        f"scenario={scenario}, d={d_val}, run={run_id}: {e}"
                    )

                flat = {
                    "dataset": "synthetic_cd",
                    "scenario": scenario,
                    "d": d_val,
                    "method": method_name,
                    "run_id": run_id,
                    **metrics,
                }
                # Attach ablation config / name (if any) to flat record for analysis
                if method_context:
                    flat_config = {}
                    shared_meta = method_context.get("_shared", {})
                    shared_cfg = shared_meta.get("causal_discovery_config", {})
                    method_cfg = method_context.get(method_name, {}).get("causal_discovery_config", {})
                    ablation_name = shared_meta.get("ablation_name")
                    if shared_cfg:
                        flat_config["causal_discovery_config_shared"] = shared_cfg
                    if method_cfg:
                        flat_config["causal_discovery_config_method"] = method_cfg
                    if flat_config:
                        flat["config"] = flat_config
                    if ablation_name is not None:
                        flat["ablation_name"] = ablation_name
                all_records.append(flat)

                # Node-level metrics
                base_node = {
                    "dataset": "synthetic_cd",
                    "scenario": scenario,
                    "d": d_val,
                    "method": method_name,
                    "run_id": run_id,
                }
                node_records.extend(
                    compute_node_level_records(nodes_hat, edges_hat, edges_true, base_node)
                )

                if save_interval > 0 and len(all_records) % save_interval == 0:
                    _save_summary_intermediate(all_records, results_dir, "synthetic_cd")

    elif dataset.lower() == "bnlearn":
        base = Path("data/raw/bnlearn")
        print(f"Running graph discovery on bnlearn datasets (base={base}), "
              f"datasets={bnlearn_datasets}, methods={methods}")

        for name, df, edges_true, nodes, meta in tqdm(
            iter_bnlearn_datasets(dataset_names=bnlearn_datasets or []),
            desc="bnlearn datasets",
            unit="dataset",
        ):
            tqdm.write(f"[INFO] Starting experiment: bnlearn dataset={name}")

            for method_name in methods:
                if method_name not in METHOD_REGISTRY:
                    tqdm.write(f"Skipping unknown method: {method_name}")
                    continue

                method_fn = get_method(method_name)
                context: Dict[str, Any] = {
                    "df": df,
                    "seed": seed,
                }
                if method_context:
                    shared_ctx = method_context.get("_shared", {})
                    per_method = method_context.get(method_name, {})
                    ctx = {**shared_ctx, **per_method}
                    context.update(ctx)
                result = method_fn(df.values, context=context)

                if isinstance(result, dict) and "error" in result:
                    error_msg = result.get("error", "Unknown error")
                    tqdm.write(
                        f"[ERROR] Method {method_name} returned error "
                        f"for bnlearn dataset={name}: {error_msg}"
                    )
                    continue

                # Extract predicted edges / nodes from result
                if isinstance(result, dict):
                    if "graph" in result:
                        g = result["graph"]
                        nodes_hat = g.get("variables", nodes)
                        edges_hat = g.get("edges", [])
                    elif "edges" in result and "variables" in result:
                        # Direct format: result["edges"] and result["variables"] (e.g., ORCA method)
                        nodes_hat = result.get("variables", nodes)
                        edges_hat = result.get("edges", [])
                    else:
                        # Fallback: assume same nodes and empty edges
                        nodes_hat = nodes
                        edges_hat = []
                else:
                    # Fallback: assume same nodes and empty edges
                    nodes_hat = nodes
                    edges_hat = []

                metrics = compute_graph_metrics(nodes_hat, edges_hat, edges_true)

                run_record = {
                    "dataset": "bnlearn",
                    "bnlearn_dataset": name,
                    "method": method_name,
                    "n_samples": df.shape[0],
                    "n_features": df.shape[1],
                    "metrics": metrics,
                    "meta": meta,
                }

                out_dir = results_dir / "bnlearn" / name
                out_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = out_dir / f"{method_name}_{timestamp}.json"
                try:
                    with open(out_path, "w") as f:
                        json.dump(run_record, f, indent=2)
                except Exception as e:
                    tqdm.write(
                        f"[ERROR] Failed to save results for method={method_name}, "
                        f"bnlearn dataset={name}: {e}"
                    )

                flat = {
                    "dataset": "bnlearn",
                    "bnlearn_dataset": name,
                    "method": method_name,
                    **metrics,
                }
                if method_context:
                    flat_config = {}
                    shared_meta = method_context.get("_shared", {})
                    shared_cfg = shared_meta.get("causal_discovery_config", {})
                    method_cfg = method_context.get(method_name, {}).get("causal_discovery_config", {})
                    ablation_name = shared_meta.get("ablation_name")
                    if shared_cfg:
                        flat_config["causal_discovery_config_shared"] = shared_cfg
                    if method_cfg:
                        flat_config["causal_discovery_config_method"] = method_cfg
                    if flat_config:
                        flat["config"] = flat_config
                    if ablation_name is not None:
                        flat["ablation_name"] = ablation_name
                all_records.append(flat)

                base_node = {
                    "dataset": "bnlearn",
                    "bnlearn_dataset": name,
                    "method": method_name,
                }
                node_records.extend(
                    compute_node_level_records(nodes_hat, edges_hat, edges_true, base_node)
                )

                if save_interval > 0 and len(all_records) % save_interval == 0:
                    _save_summary_intermediate(all_records, results_dir, "bnlearn")

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Final summary (scenario / dataset level)
    if all_records:
        df_all = pd.DataFrame(all_records)
        # Base grouping columns (scenario is included for multi-scenario experiments)
        group_cols = [c for c in ["dataset", "scenario", "d", "method", "bnlearn_dataset"] if c in df_all.columns]
        
        # If config variations are present, extract key config values for grouping
        # Config is stored as nested dict in "config" column, so we flatten it for grouping
        if "config" in df_all.columns:
            # Extract composite_weights and execution_plan_thresholds for grouping
            def _extract_config_key(row, key_path: List[str]) -> Any:
                """Extract nested config value by path."""
                config = row.get("config", {})
                if not isinstance(config, dict):
                    return None
                shared_cfg = config.get("causal_discovery_config_shared", {})
                method_cfg = config.get("causal_discovery_config_method", {})
                cfg = {**shared_cfg, **method_cfg} if method_cfg else shared_cfg
                cursor = cfg
                for k in key_path:
                    if isinstance(cursor, dict):
                        cursor = cursor.get(k)
                    else:
                        return None
                return cursor
            
            # Add flattened config columns for grouping
            if any("composite_weights" in str(cfg) for cfg in df_all["config"] if isinstance(cfg, dict)):
                for weight_key in ["markov_consistency", "sampling_stability", "structural_stability"]:
                    col_name = f"config_composite_weights_{weight_key}"
                    df_all[col_name] = df_all.apply(
                        lambda row: _extract_config_key(row, ["composite_weights", weight_key]),
                        axis=1
                    )
                    if df_all[col_name].notna().any():
                        group_cols.append(col_name)
            
            if any("execution_plan_thresholds" in str(cfg) for cfg in df_all["config"] if isinstance(cfg, dict)):
                for threshold_key in ["linearity_score", "non_gaussian_score", "anm_score"]:
                    col_name = f"config_execution_plan_thresholds_{threshold_key}"
                    df_all[col_name] = df_all.apply(
                        lambda row: _extract_config_key(row, ["execution_plan_thresholds", threshold_key]),
                        axis=1
                    )
                    if df_all[col_name].notna().any():
                        group_cols.append(col_name)
        
        agg_dict = {
            "shd": ["mean", "std"],
            "sid": ["mean", "std"],
            "precision": ["mean", "std"],
            "recall": ["mean", "std"],
            "f1": ["mean", "std"],
        }
        summary = df_all.groupby(group_cols).agg(agg_dict)
        summary.columns = ["_".join([c for c in col if c]) for col in summary.columns.values]
        summary = summary.reset_index()

        summary_dir = results_dir / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary.to_csv(summary_dir / f"summary_{dataset.lower()}_{timestamp}.csv", index=False)

        # Node-level summary
        if node_records:
            df_nodes = pd.DataFrame(node_records)
            node_group_cols = [
                c
                for c in [
                    "dataset",
                    "scenario",
                    "d",
                    "method",
                    "bnlearn_dataset",
                    "node",
                ]
                if c in df_nodes.columns
            ]
            node_agg = {
                "precision_node": ["mean", "std"],
                "recall_node": ["mean", "std"],
                "f1_node": ["mean", "std"],
            }
            node_summary = df_nodes.groupby(node_group_cols).agg(node_agg)
            node_summary.columns = [
                "_".join([c for c in col if c]) for col in node_summary.columns.values
            ]
            node_summary = node_summary.reset_index()
            node_summary.to_csv(
                summary_dir / f"node_summary_{dataset.lower()}_{timestamp}.csv", index=False
            )

        return summary

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


# === Main entry point ===

def main():
    import argparse
    """Parse YAML config and dispatch graph discovery experiments."""
    parser = argparse.ArgumentParser(description="Run graph discovery experiments from YAML config.")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/graph_discovery/graph_discovery_experiments.yaml",
        help="Path to YAML configuration file.",
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
        help="Save intermediate summary every N records (default: 50). Set to 0 to disable.",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    global_settings = config.get("global", {})
    global_results_dir = global_settings.get("results_dir", "experiments/results/graph_discovery")
    global_seed = global_settings.get("seed", 0)

    experiments_cfg = config.get("experiments", [])
    if args.experiment:
        experiments_cfg = [e for e in experiments_cfg if e.get("name") == args.experiment]
        if not experiments_cfg:
            raise ValueError(f"Experiment '{args.experiment}' not found in config.")

    for exp in experiments_cfg:
        name = exp.get("name", "unnamed")
        print("\n" + "=" * 60)
        print(f"Running graph discovery experiment: {name}")
        print("=" * 60)

        dataset = exp.get("dataset")
        if dataset is None:
            print(f"Skipping experiment '{name}': missing 'dataset' field")
            continue

        methods = exp.get("methods", [])
        results_dir = Path(exp.get("results_dir", global_results_dir))
        seed = exp.get("seed", global_seed)

        # Optional: external config list for ablation
        config_list_path = exp.get("config_list_path")

        def _iter_method_contexts() -> List[Optional[Dict[str, Any]]]:
            """
            If config_list_path is provided, load a list of predefined
            causal_discovery_config entries and iterate over them.
            Otherwise, yield a single None (no overrides).
            """
            if not config_list_path:
                return [None]

            cfg_path = Path(config_list_path)
            if not cfg_path.is_absolute():
                cfg_path = ROOT_DIR / cfg_path

            if not cfg_path.exists():
                raise FileNotFoundError(f"Config list file not found: {cfg_path}")

            with open(cfg_path, "r") as f:
                cfg_list = json.load(f)

            if not isinstance(cfg_list, list):
                raise ValueError(f"Config list at {cfg_path} must be a list of objects.")

            contexts: List[Dict[str, Any]] = []
            for item in cfg_list:
                if not isinstance(item, dict):
                    continue
                cd_cfg = item.get("causal_discovery_config", {})
                if not isinstance(cd_cfg, dict):
                    continue
                ablation_name = item.get("name")
                ctx: Dict[str, Any] = {
                    "_shared": {
                        "causal_discovery_config": cd_cfg,
                    }
                }
                if ablation_name is not None:
                    ctx["_shared"]["ablation_name"] = ablation_name
                contexts.append(ctx)

            return contexts or [None]

        if dataset.lower() == "synthetic_cd":
            scenarios = exp.get("scenarios")
            d_list = exp.get("d_list")
            runs = exp.get("runs")
            run_indices = exp.get("run_indices")

            # Iterate over ablation grid (or single default context)
            for mc in _iter_method_contexts():
                run_experiments(
                    dataset="synthetic_cd",
                    methods=methods,
                    results_dir=results_dir,
                    scenarios=scenarios,
                    d_list=d_list,
                    runs=runs,
                    run_indices=run_indices,
                    seed=seed,
                    save_interval=args.save_interval,
                    method_context=mc,
                )
        elif dataset.lower() == "bnlearn":
            bn_datasets = exp.get("datasets")
            for mc in _iter_method_contexts():
                run_experiments(
                    dataset="bnlearn",
                    methods=methods,
                    results_dir=results_dir,
                    bnlearn_datasets=bn_datasets,
                    seed=seed,
                    save_interval=args.save_interval,
                    method_context=mc,
                )
        else:
            print(f"Skipping experiment '{name}': unknown dataset '{dataset}'")
            continue

        print(f"Completed experiment: {name}")


if __name__ == "__main__":
    main()