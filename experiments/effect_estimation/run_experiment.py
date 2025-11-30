# experiments/run_experiment.py

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import yaml

from experiments.effect_estimation.methods import get_method, METHOD_REGISTRY

# ORCA LLM / graph
from agents.causal_analysis.graph import generate_causal_analysis_graph
from utils.llm import get_llm
import networkx as nx


# === 1. Data loaders ===

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
    seed: int = 0,
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
        
        for run_id in replication_indices:
            X, T, Y, tau_true_cate, tau_true_ate, meta = load_ihdp_dataset(ihdp_path, replicate_idx=run_id)

            # ORCA df / causal_graph (Oracle setting uses all confounders)
            df = pd.DataFrame(X, columns=[f"W{i}" for i in range(X.shape[1])])
            df["T"] = T
            df["Y"] = Y

            if setting == "oracle_graph":
                causal_graph = create_oracle_causal_graph(df, treatment="T", outcome="Y")
            elif setting == "agent_graph":
                # TODO: load \hat G from discovery agent
                causal_graph = create_oracle_causal_graph(df, treatment="T", outcome="Y")
            else:
                raise ValueError(f"Unknown setting: {setting}")

            for method_name in methods:
                if method_name not in METHOD_REGISTRY:
                    print(f"Skipping unknown method: {method_name}")
                    continue
                method_fn = get_method(method_name)

                context: Dict[str, Any] = {
                    "seed": seed + run_id,
                }
                if method_name == "orca":
                    context.update(
                        {
                            "df": df,
                            "causal_graph": causal_graph,
                            "app": orca_app,
                            "treatment_name": "T",
                            "outcome_name": "Y",
                        }
                    )

                result = method_fn(X, T, Y, context=context)
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
                with open(out_path, "w") as f:
                    json.dump(run_record, f, indent=2)

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

    elif dataset.lower() == "synthetic_ci":
        if synthetic_base_dir is None:
            raise ValueError("synthetic_base_dir must be provided for dataset='synthetic_ci'.")

        base_dir = Path(synthetic_base_dir)
        if scenarios is None or len(scenarios) == 0:
            scenario_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()])
        else:
            scenario_dirs = [base_dir / s for s in scenarios]

        print(f"Running experiments on Synthetic CI scenarios={ [d.name for d in scenario_dirs] }, methods={methods}, setting={setting}")

        for scenario_dir in scenario_dirs:
            # Get available runs for this scenario
            available_runs = get_available_runs(scenario_dir)
            
            if not available_runs:
                print(f"Warning: No valid run directories found in {scenario_dir}. Skipping.")
                continue
            
            # Determine which run indices to use
            if run_indices is not None:
                # Use specified run indices (filter to only those that exist)
                scenario_run_indices = [idx for idx in run_indices if idx in available_runs]
                if len(scenario_run_indices) != len(run_indices):
                    missing = [idx for idx in run_indices if idx not in scenario_run_indices]
                    print(f"Warning: Some run indices not found in {scenario_dir.name}: {missing}")
            elif runs is not None and runs > 0:
                # Use first N runs
                scenario_run_indices = available_runs[:min(runs, len(available_runs))]
            else:
                # Use all available runs
                scenario_run_indices = available_runs
            
            if not scenario_run_indices:
                print(f"Warning: No valid run indices for {scenario_dir.name}. Skipping.")
                continue
            
            print(f"  Scenario {scenario_dir.name}: using runs {scenario_run_indices} (out of {len(available_runs)} available)")
            
            for run_id in scenario_run_indices:
                X, T, Y, tau_true_cate, tau_true_ate, meta = load_synthetic_ci_dataset(
                    scenario_dir, run_idx=run_id
                )
                df = meta["df"]
                
                if setting == "oracle_graph":
                    causal_graph = create_oracle_causal_graph(df, treatment="T", outcome="Y")
                elif setting == "agent_graph":
                    # TODO: load \hat G from discovery result
                    causal_graph = create_oracle_causal_graph(df, treatment="T", outcome="Y")
                else:
                    raise ValueError(f"Unknown setting: {setting}")

                for method_name in methods:
                    if method_name not in METHOD_REGISTRY:
                        print(f"Skipping unknown method: {method_name}")
                        continue
                    method_fn = get_method(method_name)

                    context: Dict[str, Any] = {
                        "seed": seed + run_id,
                    }
                    if method_name == "orca":
                        context.update(
                            {
                                "df": df,
                                "causal_graph": causal_graph,
                                "app": orca_app,
                                "treatment_name": "T",
                                "outcome_name": "Y",
                            }
                        )

                    result = method_fn(X, T, Y, context=context)
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
                    with open(out_path, "w") as f:
                        json.dump(run_record, f, indent=2)

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
                seed=seed,
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
                seed=seed,
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