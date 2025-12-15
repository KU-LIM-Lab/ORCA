# experiments/effect_estimation/generate_summary.py

"""
Generate summary CSV from existing experiment result JSON files.

This script reads all JSON result files and generates summary CSV files
that would have been created if run_experiments() completed successfully.

Usage:
    python -m experiments.effect_estimation.generate_summary
    python -m experiments.effect_estimation.generate_summary --dataset ihdp --setting oracle_graph
    python -m experiments.effect_estimation.generate_summary --overwrite
"""

from __future__ import annotations
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np


def load_result_file(result_path: Path) -> Dict[str, Any] | None:
    """Load a single result JSON file."""
    try:
        with open(result_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load {result_path}: {e}")
        return None


def collect_all_records(results_dir: Path, dataset: str | None = None, setting: str | None = None) -> List[Dict[str, Any]]:
    """
    Collect all records from JSON result files.
    
    Args:
        results_dir: Base results directory (e.g., experiments/results/effect_estimation)
        dataset: Optional dataset filter (e.g., "ihdp", "synthetic_ci")
        setting: Optional setting filter (e.g., "oracle_graph", "agent_graph")
    
    Returns:
        List of flat records (same format as all_records in run_experiments)
    """
    all_records: List[Dict[str, Any]] = []
    
    results_dir = Path(results_dir)
    
    # Determine which datasets to process
    if dataset:
        dataset_name_map = {
            "ihdp": "IHDP",
            "synthetic_ci": "synthetic_ci"
        }
        actual_dataset_name = dataset_name_map.get(dataset.lower(), dataset)
        dataset_dirs = [results_dir / actual_dataset_name]
    else:
        dataset_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name in ["IHDP", "synthetic_ci"]]
    
    for dataset_dir in dataset_dirs:
        if not dataset_dir.exists():
            continue
        
        dataset_name = dataset_dir.name
        
        # For IHDP: results_dir/IHDP/setting/run_XXX/method.json
        if dataset_name == "IHDP":
            # Determine which settings to process
            if setting:
                setting_dirs = [dataset_dir / setting]
            else:
                setting_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
            
            for setting_dir in setting_dirs:
                if not setting_dir.exists():
                    continue
                
                setting_name = setting_dir.name
                
                # Process all run directories
                for run_dir in sorted(setting_dir.iterdir()):
                    if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                        continue
                    
                    # Process all method JSON files in this run
                    for method_file in run_dir.glob("*.json"):
                        method_name = method_file.stem
                        result = load_result_file(method_file)
                        
                        if result is None:
                            continue
                        
                        # Extract metrics and create flat record
                        metrics = result.get("metrics", {})
                        flat = {
                            "dataset": result.get("dataset", "IHDP"),
                            "scenario": result.get("scenario"),
                            "setting": result.get("setting", setting_name),
                            "method": result.get("method", method_name),
                            "run_id(replication_idx)": result.get("run_id(replication_idx)"),
                            "replicate_idx": result.get("replicate_idx", result.get("run_id(replication_idx)")),
                            **metrics,
                        }
                        all_records.append(flat)
        
        # For synthetic_ci: results_dir/synthetic_ci/scenario/setting/run_XXX/method.json
        elif dataset_name == "synthetic_ci":
            # Process all scenarios
            for scenario_dir in sorted(dataset_dir.iterdir()):
                if not scenario_dir.is_dir():
                    continue
                
                scenario_name = scenario_dir.name
                
                # Determine which settings to process
                if setting:
                    setting_dirs = [scenario_dir / setting]
                else:
                    setting_dirs = [d for d in scenario_dir.iterdir() if d.is_dir()]
                
                for setting_dir in setting_dirs:
                    if not setting_dir.exists():
                        continue
                    
                    setting_name = setting_dir.name
                    
                    # Process all run directories
                    for run_dir in sorted(setting_dir.iterdir()):
                        if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                            continue
                        
                        # Process all method JSON files in this run
                        for method_file in run_dir.glob("*.json"):
                            method_name = method_file.stem
                            result = load_result_file(method_file)
                            
                            if result is None:
                                continue
                            
                            # Extract metrics and create flat record
                            metrics = result.get("metrics", {})
                            flat = {
                                "dataset": result.get("dataset", "synthetic_ci"),
                                "scenario": result.get("scenario", scenario_name),
                                "setting": result.get("setting", setting_name),
                                "method": result.get("method", method_name),
                                "run_id(replication_idx)": result.get("run_id(replication_idx)"),
                                **metrics,
                            }
                            all_records.append(flat)
    
    return all_records


def generate_summary(all_records: List[Dict[str, Any]], results_dir: Path, 
                     dataset: str, setting: str, overwrite: bool = False) -> pd.DataFrame | None:
    """
    Generate summary CSV from all_records.
    
    This uses the same logic as run_experiments() function.
    """
    if not all_records:
        print(f"No records found for dataset={dataset}, setting={setting}")
        return None
    
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
        print(f"Warning: No aggregatable columns found for dataset={dataset}, setting={setting}")
        return None
    
    # Replace None in scenario with empty string for groupby (IHDP has None scenario)
    if "scenario" in df.columns:
        df = df.copy()
        df["scenario"] = df["scenario"].fillna("")
    
    # Filter out groups that have no valid data for aggregation
    # Check if at least one aggregation column has non-null values in each group
    grouped = df.groupby(group_cols)
    valid_groups = []
    for name, group in grouped:
        # Check if at least one aggregation column has non-null values
        has_data = any(group[col].notna().any() for col in filtered_agg_dict.keys() if col in group.columns)
        if has_data:
            valid_groups.append(name)
    
    if not valid_groups:
        print(f"Warning: No groups with valid data for aggregation")
        return None
    
    # Filter dataframe to only valid groups
    df_filtered = df[df.set_index(group_cols).index.isin(valid_groups)].reset_index(drop=True)
    
    summary = df_filtered.groupby(group_cols).agg(filtered_agg_dict)
    summary.columns = ["_".join([c for c in col if c]) for col in summary.columns.values]
    summary = summary.reset_index()
    
    # Add additional performance metrics
    import numpy as np
    
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
    if "ate_ci_width" in df_filtered.columns:
        ci_width_agg = df_filtered.groupby(group_cols)["ate_ci_width"].agg(["mean", "std"])
        summary = summary.merge(ci_width_agg, left_on=group_cols, right_index=True, how="left")
        summary = summary.rename(columns={"mean": "ate_ci_width_mean", "std": "ate_ci_width_std"})
    
    # Restore None for scenario if it was originally None
    if "scenario" in summary.columns:
        summary["scenario"] = summary["scenario"].replace("", None)
    
    summary_dir = results_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"summary_{dataset}_{setting}.csv"
    
    if summary_path.exists() and not overwrite:
        print(f"Summary file already exists: {summary_path}")
        print("Use --overwrite to regenerate it.")
        return summary
    
    summary.to_csv(summary_path, index=False)
    print(f"Generated summary: {summary_path}")
    print(f"  Total records: {len(all_records)}")
    print(f"  Summary rows: {len(summary)}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Generate summary CSV from existing experiment result JSON files"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="experiments/results/effect_estimation",
        help="Base results directory",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["ihdp", "synthetic_ci"],
        help="Dataset to process (if not specified, process all)",
    )
    parser.add_argument(
        "--setting",
        type=str,
        default=None,
        help="Setting to process (e.g., oracle_graph, agent_graph). If not specified, process all.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing summary files",
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    print(f"Collecting records from: {results_dir}")
    
    # Collect all records
    all_records = collect_all_records(results_dir, dataset=args.dataset, setting=args.setting)
    
    if not all_records:
        print("No records found. Check if results directory contains JSON files.")
        return
    
    print(f"Total records collected: {len(all_records)}")
    
    # Group by dataset and setting to generate separate summary files
    df = pd.DataFrame(all_records)
    
    if args.dataset and args.setting:
        # Generate single summary file
        generate_summary(all_records, results_dir, args.dataset, args.setting, args.overwrite)
    else:
        # Generate summary files for each dataset-setting combination
        grouped = df.groupby(["dataset", "setting"])
        for (dataset, setting), group_df in grouped:
            dataset_records = group_df.to_dict("records")
            generate_summary(
                dataset_records, 
                results_dir, 
                dataset.lower(), 
                setting, 
                args.overwrite
            )
    
    print("\nSummary generation completed!")


if __name__ == "__main__":
    main()

