"""
Test causal analysis pipeline using synthetic CI benchmark data.

This test:
1. Loads synthetic CI benchmark datasets
2. Runs causal analysis pipeline
3. Compares estimated ATE with ground truth
4. Reports performance metrics (MAE, MSE, CI coverage)
"""

import sys
import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import time

from agents.causal_analysis.graph import generate_causal_analysis_graph
from utils.llm import get_llm
import networkx as nx


def load_ci_dataset(scenario_dir: str, run_idx: int = 0) -> Tuple[pd.DataFrame, Dict]:
    """
    Load a synthetic CI dataset.
    
    Args:
        scenario_dir: Path to scenario directory (e.g., "data/synthetic_ci/CI-1_Baseline")
        run_idx: Which run to load (0-4)
    
    Returns:
        (dataframe, metadata) where metadata contains ground truth ATE
    """
    run_dir = Path(scenario_dir) / f"run_{run_idx:03d}"
    
    # Load data
    data_path = run_dir / "data.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    df = pd.read_csv(data_path)
    
    # Load stats (ground truth)
    stats_path = run_dir / "stats.json"
    if stats_path.exists():
        with open(stats_path, 'r') as f:
            stats = json.load(f)
    else:
        stats = {}
    
    # Load config
    config_path = run_dir / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    metadata = {
        "ground_truth_ate": stats.get("ATE_true"),
        "treat_rate": stats.get("treat_rate_actual"),
        "config": config
    }
    
    return df, metadata


def create_simple_causal_graph(df: pd.DataFrame, treatment: str, outcome: str) -> Dict:
    """
    Create a simple causal graph for CI data.
    Structure: W -> T, W -> Y, T -> Y
    """
    # Identify confounders (W variables)
    confounders = [col for col in df.columns if col.startswith('W')]
    
    # Create edges
    edges = []
    for conf in confounders:
        edges.append({"from": conf, "to": treatment})
        edges.append({"from": conf, "to": outcome})
    edges.append({"from": treatment, "to": outcome})
    
    # Create NetworkX graph
    G = nx.DiGraph()
    all_nodes = confounders + [treatment, outcome]
    G.add_nodes_from(all_nodes)
    for edge in edges:
        G.add_edge(edge["from"], edge["to"])
    
    # Create dot format string
    dot_lines = ["digraph {"]
    for edge in edges:
        dot_lines.append(f'  {edge["from"]} -> {edge["to"]};')
    dot_lines.append("}")
    dot_graph = "\n".join(dot_lines)
    
    return {
        "nodes": all_nodes,
        "edges": edges,
        "variables": all_nodes,
        "graph": {
            "variables": all_nodes,
            "edges": edges
        },
        "nx_graph": G,
        "dot_graph": dot_graph
    }


def test_single_ci_dataset(
    scenario_dir: str,
    run_idx: int,
    llm,
    app,
    verbose: bool = True
) -> Dict:
    """
    Test causal analysis on a single CI dataset.
    
    Returns:
        Dictionary with results and performance metrics
    """
    try:
        # Load data
        df, metadata = load_ci_dataset(scenario_dir, run_idx)
        ground_truth_ate = metadata.get("ground_truth_ate")
        
        if ground_truth_ate is None:
            if verbose:
                print(f"  ⚠ Warning: No ground truth ATE found, skipping")
            return {"success": False, "reason": "no_ground_truth"}
        
        # Identify variables
        treatment = "T"
        outcome = "Y"
        confounders = [col for col in df.columns if col.startswith('W')]
        
        # Create causal graph
        causal_graph = create_simple_causal_graph(df, treatment, outcome)
        
        # Prepare input
        question = f"What is the causal effect of {treatment} on {outcome}?"
        
        state_input = {
            "input": question,
            "df_preprocessed": df,
            "causal_graph": causal_graph,
        }
        
        # Run pipeline
        start_time = time.time()
        result = app.invoke(state_input)
        elapsed_time = time.time() - start_time
        
        # Extract results
        estimated_ate = result.get("causal_effect_ate")
        estimated_ci = result.get("causal_effect_ci")
        strategy = result.get("strategy")
        
        # Calculate errors
        if estimated_ate is not None:
            absolute_error = abs(estimated_ate - ground_truth_ate)
            squared_error = (estimated_ate - ground_truth_ate) ** 2
            
            # Check if ground truth is within confidence interval
            # CI should already be normalized to [lower, upper] format by dowhy_analysis
            if estimated_ci and isinstance(estimated_ci, (list, tuple)) and len(estimated_ci) == 2:
                ci_lower, ci_upper = float(estimated_ci[0]), float(estimated_ci[1])
                within_ci = ci_lower <= ground_truth_ate <= ci_upper
            else:
                within_ci = None
        else:
            absolute_error = None
            squared_error = None
            within_ci = None
        
        return {
            "success": True,
            "scenario": Path(scenario_dir).name,
            "run_idx": run_idx,
            "ground_truth_ate": ground_truth_ate,
            "estimated_ate": estimated_ate,
            "estimated_ci": estimated_ci,
            "absolute_error": absolute_error,
            "squared_error": squared_error,
            "within_ci": within_ci,
            "elapsed_time": elapsed_time,
            "strategy": {
                "task": getattr(strategy, "task", None) if strategy else None,
                "identification": getattr(strategy, "identification_method", None) if strategy else None,
                "estimator": getattr(strategy, "estimator", None) if strategy else None,
            } if strategy else None,
            "n_samples": len(df),
            "n_confounders": len(confounders)
        }
        
    except Exception as e:
        if verbose:
            print(f"  ✗ Error: {e}")
        import traceback
        return {
            "success": False,
            "scenario": Path(scenario_dir).name,
            "run_idx": run_idx,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def test_causal_analysis_synthetic_ci(
    scenarios: List[str] = None,
    runs_per_scenario: int = 1,
    verbose: bool = True
) -> Dict:
    """
    Test causal analysis on synthetic CI benchmark datasets.
    
    Args:
        scenarios: List of scenario names to test (e.g., ["CI-1_Baseline"]). 
                  If None, tests all scenarios.
        runs_per_scenario: Number of runs to test per scenario
        verbose: Whether to print progress
    
    Returns:
        Dictionary with aggregated results and performance metrics
    """
    print("=" * 70)
    print("Testing Causal Analysis on Synthetic CI Benchmarks")
    print("=" * 70)
    
    # Initialize LLM and graph
    if verbose:
        print("\n[1/3] Initializing LLM and causal analysis graph...")
    try:
        llm = get_llm(model="gpt-4o-mini", temperature=0.7, provider="openai")
        app = generate_causal_analysis_graph(llm=llm)
        if verbose:
            print("  ✓ LLM and graph initialized")
    except Exception as e:
        print(f"  ✗ ERROR: Failed to initialize: {e}", file=sys.stderr)
        return {"error": str(e)}
    
    # Find scenarios
    base_dir = Path("data/synthetic_ci")
    if not base_dir.exists():
        print(f"  ✗ ERROR: Directory not found: {base_dir}", file=sys.stderr)
        return {"error": f"Directory not found: {base_dir}"}
    
    if scenarios is None:
        scenario_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()])
    else:
        scenario_dirs = [base_dir / s for s in scenarios]
        scenario_dirs = [d for d in scenario_dirs if d.exists()]
    
    if verbose:
        print(f"\n[2/3] Found {len(scenario_dirs)} scenarios")
        print(f"  Scenarios: {[d.name for d in scenario_dirs]}")
    
    # Run tests
    if verbose:
        print(f"\n[3/3] Running causal analysis on {len(scenario_dirs)} scenarios...")
    
    all_results = []
    for scenario_dir in scenario_dirs:
        scenario_name = scenario_dir.name
        if verbose:
            print(f"\n  Testing {scenario_name}...")
        
        for run_idx in range(min(runs_per_scenario, 5)):  # Max 5 runs per scenario
            if verbose:
                print(f"    Run {run_idx}...", end=" ", flush=True)
            
            result = test_single_ci_dataset(
                str(scenario_dir),
                run_idx,
                llm,
                app,
                verbose=False
            )
            
            all_results.append(result)
            
            if result.get("success"):
                if verbose:
                    gt = result["ground_truth_ate"]
                    est = result["estimated_ate"]
                    err = result["absolute_error"]
                    print(f"✓ ATE: {est:.4f} (GT: {gt:.4f}, Error: {err:.4f})")
            else:
                if verbose:
                    print(f"✗ Failed: {result.get('error', result.get('reason', 'unknown'))}")
    
    # Aggregate results
    successful_results = [r for r in all_results if r.get("success")]
    
    if len(successful_results) == 0:
        print("\n❌ No successful runs!")
        return {
            "total_runs": len(all_results),
            "successful_runs": 0,
            "results": all_results
        }
    
    # Calculate performance metrics
    absolute_errors = [r["absolute_error"] for r in successful_results if r["absolute_error"] is not None]
    squared_errors = [r["squared_error"] for r in successful_results if r["squared_error"] is not None]
    within_ci_counts = [r["within_ci"] for r in successful_results if r["within_ci"] is not None]
    elapsed_times = [r["elapsed_time"] for r in successful_results if r["elapsed_time"] is not None]
    
    mae = np.mean(absolute_errors) if absolute_errors else None
    mse = np.mean(squared_errors) if squared_errors else None
    rmse = np.sqrt(mse) if mse else None
    ci_coverage = np.mean(within_ci_counts) if within_ci_counts else None
    avg_time = np.mean(elapsed_times) if elapsed_times else None
    
    # Print summary
    print("\n" + "=" * 70)
    print("Performance Summary")
    print("=" * 70)
    print(f"Total runs: {len(all_results)}")
    print(f"Successful runs: {len(successful_results)}")
    print(f"Success rate: {len(successful_results) / len(all_results) * 100:.1f}%")
    
    if mae is not None:
        print(f"\nMean Absolute Error (MAE): {mae:.6f}")
        print(f"Mean Squared Error (MSE): {mse:.6f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
    
    if ci_coverage is not None:
        print(f"CI Coverage: {ci_coverage * 100:.1f}%")
    
    if avg_time is not None:
        print(f"Average execution time: {avg_time:.2f} seconds")
    
    # Per-scenario summary with detailed configuration
    print("\n" + "=" * 70)
    print("Per-Scenario Detailed Summary")
    print("=" * 70)
    
    for scenario_dir in scenario_dirs:
        scenario_name = scenario_dir.name
        scenario_results = [r for r in successful_results if r.get("scenario") == scenario_name]
        
        if not scenario_results:
            print(f"\n{scenario_name}: No successful runs")
            continue
        
        print(f"\n{scenario_name}:")
        print("-" * 70)
        
        # Aggregate metrics for this scenario
        scenario_errors = [r["absolute_error"] for r in scenario_results if r["absolute_error"] is not None]
        scenario_squared_errors = [r["squared_error"] for r in scenario_results if r["squared_error"] is not None]
        scenario_within_ci = [r["within_ci"] for r in scenario_results if r["within_ci"] is not None]
        scenario_times = [r["elapsed_time"] for r in scenario_results if r["elapsed_time"] is not None]
        
        if scenario_errors:
            scenario_mae = np.mean(scenario_errors)
            scenario_mse = np.mean(scenario_squared_errors) if scenario_squared_errors else None
            scenario_rmse = np.sqrt(scenario_mse) if scenario_mse else None
            scenario_ci_coverage = np.mean(scenario_within_ci) if scenario_within_ci else None
            scenario_avg_time = np.mean(scenario_times) if scenario_times else None
            
            print(f"  Runs: {len(scenario_results)}")
            print(f"  MAE: {scenario_mae:.6f}")
            if scenario_rmse:
                print(f"  RMSE: {scenario_rmse:.6f}")
            if scenario_ci_coverage is not None:
                print(f"  CI Coverage: {scenario_ci_coverage * 100:.1f}%")
            if scenario_avg_time:
                print(f"  Avg Time: {scenario_avg_time:.2f}s")
        
        # Show configuration used
        print(f"\n  Configuration (Strategy):")
        strategies_used = {}
        for r in scenario_results:
            strategy = r.get("strategy")
            if strategy:
                strategy_key = (
                    strategy.get("task", "unknown"),
                    strategy.get("identification", "unknown"),
                    strategy.get("estimator", "unknown")
                )
                if strategy_key not in strategies_used:
                    strategies_used[strategy_key] = []
                strategies_used[strategy_key].append(r.get("run_idx"))
        
        if strategies_used:
            for (task, ident, est), run_indices in strategies_used.items():
                runs_str = ", ".join([f"Run {idx}" for idx in sorted(run_indices)])
                print(f"    - Task: {task}")
                print(f"      Identification: {ident}")
                print(f"      Estimator: {est}")
                print(f"      Used in: {runs_str} ({len(run_indices)} time(s))")
        else:
            print("    - No strategy information available")
        
        # Show individual run results with configuration
        print(f"\n  Individual Run Results:")
        for r in scenario_results:
            run_idx = r.get("run_idx", "?")
            gt = r.get("ground_truth_ate", "?")
            est = r.get("estimated_ate", "?")
            err = r.get("absolute_error", "?")
            within_ci = r.get("within_ci")
            time_taken = r.get("elapsed_time", "?")
            strategy = r.get("strategy", {})
            
            ci_status = "✓" if within_ci else "✗" if within_ci is False else "?"
            
            print(f"    Run {run_idx}:")
            print(f"      Ground Truth ATE: {gt:.4f}")
            print(f"      Estimated ATE: {est:.4f}")
            print(f"      Absolute Error: {err:.6f}")
            print(f"      CI Coverage: {ci_status}")
            print(f"      Execution Time: {time_taken:.2f}s")
            if strategy:
                print(f"      Strategy: {strategy.get('task', 'N/A')} / {strategy.get('identification', 'N/A')} / {strategy.get('estimator', 'N/A')}")
            print()
    
    return {
        "total_runs": len(all_results),
        "successful_runs": len(successful_results),
        "success_rate": len(successful_results) / len(all_results) if all_results else 0,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "ci_coverage": ci_coverage,
        "avg_time": avg_time,
        "results": all_results
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test causal analysis on synthetic CI benchmarks")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help="Scenarios to test (e.g., CI-1_Baseline CI-2_Nonlinear_T). If not specified, tests all."
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of runs per scenario (default: 1, max: 5)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    result = test_causal_analysis_synthetic_ci(
        scenarios=args.scenarios,
        runs_per_scenario=args.runs,
        verbose=not args.quiet
    )
    
    # Exit with error code if no successful runs
    if result.get("successful_runs", 0) == 0:
        sys.exit(1)
    else:
        sys.exit(0)

