from __future__ import annotations

import pprint
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import time
from datetime import datetime

from utils.settings import CONFIG
from agents.causal_discovery.agent import CausalDiscoveryAgent
from tests.synthetic_data import generate_er_synthetic

"""
This script runs multiple synthetic datasets to evaluate causal discovery performance:
- Runs 100 synthetic datasets with minimal print output
- Collects performance metrics across all runs
- Provides summary statistics and analysis
"""


def print_step(header: str, payload: Dict[str, Any]):
    """Minimal print function for debugging - only prints essential info"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {header}: {payload}")


def extract_edges_from_graph(graph: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Extract edge list from graph structure"""
    if not graph:
        return []
    
    # Handle different graph structures:
    # 1. Standard structure: graph["graph"]["edges"] (algorithm results)
    # 2. Ensemble structure: graph["edges"] (consensus_pag, selected_graph)
    edges_list = []
    if "graph" in graph and "edges" in graph["graph"]:
        edges_list = graph["graph"]["edges"]
    elif "edges" in graph:
        edges_list = graph["edges"]
    else:
        return []
    
    edges = []
    for edge in edges_list:
        if isinstance(edge, dict) and "from" in edge and "to" in edge:
            edges.append((edge["from"], edge["to"]))
        elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
            edges.append((str(edge[0]), str(edge[1])))
    
    return edges


def calculate_graph_metrics(predicted_edges: List[Tuple[str, str]], 
                          true_edges: List[Tuple[str, str]], 
                          all_variables: List[str]) -> Dict[str, float]:
    """Calculate precision, recall, F1, and structural Hamming distance"""
    
    # Convert to sets for easier comparison
    pred_set = set(predicted_edges)
    true_set = set(true_edges)
    
    # Calculate basic metrics
    true_positives = len(pred_set & true_set)
    false_positives = len(pred_set - true_set)
    false_negatives = len(true_set - pred_set)
    
    # Precision, Recall, F1
    precision = true_positives / len(pred_set) if pred_set else 0.0
    recall = true_positives / len(true_set) if true_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Structural Hamming Distance (SHD)
    # SHD = |E_pred - E_true| + |E_true - E_pred|
    shd = len(pred_set - true_set) + len(true_set - pred_set)
    
    # Normalize SHD by maximum possible edges
    max_edges = len(all_variables) * (len(all_variables) - 1)
    normalized_shd = shd / max_edges if max_edges > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "shd": shd,
        "normalized_shd": normalized_shd,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "total_predicted": len(pred_set),
        "total_true": len(true_set)
    }


def evaluate_against_ground_truth(discovered_graph: Dict[str, Any], 
                                ground_truth: Dict[str, Any], 
                                variables: List[str]) -> Dict[str, Any]:
    """Evaluate discovered graph against ground truth"""
    
    # Extract edges
    pred_edges = extract_edges_from_graph(discovered_graph)
    true_edges = ground_truth.get("edges", [])
    
    # Convert true edges to tuples if needed
    if true_edges and isinstance(true_edges[0], dict):
        true_edges = [(e["from"], e["to"]) for e in true_edges if "from" in e and "to" in e]
    
    # Calculate metrics
    metrics = calculate_graph_metrics(pred_edges, true_edges, variables)
    
    return {
        "metrics": metrics,
        "predicted_edges": pred_edges,
        "true_edges": true_edges,
        "edge_comparison": {
            "correct_edges": list(set(pred_edges) & set(true_edges)),
            "missing_edges": list(set(true_edges) - set(pred_edges)),
            "extra_edges": list(set(pred_edges) - set(true_edges))
        }
    }


def run_single_experiment(agent: CausalDiscoveryAgent, df: pd.DataFrame, 
                         ground_truth: Dict[str, Any], run_id: int) -> Dict[str, Any]:
    """Run a single causal discovery experiment and return results"""
    
    state: Dict[str, Any] = {"df_preprocessed": df}

    substeps = [
        "data_profiling",
        "algorithm_tiering",
        "run_algorithms_portfolio",
        "candidate_pruning",
        "scorecard_evaluation",
        "ensemble_synthesis",
    ]

    start_time = time.time()
    results = {
        "run_id": run_id,
        "success": False,
        "error": None,
        "execution_time": 0,
        "final_metrics": None,
        "algorithm_results": {},
        "data_profile": None,
        "algorithm_tiers": None
    }
    
    try:
        for sub in substeps:
            state["current_substep"] = sub
            state = agent.step(state)
            
            if state.get("error"):
                results["error"] = state["error"]
                break
                
            # Store key results
            if sub == "data_profiling":
                results["data_profile"] = state.get("data_profile", {})
            elif sub == "algorithm_tiering":
                results["algorithm_tiers"] = state.get("algorithm_tiers", {})
            elif sub == "run_algorithms_portfolio":
                results["algorithm_results"] = state.get("algorithm_results", {})
            elif sub == "ensemble_synthesis":
                dag = state.get("selected_graph", {})
                if dag and isinstance(dag, dict):
                    evaluation = evaluate_against_ground_truth(dag, ground_truth, list(df.columns))
                    results["final_metrics"] = evaluation["metrics"]
                    results["success"] = True
        
        results["execution_time"] = time.time() - start_time
        
    except Exception as e:
        results["error"] = str(e)
        results["execution_time"] = time.time() - start_time
    
    return results


def run_batch_experiments(n_experiments: int = 100, 
                         n_nodes: int = 8, 
                         edge_prob: float = 0.25, 
                         n_samples: int = 1500,
                         fast_mode: bool = True) -> List[Dict[str, Any]]:
    """Run multiple causal discovery experiments"""
    
    print(f"Starting batch of {n_experiments} experiments...")
    print(f"Parameters: n_nodes={n_nodes}, edge_prob={edge_prob}, n_samples={n_samples}")
    
    # Load config and build agent once
    cd_config = (CONFIG.get("agents", {}) or {}).get("causal_discovery", {})
    
    # Reduce bootstrap iterations for faster batch processing
    if fast_mode:
        cd_config = cd_config.copy()
        cd_config["bootstrap_iterations"] = 5  # Use fewer iterations instead of 0
    
    agent = CausalDiscoveryAgent(config=cd_config)
    
    all_results = []
    successful_runs = 0
    
    for i in range(n_experiments):
        if i % 10 == 0:
            print(f"Progress: {i}/{n_experiments} experiments completed")
        
        # Generate synthetic data
        df, meta = generate_er_synthetic(
            n_nodes=n_nodes, 
            edge_prob=edge_prob, 
            n_samples=n_samples, 
            seed=i + 42  # Different seed for each run
        )
        
        # Store ground truth
        ground_truth = {
            "edges": meta.get("edges", []),
            "variables": list(df.columns)
        }
        
        # Run experiment
        result = run_single_experiment(agent, df, ground_truth, i)
        all_results.append(result)
        
        if result["success"]:
            successful_runs += 1
        
        # Print minimal progress for failed runs
        if not result["success"]:
            print(f"Run {i} failed: {result.get('error', 'Unknown error')}")
    
    print(f"\nBatch completed: {successful_runs}/{n_experiments} successful runs")
    return all_results


def analyze_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze and summarize batch experiment results"""
    
    successful_results = [r for r in results if r["success"]]
    failed_results = [r for r in results if not r["success"]]
    
    if not successful_results:
        return {
            "summary": "No successful runs to analyze",
            "success_rate": 0.0,
            "total_runs": len(results),
            "failed_runs": len(failed_results)
        }
    
    # Extract metrics from successful runs
    metrics_data = []
    execution_times = []
    
    for result in successful_results:
        if result["final_metrics"]:
            metrics_data.append(result["final_metrics"])
            execution_times.append(result["execution_time"])
    
    if not metrics_data:
        return {
            "summary": "No valid metrics to analyze",
            "success_rate": len(successful_results) / len(results),
            "total_runs": len(results)
        }
    
    # Calculate summary statistics
    metrics_df = pd.DataFrame(metrics_data)
    
    summary_stats = {
        "precision": {
            "mean": float(metrics_df["precision"].mean()),
            "std": float(metrics_df["precision"].std()),
            "min": float(metrics_df["precision"].min()),
            "max": float(metrics_df["precision"].max())
        },
        "recall": {
            "mean": float(metrics_df["recall"].mean()),
            "std": float(metrics_df["recall"].std()),
            "min": float(metrics_df["recall"].min()),
            "max": float(metrics_df["recall"].max())
        },
        "f1_score": {
            "mean": float(metrics_df["f1_score"].mean()),
            "std": float(metrics_df["f1_score"].std()),
            "min": float(metrics_df["f1_score"].min()),
            "max": float(metrics_df["f1_score"].max())
        },
        "shd": {
            "mean": float(metrics_df["shd"].mean()),
            "std": float(metrics_df["shd"].std()),
            "min": float(metrics_df["shd"].min()),
            "max": float(metrics_df["shd"].max())
        },
        "normalized_shd": {
            "mean": float(metrics_df["normalized_shd"].mean()),
            "std": float(metrics_df["normalized_shd"].std()),
            "min": float(metrics_df["normalized_shd"].min()),
            "max": float(metrics_df["normalized_shd"].max())
        }
    }
    
    execution_stats = {
        "mean_time": float(np.mean(execution_times)),
        "std_time": float(np.std(execution_times)),
        "min_time": float(np.min(execution_times)),
        "max_time": float(np.max(execution_times))
    }
    
    # Error analysis
    error_counts = {}
    for failed in failed_results:
        error = failed.get("error", "Unknown")
        error_counts[error] = error_counts.get(error, 0) + 1
    
    return {
        "summary": f"Analyzed {len(successful_results)}/{len(results)} successful runs",
        "success_rate": len(successful_results) / len(results),
        "total_runs": len(results),
        "failed_runs": len(failed_results),
        "metrics_summary": summary_stats,
        "execution_time_stats": execution_stats,
        "error_analysis": error_counts,
        "detailed_results": results  # Keep all results for further analysis
    }


def main():
    """Run batch experiments and analyze results"""
    
    # Configuration for batch experiments
    n_experiments = 100
    n_nodes = 8
    edge_prob = 0.25
    n_samples = 1500
    
    print(f"Starting causal discovery batch evaluation...")
    print(f"Configuration: {n_experiments} experiments, {n_nodes} nodes, edge_prob={edge_prob}, {n_samples} samples")
    print("Note: Using reduced bootstrap iterations (5) for faster batch processing")
    
    # Run batch experiments
    start_time = time.time()
    results = run_batch_experiments(
        n_experiments=n_experiments,
        n_nodes=n_nodes,
        edge_prob=edge_prob,
        n_samples=n_samples,
        fast_mode=True  # Use reduced bootstrap iterations for faster processing
    )
    total_time = time.time() - start_time
    
    # Analyze results
    print("\n" + "="*60)
    print("BATCH EXPERIMENT ANALYSIS")
    print("="*60)
    
    analysis = analyze_results(results)
    
    # Print summary
    print(f"\nSUMMARY:")
    print(f"Total experiments: {analysis['total_runs']}")
    print(f"Successful runs: {analysis['total_runs'] - analysis['failed_runs']}")
    print(f"Success rate: {analysis['success_rate']:.2%}")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Average time per experiment: {total_time/n_experiments:.2f} seconds")
    
    if analysis['success_rate'] > 0:
        print(f"\nPERFORMANCE METRICS (mean ± std):")
        metrics = analysis['metrics_summary']
        print(f"Precision: {metrics['precision']['mean']:.3f} ± {metrics['precision']['std']:.3f}")
        print(f"Recall: {metrics['recall']['mean']:.3f} ± {metrics['recall']['std']:.3f}")
        print(f"F1 Score: {metrics['f1_score']['mean']:.3f} ± {metrics['f1_score']['std']:.3f}")
        print(f"SHD: {metrics['shd']['mean']:.1f} ± {metrics['shd']['std']:.1f}")
        print(f"Normalized SHD: {metrics['normalized_shd']['mean']:.3f} ± {metrics['normalized_shd']['std']:.3f}")
        
        print(f"\nEXECUTION TIME STATS:")
        exec_stats = analysis['execution_time_stats']
        print(f"Mean: {exec_stats['mean_time']:.2f}s")
        print(f"Std: {exec_stats['std_time']:.2f}s")
        print(f"Min: {exec_stats['min_time']:.2f}s")
        print(f"Max: {exec_stats['max_time']:.2f}s")
    
    if analysis['error_analysis']:
        print(f"\nERROR ANALYSIS:")
        for error, count in analysis['error_analysis'].items():
            print(f"  {error}: {count} occurrences")
    
    # Save detailed results to file
    import json
    output_file = f"causal_discovery_batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {output_file}")
    
    return analysis


def run_custom_batch(n_experiments: int = 100, 
                    n_nodes: int = 8, 
                    edge_prob: float = 0.25, 
                    n_samples: int = 1500,
                    save_results: bool = True,
                    fast_mode: bool = True) -> Dict[str, Any]:
    """Run custom batch experiments with specified parameters"""
    
    print(f"Starting custom batch evaluation...")
    print(f"Configuration: {n_experiments} experiments, {n_nodes} nodes, edge_prob={edge_prob}, {n_samples} samples")
    if fast_mode:
        print("Note: Using reduced bootstrap iterations (5) for faster batch processing")
    
    # Run batch experiments
    start_time = time.time()
    results = run_batch_experiments(
        n_experiments=n_experiments,
        n_nodes=n_nodes,
        edge_prob=edge_prob,
        n_samples=n_samples,
        fast_mode=fast_mode
    )
    total_time = time.time() - start_time
    
    # Analyze results
    analysis = analyze_results(results)
    
    # Print summary
    print(f"\nSUMMARY:")
    print(f"Total experiments: {analysis['total_runs']}")
    print(f"Successful runs: {analysis['total_runs'] - analysis['failed_runs']}")
    print(f"Success rate: {analysis['success_rate']:.2%}")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Average time per experiment: {total_time/n_experiments:.2f} seconds")
    
    if analysis['success_rate'] > 0:
        print(f"\nPERFORMANCE METRICS (mean ± std):")
        metrics = analysis['metrics_summary']
        print(f"Precision: {metrics['precision']['mean']:.3f} ± {metrics['precision']['std']:.3f}")
        print(f"Recall: {metrics['recall']['mean']:.3f} ± {metrics['recall']['std']:.3f}")
        print(f"F1 Score: {metrics['f1_score']['mean']:.3f} ± {metrics['f1_score']['std']:.3f}")
        print(f"SHD: {metrics['shd']['mean']:.1f} ± {metrics['shd']['std']:.1f}")
        print(f"Normalized SHD: {metrics['normalized_shd']['mean']:.3f} ± {metrics['normalized_shd']['std']:.3f}")
    
    # Save results if requested
    if save_results:
        import json
        output_file = f"causal_discovery_batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"\nDetailed results saved to: {output_file}")
    
    return analysis


if __name__ == "__main__":
    # Run default batch of 100 experiments
    main()
    
    # Uncomment below to run custom experiments with different parameters
    # run_custom_batch(n_experiments=50, n_nodes=6, edge_prob=0.3, n_samples=1000)


